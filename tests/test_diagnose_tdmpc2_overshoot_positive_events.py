"""Synthetic-only tests: no real checkpoint load or environment construction."""

import hashlib
import json

import numpy as np
import pytest
import torch

from scripts import diagnose_tdmpc2_overshoot_positive_events as diagnostic


def episodes(length=10):
    return [{"episode": i, "track_id": 1, "geometry_seed": road, "length": length}
            for i, road in enumerate(diagnostic.raw.ROADS)]


def replay(length=10, *, ending="terminated"):
    result = []
    for i in range(4):
        actions = np.full((length, 3), i / 4, np.float32)
        observations = np.zeros((length + 1, 4, 64, 64), np.uint8)
        for step in range(length + 1):
            observations[step].fill(step + i * 20)
        rewards = np.arange(length, dtype=np.float32) - 3
        flags = {key: np.zeros(length, np.bool_) for key in ("terminated", "truncated", "terminal")}
        flags[ending][-1] = True
        flags["terminal"][-1] = ending == "terminated"
        result.append({"observations": observations, "actions": actions, "rewards": rewards, **flags})
    return {"episodes": result}


def test_reward_blind_randomstate_sampling_and_no_cross_episode():
    sample = diagnostic.select(episodes(), count=5)
    assert sample == diagnostic.select(episodes(), count=5)
    assert len(sample) == len(set(sample)) == 20
    assert all(0 <= start <= 5 for _, start in sample)
    # A reward/finish field must never influence the selection.
    annotated = [dict(ep, reward=1e10, finished=True) for ep in episodes()]
    assert diagnostic.select(annotated, count=5) == sample
    assert len(diagnostic.select(episodes(length=5), count=3)) == 12
    assert {start for _, start in diagnostic.select(episodes(length=5), count=3)} == {0}
    with pytest.raises(ValueError, match="no H5"):
        diagnostic.select(episodes(length=4), count=1)


def test_frame_step_alignment_timeout_and_terminal_mask():
    data = replay()
    selected = [(0, 0), (0, 5), (1, 5)]
    windows, digest = diagnostic.bind_windows(data, selected)
    assert len(digest) == 64
    assert windows[0]["obs"][:, 0, 0, 0].tolist() == [0, 1, 2, 3, 4]
    assert windows[1]["obs"][:, 0, 0, 0].tolist() == [5, 6, 7, 8, 9]
    assert windows[1]["rewards"].tolist() == [2, 3, 4, 5, 6]
    assert windows[1]["target_mask"] == (True,) * 5
    assert windows[1]["flags"][2].tolist() == [False] * 4 + [True]
    timeout = replay(ending="truncated")
    time_windows, _ = diagnostic.bind_windows(timeout, [(0, 5)])
    assert time_windows[0]["flags"][1][-1] and not time_windows[0]["flags"][2][-1]
    data["episodes"][0]["terminated"][3] = True
    with pytest.raises(ValueError, match="crosses"):
        diagnostic.bind_windows(data, [(0, 0)])
    data["episodes"][0]["terminated"][3] = False
    data["episodes"][0]["terminal"][3] = True
    with pytest.raises(ValueError, match="crosses"):
        diagnostic.bind_windows(data, [(0, 0)])


class FakeModel:
    def eval(self):
        pass

    def encode(self, obs, task):
        return obs[:, 0, 0, 0].float().unsqueeze(1)

    def reward(self, latent, action, task):
        # Categorical logits peaked at the symlog bin indexed by latent.
        logits = torch.full((len(latent), 101), -1000.0)
        logits.scatter_(1, (50 + latent.long()), 1000.0)
        return logits

    def next(self, latent, action, task):
        return latent + 2

    cfg = type("Config", (), {"num_bins": 101, "vmin": -10.0, "vmax": 10.0})()


def test_reward_ce_decoding_and_true_vs_imagined_frame_alignment():
    windows, _ = diagnostic.bind_windows(replay(), [(0, 0)])
    actual, imagined = diagnostic.predict(FakeModel(), windows, batch_size=1)
    assert np.allclose(actual[0], np.expm1(np.arange(5) * 0.2), atol=1e-6)
    assert np.allclose(imagined[0], np.expm1(np.arange(5) * 0.4), atol=1e-6)
    assert actual[0, 4] != imagined[0, 4]
    assert windows[0]["obs"][0, 0, 0, 0] == 0


def test_finite_and_count_gate_road_thresholds():
    roads = diagnostic.raw.ROADS
    windows = []
    for road in roads:
        for n in range(diagnostic.COUNT):
            windows.append({"road": road, "episode_id": roads.index(road), "start_step": n,
                            "rewards": np.array([1, 0, 0, 0, 0], np.float32),
                            "flags": tuple(np.zeros(5, np.bool_) for _ in range(3))})
    target = np.stack([w["rewards"] for w in windows])
    old = (target.copy(), target.copy())
    new = (target - 0.30, target - 0.30)
    summary, counts = diagnostic.summarize(windows, {"old": old, "new": new})
    gate = diagnostic.mechanism_gate(summary, counts)
    assert gate["status"] == "PASS" and gate["qualifying_roads"] == 4
    assert counts[str(roads[0])]["unique_transitions"] == 2052
    assert sum(row["valid_reward_targets"] for row in counts[str(roads[0])]["by_depth"]) == 5 * diagnostic.COUNT
    assert summary[f"{roads[0]}/new/true_latent/1/positive"]["count"] == 2048
    counts[str(roads[0])]["positive"] = 99
    assert diagnostic.mechanism_gate(summary, counts)["status"] == "FAIL"
    counts[str(roads[0])]["positive"] = 100
    assert diagnostic.mechanism_gate(summary, counts)["status"] == "PASS"
    counts[str(roads[0])]["positive"] = 2048
    for road in roads[1:]:
        summary[f"{road}/new/true_latent/all/positive"]["mae"] = 0.04
    assert diagnostic.mechanism_gate(summary, counts)["status"] == "FAIL"
    new[0][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        diagnostic.summarize(windows, {"old": old, "new": new})


def test_source_check_before_deserialization_and_zero_reset(monkeypatch, tmp_path):
    (tmp_path / "runs").mkdir()
    visited = []
    monkeypatch.setattr(diagnostic, "preflight", lambda root, sha: (_ for _ in ()).throw(
        ValueError("source SHA mismatch")))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: visited.append("deserialize"))
    with pytest.raises(ValueError, match="source SHA"):
        diagnostic.score(tmp_path, "0" * 64)
    assert not visited and not list((tmp_path / "runs").iterdir())


def test_original_replay_episode_identity_is_not_new_training_identity(monkeypatch, tmp_path):
    original = episodes()
    newer = [dict(ep, length=ep["length"] + 1) for ep in original] + [dict(original[0], episode=4)]
    monkeypatch.setattr(diagnostic.raw, "FINAL_EPISODES", len(original))
    monkeypatch.setattr(diagnostic, "_episodes", lambda root: original)
    monkeypatch.setattr(diagnostic, "_proposal", lambda root, selected: {"source": "synthetic"})
    monkeypatch.setattr(diagnostic.source, "preflight", lambda *args: {"episodes": newer})
    monkeypatch.setattr(diagnostic.raw, "_prepared", lambda root: (None, None, {
        "episodes": original, "pin": {"step_prefix_sha256": diagnostic.raw.STEPS_SHA}}))
    report = diagnostic.preflight(tmp_path)
    assert report["original_episodes"] == original
    assert len(report["bundle"]["episodes"]) == len(original) + 1
    monkeypatch.setattr(diagnostic.raw, "_prepared", lambda root: (None, None, {
        "episodes": newer, "pin": {"step_prefix_sha256": diagnostic.raw.STEPS_SHA}}))
    with pytest.raises(ValueError, match="metadata differs"):
        diagnostic.preflight(tmp_path)


def test_exclusive_fsynced_body_sha_receipt(monkeypatch, tmp_path):
    (tmp_path / "runs").mkdir()
    calls = []
    monkeypatch.setattr(diagnostic.os, "fsync", lambda fd: calls.append(fd))
    body = {"environment_resets": 0, "optimizer_updates": 0, "status": "FAIL"}
    diagnostic.write_receipt(tmp_path, body)
    path = tmp_path / diagnostic.OUTPUT
    saved = json.loads(path.read_text())
    assert saved["body_sha256"] == hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert len(calls) == 2
    with pytest.raises(ValueError, match="exclusive"):
        diagnostic.write_receipt(tmp_path, body)
