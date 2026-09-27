"""Synthetic final-source contracts; no real HAIC reset or learner update."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from common_adapter import ActionAdapter, Transition
from drq_v2 import Uint8Replay
from scripts import collect_drq_final_source_replay as collect
from scripts import train_drq_final_source_replay as train


def test_sparse_checkpoint_allocation_preserves_bytes_and_mmap(tmp_path):
    """Reclaim never-inserted replay slots without changing sealed checkpoint bytes."""
    path = tmp_path / "checkpoint.pt"
    frames = np.zeros((1024, 84, 84), dtype=np.uint8)
    frames[0] = 42
    torch.save({"replay": {"frames": frames}, "gradient_steps": 22768}, path)
    old_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    allocated_before = path.stat().st_blocks * 512
    subprocess.run(["fallocate", "--dig-holes", str(path)], check=True)
    allocated_after = path.stat().st_blocks * 512
    assert allocated_after < allocated_before // 2
    assert hashlib.sha256(path.read_bytes()).hexdigest() == old_sha
    restored = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    np.testing.assert_array_equal(restored["replay"]["frames"], frames)
    assert restored["gradient_steps"] == 22768


def _state_and_ledger():
    first, last = collect.FIRST_SOURCE_SEQUENCE, collect.LAST_SOURCE_SEQUENCE + 1
    cap = collect.CAPACITY
    # Episode 0 crosses the rolling replay start; 1/2 are complete, 3 is
    # the final original checkpoint's unfinished episode.
    ledger = [{"episode_id": 0, "track_id": 1, "geometry_seed": 81,
               "start_sequence": 30900, "end_step": 31300},
              {"episode_id": 1, "track_id": 2, "geometry_seed": 82,
               "start_sequence": 31300, "end_step": 90000},
              {"episode_id": 2, "track_id": 3, "geometry_seed": 83,
               "start_sequence": 90000, "end_step": 131000},
              {"episode_id": 3, "track_id": 4, "geometry_seed": 84,
               "start_sequence": 131000}]
    sequences = np.arange(first, last, dtype=np.int64)
    state = {"capacity": cap, "size": cap, "next_sequence": last,
             "action_dim": 3, "n_step": 3, "gamma": .99,
             "sequence_ids": sequences % cap,
             "episode_ids": np.zeros(cap, dtype=np.int64),
             "episode_steps": np.zeros(cap, dtype=np.int64)}
    state["sequence_ids"][sequences % cap] = sequences
    for row in ledger:
        selected = sequences >= row["start_sequence"]
        if "end_step" in row:
            selected &= sequences < row["end_step"]
        state["episode_ids"][sequences[selected] % cap] = row["episode_id"]
        state["episode_steps"][sequences[selected] % cap] = sequences[selected] - row["start_sequence"]
    return state, ledger


def test_schedule_starts_at_boundary_then_visits_earlier_source_train_once():
    state, ledger = _state_and_ledger()
    pairs = collect.derive_schedule(state, ledger, excluded=set())
    assert [row["original_episode_id"] for row in pairs] == [0, 1, 2, 3]
    assert [(r["track_id"], r["geometry_seed"]) for r in pairs] == [
        (1, 81), (2, 82), (3, 83), (4, 84)]
    assert collect.schedule_sha256(pairs) == collect.schedule_sha256(pairs)
    assert collect.schedule_sha256(pairs) != collect.schedule_sha256(list(reversed(pairs)))
    with pytest.raises(ValueError, match="diagnostic/protected/reserved"):
        collect.derive_schedule(state, ledger, excluded={(3, 83)})
    tampered = dict(state, episode_steps=state["episode_steps"].copy())
    tampered["episode_steps"][31300 % collect.CAPACITY] = 12
    with pytest.raises(ValueError, match="episode steps|reset"):
        collect.derive_schedule(tampered, ledger, excluded=set())


def test_schedule_finite_historical_prefix_only_after_original_retained_order():
    state, ledger = _state_and_ledger()
    ledger.insert(0, {"episode_id": 0, "track_id": 2, "geometry_seed": 80,
                      "start_sequence": 0, "end_step": 30900})
    for row in ledger[1:]:
        row["episode_id"] += 1
    selected = np.asarray(state["episode_ids"])
    state["episode_ids"] = selected + 1
    pairs = collect.derive_schedule(state, ledger, excluded=set())
    assert [row["original_episode_id"] for row in pairs] == [1, 2, 3, 4, 0]
    assert [(r["track_id"], r["geometry_seed"]) for r in pairs] == [
        (1, 81), (2, 82), (3, 83), (4, 84), (2, 80)]
    assert len({(r["track_id"], r["geometry_seed"]) for r in pairs}) == len(pairs)


def _synthetic_pool(tmp_path: Path):
    schedule = [{"original_episode_id": 6, "track_id": 2, "geometry_seed": 111},
                {"original_episode_id": 7, "track_id": 4, "geometry_seed": 222}]
    replay = Uint8Replay(capacity=12, n_step=3, gamma=.99)
    action_adapter = ActionAdapter()
    reset = []
    steps = []
    for ep, size in enumerate((4, 2)):
        road = schedule[ep]
        reset.append({"event": "reset", "episode_id": ep, "schedule_index": ep,
                      "original_episode_id": road["original_episode_id"], "source_seed": 0,
                      "track_id": road["track_id"], "geometry_seed": road["geometry_seed"],
                      "seed": road["geometry_seed"], "partition": "TRAIN",
                      "collection_step": len(steps)})
        for step in range(size):
            deterministic = np.asarray([0., .3, -.7], dtype=np.float32)
            noise = np.asarray([.01, -.01, .02], dtype=np.float64)
            noisy = deterministic.astype(np.float64) + noise
            pre_adapter = np.clip(noisy, -1., 1.).astype(np.float32)
            official = action_adapter.to_official(pre_adapter)
            native = action_adapter.to_native(official)
            done = ep == 0 and step == size - 1
            observation = np.full((4, 84, 84), (len(steps) + 1) / 255, dtype=np.float32)
            replay.add(Transition(observation, native, 1., observation, done, False,
                                  terminal=done, episode_id=ep, step=step))
            steps.append({"decision": len(steps) + 1, "sequence_id": len(steps),
                          "source_seed": 0, "schedule_index": ep, "episode_id": ep,
                          "original_episode_id": road["original_episode_id"],
                          "episode_step": step, "track_id": road["track_id"],
                          "geometry_seed": road["geometry_seed"], "partition": "TRAIN",
                          "unnoised_native_action": deterministic.tolist(), "noise": noise.tolist(),
                          "noised_native_action": noisy.tolist(),
                          "pre_adapter_native_action": pre_adapter.tolist(),
                          "native_action": native.tolist(), "official_action": official.tolist(),
                          "reward": 1., "terminated": done, "truncated": False, "terminal": done})
        reset.append({"event": "end" if ep == 0 else "capped_partial", "episode_id": ep,
                      "schedule_index": ep, "original_episode_id": road["original_episode_id"],
                      "source_seed": 0, "track_id": road["track_id"], "seed": road["geometry_seed"],
                      "geometry_seed": road["geometry_seed"], "collection_step": len(steps),
                      "steps": size, "reward": float(size)})
    episodes = tmp_path / "episodes.jsonl"
    actions = tmp_path / "steps.jsonl"
    episodes.write_text("".join(json.dumps(row) + "\n" for row in reset), encoding="utf-8")
    actions.write_text("".join(json.dumps(row) + "\n" for row in steps), encoding="utf-8")
    return replay, episodes, actions, schedule, reset, steps


def test_pool_join_accepts_complete_and_capped_partial_with_actual_adapter_action(tmp_path):
    replay, episodes, actions, schedule, _, _ = _synthetic_pool(tmp_path)
    train.check_pool_lineage(replay, episodes, actions, schedule, count=6, seed=0)
    assert len(replay.valid_indices()) >= 4


@pytest.mark.parametrize("field,value,fragment", [
    ("geometry_seed", 991, "unknown, diagnostic or non-TRAIN"),
    ("partition", "TRAIN-DIAGNOSTIC", "unknown, diagnostic or non-TRAIN"),
    ("reward", 80., "step ledger"),
    ("source_seed", 1, "unknown, diagnostic or non-TRAIN"),
    ("official_action", [1., 1., 1.], "action/reward/terminal"),
    ("terminal", True, "action/reward/terminal"),
])
def test_pool_join_rejects_tampered_road_reward_action_terminal_and_seed(tmp_path, field, value, fragment):
    replay, episodes, actions, schedule, _, steps = _synthetic_pool(tmp_path)
    steps[0][field] = value
    actions.write_text("".join(json.dumps(row) + "\n" for row in steps), encoding="utf-8")
    with pytest.raises(ValueError, match=fragment):
        train.check_pool_lineage(replay, episodes, actions, schedule, count=6, seed=0)


def test_pool_join_rejects_foreign_reset_or_missing_end(tmp_path):
    replay, episodes, actions, schedule, rows, _ = _synthetic_pool(tmp_path)
    rows[0]["seed"] = 888
    episodes.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="foreign road"):
        train.check_pool_lineage(replay, episodes, actions, schedule, count=6, seed=0)
    rows[0]["seed"] = 111
    episodes.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        train.check_pool_lineage(replay, episodes, actions, schedule, count=6, seed=0)


def test_collection_protocol_fails_closed_without_candidate_specific_audit(tmp_path, monkeypatch):
    state, ledger = _state_and_ledger()
    schedule = collect.derive_schedule(state, ledger, set())
    (tmp_path / "r6.json").write_text(json.dumps({"source_actors": [
        {"learner_seed": seed, "actor_sha256": "b" * 64,
         "checkpoint_sha256": "a" * 64, "checkpoint_path": "checkpoint.pt"}
        for seed in (0, 1)], "catalog": {"path": "catalog.json", "sha256": "c" * 64},
        "runtime": {"training_dependencies": {"python": sys.version.split()[0]}}}), encoding="utf-8")
    (tmp_path / "r7.json").write_text(json.dumps({"r6_protocol_path": "r6.json",
        "r6_protocol_sha256": collect.r7.R6_SHA,
        "source_replay": {str(seed): {"checkpoint_sha256": "a" * 64,
            "episode_ledger_path": "ledger.jsonl", "episode_ledger_sha256": "d" * 64}
            for seed in (0, 1)}}), encoding="utf-8")
    (tmp_path / "catalog.json").write_text(json.dumps({"train_diagnostic": [
        {"track_id": 1, "geometry_seed": i + 1000} for i in range(16)]}), encoding="utf-8")
    study = {"format": collect.FORMAT, "study_id": collect.STUDY_ID,
             "run_root": collect.RUN_ROOT.as_posix(),
             "schedule_algorithm": collect.SCHEDULE_ALGORITHM,
             "r6_protocol_path": "r6.json", "r6_protocol_sha256": collect.r7.R6_SHA,
             "r7_protocol_path": "r7.json", "r7_protocol_sha256":
             "774b4a7119b32bde359d5a6e5b789b69badd6c899bf15eebf5e5825b6e6a59c9",
             "catalog_sha256": "c" * 64, "decisions": 100000, "capacity": 100000,
             "n_step": 3, "gamma": .99, "collection_noise_std": .05,
             "environment": {"frame_skip": 4, "max_steps": 2000, "obstacles": True,
                             "raw_reward": True, "track_ids": [1, 2, 3, 4]},
             "runtime": {"python": sys.version.split()[0]},
             "code_sha256": {name: "e" * 64 for name in (
                 "scripts/collect_drq_final_source_replay.py", "train.py",
                 "common_adapter.py", "drq_v2.py")},
             "sources": {str(seed): {"source_checkpoint_sha256": "a" * 64,
                 "source_actor_sha256": "b" * 64, "original_ledger_sha256": "d" * 64,
                 "schedule_sha256": collect.schedule_sha256(schedule),
                 "collection_rng_seeds": {"action_noise_seed": 50 + seed}}
                 for seed in (0, 1)}}
    protocol_path = tmp_path / "collection.json"
    protocol_path.write_text(json.dumps(study), encoding="utf-8")
    monkeypatch.setattr(collect.r6, "_validate_protocol", lambda *a: None)
    monkeypatch.setattr(collect.r6, "_validate_catalog", lambda *a: None)
    monkeypatch.setattr(collect.r6, "_checkpoint_source_paths", lambda *a: None)
    monkeypatch.setattr(collect.r7, "pinned", lambda root, name, digest: root / name)
    monkeypatch.setattr(collect, "_original_ledger", lambda *a: ledger)
    monkeypatch.setattr(collect.torch, "load", lambda *a, **k: {"format": "haic-drq-v2-checkpoint-v1",
                                                                  "environment_steps": 131072,
                                                                  "replay": state})
    monkeypatch.setattr(collect, "build_env", lambda **k: pytest.fail("environment reset"))
    with pytest.raises(ValueError, match="prospective cross-lane audit unavailable"):
        collect.load_protocol(tmp_path, protocol_path)
    study["cross_lane_audit"] = {"path": "audit.json", "sha256": "f" * 64}
    (tmp_path / "audit.json").write_text(json.dumps({"format":
        "haic-drq-final-source-cross-lane-audit-v1", "passed": False,
        "partition": "TRAIN"}), encoding="utf-8")
    protocol_path.write_text(json.dumps(study), encoding="utf-8")
    with pytest.raises(ValueError, match="cross-lane audit did not clear"):
        collect.load_protocol(tmp_path, protocol_path)


def test_collector_exhaustion_preserves_partial_ledgers_without_pool_or_success_receipt(tmp_path, monkeypatch):
    class Actor(torch.nn.Module):
        def forward(self, observation):
            return torch.zeros((len(observation), 3), dtype=torch.float32)

    class FakeEnv:
        def __init__(self, track_id, seed):
            self.track_id, self.seed = track_id, seed
            self.closed = False

        def reset(self, *, seed=None, options=None):
            return np.zeros((4, 84, 84), dtype=np.float32), {
                "track_id": self.track_id, "seed": self.seed}

        def step(self, action):
            return np.zeros((4, 84, 84), dtype=np.float32), 1., True, False, {
                "track_id": self.track_id, "seed": self.seed, "finished": False}

        def close(self):
            self.closed = True

    root = tmp_path
    protocol = root / "collection.json"
    protocol.write_text("{}", encoding="utf-8")
    schedule = [{"original_episode_id": i + 100, "track_id": i + 1,
                 "geometry_seed": i + 31} for i in range(2)]
    sources = [{"learner_seed": 0, "checkpoint_path": "old.pt", "checkpoint_sha256": "a" * 64,
                "actor_path": "actor.pt", "actor_sha256": "b" * 64, "source_revision": "original"}]
    (root / "old.json").write_text(json.dumps({"source_replay": {
        "0": {"episode_ledger_path": "old-episodes.jsonl"}}}), encoding="utf-8")
    mock = {"r7_protocol_path": "old.json",
            "sources": {"0": {"collection_rng_seeds": {"action_noise_seed": 123},
                              "schedule_sha256": collect.schedule_sha256(schedule),
                              "original_ledger_sha256": "c" * 64}}}
    monkeypatch.setattr(collect, "load_protocol", lambda *a, **k: (mock, {"source_actors": sources},
                                                                   {0: schedule}, "d" * 64))
    monkeypatch.setattr(collect.r7, "pinned", lambda root, name, digest: root / name)
    monkeypatch.setattr(collect, "audit_source_actor_pair", lambda *a, **k: {})
    monkeypatch.setattr(collect, "load_exported_actor", lambda *a, **k: (Actor(), ActionAdapter(), None))
    built = []

    def build_env(**kwargs):
        env = FakeEnv(kwargs["track_id"], kwargs["seed"])
        built.append(env)
        return env

    monkeypatch.setattr(collect, "build_env", build_env)
    with pytest.raises(ValueError, match="finite original TRAIN reset schedule exhausted"):
        collect.collect(root, protocol, 0)
    output = root / collect.RUN_ROOT / "collection" / "seed0"
    failure = json.loads((output / "failure.json").read_text())
    assert failure["completed"] is False and failure["decisions"] == 2
    assert len((output / "steps.jsonl").read_text().splitlines()) == 2
    assert len((output / "episodes.jsonl").read_text().splitlines()) == 4
    assert not (output / "pool.pt").exists() and not (output / "receipt.json").exists()
    assert all(env.closed for env in built)


def test_preflight_only_never_builds_environment_or_trains(monkeypatch, capsys):
    monkeypatch.setattr(collect, "load_protocol", lambda *a, **k: (
        {}, {}, {0: [{"track_id": 1}], 1: [{"track_id": 2}]}, "e" * 64))
    monkeypatch.setattr(collect, "build_env", lambda **kwargs: pytest.fail("environment reset"))
    assert collect.main(["--preflight-only"]) == 0
    assert '"environment_decisions": 0' in capsys.readouterr().out
    monkeypatch.setattr(train, "load_protocol", lambda *a, **k: (
        {"source_replay": {"0": {}, "1": {}}}, {}, {}, {0: {}, 1: {}},
        {0: [], 1: []}, "f" * 64))
    monkeypatch.setattr(train, "load_source_pool", lambda *a, **k: None)
    monkeypatch.setattr(train, "train", lambda *a, **k: pytest.fail("learner update"))
    assert train.main(["--preflight-only"]) == 0
    assert '"learner_updates": 0' in capsys.readouterr().out
