"""Synthetic semantic-terminal and source-bound checks; never reset an environment."""

import hashlib
import json
import shutil
import sys

import numpy as np
import pytest
import torch

from scripts import diagnose_tdmpc2_terminal_probe as diagnosis


def _line(row):
    return (json.dumps(row, sort_keys=True) + "\n").encode()


def _fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnosis.cursor, "RUN", "runs/synthetic")
    run = tmp_path / "runs/synthetic"
    run.mkdir(parents=True)
    episodes = []
    snapshots = []
    steps = []
    for eid, (length, terminated, truncated) in enumerate(((5, True, False), (6, False, True))):
        images = np.full((length + 1, 4, 64, 64), eid, dtype=np.uint8)
        actions = np.zeros((length, 3), dtype=np.float32)
        actions[:, 0] = eid / 2
        rewards = np.full(length, .25, np.float32)
        flags = {key: np.zeros(length, dtype=np.bool_) for key in ("terminated", "truncated", "terminal")}
        flags["terminated"][-1] = terminated
        flags["truncated"][-1] = truncated
        flags["terminal"][-1] = terminated
        snapshots.append({"episode_id": eid, "start_step": 0, "observations": images,
                          "actions": actions, "rewards": rewards, **flags})
        before = sum(row["length"] for row in episodes)
        episodes.append({"event": "episode", "episode": eid, "length": length,
                         "decisions": before + length, "finished": False, "finish_time_s": None,
                         "terminated": terminated, "truncated": truncated, "terminal": terminated})
        for offset in range(length):
            steps.append({"decision": before + offset + 1, "episode": eid,
                          "action_f32_hex": actions[offset].tobytes().hex(), "reward": .25,
                          **{k: bool(v[offset]) for k, v in flags.items()}})
    step_bytes = b"".join(map(_line, steps))
    training_bytes = b"".join(map(_line, episodes))
    (run / "steps.jsonl").write_bytes(step_bytes)
    (run / "training.jsonl").write_bytes(training_bytes)
    replay = {"episodes": snapshots}
    pin = {"line": len(episodes), "row": {"episodes": 2, "decisions": 11},
           "ledger_prefix_sha256": hashlib.sha256(training_bytes).hexdigest(),
           "step_prefix_sha256": hashlib.sha256(step_bytes).hexdigest()}
    return replay, episodes, steps, pin


def test_terminal_vs_timeout_semantics_and_replay_step_binding(tmp_path, monkeypatch):
    replay, episodes, steps, pin = _fixture(tmp_path, monkeypatch)
    rows = diagnosis._episodes(tmp_path, pin, max_steps=6)
    assert rows == episodes
    diagnosis._validate_anchor(tmp_path, replay, rows, pin)
    ids = [(0, 2), (1, 3)]
    probe, tags, _ = diagnosis._batch(replay, rows, ids)
    assert probe["terminal"][:, :, 0].tolist() == [[0, 0], [0, 0], [1, 0]]
    assert tags[-1].tolist() == ["raw_termination", "time_limit_truncation"]
    steps[4]["terminal"] = False
    (tmp_path / "runs/synthetic/steps.jsonl").write_bytes(b"".join(map(_line, steps)))
    with pytest.raises(ValueError, match="semantic label differs"):
        diagnosis._validate_anchor(tmp_path, replay, rows, pin)
    episodes[1]["length"] = 5
    (tmp_path / "runs/synthetic/training.jsonl").write_bytes(b"".join(map(_line, episodes)))
    with pytest.raises(ValueError, match="not a time limit"):
        diagnosis._episodes(tmp_path, pin, max_steps=6)
    episodes[1].update(length=6, terminated=True, terminal=True)
    (tmp_path / "runs/synthetic/training.jsonl").write_bytes(b"".join(map(_line, episodes)))
    with pytest.raises(ValueError, match="ambiguous event class"):
        diagnosis._episodes(tmp_path, pin, max_steps=6)


def test_sealed_ledger_prefix_is_rechecked_after_cursor(tmp_path, monkeypatch):
    replay, episodes, steps, pin = _fixture(tmp_path, monkeypatch)
    (tmp_path / "runs/synthetic/training.jsonl").write_bytes(
        b"".join(map(_line, episodes)).replace(b'"event": "episode"', b'"event" : "episode"'))
    with pytest.raises(ValueError, match="prefix SHA"):
        diagnosis._episodes(tmp_path, pin, max_steps=6)
    (tmp_path / "runs/synthetic/steps.jsonl").write_bytes(
        b"".join(map(_line, steps)).replace(b'"decision":', b'"decision" :'))
    with pytest.raises(ValueError, match="prefix SHA"):
        diagnosis._validate_anchor(tmp_path, replay, episodes, pin)


def test_finish_is_positive_but_unconfirmed_finish_or_short_terminal_fails(tmp_path, monkeypatch):
    replay, episodes, _, pin = _fixture(tmp_path, monkeypatch)
    episodes[1].update(finished=True, finish_time_s=1.2, terminal=True)
    replay["episodes"][1]["terminal"][-1] = True
    updated = b"".join(map(_line, episodes))
    (tmp_path / "runs/synthetic/training.jsonl").write_bytes(updated)
    pin["ledger_prefix_sha256"] = hashlib.sha256(updated).hexdigest()
    rows = diagnosis._episodes(tmp_path, pin, max_steps=6)
    probe, tags, _ = diagnosis._batch(replay, rows, [(1, 3)])
    assert probe["terminal"][-1, 0, 0] == 1 and tags[-1, 0] == "finish"
    episodes[1]["finish_time_s"] = None
    (tmp_path / "runs/synthetic/training.jsonl").write_bytes(b"".join(map(_line, episodes)))
    with pytest.raises(ValueError, match="confirmed truncation"):
        diagnosis._episodes(tmp_path, pin, max_steps=6)
    episodes[0]["length"] = 2
    ep = replay["episodes"][0]
    for name in ("actions", "rewards", "terminated", "truncated", "terminal"):
        ep[name] = ep[name][:2].copy()
    ep["observations"] = ep["observations"][:3].copy()
    with pytest.raises(ValueError, match="no complete H3"):
        diagnosis._windows(replay, episodes, 834, diagnosis.ANCHOR_SHA256)


def test_balance_ids_deterministic_and_same_raw_window_bytes(tmp_path, monkeypatch):
    replay, rows, _, _ = _fixture(tmp_path, monkeypatch)
    first, meta = diagnosis._windows(replay, rows, 834, diagnosis.ANCHOR_SHA256)
    again, same = diagnosis._windows(replay, rows, 834, diagnosis.ANCHOR_SHA256)
    assert first == again and meta == same
    assert meta["positive_windows"] == meta["negative_windows"] == 1
    assert meta["eligible_negative_windows"] == 6
    assert len(set(first)) == len(first) == 2
    probe, tags, digest = diagnosis._batch(replay, rows, first)
    assert digest == diagnosis._batch(replay, rows, first)[2]
    assert int(probe["terminal"].sum()) == 1
    assert probe["terminal"].numel() == 6  # one positive across two balanced H3 windows
    replay["episodes"][0]["observations"][2, 0, 0, 0] ^= 1
    assert diagnosis._batch(replay, rows, first)[2] != digest
    assert tags.shape == (3, 2)
    rows[0]["length"] = 3
    ep = replay["episodes"][0]
    ep["observations"] = ep["observations"][:4]
    for name in ("actions", "rewards", "terminated", "truncated", "terminal"):
        ep[name] = ep[name][:3].copy()
    ep["terminated"][:] = [False, False, True]
    ep["terminal"][:] = ep["terminated"]
    with pytest.raises(ValueError, match="balanced"):
        diagnosis._windows({"episodes": [replay["episodes"][0]]}, rows[:1], 834, diagnosis.ANCHOR_SHA256)


def test_class_denominators_recall_specificity_and_missing_finish():
    logits = torch.tensor([[[2.], [-2.]], [[2.], [-2.]], [[2.], [2.]]])
    labels = torch.tensor([[[0.], [0.]], [[0.], [0.]], [[1.], [0.]]])
    classes = np.array([["ordinary_nonterminal", "ordinary_nonterminal"],
                        ["ordinary_nonterminal", "ordinary_nonterminal"],
                        ["raw_termination", "time_limit_truncation"]])
    result = diagnosis._summarize(logits, labels, classes)
    assert (result["positive_transitions"], result["negative_transitions"]) == (1, 5)
    assert result["positive_recall"] == 1
    assert result["specificity"] == pytest.approx(2 / 5)
    assert result["true_positive_count"] == 1 and result["true_negative_count"] == 2
    assert result["per_class"]["ordinary_nonterminal"]["specificity"] == .5
    assert result["per_class"]["time_limit_truncation"]["specificity"] == 0
    assert result["per_class"]["finish"]["count"] == 0
    assert result["per_class"]["finish"]["positive_recall"] is None
    assert result["bce"] > 0 and result["per_class"]["raw_termination"]["bce"] > 0
    classes[-1, 0] = "time_limit_truncation"
    with pytest.raises(ValueError, match="contradicts"):
        diagnosis._summarize(logits, labels, classes)


class _SyntheticModel:
    def eval(self):
        return self

    def encode(self, images, task):
        # Simulate the real pixel encoder's random shift without any environment import.
        return images.float().flatten(start_dim=-3).mean(-1, keepdim=True) + torch.randint(0, 7, images.shape[:-3]).unsqueeze(-1)

    def next(self, z, action, task):
        return z + action[:, :1]

    def termination(self, z, task, *, unnormalized):
        return z / 255


def test_prediction_repeated_rng_true_next_vs_predicted_without_environment(tmp_path, monkeypatch):
    replay, rows, _, _ = _fixture(tmp_path, monkeypatch)
    probe, _, _ = diagnosis._batch(replay, rows, [(0, 2), (1, 3)])
    rng = torch.get_rng_state().clone()
    env_modules = {key for key in sys.modules if key in ("env_wrapper", "gymnasium", "core.vendor.car_racing")}
    first = diagnosis._predict(_SyntheticModel(), probe, 834, batch_size=1)
    second = diagnosis._predict(_SyntheticModel(), probe, 834, batch_size=1)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert not torch.equal(first[0], first[1])  # distinct true-next/rolled-latent paths
    assert torch.equal(torch.get_rng_state(), rng)
    assert {key for key in sys.modules if key in ("env_wrapper", "gymnasium", "core.vendor.car_racing")} == env_modules


def test_source_tamper_fails_before_deserialization(tmp_path, monkeypatch):
    helper = tmp_path / diagnosis.HELPER
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b"tampered source")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("deserialized before source pin"))
    with pytest.raises(ValueError, match="helper SHA"):
        diagnosis._bind(tmp_path, (20000,))
    # Even with the helper intact, the frozen source map must reject a modified model.
    helper.write_bytes((diagnosis.ROOT / diagnosis.HELPER).read_bytes())
    protocol = tmp_path / diagnosis.cursor.PROTOCOL
    protocol.parent.mkdir(parents=True)
    protocol.write_bytes((diagnosis.ROOT / diagnosis.cursor.PROTOCOL).read_bytes())
    for source in diagnosis.cursor.SOURCE_PATHS:
        target = tmp_path / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(diagnosis.ROOT / source, target)
    (tmp_path / "haic/algorithms/tdmpc2/model.py").write_bytes(b"tampered model")
    with pytest.raises(ValueError, match="source hash mismatch: haic/algorithms/tdmpc2/model.py"):
        diagnosis._bind(tmp_path, (20000,))


def test_real_source_pins_and_default_unsealed_target_fail_before_torch_load(monkeypatch):
    assert diagnosis.cursor._digest(diagnosis.ROOT / diagnosis.HELPER) == diagnosis.HELPER_SHA256
    assert diagnosis.cursor._protocol(diagnosis.ROOT)["source_sha256"]
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("loaded missing checkpoint"))
    def unavailable(root, protocol, target):
        if target == 70000:
            raise ValueError("complete checkpoint ledger cursor not available")
        return {"checkpoint_sha256": diagnosis.ANCHOR_SHA256}
    monkeypatch.setattr(diagnosis.cursor, "_cursor", unavailable)
    with pytest.raises(ValueError, match="not available"):
        diagnosis._bind(diagnosis.ROOT, diagnosis.TARGETS)


def test_wrong_anchor_sha_is_rejected_after_cursor_before_deserialization(monkeypatch):
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("loaded unsealed checkpoint"))
    monkeypatch.setattr(diagnosis.cursor, "_protocol", lambda root: {})
    monkeypatch.setattr(diagnosis.cursor, "_cursor", lambda root, protocol, target: {"checkpoint_sha256": "b" * 64})
    with pytest.raises(ValueError, match="20k sealed checkpoint SHA"):
        diagnosis._bind(diagnosis.ROOT, (20000,))


def test_cli_exclusive_only_direct_source_bound_runs_output(tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs"
    runs.mkdir()
    output = runs / "tdmpc2-long-v2-terminal-20000-40000.json"
    monkeypatch.setattr(diagnosis, "ROOT", tmp_path)
    monkeypatch.setattr(diagnosis, "score", lambda **kwargs: {"checkpoints": [{"target": 20000}]})
    monkeypatch.setattr(sys, "argv", ["probe", "--targets", "20000", "40000", "--output", str(output)])
    diagnosis.main()
    assert json.loads(output.read_text())["checkpoints"][0]["target"] == 20000
    assert json.loads(capsys.readouterr().out)["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        diagnosis.main()
    monkeypatch.setattr(sys, "argv", ["probe", "--targets", "20000", "--output", str(runs / "other.json")])
    with pytest.raises(SystemExit):
        diagnosis.main()
