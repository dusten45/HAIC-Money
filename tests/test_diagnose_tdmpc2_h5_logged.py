"""Synthetic no-reset H5 calibration and trust-boundary tests."""

import builtins
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import diagnose_tdmpc2_h5_logged as h5


def _synthetic(tmp_path):
    episodes = []
    lines = []
    snapshots = []
    total = 0
    for eid, (length, reward_values, ending) in enumerate(((7, (1, 2, 3, 4, 5, 6, 7), "terminated"),
                                                           (5, (-1, -2, -3, -4, -5), "timeout"))):
        actions = np.full((length, 3), .25 + eid * .5, dtype=np.float32)
        rewards = np.asarray(reward_values, dtype=np.float32)
        terminated = np.zeros(length, dtype=np.bool_)
        truncated = np.zeros(length, dtype=np.bool_)
        terminal = np.zeros(length, dtype=np.bool_)
        if ending == "terminated":
            terminated[-1] = terminal[-1] = True
        else:
            truncated[-1] = True  # Plain time limit still bootstraps; not terminal.
        for offset in range(length):
            lines.append({"episode": eid, "decision": total + offset + 1,
                          "track_id": 1, "geometry_seed": h5.ROADS[eid],
                          "action_f32_hex": actions[offset].tobytes().hex(),
                          "native_action_f32_hex": actions[offset].tobytes().hex(),
                          "reward": float(rewards[offset]), "terminated": bool(terminated[offset]),
                          "truncated": bool(truncated[offset]), "terminal": bool(terminal[offset])})
        total += length
        episodes.append({"event": "episode", "episode": eid, "length": length, "decisions": total,
                         "updates": 0, "return": float(rewards.sum()),
                         "track_id": 1, "geometry_seed": h5.ROADS[eid],
                         "terminated": bool(terminated[-1]), "truncated": bool(truncated[-1]),
                         "terminal": bool(terminal[-1]),
                         "action_trace_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                         "native_action_trace_sha256": hashlib.sha256(actions.tobytes()).hexdigest()})
        snapshots.append({"episode_id": eid, "start_step": 0,
                          "observations": np.full((length + 1, 4, 64, 64), eid, np.uint8),
                          "actions": actions, "rewards": rewards, "terminated": terminated,
                          "truncated": truncated, "terminal": terminal})
    path = tmp_path / "steps.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in lines))
    replay = {"format": "haic-tdmpc2-episode-replay-v1", "capacity": 120000,
              "horizon": 3, "action_dim": 3, "augmentation_pad": 3, "include_partial": False,
              "bootstrap_on_truncation": True, "observation_shape": (4, 64, 64),
              "next_episode_id": 2, "size": 12, "active": None, "episodes": snapshots}
    return episodes, path, replay


def _bound_fixture(tmp_path, monkeypatch):
    """Small sealed RAW-shaped source; do not replace _prepared, _windows, or _cursor."""
    from scripts import diagnose_tdmpc2_checkpoint_losses as bound

    episodes, steps, replay = _synthetic(tmp_path)
    run = tmp_path / h5.RUN
    run.mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    (run / "steps.jsonl").write_bytes(steps.read_bytes())
    original_sources = {}
    for name in bound.SOURCE_PATHS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic pinned source: {name}".encode())
        original_sources[name] = h5._digest(path)
    for name, source in ((h5.HELPER, bound.__file__), ("scripts/diagnose_tdmpc2_h5_logged.py", h5.__file__)):
        path = tmp_path / name
        path.write_bytes(Path(source).read_bytes())
    training = {"seed": 733, "horizon": 3, "rho": .5, "discount": .995,
                "action_dim": 3, "observation_shape": [4, 64, 64], "episodic": True,
                "model_size": 5, "num_bins": 101, "augmentation_pad": 3,
                "batch_size": 256, "max_steps": 2000, "replay_capacity": 120000,
                "seed_steps": 10000, "pretrain_updates": 10000}
    source_protocol = {"format": "haic-tdmpc2-long-train-v1", "purpose": "consumed-TRAIN-development",
                       "source_sha256": original_sources, "run_dir": h5.RUN,
                       "selection": {"arm": "independent_3d", "action_dim": 3},
                       "cells": [{"track_id": 1, "geometry_seed": seed} for seed in h5.ROADS],
                       "episode_schedule": [0, 1, 2, 3],
                       "environment": {"track_id": 1, "obstacles": True, "frame_skip": 4, "reward": "raw"},
                       "checkpoint_targets": [10], "training": training}
    original_protocol = tmp_path / bound.PROTOCOL
    original_protocol.write_text(json.dumps(source_protocol))
    original_sha = h5._digest(original_protocol)
    monkeypatch.setattr(bound, "PROTOCOL_SHA256", original_sha)
    monkeypatch.setattr(bound, "TARGETS", (10,))
    monkeypatch.setattr(h5, "SOURCE_PROTOCOL_SHA", original_sha)
    monkeypatch.setattr(h5, "FINAL_TARGET", 10)
    monkeypatch.setattr(h5, "FINAL_DECISIONS", 12)
    monkeypatch.setattr(h5, "FINAL_UPDATES", 0)
    monkeypatch.setattr(h5, "FINAL_EPISODES", 2)
    monkeypatch.setattr(h5, "COUNT", 3)
    monkeypatch.setattr(h5, "END_ALIGNED", 2)
    checkpoint = run / "checkpoint-at-least-000010-step-000012.pt"
    checkpoint.write_bytes(b"synthetic trusted checkpoint; never deserialized")
    monkeypatch.setattr(h5, "CHECKPOINT", checkpoint.relative_to(tmp_path).as_posix())
    monkeypatch.setattr(h5, "CHECKPOINT_SHA", h5._digest(checkpoint))
    steps_sha = h5._digest(run / "steps.jsonl")
    monkeypatch.setattr(h5, "STEPS_SHA", steps_sha)
    rows = [{"event": "start", "protocol_sha256": original_sha,
             "source_sha256": original_sources, "resume_supported": False}]
    before = 0
    for ep in episodes:
        cell = {"track_id": ep["track_id"], "geometry_seed": ep["geometry_seed"]}
        rows.extend({"event": event, "episode": ep["episode"], "decisions": before, **cell}
                    for event in ("reset_intent", "reset"))
        rows.append(ep)
        before = ep["decisions"]
    checkpoint_row = {"event": "checkpoint", "target": 10, "decisions": 12, "updates": 0,
                      "episodes": 2, "rolling_update_count": 0, "train_episodes_since_previous": [{}, {}],
                      "path": checkpoint.name, "sha256": h5.CHECKPOINT_SHA,
                      "step_ledger_sha256": steps_sha}
    rows.append(checkpoint_row)
    ledger = run / "training.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(h5, "TRAIN_SHA", h5._digest(ledger))
    result = {"status": "completed_boundary_at_least_100k", "protocol_sha256": original_sha,
              "source_sha256": original_sources, "reused_train_only": True, "evaluation": None,
              "resume_supported": False, "decisions": 12, "updates": 0, "episodes": 2,
              "action_dim": 3, "checkpoints": [{k: v for k, v in checkpoint_row.items() if k != "event"}],
              "training_ledger_sha256": h5.TRAIN_SHA, "step_ledger_sha256": steps_sha}
    (run / "result.json").write_text(json.dumps(result))
    monkeypatch.setattr(h5, "RESULT_SHA", h5._digest(run / "result.json"))
    return run, replay, checkpoint, tmp_path / "haic/algorithms/tdmpc2/model.py"


def test_selection_is_fixed_unique_episode_contained_and_outcome_blind():
    episodes = [{"episode": i, "length": 5 + i % 7} for i in range(40)]
    selected = h5._select(episodes, seed=92, count=16, end_aligned=4)
    assert selected == h5._select(episodes, seed=92, count=16, end_aligned=4)
    assert len(selected) == len(set(selected)) == 16
    assert sum(start == episodes[eid]["length"] - 5 for eid, start in selected) >= 4
    assert all(0 <= start <= episodes[eid]["length"] - 5 for eid, start in selected)
    with pytest.raises(ValueError, match="complete H5"):
        h5._select([{"episode": 0, "length": 4}], count=1, end_aligned=1)


def test_same_anchor_h3_h5_discount_reward_baseline_and_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(h5, "COUNT", 3)
    episodes, steps, replay = _synthetic(tmp_path)
    windows = h5._windows(episodes, [(0, 0), (0, 2), (1, 0)], steps)
    _, anchor_hashes = h5._replay({"replay": replay}, episodes, windows)
    assert len(anchor_hashes) == 3
    targets = h5._targets(windows)
    assert targets["h3"]["terminal_positive_transitions"] == 0
    assert targets["h5"]["terminal_positive_transitions"] == 1
    assert targets["h5"]["truncated_positive_transitions"] == 1
    assert targets["h5"]["ordinary_timeout_transitions"] == 1
    actual = np.asarray([w["raw_rewards"] for w in windows], dtype=np.float64).T
    predicted = actual + .25
    probability = np.zeros_like(predicted)
    probability[4, 1:] = .75
    summary, rows = h5._metrics(windows, predicted, probability)
    for horizon in (3, 5):
        weights = h5.GAMMA ** np.arange(horizon)
        assert rows[1][f"h{horizon}"]["actual_discounted_reward"] == pytest.approx(
            sum(weights[t] * value for t, value in enumerate(windows[1]["raw_rewards"][:horizon])))
        assert summary[f"h{horizon}"]["discounted_return_mae"] == pytest.approx(.25 * weights.sum())
        assert summary[f"h{horizon}"]["mae_per_discounted_step"] == pytest.approx(.25)
        assert summary[f"h{horizon}"]["constant_reward_discounted_return_mae"] == pytest.approx(
            np.mean(np.abs(weights @ actual[:horizon])))
    assert summary["h5"]["terminal_true_positive"] == 1
    assert summary["h5"]["terminal_false_positive"] == 1  # Timeout is not semantic termination.
    assert summary["h3"]["terminal_true_positive"] == 0


class _FixedRolloutModel(torch.nn.Module):
    def __init__(self, rewards, probabilities):
        super().__init__()
        self.cfg = SimpleNamespace(num_bins=0)
        self.rewards = torch.as_tensor(rewards, dtype=torch.float32)
        self.probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
        self.reward_depth = self.termination_depth = self.encode_calls = 0

    def encode(self, obs, task):
        self.encode_calls += 1
        return torch.zeros((len(obs), 1))

    def next(self, z, action, task):
        return z

    def reward(self, z, action, task):
        result = self.rewards[self.reward_depth, :, None]
        self.reward_depth += 1
        return result

    def termination(self, z, task):
        result = self.probabilities[self.termination_depth, :, None]
        self.termination_depth += 1
        return result


def test_predeclared_gate_fake_model_pass_failure_and_full_receipt(tmp_path):
    episodes, steps, replay = _synthetic(tmp_path)
    windows = h5._windows(episodes, [(0, 0), (0, 2), (1, 0)], steps)
    raw = np.asarray([w["raw_rewards"] for w in windows], dtype=np.float32).T
    prob = np.zeros_like(raw)
    prob[4, 1] = .75  # One semantic termination, no false positives.
    model = _FixedRolloutModel(raw, prob)
    predicted, terminal = h5._rollout(model, replay, windows, 834)
    summary, rows = h5._metrics(windows, predicted, terminal)
    checks = h5._quality_gate(summary["h5"], h5.QUALITY_GATE)
    assert model.encode_calls == 1
    assert checks["quality_gate_passed"] and checks["h5_terminal_tpr"] == 1
    assert checks["h5_terminal_fpr"] == 0 and checks["h5_terminal_positive_denominator"] == 1
    assert len(rows) == 3 and not checks["failure_reasons"]

    for bad_probability in (np.zeros_like(prob), np.ones_like(prob)):
        predicted, terminal = h5._rollout(_FixedRolloutModel(raw, bad_probability), replay, windows, 834)
        summary, rows = h5._metrics(windows, predicted, terminal)
        checks = h5._quality_gate(summary["h5"], h5.QUALITY_GATE)
        assert checks["h5_return_below_constant"]
        assert not checks["quality_gate_passed"] and len(rows) == 3
        assert "h5_semantic_terminal_tpr_not_strictly_above_fpr" in checks["failure_reasons"]

    predicted, terminal = h5._rollout(_FixedRolloutModel(np.zeros_like(raw), prob), replay, windows, 834)
    summary, rows = h5._metrics(windows, predicted, terminal)
    checks = h5._quality_gate(summary["h5"], h5.QUALITY_GATE)
    assert not checks["quality_gate_passed"] and len(rows) == 3
    assert "h5_reward_return_mae_not_strictly_below_zero_constant" in checks["failure_reasons"]

    broken = raw.copy()
    broken[4, 1] = np.nan
    predicted, terminal = h5._rollout(_FixedRolloutModel(broken, prob), replay, windows, 834)
    summary, rows = h5._metrics(windows, predicted, terminal)
    checks = h5._quality_gate(summary["h5"], h5.QUALITY_GATE)
    assert not checks["quality_gate_passed"] and summary["h5"]["discounted_return_mae"] is None
    assert rows[1]["predicted_rewards"][4] is None and len(rows) == 3
    json.dumps({"horizons": summary, "windows": rows, "quality_gate_checks": checks}, allow_nan=False)


def test_planner_strict_point_five_and_depth_finish_timeout_labels(tmp_path):
    episodes, steps, _ = _synthetic(tmp_path)
    windows = h5._windows(episodes, [(0, 0), (0, 2), (1, 0)], steps)
    finished = {**windows[2], "episode_id": 3, "terminal": (False, False, False, False, True)}
    windows = [*windows, finished]
    rewards = np.asarray([w["raw_rewards"] for w in windows], dtype=np.float64).T
    probability = np.zeros_like(rewards)
    probability[4] = [.5, .5, .5, .5]
    summary, _ = h5._metrics(windows, rewards, probability)
    h5_depth = summary["h5"]["terminal_by_depth"]
    assert all(row["semantic_terminal_positive"] == 0 for row in h5_depth[:4])
    assert h5_depth[4]["raw_terminated_positive"] == 1
    assert h5_depth[4]["finished_truncation_positive"] == 1
    assert h5_depth[4]["ordinary_time_limit_positive"] == 1
    assert h5_depth[4]["predicted_positive_gt_0_5"] == 0
    assert (h5_depth[4]["true_positive"], h5_depth[4]["false_negative"]) == (0, 2)
    probability[4, 1] = .5001
    probability[4, 3] = .75
    summary, _ = h5._metrics(windows, rewards, probability)
    assert summary["h5"]["terminal_by_depth"][4]["true_positive"] == 2
    assert summary["h5"]["finished_truncation_true_positive"] == 1
    assert summary["h5"]["raw_terminated_true_positive"] == 1
    assert summary["h3"]["semantic_terminal_positive_transitions"] == 0


def test_replay_boundary_and_action_bytes_fail_closed(tmp_path):
    episodes, steps, replay = _synthetic(tmp_path)
    selected = [(0, 2), (1, 0)]
    windows = h5._windows(episodes, selected, steps)
    with pytest.raises(ValueError, match="missing"):
        h5._windows(episodes, [(0, 3)], steps)  # Would extend beyond the first reset.
    replay["episodes"][0]["terminal"][3] = True
    with pytest.raises(ValueError, match="crosses a reset"):
        h5._replay({"replay": replay}, episodes, windows)
    replay["episodes"][0]["terminal"][3] = False
    windows[0]["actions_bytes"] = b"\0" * 60
    with pytest.raises(ValueError, match="differs from logged"):
        h5._replay({"replay": replay}, episodes, windows)


def test_degenerate_h5_terminal_or_return_targets_rejected(monkeypatch):
    monkeypatch.setattr(h5, "COUNT", 2)
    good = {"raw_rewards": (1., 2., 3., 4., 5.), "terminated": (False,) * 5,
            "truncated": (False,) * 5, "terminal": (False,) * 5}
    bad = {**good, "raw_rewards": (-1., -2., -3., -4., -5.)}
    with pytest.raises(ValueError, match="semantic-terminal"):
        h5._targets([good, bad])
    bad = {**good, "terminal": (False, False, False, False, True)}
    with pytest.raises(ValueError, match="degenerate H3"):
        h5._targets([good, bad])


class _StochasticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(num_bins=0)
        self.encode_count = 0

    def encode(self, obs, task):
        self.encode_count += 1
        return torch.rand((len(obs), 1)) + obs[:, :1, :1, :1].reshape(len(obs), 1).float()

    def next(self, z, action, task):
        return z + action[:, :1]

    def reward(self, z, action, task):
        return z + action[:, :1]

    def termination(self, z, task):
        return z.sigmoid()


def test_anchor_encodes_once_and_seeded_augmentation_rng_is_restored(tmp_path):
    episodes, steps, replay = _synthetic(tmp_path)
    windows = h5._windows(episodes, [(0, 2), (1, 0)], steps)
    model = _StochasticModel()
    saved = torch.get_rng_state().clone()
    first = h5._rollout(model, replay, windows, 42)
    assert torch.equal(torch.get_rng_state(), saved)
    torch.manual_seed(999)
    second = h5._rollout(model, replay, windows, 42)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert model.encode_count == 2  # Once per window batch, once per invocation.
    assert first[0].shape == first[1].shape == (5, 2)


def test_real_tiny_cpu_world_model_can_predict_five_steps_without_mutating_weights(tmp_path):
    from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel

    episodes, steps, replay = _synthetic(tmp_path)
    windows = h5._windows(episodes, [(0, 2), (1, 0)], steps)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": (4, 64, 64)},
                                               num_channels=1, latent_dim=16, mlp_dim=16,
                                               num_bins=11, episodic=True))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        first = h5._rollout(model, replay, windows, 834)
        assert torch.equal(torch.get_rng_state(), rng)
        assert all(np.array_equal(left, right) for left, right in zip(first, h5._rollout(model, replay, windows, 834)))
        assert all(np.isfinite(values).all() and values.shape == (5, 2) for values in first)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    finally:
        torch.set_num_threads(threads)


def test_integrated_synthetic_source_cursor_windows_and_replay(tmp_path, monkeypatch):
    _, replay, _, _ = _bound_fixture(tmp_path, monkeypatch)
    proposal, windows, context = h5._prepared(tmp_path)
    assert len(windows) == 3
    assert context["targets"]["h5"]["terminal_positive_transitions"] == 1
    assert context["pin"]["step_prefix_sha256"] == h5.STEPS_SHA
    _, anchor_hashes = h5._replay({"replay": replay}, context["episodes"], windows)
    assert len(anchor_hashes) == 3
    assert proposal["selection"]["quality_gate"] == h5.QUALITY_GATE


@pytest.mark.parametrize("tamper", ["selected_action", "producer_source", "checkpoint", "protocol"])
def test_integrated_tamper_refused_before_torch_load(tmp_path, monkeypatch, tamper):
    run, _, checkpoint, source = _bound_fixture(tmp_path, monkeypatch)
    proposal, windows, _ = h5._prepared(tmp_path)
    frozen = tmp_path / h5.PROTOCOL
    frozen.write_text(json.dumps(proposal))
    frozen_sha = h5._digest(frozen)
    if tamper == "selected_action":
        lines = (run / "steps.jsonl").read_text().splitlines()
        selected = windows[0]
        index = sum(7 if eid == 0 else 5 for eid in range(selected["episode_id"])) + selected["start_step"]
        row = json.loads(lines[index])
        row["action_f32_hex"] = np.ones(3, np.float32).tobytes().hex()
        lines[index] = json.dumps(row)
        (run / "steps.jsonl").write_text("\n".join(lines) + "\n")
    elif tamper == "producer_source":
        source.write_bytes(b"modified original model source")
    elif tamper == "checkpoint":
        checkpoint.write_bytes(b"modified untrusted pickle")
    else:
        frozen.write_bytes(b"changed H5 draft")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("untrusted pickle was loaded"))
    with pytest.raises(ValueError, match="SHA mismatch|source hash mismatch|changed before torch.load"):
        h5.score(root=tmp_path, protocol_sha256=frozen_sha)


def test_rehashed_selected_action_still_fails_real_step_episode_trace_before_load(tmp_path, monkeypatch):
    run, _, _, _ = _bound_fixture(tmp_path, monkeypatch)
    proposal, windows, _ = h5._prepared(tmp_path)
    frozen = tmp_path / h5.PROTOCOL
    frozen.write_text(json.dumps(proposal))
    steps = run / "steps.jsonl"
    lines = steps.read_text().splitlines()
    selected = windows[0]
    index = (7 if selected["episode_id"] else 0) + selected["start_step"]
    row = json.loads(lines[index])
    row["action_f32_hex"] = np.ones(3, np.float32).tobytes().hex()
    lines[index] = json.dumps(row)
    steps.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(h5, "STEPS_SHA", h5._digest(steps))
    ledger = run / "training.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[-1]["step_ledger_sha256"] = h5.STEPS_SHA
    ledger.write_text("".join(json.dumps(item) + "\n" for item in rows))
    monkeypatch.setattr(h5, "TRAIN_SHA", h5._digest(ledger))
    result = run / "result.json"
    summary = json.loads(result.read_text())
    summary["step_ledger_sha256"] = summary["checkpoints"][-1]["step_ledger_sha256"] = h5.STEPS_SHA
    summary["training_ledger_sha256"] = h5.TRAIN_SHA
    result.write_text(json.dumps(summary))
    monkeypatch.setattr(h5, "RESULT_SHA", h5._digest(result))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("mismatched source action loaded"))
    with pytest.raises(ValueError, match="episode row differs from step-ledger actions"):
        h5.score(root=tmp_path, protocol_sha256=h5._digest(frozen))


def test_default_preflight_uses_real_binder_without_deserialize_or_env(tmp_path, monkeypatch, capsys):
    _bound_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(h5, "ROOT", tmp_path)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("preflight deserialized"))
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.endswith("haic_env") or name == "env_wrapper" or name.endswith("train_tdmpc2_long"):
            pytest.fail("preflight imported an environment/learner runner")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr("sys.argv", ["h5-calibration"])
    h5.main()
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "preflight_only"
    assert (output["environment_resets"], output["torch_load_calls"]) == (0, 0)
    assert output["window_count"] == 3
    assert output["target_coverage"]["h5"]["terminal_positive_transitions"] == 1
    assert not (tmp_path / h5.PROTOCOL).exists()


def test_checkpoint_same_descriptor_hash_and_load_survives_path_swap(tmp_path, monkeypatch):
    _, _, checkpoint, _ = _bound_fixture(tmp_path, monkeypatch)
    proposal, _, _ = h5._prepared(tmp_path)
    frozen = tmp_path / h5.PROTOCOL
    frozen.write_text(json.dumps(proposal))
    old_bytes = checkpoint.read_bytes()
    original_digest = h5._digest_stream
    visits = 0

    def swap_after_hash(stream):
        nonlocal visits
        digest = original_digest(stream)
        if stream.name == str(checkpoint):
            visits += 1
            if visits == 2:  # First: _prepared. Second: score's final, open-descriptor check.
                replacement = checkpoint.with_name("replacement.pt")
                replacement.write_bytes(b"replaced checkpoint path, never deserialized")
                replacement.replace(checkpoint)
        return digest

    def assert_original_open_file(stream, **kwargs):
        assert stream.read() == old_bytes
        assert checkpoint.read_bytes() != old_bytes
        raise RuntimeError("same sealed descriptor used")

    monkeypatch.setattr(h5, "_digest_stream", swap_after_hash)
    monkeypatch.setattr(torch, "load", assert_original_open_file)
    with pytest.raises(RuntimeError, match="same sealed descriptor"):
        h5.score(root=tmp_path, protocol_sha256=h5._digest(frozen))
    assert visits == 2


def test_synthetic_score_keeps_full_receipt_when_branch_design_gate_fails(tmp_path, monkeypatch):
    from haic.algorithms.tdmpc2 import model as model_module

    _, replay, checkpoint, _ = _bound_fixture(tmp_path, monkeypatch)
    proposal, windows, context = h5._prepared(tmp_path)
    frozen = tmp_path / h5.PROTOCOL
    frozen.write_text(json.dumps(proposal))  # Synthetic fixture only, never main's draft.
    raw = np.asarray([w["raw_rewards"] for w in windows], dtype=np.float32).T
    probability = np.zeros_like(raw)
    probability[4, next(i for i, w in enumerate(windows) if w["terminal"][4])] = .75
    fake_state = {"format": "haic-tdmpc2-long-train-v1", "protocol_sha256": h5.SOURCE_PROTOCOL_SHA,
                  "source_sha256": context["original_sources"], "target": h5.FINAL_TARGET,
                  "decisions": h5.FINAL_DECISIONS, "episodes": h5.FINAL_EPISODES,
                  "updates": h5.FINAL_UPDATES, "action_dim": 3, "resume_supported": False,
                  "replay": replay, "learner": {"q_scale": torch.ones(1)}}

    def fake_load(stream, **kwargs):
        assert stream.read() == checkpoint.read_bytes()
        assert kwargs["map_location"] == "cpu"
        return fake_state

    monkeypatch.setattr(torch, "load", fake_load)
    monkeypatch.setattr(model_module, "WorldModel", lambda cfg: _FixedRolloutModel(raw, probability))
    first = h5.score(root=tmp_path, protocol_sha256=h5._digest(frozen))
    assert first["quality_gate_passed"] and len(first["windows"]) == 3
    assert first["horizons"]["h5"]["terminal_by_depth"][-1]["true_positive"] == 1
    assert "plain timeouts" in first["terminal_label_limitations"][1]
    assert first["environment_resets"] == first["optimizer_steps"] == 0

    monkeypatch.setattr(model_module, "WorldModel", lambda cfg: _FixedRolloutModel(raw, np.ones_like(probability)))
    failed = h5.score(root=tmp_path, protocol_sha256=h5._digest(frozen))
    assert not failed["quality_gate_passed"]
    assert failed["quality_gate_checks"]["h5_terminal_tpr"] == 1
    assert failed["quality_gate_checks"]["h5_terminal_fpr"] == 1
    assert len(failed["windows"]) == 3 and failed["horizons"]["h5"]["windows"] == 3
    json.dumps(failed, allow_nan=False)


def test_score_cli_requires_frozen_protocol_and_unused_receipt(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(h5, "ROOT", tmp_path)
    monkeypatch.setattr(h5, "score", lambda **kwargs: pytest.fail("score ran without valid output"))
    monkeypatch.setattr("sys.argv", ["h5-calibration", "--score"])
    with pytest.raises(SystemExit):
        h5.main()
    used = tmp_path / "runs/tdmpc2-long-v2-h5-logged-test.json"
    used.write_text("existing receipt")
    monkeypatch.setattr("sys.argv", ["h5-calibration", "--score", "--protocol-sha256", "a" * 64,
                                        "--output", str(used)])
    with pytest.raises(SystemExit):
        h5.main()
    assert used.read_text() == "existing receipt"
