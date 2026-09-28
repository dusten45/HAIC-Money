"""Synthetic only: no checkpoint pickle or HAIC environment interaction."""

import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from haic.algorithms.tdmpc2.model import ShiftAug
from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner
from scripts import diagnose_tdmpc2_freeze_parity as diagnostic


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def synthetic_source(tmp_path, monkeypatch):
    """The validator is stubbed only AFTER a synthetic three-file SHA gate."""
    for name, payload in ((diagnostic.PROTOCOL, b"protocol"),
                          (f"{diagnostic.RUN}/result.json", b"result"),
                          ("scripts/evaluate_tdmpc2_full_train.py", b"binder"),
                          (diagnostic.CHECKPOINT, b"checkpoint")):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    monkeypatch.setattr(diagnostic, "PROTOCOL_SHA", digest(tmp_path / diagnostic.PROTOCOL))
    monkeypatch.setattr(diagnostic, "RESULT_SHA", digest(tmp_path / diagnostic.RUN / "result.json"))
    monkeypatch.setattr(diagnostic, "BINDER_SHA", digest(tmp_path / "scripts/evaluate_tdmpc2_full_train.py"))
    monkeypatch.setattr(diagnostic, "CHECKPOINT_SHA", digest(tmp_path / diagnostic.CHECKPOINT))
    from scripts import evaluate_tdmpc2_full_train as binder

    calls = []

    def source(root, ref):
        calls.append("source")
        assert ref["source"]["protocol"]["sha256"] == diagnostic.PROTOCOL_SHA
        assert ref["source"]["result"]["sha256"] == diagnostic.RESULT_SHA
        return ({"run_dir": diagnostic.RUN, "source_sha256": {}},
                {"decisions": diagnostic.DECISIONS, "updates": diagnostic.DECISIONS,
                 "episodes": diagnostic.EPISODES, "training_ledger_sha256": diagnostic.TRAIN_SHA,
                 "step_ledger_sha256": diagnostic.STEP_SHA,
                 "checkpoints": [{"sha256": diagnostic.CHECKPOINT_SHA}]}, root / diagnostic.RUN)

    def ledger(root, protocol, result, run, ref):
        calls.append("ledger")
        return [{"target": 100000, "decisions": diagnostic.DECISIONS,
                 "updates": diagnostic.DECISIONS, "episodes": diagnostic.EPISODES,
                 **ref["checkpoints"][0]}]

    monkeypatch.setattr(binder, "_source", source)
    monkeypatch.setattr(binder, "_ledger", ledger)
    return tmp_path, calls


@pytest.mark.parametrize("tamper", ["protocol", "result", "binder", "ledger", "checkpoint"])
def test_sha_tamper_and_cursor_rejection_precede_torch_load(synthetic_source, monkeypatch, tamper):
    root, calls = synthetic_source
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("unverified pickle loaded"))
    if tamper == "ledger":
        from scripts import evaluate_tdmpc2_full_train as binder

        def reject(*args):
            raise ValueError("full step-ledger cursor drift")

        monkeypatch.setattr(binder, "_ledger", reject)
    elif tamper == "checkpoint":
        monkeypatch.setattr(diagnostic, "CHECKPOINT_SHA", "0" * 64)
    else:
        name = {"protocol": diagnostic.PROTOCOL, "result": f"{diagnostic.RUN}/result.json",
                "binder": "scripts/evaluate_tdmpc2_full_train.py"}[tamper]
        (root / name).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA|ledger|cursor|checkpoint"):
        bound = diagnostic._bound_files(root)
        diagnostic._load_checkpoint(bound)
    if tamper in ("protocol", "result", "binder"):
        assert calls == []
    elif tamper == "ledger":
        assert calls == ["source"]


def test_valid_synthetic_source_binds_all_cursors_before_deserialization(synthetic_source, monkeypatch):
    root, calls = synthetic_source
    bound = diagnostic._bound_files(root)
    assert calls == ["source", "ledger"]
    seen = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: seen.append("loaded") or {})
    with pytest.raises(ValueError, match="checkpoint payload"):
        diagnostic._load_checkpoint(bound)
    assert seen == ["loaded"]


@pytest.fixture
def archived_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "DECISIONS", 42)
    monkeypatch.setattr(diagnostic, "EPISODES", 2)
    monkeypatch.setattr(diagnostic, "WINDOWS", (("seed-random-ep0", 0, 16, (0, 0)),
                                             ("seed-random-ep1", 1, 16, (-1, 2))))
    folder = tmp_path / diagnostic.RUN
    folder.mkdir(parents=True)
    episodes, rows, steps = [], [], []
    for eid in (0, 1):
        images = np.zeros((22, 4, 64, 64), np.uint8)
        images[:, :, 20, 21] = np.arange(22, dtype=np.uint8)[:, None] + eid
        actions = np.tile(np.array([[.25, -.5, .5]], np.float32), (21, 1))
        rewards = np.full(21, .5, np.float32)
        terminated = np.zeros(21, np.bool_)
        terminated[-1] = True
        ep = {"episode_id": eid, "start_step": 0, "observations": images,
              "actions": actions, "rewards": rewards, "terminated": terminated,
              "truncated": np.zeros(21, np.bool_), "terminal": terminated.copy()}
        episodes.append(ep)
        rows.append({"event": "episode", "episode": eid, "length": 21, "decisions": (eid + 1) * 21,
                     "track_id": 1, "geometry_seed": 3910800001 + eid * 3,
                     "finished": False, "terminated": True, "truncated": False, "terminal": True,
                     "action_trace_sha256": hashlib.sha256(actions.tobytes()).hexdigest()})
        steps.extend({"decision": eid * 21 + t + 1, "episode": eid,
                      "geometry_seed": rows[-1]["geometry_seed"],
                      "action_f32_hex": actions[t].tobytes().hex(), "reward": .5,
                      "terminated": bool(terminated[t]), "truncated": False,
                      "terminal": bool(terminated[t])} for t in range(21))
    (folder / "training.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (folder / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in steps))
    monkeypatch.setattr(diagnostic, "TRAIN_SHA", digest(folder / "training.jsonl"))
    monkeypatch.setattr(diagnostic, "STEP_SHA", digest(folder / "steps.jsonl"))
    replay = {"format": "haic-tdmpc2-episode-replay-v1", "capacity": 120000, "horizon": 3,
              "action_dim": 3, "augmentation_pad": 3, "observation_shape": (4, 64, 64),
              "active": None, "size": 42, "next_episode_id": 2, "episodes": episodes}
    from scripts import evaluate_tdmpc2_full_train as binder

    bound = {"root": tmp_path, "binder": binder, "protocol": {"cells": [
        {"track_id": 1, "geometry_seed": 3910800001},
        {"track_id": 1, "geometry_seed": 3910800004}]}}
    return {"replay": replay}, bound, steps


def test_fixed_complete_h3_windows_bind_exact_pixel_and_action_bytes(archived_windows):
    state, bound, _ = archived_windows
    windows = diagnostic._windows(state, bound)
    assert [(w["episode_id"], w["start_step"], w["shift_yx"]) for w in windows] == [
        (0, 16, (0, 0)), (1, 16, (-1, 2))]
    assert all(w["obs"].shape == (4, 4, 64, 64) and w["actions"].shape == (3, 3) for w in windows)
    assert all(w["pixel_sha256"] == hashlib.sha256(w["obs"].tobytes()).hexdigest() for w in windows)
    assert all(w["action_bytes_hex"] == w["actions"].tobytes().hex() for w in windows)


def test_changed_archived_step_or_replay_action_fails_closed(archived_windows):
    state, bound, steps = archived_windows
    steps[16]["action_f32_hex"] = np.zeros(3, np.float32).tobytes().hex()
    (bound["root"] / diagnostic.RUN / "steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in steps))
    with pytest.raises(ValueError, match="ledger"):
        diagnostic._windows(state, bound)
    steps[16]["action_f32_hex"] = state["replay"]["episodes"][0]["actions"][16].tobytes().hex()
    (bound["root"] / diagnostic.RUN / "steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in steps))
    state["replay"]["episodes"][0]["actions"][16, 0] = .9
    with pytest.raises(ValueError, match="TRAIN ledger"):
        diagnostic._windows(state, bound)


class ToyModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = SimpleNamespace(num_bins=0)
        self.w = nn.Parameter(torch.tensor(.2))
        self.shift = ShiftAug(pad=3)
        self._encoder = nn.ModuleDict({"rgb": nn.Sequential(self.shift)})

    def encode(self, obs, task=None):
        pixels: torch.Tensor = self._encoder["rgb"](obs)
        return torch.stack((pixels[:, 0].mean((1, 2)) * self.w,
                            pixels[:, 1].mean((1, 2)) * self.w), dim=-1)

    def next(self, z, action, task=None):
        return z + action[:, :2] * self.w

    def _pi(self, z):
        return torch.cat((z, z[:, :1], torch.zeros((z.shape[0], 3), device=z.device)), dim=-1)

    def reward(self, z, action, task=None):
        return (z[:, :1] + action[:, :1]) * self.w

    def Q(self, z, action, task=None, return_type="all"):
        assert return_type == "all"
        return self.reward(z, action).unsqueeze(0).expand(5, -1, -1)

    def termination(self, z, task=None, unnormalized=False):
        assert unnormalized
        return z[:, :1] * self.w

    def reset(self):
        pytest.fail("environment/model reset is forbidden")

    def update(self, *args):
        pytest.fail("optimizer update is forbidden")


def test_explicit_shift_fixture_restores_encoder_and_preserves_rng(archived_windows):
    state, bound, _ = archived_windows
    window = diagnostic._windows(state, bound)[1]
    cpu, same_weights = ToyModel().eval(), ToyModel().eval()
    same_weights.load_state_dict(cpu.state_dict(), strict=True)
    torch.manual_seed(87392)
    global_rng = torch.get_rng_state().clone()
    before_np = np.random.get_state()
    before_python = random.getstate()
    first = diagnostic._forward(cpu, window, "cpu")
    assert torch.equal(torch.get_rng_state(), global_rng)
    torch.manual_seed(87392)
    second = diagnostic._forward(same_weights, window, "cpu")
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert all(np.array_equal(first[name], second[name]) for name in diagnostic.TOLERANCES)
    assert cpu.shift.forward.__func__ is ShiftAug.forward
    assert same_weights.shift.forward.__func__ is ShiftAug.forward
    assert random.getstate() == before_python
    np.testing.assert_equal(np.random.get_state(), before_np)


def test_aligned_pad_three_crop_and_l2_abs_statistics():
    pixels = torch.zeros(1, 4, 64, 64, dtype=torch.uint8)
    pixels[0, :, 20, 21] = 173
    shifted = diagnostic.aligned_shift(pixels, -1, 2)
    assert shifted.dtype == torch.float32 and shifted[0, 0, 21, 19] == 173
    with pytest.raises(ValueError, match="offset"):
        diagnostic.aligned_shift(pixels, 4, 0)
    left = np.zeros((2, 3), dtype=np.float32)
    right = np.array([[3., 4., 0.], [0., 0., 12.]], dtype=np.float32)
    stats = diagnostic.difference_stats(left, right, 5.)
    assert stats["max_abs"] == 12 and stats["mean_abs"] == pytest.approx(19 / 6)
    assert stats["vector_l2_max"] == 12 and stats["vector_l2_mean"] == 8.5
    assert stats["elements"] == 6 and not stats["within_tolerance"]


class TinyPlannerModel:
    def encode(self, obs, task=None):
        return obs.float()

    def pi(self, z, task=None):
        return torch.full((z.shape[0], 1), .25), {}

    def next(self, z, a, task=None):
        return z + a

    def reward(self, z, a, task=None):
        return a * .2

    def Q(self, z, a, task=None, return_type="avg"):
        assert return_type == "avg"
        return torch.ones((z.shape[0], 1))

    def reset(self):
        pytest.fail("environment reset is forbidden")


def test_planner_mode_toggle_is_final_noise_only_and_preserves_state_rng():
    planner = TDMPC2Planner(TinyPlannerModel(), PlannerConfig(
        action_dim=1, discount=.995, horizon=3, num_bins=0,
        num_samples=8, num_elites=2, num_pi_trajs=1,
        iterations=2, min_std=.05, max_std=.25))
    planner.prev_mean = torch.tensor([[.1], [.2], [.3]])
    previous = planner.prev_mean.clone()
    global_rng = torch.get_rng_state().clone()
    report = diagnostic._paired_plan(planner, torch.zeros(1, 1))
    assert report["selected_elite_bit_identical"] and report["selected_elite_index_identical"]
    assert report["prev_mean_bit_identical"] and report["final_additive_noise_only_verified"]
    assert not report["final_action_bit_identical"] and report["final_action_delta_l2"] > 0
    assert torch.equal(planner.prev_mean, previous)
    assert torch.equal(torch.get_rng_state(), global_rng)


def test_cpu_only_score_reports_unavailable_not_fabricated_gpu_proof(archived_windows, monkeypatch):
    state, bound, _ = archived_windows
    windows = diagnostic._windows(state, bound)
    import haic.algorithms.tdmpc2.model as model_module

    monkeypatch.setattr(diagnostic, "_bound_files", lambda root: {"protocol": {"source_sha256": {}},
                                                                     "pin": {}, "root": bound["root"]})
    monkeypatch.setattr(diagnostic, "_load_checkpoint", lambda bound: {"learner": {"model.w": torch.tensor(.2)}})
    monkeypatch.setattr(diagnostic, "_windows", lambda state, bound: windows)
    monkeypatch.setattr(model_module, "WorldModel", ToyModel)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    before = torch.get_rng_state().clone()
    report = diagnostic.score(root=bound["root"])
    assert report["environment_resets"] == report["optimizer_steps"] == 0
    assert report["cpu_gpu"]["status"] == "unavailable_no_parity_proof"
    assert report["cpu_gpu"]["within_all_tolerances"] is None
    assert report["cpu_gpu"]["aggregate"] is None
    assert report["cpu_gpu"]["per_window"] == [] and len(report["windows"]) == 2
    assert torch.equal(before, torch.get_rng_state())
