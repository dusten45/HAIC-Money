"""Synthetic-only overshoot runner checks; never construct or reset HAIC CarRacing."""

from copy import deepcopy
import hashlib
import json
import math
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haic.algorithms.tdmpc2.reward_overshoot import OvershootLearner, OvershootReplay
from scripts import train_tdmpc2_reward_overshoot as operator


def _root(tmp_path):
    for name in operator.SOURCE_PATHS:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(operator.ROOT / name, dest)
    for name in (operator.BASE_PROTOCOL[0], operator.BASE_RESULT[0], operator.AUDIT[0]):
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(operator.ROOT / name, dest)
    (tmp_path / "runs").mkdir(exist_ok=True)
    return tmp_path


def _protocol(root):
    original = operator.base._json(root / operator.BASE_PROTOCOL[0])
    return {
        "format": operator.FORMAT, "purpose": "consumed-TRAIN-development",
        "upstream_revision": operator.base.UPSTREAM,
        "source_sha256": {name: operator.base.digest(root / name) for name in operator.SOURCE_PATHS},
        "baseline_training_protocol": {"path": operator.BASE_PROTOCOL[0], "sha256": operator.BASE_PROTOCOL[1]},
        "baseline_training_result": {"path": operator.BASE_RESULT[0], "sha256": operator.BASE_RESULT[1]},
        "replay_audit": {"path": operator.AUDIT[0], "sha256": operator.AUDIT[1]},
        "throughput_benchmark": {"path": operator.BENCHMARK, "sha256": "a" * 64},
        "selection": deepcopy(original["selection"]), "cells": deepcopy(original["cells"]),
        "episode_schedule": deepcopy(original["episode_schedule"]),
        "environment": deepcopy(original["environment"]), "run_dir": operator.RUN_DIR,
        "training": deepcopy(original["training"]), "checkpoint_targets": deepcopy(original["checkpoint_targets"]),
        "resources": deepcopy(original["resources"]), "startup_gate": deepcopy(operator.STARTUP_GATE),
        "runtime": {"python": "3.11.14", "torch": "2.1.0+cu121", "numpy": "1.26.0",
                    "gymnasium": "0.29.1", "torch_cuda": "12.1", "device": "cuda",
                    "gpu_name": "Synthetic CUDA", "cuda_capability": [8, 9],
                    "cuda_visible_devices": None, "torch_num_threads": 2, "torch_num_interop_threads": 2},
        "seed_schedule": operator.base.SEED_SCHEDULE, "reward_overshoot": deepcopy(operator.OVERSHOOT),
    }


def _freeze(root, protocol):
    path = root / "experiments/tdmpc2-overshoot-test.json"
    path.write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")
    return path, operator.base.digest(path)


def _receipt(root, protocol):
    script = root / "scripts/benchmark_tdmpc2_reward_overshoot.py"
    script.write_text("synthetic benchmark producer only", encoding="utf-8")
    names = ("scripts/benchmark_tdmpc2_reward_overshoot.py",
             "haic/algorithms/tdmpc2/learner.py", "haic/algorithms/tdmpc2/model.py",
             "haic/algorithms/tdmpc2/replay.py", "haic/algorithms/tdmpc2/reward_overshoot.py")
    sources = {name: operator.base.digest(root / name) for name in names}
    inputs = {"base": "b" * 64, "variant": "b" * 64}
    t = 128
    base = {"mean_seconds": .01, "median_seconds": .01, "update_wall_seconds": [.01] * t}
    variant = {"mean_seconds": .02, "median_seconds": .02, "update_wall_seconds": [.02] * t}
    receipt = {
        "format": "haic-tdmpc2-reward-overshoot-throughput-v1", "status": "PASS",
        "environment_resets": 0, "real_replay_reads": 0,
        "expected_script_sha256": sources["scripts/benchmark_tdmpc2_reward_overshoot.py"],
        "output_path": operator.BENCHMARK,
        "source_sha256_before": sources, "source_sha256_after": sources,
        "input_sha256_before": inputs, "input_sha256_after": inputs,
        "sample_rng_sha256_before": {"base": "d" * 64, "variant": "d" * 64},
        "sample_rng_sha256_after": {"base": "e" * 64, "variant": "e" * 64},
        "runtime": {"torch": protocol["runtime"]["torch"],
                    "torch_cuda": protocol["runtime"]["torch_cuda"], "device": "cuda:0",
                    "cuda_name": protocol["runtime"]["gpu_name"], "cuda_visible_devices": None,
                    "torch_num_threads": 2, "torch_num_interop_threads": 2},
        "fixture": {"batch_size": 256, "horizon": 3, "action_dim": 3,
                    "obs_shape": [4, 64, 64], "model_size": 5, "num_bins": 101,
                    "valid_step4": 251, "valid_step5": 249, "synthetic_only": True,
                    "trainable_parameters": 5_000_000,
                    "proof_h3_sample_sha256": "f" * 64,
                    "proof_extended_sample_sha256": "1" * 64},
        "timing": {"scope": "full_update_with_replay_sample",
                   "synchronization": "torch.cuda.synchronize per update",
                   "warmup_updates_each": 32, "timed_updates_each": t,
                   "block_order": ["base", "variant", "variant", "base"] * 2,
                   "base": base, "variant": variant},
        "resources": {"cgroup_available_bytes_before": 32 * 1024**3,
                      "cgroup_available_bytes_after": 32 * 1024**3,
                      "disk_available_bytes_before": 32 * 1024**3,
                      "disk_available_bytes_after": 32 * 1024**3,
                      "cuda_free_bytes_before": 8 * 1024**3,
                      "cuda_free_bytes_after": 8 * 1024**3,
                      "cuda_total_bytes": 16 * 1024**3, "max_cuda_allocated_bytes": 1 * 1024**3,
                      "min_cgroup_available_bytes": 24 * 1024**3,
                      "min_disk_available_bytes": 24 * 1024**3,
                      "gpu_reserve_bytes": 2 * 1024**3},
        "forecast": {"baseline_elapsed_seconds": 17459.373508695047,
                     "baseline_updates": 100354,
                     "positive_update_delta_seconds": .01,
                     "forecast_seconds": 17459.373508695047 + 100354 * .01,
                     "threshold_seconds_exclusive": 20600,
                     "wall_cap_seconds": 21600, "passed": True},
    }
    return receipt


def _freeze_receipt(root, protocol, receipt):
    body = {key: value for key, value in receipt.items() if key != "body_sha256"}
    receipt["body_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    output = root / operator.BENCHMARK
    output.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    protocol["throughput_benchmark"]["sha256"] = operator.base.digest(output)


def _preflight_fixture(tmp_path, monkeypatch):
    root = _root(tmp_path)
    p = _protocol(root)
    original = operator.base._json(root / operator.BASE_PROTOCOL[0])
    monkeypatch.setattr(operator, "_evidence", lambda *_: original)
    monkeypatch.setattr(operator, "runtime", lambda: p["runtime"])
    monkeypatch.setattr(operator, "_startup_resources", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(operator.base, "_resources", lambda *_: None)
    monkeypatch.setattr(operator, "make_training_env", lambda *_: pytest.fail("preflight constructed HAIC env"))
    receipt = _receipt(root, p)
    _freeze_receipt(root, p, receipt)
    path, sha = _freeze(root, p)
    return root, p, receipt, path, sha


def test_preflight_validates_exact_protocol_source_and_exclusive_output(tmp_path, monkeypatch):
    root, p, receipt, path, sha = _preflight_fixture(tmp_path, monkeypatch)
    output = root / operator.RUN_DIR
    assert operator.preflight(path, sha, output, root=root) == p
    assert not output.exists()
    with pytest.raises(ValueError, match="protocol SHA mismatch"):
        operator.preflight(path, "0" * 64, output, root=root)
    with pytest.raises(ValueError, match="exclusive output"):
        operator.preflight(path, sha, root / "runs/tdmpc2-long-20260928-v2", root=root)
    for section, key, altered in (("training", "seed", 734), ("training", "decision_cap", 102001),
                                  ("training", "horizon", 5), ("resources", "max_wall_seconds", 21601),
                                  ("environment", "obstacles", False),
                                  ("reward_overshoot", "reward_target", "shaped"),
                                  ("startup_gate", "min_disk_available_bytes", 16 * 1024**3)):
        bad = deepcopy(p)
        bad[section][key] = altered
        modified, checksum = _freeze(root, bad)
        with pytest.raises(ValueError):
            operator.preflight(modified, checksum, output, root=root)
    path, sha = _freeze(root, p)
    (root / "env_wrapper.py").write_text("tampered source", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch: env_wrapper.py"):
        operator.preflight(path, sha, output, root=root)
    shutil.copyfile(operator.ROOT / "env_wrapper.py", root / "env_wrapper.py")
    (root / operator.BENCHMARK).write_text("tampered receipt", encoding="utf-8")
    with pytest.raises(ValueError, match="reference hash mismatch"):
        operator.preflight(path, sha, output, root=root)
    _freeze_receipt(root, p, receipt)
    path, sha = _freeze(root, p)
    output.mkdir()
    with pytest.raises(ValueError, match="exclusive output"):
        operator.preflight(path, sha, output, root=root)


def test_benchmark_report_rejects_missing_full_update_or_unsafe_forecast(tmp_path, monkeypatch):
    root, p, receipt, _, _ = _preflight_fixture(tmp_path, monkeypatch)
    for field, changed, message in (("timing", {"scope": "precomputed_batch_only"}, "counts or synchronization"),
                                    ("fixture", {"valid_step5": 0}, "active suffix"),
                                    ("forecast", {"passed": False}, "wall budget"),
                                    ("resources", {"cuda_free_bytes_after": 2 * 1024**3}, "reserve"),
                                    ("source_sha256_after", {"haic/algorithms/tdmpc2/reward_overshoot.py": "0" * 64}, "sources")):
        bad = deepcopy(receipt)
        bad[field].update(changed)
        _freeze_receipt(root, p, bad)
        path, sha = _freeze(root, p)
        with pytest.raises(ValueError, match=message):
            operator.preflight(path, sha, root / operator.RUN_DIR, root=root)
    bad = deepcopy(receipt)
    bad["body_sha256"] = "0" * 64
    (root / operator.BENCHMARK).write_text(json.dumps(bad), encoding="utf-8")
    p["throughput_benchmark"]["sha256"] = operator.base.digest(root / operator.BENCHMARK)
    path, sha = _freeze(root, p)
    with pytest.raises(ValueError, match="body SHA"):
        operator.preflight(path, sha, root / operator.RUN_DIR, root=root)


def test_startup_resource_floor_uses_raw_cgroup_disk_and_available_vram(tmp_path, monkeypatch):
    gate = operator.STARTUP_GATE
    original = type(tmp_path).read_text
    readings = {"memory.max": str(40 * 1024**3), "memory.current": str(18 * 1024**3)}

    def read(path, *args, **kwargs):
        if str(path).startswith("/sys/fs/cgroup/") and path.name in readings:
            return readings[path.name]
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "read_text", read)
    monkeypatch.setattr(operator.os, "statvfs", lambda _: SimpleNamespace(
        f_bavail=30 * 1024, f_frsize=1024**2))
    monkeypatch.setattr(operator.torch.cuda, "mem_get_info", lambda: (4 * 1024**3, 16 * 1024**3))
    output = tmp_path / "runs/unused"
    (tmp_path / "runs").mkdir()
    with pytest.raises(ValueError, match="cgroup"):
        operator._startup_resources(output, gate, cuda_peak=1 * 1024**3)
    readings["memory.current"] = str(10 * 1024**3)
    monkeypatch.setattr(operator.os, "statvfs", lambda _: SimpleNamespace(
        f_bavail=23 * 1024, f_frsize=1024**2))
    with pytest.raises(ValueError, match="disk"):
        operator._startup_resources(output, gate, cuda_peak=1 * 1024**3)
    monkeypatch.setattr(operator.os, "statvfs", lambda _: SimpleNamespace(
        f_bavail=30 * 1024, f_frsize=1024**2))
    with pytest.raises(ValueError, match="CUDA memory"):
        operator._startup_resources(output, gate, cuda_peak=3 * 1024**3)
    operator._startup_resources(output, gate, cuda_peak=1 * 1024**3)


def test_real_components_are_variant_only_and_keep_original_h3_planner_defaults():
    original = operator.base._json(operator.ROOT / operator.BASE_PROTOCOL[0])
    cfg = deepcopy(original["training"])
    cfg["device"] = "cpu"  # Synthetic construction only; never a TRAIN protocol.
    learner, replay, planner = operator._components({"training": cfg})
    assert type(learner) is OvershootLearner and learner.aux_enabled is True
    assert type(replay) is OvershootReplay and replay.capacity == 120000
    assert replay.horizon == replay.action_dim == planner.config.horizon == planner.config.action_dim == 3
    assert (planner.config.num_samples, planner.config.num_pi_trajs,
            planner.config.iterations, planner.config.num_elites) == (512, 24, 6, 64)
    assert learner.model.cfg.episodic is True and learner.model.cfg.num_bins == 101
    assert learner.cfg.batch_size == 256 and learner.cfg.reward_coef == .1
    assert learner.discount == .995


class FakeLearner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(1, 1)
        self.register_buffer("q_scale", torch.ones(1))
        self.optim = torch.optim.Adam(self.model.parameters())
        self.pi_optim = torch.optim.Adam(self.model.parameters())
        self.discount = .995
        self.samples = []

    def update(self, replay):
        assert type(replay) is OvershootReplay and replay.horizon == replay.action_dim == 3
        sample = replay.sample(4)
        self.samples.append((replay.num_episodes, int(sample["overshoot_mask"].sum().item())))
        return {"overshoot_reward_loss": 0.2, "reward_loss": 0.5,
                "total_loss": 1.0, "pi_entropy": -2.0}


class FakePlanner:
    def reset(self):
        pass


class FakeEnv:
    def __init__(self, *, failure=None):
        self.unwrapped = self
        self.track_id = self.track_seed = None
        self.length = self.resets = 0
        self.closed = False
        self.failure = failure

    def reset(self, *, seed, options):
        self.track_seed, self.track_id = seed, options["track_id"]
        self.length = 0
        self.resets += 1
        if self.failure == "wrong_road":
            self.track_seed += 1
        return np.full((4, 84, 84), .1, np.float32), {}

    def step(self, action):
        if self.failure == "interrupted" and self.resets == 4 and self.length == 0:
            raise RuntimeError("synthetic interrupted episode")
        assert action.shape == (3,) and np.isfinite(action).all()
        self.length += 1
        info = {"progress": .25, "damage": .1, "finished": False, "finish_time_s": None}
        terminated = truncated = False
        if self.failure == "bad_finish":
            info.update(finished=True, finish_time_s=1.0)
        if self.failure == "early_timeout":
            truncated = True
        if self.length == 5:
            truncated = True
            if self.track_seed == operator.base.ROADS[1]:
                info.update(finished=True, finish_time_s=1.0)
        return np.full((4, 84, 84), .2, np.float32), .25, terminated, truncated, info

    def close(self):
        self.closed = True


def _synthetic_run(tmp_path, monkeypatch, *, failure=None):
    root = _root(tmp_path)
    p = _protocol(root)
    p["training"].update(seed=7, seed_steps=10, pretrain_updates=10,
                         batch_size=4, max_steps=5, replay_capacity=35,
                         decision_cap=35, update_cap=35)
    monkeypatch.setattr(operator, "preflight", lambda *_args, **_kwargs: p)
    monkeypatch.setattr(operator, "_evidence", lambda *_: None)
    monkeypatch.setattr(operator, "_recheck", lambda *_: None)
    monkeypatch.setattr(operator, "_startup_resources", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(operator, "_benchmark", lambda *_: {"resources": {"max_cuda_allocated_bytes": 1}})
    monkeypatch.setattr(operator.base, "_resources", lambda *_: None)
    monkeypatch.setattr(operator.base, "TARGETS", [15, 20, 25, 30])
    learner = FakeLearner()
    replay = OvershootReplay(35, seed=7)
    monkeypatch.setattr(operator, "_components", lambda _: (learner, replay, FakePlanner()))
    monkeypatch.setattr(operator.base, "_diagnostic", lambda _, probe, __: {
        "sample_count": int(probe["action"].shape[1])})

    def planned(_, __, ___, ____, *, t0):
        action = np.array([0.0, .2, -.2], np.float32)
        delta = {"prior_mean": [0.0, .1, -.1], "mppi_weighted_elite_mean": [0.0, .2, -.2],
                 "delta": [0.0, .1, -.1], "delta_l2": math.sqrt(.02),
                 "applied_delta_l2": math.sqrt(.02)}
        return action, delta

    monkeypatch.setattr(operator.base, "_planned_action", planned)
    env = FakeEnv(failure=failure)
    output = root / operator.RUN_DIR
    return p, root, output, env, learner


def test_synthetic_loop_has_exact_seed_pretrain_one_update_per_decision_and_variant_checkpoints(tmp_path, monkeypatch):
    p, root, output, env, learner = _synthetic_run(tmp_path, monkeypatch)
    result = operator.run(root / "experiments/unused.json", "a" * 64, output,
                          root=root, env_factory=lambda _: env, clock=lambda: 0)
    assert env.closed and env.resets == 6
    assert (result["decisions"], result["updates"], result["pretrain_updates"]) == (30, 30, 10)
    assert result["reward_overshoot"] == operator.OVERSHOOT and result["evaluation"] is None
    assert not result["resume_supported"]
    assert learner.samples[0][0] == 1 and any(valid > 0 for _, valid in learner.samples)
    assert len(learner.samples) == 30
    ledger = [json.loads(row) for row in (output / "training.jsonl").read_text().splitlines()]
    steps = [json.loads(row) for row in (output / "steps.jsonl").read_text().splitlines()]
    assert len(steps) == 30 and [r["decisions"] for r in ledger if r["event"] == "checkpoint"] == [15, 20, 25, 30]
    assert [r["geometry_seed"] for r in ledger if r["event"] == "episode"] == list(operator.base.ROADS) + list(operator.base.ROADS[:2])
    assert [r["terminal"] for r in steps if r["truncated"]] == [False, True, False, False, False, True]
    rng = np.random.RandomState(7)
    assert [r["action_f32_hex"] for r in steps[:10]] == [
        rng.uniform(-1, 1, size=3).astype(np.float32).tobytes().hex() for _ in range(10)]
    for target, report in zip([15, 20, 25, 30], result["checkpoints"]):
        state = torch.load(output / report["path"], map_location="cpu", weights_only=False)
        assert report["sha256"] == operator.base.digest(output / report["path"])
        assert (state["format"], state["target"], state["decisions"], state["updates"]) == (
            operator.FORMAT, target, target, target)
        assert state["replay"]["active"] is None and state["replay"]["horizon"] == 3
        assert state["reward_overshoot"] == p["reward_overshoot"]
        assert state["protocol_sha256"] == "a" * 64 and state["probe"]["overshoot_mask"].shape[0] == 2
        assert state["step_ledger_sha256"] == report["step_ledger_sha256"]
        assert (state["training_ledger_sha256_before_checkpoint"]
                == report["training_ledger_sha256_before_checkpoint"])
        assert set(state) >= {"learner", "optim", "pi_optim", "replay", "probe", "rng", "source_sha256"}
        assert report["rolling_update_means"]["overshoot_reward_loss"] == pytest.approx(.2)
    with pytest.raises(FileExistsError):
        operator.run(root / "experiments/unused.json", "a" * 64, output, root=root,
                     env_factory=lambda _: pytest.fail("existing completed run reset"), clock=lambda: 0)


@pytest.mark.parametrize("failure,pattern", [
    ("wrong_road", "reset did not honor"), ("bad_finish", "terminal outcome"),
    ("early_timeout", "timeout occurred before"), ("interrupted", "synthetic interrupted episode"),
])
def test_wrong_cell_finish_timeout_or_interruption_preserves_partial_and_refuses_resume(tmp_path, monkeypatch,
                                                                                          failure, pattern):
    p, root, output, env, _ = _synthetic_run(tmp_path, monkeypatch, failure=failure)
    with pytest.raises((ValueError, RuntimeError), match=pattern):
        operator.run(root / "experiments/unused.json", "a" * 64, output,
                     root=root, env_factory=lambda _: env, clock=lambda: 0)
    assert env.closed and output.exists() and not (output / "result.json").exists()
    ledger = [json.loads(row) for row in (output / "training.jsonl").read_text().splitlines()]
    assert ledger[-1]["event"] == "partial" and ledger[-1]["resume_supported"] is False
    if failure == "interrupted":
        assert ledger[-1]["reset_intents"] == 4 and ledger[-1]["decisions"] == 15
        assert ledger[-1]["pending_step"]["decision"] == 16
        assert ledger[-1]["active_length"] == 0
        checkpoint = next(output.glob("checkpoint-at-least-*.pt"))
        assert operator.base.digest(checkpoint) == next(
            r["sha256"] for r in ledger if r["event"] == "checkpoint")
    with pytest.raises(FileExistsError):
        operator.run(root / "experiments/unused.json", "a" * 64, output,
                     root=root, env_factory=lambda _: pytest.fail("existing partial run reset"), clock=lambda: 0)


@pytest.mark.parametrize("drop,pattern", [
    ("cgroup", "cgroup headroom"), ("disk", "disk headroom"), ("cuda", "CUDA memory"),
])
def test_resource_drop_before_second_reset_records_partial_without_second_intent(tmp_path, monkeypatch,
                                                                                  drop, pattern):
    real_startup = operator._startup_resources
    p, root, output, _, _ = _synthetic_run(tmp_path, monkeypatch)
    low = {"now": False}

    class ResourceDropEnv(FakeEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            if truncated and self.resets == 1:
                low["now"] = True
            return obs, reward, terminated, truncated, info

    env = ResourceDropEnv()
    original = type(root).read_text

    def read(path, *args, **kwargs):
        if str(path).startswith("/sys/fs/cgroup/"):
            if path.name == "memory.max":
                return str(40 * 1024**3)
            if path.name == "memory.current":
                return str((18 if low["now"] and drop == "cgroup" else 10) * 1024**3)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(root), "read_text", read)
    monkeypatch.setattr(operator.os, "statvfs", lambda _: SimpleNamespace(
        f_bavail=(20 if low["now"] and drop == "disk" else 30) * 1024,
        f_frsize=1024**2))
    monkeypatch.setattr(operator.torch.cuda, "mem_get_info", lambda: (
        (2 if low["now"] and drop == "cuda" else 4) * 1024**3, 16 * 1024**3))
    monkeypatch.setattr(operator, "_startup_resources", real_startup)
    with pytest.raises(ValueError, match=pattern):
        operator.run(root / "experiments/unused.json", "a" * 64, output,
                     root=root, env_factory=lambda _: env, clock=lambda: 0)
    rows = [json.loads(row) for row in (output / "training.jsonl").read_text().splitlines()]
    assert env.resets == 1 and env.closed
    assert [row["event"] for row in rows] == ["start", "reset_intent", "reset", "episode", "partial"]
    assert rows[-1]["decisions"] == 5 and rows[-1]["updates"] == 0
    assert rows[-1]["reset_intents"] == 1 and rows[-1]["environment_steps_applied"] == 5
    assert not (output / "result.json").exists()
