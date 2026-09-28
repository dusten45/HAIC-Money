"""Synthetic-only checks for the independent TD-MPC2 long operator; no real resets."""

from copy import deepcopy
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haic.algorithms.tdmpc2.replay import EpisodeReplay
from scripts import train_tdmpc2_long as operator


def _protocol(root, dim=3):
    arm = "independent_3d" if dim == 3 else "exclusive_2d"
    names = ("scripts/train_tdmpc2_long.py", "haic/algorithms/tdmpc2/action_2d.py", "env_wrapper.py")
    return {
        "format": operator.FORMAT, "purpose": "consumed-TRAIN-development",
        "upstream_revision": operator.UPSTREAM,
        "source_sha256": {name: operator.digest(root / name) for name in names},
        "r6_protocol": {"path": "experiments/drqv2-geometry-mix-v1-r6.json", "sha256": operator.R6_SHA},
        "prior_consumption": {"path": "experiments/dreamerv3-reused-train-diagnostic-v1.json",
                              "sha256": operator.PRIOR_SHA},
        "exploration_protocol": {"path": "experiments/tdmpc2-exploration-v2.json",
                                 "sha256": operator.EXPLORATION_SHA},
        "exploration_result": {"path": "experiments/tdmpc2-exploration-v2-result.json",
                               "sha256": operator.digest(root / "experiments/tdmpc2-exploration-v2-result.json")},
        "selection": {"arm": arm, "action_dim": dim},
        "cells": deepcopy(operator.CELLS), "episode_schedule": [0, 1, 2, 3],
        "environment": operator.ENVIRONMENT.copy(), "run_dir": "runs/tdmpc2-long-test",
        "training": {"seed": 733, "decision_cap": 102000, "seed_steps": 10000,
                     "pretrain_updates": 10000, "update_cap": 102000,
                     "updates_per_post_seed_decision": 1, "max_steps": 2000,
                     "replay_capacity": 120000, "batch_size": 256, "device": "cpu",
                     "horizon": 3, "discount": .995, "rho": .5, "model_size": 5,
                     "num_bins": 101, "augmentation_pad": 3, "episodic": True,
                     "observation_shape": [4, 64, 64], "action_dim": dim},
        "checkpoint_targets": [20000, 40000, 70000, 100000],
        "seed_schedule": operator.SEED_SCHEDULE,
        "resources": {"min_cgroup_available_bytes": 16 * 1024**3,
                      "min_disk_available_bytes": 16 * 1024**3, "max_wall_seconds": 21600},
    }


def _fixture_root(tmp_path, monkeypatch):
    for directory in ("experiments", "runs", "scripts", "haic/algorithms/tdmpc2",
                      "runs/tdmpc2-exploration-20260928-v2"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    names = ("experiments/drqv2-geometry-mix-v1-r6.json",
             "experiments/dreamerv3-reused-train-diagnostic-v1.json",
             "experiments/tdmpc2-exploration-v2.json",
             "experiments/tdmpc2-exploration-v2-result.json",
             "runs/tdmpc2-exploration-20260928-v2/result.json",
             "runs/tdmpc2-exploration-20260928-v2/episodes.jsonl",
             "runs/tdmpc2-exploration-20260928-v2/steps.jsonl")
    for name in names:
        shutil.copyfile(operator.ROOT / name, tmp_path / name)
    for name in ("scripts/train_tdmpc2_long.py", "haic/algorithms/tdmpc2/action_2d.py", "env_wrapper.py"):
        path = tmp_path / name
        path.write_text("test-only executable or environment stub", encoding="utf-8")
    monkeypatch.setattr(operator, "SOURCE_PATHS", frozenset({
        "scripts/train_tdmpc2_long.py", "haic/algorithms/tdmpc2/action_2d.py", "env_wrapper.py"}))
    monkeypatch.setattr(operator, "_resources", lambda *_: None)
    return tmp_path


def test_preflight_binds_selected_arm_sources_and_existing_consumed_evidence(tmp_path, monkeypatch):
    root = _fixture_root(tmp_path, monkeypatch)
    protocol = _protocol(root)
    frozen = root / "experiments/tdmpc2-long-test.json"
    output = root / protocol["run_dir"]
    monkeypatch.setattr(operator, "make_training_env", lambda *_: pytest.fail("preflight constructed env"))

    def freeze():
        frozen.write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")
        return operator.digest(frozen)

    sha = freeze()
    assert operator.preflight(frozen, sha, output, root=root) == protocol
    assert not output.exists()
    with pytest.raises(ValueError, match="protocol SHA mismatch"):
        operator.preflight(frozen, "0" * 64, output, root=root)
    with pytest.raises(ValueError, match="run directory differs"):
        operator.preflight(frozen, sha, root / "runs/tdmpc2-long-other", root=root)
    protocol["seed_schedule"] = "v2's 10001 random decisions"
    with pytest.raises(ValueError, match="unrecognized long TRAIN protocol"):
        operator.preflight(frozen, freeze(), output, root=root)
    protocol["seed_schedule"] = operator.SEED_SCHEDULE

    protocol["selection"] = {"arm": "exclusive_2d", "action_dim": 2}
    protocol["training"]["action_dim"] = 2
    with pytest.raises(ValueError, match="selected exploration arm"):
        operator.preflight(frozen, freeze(), output, root=root)
    protocol["selection"] = {"arm": "independent_3d", "action_dim": 3}
    protocol["training"]["action_dim"] = 3
    protocol["environment"]["obstacles"] = False
    with pytest.raises(ValueError, match="four consumed"):
        operator.preflight(frozen, freeze(), output, root=root)
    protocol["environment"]["obstacles"] = True
    protocol["training"]["replay_capacity"] = 99999
    with pytest.raises(ValueError, match="budget settings"):
        operator.preflight(frozen, freeze(), output, root=root)
    protocol["training"]["replay_capacity"] = 120000
    (root / "env_wrapper.py").write_text("source drift", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch: env_wrapper.py"):
        operator.preflight(frozen, freeze(), output, root=root)
    (root / "env_wrapper.py").write_text("test-only executable or environment stub", encoding="utf-8")
    receipt = root / "runs/tdmpc2-exploration-20260928-v2/steps.jsonl"
    with receipt.open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="selection evidence or ledger hashes"):
        operator.preflight(frozen, freeze(), output, root=root)
    assert not output.exists()


def test_raw_memory_and_disk_floors_are_measured_not_declared(tmp_path, monkeypatch):
    resources = {"resources": {"min_cgroup_available_bytes": 16 * 1024**3,
                               "min_disk_available_bytes": 16 * 1024**3}}
    original = type(tmp_path).read_text
    readings = {"memory.max": str(32 * 1024**3), "memory.current": str(17 * 1024**3)}

    def read(path, *args, **kwargs):
        if path.name in readings and str(path).startswith("/sys/fs/cgroup/"):
            return readings[path.name]
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "read_text", read)
    monkeypatch.setattr(operator.os, "statvfs", lambda _: SimpleNamespace(
        f_bavail=32 * 1024, f_frsize=1024**2))
    with pytest.raises(ValueError, match="cgroup"):
        operator._resources(tmp_path / "runs/run", resources)
    readings["memory.current"] = str(10 * 1024**3)
    monkeypatch.setattr(operator.os, "statvfs", lambda _: SimpleNamespace(
        f_bavail=15 * 1024, f_frsize=1024**2))
    with pytest.raises(ValueError, match="disk"):
        operator._resources(tmp_path / "runs/run", resources)
    monkeypatch.setattr(operator.os, "statvfs", lambda _: SimpleNamespace(
        f_bavail=17 * 1024, f_frsize=1024**2))
    operator._resources(tmp_path / "runs/run", resources)


@pytest.mark.parametrize("dim", [2, 3])
def test_component_dimensions_match_model_replay_and_official_planner_defaults(dim):
    cfg = {"training": {"action_dim": dim, "device": "cpu", "max_steps": 2000,
                        "horizon": 3, "batch_size": 256, "replay_capacity": 120000, "seed": 733,
                        "discount": .995}}
    learner, replay, planner = operator._components(cfg)
    assert learner.model.cfg.action_dim == replay.action_dim == planner.config.action_dim == dim
    assert learner.cfg.rho == .5 and learner.cfg.batch_size == 256
    assert replay.horizon == planner.config.horizon == 3 and replay.augmentation_pad == 3
    assert (planner.config.num_samples, planner.config.num_pi_trajs,
            planner.config.iterations, planner.config.num_elites) == (512, 24, 6, 64)
    assert learner.discount == .995


class FakeModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cfg = SimpleNamespace(action_dim=dim)
        self._pi = torch.nn.Linear(1, 2 * dim)
        self.weight = torch.nn.Parameter(torch.tensor(.125))

    def pi(self, z, task=None):
        raw = self._pi(z)
        mean, log_std = raw.chunk(2, dim=-1)
        action = (mean + torch.randn_like(log_std) * log_std.exp() * .01).tanh()
        return action, {"mean": mean.tanh()}


class FakePlanner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.calls = 0

    def plan(self, obs, *, t0=False):
        self.calls += 1
        z = torch.ones((24, 1), device=obs.device) * self.model.weight
        action, _ = self.model.pi(z)
        self.prev_mean = action[0].detach().clone().repeat(3, 1)
        return (action[0] + .05 * torch.randn_like(action[0])).clamp(-1, 1)


@pytest.mark.parametrize("dim", [2, 3])
def test_prior_hook_has_identical_actions_and_rng_without_extra_forward(dim):
    operator._seed(11)
    model = FakeModel(dim)
    pixels = np.zeros((4, 64, 64), np.uint8)
    initial = operator._rng_state()

    def trace(capture):
        operator._restore_rng(initial)
        planner = FakePlanner(model)
        values = [operator._planned_action(model, planner, pixels, "cpu", t0=(index == 0),
                                           capture_prior=capture) for index in range(6)]
        return values, operator._rng_state()

    off, off_rng = trace(False)
    on, on_rng = trace(True)
    assert [action.tobytes() for action, _ in off] == [action.tobytes() for action, _ in on]
    torch.testing.assert_close(off_rng["torch"], on_rng["torch"], atol=0, rtol=0)
    assert off_rng["python"] == on_rng["python"]
    np.testing.assert_array_equal(off_rng["numpy"][1], on_rng["numpy"][1])
    for action, report in on:
        assert report is not None
        assert report["kind"] == "first_planner_pi_tanh_mean_vs_mppi_weighted_elite_mean"
        np.testing.assert_allclose(np.array(report["delta"]),
                                   np.array(report["mppi_weighted_elite_mean"]) - report["prior_mean"], atol=1e-7)
        assert report["delta_l2"] >= 0
        assert report["applied_delta_l2"] >= 0
        np.testing.assert_allclose(report["applied_exploration_noised_action"], action)
    if dim == 2:
        assert operator.action_2d(on[0][0])[1] * operator.action_2d(on[0][0])[2] == 0
    else:
        assert operator.action_3d(on[0][0]).shape == (3,)


class FakeLearner(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.model = FakeModel(dim)
        self.register_buffer("q_scale", torch.ones(1))
        self.optim = torch.optim.Adam([self.model.weight])
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters())
        self.discount = .995
        self.eligible_at_update = []

    def update(self, replay):
        self.eligible_at_update.append(replay.num_episodes)
        sample = replay.sample(2)
        assert sample["action"].shape[-1] == self.model.cfg.action_dim
        return {"total_loss": float(torch.rand(())), "pi_entropy": .5,
                "pi_scale": self.q_scale.item()}


class FakeEnv:
    def __init__(self, *, fail_after=None):
        self.unwrapped = self
        self.track_id = self.track_seed = None
        self.length = self.resets = 0
        self.closed = False
        self.fail_after = fail_after

    def reset(self, *, seed, options):
        self.track_seed, self.track_id = seed, options["track_id"]
        self.length = 0
        self.resets += 1
        return np.full((4, 84, 84), .1, np.float32), {}

    def step(self, action):
        assert action.shape == (3,) and np.isfinite(action).all()
        if self.fail_after and self.resets == self.fail_after and self.length == 1:
            raise RuntimeError("synthetic interrupted episode")
        self.length += 1
        finished = self.track_seed == operator.ROADS[1] and self.length == 2
        return np.full((4, 84, 84), .2, np.float32), .25, False, self.length == 2, {
            "progress": .25, "damage": .1, "finished": finished,
            "finish_time_s": 1.0 if finished else None,
        }

    def close(self):
        self.closed = True


@pytest.mark.parametrize("dim", [2, 3])
def test_four_immutable_boundary_checkpoints_and_exact_seed_pretrain(tmp_path, monkeypatch, dim):
    p = _protocol(_fixture_root(tmp_path, monkeypatch))
    p["training"].update({"seed": 7, "action_dim": dim, "decision_cap": 18, "update_cap": 18,
                          "seed_steps": 4, "pretrain_updates": 4, "max_steps": 2,
                          "batch_size": 2, "horizon": 1, "replay_capacity": 18})
    monkeypatch.setattr(operator, "TARGETS", [6, 8, 12, 16])
    monkeypatch.setattr(operator, "ROOT", tmp_path)
    learner = FakeLearner(dim)
    monkeypatch.setattr(operator, "_components", lambda _: (
        learner, EpisodeReplay(18, 1, action_dim=dim, seed=7), FakePlanner(learner.model)))
    monkeypatch.setattr(operator, "_diagnostic", lambda _, probe, __: {
        "sample_count": int(probe["action"].shape[1]), "q_scale": 1.0})
    env = FakeEnv()
    output = tmp_path / "runs/tdmpc2-long-test"
    result = operator.run(p, "a" * 64, output, env_factory=lambda _: env, clock=lambda: 0)
    assert env.closed and env.resets == 8
    assert (result["decisions"], result["updates"], result["pretrain_updates"]) == (16, 16, 4)
    assert result["seed_schedule"] == operator.SEED_SCHEDULE
    assert result["runtime"]["device"] == "cpu" and result["resource_usage"]["cpu_seconds"] >= 0
    assert len(result["checkpoints"]) == 4 and not result["resume_supported"]
    assert learner.eligible_at_update[:4] == [1] * 4
    rows = [json.loads(row) for row in (output / "training.jsonl").read_text().splitlines()]
    step_rows = [json.loads(row) for row in (output / "steps.jsonl").read_text().splitlines()]
    assert len(step_rows) == 16
    seeded = np.random.RandomState(7)
    expected = [seeded.uniform(-1, 1, size=dim).astype(np.float32).tobytes().hex()
                for _ in range(4)]
    assert [row["action_f32_hex"] for row in step_rows[:4]] == expected
    assert [row["decisions"] for row in rows if row["event"] == "checkpoint"] == [6, 8, 12, 16]
    assert rows[0]["runtime"]["torch"] == str(torch.__version__)
    assert [row["event"] for row in rows].count("episode") == 8
    assert [row["geometry_seed"] for row in rows if row["event"] == "episode"] == list(operator.ROADS) * 2
    for index, report in enumerate(result["checkpoints"]):
        path = output / report["path"]
        assert operator.digest(path) == report["sha256"]
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert state["replay"]["action_dim"] == dim and state["replay"]["active"] is None
        assert state["target"] == report["target"] and not state["resume_supported"]
        assert report["rolling_update_count"] == [6, 2, 4, 4][index]
        assert len(report["train_episodes_since_previous"]) == [3, 1, 2, 2][index]
        assert report["frozen_train_probe"]["sample_count"] == 2
        assert report["same_observation_prior_vs_mppi"]["delta_l2"] >= 0
        interval = report["same_observation_prior_vs_mppi_interval"]
        assert interval["planned_actions"] == [2, 2, 4, 4][index]
        assert interval["mean_model_delta_l2"] >= 0
        assert interval["mean_applied_exploration_noised_delta_l2"] >= 0
        assert report["resource_usage"]["peak_process_rss_mib"] > 0
        assert len(interval["mean_absolute_model_delta"]) == dim
        assert len(interval["mean_absolute_native_delta"]) == 3
    assert bytes.fromhex(step_rows[0]["action_f32_hex"]) == np.frombuffer(
        bytes.fromhex(step_rows[0]["action_f32_hex"]), dtype=np.float32).tobytes()
    if dim == 2:
        native = np.frombuffer(bytes.fromhex(step_rows[0]["native_action_f32_hex"]), dtype=np.float32)
        assert native[1] * native[2] == 0
    with pytest.raises(FileExistsError):
        operator.run(p, "a" * 64, output, env_factory=lambda _: pytest.fail("existing run reset"))


def test_partial_ledger_and_frozen_checkpoint_preserved_on_failure(tmp_path, monkeypatch):
    p = _protocol(_fixture_root(tmp_path, monkeypatch))
    p["training"].update({"seed_steps": 4, "pretrain_updates": 4, "max_steps": 2,
                          "batch_size": 2, "horizon": 1, "replay_capacity": 10,
                          "decision_cap": 10, "update_cap": 10})
    monkeypatch.setattr(operator, "ROOT", tmp_path)
    monkeypatch.setattr(operator, "TARGETS", [6, 8, 10, 12])
    p["training"]["decision_cap"] = p["training"]["update_cap"] = 14
    learner = FakeLearner(3)
    monkeypatch.setattr(operator, "_components", lambda _: (
        learner, EpisodeReplay(10, 1, action_dim=3, seed=7), FakePlanner(learner.model)))
    monkeypatch.setattr(operator, "_diagnostic", lambda *_: {"sample_count": 2})
    output = tmp_path / "runs/tdmpc2-long-test"
    with pytest.raises(RuntimeError, match="synthetic interrupted episode"):
        operator.run(p, "a" * 64, output, env_factory=lambda _: FakeEnv(fail_after=4), clock=lambda: 0)
    checkpoint = next(output.glob("checkpoint-*.pt"))
    original = operator.digest(checkpoint)
    rows = [json.loads(row) for row in (output / "training.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "partial" and rows[-1]["resume_supported"] is False
    assert rows[-1]["active_length"] == 1 and rows[-1]["decisions"] == 7
    assert operator.digest(checkpoint) == original
    assert not (output / "result.json").exists()
    with pytest.raises(FileExistsError):
        operator.run(p, "a" * 64, output, env_factory=lambda _: pytest.fail("partial cannot resume"))


def test_frozen_probe_reuses_sampler_and_restores_torch_rng():
    from haic.algorithms.tdmpc2.learner import TDMPC2Learner, TDMPC2LearnerConfig
    from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = WorldModel(TDMPC2ModelConfig(action_dim=2, obs_shape={"rgb": (4, 64, 64)},
                                             num_channels=2, latent_dim=32, mlp_dim=32,
                                             num_q=3, num_bins=11, episodic=True, dropout=0))
        learner = TDMPC2Learner(model, TDMPC2LearnerConfig(episode_length=2000, horizon=1, batch_size=2))
        replay = EpisodeReplay(4, 1, action_dim=2, seed=41)
        for index in range(3):
            image = np.full((4, 64, 64), 32 + index, np.uint8)
            replay.start_episode(image)
            replay.add_step(image, np.zeros(2, np.float32), float(index), truncated=True)
        sampler = deepcopy(replay.rng.bit_generator.state)
        probe = operator._freeze_probe(replay, 2)
        assert replay.rng.bit_generator.state == sampler
        saved = operator._rng_state()
        one = operator._diagnostic(learner, probe, 101)
        two = operator._diagnostic(learner, probe, 101)
        assert one == two and one["sample_count"] == 2
        assert one["terminal_positive_transitions"] == 0
        assert one["h3_unique_raw_returns_at_1e_minus_6"] >= 1
        assert one["h3_discounted_return_max"] >= one["h3_discounted_return_min"]
        assert all(np.isfinite(value) for value in one.values() if isinstance(value, float))
        torch.testing.assert_close(operator._rng_state()["torch"], saved["torch"], atol=0, rtol=0)
        assert replay.rng.bit_generator.state == sampler
    finally:
        torch.set_num_threads(threads)
