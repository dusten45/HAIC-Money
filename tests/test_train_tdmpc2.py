"""Synthetic TD-MPC2 operator checks: never construct or reset a real HAIC env."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch

from scripts import train_tdmpc2 as operator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_template():
    return {
        "format": operator.FORMAT,
        "purpose": "reused-TRAIN-engineering-pilot",
        "upstream_revision": operator.UPSTREAM,
        "source_sha256": {},
        "r6_protocol": {"path": "experiments/drqv2-geometry-mix-v1-r6.json", "sha256": operator.R6_SHA},
        "prior_consumption": {"path": "experiments/dreamerv3-reused-train-diagnostic-v1.json", "sha256": operator.PRIOR_SHA},
        "cells": [{"track_id": 1, "geometry_seed": road} for road in operator.ROADS],
        "episode_schedule": [0, 1, 2, 3],
        "environment": {"track_id": 1, "obstacles": True, "frame_skip": 4, "reward": "raw"},
        "run_dir": "runs/tdmpc2-reused-train-synthetic",
        "training": {"seed": 13, "decision_cap": 14000, "seed_steps": 10000,
                     "pretrain_updates": 10000, "update_cap": 14000, "min_post_seed_updates": 1000,
                     "updates_per_post_seed_decision": 1, "max_steps": 2000,
                     "replay_capacity": 14000, "batch_size": 256, "device": "cpu",
                     "horizon": 3, "discount": .995, "rho": .5, "model_size": 5, "num_bins": 101,
                     "augmentation_pad": 3, "episodic": True, "observation_shape": [4, 64, 64]},
        "evaluation": {"seed": 83, "repeats": 2, "max_episodes": 16, "max_steps": 500},
        "resources": {"min_cgroup_available_bytes": 16 * 1024**3,
                      "min_disk_available_bytes": 5 * 1024**3, "max_wall_seconds": 7200},
    }


def test_preflight_rejects_source_drift_and_unapproved_cells_before_env_creation(tmp_path, monkeypatch):
    root = tmp_path
    (root / "experiments").mkdir()
    (root / "scripts").mkdir()
    (root / "runs").mkdir()
    for name in ("drqv2-geometry-mix-v1-r6.json", "dreamerv3-reused-train-diagnostic-v1.json"):
        shutil.copyfile(operator.ROOT / "experiments" / name, root / "experiments" / name)
    src = root / "scripts" / "train_tdmpc2.py"
    src.write_text("frozen source", encoding="utf-8")
    environment = root / "env_wrapper.py"
    environment.write_text("frozen environment", encoding="utf-8")
    monkeypatch.setattr(operator, "SOURCE_PATHS", frozenset({"scripts/train_tdmpc2.py", "env_wrapper.py"}))
    untouched = []
    monkeypatch.setattr(operator, "make_training_env", lambda _: untouched.append("constructed"))
    protocol = protocol_template()
    protocol["source_sha256"] = {"scripts/train_tdmpc2.py": sha(src), "env_wrapper.py": sha(environment)}
    frozen = root / "experiments" / "tdmpc2.json"

    def freeze():
        frozen.write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")
        return sha(frozen)

    out = root / "runs" / "tdmpc2-reused-train-synthetic"
    stamp = freeze()
    assert operator.preflight(frozen, stamp, out, root=root) == protocol
    assert not out.exists()
    assert untouched == []
    with pytest.raises(ValueError, match="differs from frozen"):
        operator.preflight(frozen, stamp, root / "runs" / "tdmpc2-reused-train-other", root=root)
    protocol["training"]["device"] = "cuda"
    stamp = freeze()
    monkeypatch.setattr(operator.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="pinned CUDA device unavailable"):
        operator.preflight(frozen, stamp, out, root=root)
    assert operator.preflight(frozen, stamp, out, root=root, require_training_device=False) == protocol
    protocol["training"]["device"] = "cpu"
    stamp = freeze()
    src.write_text("modified executable", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        operator.preflight(frozen, stamp, out, root=root)
    src.write_text("frozen source", encoding="utf-8")
    environment.write_text("changed physics wrapper", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch: env_wrapper.py"):
        operator.preflight(frozen, stamp, out, root=root)
    environment.write_text("frozen environment", encoding="utf-8")
    protocol["cells"][0]["geometry_seed"] = 123
    with pytest.raises(ValueError, match="consumed TRAIN"):
        operator.preflight(frozen, freeze(), out, root=root)
    protocol["cells"][0]["geometry_seed"] = operator.ROADS[0]
    protocol["environment"]["obstacles"] = False
    with pytest.raises(ValueError, match="TRAIN environment conditions"):
        operator.preflight(frozen, freeze(), out, root=root)
    protocol["environment"]["obstacles"] = True
    protocol["training"]["seed_steps"] = 5000
    with pytest.raises(ValueError, match="upstream seed"):
        operator.preflight(frozen, freeze(), out, root=root)
    assert untouched == []


class FakeEnv:
    def __init__(self, max_steps, *, clock=None):
        self.max_steps = max_steps
        self.episode = 0
        self.track = None
        self.clock = clock
        self.closed = False
        self.resets = []

    def reset(self, *, seed, options):
        assert options == {"track_id": 1}
        assert seed in operator.ROADS
        self.track = seed
        self.episode = 0
        self.resets.append(seed)
        return self.image(), {}

    def image(self):
        return np.full((4, 84, 84), .125 + self.episode * .125, np.float32)

    def step(self, action):
        assert action.shape == (3,) and -1 <= action[0] <= 1
        assert np.all((action[1:] >= 0) & (action[1:] <= 1))
        self.episode += 1
        if self.clock is not None and len(self.resets) == 1 and self.episode == self.max_steps:
            self.clock.now = 1001
        finished = self.track == operator.ROADS[1] and self.episode == self.max_steps
        return self.image(), float(action[0]) + .25, False, self.episode == self.max_steps, {
            "progress": .75 if finished else .5, "finished": finished,
            "finish_time_s": 1.25 if finished else None,
        }

    def close(self):
        self.closed = True


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.2]))
        self.policy = torch.nn.Parameter(torch.tensor([.1]))

    def encode(self, obs, task=None):
        return torch.ones((obs.shape[0], 1), device=obs.device) * self.weight

    def pi(self, z, task=None):
        action = (torch.rand((z.shape[0], 3), device=z.device) - .5) + self.policy
        return action.clamp(-1, 1), {"mean": self.policy.expand(z.shape[0], 3).tanh()}


class FakeLearner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeModel()
        self.optim = torch.optim.Adam([self.model.weight], lr=.01)
        self.pi_optim = torch.optim.Adam([self.model.policy], lr=.01)
        self.discount = .995
        self.samples = []
        self.eligible_at_update = []

    def update(self, replay):
        self.eligible_at_update.append(replay.num_episodes)
        sampled = replay.sample(2)
        self.samples.append(sampled["start_step"].tolist())
        self.optim.zero_grad()
        self.pi_optim.zero_grad()
        loss = (self.model.weight - torch.rand(()) - random.random() - np.random.rand()).square()
        loss = loss + self.model.policy.square()
        loss.backward()
        self.optim.step()
        self.pi_optim.step()
        return {"loss": float(loss.detach())}


class FakePlanner:
    def __init__(self, model, config):
        self.model = model
        self.reset()

    def reset(self):
        self.prev_mean = 0

    def plan(self, obs, *, t0=False, eval_mode=False):
        if t0:
            self.reset()
        self.prev_mean += 1
        return (torch.rand(3) - .5 + self.model.policy.detach()).clamp(-1, 1)


def small_training_protocol():
    p = protocol_template()
    p["training"].update({"decision_cap": 8, "seed_steps": 2, "pretrain_updates": 2,
                          "update_cap": 6, "min_post_seed_updates": 1,
                          "max_steps": 2, "replay_capacity": 8, "batch_size": 2, "horizon": 1})
    p["resources"]["max_wall_seconds"] = 1000
    p["evaluation"]["max_steps"] = 2
    return p


def synthetic_components(monkeypatch):
    from haic.algorithms.tdmpc2.replay import EpisodeReplay

    def build(protocol):
        learner = FakeLearner()
        train = protocol["training"]
        replay = EpisodeReplay(train["replay_capacity"], train["horizon"], seed=train["seed"])
        return learner, replay, FakePlanner(learner.model, None)

    monkeypatch.setattr(operator, "_components", build)
    monkeypatch.setattr(operator, "_resources", lambda *_: None)
    monkeypatch.setattr(operator, "evaluate", lambda *args, **kwargs: {"synthetic": True})
    monkeypatch.setattr(operator, "_diagnostic", lambda learner, probe, seed: {
        "sample_count": len(probe["episode_id"]), "q_scale": 1.0,
    })
    monkeypatch.setattr(operator, "SOURCE_PATHS", frozenset())


def checkpoint_state(path):
    return torch.load(path / "boundary.pt", weights_only=False)


def test_boundary_resume_reproduces_model_optimizer_replay_and_random_stream(tmp_path, monkeypatch):
    synthetic_components(monkeypatch)
    p = small_training_protocol()
    p["source_sha256"] = {}
    monkeypatch.setattr(operator, "ROOT", tmp_path)
    full = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    first_envs, second_envs = [], []

    def full_env(max_steps):
        env = FakeEnv(max_steps)
        first_envs.append(env)
        return env

    first = operator.run(p, "c" * 64, full, env_factory=full_env, clock=lambda: 0)
    assert first["decisions"] == 8 and first["updates"] == 6
    assert first["pretrain_updates"] == 2 and first["episodes"] == 4
    assert first["min_post_seed_updates_met"] and first["evaluation"] is None
    assert first["evaluation_status"] == "separate_cpu_export_and_evaluation_required"
    assert first_envs[0].closed

    class Clock:
        now = 0

        def __call__(self):
            return self.now

    clock = Clock()

    def cut_env(max_steps):
        env = FakeEnv(max_steps, clock=clock)
        second_envs.append(env)
        return env

    stopped = operator.run(p, "c" * 64, resumed, env_factory=cut_env, clock=clock)
    assert stopped["status"] == "boundary_wall_budget" and stopped["decisions"] == 2
    assert stopped["episodes"] == 1 and checkpoint_state(resumed)["replay"]["active"] is None
    with pytest.raises(ValueError, match="source/protocol/cursor mismatch"):
        operator.run(p, "e" * 64, resumed, resume=True,
                     env_factory=cut_env, clock=lambda: 0)
    assert len(second_envs) == 1
    continued = operator.run(p, "c" * 64, resumed, resume=True,
                             env_factory=cut_env, clock=lambda: 0)
    assert continued["decisions"] == first["decisions"]
    assert continued["updates"] == first["updates"]
    assert continued["episodes"] == first["episodes"]
    assert second_envs[-1].resets == list(operator.ROADS[1:])

    a, b = checkpoint_state(full), checkpoint_state(resumed)
    for key in ("decisions", "updates", "pretrain_updates", "episodes"):
        assert a[key] == b[key]
    for key, value in a["learner"].items():
        torch.testing.assert_close(value, b["learner"][key], rtol=0, atol=0)
    assert a["optim"]["state"][0]["step"] == b["optim"]["state"][0]["step"]
    assert a["pi_optim"]["state"][0]["step"] == b["pi_optim"]["state"][0]["step"]
    assert a["rng"]["python"] == b["rng"]["python"]
    assert a["rng"]["numpy"][2:] == b["rng"]["numpy"][2:]
    np.testing.assert_array_equal(a["rng"]["numpy"][1], b["rng"]["numpy"][1])
    torch.testing.assert_close(a["rng"]["torch"], b["rng"]["torch"], rtol=0, atol=0)
    assert a["replay"]["rng_state"] == b["replay"]["rng_state"]
    for key in ("observations", "actions", "rewards", "terminated", "truncated", "terminal"):
        for left, right in zip(a["replay"]["episodes"], b["replay"]["episodes"]):
            np.testing.assert_array_equal(left[key], right[key])


def test_timeout_bootstraps_but_finish_is_terminal_and_partial_cannot_resume(tmp_path, monkeypatch):
    synthetic_components(monkeypatch)
    p = small_training_protocol()
    p["source_sha256"] = {}
    monkeypatch.setattr(operator, "ROOT", tmp_path)
    out = tmp_path / "part"
    class Clock:
        now = 0

        def __call__(self):
            return self.now

    clock = Clock()

    class Partial(FakeEnv):
        def step(self, action):
            value = super().step(action)
            if len(self.resets) == 2 and self.episode == 1:
                clock.now = 1001
            return value

    result = operator.run(p, "d" * 64, out, env_factory=lambda max_steps: Partial(max_steps), clock=clock)
    assert result["status"] == "partial_wall_budget"
    lines = [json.loads(line) for line in (out / "training.jsonl").read_text().splitlines()]
    assert lines[-1]["event"] == "partial"
    with pytest.raises(ValueError, match="partial ledger"):
        operator.run(p, "d" * 64, out, resume=True,
                     env_factory=lambda max_steps: FakeEnv(max_steps), clock=lambda: 0)

    p["training"]["decision_cap"] = 4
    p["training"]["update_cap"] = 3
    p["training"]["min_post_seed_updates"] = 0
    rows = tmp_path / "four"
    operator.run(p, "d" * 64, rows, env_factory=lambda n: FakeEnv(n), clock=lambda: 0)
    state = checkpoint_state(rows)["replay"]
    first, second = state["episodes"]
    assert first["truncated"][-1] and not first["terminal"][-1]
    assert second["truncated"][-1] and second["terminal"][-1]


def test_seed_boundary_without_completed_window_fails_before_planner(tmp_path, monkeypatch):
    synthetic_components(monkeypatch)
    p = small_training_protocol()
    p["source_sha256"] = {}
    p["training"]["horizon"] = 3
    monkeypatch.setattr(operator, "ROOT", tmp_path)
    calls = []

    def forbidden(self, *args, **kwargs):
        calls.append("planner")
        raise AssertionError("planner cannot act without source-pinned seed pretraining")

    monkeypatch.setattr(FakePlanner, "plan", forbidden)
    out = tmp_path / "seed-window-absent"
    with pytest.raises(RuntimeError, match=r"no completed H\+1 replay window"):
        operator.run(p, "f" * 64, out, env_factory=lambda max_steps: FakeEnv(max_steps), clock=lambda: 0)
    assert not calls
    rows = [json.loads(line) for line in (out / "training.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "partial" and rows[-1]["decisions"] == 3
    with pytest.raises(ValueError, match="partial ledger"):
        operator.run(p, "f" * 64, out, resume=True,
                     env_factory=lambda max_steps: FakeEnv(max_steps), clock=lambda: 0)


def test_newly_ended_episode_becomes_sampleable_only_after_decision_updates(tmp_path, monkeypatch):
    synthetic_components(monkeypatch)
    monkeypatch.setattr(operator, "ROOT", tmp_path)
    saved_build = operator._components
    learner_holder = {}

    def capture(protocol):
        components = saved_build(protocol)
        learner_holder["learner"] = components[0]
        return components

    monkeypatch.setattr(operator, "_components", capture)
    p = small_training_protocol()
    p["source_sha256"] = {}
    p["training"].update({"decision_cap": 4, "update_cap": 3, "min_post_seed_updates": 0})
    out = tmp_path / "deferred-final-step"
    operator.run(p, "f" * 64, out, env_factory=lambda max_steps: FakeEnv(max_steps), clock=lambda: 0)
    assert learner_holder["learner"].eligible_at_update == [1, 1, 1]
    assert checkpoint_state(out)["replay"]["next_episode_id"] == 2


def test_paired_evaluation_seeds_rng_and_restores_all_streams(tmp_path, monkeypatch):
    monkeypatch.setattr(operator, "TDMPC2Planner", FakePlanner)
    monkeypatch.setattr(operator, "_resources", lambda *_: None)
    p = small_training_protocol()
    p["evaluation"].update({"repeats": 2, "max_episodes": 16})
    _ = random.random(), np.random.rand(), torch.rand(())
    before = operator._rng_state()
    created = []

    def factory(max_steps):
        env = FakeEnv(max_steps)
        created.append(env)
        return env

    report = operator.evaluate(p, FakeModel(), tmp_path / "eval.jsonl", env_factory=factory)
    assert report["episodes"] == 16
    assert report["per_mode"]["mppi"]["episodes"] == report["per_mode"]["prior"]["episodes"] == 8
    assert report["per_mode"]["prior"]["finishes"] == 2
    assert report["per_mode"]["mppi"]["censored"] == 6
    assert report["capped_finish_comparison_valid"] is False
    rows = [json.loads(line) for line in (tmp_path / "eval.jsonl").read_text().splitlines()
            if json.loads(line)["event"] == "episode"]
    assert len(rows) == 16 and created[0].closed
    for pair in range(0, len(rows), 2):
        assert rows[pair]["mode"] == "prior" and rows[pair + 1]["mode"] == "mppi"
        assert rows[pair]["episode_seed"] == rows[pair + 1]["episode_seed"]
        assert rows[pair]["geometry_seed"] == rows[pair + 1]["geometry_seed"]
        assert rows[pair]["decisions"] == rows[pair + 1]["decisions"] == 2
        assert rows[pair]["action_latency_max_s"] >= 0
        assert rows[pair]["process_rss_bytes"] > 0
        assert len(rows[pair]["native_action_trace_sha256"]) == 64
        assert rows[pair]["censored"] == (rows[pair]["geometry_seed"] != operator.ROADS[1])
        if rows[pair]["mode"] == "prior":
            assert rows[pair]["reward"] == pytest.approx(2 * (.25 + np.tanh(.1)), abs=1e-6)
    after = operator._rng_state()
    assert before["python"] == after["python"]
    np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
    torch.testing.assert_close(before["torch"], after["torch"])
    with pytest.raises(ValueError, match="already exists"):
        operator.evaluate(p, FakeModel(), tmp_path / "eval.jsonl", env_factory=factory)
    operator.evaluate(p, FakeModel(), tmp_path / "fresh-reload.jsonl", env_factory=factory)
    reloaded = [json.loads(line) for line in (tmp_path / "fresh-reload.jsonl").read_text().splitlines()
                if json.loads(line)["event"] == "episode"]
    assert [(row["action_trace_sha256"], row["native_action_trace_sha256"]) for row in rows] == [
        (row["action_trace_sha256"], row["native_action_trace_sha256"]) for row in reloaded]


def test_frozen_train_probe_reports_fit_and_restores_rng():
    from haic.algorithms.tdmpc2.learner import TDMPC2Learner, TDMPC2LearnerConfig
    from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel
    from haic.algorithms.tdmpc2.replay import EpisodeReplay

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = WorldModel(TDMPC2ModelConfig(
            obs_shape={"rgb": (4, 64, 64)}, num_channels=2, latent_dim=32,
            mlp_dim=32, num_q=3, num_bins=11, episodic=True, dropout=0))
        learner = TDMPC2Learner(model, TDMPC2LearnerConfig(episode_length=2000, horizon=1, batch_size=3))
        replay = EpisodeReplay(capacity=4, horizon=1, seed=101)
        for road in range(4):
            image = np.full((4, 64, 64), 32 + road * 16, np.uint8)
            replay.start_episode(image)
            replay.add_step(image + 4, np.zeros(3, np.float32), float(road), truncated=True,
                            terminal=(road == 1))
        saved = operator._rng_state()
        previous_sampler = deepcopy(replay.rng.bit_generator.state)
        probe = operator._freeze_probe(replay, 3)
        assert replay.rng.bit_generator.state == previous_sampler
        report = operator._diagnostic(learner, probe, 77)
        assert report["sample_count"] == 3
        assert report["episode_ids"] == probe["episode_id"].tolist()
        for key in ("latent_mse", "latent_frozen_anchor_mse", "reward_mae",
                    "constant_reward_mae", "q_td_residual_abs_mean", "q_scale", "termination_bce"):
            assert np.isfinite(report[key])
        assert report == operator._diagnostic(learner, probe, 77)
        assert saved["python"] == operator._rng_state()["python"]
        torch.testing.assert_close(saved["torch"], operator._rng_state()["torch"])
    finally:
        torch.set_num_threads(old_threads)


def test_model_only_cpu_export_and_separate_reload_do_not_require_cuda_state(tmp_path, monkeypatch):
    actual_evaluate = operator.evaluate
    synthetic_components(monkeypatch)
    monkeypatch.setattr(operator, "evaluate", actual_evaluate)
    monkeypatch.setattr(operator, "ROOT", tmp_path)
    monkeypatch.setattr(operator, "WorldModel", lambda *_: FakeModel())
    monkeypatch.setattr(operator, "TDMPC2ModelConfig", lambda **_: None)
    monkeypatch.setattr(operator, "TDMPC2Planner", FakePlanner)
    monkeypatch.setattr(operator.torch.version, "cuda", None)
    p = small_training_protocol()
    p["source_sha256"] = {}
    p["training"]["device"] = "cuda"  # Protocol is training-only; CPU evaluator ignores it.
    out = tmp_path / "cpu-export"
    operator.run(p, "a" * 64, out, env_factory=lambda max_steps: FakeEnv(max_steps), clock=lambda: 0)
    assert not (out / "train-evaluation.jsonl").exists()
    exported = operator.export_cpu(p, "a" * 64, out)
    assert exported["environment_resets"] == 0
    assert exported["sha256"] == sha(out / "cpu-model.pt")
    payload = torch.load(out / "cpu-model.pt", map_location="cpu", weights_only=False)
    assert payload["format"] == "haic-tdmpc2-cpu-model-v1"
    assert payload["checkpoint_sha256"] == exported["checkpoint_sha256"]
    assert set(payload["model_state"]) == {"weight", "policy"}
    assert all(value.device.type == "cpu" for value in payload["model_state"].values())
    with pytest.raises(ValueError, match="already exists"):
        operator.export_cpu(p, "a" * 64, out)
    monkeypatch.setattr(operator, "_boundary_state", lambda *args: (_ for _ in ()).throw(
        AssertionError("CPU-only evaluation must not load training replay/optimizer/CUDA RNG")))
    monkeypatch.setattr(operator.torch.version, "cuda", "fake-cuda-build")
    with pytest.raises(ValueError, match="CPU-only PyTorch 2.1.0 interpreter"):
        operator.evaluate_only(p, "a" * 64, out,
                               env_factory=lambda max_steps: FakeEnv(max_steps))
    monkeypatch.setattr(operator.torch.version, "cuda", None)
    monkeypatch.setattr(operator.torch, "__version__", "2.1.0+cpu")
    report = operator.evaluate_only(p, "a" * 64, out,
                                    env_factory=lambda max_steps: FakeEnv(max_steps))
    assert report["episodes"] == 16 and report["cpu_export_sha256"] == exported["sha256"]
    with pytest.raises(ValueError, match="already exists"):
        operator.evaluate_only(p, "a" * 64, out,
                               env_factory=lambda max_steps: FakeEnv(max_steps))
