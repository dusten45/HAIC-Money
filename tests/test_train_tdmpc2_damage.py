"""Synthetic-only damage TRAIN runner tests. No real HAIC environment construction."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from haic.algorithms.tdmpc2.learner import TDMPC2Learner, TDMPC2LearnerConfig
from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel, two_hot_inv
from haic.algorithms.tdmpc2.replay import EpisodeReplay
from haic.algorithms.tdmpc2.reward import DamageIncrementReward
from scripts import train_tdmpc2_damage as operator


def _protocol() -> dict:
    baseline = operator.base._json(operator.ROOT / operator.BASELINE["baseline_training_protocol"][0])
    return {
        "format": operator.FORMAT, "purpose": "consumed-TRAIN-development",
        "upstream_revision": operator.base.UPSTREAM,
        "source_sha256": dict(baseline["source_sha256"]),
        **{key: {"path": name, "sha256": sha} for key, (name, sha) in operator.BASELINE.items()},
        "selection": {"arm": "independent_3d", "action_dim": 3},
        "cells": deepcopy(operator.base.CELLS), "episode_schedule": [0, 1, 2, 3],
        "environment": dict(operator.base.ENVIRONMENT),
        "run_dir": "runs/tdmpc2-damage-synthetic",
        "training": deepcopy(baseline["training"]),
        "checkpoint_targets": list(operator.base.TARGETS),
        "resources": dict(baseline["resources"]),
        "seed_schedule": operator.base.SEED_SCHEDULE,
        "reward_target": dict(operator.REWARD_TARGET),
    }


def _fixture(tmp_path, monkeypatch):
    p = _protocol()
    root = tmp_path
    for name in operator.SOURCE_PATHS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("ascii"))
        p["source_sha256"][name] = operator.base.digest(path)
    sources = {key: p["source_sha256"][key] for key in operator.base.SOURCE_PATHS}
    fake_baseline = {"source_sha256": sources, "training": deepcopy(p["training"]),
                     "resources": deepcopy(p["resources"]),
                     "selection": deepcopy(p["selection"]), "cells": deepcopy(p["cells"]),
                     "checkpoint_targets": list(p["checkpoint_targets"])}
    (root / "runs").mkdir()
    (root / "experiments").mkdir()
    path = root / "experiments/tdmpc2-damage-synthetic.json"
    monkeypatch.setattr(operator, "_evidence", lambda *_: fake_baseline)
    monkeypatch.setattr(operator.base, "_resources", lambda *_: None)
    return p, path, root / p["run_dir"]


def _freeze(path, p):
    path.write_text(json.dumps(p, sort_keys=True), encoding="utf-8")
    return operator.base.digest(path)


def test_no_reset_preflight_requires_exact_reward_source_and_resource_contract(tmp_path, monkeypatch):
    p, path, output = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(operator, "make_training_env", lambda *_: pytest.fail("constructed real env"))
    sha = _freeze(path, p)
    assert operator.preflight(path, sha, output, root=tmp_path) == p
    assert not output.exists()
    with pytest.raises(ValueError, match="protocol SHA mismatch"):
        operator.preflight(path, "0" * 64, output, root=tmp_path)
    p["reward_target"]["coefficient"] = 4.0
    with pytest.raises(ValueError, match="damage-only"):
        operator.preflight(path, _freeze(path, p), output, root=tmp_path)
    p["reward_target"]["coefficient"] = 5.0
    p["training"]["horizon"] = 5
    with pytest.raises(ValueError, match="training, action"):
        operator.preflight(path, _freeze(path, p), output, root=tmp_path)
    p["training"]["horizon"] = 3
    p["training"]["pretrain_updates"] = 10000.0
    with pytest.raises(ValueError, match="training, action"):
        operator.preflight(path, _freeze(path, p), output, root=tmp_path)
    p["training"]["pretrain_updates"] = 10000
    p["training"]["action_dim"] = 2
    with pytest.raises(ValueError, match="training, action"):
        operator.preflight(path, _freeze(path, p), output, root=tmp_path)
    p["training"]["action_dim"] = 3
    p["resources"]["min_disk_available_bytes"] -= 1
    with pytest.raises(ValueError, match="training, action"):
        operator.preflight(path, _freeze(path, p), output, root=tmp_path)
    p["resources"]["min_disk_available_bytes"] += 1
    (tmp_path / "env_wrapper.py").write_text("source drift", encoding="ascii")
    with pytest.raises(ValueError, match="source hash mismatch: env_wrapper.py"):
        operator.preflight(path, _freeze(path, p), output, root=tmp_path)
    (tmp_path / "env_wrapper.py").write_bytes(b"env_wrapper.py")
    monkeypatch.setattr(operator.base, "_resources", lambda *_: (_ for _ in ()).throw(
        ValueError("insufficient raw cgroup headroom")))
    with pytest.raises(ValueError, match="cgroup"):
        operator.preflight(path, _freeze(path, p), output, root=tmp_path)
    assert not output.exists()


def test_runtime_protocol_drift_fails_before_any_reset(tmp_path, monkeypatch):
    p, path, output = _fixture(tmp_path, monkeypatch)
    old_sha = _freeze(path, p)
    p["training"]["seed"] = 734
    _freeze(path, p)
    with pytest.raises(ValueError, match="damage protocol changed during run"):
        operator._runtime_sources(tmp_path, p["source_sha256"], path, old_sha)
    assert not output.exists()


class FakeLearner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(1, 1)
        self.optim = torch.optim.Adam(self.model.parameters())
        self.pi_optim = torch.optim.Adam(self.model.parameters())
        self.register_buffer("q_scale", torch.ones(1))
        self.discount = .995
        self.sampled_rewards = []
        self.eligible_at_update = []

    def update(self, replay):
        self.eligible_at_update.append(replay.num_episodes)
        sampled = replay.sample(2)
        self.sampled_rewards.extend(sampled["reward"].flatten().tolist())
        return {"total_loss": .1, "reward_loss": .2, "pi_entropy": .3}


class FakePlanner:
    def reset(self):
        pass


class FakeEnv:
    def __init__(self, *, invalid=False):
        self.unwrapped = self
        self.invalid = invalid
        self.resets = self.length = 0
        self.closed = False

    def reset(self, *, seed, options):
        self.track_seed, self.track_id = seed, options["track_id"]
        self.resets += 1
        self.length = 0
        return np.zeros((4, 84, 84), np.float32), {}

    def step(self, action):
        assert action.shape == (3,) and np.isfinite(action).all()
        self.length += 1
        finished = self.resets % 4 == 2 and self.length == 2
        terminated = self.resets % 4 == 3 and self.length == 2
        damage = (1.0 if self.resets % 4 == 0 else 0.0 if self.length == 1 else .2)
        if self.invalid == "decreasing" and self.resets == 1:
            damage = .4 if self.length == 1 else .2
        elif self.invalid is True and self.resets == 1 and self.length == 2:
            damage = -0.1
        return np.zeros((4, 84, 84), np.float32), (.25 if self.length == 1 else .5), (
            terminated), self.length == 2 and not terminated, {
                "damage": damage, "progress": .1 * self.length, "finished": finished,
                "finish_time_s": 1.0 if finished else None,
            }

    def close(self):
        self.closed = True


def _fake_run(tmp_path, monkeypatch, *, invalid=False):
    p = _protocol()
    p["training"].update({"device": "cpu", "seed_steps": 4, "pretrain_updates": 4,
                          "max_steps": 2, "horizon": 1, "batch_size": 2,
                          "decision_cap": 18, "update_cap": 18, "replay_capacity": 18})
    monkeypatch.setattr(operator, "preflight", lambda *_args, **_kwargs: p)
    monkeypatch.setattr(operator, "_runtime_sources", lambda *_: None)
    monkeypatch.setattr(operator.base, "_resources", lambda *_: None)
    monkeypatch.setattr(operator.base, "TARGETS", [6, 8, 12, 16])
    learner = FakeLearner()
    replay = EpisodeReplay(18, 1, action_dim=3, seed=733)
    monkeypatch.setattr(operator.base, "_components", lambda _: (learner, replay, FakePlanner()))
    gap = {"delta": [0.1, 0.0, 0.0], "prior_mean": [0.0, 0.0, 0.0],
           "mppi_weighted_elite_mean": [0.1, 0.0, 0.0],
           "delta_l2": .1, "applied_delta_l2": .2}
    monkeypatch.setattr(operator.base, "_planned_action", lambda *_args, **_kwargs: (
        np.zeros(3, dtype=np.float32), gap))
    monkeypatch.setattr(operator, "_diagnostic", lambda *_: {
        "training_reward_mae": .1, "sample_count": 2})
    env = FakeEnv(invalid=invalid)
    (tmp_path / "runs").mkdir()
    output = tmp_path / p["run_dir"]
    return p, env, learner, replay, output


def test_raw_journal_and_episode_return_shaped_replay_and_semantic_bootstrap(tmp_path, monkeypatch):
    p, env, learner, replay, output = _fake_run(tmp_path, monkeypatch)
    result = operator.run(Path("experiments/unused.json"), "a" * 64, output, root=tmp_path,
                          env_factory=lambda _: env, clock=lambda: 0)
    steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    rows = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    episodes = [row for row in rows if row["event"] == "episode"]
    assert env.closed and env.resets == 8 and len(steps) == 16
    assert (result["decisions"], result["updates"], result["episodes"]) == (16, 16, 8)
    assert len(result["checkpoints"]) == 4 and result["evaluation"] is None
    assert (steps[0]["reward"], steps[0]["training_reward"], steps[0]["damage_delta"]) == (.25, .25, 0)
    assert steps[1]["reward"] == .5 and steps[1]["training_reward"] == pytest.approx(-.5)
    assert steps[1]["damage_delta"] == pytest.approx(.2) and steps[1]["damage"] == pytest.approx(.2)
    assert episodes[0]["return"] == .75 and episodes[0]["training_return"] == pytest.approx(-.25)
    assert episodes[1]["return"] == .75 and episodes[1]["training_return"] == pytest.approx(-.25)
    assert steps[6]["reward"] == .25 and steps[6]["training_reward"] == pytest.approx(-4.75)
    assert steps[7]["damage_delta"] == 0 and steps[7]["training_reward"] == .5
    assert episodes[3]["damage"] == 1 and episodes[4]["damage"] == pytest.approx(.2)
    for episode in episodes:
        selected = [step for step in steps if step["episode"] == episode["episode"]]
        assert len(selected) == episode["length"]
        assert sum(step["reward"] for step in selected) == pytest.approx(episode["return"])
        assert sum(step["training_reward"] for step in selected) == pytest.approx(episode["training_return"])
        assert sum(step["damage_delta"] for step in selected) == pytest.approx(episode["damage"])
        assert episode["training_return"] == pytest.approx(episode["return"] - 5 * episode["damage"])
    assert learner.eligible_at_update[:4] == [1] * 4
    assert all(value in (.25, .5, -.5, -4.75) for value in learner.sampled_rewards)
    assert replay.active_length == 0
    for index, cp in enumerate(result["checkpoints"]):
        assert cp["target"] == [6, 8, 12, 16][index]
        assert cp["reward_prediction_target"] == "training_reward"
        assert "reward_loss" not in cp["rolling_update_means"]
        assert cp["rolling_update_means"]["training_reward_loss"] == pytest.approx(.2)
        assert cp["frozen_train_probe"]["training_reward_mae"] == .1
        assert cp["same_observation_prior_vs_mppi_interval"]["planned_actions"] == [2, 2, 4, 4][index]
        assert cp["same_observation_prior_vs_mppi"]["delta_l2"] == .1
        state = torch.load(output / cp["path"], map_location="cpu", weights_only=False)
        assert state["format"] == operator.FORMAT and state["reward_target"] == operator.REWARD_TARGET
        assert state["probe"]["reward"].shape == (1, 2, 1)
        assert state["replay"]["active"] is None and not state["resume_supported"]
        assert operator.base.digest(output / cp["path"]) == cp["sha256"]
    states = replay.state_dict()["episodes"]
    assert states[0]["rewards"].tolist() == pytest.approx([.25, -.5])
    assert states[0]["truncated"].tolist() == [False, True]
    assert states[0]["terminal"].tolist() == [False, False]  # ordinary time limit bootstraps
    assert states[1]["truncated"].tolist() == [False, True]
    assert states[1]["terminal"].tolist() == [False, True]  # HAIC finish does not bootstrap
    assert states[2]["terminated"].tolist() == [False, True]
    assert states[2]["terminal"].tolist() == [False, True]

    class ConstantQ:
        def pi(self, next_z, task):
            return torch.zeros_like(next_z), {}

        def Q(self, next_z, action, task, *, return_type, target):
            return torch.ones_like(next_z)

    dummy = SimpleNamespace(model=ConstantQ(), discount=.995)
    reward = torch.tensor([[states[i]["rewards"][-1]] for i in range(3)]).reshape(3, 1, 1)
    terminal = torch.tensor([[states[i]["terminal"][-1]] for i in range(3)], dtype=torch.float32).reshape(3, 1, 1)
    td = TDMPC2Learner._td_target(cast(TDMPC2Learner, dummy), torch.zeros_like(reward), reward, terminal)
    assert td.flatten().tolist() == pytest.approx([-.5 + .995, -.5, -.5])
    with pytest.raises(FileExistsError):
        operator.run(Path("experiments/unused.json"), "a" * 64, output, root=tmp_path,
                     env_factory=lambda _: pytest.fail("resumed partial run"))


@pytest.mark.parametrize("invalid,pattern,first_damage", [
    (True, "out-of-bounds", 0.0), ("decreasing", "decreased within episode", .4),
])
def test_invalid_telemetry_preserves_consumed_step_intent_and_no_exact_resume(
        tmp_path, monkeypatch, invalid, pattern, first_damage):
    _p, env, _learner, _replay, output = _fake_run(tmp_path, monkeypatch, invalid=invalid)
    with pytest.raises(ValueError, match=pattern):
        operator.run(Path("experiments/unused.json"), "a" * 64, output, root=tmp_path,
                     env_factory=lambda _: env, clock=lambda: 0)
    rows = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    partial = rows[-1]
    assert env.closed and env.resets == 1
    assert partial["event"] == "partial" and not partial["resume_supported"]
    assert partial["decisions"] == 1 and partial["environment_steps_applied"] == 2
    assert partial["reset_intents"] == 1 and partial["environment_resets"] is None
    assert partial["pending_step"]["decision"] == 2 and partial["active_length"] == 1
    assert partial["pending_step"]["damage_repr"] == ("0.2" if invalid == "decreasing" else "-0.1")
    recorded = json.loads((output / "steps.jsonl").read_text().splitlines()[0])
    assert recorded["reward"] == .25
    assert recorded["damage"] == pytest.approx(first_damage)
    assert recorded["training_reward"] == pytest.approx(.25 - 5 * first_damage)
    assert len((output / "steps.jsonl").read_text().splitlines()) == 1
    assert not (output / "result.json").exists()


def test_source_recheck_failure_journals_partial_without_environment(tmp_path, monkeypatch):
    _p, _env, _learner, _replay, output = _fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(operator, "_runtime_sources", lambda *_: (_ for _ in ()).throw(
        ValueError("source drift")))
    with pytest.raises(ValueError, match="source drift"):
        operator.run(Path("experiments/unused.json"), "a" * 64, output, root=tmp_path,
                     env_factory=lambda _: pytest.fail("constructed environment before source recheck"),
                     clock=lambda: 0)
    rows = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["start", "partial"]
    assert rows[-1]["environment_steps_applied"] == 0 and rows[-1]["pending_step"] is None
    assert not (output / "steps.jsonl").exists()


def test_resource_race_after_start_is_journaled_with_zero_resets(tmp_path, monkeypatch):
    _p, env, _learner, _replay, output = _fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(operator.base, "_resources", lambda *_: (_ for _ in ()).throw(
        ValueError("synthetic cgroup floor failure")))
    with pytest.raises(ValueError, match="synthetic cgroup"):
        operator.run(Path("experiments/unused.json"), "a" * 64, output, root=tmp_path,
                     env_factory=lambda _: pytest.fail("resource failure constructed env"), clock=lambda: 0)
    rows = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["start", "partial"]
    assert rows[-1]["environment_resets"] == rows[-1]["environment_steps_applied"] == 0
    assert rows[-1]["reset_intents"] == 0 and rows[-1]["resume_supported"] is False
    assert not (output / "failure.json").exists() and env.resets == 0


def test_cuda_identity_failure_has_exclusive_zero_interaction_receipt(tmp_path, monkeypatch):
    p, env, _learner, _replay, output = _fake_run(tmp_path, monkeypatch)
    p["training"]["device"] = "cuda"  # synthetic preflight only; no CUDA model is constructed
    monkeypatch.setattr(operator.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(operator.torch.cuda, "get_device_name", lambda *_: (_ for _ in ()).throw(
        RuntimeError("synthetic CUDA name failure")))
    with pytest.raises(RuntimeError, match="synthetic CUDA name failure"):
        operator.run(Path("experiments/unused.json"), "a" * 64, output, root=tmp_path,
                     env_factory=lambda _: pytest.fail("device failure constructed env"), clock=lambda: 0)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["event"] == "partial" and failure["reason"] == "RuntimeError"
    assert failure["protocol_sha256"] == "a" * 64
    assert failure["environment_resets"] == failure["environment_steps_applied"] == 0
    assert failure["reset_intents"] == 0 and failure["resume_supported"] is False
    assert env.resets == 0 and not (output / "result.json").exists()
    with pytest.raises(FileExistsError):
        operator.run(Path("experiments/unused.json"), "a" * 64, output, root=tmp_path,
                     env_factory=lambda _: pytest.fail("restarted failed output"))


@pytest.mark.parametrize("failed_event", ["start", "partial"])
def test_journal_failure_uses_independent_fsynced_receipt(tmp_path, monkeypatch, failed_event):
    _p, env, _learner, _replay, output = _fake_run(tmp_path, monkeypatch)
    journal = operator.base._journal

    def fail_selected(path, row):
        if row["event"] == failed_event:
            raise OSError("synthetic journal failure")
        journal(path, row)

    monkeypatch.setattr(operator.base, "_journal", fail_selected)
    if failed_event == "partial":
        monkeypatch.setattr(operator.base, "_resources", lambda *_: (_ for _ in ()).throw(
            ValueError("synthetic resource race")))
    with pytest.raises((OSError, ValueError), match="synthetic"):
        operator.run(Path("experiments/unused.json"), "a" * 64, output, root=tmp_path,
                     env_factory=lambda _: pytest.fail("journal failure constructed env"), clock=lambda: 0)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["event"] == "partial" and not failure["resume_supported"]
    assert failure["environment_resets"] == failure["environment_steps_applied"] == 0
    assert failure["reset_intents"] == 0 and env.resets == 0
    assert failure["reason"] == ("OSError" if failed_event == "start" else "ValueError")
    assert not (output / "result.json").exists()


def test_real_shaped_probe_reports_shaped_mae_returns_and_preserves_rng():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(17)
        model = WorldModel(TDMPC2ModelConfig(
            action_dim=3, obs_shape={"rgb": (4, 64, 64)}, num_channels=2,
            latent_dim=32, mlp_dim=32, num_q=3, num_bins=11, episodic=True, dropout=0))
        learner = TDMPC2Learner(model, TDMPC2LearnerConfig(
            episode_length=2000, horizon=3, batch_size=8))
        replay = EpisodeReplay(6, 3, action_dim=3, seed=41)
        for episode, raw_rewards in enumerate(((.25, .5, 1.5), (-.25, .25, .5))):
            shaper = DamageIncrementReward()
            shaper.reset()
            image = np.full((4, 64, 64), 32 + episode, np.uint8)
            replay.start_episode(image)
            for step, raw in enumerate(raw_rewards):
                shaped, damage_delta = shaper.step(raw, .2 * (step + 1))
                assert damage_delta == pytest.approx(.2)
                replay.add_step(image, np.zeros(3, np.float32), shaped, truncated=step == 2)
        sampler = deepcopy(replay.rng.bit_generator.state)
        probe = operator.base._freeze_probe(replay, 8)
        assert replay.rng.bit_generator.state == sampler
        assert set(probe["episode_id"].tolist()) == {0, 1}
        shaped = probe["reward"].float().squeeze(-1)
        h3 = shaped.sum(dim=0)
        assert float(h3.max()) < 0  # raw H3 returns are +2.25 and +0.5
        before = operator.base._rng_state()

        operator.base._seed(834)
        model.eval()
        with torch.inference_mode():
            z = model.encode(probe["obs"][0], None)
            next_z = model.encode(probe["obs"][1], None)
            model.next(z, probe["action"][0], None)
            prediction = two_hot_inv(model.reward(z, probe["action"][0], None), model.cfg)
            expected_mae = (prediction - probe["reward"][0]).abs().mean().item()
            expected_constant = (probe["reward"][0].mean() - probe["reward"][0]).abs().mean().item()
        operator.base._restore_rng(before)

        metrics = operator._diagnostic(learner, probe, 834)
        assert metrics == operator._diagnostic(learner, probe, 834)
        assert metrics["training_reward_mae"] == pytest.approx(expected_mae, abs=1e-6)
        assert metrics["constant_training_reward_mae"] == pytest.approx(expected_constant, abs=1e-6)
        assert metrics["training_reward_min"] == pytest.approx(float(shaped.min()))
        assert metrics["training_reward_max"] == pytest.approx(float(shaped.max()))
        assert metrics["h3_training_return_min"] == pytest.approx(float(h3.min()))
        assert metrics["h3_training_return_max"] == pytest.approx(float(h3.max()))
        assert metrics["h3_unique_training_returns_at_1e_minus_6"] == int(torch.unique(
            (h3.double() * 1e6).round()).numel())
        discounted = shaped[0] + learner.discount * shaped[1] + learner.discount**2 * shaped[2]
        assert metrics["h3_discounted_training_return_min"] == pytest.approx(float(discounted.min()))
        assert metrics["h3_discounted_training_return_max"] == pytest.approx(float(discounted.max()))
        assert not any("raw" in key or key == "reward_mae" for key in metrics)
        after = operator.base._rng_state()
        assert after["python"] == before["python"]
        np.testing.assert_array_equal(after["numpy"][1], before["numpy"][1])
        torch.testing.assert_close(after["torch"], before["torch"], atol=0, rtol=0)
        if before["cuda"] is not None:
            for actual, saved in zip(after["cuda"], before["cuda"]):
                torch.testing.assert_close(actual, saved, atol=0, rtol=0)
        assert replay.rng.bit_generator.state == sampler
    finally:
        torch.set_num_threads(threads)


def test_diagnostic_renames_all_raw_reward_terms_to_training_targets(monkeypatch):
    metrics = {"reward_min": -1, "reward_max": 2, "reward_mae": .2,
               "constant_reward_mae": .3, "h3_raw_return_min": -3,
               "h3_raw_return_max": 4, "h3_discounted_return_min": -2,
               "h3_discounted_return_max": 3, "h3_unique_raw_returns_at_1e_minus_6": 5,
               "terminal_positive_transitions": 0, "q_scale": 1.0}
    monkeypatch.setattr(operator.base, "_diagnostic", lambda *_: metrics)
    renamed = operator._diagnostic(None, {}, 733)
    assert renamed["training_reward_mae"] == .2
    assert renamed["h3_unique_training_returns_at_1e_minus_6"] == 5
    assert renamed["h3_discounted_training_return_max"] == 3
    assert renamed["terminal_positive_transitions"] == 0
    assert not any("raw" in key or key.startswith("reward_") for key in renamed)
