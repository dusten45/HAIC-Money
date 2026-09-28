"""Synthetic full TRAIN evaluator contracts; never construct a real HAIC environment."""

import hashlib
import json
from copy import deepcopy
import shutil

import numpy as np
import pytest
import torch

from scripts import evaluate_tdmpc2_full_train as operator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line(row):
    return (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(operator, "TARGETS", (2,))
    monkeypatch.setattr(operator, "TRAIN_SETTINGS", {
        **operator.TRAIN_SETTINGS, "seed_steps": 2, "pretrain_updates": 2})
    root = tmp_path
    (root / "runs").mkdir()
    (root / "experiments").mkdir()
    source = json.loads((operator.ROOT / "experiments/tdmpc2-long-reused-train-v1.json").read_text())
    for name in (*operator.SOURCE_PATHS, "scripts/evaluate_tdmpc2_full_train.py",
                 "experiments/drqv2-geometry-mix-v1-r6.json",
                 "experiments/dreamerv3-reused-train-diagnostic-v1.json"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(operator.ROOT / name, destination)
    source["source_sha256"] = {name: sha(root / name) for name in operator.SOURCE_PATHS}
    source["run_dir"] = "runs/tdmpc2-long-synthetic"
    source["training"].update(seed_steps=2, pretrain_updates=2)
    source["checkpoint_targets"] = [2]
    source_path = root / "experiments/tdmpc2-long-reused-train-synthetic.json"
    source_path.write_bytes(line(source))
    run = root / source["run_dir"]
    run.mkdir()
    source_hash = sha(source_path)
    action = np.zeros(3, np.float32).tobytes()
    native = np.array([0, .5, .5], np.float32).tobytes()
    step_rows = [{"decision": i + 1, "episode": 0, "track_id": 1,
                  "geometry_seed": operator.ROADS[0], "action_f32_hex": action.hex(),
                  "native_action_f32_hex": native.hex(), "reward": .25,
                  "terminated": i == 1, "truncated": False, "terminal": i == 1} for i in range(2)]
    steps = run / "steps.jsonl"
    steps.write_bytes(b"".join(map(line, step_rows)))
    episode = {"event": "episode", "episode": 0, **operator.CELLS[0], "decisions": 2,
               "updates": 2, "length": 2, "return": .5, "progress": .1, "damage": 0.,
               "finished": False, "terminated": True, "truncated": False, "terminal": True,
               "action_trace_sha256": hashlib.sha256(action * 2).hexdigest(),
               "native_action_trace_sha256": hashlib.sha256(native * 2).hexdigest()}
    name = "checkpoint-at-least-000002-step-000002.pt"
    checkpoint = run / name
    torch.save({"format": source["format"], "protocol_sha256": source_hash,
                "source_sha256": source["source_sha256"], "resume_supported": False,
                "action_dim": 3, "target": 2, "decisions": 2, "updates": 2, "episodes": 1,
                "learner": {"q_scale": torch.ones(1), "model.weight": torch.tensor([.2])}}, checkpoint)
    report = {"target": 2, "decisions": 2, "updates": 2, "episodes": 1,
              "rolling_update_count": 2, "train_episodes_since_previous": [{
                  k: episode[k] for k in ("episode", "track_id", "geometry_seed", "decisions",
                                       "length", "return", "progress", "damage", "finished")}],
              "path": name, "sha256": sha(checkpoint), "step_ledger_sha256": sha(steps)}
    ledger_rows = [
        {"event": "start", "protocol_sha256": source_hash, "source_sha256": source["source_sha256"],
         "resume_supported": False, "seed_schedule": source["seed_schedule"]},
        {"event": "reset_intent", "episode": 0, "decisions": 0, **operator.CELLS[0]},
        {"event": "reset", "episode": 0, "decisions": 0, **operator.CELLS[0]},
        episode, {"event": "checkpoint", **report}]
    ledger = run / "training.jsonl"
    ledger.write_bytes(b"".join(map(line, ledger_rows)))
    cursor = hashlib.sha256(ledger.read_bytes()).hexdigest()
    result = {"status": "completed_boundary_at_least_100k", "protocol_sha256": source_hash,
              "source_sha256": source["source_sha256"], "action_dim": 3,
              "reused_train_only": True, "resume_supported": False, "evaluation": None,
              "seed_schedule": source["seed_schedule"], "pretrain_updates": 2,
              "decisions": 2, "episodes": 1, "updates": 2, "checkpoints": [report],
              "training_ledger_sha256": sha(ledger), "step_ledger_sha256": sha(steps)}
    result_path = run / "result.json"
    result_path.write_bytes(line(result))
    protocol = {"format": operator.FORMAT, "purpose": "consumed-TRAIN-development",
                "evaluation_source_sha256": sha(root / "scripts/evaluate_tdmpc2_full_train.py"),
                "source": {
                    "protocol": {"path": source_path.relative_to(root).as_posix(), "sha256": source_hash},
                    "result": {"path": result_path.relative_to(root).as_posix(), "sha256": sha(result_path)},
                    "checkpoints": [{"target": 2, "path": checkpoint.relative_to(root).as_posix(),
                                     "sha256": sha(checkpoint), "training_cursor_sha256": cursor,
                                     "step_cursor_sha256": sha(steps)}]},
                "cells": deepcopy(operator.CELLS), "environment": deepcopy(operator.ENVIRONMENT),
                "modes": list(operator.MODES), "repeats": 2, "seed": 83100,
                "max_steps": 2000, "output_dir": "runs/tdmpc2-full-train-synthetic"}
    path = root / "experiments/tdmpc2-full-synthetic.json"

    def freeze():
        path.write_bytes(line(protocol))
        return sha(path)

    return root, protocol, freeze, path, source_path, result_path, ledger, steps, checkpoint, ledger_rows


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.]))

    def encode(self, obs, task):
        return self.weight.expand(obs.shape[0], 1)

    def pi(self, latent, task):
        mean = torch.tensor([[.2, .1, -.1]]).expand(latent.shape[0], 3)
        return mean, {"mean": mean}


class FakePlanner:
    instances = []

    def __init__(self, model, config):
        assert config.action_dim == 3 and config.horizon == 3 and config.num_samples == 512
        assert config.iterations == 6 and config.num_pi_trajs == 24 and config.num_elites == 64
        assert config.episodic and config.discount == .995
        self.reset_count = 0
        self.calls = []
        self.instances.append(self)

    def reset(self):
        self.reset_count += 1

    def plan(self, obs, *, t0, eval_mode):
        assert eval_mode
        self.calls.append(t0)
        return torch.tensor([.3, .2, -.2])


class FakeEnv:
    def __init__(self, max_steps):
        assert max_steps == 2000
        self.unwrapped = self
        self.resets = []
        self.track_id = self.track_seed = None
        self.closed = False
        self.step_count = 0

    def reset(self, *, seed, options):
        assert options == {"track_id": 1} and seed in operator.ROADS
        self.track_id, self.track_seed = 1, seed
        self.resets.append(seed)
        self.step_count = 0
        return np.zeros((4, 84, 84), np.float32), {}

    def step(self, action):
        assert action.shape == (3,) and action.dtype == np.float32
        assert np.all((-1 <= action) & (action <= 1))
        self.step_count += 1
        index = len(self.resets) - 1
        finish = index == 0 and self.step_count == 2000
        censored = index == 4 and self.step_count == 2000
        terminate = index not in (0, 4) and self.step_count == 2
        return np.zeros((4, 84, 84), np.float32), 1., terminate, finish or censored, {
            "progress": 1. if finish else .25, "damage": .1,
            "finished": finish, "finish_time_s": 8. if finish else None}

    def close(self):
        self.closed = True


def test_read_only_preflight_rejects_partial_and_no_checkpoint_before_env(frozen, monkeypatch):
    root, protocol, freeze, path, _, result, _, _, _, _ = frozen
    constructed = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("preflight deserialized"))
    expected = freeze()
    checked = operator.preflight(path.relative_to(root), expected, root=root)
    assert operator.preflight(path, expected, root=root) == checked
    assert checked["environment_resets"] == 0 and checked["planned_episodes"] == 16
    assert not (root / protocol["output_dir"]).exists()
    result.unlink()
    with pytest.raises(ValueError, match="missing artifact"):
        operator.preflight(path.relative_to(root), expected, root=root)
    assert constructed == []


@pytest.mark.parametrize("tamper", ["evaluation_hash", "source_hash", "source_code", "result_hash",
                                    "checkpoint_hash", "checkpoint_missing", "cursor", "step_hash", "partial", "road",
                                    "capped", "blind", "output_exists"])
def test_tampering_fails_closed_before_any_reset(frozen, tamper, monkeypatch):
    root, p, freeze, path, source_path, result_path, ledger, steps, checkpoint, rows = frozen
    output = root / p["output_dir"]
    if tamper == "evaluation_hash":
        p["evaluation_source_sha256"] = "0" * 64
    elif tamper == "source_hash":
        p["source"]["protocol"]["sha256"] = "0" * 64
    elif tamper == "source_code":
        (root / "haic/algorithms/tdmpc2/planner.py").write_bytes(b"drift")
    elif tamper == "result_hash":
        result_path.write_bytes(b"{}\n")
    elif tamper == "checkpoint_hash":
        checkpoint.write_bytes(b"drift")
    elif tamper == "checkpoint_missing":
        checkpoint.unlink()
    elif tamper == "cursor":
        p["source"]["checkpoints"][0]["training_cursor_sha256"] = "0" * 64
    elif tamper == "step_hash":
        steps.write_bytes(steps.read_bytes() + b"{}\n")
    elif tamper == "partial":
        rows.append({"event": "partial", "decisions": 3})
        ledger.write_bytes(b"".join(map(line, rows)))
        result = json.loads(result_path.read_text())
        result["training_ledger_sha256"] = sha(ledger)
        result_path.write_bytes(line(result))
        p["source"]["result"]["sha256"] = sha(result_path)
    elif tamper == "road":
        p["cells"][0] = {"track_id": 1, "geometry_seed": 123}
    elif tamper == "capped":
        p["max_steps"] = 500
    elif tamper == "blind":
        p["purpose"] = "blind"
    elif tamper == "output_exists":
        output.mkdir()
    constructed = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("loaded before complete preflight"))
    monkeypatch.setattr(operator.torch.version, "cuda", None)
    with pytest.raises(ValueError):
        operator.execute(path.relative_to(root), freeze(), root=root,
                         env_factory=lambda n: constructed.append(n))
    assert not constructed


def test_full_episodes_separate_resets_finish_at_horizon_and_denominators(frozen, monkeypatch):
    root, p, freeze, path, *_ = frozen
    monkeypatch.setattr(operator.torch.version, "cuda", None)
    FakePlanner.instances.clear()
    envs = []

    def make_env(max_steps):
        env = FakeEnv(max_steps)
        envs.append(env)
        return env

    result = operator.execute(path.relative_to(root), freeze(), root=root, env_factory=make_env,
                              model_factory=FakeModel, planner_factory=FakePlanner)
    assert len(envs) == 1 and envs[0].closed
    assert envs[0].resets == [seed for _repeat in range(2) for seed in operator.ROADS for _mode in operator.MODES]
    assert result["denominators"] == {"checkpoints": 1, "distinct_roads": 4,
                                      "repeats_per_road_mode_checkpoint": 2, "modes": ["prior", "mppi"],
                                      "planned_episodes": 16, "completed_episodes": 16}
    ledger = (root / p["output_dir"] / "episodes.jsonl")
    rows = [json.loads(row) for row in ledger.read_text().splitlines()]
    assert len(rows) == 32 and [row["event"] for row in rows] == ["reset_intent", "episode"] * 16
    episodes = rows[1::2]
    assert [(r["mode"], r["repeat"]) for r in episodes[:2]] == [("prior", 0), ("mppi", 0)]
    assert episodes[0]["decisions"] == 2000 and episodes[0]["finished"] is True
    assert episodes[0]["terminal"] is True and episodes[0]["truncated"] is True
    assert episodes[0]["censored"] is False and episodes[0]["raw_return"] == 2000
    assert episodes[4]["decisions"] == 2000 and episodes[4]["censored"] is True
    assert episodes[4]["terminal"] is False and not episodes[4]["finished"]
    mode = result["per_checkpoint"]["2"]["per_mode"]
    assert mode["prior"]["finishes"] == 1 and mode["prior"]["episodes"] == 8
    assert mode["prior"]["uncensored"] == 7 and mode["prior"]["censored"] == 1
    assert mode["mppi"]["finishes"] == 0 and mode["mppi"]["episodes"] == 8
    assert mode["prior"]["roads"][0]["finishes"] == 1
    assert mode["prior"]["roads"][0]["episodes"] == 2
    assert mode["prior"]["roads"][0]["mean_raw_return"] == 1001
    assert mode["prior"]["decisions"] == 2000 + 2000 + 6 * 2
    assert FakePlanner.instances[0].reset_count == 16
    assert len(FakePlanner.instances[0].calls) == mode["mppi"]["decisions"]
    assert sum(FakePlanner.instances[0].calls) == 8
    assert result["full_episode_finish_comparison_valid"] and not result["generalization_claim"]
    with pytest.raises(ValueError, match="unique"):
        operator.execute(path.relative_to(root), sha(path), root=root,
                         env_factory=lambda _: pytest.fail("duplicate reset"))


def test_malformed_full_episode_preserves_partial_without_extra_reset(frozen, monkeypatch):
    root, p, freeze, path, *_ = frozen
    monkeypatch.setattr(operator.torch.version, "cuda", None)
    class EarlyTruncation(FakeEnv):
        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            return obs, reward, False, self.step_count == 1, {**info, "finished": False}

    envs = []

    def make_env(n):
        envs.append(EarlyTruncation(n))
        return envs[-1]

    with pytest.raises(ValueError, match="finish/truncation"):
        operator.execute(path.relative_to(root), freeze(), root=root,
                         env_factory=make_env, model_factory=FakeModel, planner_factory=FakePlanner)
    assert len(envs) == 1 and envs[0].resets == [operator.ROADS[0]] and envs[0].closed
    assert not (root / p["output_dir"] / "result.json").exists()
    rows = [json.loads(row) for row in (root / p["output_dir"] / "episodes.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["reset_intent", "partial"]


def test_source_drift_after_first_episode_prevents_next_reset(frozen, monkeypatch):
    root, p, freeze, path, *_ = frozen
    monkeypatch.setattr(operator.torch.version, "cuda", None)
    class DriftAfterFirstEpisode(FakeEnv):
        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            if self.step_count == 2:
                (root / "haic/algorithms/tdmpc2/model.py").write_bytes(b"changed between resets")
            return obs, reward, self.step_count == 2, False, info

    envs = []

    def make_env(n):
        envs.append(DriftAfterFirstEpisode(n))
        return envs[-1]

    with pytest.raises(ValueError, match="source hash mismatch"):
        operator.execute(path.relative_to(root), freeze(), root=root, env_factory=make_env,
                         model_factory=FakeModel, planner_factory=FakePlanner)
    assert len(envs) == 1 and envs[0].resets == [operator.ROADS[0]] and envs[0].closed
    rows = [json.loads(row) for row in (root / p["output_dir"] / "episodes.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["reset_intent", "episode", "partial"]
