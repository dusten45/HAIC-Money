"""Synthetic H3-noise development operator contracts; no real HAIC reset."""

from copy import deepcopy
import hashlib
import json
import os
import shutil

import numpy as np
import pytest
import torch

from scripts import evaluate_tdmpc2_full_train as raw
from scripts import evaluate_tdmpc2_h3_noise_train as operator
from tests.test_evaluate_tdmpc2_full_train import FakeModel, frozen as raw_frozen


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line(record):
    return (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()


@pytest.fixture
def frozen(raw_frozen, monkeypatch):
    root, source_eval, _, _, source_path, source_result_path, training, steps, checkpoint, _ = raw_frozen
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch, "__version__", "2.1.0+cpu")
    monkeypatch.setattr(operator, "_headroom", lambda: 2_000_000_000)
    copied = root / "scripts/evaluate_tdmpc2_h3_noise_train.py"
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(operator.ROOT / "scripts/evaluate_tdmpc2_h3_noise_train.py", copied)
    source = deepcopy(source_eval["source"])
    monkeypatch.setattr(operator, "SOURCE", source)
    monkeypatch.setattr(operator, "SOURCE_CURSOR", (2, 2, 1))

    historical = deepcopy(source_eval)
    historical["output_dir"] = "runs/tdmpc2-full-train-synthetic-baseline"
    historical["seed"] = 20260928
    historical_protocol = root / "experiments/tdmpc2-full-consumed-train-synthetic.json"
    historical_protocol.write_bytes(line(historical))
    output = root / historical["output_dir"]
    output.mkdir()
    episodes = []
    records = []
    for repeat in range(2):
        for index, cell in enumerate(raw.CELLS):
            for mode in raw.MODES:
                expected = {"target": 2, "mode": mode, "repeat": repeat,
                            "episode_seed": 20260928 + repeat * 4 + index, **cell}
                records.append({"event": "reset_intent", **expected})
                episode = {"event": "episode", **expected, "decisions": 2, "max_steps": 2000,
                           "raw_return": 2., "progress": .2, "damage": .1,
                           "finished": False, "terminated": True, "truncated": False,
                           "terminal": True, "censored": False, "finish_time_s": None,
                           "action_trace_sha256": "1" * 64, "native_action_trace_sha256": "2" * 64,
                           "action_latency_first_s": .1, "action_latency_mean_s": .1,
                           "action_latency_max_s": .1, "action_latency_total_s": .2}
                episodes.append(episode)
                records.append(episode)
    ledger = output / "episodes.jsonl"
    ledger.write_bytes(b"".join(map(line, records)))
    pin = {**source["checkpoints"][0], "line": 5, "decisions": 2, "updates": 2, "episodes": 1}
    primary = {"status": "complete", "protocol_sha256": sha(historical_protocol),
               "source_result_sha256": sha(source_result_path), "episodes_sha256": sha(ledger),
               "checkpoints": [pin], "full_episode_finish_comparison_valid": True,
               "denominators": {"checkpoints": 1, "completed_episodes": 16, "distinct_roads": 4,
                                "modes": ["prior", "mppi"], "planned_episodes": 16,
                                "repeats_per_road_mode_checkpoint": 2},
               "per_checkpoint": raw._summary(episodes, [pin], 2)}
    primary_path = output / "result.json"
    primary_path.write_bytes(line(primary))

    wrapper = {"status": "complete_valid_full_episode_finish_comparison",
               "evaluation_protocol_sha256": sha(historical_protocol),
               "evaluation_operator_sha256": sha(root / "scripts/evaluate_tdmpc2_full_train.py"),
               "source_model_sha256": sha(checkpoint),
               "source_training_result_sha256": sha(source_result_path),
               "primary_result": primary_path.relative_to(root).as_posix(),
               "primary_result_sha256": sha(primary_path),
               "episode_ledger": ledger.relative_to(root).as_posix(),
               "episode_ledger_sha256": sha(ledger),
               "denominators": {"distinct_training_roads": 4, "repeats_per_road_per_mode": 2,
                                "prior_episodes": 8, "mppi_episodes": 8, "total_episodes": 16,
                                "max_decisions_per_episode": 2000},
               "validity": {"all_scheduled_episodes_complete": True, "full_episode_finish_comparison_valid": True,
                            "censored_episodes": 0, "generalization_claim": False},
               "prior": {"finishes": 0, "episodes": 8}, "mppi": {"finishes": 0, "episodes": 8}}
    wrapper_path = root / "experiments/tdmpc2-full-consumed-train-synthetic-result.json"
    wrapper_path.write_bytes(line(wrapper))

    training_result = {"run_status": "completed_boundary_at_least_100k",
                       "training_protocol_sha256": sha(source_path),
                       "run_result_sha256": sha(source_result_path),
                       "checkpoint_sha256": sha(checkpoint),
                       "training_ledger_sha256": sha(training), "step_ledger_sha256": sha(steps),
                       "decisions": 2, "updates": 2, "episodes": 1}
    summary_path = root / "experiments/tdmpc2-long-reused-train-synthetic-final.json"
    summary_path.write_bytes(line(training_result))
    refs = {"training_result": {"path": summary_path.relative_to(root).as_posix(), "sha256": sha(summary_path)},
            "full_eval_protocol": {"path": historical_protocol.relative_to(root).as_posix(),
                                   "sha256": sha(historical_protocol)},
            "full_eval_result": {"path": wrapper_path.relative_to(root).as_posix(), "sha256": sha(wrapper_path)}}
    monkeypatch.setattr(operator, "BASELINE", refs)
    eval_protocol = {"format": operator.FORMAT, "purpose": "consumed-TRAIN-development",
                     "evaluation_source_sha256": sha(copied),
                     "raw_evaluator_sha256": operator.RAW_EVALUATOR_SHA,
                     "source": deepcopy(source), "baseline": deepcopy(refs),
                     "cells": deepcopy(raw.CELLS), "environment": deepcopy(raw.ENVIRONMENT),
                     "modes": ["mppi"], "repeats": 2, "seed": 20260928,
                     "max_steps": 2000, "planner": deepcopy(operator.PLANNER),
                     "resources": {"measured_peak_rss_bytes": 100_000_000,
                                   "additional_memory_bytes": 100_000_000, "memory_reserve_bytes": 10_000_000,
                                   "remaining_disk_bytes": 10000, "disk_reserve_bytes": 10000,
                                   "max_wall_seconds": 3600},
                     "output_dir": "runs/tdmpc2-h3-noise-full-train-synthetic"}
    path = root / "experiments/tdmpc2-h3-noise-synthetic.json"

    def freeze():
        path.write_bytes(line(eval_protocol))
        return sha(path)

    return locals()


class FakePlanner:
    def __init__(self):
        from haic.algorithms.tdmpc2.planner import PlannerConfig

        self.config = PlannerConfig(action_dim=3, discount=.995, horizon=3, episodic=True,
                                    num_samples=512, num_pi_trajs=24, iterations=6, num_elites=64)
        self.calls = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def plan(self, obs, *, t0, eval_mode):
        self.calls.append((t0, eval_mode))
        return torch.tensor([.2 if eval_mode else .5, .1, -.1])


class FakeEnv:
    def __init__(self, max_steps, *, broken=False, drift=None):
        assert max_steps == 2000
        self.unwrapped = self
        self.resets = []
        self.actions = []
        self.steps = 0
        self.closed = False
        self.broken = broken
        self.drift = drift

    def reset(self, *, seed, options):
        assert options == {"track_id": 1} and seed in raw.ROADS
        self.track_seed = seed
        self.track_id = 1
        self.resets.append(seed)
        self.steps = 0
        return np.zeros((4, 84, 84), np.float32), {}

    def step(self, action):
        assert action.dtype == np.float32 and action.shape == (3,)
        self.actions.append(action.copy())
        self.steps += 1
        if self.drift is not None and self.steps == 2:
            self.drift()
        finish = len(self.resets) - 1 in (0, 1, 4, 5) and self.steps == 2
        truncated = self.broken or finish
        return np.zeros((4, 84, 84), np.float32), 1., not truncated and self.steps == 2, truncated, {
            "progress": 1. if finish else .2, "damage": .1, "finished": finish,
            "finish_time_s": 1. if finish else None}

    def close(self):
        self.closed = True


def test_read_only_preflight_verifies_real_raw_evidence_without_reset_or_model_load(monkeypatch):
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("preflight deserialized"))
    _, pin = operator._source(operator.ROOT)
    operator._baseline(operator.ROOT, pin)
    assert pin["sha256"] == operator.SOURCE["checkpoints"][0]["sha256"]


def test_synthetic_preflight_and_mode_difference(frozen, monkeypatch):
    f = frozen
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("preflight deserialized"))
    checked = operator.preflight(f["path"], f["freeze"](), root=f["root"])
    assert checked["environment_resets"] == 0 and checked["torch_load_calls"] == 0
    assert checked["planned_episodes"] == 8 and checked["planner"]["eval_mode"] is False
    assert not (f["root"] / f["eval_protocol"]["output_dir"]).exists()

    baseline_env, noise_env = FakeEnv(2000), FakeEnv(2000)
    baseline, noise = FakePlanner(), FakePlanner()
    old = raw._episode(baseline_env, FakeModel(), baseline, raw.CELLS[0], mode="mppi",
                       repeat=0, seed=20260928, checkpoint_target=2)
    new = operator._episode(noise_env, FakeModel(), noise, raw.CELLS[0], repeat=0,
                            seed=20260928, budget=lambda: None)
    assert old["finished"] == new["finished"]
    assert old["action_trace_sha256"] != new["action_trace_sha256"]
    assert baseline.calls == [(True, True), (False, True)]
    assert noise.calls == [(True, False), (False, False)]
    assert np.allclose(baseline_env.actions[0], [.2, .55, .45])
    assert np.allclose(noise_env.actions[0], [.5, .55, .45])


def test_action_latency_above_five_seconds_stops_before_step():
    env = FakeEnv(2000)
    times = iter((0.0, 5.001))
    with pytest.raises(TimeoutError, match="latency"):
        operator._episode(env, FakeModel(), FakePlanner(), raw.CELLS[0], repeat=0,
                          seed=20260928, budget=lambda: None, clock=lambda: next(times))
    assert env.resets == [raw.ROADS[0]] and env.actions == []


@pytest.mark.parametrize("tamper", ["protocol_sha", "source_code", "source_result", "training_ledger",
                                     "checkpoint", "historical_episode", "historical_result", "historical_protocol",
                                     "wrong_mode", "wrong_seed", "output_exists", "resource_telemetry"])
def test_source_tamper_and_scope_fail_before_output(frozen, monkeypatch, tamper):
    f = frozen
    expected = f["freeze"]()
    if tamper == "protocol_sha":
        expected = "0" * 64
    elif tamper == "source_code":
        (f["root"] / "haic/algorithms/tdmpc2/planner.py").write_bytes(b"drift")
    elif tamper == "source_result":
        f["source_result_path"].write_bytes(b"{}\n")
    elif tamper == "training_ledger":
        f["training"].write_bytes(f["training"].read_bytes() + b"{}\n")
    elif tamper == "checkpoint":
        f["checkpoint"].write_bytes(b"drift")
    elif tamper == "historical_episode":
        f["ledger"].write_bytes(f["ledger"].read_bytes() + b"{}\n")
    elif tamper == "historical_result":
        f["primary_path"].write_bytes(b"{}\n")
    elif tamper == "historical_protocol":
        f["historical_protocol"].write_bytes(b"{}\n")
    elif tamper == "wrong_mode":
        f["eval_protocol"]["planner"]["eval_mode"] = True
        expected = f["freeze"]()
    elif tamper == "wrong_seed":
        f["eval_protocol"]["seed"] = 20260929
        expected = f["freeze"]()
    elif tamper == "output_exists":
        (f["root"] / f["eval_protocol"]["output_dir"]).mkdir()
    elif tamper == "resource_telemetry":
        monkeypatch.setattr(operator, "_headroom", lambda: 0)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("loaded before preflight"))
    with pytest.raises(ValueError):
        operator.preflight(f["path"], expected, root=f["root"])
    if tamper != "output_exists":
        assert not (f["root"] / f["eval_protocol"]["output_dir"]).exists()


def test_private_core_eight_full_episodes_two_roads_and_exclusive_receipt(frozen):
    f = frozen
    digest = f["freeze"]()
    p, checked = operator._check(f["path"], digest, root=f["root"])
    planner = FakePlanner()
    envs = []

    def env_factory(n):
        envs.append(FakeEnv(n))
        return envs[-1]

    result = operator._run_core(f["root"], p, f["path"], digest, checked["checkpoint"],
                                FakeModel(), planner, env_factory=env_factory)
    assert len(envs) == 1 and envs[0].closed
    assert envs[0].resets == list(raw.ROADS) * 2
    assert result["primary_mppi"]["episodes"] == 8
    assert result["primary_mppi"]["finishes"] == 4
    assert result["primary_local_target"] == {
        "min_finishes": 4, "min_distinct_finished_roads": 2,
        "observed_finishes": 4, "observed_distinct_finished_roads": 2,
        "met_on_reused_train": True, "fresh_generalization_claim": False}
    assert all(not eval_mode for _, eval_mode in planner.calls)
    assert planner.resets == 8 and sum(t0 for t0, _ in planner.calls) == 8
    ledger = f["root"] / p["output_dir"] / "episodes.jsonl"
    rows = [json.loads(row) for row in ledger.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["reset_intent", "episode"] * 8
    assert sha(ledger) == result["episodes_sha256"]
    assert json.loads((ledger.parent / "result.json").read_text()) == result
    with pytest.raises(ValueError, match="exclusive"):
        operator.preflight(f["path"], digest, root=f["root"])


def test_source_drift_stops_before_second_reset_and_keeps_partial(frozen):
    f = frozen
    digest = f["freeze"]()
    p, checked = operator._check(f["path"], digest, root=f["root"])
    envs = []

    def env_factory(n):
        envs.append(FakeEnv(n, drift=lambda: (f["root"] / "haic/algorithms/tdmpc2/model.py").write_bytes(b"drift")))
        return envs[-1]

    with pytest.raises(ValueError, match="source hash mismatch"):
        operator._run_core(f["root"], p, f["path"], digest, checked["checkpoint"],
                           FakeModel(), FakePlanner(), env_factory=env_factory)
    assert envs[0].resets == [raw.ROADS[0]] and envs[0].closed
    output = f["root"] / p["output_dir"]
    assert not (output / "result.json").exists()
    events = [json.loads(row)["event"] for row in (output / "episodes.jsonl").read_text().splitlines()]
    assert events == ["reset_intent", "episode", "partial"]


def test_source_drift_during_last_episode_prevents_complete_receipt(frozen):
    f = frozen
    digest = f["freeze"]()
    p, checked = operator._check(f["path"], digest, root=f["root"])
    envs = []

    def env_factory(n):
        envs.append(FakeEnv(n, drift=lambda: (
            f["root"] / "haic/algorithms/tdmpc2/model.py").write_bytes(b"last-episode drift")
            if len(envs[0].resets) == 8 else None))
        return envs[-1]

    with pytest.raises(ValueError, match="source hash mismatch"):
        operator._run_core(f["root"], p, f["path"], digest, checked["checkpoint"],
                           FakeModel(), FakePlanner(), env_factory=env_factory)
    assert len(envs[0].resets) == 8 and envs[0].closed
    output = f["root"] / p["output_dir"]
    assert not (output / "result.json").exists()
    rows = [json.loads(row) for row in (output / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 17 and rows[-1]["event"] == "partial" and rows[-1]["complete_episodes"] == 8


def test_malformed_episode_preserves_reset_intent_and_partial(frozen):
    f = frozen
    digest = f["freeze"]()
    p, checked = operator._check(f["path"], digest, root=f["root"])
    envs = []

    def env_factory(n):
        envs.append(FakeEnv(n, broken=True))
        return envs[-1]

    with pytest.raises(ValueError, match="finish/truncation"):
        operator._run_core(f["root"], p, f["path"], digest, checked["checkpoint"],
                           FakeModel(), FakePlanner(), env_factory=env_factory)
    assert envs[0].resets == [raw.ROADS[0]] and envs[0].closed
    rows = [json.loads(row) for row in (f["root"] / p["output_dir"] / "episodes.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rows] == ["reset_intent", "partial"]
    assert rows[-1]["reset_intents"] == 1 and rows[-1]["resume_supported"] is False


def test_final_fsync_failure_does_not_erase_episode_evidence(frozen, monkeypatch):
    f = frozen
    digest = f["freeze"]()
    p, checked = operator._check(f["path"], digest, root=f["root"])
    fsync = os.fsync
    failed = False

    def fail_final(fd):
        nonlocal failed
        if not failed and "/result.json" in os.readlink(f"/proc/self/fd/{fd}"):
            failed = True
            raise OSError("synthetic final receipt fsync failure")
        return fsync(fd)

    monkeypatch.setattr(operator.os, "fsync", fail_final)
    with pytest.raises(OSError, match="synthetic final receipt fsync failure"):
        operator._run_core(f["root"], p, f["path"], digest, checked["checkpoint"],
                           FakeModel(), FakePlanner(), env_factory=FakeEnv)
    assert failed
    output = f["root"] / p["output_dir"]
    rows = [json.loads(row) for row in (output / "episodes.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["reset_intent", "episode"] * 8 + ["partial"]
    assert rows[-1]["complete_episodes"] == 8 and rows[-1]["resume_supported"] is False
    assert (output / "result.json").exists()


def test_partial_fsync_failure_writes_independent_failure_receipt(frozen, monkeypatch):
    f = frozen
    digest = f["freeze"]()
    p, checked = operator._check(f["path"], digest, root=f["root"])
    journal = raw._journal

    def fail_partial(path, row):
        if row["event"] == "partial":
            raise OSError("synthetic partial fsync failure")
        return journal(path, row)

    monkeypatch.setattr(operator.raw, "_journal", fail_partial)
    with pytest.raises(ValueError, match="finish/truncation"):
        operator._run_core(f["root"], p, f["path"], digest, checked["checkpoint"],
                           FakeModel(), FakePlanner(), env_factory=lambda n: FakeEnv(n, broken=True))
    output = f["root"] / p["output_dir"]
    fallback = json.loads((output / "failure.json").read_text())
    assert fallback["reset_intents"] == 1 and fallback["complete_episodes"] == 0
    assert fallback["environment_resets"] is None and fallback["resume_supported"] is False
    assert fallback["episode_ledger_sha256"] == sha(output / "episodes.jsonl")
