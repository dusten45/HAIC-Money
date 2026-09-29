"""Synthetic H5 planner-only contracts; no real environment construction/reset."""

from copy import deepcopy
import hashlib
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import evaluate_tdmpc2_h5_planner as operator
from tests.test_evaluate_tdmpc2_full_train import frozen as raw_frozen


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line(record):
    return (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()


@pytest.fixture
def frozen(raw_frozen, monkeypatch):
    root, raw_protocol, raw_freeze, _, source_path, result_path, training, steps, checkpoint, _ = raw_frozen
    source = deepcopy(raw_protocol["source"])
    monkeypatch.setattr(operator, "SOURCE", source)
    monkeypatch.setattr(operator, "SOURCE_CURSOR", (2, 2, 1))
    # Baseline's REAL frozen proof is tested separately. The synthetic RAW
    # source below still passes the actual unchanged H3 _source/_ledger gate.
    monkeypatch.setattr(operator, "_baseline", lambda _root: None)
    monkeypatch.setattr(operator.torch.version, "cuda", None)
    monkeypatch.setattr(operator.torch, "__version__", "2.1.0+cpu")
    h5_source = root / "scripts/evaluate_tdmpc2_h5_planner.py"
    shutil.copyfile(operator.ROOT / "scripts/evaluate_tdmpc2_h5_planner.py", h5_source)
    gates = {}
    for kind in ("quality", "branch"):
        slug = "logged" if kind == "quality" else "branch"
        protocol = root / f"experiments/tdmpc2-h5-{slug}-fake-protocol.json"
        result = root / f"experiments/tdmpc2-h5-{slug}-fake-result.json"
        protocol.write_bytes(line({"format": "synthetic-only", "source": source,
                                   "baseline": operator.BASELINE,
                                   **({"horizons": [3, 5]} if kind == "quality" else
                                      {"horizon": 5, "quality_result": gates["quality"]["result"]})}))
        result.write_bytes(line({"gate_passed": True, "protocol_sha256": sha(protocol),
                                 "source_checkpoint_sha256": sha(checkpoint),
                                 "source_training_result_sha256": sha(result_path),
                                 **({"environment_resets": 0} if kind == "quality" else
                                    {"real_horizon": 5,
                                     "quality_result_sha256": gates["quality"]["result"]["sha256"]})}))
        gates[kind] = {"protocol": {"path": protocol.relative_to(root).as_posix(), "sha256": sha(protocol)},
                       "result": {"path": result.relative_to(root).as_posix(), "sha256": sha(result)}}
    p = {"format": operator.FORMAT, "purpose": "consumed-TRAIN-development",
         "evaluation_source_sha256": sha(h5_source), "raw_evaluator_sha256": operator.RAW_EVALUATOR_SHA,
         "source": source, "baseline": deepcopy(operator.BASELINE), "h5_gates": gates,
         "cells": deepcopy(operator.raw.CELLS), "environment": deepcopy(operator.raw.ENVIRONMENT),
         "modes": ["mppi"], "repeats": 2, "seed": 20260928, "max_steps": 2000,
         "planner": deepcopy(operator.PLANNER), "budget": deepcopy(operator.BUDGET),
         "output_dir": "runs/tdmpc2-h5-planner-full-train-synthetic"}
    path = root / "experiments/tdmpc2-h5-planner-synthetic.json"

    def freeze():
        path.write_bytes(line(p))
        return sha(path)

    return SimpleNamespace(root=root, p=p, path=path, freeze=freeze, gates=gates,
                           source_path=source_path, result_path=result_path, checkpoint=checkpoint,
                           training=training, steps=steps, raw_freeze=raw_freeze)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.]))

    def encode(self, obs, task):
        return self.weight.expand(obs.shape[0], 1)


class FakePlanner:
    def __init__(self, model, config):
        assert (config.action_dim, config.horizon, config.num_samples, config.num_pi_trajs,
                config.iterations, config.num_elites, config.discount, config.episodic) == (
                    3, 5, 512, 24, 6, 64, .995, True)
        self.config = config
        self.reset_count = 0
        self.calls = []
        self.random_first = []

    def reset(self):
        self.reset_count += 1

    def plan(self, obs, *, t0, eval_mode):
        assert eval_mode and obs.shape == (1, 4, 64, 64)
        self.calls.append(t0)
        if t0:
            self.random_first.append(torch.rand(1).item())
        return torch.tensor([.5, .25, -.25])


class FakeEnv:
    def __init__(self, max_steps):
        assert max_steps == 2000
        self.unwrapped = self
        self.resets = []
        self.steps = 0
        self.closed = False
        self.track_id = self.track_seed = None

    def reset(self, *, seed, options):
        assert options == {"track_id": 1} and seed in operator.raw.ROADS
        self.track_id, self.track_seed = 1, seed
        self.resets.append(seed)
        self.steps = 0
        return np.zeros((4, 84, 84), np.float32), {}

    def step(self, action):
        assert action.dtype == np.float32 and action.shape == (3,)
        np.testing.assert_array_equal(action, np.array([.5, .625, .375], np.float32))
        self.steps += 1
        index = len(self.resets) - 1
        finish = index == 0 and self.steps == 2000
        censor = index == 4 and self.steps == 2000
        terminate = index not in (0, 4) and self.steps == 2
        return np.zeros((4, 84, 84), np.float32), 1., terminate, finish or censor, {
            "progress": 1. if finish else .25, "damage": .2,
            "finished": finish, "finish_time_s": 12. if finish else None}

    def close(self):
        self.closed = True


def test_real_frozen_raw_baseline_and_h3_source_are_hash_bound():
    assert sha(operator.ROOT / "scripts/evaluate_tdmpc2_full_train.py") == operator.RAW_EVALUATOR_SHA
    assert sha(operator.ROOT / operator.BASELINE["full_eval_result"]["path"]) == (
        operator.BASELINE["full_eval_result"]["sha256"])
    operator._baseline(operator.ROOT)
    pin = operator._source(operator.ROOT)
    assert pin["sha256"] == operator.SOURCE["checkpoints"][0]["sha256"]
    assert (pin["decisions"], pin["updates"], pin["episodes"]) == (100354, 100354, 307)


def test_preflight_blocks_even_self_declared_passing_fake_h5_gates(frozen, monkeypatch):
    f = frozen
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("torch.load before H5 gates"))
    checked = operator.preflight(f.path, f.freeze(), root=f.root)
    assert checked["status"] == "blocked" and checked["environment_resets"] == 0
    assert checked["torch_load_calls"] == 0 and checked["planned_episodes"] == 8
    assert checked["reason"] == "h5_quality_and_real_branch_gate_schemas_not_frozen"
    assert not (f.root / f.p["output_dir"]).exists()
    with pytest.raises(ValueError, match="DORMANT.*environment_resets=0"):
        operator.execute(f.path, sha(f.path), root=f.root)
    f.p["h5_gates"] = {}
    missing = operator.preflight(f.path, f.freeze(), root=f.root)
    assert missing["reason"] == "missing_h5_quality_or_real_branch_gate"
    assert missing["environment_resets"] == 0 and not (f.root / f.p["output_dir"]).exists()


def test_unsatisfied_h5_result_cannot_self_authorize(frozen, monkeypatch):
    f = frozen
    result = f.root / f.gates["quality"]["result"]["path"]
    failed = json.loads(result.read_text())
    failed["gate_passed"] = False
    result.write_bytes(line(failed))
    f.p["h5_gates"]["quality"]["result"]["sha256"] = sha(result)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("loaded failed gate"))
    with pytest.raises(ValueError):
        operator.execute(f.path, f.freeze(), root=f.root)
    assert not (f.root / f.p["output_dir"]).exists()


@pytest.mark.parametrize("tamper", ["protocol_sha", "h5_source", "raw_source", "checkpoint",
                                    "training", "steps", "result", "source_protocol", "cursor",
                                    "gate_sha", "gate_missing", "gate_rebound_source", "h3_relabel", "damage_relabel",
                                    "horizon", "samples", "modes", "road", "seed", "capped",
                                    "output_exists", "gpu_runtime", "disk_budget"])
def test_sha_gate_source_and_runtime_tamper_denied_before_load_or_reset(frozen, monkeypatch, tamper):
    f = frozen
    if tamper == "h5_source":
        (f.root / "scripts/evaluate_tdmpc2_h5_planner.py").write_bytes(b"drift")
    elif tamper == "raw_source":
        (f.root / "scripts/evaluate_tdmpc2_full_train.py").write_bytes(b"drift")
    elif tamper in ("checkpoint", "training", "steps", "result", "source_protocol"):
        path = {"checkpoint": f.checkpoint, "training": f.training, "steps": f.steps,
                "result": f.result_path, "source_protocol": f.source_path}[tamper]
        path.write_bytes(path.read_bytes() + b"drift")
    elif tamper == "cursor":
        f.p["source"]["checkpoints"][0]["step_cursor_sha256"] = "0" * 64
    elif tamper in ("gate_sha", "gate_missing", "gate_rebound_source"):
        path = f.root / f.gates["quality"]["result"]["path"]
        if tamper == "gate_sha":
            path.write_bytes(b"{}\n")
        elif tamper == "gate_missing":
            path.unlink()
        else:
            result = json.loads(path.read_text())
            result["source_checkpoint_sha256"] = "0" * 64
            path.write_bytes(line(result))
            f.p["h5_gates"]["quality"]["result"]["sha256"] = sha(path)
    elif tamper in ("h3_relabel", "damage_relabel"):
        f.p["h5_gates"]["branch"]["result"] = deepcopy(
            operator.BASELINE["full_eval_result"] if tamper == "h3_relabel" else
            {"path": "experiments/tdmpc2-damage-full-consumed-train-v1-result.json", "sha256": "0" * 64})
    elif tamper == "horizon":
        f.p["planner"]["horizon"] = 3
    elif tamper == "samples":
        f.p["planner"]["num_samples"] = 1024
    elif tamper == "modes":
        f.p["modes"] = ["prior", "mppi"]
    elif tamper == "road":
        f.p["cells"][0]["geometry_seed"] = 42
    elif tamper == "seed":
        f.p["seed"] += 1
    elif tamper == "capped":
        f.p["max_steps"] = 500
    elif tamper == "output_exists":
        (f.root / f.p["output_dir"]).mkdir()
    elif tamper == "gpu_runtime":
        monkeypatch.setattr(torch.version, "cuda", "12.1")
    elif tamper == "disk_budget":
        f.p["budget"]["min_disk_available_bytes"] = 0
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("deserialized before H5 preflight"))
    with pytest.raises(ValueError):
        operator.execute(f.path, "0" * 64 if tamper == "protocol_sha" else f.freeze(), root=f.root)
    assert not (f.root / f.p["output_dir"] / "episodes.jsonl").exists()


def test_partial_training_ledger_tamper_fails_even_if_result_rehashed(frozen, monkeypatch):
    f = frozen
    ledger = f.training
    ledger.write_bytes(ledger.read_bytes() + line({"event": "partial", "decisions": 3}))
    result = json.loads(f.result_path.read_text())
    result["training_ledger_sha256"] = sha(ledger)
    f.result_path.write_bytes(line(result))
    f.p["source"]["result"]["sha256"] = sha(f.result_path)
    # Synthetic SOURCE is deliberately re-pinned to force the *actual* H3
    # complete-cursor checker, not only the earlier result hash check.
    monkeypatch.setattr(operator, "SOURCE", deepcopy(f.p["source"]))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("loaded partial checkpoint"))
    with pytest.raises(ValueError, match="partial or unexpected training event"):
        operator.preflight(f.path, f.freeze(), root=f.root)


def test_h5_config_and_real_planner_action_forks_from_h3_without_new_config_rng():
    class Model:
        def encode(self, obs, task):
            return torch.zeros((obs.shape[0], 1))

        def pi(self, z, task):
            action = torch.full((z.shape[0], 3), .2)
            return action, {"mean": action}

        def next(self, z, action, task):
            return z + action[:, :1]

        def reward(self, z, action, task):
            # A finite, action-sensitive distributional raw reward model.
            bins = torch.zeros((z.shape[0], 101))
            bins[:, 45] = -5 * action[:, 0]
            bins[:, 55] = 5 * action[:, 0]
            return bins

        def termination(self, z, task):
            return torch.zeros_like(z)

        def Q(self, z, action, task, return_type="avg"):
            return z * 0

    from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner

    model = Model()
    torch.manual_seed(71)
    before = torch.get_rng_state().clone()
    h5 = operator._planner(model)
    assert torch.equal(torch.get_rng_state(), before)
    assert h5.config == PlannerConfig(action_dim=3, discount=.995, horizon=5, episodic=True,
                                      num_samples=512, num_pi_trajs=24, iterations=6, num_elites=64)
    h3 = TDMPC2Planner(model, PlannerConfig(action_dim=3, discount=.995, horizon=3, episodic=True))
    obs = torch.zeros(1, 4, 64, 64)
    torch.manual_seed(71)
    action_h5 = h5.plan(obs, t0=True, eval_mode=True)
    torch.manual_seed(71)
    action_h3 = h3.plan(obs, t0=True, eval_mode=True)
    assert action_h5.shape == action_h3.shape == (3,)
    assert not torch.equal(action_h5, action_h3)
    assert h5.prev_mean.shape == (5, 3) and h3.prev_mean.shape == (3, 3)


def test_synthetic_h5_full_episodes_finish_at_2000_and_no_extra_rng(frozen):
    f = frozen
    model = FakeModel()
    planner = operator._planner(model, factory=FakePlanner)
    assert isinstance(planner, FakePlanner)
    envs = []

    def make_env(n):
        envs.append(FakeEnv(n))
        return envs[-1]

    pin = {"sha256": operator.SOURCE["checkpoints"][0]["sha256"]}
    report = operator._run_core(f.root, f.p, f.path, f.freeze(), pin, model, planner, env_factory=make_env,
                                recheck=lambda: True)
    assert len(envs) == 1 and envs[0].closed
    assert envs[0].resets == list(operator.raw.ROADS) * 2
    assert planner.reset_count == 8 and len(planner.calls) == report["primary_mppi"]["decisions"]
    assert sum(planner.calls) == 8
    for index, observed in enumerate(planner.random_first):
        torch.manual_seed(20260928 + index)
        assert observed == torch.rand(1).item()
    ledger = f.root / f.p["output_dir"] / "episodes.jsonl"
    rows = [json.loads(row) for row in ledger.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["reset_intent", "episode"] * 8
    assert rows[1]["decisions"] == 2000 and rows[1]["finished"] and rows[1]["terminal"]
    assert rows[1]["truncated"] and not rows[1]["censored"]
    assert rows[9]["decisions"] == 2000 and rows[9]["censored"] and not rows[9]["terminal"]
    summary = report["primary_mppi"]
    assert (summary["episodes"], summary["finishes"], summary["censored"], summary["uncensored"]) == (8, 1, 1, 7)
    assert summary["roads"][0]["episodes"] == 2 and summary["roads"][0]["finishes"] == 1
    assert summary["roads"][0]["mean_raw_return"] == 2000
    assert report["primary_local_target"]["finishes_required"] == 4
    assert not report["primary_local_target"]["met_on_reused_train"]
    assert not report["generalization_claim"] and not report["official_score"]
    assert report["protocol_sha256"] == sha(f.path)
    assert report["episodes_sha256"] == sha(ledger)
    assert json.loads((f.root / f.p["output_dir"] / "result.json").read_text()) == report
    with pytest.raises(FileExistsError):
        operator._run_core(f.root, f.p, f.path, sha(f.path), pin, model, planner,
                           env_factory=make_env, recheck=lambda: True)


def test_latency_budget_stops_before_action_applied_and_preserves_reset_intent(frozen):
    f = frozen
    model = FakeModel()
    planner = operator._planner(model, factory=FakePlanner)
    envs = []

    def make_env(n):
        envs.append(FakeEnv(n))
        return envs[-1]

    times = iter([0., 0., 0., 5.1])
    with pytest.raises(TimeoutError, match="action latency"):
        operator._run_core(f.root, f.p, f.path, f.freeze(), operator.SOURCE["checkpoints"][0], model, planner,
                           env_factory=make_env, recheck=lambda: True, clock=lambda: next(times))
    assert len(envs) == 1 and envs[0].closed and envs[0].resets == [operator.raw.ROADS[0]]
    assert envs[0].steps == 0
    ledger = f.root / f.p["output_dir"] / "episodes.jsonl"
    rows = [json.loads(row) for row in ledger.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["reset_intent", "partial"]
    assert rows[-1]["reset_intents"] == 1 and rows[-1]["environment_resets"] is None
    assert not (ledger.parent / "result.json").exists()


def test_source_gate_recheck_blocks_next_reset_and_keeps_first_episode(frozen):
    f = frozen
    model = FakeModel()
    planner = operator._planner(model, factory=FakePlanner)
    envs = []
    calls = 0

    def recheck():
        nonlocal calls
        calls += 1
        return calls <= 2

    def make_env(n):
        envs.append(FakeEnv(n))
        return envs[-1]

    with pytest.raises(ValueError, match="gate changed before reset"):
        operator._run_core(f.root, f.p, f.path, f.freeze(), operator.SOURCE["checkpoints"][0],
                           model, planner, env_factory=make_env, recheck=recheck)
    assert calls == 3 and envs[0].closed and envs[0].resets == [operator.raw.ROADS[0]]
    ledger = f.root / f.p["output_dir"] / "episodes.jsonl"
    records = [json.loads(row) for row in ledger.read_text().splitlines()]
    assert [row["event"] for row in records] == ["reset_intent", "episode", "partial"]
    assert records[-1]["complete_episodes"] == 1 and records[-1]["reset_intents"] == 1
    assert not (ledger.parent / "result.json").exists()


def test_core_refuses_stale_protocol_sha_before_env_or_output(frozen):
    f = frozen
    model = FakeModel()
    planner = operator._planner(model, factory=FakePlanner)
    f.freeze()
    with pytest.raises(ValueError, match="protocol SHA"):
        operator._run_core(f.root, f.p, f.path, "0" * 64, operator.SOURCE["checkpoints"][0],
                           model, planner, env_factory=lambda _: pytest.fail("env constructed"),
                           recheck=lambda: True)
    assert not (f.root / f.p["output_dir"]).exists()


def test_wall_and_memory_budget_stop_before_reset(frozen, monkeypatch):
    f = frozen
    model = FakeModel()
    planner = operator._planner(model, factory=FakePlanner)
    made = []
    times = iter([0., 21601.])
    with pytest.raises(TimeoutError, match="wall budget"):
        operator._run_core(f.root, f.p, f.path, f.freeze(), operator.SOURCE["checkpoints"][0], model, planner,
                           env_factory=lambda n: made.append(n), recheck=lambda: True,
                           clock=lambda: next(times))
    assert not made
    receipt = [json.loads(row) for row in (f.root / f.p["output_dir"] / "episodes.jsonl").read_text().splitlines()]
    assert receipt[-1]["event"] == "partial" and receipt[-1]["environment_resets"] == 0
    f.p["output_dir"] += "-rss"
    monkeypatch.setattr(operator.resource, "getrusage", lambda _: SimpleNamespace(
        ru_maxrss=operator.BUDGET["max_process_rss_mib"] * 1024 + 1))
    with pytest.raises(MemoryError, match="RSS budget"):
        operator._run_core(f.root, f.p, f.path, f.freeze(), operator.SOURCE["checkpoints"][0], model, planner,
                           env_factory=lambda n: made.append(n), recheck=lambda: True)
    assert not made


def test_failure_fallback_fsyncs_conservative_partial_receipt(frozen, monkeypatch):
    f = frozen
    model = FakeModel()
    planner = operator._planner(model, factory=FakePlanner)
    envs = []

    class BrokenEnv(FakeEnv):
        def step(self, action):
            raise RuntimeError("synthetic applied-step failure")

    def make_env(n):
        envs.append(BrokenEnv(n))
        return envs[-1]

    journal = operator.raw._journal

    def fail_partial(path, row):
        if row["event"] == "partial":
            raise OSError("cannot append partial")
        journal(path, row)

    monkeypatch.setattr(operator.raw, "_journal", fail_partial)
    with pytest.raises(RuntimeError, match="synthetic applied-step failure"):
        operator._run_core(f.root, f.p, f.path, f.freeze(), operator.SOURCE["checkpoints"][0], model, planner,
                           env_factory=make_env, recheck=lambda: True)
    assert envs[0].closed and envs[0].resets == [operator.raw.ROADS[0]]
    output = f.root / f.p["output_dir"]
    failure = json.loads((output / "failure.json").read_text())
    assert failure["event"] == "failure" and failure["reset_intents"] == 1
    assert failure["protocol_sha256"] == sha(f.path)
    assert failure["environment_resets"] is None and failure["resume_supported"] is False
    assert failure["episode_ledger_sha256"] == sha(output / "episodes.jsonl")
    assert not (output / "result.json").exists()
