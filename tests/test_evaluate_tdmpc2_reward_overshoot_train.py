"""Synthetic-only overshoot full-episode safety; never construct the HAIC env."""

from copy import deepcopy
import hashlib
import json
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import evaluate_tdmpc2_full_train as raw
from scripts import evaluate_tdmpc2_reward_overshoot_train as operator
from tests.test_evaluate_tdmpc2_full_train import FakeModel


REAL_SCORE = operator._score
REAL_SOURCE = operator._source


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line(row):
    return (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path
    (root / "experiments").mkdir()
    (root / "runs").mkdir()
    executable = root / "scripts/evaluate_tdmpc2_reward_overshoot_train.py"
    executable.parent.mkdir()
    shutil.copyfile(operator.ROOT / "scripts/evaluate_tdmpc2_reward_overshoot_train.py", executable)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch, "__version__", "2.1.0+cpu")
    monkeypatch.setattr(operator, "_runtime", lambda: {"runtime": "synthetic-native-box2d"})
    monkeypatch.setattr(operator, "_baseline", lambda _root: {
        "baseline_checkpoint_sha256": "a" * 64, "mppi_episodes": 8, "mppi_finishes": 0,
        "mppi_censored": 0, "measured_peak_rss_bytes": 6462996480})
    pin = {"target": 100000, "path": "runs/tdmpc2-overshoot-20260929-v1/checkpoint-at-least-100000-step-100001.pt",
           "sha256": "b" * 64, "training_cursor_sha256": "c" * 64,
           "step_cursor_sha256": "d" * 64, "line": 900, "decisions": 100001,
           "updates": 100001, "episodes": 300, "training_ledger_sha256_before_checkpoint": "e" * 64}
    variant = {"source_sha256": {"scripts/train_tdmpc2_reward_overshoot.py": "f" * 64},
               "reward_overshoot": {"reward_target": "raw"}}
    result = {"training_ledger_sha256": "1" * 64, "step_ledger_sha256": "2" * 64}
    monkeypatch.setattr(operator, "_source", lambda *_: (variant, result, pin))
    gate = {"receipt_sha256": "3" * 64, "body_sha256": "4" * 64,
            "h5_concordant": 66, "h3_concordant": 31,
            "nonregressing_roads": 3, "normalized_h5_error": .3}
    monkeypatch.setattr(operator, "_score", lambda *_: gate)
    monkeypatch.setattr(operator.noise, "_headroom", lambda: 48 * 1024**3)
    monkeypatch.setattr(operator.shutil, "disk_usage", lambda _: SimpleNamespace(free=48 * 1024**3))
    protocol = {"format": operator.FORMAT, "purpose": "consumed-TRAIN-development",
                "evaluation_source_sha256": sha(executable),
                "raw_evaluator_sha256": operator.RAW_EVALUATOR_SHA,
                "noise_evaluator_sha256": operator.NOISE_EVALUATOR_SHA,
                "source": {"protocol": deepcopy(operator.TRAIN_PROTOCOL),
                           "result": {"path": "runs/tdmpc2-overshoot-20260929-v1/result.json", "sha256": "5" * 64},
                           "checkpoint": {k: pin[k] for k in (
                               "target", "path", "sha256", "training_cursor_sha256", "step_cursor_sha256")}},
                "branch_gate": {"protocol": {"path": "experiments/tdmpc2-overshoot-old-branch-score-v1.json",
                                             "sha256": "6" * 64},
                                "receipt": {"path": "runs/tdmpc2-overshoot-old-branch-score-20260929-v1.json",
                                            "sha256": gate["receipt_sha256"]}},
                "baseline": deepcopy(operator.noise.BASELINE),
                "resource_reference": deepcopy(operator.RESOURCE_PROTOCOL),
                "cells": deepcopy(raw.CELLS), "environment": deepcopy(raw.ENVIRONMENT),
                "modes": ["mppi"], "repeats": 2, "seed": 20260928, "max_steps": 2000,
                "planner": deepcopy(operator.PLANNER), "runtime": operator._runtime(),
                "resources": {"measured_peak_rss_bytes": 6462996480,
                              "additional_memory_bytes": 8589934592,
                              "memory_reserve_bytes": 24 * 1024**3,
                              "remaining_disk_bytes": 67108864,
                              "disk_reserve_bytes": 24 * 1024**3,
                              "max_wall_seconds": 21600},
                "output_dir": "runs/tdmpc2-overshoot-full-train-synthetic"}
    path = root / "experiments/tdmpc2-overshoot-full-test.json"

    def freeze():
        path.write_bytes(line(protocol))
        return sha(path)

    return SimpleNamespace(root=root, protocol=protocol, path=path, freeze=freeze,
                           pin=pin, gate=gate, result=result, variant=variant, executable=executable)


class FakePlanner:
    def __init__(self):
        from haic.algorithms.tdmpc2.planner import PlannerConfig
        self.config = PlannerConfig(action_dim=3, discount=.995, episodic=True, horizon=3,
                                    num_samples=512, num_pi_trajs=24, iterations=6, num_elites=64)
        self.calls = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def plan(self, obs, *, t0, eval_mode):
        self.calls.append((t0, eval_mode))
        return torch.tensor([.1, .2, -.3])


class FakeEnv:
    def __init__(self, max_steps, *, break_finish=False, drift=None):
        assert max_steps == 2000
        self.unwrapped = self
        self.resets = []
        self.actions = []
        self.closed = False
        self.break_finish = break_finish
        self.drift = drift

    def reset(self, *, seed, options):
        assert seed in raw.ROADS and options == {"track_id": 1}
        self.track_seed = seed
        self.track_id = 1
        self.resets.append(seed)
        self.steps = 0
        return np.zeros((4, 84, 84), np.float32), {}

    def step(self, action):
        assert action.shape == (3,) and action.dtype == np.float32
        self.actions.append(action.copy())
        self.steps += 1
        if self.drift and self.steps == 2:
            self.drift()
        finish = len(self.resets) - 1 in (0, 1, 4, 5) and self.steps == 2
        truncated = finish or self.break_finish
        return np.zeros((4, 84, 84), np.float32), 1., not truncated and self.steps == 2, truncated, {
            "progress": 1. if finish else .25, "damage": .1, "finished": finish,
            "finish_time_s": 2. if finish else None}

    def close(self):
        self.closed = True


def test_preflight_requires_score_and_complete_source_before_any_load(frozen, monkeypatch):
    f = frozen
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("deserialized before preflight"))
    checked = operator.preflight(f.path, f.freeze(), root=f.root)
    assert checked["environment_resets"] == checked["torch_load_calls"] == 0
    assert checked["planned_episodes"] == 8
    assert not (f.root / f.protocol["output_dir"]).exists()
    monkeypatch.setattr(operator, "_score", lambda *_: (_ for _ in ()).throw(ValueError("score FAIL")))
    with pytest.raises(ValueError, match="score FAIL"):
        operator.execute(f.path, sha(f.path), root=f.root)
    assert not (f.root / f.protocol["output_dir"]).exists()


def test_absent_actual_trainer_result_refuses_before_torch_load_or_env_import(frozen, monkeypatch):
    f = frozen
    dest = f.root / operator.TRAIN_PROTOCOL["path"]
    shutil.copyfile(operator.ROOT / operator.TRAIN_PROTOCOL["path"], dest)
    monkeypatch.setattr(operator, "_source", REAL_SOURCE)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("loaded incomplete checkpoint"))
    with pytest.raises(ValueError, match="missing artifact"):
        operator.execute(f.path, f.freeze(), root=f.root)
    assert not (f.root / f.protocol["output_dir"]).exists()


def test_runtime_cli_is_protocol_free_and_does_not_build_environment(frozen, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["synthetic", "--print-runtime"])
    operator.main()
    assert json.loads(capsys.readouterr().out) == {"runtime": "synthetic-native-box2d"}
    monkeypatch.setattr(sys, "argv", ["synthetic", "--print-runtime", "--execute"])
    with pytest.raises(SystemExit):
        operator.main()


@pytest.mark.parametrize("tamper", ["eval_sha", "source_code", "protocol_sha", "wrong_road", "mode",
                                    "seed", "horizon", "resource_reserve", "cgroup_drop", "disk_drop",
                                    "output_exists", "cuda_runtime"])
def test_protocol_scope_sha_runtime_budget_refuse_before_model(frozen, monkeypatch, tamper):
    f = frozen
    expected = f.freeze()
    if tamper == "eval_sha":
        f.protocol["evaluation_source_sha256"] = "0" * 64
    elif tamper == "source_code":
        f.executable.write_bytes(b"drift")
    elif tamper == "protocol_sha":
        expected = "0" * 64
    elif tamper == "wrong_road":
        f.protocol["cells"][0]["geometry_seed"] = 3
    elif tamper == "mode":
        f.protocol["modes"] = ["prior"]
    elif tamper == "seed":
        f.protocol["seed"] = 20260929
    elif tamper == "horizon":
        f.protocol["planner"]["horizon"] = 5
    elif tamper == "resource_reserve":
        f.protocol["resources"]["memory_reserve_bytes"] = 4 * 1024**3
    elif tamper == "cgroup_drop":
        monkeypatch.setattr(operator.noise, "_headroom", lambda: 20 * 1024**3)
    elif tamper == "disk_drop":
        monkeypatch.setattr(operator.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    elif tamper == "output_exists":
        (f.root / f.protocol["output_dir"]).mkdir()
    elif tamper == "cuda_runtime":
        monkeypatch.setattr(torch.version, "cuda", "12.1")
    if tamper in {"eval_sha", "wrong_road", "mode", "seed", "horizon", "resource_reserve"}:
        expected = f.freeze()
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("loaded before SHA/score gate"))
    with pytest.raises(ValueError):
        operator.execute(f.path, expected, root=f.root)
    if tamper != "output_exists":
        assert not (f.root / f.protocol["output_dir"]).exists()


def test_fake_mppi_eight_reused_episodes_two_road_gate(frozen):
    f = frozen
    digest = f.freeze()
    p, checked = operator._check(f.path, digest, root=f.root)
    envs = []

    def make_env(n):
        envs.append(FakeEnv(n))
        return envs[-1]

    planner = FakePlanner()
    report = operator._run_core(f.root, p, f.path, digest, checked, FakeModel(), planner,
                                env_factory=make_env)
    assert len(envs) == 1 and envs[0].closed
    assert envs[0].resets == list(raw.ROADS) * 2
    assert planner.resets == 8 and len(planner.calls) == 16
    assert all(eval_mode for _, eval_mode in planner.calls)
    assert sum(t0 for t0, _ in planner.calls) == 8
    assert report["primary_mppi"]["finishes"] == 4
    assert report["primary_local_gate"]["met_on_reused_train"] is True
    assert report["primary_local_gate"]["observed_distinct_finished_roads"] == 2
    assert report["full_episode_finish_comparison_valid"] and not report["generalization_claim"]
    output = f.root / p["output_dir"]
    ledger = output / "episodes.jsonl"
    rows = [json.loads(row) for row in ledger.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["start"] + ["reset_intent", "episode"] * 8
    assert sha(ledger) == report["episodes_sha256"]
    assert json.loads((output / "result.json").read_text()) == report
    with pytest.raises(ValueError, match="exclusive"):
        operator.preflight(f.path, digest, root=f.root)


def test_five_second_action_latency_stops_before_step():
    env = FakeEnv(2000)
    times = iter((0.0, 5.001))
    with pytest.raises(TimeoutError, match="latency"):
        operator._episode(env, FakeModel(), FakePlanner(), raw.CELLS[0], repeat=0,
                          seed=20260928, budget=lambda: None, clock=lambda: next(times))
    assert env.resets == [raw.ROADS[0]] and env.actions == []


def test_source_or_score_drift_before_second_reset_preserves_partial(frozen, monkeypatch):
    f = frozen
    digest = f.freeze()
    p, checked = operator._check(f.path, digest, root=f.root)
    envs = []

    def make_env(n):
        envs.append(FakeEnv(n, drift=lambda: f.executable.write_bytes(b"drift")))
        return envs[-1]

    with pytest.raises(ValueError, match="executable SHA mismatch"):
        operator._run_core(f.root, p, f.path, digest, checked, FakeModel(), FakePlanner(), env_factory=make_env)
    assert envs[0].resets == [raw.ROADS[0]] and envs[0].closed
    output = f.root / p["output_dir"]
    assert not (output / "result.json").exists()
    rows = [json.loads(row) for row in (output / "episodes.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rows] == ["start", "reset_intent", "episode", "partial"]
    assert rows[-1]["complete_episodes"] == rows[-1]["reset_intents"] == 1


def test_malformed_fake_episode_and_latency_keep_intent_partial(frozen):
    f = frozen
    digest = f.freeze()
    p, checked = operator._check(f.path, digest, root=f.root)
    envs = []

    def make_env(n):
        envs.append(FakeEnv(n, break_finish=True))
        return envs[-1]

    with pytest.raises(ValueError, match="finish/truncation"):
        operator._run_core(f.root, p, f.path, digest, checked, FakeModel(), FakePlanner(), env_factory=make_env)
    assert envs[0].resets == [raw.ROADS[0]] and envs[0].closed
    output = f.root / p["output_dir"]
    rows = [json.loads(row) for row in (output / "episodes.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["start", "reset_intent", "partial"]
    assert rows[-1]["reset_intents"] == 1 and rows[-1]["resume_supported"] is False


def test_partial_fsync_failure_creates_independent_failure_receipt(frozen, monkeypatch):
    f = frozen
    digest = f.freeze()
    p, checked = operator._check(f.path, digest, root=f.root)
    journal = raw._journal

    def broken(path, record):
        if record["event"] == "partial":
            raise OSError("synthetic journal sync failure")
        return journal(path, record)

    monkeypatch.setattr(operator.raw, "_journal", broken)
    with pytest.raises(ValueError, match="finish/truncation"):
        operator._run_core(f.root, p, f.path, digest, checked, FakeModel(), FakePlanner(),
                           env_factory=lambda n: FakeEnv(n, break_finish=True))
    output = f.root / p["output_dir"]
    failure = json.loads((output / "failure.json").read_text())
    assert failure["reset_intents"] == 1 and failure["environment_resets"] is None
    assert failure["episode_ledger_sha256"] == sha(output / "episodes.jsonl")


def test_model_contract_checks_overshoot_checkpoint_format_before_weights(frozen, monkeypatch):
    f = frozen
    checkpoint = f.root / f.pin["path"]
    checkpoint.parent.mkdir(parents=True)
    state = {"format": operator.TRAIN_FORMAT, "protocol_sha256": operator.TRAIN_PROTOCOL["sha256"],
             "source_sha256": f.variant["source_sha256"], "target": 100000,
             "decisions": f.pin["decisions"], "updates": f.pin["updates"],
             "episodes": f.pin["episodes"], "action_dim": 3, "resume_supported": False,
             "step_ledger_sha256": f.pin["step_cursor_sha256"],
             "training_ledger_sha256_before_checkpoint": f.pin["training_ledger_sha256_before_checkpoint"],
             "learner": {"q_scale": torch.ones(1), "model.weight": torch.tensor([.3])},
             "optim": {}, "pi_optim": {}, "probe": {}, "rng": {},
             "replay": {"format": "haic-tdmpc2-episode-replay-v1", "active": None,
                        "next_episode_id": f.pin["episodes"], "size": f.pin["decisions"]},
             **{k: {"reward_target": "raw"} for k in (
                 "baseline_training_protocol", "baseline_training_result", "replay_audit",
                 "throughput_benchmark", "reward_overshoot")}}
    torch.save(state, checkpoint)
    f.pin["sha256"] = sha(checkpoint)
    p = {"source_sha256": f.variant["source_sha256"], **{
        k: state[k] for k in ("baseline_training_protocol", "baseline_training_result", "replay_audit",
                              "throughput_benchmark", "reward_overshoot")}}
    model = operator._model(f.pin, p, f.root, model_factory=FakeModel)
    assert model.weight.item() == pytest.approx(.3)
    state["reward_overshoot"] = {"reward_target": "shaped"}
    torch.save(state, checkpoint)
    f.pin["sha256"] = sha(checkpoint)
    with pytest.raises(ValueError, match="bound complete reward-overshoot"):
        operator._model(f.pin, p, f.root, model_factory=lambda: pytest.fail("constructed untrusted model"))


def test_dynamic_resource_subtracts_loaded_peak_not_reserve(frozen, monkeypatch):
    forecast = frozen.protocol["resources"]
    monkeypatch.setattr(operator.noise, "_headroom", lambda: forecast["memory_reserve_bytes"])
    monkeypatch.setattr(operator.shutil, "disk_usage", lambda _: SimpleNamespace(
        free=forecast["remaining_disk_bytes"] + forecast["disk_reserve_bytes"]))
    reading = operator._resources(frozen.root / "runs", forecast,
                                  allocated_bytes=forecast["additional_memory_bytes"])
    assert reading["required_memory_bytes"] == 24 * 1024**3
    with pytest.raises(ValueError, match="reserve"):
        operator._resources(frozen.root / "runs", forecast,
                            allocated_bytes=forecast["additional_memory_bytes"] - 1)


def _synthetic_score(frozen, monkeypatch):
    """Synthetic perfect predictions over pinned old raw outcomes; no model load."""
    f = frozen
    root = f.root
    names = (operator.TRAIN_PROTOCOL["path"], operator.BRANCH_PROTOCOL["path"],
             operator.BRANCH_PRIMARY["path"],
             "scripts/diagnose_tdmpc2_h5_branches.py",
             "experiments/tdmpc2-h5-branches-v1-result.json")
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(operator.ROOT / name, path)
    trainer = json.loads((root / operator.TRAIN_PROTOCOL["path"]).read_text())
    old_protocol = json.loads((root / operator.BRANCH_PROTOCOL["path"]).read_text())
    for name in set(trainer["source_sha256"]) | set(old_protocol["source_sha256"]):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            shutil.copyfile(operator.ROOT / name, path)
    scorer = "scripts/score_tdmpc2_overshoot_archived_branches.py"
    scorer_test = "tests/test_score_tdmpc2_overshoot_archived_branches.py"
    for name in (scorer, scorer_test):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic no-reset scorer source")
    new_run = root / "runs/tdmpc2-overshoot-20260929-v1"
    new_run.mkdir()
    for name in ("training.jsonl", "steps.jsonl", "result.json"):
        (new_run / name).write_bytes(b"synthetic completed-source evidence\n")
    checkpoint = root / f.pin["path"]
    checkpoint.write_bytes(b"synthetic variant checkpoint")
    old_model = root / operator.noise.SOURCE["checkpoints"][0]["path"]
    old_model.parent.mkdir(parents=True, exist_ok=True)
    old_model.write_bytes(b"synthetic original model; never deserialized")
    original_digest = raw._digest

    def fake_old_sha(path):
        if path == old_model:
            return operator.noise.SOURCE["checkpoints"][0]["sha256"]
        return original_digest(path)

    monkeypatch.setattr(operator.raw, "_digest", fake_old_sha)
    training_source = {"protocol": operator.TRAIN_PROTOCOL,
                       "result": {"path": f"{new_run.relative_to(root)}/result.json",
                                  "sha256": sha(new_run / "result.json")},
                       "checkpoint": {"path": f.pin["path"], "sha256": sha(checkpoint)},
                       "training_ledger": {"path": f"{new_run.relative_to(root)}/training.jsonl",
                                           "sha256": sha(new_run / "training.jsonl")},
                       "step_ledger": {"path": f"{new_run.relative_to(root)}/steps.jsonl",
                                       "sha256": sha(new_run / "steps.jsonl")}}
    original_branch = {"primary": operator.BRANCH_PRIMARY, "protocol": operator.BRANCH_PROTOCOL,
                       "operator": {"path": "scripts/diagnose_tdmpc2_h5_branches.py",
                                    "sha256": sha(root / "scripts/diagnose_tdmpc2_h5_branches.py")},
                       "model": {"path": old_model.relative_to(root).as_posix(),
                                 "sha256": operator.noise.SOURCE["checkpoints"][0]["sha256"]},
                       "summary": {"path": "experiments/tdmpc2-h5-branches-v1-result.json",
                                   "sha256": sha(root / "experiments/tdmpc2-h5-branches-v1-result.json")}}
    source_map = {**trainer["source_sha256"], **old_protocol["source_sha256"],
                  scorer: sha(root / scorer), scorer_test: sha(root / scorer_test)}
    score_protocol = {"format": "haic-tdmpc2-overshoot-old-branch-score-v1",
                      "purpose": "consumed-TRAIN-archived-branch-reward-only",
                      "training_source": training_source, "original_branch": original_branch,
                      "source_sha256": source_map, "runtime": old_protocol["runtime"],
                      "output": "runs/tdmpc2-overshoot-old-branch-score-20260929-v1.json",
                      "gate": {"min_h5_concordant": 66, "h5_informative_pairs": 91,
                               "min_nonregressing_roads": 3, "min_h3_concordant": 31,
                               "h3_informative_pairs": 40,
                               "h5_error_exclusive": .39667698614253405}}
    spec = root / "experiments/tdmpc2-overshoot-old-branch-score-v1.json"
    spec.write_bytes(line(score_protocol))
    old = json.loads((root / operator.BRANCH_PRIMARY["path"]).read_text())
    anchors = []
    roads = {}
    for anchor in old["anchors"]:
        road = str(anchor["geometry_seed"])
        roads.setdefault(road, {"anchors": 0, "informative_real_pairs": 0,
                                "pairs": {"concordant": 0, "discordant": 0,
                                          "real_tie": 0, "predicted_tie": 0}})
        roads[road]["anchors"] += 1
        candidates = [{"candidate": item["candidate"], "model_action_bytes_hex": item["model_action_bytes_hex"],
                       "real_h5_raw_return": item["real_discounted_raw_return"],
                       "real_h3_prefix_raw_return": item["real_h3_prefix_raw_return"],
                       "old_h5_reward_return": item["predicted_reward_return"],
                       "new_h3_reward_return": item["real_h3_prefix_raw_return"],
                       "new_h5_reward_return": item["real_discounted_raw_return"],
                       "ending": item["ending"], "steps": 5} for item in anchor["candidates"]]
        anchors.append({"episode_id": anchor["episode_id"], "start_step": anchor["start_step"],
                        "geometry_seed": anchor["geometry_seed"],
                        "anchor_model_observation_sha256": anchor["anchor_model_observation_sha256"],
                        "candidates": candidates})
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1:]:
                diff = left["real_h5_raw_return"] - right["real_h5_raw_return"]
                key = "real_tie" if abs(diff) <= 1e-6 else "concordant"
                roads[road]["pairs"][key] += 1
                roads[road]["informative_real_pairs"] += key != "real_tie"
    summary = {"h5": {"pairs": {"concordant": 91, "discordant": 0, "real_tie": 29,
                                 "predicted_tie": 0}, "informative_real_pairs": 91,
                      "by_road": roads},
               "h3": {"pairs": {"concordant": 40, "discordant": 0, "real_tie": 80,
                                 "predicted_tie": 0}, "informative_real_pairs": 40},
               "h5_absolute_error_per_discounted_step": 0.0}
    old_counts = [13, 16, 12, 15]
    gate = {"h5_reward_concordance": {"concordant": 91, "informative_real_pairs": 91,
                                      "minimum_concordant": 66, "passed": True},
            "road_nonregression": {"nonregressing_roads": 4, "minimum_roads": 3,
                                   "by_road": {str(road): {
                                       "new_concordant": roads[str(road)]["informative_real_pairs"],
                                       "old_concordant": old_counts[i],
                                       "informative_real_pairs": roads[str(road)]["informative_real_pairs"],
                                       "passed": True}
                                       for i, road in enumerate(raw.ROADS)}, "passed": True},
            "h3_prefix_concordance": {"concordant": 40, "informative_real_pairs": 40,
                                      "minimum_concordant": 31, "passed": True},
            "h5_normalized_error": {"value": 0.0, "threshold_exclusive": .39667698614253405,
                                    "passed": True}, "passed": True}
    receipt = {"format": "haic-tdmpc2-overshoot-old-branch-score-result-v1", "status": "PASS",
               "protocol": {"path": spec.relative_to(root).as_posix(), "sha256": sha(spec)},
               "training_source": training_source, "original_branch": original_branch,
               "source_sha256": source_map, "runtime": old_protocol["runtime"],
               "anchors": anchors, "summary": summary, "gate": gate,
               "environment_resets": 0, "optimizer_updates": 0,
               "scope": "reused_TRAIN_archived_branches_only"}
    output = root / score_protocol["output"]

    def freeze_score():
        receipt["body_sha256"] = hashlib.sha256(json.dumps(
            {k: v for k, v in receipt.items() if k != "body_sha256"},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        output.write_bytes(line(receipt))
        return {"protocol": receipt["protocol"],
                "receipt": {"path": output.relative_to(root).as_posix(), "sha256": sha(output)}}

    pin = {"path": checkpoint.relative_to(root).as_posix(), "sha256": sha(checkpoint)}
    source = {"result": training_source["result"],
              "training_ledger_sha256": training_source["training_ledger"]["sha256"],
              "step_ledger_sha256": training_source["step_ledger"]["sha256"]}
    return SimpleNamespace(root=root, receipt=receipt, output=output, freeze_score=freeze_score,
                           pin=pin, source=source)


def test_synthetic_archived_score_recomputes_four_gates_and_rejects_fail(frozen, monkeypatch):
    proof = _synthetic_score(frozen, monkeypatch)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("score deserialized weights"))
    accepted = REAL_SCORE(proof.root, proof.freeze_score(), proof.source, proof.pin)
    assert accepted["h5_concordant"] == 91 and accepted["h3_concordant"] == 40
    assert accepted["nonregressing_roads"] == 4 and accepted["normalized_h5_error"] == 0
    proof.receipt["status"] = "FAIL"
    with pytest.raises(ValueError, match="PASS"):
        REAL_SCORE(proof.root, proof.freeze_score(), proof.source, proof.pin)


@pytest.mark.parametrize("tamper", ["candidate", "source_checkpoint", "gate_false", "source_bytes",
                                    "old_branch_bytes", "protocol_sha", "absent_score"])
def test_synthetic_score_tamper_refuses_without_loading(frozen, monkeypatch, tamper):
    proof = _synthetic_score(frozen, monkeypatch)
    refs = proof.freeze_score()
    if tamper == "candidate":
        proof.receipt["anchors"][0]["candidates"][0]["new_h5_reward_return"] = 1e5
        refs = proof.freeze_score()
    elif tamper == "source_checkpoint":
        proof.receipt["training_source"]["checkpoint"]["sha256"] = "0" * 64
        refs = proof.freeze_score()
    elif tamper == "gate_false":
        proof.receipt["gate"]["passed"] = False
        refs = proof.freeze_score()
    elif tamper == "source_bytes":
        (proof.root / "haic/algorithms/tdmpc2/planner.py").write_bytes(b"drift")
    elif tamper == "old_branch_bytes":
        (proof.root / operator.BRANCH_PRIMARY["path"]).write_bytes(b"drift")
    elif tamper == "protocol_sha":
        refs["protocol"]["sha256"] = "0" * 64
    elif tamper == "absent_score":
        proof.output.unlink()
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("score loaded weights"))
    with pytest.raises(ValueError):
        REAL_SCORE(proof.root, refs, proof.source, proof.pin)


def test_complete_synthetic_100k_ledger_binds_four_boundaries_and_raw_steps(frozen):
    f = frozen
    run = f.root / "runs/tdmpc2-overshoot-20260929-v1"
    run.mkdir()
    protocol = {"run_dir": run.relative_to(f.root).as_posix(),
                "checkpoint_targets": [20000, 40000, 70000, 100000],
                "source_sha256": {"synthetic.py": "a" * 64},
                "reward_overshoot": {"reward_target": "raw"},
                "runtime": {"torch": "synthetic"}, "seed_schedule": "synthetic complete seed"}
    action = np.zeros(3, np.float32).tobytes()
    native = np.array([0, .5, .5], np.float32).tobytes()
    rows = [{"event": "start", "protocol_sha256": operator.TRAIN_PROTOCOL["sha256"],
             "source_sha256": protocol["source_sha256"], "reward_overshoot": protocol["reward_overshoot"],
             "runtime": protocol["runtime"], "seed_schedule": protocol["seed_schedule"],
             "resume_supported": False}]
    steps = bytearray()
    checkpoints = []
    interval = []
    final_pin = None
    for episode_index in range(50):
        cell = raw.CELLS[episode_index % 4]
        beginning = episode_index * 2000
        rows.extend(({"event": event, "episode": episode_index, "decisions": beginning, **cell}
                     for event in ("reset_intent", "reset")))
        for offset in range(2000):
            ending = offset == 1999
            steps.extend(line({"decision": beginning + offset + 1, "episode": episode_index, **cell,
                               "action_f32_hex": action.hex(), "native_action_f32_hex": native.hex(),
                               "raw_reward_repr": "0.25", "reward": .25,
                               "terminated": ending, "truncated": False, "finished": False,
                               "terminal": ending}))
        decisions = beginning + 2000
        episode = {"event": "episode", "episode": episode_index, **cell,
                   "decisions": decisions, "updates": decisions if decisions >= 10000 else 0,
                   "length": 2000, "return": 500., "progress": .2, "damage": 0.,
                   "finished": False, "terminated": True, "truncated": False, "terminal": True,
                   "finish_time_s": None,
                   "action_trace_sha256": hashlib.sha256(action * 2000).hexdigest(),
                   "native_action_trace_sha256": hashlib.sha256(native * 2000).hexdigest()}
        rows.append(episode)
        interval.append({key: episode[key] for key in (
            "episode", "track_id", "geometry_seed", "decisions", "length", "return",
            "progress", "damage", "finished")})
        if decisions in protocol["checkpoint_targets"]:
            target = decisions
            checkpoint = run / f"checkpoint-at-least-{target:06d}-step-{decisions:06d}.pt"
            checkpoint.write_bytes(f"synthetic {target} milestone".encode())
            prefix = b"".join(map(line, rows))
            step_sha = hashlib.sha256(steps).hexdigest()
            report = {"target": target, "decisions": decisions, "updates": decisions,
                      "episodes": episode_index + 1, "path": checkpoint.name, "sha256": sha(checkpoint),
                      "step_ledger_sha256": step_sha,
                      "training_ledger_sha256_before_checkpoint": hashlib.sha256(prefix).hexdigest(),
                      "train_episodes_since_previous": interval}
            rows.append({"event": "checkpoint", **report})
            checkpoints.append(report)
            interval = []
            if target == 100000:
                final_pin = {"target": target, "path": checkpoint.relative_to(f.root).as_posix(),
                             "sha256": sha(checkpoint),
                             "training_cursor_sha256": hashlib.sha256(b"".join(map(line, rows))).hexdigest(),
                             "step_cursor_sha256": step_sha}
    training = run / "training.jsonl"
    training.write_bytes(b"".join(map(line, rows)))
    step_path = run / "steps.jsonl"
    step_path.write_bytes(steps)
    result = {"decisions": 100000, "updates": 100000, "episodes": 50,
              "checkpoints": checkpoints, "training_ledger_sha256": sha(training),
              "step_ledger_sha256": sha(step_path)}
    assert final_pin is not None
    verified = operator._ledger(f.root, protocol, result, final_pin)
    assert verified["decisions"] == verified["updates"] == 100000
    assert verified["episodes"] == 50
    assert verified["training_ledger_sha256_before_checkpoint"] == checkpoints[-1][
        "training_ledger_sha256_before_checkpoint"]
    step_path.write_bytes(step_path.read_bytes().replace(b'"raw_reward_repr": "0.25"',
                                                         b'"raw_reward_repr": "1.0"', 1))
    result["step_ledger_sha256"] = sha(step_path)
    with pytest.raises(ValueError, match="producer reward"):
        operator._ledger(f.root, protocol, result, final_pin)
