"""Synthetic-only Q0 evaluator tests; never build or reset a real HAIC env."""

from copy import deepcopy
import hashlib
import json
import shutil
from types import SimpleNamespace

import pytest
import torch

from scripts import evaluate_tdmpc2_full_train as raw
from scripts import evaluate_tdmpc2_overshoot_h5_q0 as evaluator
from tests.test_evaluate_tdmpc2_reward_overshoot_train import FakeEnv, FakeModel


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    return {"path": path.relative_to(path.parents[2]).as_posix(), "sha256": digest(path)}


class FakeQ0Planner:
    def __init__(self):
        from haic.algorithms.tdmpc2.planner import PlannerConfig

        self.config = PlannerConfig(action_dim=3, discount=.995, episodic=True, horizon=5,
                                    num_samples=512, num_pi_trajs=24, iterations=6, num_elites=64)
        self.q_weight = 0
        self.calls = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def plan(self, pixels, *, t0, eval_mode):
        self.calls.append((t0, eval_mode))
        return torch.tensor([.1, .2, -.3])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path
    (root / "runs").mkdir()
    executable = root / "scripts/evaluate_tdmpc2_overshoot_h5_q0.py"
    executable.parent.mkdir()
    shutil.copyfile(evaluator.ROOT / "scripts/evaluate_tdmpc2_overshoot_h5_q0.py", executable)
    shutil.copyfile(evaluator.ROOT / "scripts/evaluate_tdmpc2_reward_overshoot_train.py",
                    root / "scripts/evaluate_tdmpc2_reward_overshoot_train.py")
    (root / "haic/algorithms/tdmpc2").mkdir(parents=True)
    (root / "haic/algorithms/tdmpc2/q_bootstrap.py").write_bytes(b"synthetic Q0 planner")
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch, "__version__", "2.1.0+cpu")
    monkeypatch.setattr(evaluator.prior, "_runtime", lambda: {"torch": "2.1.0+cpu"})
    monkeypatch.setattr(evaluator.prior, "_resources", lambda *_args, **_kwargs: {"free": 1})
    monkeypatch.setattr(evaluator.prior, "_baseline", lambda _: {"mppi_finishes": 0})
    pin = {"target": 100000, "path": "runs/tdmpc2-overshoot-20260929-v1/model.pt",
           "sha256": evaluator.OVERSHOOT_CHECKPOINT_SHA, "decisions": 100159, "episodes": 309}
    source = {"protocol": deepcopy(evaluator.prior.TRAIN_PROTOCOL),
              "result": {"path": "runs/tdmpc2-overshoot-20260929-v1/result.json", "sha256": "b" * 64},
              "checkpoint": {key: pin[key] for key in ("target", "path", "sha256")}}
    monkeypatch.setattr(evaluator.prior, "_source", lambda *_: (
        {"source_sha256": {"synthetic": "a" * 64}},
        {"training_ledger_sha256": "c" * 64, "step_ledger_sha256": "d" * 64}, pin))
    monkeypatch.setattr(evaluator, "_ref", lambda _root, ref, _prefix: (
        {"status": "complete_100k_reused_train_model_internal_only",
         "primary_result_sha256": source["result"]["sha256"],
         "selected_checkpoint_sha256": pin["sha256"],
         "training_ledger_sha256": "c" * 64, "step_ledger_sha256": "d" * 64}
        if ref == evaluator.SOURCE_SUMMARY else
        {"mppi": {"finishes": 0, "episodes": 8}} if ref == evaluator.H3_REFERENCES["fixed"] else
        {"performance_evidence": {"finishes": 2, "completed_episodes": 8,
                                   "distinct_finished_roads": 2, "censored": 0}}, root))
    gate = {"receipt_sha256": "e" * 64, "body_sha256": "f" * 64}
    monkeypatch.setattr(evaluator, "_branch", lambda *_: gate)
    p = {"format": evaluator.FORMAT, "purpose": "consumed-TRAIN-development",
         "evaluation_source_sha256": digest(executable),
         "planner_source_sha256": digest(root / "haic/algorithms/tdmpc2/q_bootstrap.py"),
         "source": source, "source_summary": deepcopy(evaluator.SOURCE_SUMMARY),
         "branch_gate": {"protocol": {"path": "experiments/tdmpc2-overshoot-q0-branches-v1.json",
                                       "sha256": "a" * 64},
                         "receipt": {"path": "runs/tdmpc2-overshoot-q0-branches-synthetic.json",
                                     "sha256": gate["receipt_sha256"]}},
         "descriptive_references": deepcopy(evaluator.H3_REFERENCES),
         "cells": deepcopy(raw.CELLS), "environment": deepcopy(raw.ENVIRONMENT),
         "modes": ["mppi"], "repeats": 2, "seed": 20260928, "max_steps": 2000,
         "planner": deepcopy(evaluator.PLANNER), "runtime": evaluator.prior._runtime(),
         "resources": {"measured_peak_rss_bytes": 6462996480,
                       "additional_memory_bytes": 8589934592, "memory_reserve_bytes": 4 * 1024**3,
                       "remaining_disk_bytes": 67108864, "disk_reserve_bytes": 2 * 1024**3,
                       "max_wall_seconds": 21600},
         "output_dir": "runs/tdmpc2-overshoot-h5-q0-full-train-synthetic"}
    protocol = root / "experiments/tdmpc2-q0-synthetic.json"
    protocol.parent.mkdir()
    protocol.write_text(json.dumps(p, sort_keys=True) + "\n")
    return SimpleNamespace(root=root, protocol=protocol, p=p, pin=pin, gate=gate,
                           sha=digest(protocol), executable=executable)


def test_preflight_is_read_only_and_q0_source_bound(setup, monkeypatch):
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: pytest.fail("torch.load before gate"))
    checked = evaluator.preflight(setup.protocol, setup.sha, root=setup.root)
    assert checked["torch_load_calls"] == checked["environment_resets"] == 0
    assert checked["planned_episodes"] == 8 and checked["checkpoint"] == setup.pin
    assert not (setup.root / setup.p["output_dir"]).exists()


@pytest.mark.parametrize("tamper", ["source", "branch", "runtime", "planner", "resource", "output"])
def test_invalid_preflight_refuses_before_torch_or_env(setup, monkeypatch, tamper):
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: pytest.fail("torch.load before source gate"))
    if tamper == "source":
        setup.executable.write_bytes(b"drift")
    elif tamper == "branch":
        monkeypatch.setattr(evaluator, "_branch", lambda *_: (_ for _ in ()).throw(ValueError("branch FAIL")))
    elif tamper == "runtime":
        setup.p["runtime"] = {"torch": "different"}
    elif tamper == "planner":
        setup.p["planner"]["q_weight"] = 1.
    elif tamper == "resource":
        monkeypatch.setattr(evaluator.prior, "_resources", lambda *_a, **_k: (
            _ for _ in ()).throw(ValueError("resource gate")))
    else:
        (setup.root / setup.p["output_dir"]).mkdir()
    if tamper in ("runtime", "planner"):
        setup.protocol.write_text(json.dumps(setup.p, sort_keys=True) + "\n")
        setup.sha = digest(setup.protocol)
    with pytest.raises(ValueError):
        evaluator.execute(setup.protocol, setup.sha, root=setup.root)
    if tamper != "output":
        assert not (setup.root / setup.p["output_dir"]).exists()


def test_fake_q0_eight_episodes_and_road_gate(setup):
    p, checked = evaluator._check(setup.protocol, setup.sha, root=setup.root)
    envs = []

    def factory(max_steps):
        envs.append(FakeEnv(max_steps))
        return envs[-1]

    planner = FakeQ0Planner()
    result = evaluator._run_core(setup.root, p, setup.protocol, setup.sha, checked,
                                 FakeModel(), planner, env_factory=factory)
    assert len(envs) == 1 and envs[0].closed
    assert envs[0].resets == list(raw.ROADS) * 2
    assert planner.resets == 8 and len(planner.calls) == 16
    assert all(mode for _, mode in planner.calls)
    assert sum(t0 for t0, _ in planner.calls) == 8
    assert result["primary_mppi"]["finishes"] == 4
    assert result["primary_local_gate"]["met_on_reused_train"] is True
    assert result["primary_local_gate"]["observed_distinct_finished_roads"] == 2
    ledger = setup.root / p["output_dir"] / "episodes.jsonl"
    assert [json.loads(line)["event"] for line in ledger.read_text().splitlines()] == (
        ["start"] + ["reset_intent", "episode"] * 8)
    assert digest(ledger) == result["episodes_sha256"]
    with pytest.raises(ValueError, match="exclusive"):
        evaluator.preflight(setup.protocol, setup.sha, root=setup.root)


def test_partial_preserved_when_resource_fails_after_one_episode(setup, monkeypatch):
    p, checked = evaluator._check(setup.protocol, setup.sha, root=setup.root)
    state = {"down": False}

    def budget(*_args, **_kwargs):
        if state["down"]:
            raise ValueError("resource exhausted")
        return {"free": 1}

    monkeypatch.setattr(evaluator.prior, "_resources", budget)
    envs = []

    def factory(max_steps):
        envs.append(FakeEnv(max_steps, drift=lambda: state.__setitem__("down", True)))
        return envs[-1]

    with pytest.raises(ValueError, match="resource exhausted"):
        evaluator._run_core(setup.root, p, setup.protocol, setup.sha, checked,
                            FakeModel(), FakeQ0Planner(), env_factory=factory)
    assert envs[0].resets == [raw.ROADS[0]] and envs[0].closed
    output = setup.root / p["output_dir"]
    assert not (output / "result.json").exists()
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rows] == ["start", "reset_intent", "episode", "partial"]
    assert rows[-1]["reset_intents"] == rows[-1]["complete_episodes"] == 1


def test_h5_q0_core_refuses_q1_before_any_reset(setup):
    p, checked = evaluator._check(setup.protocol, setup.sha, root=setup.root)
    planner = FakeQ0Planner()
    planner.q_weight = 1
    with pytest.raises(ValueError, match="q_weight=0"):
        evaluator._run_core(setup.root, p, setup.protocol, setup.sha, checked,
                            FakeModel(), planner, env_factory=lambda _: pytest.fail("made env"))
    assert not (setup.root / p["output_dir"]).exists()


def test_disjoint_branch_recomputes_complete_pass_and_rejects_tamper(tmp_path, monkeypatch):
    root = tmp_path
    (root / "runs").mkdir()
    (root / "experiments").mkdir()
    pin = {"path": "runs/tdmpc2-overshoot-20260929-v1/model.pt", "sha256": "a" * 64}
    source = {"result": {"path": "runs/tdmpc2-overshoot-20260929-v1/result.json",
                         "sha256": "b" * 64}}
    original = {"path": "runs/tdmpc2-raw100k-h5-branches-synthetic.json", "sha256": "c" * 64}
    monkeypatch.setattr(evaluator.prior, "BRANCH_PRIMARY", original)
    files = {}
    for name in ("haic/algorithms/tdmpc2/q_bootstrap.py",
                 "scripts/diagnose_tdmpc2_overshoot_q0_branches.py",
                 "tests/test_diagnose_tdmpc2_overshoot_q0_branches.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic pinned source " + name.encode())
        files[name] = digest(path)
    for ref in (pin, source["result"], original):
        path = root / ref["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic evidence")
        ref["sha256"] = digest(path)
    old_score = {"training_source": {"protocol": evaluator.prior.TRAIN_PROTOCOL,
                                      "result": source["result"], "checkpoint": pin},
                 "original_branch": {"primary": original}, "source_sha256": {}}
    training_path = root / evaluator.prior.TRAIN_PROTOCOL["path"]
    training_path.write_bytes(b"synthetic old protocol reference")
    original_old_digest = evaluator.raw._digest
    monkeypatch.setattr(evaluator.raw, "_digest", lambda path: (
        evaluator.prior.TRAIN_PROTOCOL["sha256"] if path == training_path else original_old_digest(path)))
    runtime = {"torch": "2.1.0+cpu", "torch_cuda": None}
    pairs = [[i, j] for i in range(5) for j in range(i + 1, 5)]
    labels = ["h3_q0", "h5_q0", *(f"h5_q1_{i}_{j}" for i, j in pairs)]
    names = ("logged", "coast", "gas", "brake", "left_gas")
    anchors = []
    for index in range(12):
        real = list(range(5))
        anchors.append({"episode_id": index // 3 + 4, "track_id": 1,
                        "start_step": (16, 50, 100)[index % 3],
                        "geometry_seed": raw.ROADS[index // 3],
                        **{key: "a" * 64 for key in ("road_sha256", "anchor_accessible_state_sha256",
                                                    "anchor_observation_sha256", "anchor_model_observation_sha256")},
                        "candidate_action_bytes_hex": {name: "00" for name in names},
                        "candidates": [{"candidate": name, "steps": 5, "full_h5": True,
                                        "ranking_eligible": True, "model_action_bytes_hex": "00",
                                        "raw_rewards": [float(value), 0., 0., 0., 0.],
                                        "real_discounted_raw_return": float(value)}
                                       for name, value in zip(names, real)],
                        "scores": {label: {name: float(value if label in labels[:2] else 4 - value)
                                           for name, value in zip(names, real)} for label in labels}})
    totals = {}
    for label in labels:
        is_q0 = label in labels[:2]
        totals[label] = {"pairs": {"concordant": 120 if is_q0 else 0,
                                    "discordant": 0 if is_q0 else 120,
                                    "predicted_tie": 0, "real_tie": 0},
                         "informative_real_pairs": 120, "concordance": 1. if is_q0 else 0.,
                         "tied_best": 12 if is_q0 else 0,
                         "tied_best_roads": 4 if is_q0 else 0,
                         "mean_real_h5_regret": 0. if is_q0 else 4.}
    gate = {"full_h5_suffixes": 60, "informative_real_pairs": 120,
            "informative_roads": 4, "scores": totals, "quality_gate_passed": True}
    protocol = {"format": evaluator.BRANCH_FORMAT,
                "purpose": "consumed-TRAIN-disjoint-RAW-replay-real-five-action-Q0-screen",
                "output": "runs/tdmpc2-overshoot-q0-branches-synthetic.json",
                "score_source": {"path": "experiments/tdmpc2-overshoot-old-branch-score-v1.json",
                                 "sha256": "44edd04cc7a579c3867ed35f5763645d0c92b2971fd979d5d141ea87f5802aa8"},
                "old_primary": original, "source_sha256": files,
                "runtime": {"torch": "2.1.0+cpu"},
                "max_resets": 72, "policy_release": False,
                "anchors": [{"episode_id": ep, "start_step": step}
                            for ep in range(4, 8) for step in (16, 50, 100)],
                "episode_lengths": [319, 314, 293, 303], "q_weights": {"q0": 0, "q1": 1},
                "horizons": [3, 5], "fixed_q_pairs": pairs, "discount": .995,
                "tie_tolerance": 1e-6, "bootstrap_offset": 100000,
                "environment": {"track_id": 1, "geometry_seeds": list(raw.ROADS),
                                "max_steps": 2000, "frame_skip": 4, "obstacles": True,
                                "reward_shaping": False},
                "quality_gate": {"full_h5_suffixes": 60, "min_informative_pairs": 40,
                                 "min_informative_roads": 3, "min_q0_concordance": .7,
                                 "min_q0_tied_best": 8, "min_q0_tied_best_roads": 2}}
    refs = {"protocol": {"path": "experiments/tdmpc2-overshoot-q0-branches-v1.json", "sha256": "d" * 64},
            "receipt": {"path": protocol["output"], "sha256": "e" * 64}}
    receipt = {"format": "haic-tdmpc2-overshoot-q0-real-branches-result-v1", "status": "PASS",
               "protocol": refs["protocol"], "score_source": protocol["score_source"],
               "old_primary": original, "source_sha256": files,
               "runtime": {"torch": "2.1.0+cpu"}, "environment_resets_attempted": 72,
               "reset_intents": [{"episode_id": ep, "start_step": step, "candidate": candidate,
                                  "ordinal": ordinal}
                                 for ordinal, (ep, step, candidate) in enumerate((
                                     (ep, step, candidate) for ep in range(4, 8)
                                     for step in (16, 50, 100)
                                     for candidate in ("prefix_capture", *names)), 1)],
               "optimizer_updates": 0,
               "policy_release": False, "replay_bound_before_first_reset": True,
               "anchors": anchors, "summary": gate}
    receipt["body_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in receipt.items() if key != "body_sha256"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    data = {refs["protocol"]["path"]: protocol, refs["receipt"]["path"]: receipt,
            protocol["score_source"]["path"]: old_score, original["path"]: {"status": "complete"}}
    monkeypatch.setattr(evaluator, "_ref", lambda _root, ref, _prefix: (data[ref["path"]], root / ref["path"]))
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: pytest.fail("branch verifier loaded model"))
    accepted = evaluator._branch(root, refs, source, pin, runtime)
    assert accepted["quality_gate_passed"] and accepted["informative_real_pairs"] == 120
    receipt["status"] = "FAIL"
    with pytest.raises(ValueError, match="FAIL"):
        evaluator._branch(root, refs, source, pin, runtime)
    receipt["status"] = "PASS"
    receipt["anchors"][0]["candidates"][0]["raw_rewards"][0] = 20.
    with pytest.raises(ValueError, match="body SHA"):
        evaluator._branch(root, refs, source, pin, runtime)
    receipt["anchors"][0]["candidates"][0]["raw_rewards"][0] = 0.
    receipt["summary"]["scores"]["h5_q0"]["pairs"]["concordant"] = 119
    receipt["body_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in receipt.items() if key != "body_sha256"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError, match="predeclared Q0 gates"):
        evaluator._branch(root, refs, source, pin, runtime)
    receipt["summary"]["scores"]["h5_q0"]["pairs"]["concordant"] = 120
    receipt["body_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in receipt.items() if key != "body_sha256"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    (root / "haic/algorithms/tdmpc2/q_bootstrap.py").write_bytes(b"drift")
    with pytest.raises(ValueError, match="source drift"):
        evaluator._branch(root, refs, source, pin, runtime)
