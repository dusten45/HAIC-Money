"""DORMANT RAW 100k H5-planner-only, full-episode CPU TRAIN evaluator.

Run with ``python -m scripts.evaluate_tdmpc2_h5_planner --protocol ...
--protocol-sha256 ...``. No H5 quality or same-anchor REAL H5 branch gate has
been frozen. Even a self-declared passing receipt cannot enable --execute:
the producer schemas and independent gate thresholds must first be reviewed,
implemented here, and frozen with a NEW operator/protocol SHA. The default
preflight makes no environment, output, or torch.load call. The private rollout
core is exercised only against synthetic environments until that dependency is
resolved. Never deserialize an untrusted checkpoint (torch.load uses pickle).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import shutil
import time

import torch

from scripts import evaluate_tdmpc2_full_train as raw


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-tdmpc2-h5-planner-full-consumed-train-v1"
RAW_EVALUATOR_SHA = "d2f3df9522257f2dfbffe5167571c5a491b7929acbbbcb3b84ebb0c2f8b91089"
SOURCE_CURSOR = (100354, 100354, 307)
SOURCE = {
    "protocol": {"path": "experiments/tdmpc2-long-reused-train-v2.json",
                 "sha256": "d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec"},
    "result": {"path": "runs/tdmpc2-long-20260928-v2/result.json",
               "sha256": "287470932dbba9a7eeed778a5be7762b736e6679d99fd29078c5103890a75a4b"},
    "checkpoints": [{
        "target": 100000,
        "path": "runs/tdmpc2-long-20260928-v2/checkpoint-at-least-100000-step-100354.pt",
        "sha256": "aa268b95a7398b18474fbbaea153fb5e20cc363345cf3c0e6176aaa9dc72c295",
        "training_cursor_sha256": "84f06ee4dcc8568bd5d384d1811252321ad5dc6a98ce19d18badb448d09123d7",
        "step_cursor_sha256": "a94ce156afcb6e4f706474d6cf2d2ccb4754a868eb75a3eb4b53df5d2be37945",
    }],
}
BASELINE = {
    "training_result": {"path": "experiments/tdmpc2-long-reused-train-v2-100k-result.json",
                        "sha256": "bacf4c7e7a53794ecec7797746388695bac01096287f7aff09b5feafc4e3b515"},
    "full_eval_protocol": {"path": "experiments/tdmpc2-full-consumed-train-v1.json",
                           "sha256": "874d01d1396697efc9e4b119b0cd9ce8189fe36fa8d44773dcc13d30ab9ddbcd"},
    "full_eval_result": {"path": "experiments/tdmpc2-full-consumed-train-v1-result.json",
                         "sha256": "3bc41dd2a35bfd488a6cd13a2778d9344f6929873edd77e701b067e31fd1366b"},
}
PLANNER = {"action_dim": 3, "horizon": 5, "num_samples": 512,
           "num_pi_trajs": 24, "iterations": 6, "num_elites": 64,
           "discount": .995, "episodic": True, "eval_mode": True}
# Local safety caps, NOT official Agent limits. A 16k-decision upper bound at
# five seconds/action cannot fit the wall cap; exceeding it preserves a partial.
BUDGET = {"max_wall_seconds": 21600, "max_action_latency_s": 5.0,
          "max_process_rss_mib": 4096, "min_disk_available_bytes": 268435456}


def _baseline(root: Path) -> None:
    """Bind the independently frozen RAW H3 result, including its primary ledger."""
    summary, _ = raw._reference(root, BASELINE["training_result"], "experiments/")
    protocol, _ = raw._reference(root, BASELINE["full_eval_protocol"], "experiments/")
    result, _ = raw._reference(root, BASELINE["full_eval_result"], "experiments/")
    if (summary.get("run_status") != "completed_boundary_at_least_100k"
            or summary.get("training_protocol_sha256") != SOURCE["protocol"]["sha256"]
            or summary.get("run_result_sha256") != SOURCE["result"]["sha256"]
            or summary.get("checkpoint_sha256") != SOURCE["checkpoints"][0]["sha256"]
            or summary.get("training_ledger_sha256") != SOURCE["checkpoints"][0]["training_cursor_sha256"]
            or summary.get("step_ledger_sha256") != SOURCE["checkpoints"][0]["step_cursor_sha256"]
            or (summary.get("decisions"), summary.get("updates"), summary.get("episodes")) != SOURCE_CURSOR
            or protocol.get("source") != SOURCE or protocol.get("cells") != raw.CELLS
            or protocol.get("environment") != raw.ENVIRONMENT
            or (protocol.get("modes"), protocol.get("repeats"), protocol.get("seed"),
                protocol.get("max_steps")) != (["prior", "mppi"], 2, 20260928, 2000)
            or protocol.get("evaluation_source_sha256") != RAW_EVALUATOR_SHA
            or result.get("status") != "complete_valid_full_episode_finish_comparison"
            or result.get("evaluation_protocol_sha256") != BASELINE["full_eval_protocol"]["sha256"]
            or result.get("evaluation_operator_sha256") != RAW_EVALUATOR_SHA
            or result.get("source_model_sha256") != SOURCE["checkpoints"][0]["sha256"]
            or result.get("source_training_result_sha256") != SOURCE["result"]["sha256"]
            or result.get("denominators") != {
                "distinct_training_roads": 4, "repeats_per_road_per_mode": 2,
                "prior_episodes": 8, "mppi_episodes": 8, "total_episodes": 16,
                "max_decisions_per_episode": 2000}
            or result.get("validity", {}).get("all_scheduled_episodes_complete") is not True
            or result["validity"].get("full_episode_finish_comparison_valid") is not True
            or result["validity"].get("censored_episodes") != 0
            or result["validity"].get("generalization_claim") is not False
            or any(result.get(mode, {}).get("finishes") != 0
                   or result[mode].get("episodes") != 8 for mode in ("prior", "mppi"))):
        raise ValueError("frozen RAW H3 baseline source/evaluation differs")
    primary, _ = raw._reference(root, {"path": result["primary_result"],
                                       "sha256": result["primary_result_sha256"]}, "runs/tdmpc2-full-train-")
    ledger = raw._path(root, result["episode_ledger"])
    if (raw._digest(ledger) != raw._sha(result["episode_ledger_sha256"])
            or primary.get("status") != "complete"
            or primary.get("protocol_sha256") != BASELINE["full_eval_protocol"]["sha256"]
            or primary.get("source_result_sha256") != SOURCE["result"]["sha256"]
            or primary.get("episodes_sha256") != result["episode_ledger_sha256"]
            or primary.get("full_episode_finish_comparison_valid") is not True
            or primary.get("denominators", {}).get("completed_episodes") != 16
            or any(primary.get("per_checkpoint", {}).get("100000", {}).get("per_mode", {}).get(mode, {}).get("episodes") != 8
                   or primary["per_checkpoint"]["100000"]["per_mode"][mode].get("finishes") != 0
                   for mode in ("prior", "mppi"))):
        raise ValueError("frozen RAW H3 baseline primary receipt/ledger differs")


def _source(root: Path) -> dict:
    """Use the SHA-pinned RAW H3 validator without weakening its horizon gate."""
    protocol, result, run = raw._source(root, {"source": SOURCE})
    pins = raw._ledger(root, protocol, result, run, SOURCE)
    pin = pins[0]
    if (len(pins) != 1 or {k: pin[k] for k in SOURCE["checkpoints"][0]} != SOURCE["checkpoints"][0]
            or (pin["decisions"], pin["updates"], pin["episodes"]) != SOURCE_CURSOR
            or result["checkpoints"][-1]["target"] != SOURCE["checkpoints"][0]["target"]
            or result["checkpoints"][-1]["sha256"] != pin["sha256"]):
        raise ValueError("source is not the first completed RAW 100k H3 model")
    return pin


def _gates(root: Path, gates: object) -> str:
    """SHA-check proposed H5 receipts, but never self-certify an unfrozen producer."""
    if not isinstance(gates, dict) or set(gates) != {"quality", "branch"}:
        return "missing_h5_quality_or_real_branch_gate"
    receipts = {}
    for kind in ("quality", "branch"):
        group = gates[kind]
        if not isinstance(group, dict) or set(group) != {"protocol", "result"}:
            return f"missing_h5_{kind}_protocol_or_result"
        documents = {}
        for part in ("protocol", "result"):
            ref = group[part]
            if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
                return f"missing_h5_{kind}_{part}_sha_reference"
            # An H3 branch result or damage-target result must not be rebranded H5.
            prefix = "experiments/tdmpc2-h5-logged-" if kind == "quality" else "experiments/tdmpc2-h5-branch-"
            if not isinstance(ref["path"], str) or not ref["path"].startswith(prefix):
                raise ValueError(f"H5 {kind} {part} is not a separate H5 artifact")
            documents[part], _ = raw._reference(root, ref, "experiments/tdmpc2-h5-")
        spec, result = documents["protocol"], documents["result"]
        if (spec.get("source") != SOURCE or spec.get("baseline") != BASELINE
                or result.get("protocol_sha256") != group["protocol"]["sha256"]
                or result.get("source_checkpoint_sha256") != SOURCE["checkpoints"][0]["sha256"]
                or result.get("source_training_result_sha256") != SOURCE["result"]["sha256"]):
            raise ValueError(f"H5 {kind} result does not bind its RAW100k source/protocol")
        receipts[kind] = documents
    quality = receipts["quality"]
    branch = receipts["branch"]
    if (quality["protocol"].get("horizons") != [3, 5]
            or quality["result"].get("environment_resets") != 0
            or branch["protocol"].get("horizon") != 5
            or branch["protocol"].get("quality_result") != gates["quality"]["result"]
            or branch["result"].get("real_horizon") != 5
            or branch["result"].get("quality_result_sha256") != gates["quality"]["result"]["sha256"]):
        raise ValueError("H5 logged and real-branch gate receipts are not source-linked H5 evidence")
    # The no-reset logged and real same-anchor H5 producers have not frozen
    # their metric thresholds, parity schema or result format. A hash and a
    # self-reported 'passed' flag cannot authorize additional TRAIN resets.
    return "h5_quality_and_real_branch_gate_schemas_not_frozen"


def _check(protocol_path: Path, sha: str, *, root: Path) -> tuple[dict, dict]:
    path = raw._evaluation_path(root, protocol_path)
    if path.parent != root / "experiments" or raw._digest(path) != raw._sha(sha):
        raise ValueError("separate H5 evaluation protocol SHA mismatch")
    p = raw._json(path.read_bytes())
    if (set(p) != {"format", "purpose", "evaluation_source_sha256", "raw_evaluator_sha256",
                   "source", "baseline", "h5_gates", "cells", "environment", "modes",
                   "repeats", "seed", "max_steps", "planner", "budget", "output_dir"}
            or p["format"] != FORMAT or p["purpose"] != "consumed-TRAIN-development"
            or p["source"] != SOURCE or p["baseline"] != BASELINE
            or p["raw_evaluator_sha256"] != RAW_EVALUATOR_SHA
            or p["cells"] != raw.CELLS or p["environment"] != raw.ENVIRONMENT
            or p["modes"] != ["mppi"] or type(p["repeats"]) is not int or p["repeats"] != 2
            or type(p["seed"]) is not int or p["seed"] != 20260928
            or type(p["max_steps"]) is not int or p["max_steps"] != 2000
            or p["planner"] != PLANNER or any(type(p["planner"].get(k)) is not type(v)
                                             for k, v in PLANNER.items())
            or p["budget"] != BUDGET or any(type(p["budget"].get(k)) is not type(v)
                                           for k, v in BUDGET.items())):
        raise ValueError("H5 planner-only RAW full-episode protocol differs")
    if (raw._digest(raw._path(root, "scripts/evaluate_tdmpc2_h5_planner.py"))
            != raw._sha(p["evaluation_source_sha256"])
            or raw._digest(raw._path(root, "scripts/evaluate_tdmpc2_full_train.py")) != RAW_EVALUATOR_SHA):
        raise ValueError("H5 or trusted RAW evaluator executable SHA mismatch")
    output = raw._path(root, p["output_dir"], existing=False)
    if (output.parent != root / "runs" or not output.name.startswith("tdmpc2-h5-planner-full-train-")
            or output.exists() or not output.parent.is_dir()):
        raise ValueError("H5 output requires a new exclusive runs/ directory")
    if torch.version.cuda is not None or not str(torch.__version__).startswith("2.1.0+cpu"):
        raise ValueError("H5 evaluation requires isolated CPU-only Torch 2.1.0")
    if shutil.disk_usage(output.parent).free < BUDGET["min_disk_available_bytes"]:
        raise ValueError("insufficient disk for durable H5 receipts")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > BUDGET["max_process_rss_mib"] * 1024:
        raise ValueError("CPU process already exceeds H5 RSS budget")
    _baseline(root)
    pin = _source(root)
    reason = _gates(root, p["h5_gates"])
    return p, {"status": "blocked", "reason": reason, "environment_resets": 0,
               "torch_load_calls": 0, "protocol_sha256": sha,
               "source_checkpoint_sha256": pin["sha256"], "source_decisions": pin["decisions"],
               "baseline_mppi_finishes": 0, "baseline_mppi_episodes": 8,
               "modes": ["mppi"], "planned_episodes": 8, "output_dir": p["output_dir"],
               "generalization_claim": False, "official_score": False}


def preflight(protocol_path: Path, protocol_sha256: str, *, root: Path = ROOT) -> dict:
    """File-only, fail-closed preflight: absent/unfrozen H5 gates never pass."""
    return _check(protocol_path, protocol_sha256, root=root.resolve(strict=True))[1]


def _planner(model, *, factory=None):
    from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner

    config = PlannerConfig(action_dim=3, discount=.995, horizon=5, episodic=True,
                           num_samples=512, num_pi_trajs=24, iterations=6, num_elites=64)
    planner = (TDMPC2Planner if factory is None else factory)(model, config)
    if getattr(planner, "config", None) != config:
        raise ValueError("H5 planner config changed or silently fell back to H3")
    return planner


def _summary(rows: list[dict], pin: dict) -> dict:
    schedule = [(repeat, cell) for repeat in range(2) for cell in raw.CELLS]
    if (len(rows) != 8 or any(row.get("event") != "episode" or row.get("target") != 100000
            or row.get("mode") != "mppi" or row.get("repeat") != repeat
            or row.get("episode_seed") != 20260928 + index
            or any(row.get(key) != value for key, value in cell.items())
            or row.get("max_steps") != 2000 or row.get("decisions", 0) not in range(1, 2001)
            or "training_return" in row or "training_reward" in row
            for index, (row, (repeat, cell)) in enumerate(zip(rows, schedule)))):
        raise ValueError("incomplete or altered eight-episode RAW H5 MPPI denominator")
    roads = []
    for cell in raw.CELLS:
        cohort = [r for r in rows if r["geometry_seed"] == cell["geometry_seed"]]
        decisions = sum(r["decisions"] for r in cohort)
        roads.append({**cell, "episodes": 2, "finishes": sum(r["finished"] for r in cohort),
                      "censored": sum(r["censored"] for r in cohort),
                      "uncensored": sum(not r["censored"] for r in cohort),
                      "decisions": decisions,
                      "mean_progress": sum(r["progress"] for r in cohort) / 2,
                      "mean_raw_return": sum(r["raw_return"] for r in cohort) / 2,
                      "mean_damage": sum(r["damage"] for r in cohort) / 2,
                      "mean_action_latency_s": sum(r["action_latency_total_s"] for r in cohort) / decisions,
                      "max_action_latency_s": max(r["action_latency_max_s"] for r in cohort)})
    decisions = sum(r["decisions"] for r in rows)
    return {"checkpoint_sha256": pin["sha256"], "mode": "mppi",
            "episodes": 8, "finishes": sum(r["finished"] for r in rows),
            "censored": sum(r["censored"] for r in rows),
            "uncensored": sum(not r["censored"] for r in rows), "decisions": decisions,
            "mean_progress": sum(r["progress"] for r in rows) / 8,
            "mean_raw_return": sum(r["raw_return"] for r in rows) / 8,
            "mean_damage": sum(r["damage"] for r in rows) / 8,
            "mean_action_latency_s": sum(r["action_latency_total_s"] for r in rows) / decisions,
            "max_action_latency_s": max(r["action_latency_max_s"] for r in rows), "roads": roads}


def _run_core(root: Path, p: dict, protocol_path: Path, protocol_sha256: str,
              pin: dict, model, planner, *, env_factory, recheck,
              clock=time.perf_counter) -> dict:
    """Synthetic-tested rollout core; the CLI cannot call it until both gates exist.

    ``recheck`` must independently verify source AND H5 gates before each reset;
    the current public preflight always blocks, so no real caller exists yet.
    """
    if not callable(recheck) or not callable(env_factory):
        raise ValueError("H5 source/gate recheck and environment factory are required")
    path = raw._evaluation_path(root, protocol_path)
    if (path.parent != root / "experiments" or raw._digest(path) != raw._sha(protocol_sha256)
            or raw._json(path.read_bytes()) != p):
        raise ValueError("H5 core protocol SHA changed before output creation")
    from haic.algorithms.tdmpc2.planner import PlannerConfig

    expected = PlannerConfig(action_dim=3, discount=.995, horizon=5, episodic=True,
                             num_samples=512, num_pi_trajs=24, iterations=6, num_elites=64)
    if (getattr(planner, "config", None) != expected or p.get("planner") != PLANNER
            or p.get("budget") != BUDGET or p.get("modes") != ["mppi"]
            or p.get("max_steps") != 2000 or pin.get("sha256") != SOURCE["checkpoints"][0]["sha256"]):
        raise ValueError("H5 core requires the exact RAW model and unchanged MPPI settings")
    output = raw._path(root, p["output_dir"], existing=False)
    if output.parent != root / "runs" or not output.name.startswith("tdmpc2-h5-planner-full-train-"):
        raise ValueError("invalid exclusive H5 output path")
    output.mkdir(mode=0o700, exist_ok=False)
    ledger = output / "episodes.jsonl"
    started = clock()
    rows: list[dict] = []
    env = None
    reset_intents = 0

    def budget() -> float:
        now = clock()
        if not math.isfinite(now) or now - started > BUDGET["max_wall_seconds"]:
            raise TimeoutError("H5 CPU wall budget exceeded")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > BUDGET["max_process_rss_mib"] * 1024:
            raise MemoryError("H5 CPU RSS budget exceeded")
        if shutil.disk_usage(output).free < BUDGET["min_disk_available_bytes"]:
            raise OSError("H5 receipt disk budget exceeded")
        return now

    action_started = None

    def action_clock() -> float:
        nonlocal action_started
        now = budget()
        if action_started is None:
            action_started = now
        else:
            elapsed = now - action_started
            action_started = None
            if elapsed < 0 or elapsed > BUDGET["max_action_latency_s"]:
                raise TimeoutError("H5 CPU action latency budget exceeded")
        return now

    try:
        for repeat in range(2):
            for index, cell in enumerate(raw.CELLS):
                budget()
                if recheck() is not True:
                    raise ValueError("H5 quality/branch/source gate changed before reset")
                seed = 20260928 + repeat * 4 + index
                reset_intents += 1
                raw._journal(ledger, {"event": "reset_intent", "target": 100000,
                                      "mode": "mppi", "repeat": repeat, "episode_seed": seed, **cell})
                if env is None:
                    env = env_factory(2000)
                if recheck() is not True:
                    raise ValueError("H5 quality/branch/source gate changed immediately before reset")
                row = raw._episode(env, model, planner, cell, mode="mppi", repeat=repeat,
                                   seed=seed, checkpoint_target=100000, max_steps=2000,
                                   clock=action_clock)
                raw._journal(ledger, row)
                rows.append(row)
        budget()
        outcome = _summary(rows, pin)
        report = {"status": "complete", "scope": "reused_consumed_TRAIN_development_only",
                  "reused_train_only": True, "generalization_claim": False, "official_score": False,
                  "protocol_sha256": protocol_sha256,
                  "evaluation_source_sha256": p["evaluation_source_sha256"],
                  "source_training_result_sha256": SOURCE["result"]["sha256"],
                  "h5_gates": p["h5_gates"],
                  "source_model_sha256": pin["sha256"], "source_model_trained_horizon": 3,
                  "planner_horizon": 5, "planner": PLANNER, "budget": BUDGET,
                  "evaluation_reward": "raw_environment_only", "checkpoint_target": 100000,
                  "baseline_full_eval_result_sha256": BASELINE["full_eval_result"]["sha256"],
                  "baseline_h3_mppi": {"episodes": 8, "finishes": 0},
                  "pairing": "same four consumed TRAIN road/reset seeds, not actions or trajectories",
                  "denominators": {"distinct_training_roads": 4, "repeats_per_road": 2,
                                   "planned_episodes": 8, "completed_episodes": len(rows), "max_steps": 2000},
                  "episodes_sha256": raw._digest(ledger), "primary_mppi": outcome,
                  "primary_local_target": {"finishes_required": 4, "episodes": 8,
                                           "observed_finishes": outcome["finishes"],
                                           "met_on_reused_train": outcome["finishes"] >= 4,
                                           "generalization_claim": False}}
        with (output / "result.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return report
    except BaseException as exc:
        try:
            raw._journal(ledger, {"event": "partial", "reason": type(exc).__name__,
                                  "complete_episodes": len(rows), "reset_intents": reset_intents,
                                  "environment_resets": None if reset_intents else 0,
                                  "resume_supported": False})
        except BaseException:
            try:
                with (output / "failure.json").open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps({"event": "failure", "reason": type(exc).__name__,
                                             "protocol_sha256": protocol_sha256,
                                             "episode_ledger_sha256": raw._digest(ledger) if ledger.is_file() else None,
                                             "complete_episodes": len(rows), "reset_intents": reset_intents,
                                             "environment_resets": None if reset_intents else 0,
                                             "resume_supported": False}, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException as preservation_error:
                raise RuntimeError("cannot preserve H5 exposure receipt; exposure unknown") from preservation_error
        raise
    finally:
        try:
            if env is not None:
                env.close()  # type: ignore[union-attr]
        finally:
            fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


def execute(protocol_path: Path, protocol_sha256: str, *, root: Path = ROOT) -> dict:
    """Deliberate hard gate: no producer schema, hence no checkpoint load/reset."""
    checked = preflight(protocol_path, protocol_sha256, root=root)
    raise ValueError(f"H5 evaluation DORMANT: {checked['reason']}; environment_resets=0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path, help="externally frozen experiments/*.json")
    parser.add_argument("--protocol-sha256", required=True, help="external evaluation protocol SHA-256")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="default: no-reset dormant preflight")
    mode.add_argument("--execute", action="store_true", help="blocked until both H5 gates are implemented")
    args = parser.parse_args()
    report = (execute(args.protocol, args.protocol_sha256) if args.execute else
              preflight(args.protocol, args.protocol_sha256))
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
