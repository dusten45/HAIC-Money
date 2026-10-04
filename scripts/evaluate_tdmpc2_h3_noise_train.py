"""Frozen RAW100k H3 MPPI with training-only final action noise on consumed TRAIN.

Run from the repository root with ``python -m scripts.evaluate_tdmpc2_h3_noise_train``.
The default preflight cannot construct/reset an environment or deserialize a model.
Execution requires a separately frozen SHA-bound protocol and CPU-only Torch 2.1.
These eight reused-road episodes cannot establish fresh-road or official performance.
Only load the trusted, fully SHA-checked RAW checkpoint: torch.load uses pickle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import time

import numpy as np
import torch

from scripts import evaluate_tdmpc2_full_train as raw


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-tdmpc2-h3-noise-full-consumed-train-v1"
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
PLANNER = {"action_dim": 3, "horizon": 3, "num_samples": 512,
           "num_pi_trajs": 24, "iterations": 6, "num_elites": 64,
           "discount": .995, "episodic": True, "eval_mode": False}
RESOURCE_KEYS = {"measured_peak_rss_bytes", "additional_memory_bytes", "memory_reserve_bytes",
                 "remaining_disk_bytes", "disk_reserve_bytes", "max_wall_seconds"}


def _source(root: Path) -> tuple[dict, dict]:
    protocol, result, run = raw._source(root, {"source": SOURCE})
    pins = raw._ledger(root, protocol, result, run, SOURCE)
    pin = pins[0]
    if (len(pins) != 1 or {key: pin[key] for key in SOURCE["checkpoints"][0]} != SOURCE["checkpoints"][0]
            or (pin["decisions"], pin["updates"], pin["episodes"]) != SOURCE_CURSOR
            or result["checkpoints"][-1]["target"] != SOURCE["checkpoints"][0]["target"]
            or result["checkpoints"][-1]["sha256"] != pin["sha256"]):
        raise ValueError("not the completed first RAW100k H3 checkpoint and full TRAIN cursor")
    return protocol, pin


def _baseline(root: Path, pin: dict) -> None:
    """Check the historical H3 MPPI0/8 at both primary-result and episode level."""
    summary, _ = raw._reference(root, BASELINE["training_result"], "experiments/")
    protocol, _ = raw._reference(root, BASELINE["full_eval_protocol"], "experiments/")
    result, _ = raw._reference(root, BASELINE["full_eval_result"], "experiments/")
    if (raw._digest(raw._path(root, "scripts/evaluate_tdmpc2_full_train.py")) != RAW_EVALUATOR_SHA
            or summary.get("run_status") != "completed_boundary_at_least_100k"
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
            or any(result.get(mode, {}).get("finishes") != 0 or result[mode].get("episodes") != 8
                   for mode in ("prior", "mppi"))):
        raise ValueError("historical RAW H3 baseline evidence differs")
    primary, _ = raw._reference(root, {"path": result["primary_result"],
                                       "sha256": result["primary_result_sha256"]}, "runs/tdmpc2-full-train-")
    ledger = raw._path(root, result["episode_ledger"])
    if (result["episode_ledger"] != f"{protocol['output_dir']}/episodes.jsonl"
            or result["primary_result"] != f"{protocol['output_dir']}/result.json"
            or raw._digest(ledger) != raw._sha(result["episode_ledger_sha256"])
            or primary.get("status") != "complete" or primary.get("protocol_sha256") != BASELINE["full_eval_protocol"]["sha256"]
            or primary.get("source_result_sha256") != SOURCE["result"]["sha256"]
            or primary.get("episodes_sha256") != result["episode_ledger_sha256"]
            or primary.get("checkpoints") != [pin]
            or primary.get("full_episode_finish_comparison_valid") is not True
            or primary.get("denominators") != {
                "checkpoints": 1, "completed_episodes": 16, "distinct_roads": 4,
                "modes": ["prior", "mppi"], "planned_episodes": 16,
                "repeats_per_road_mode_checkpoint": 2}):
        raise ValueError("historical H3 primary result or episode ledger changed")
    lines = ledger.read_bytes().splitlines(keepends=True)
    if len(lines) != 32 or any(not line.endswith(b"\n") for line in lines):
        raise ValueError("historical H3 episode ledger is partial")
    rows = []
    for number, (repeat, cell, mode) in enumerate((repeat, cell, mode)
                                                for repeat in range(2) for cell in raw.CELLS
                                                for mode in ("prior", "mppi")):
        expected = {"target": SOURCE["checkpoints"][0]["target"], "mode": mode, "repeat": repeat,
                    "episode_seed": 20260928 + repeat * 4 + raw.CELLS.index(cell), **cell}
        intent, row = raw._json(lines[2 * number]), raw._json(lines[2 * number + 1])
        if intent != {"event": "reset_intent", **expected} or any(row.get(k) != v for k, v in expected.items()):
            raise ValueError("historical H3 reset schedule changed")
        if (row.get("event") != "episode" or row.get("max_steps") != 2000
                or type(row.get("decisions")) is not int or not 1 <= row["decisions"] <= 2000
                or any(type(row.get(k)) is not bool for k in ("finished", "terminated", "truncated", "terminal", "censored"))
                or not (row["terminated"] or row["truncated"])
                or row["terminal"] != (row["terminated"] or row["finished"])
                or row["censored"] != (row["truncated"] and not row["finished"] and not row["terminated"])
                or row["finished"] and not row["truncated"]
                or row["censored"] and row["decisions"] != 2000
                or any(type(row.get(k)) not in (float, int) or not math.isfinite(row[k])
                       for k in ("raw_return", "progress", "damage", "action_latency_total_s", "action_latency_max_s"))
                or row["action_latency_total_s"] < 0 or row["action_latency_max_s"] < 0
                or any(raw._sha(row.get(k)) != row[k] for k in ("action_trace_sha256", "native_action_trace_sha256"))):
            raise ValueError("historical H3 primary episode is incomplete or inconsistent")
        rows.append(row)
    target = SOURCE["checkpoints"][0]["target"]
    calculated = raw._summary(rows, [{"target": target, "sha256": SOURCE["checkpoints"][0]["sha256"]}], 2)
    if (calculated != primary.get("per_checkpoint")
            or any(primary["per_checkpoint"][str(target)]["per_mode"][mode]["finishes"] != 0
                   or primary["per_checkpoint"][str(target)]["per_mode"][mode]["censored"] != 0
                   for mode in ("prior", "mppi"))):
        raise ValueError("historical H3 0/8 primary ledger does not match result")


def _headroom() -> int:
    """Use the minimum actual remaining memory of host and finite cgroup ancestors."""
    lines = Path("/proc/meminfo").read_text().splitlines()
    available = next((int(line.split()[1]) * 1024 for line in lines if line.startswith("MemAvailable:")), None)
    if available is None:
        raise ValueError("host memory telemetry unavailable")
    group = next((line.split("::", 1)[1].strip("/") for line in Path("/proc/self/cgroup").read_text().splitlines()
                  if line.startswith("0::")), None)
    if group is None:
        raise ValueError("cgroup v2 memory telemetry unavailable")
    root = Path("/sys/fs/cgroup")
    path = root / group
    while True:
        maximum = (path / "memory.max").read_text().strip()
        current = int((path / "memory.current").read_text().strip())
        if maximum != "max":
            available = min(available, int(maximum) - current)
        if path == root:
            break
        path = path.parent
    return available


def _resources(output: Path, forecast: dict, *, allocated_bytes: int = 0) -> dict:
    memory = _headroom()
    disk = shutil.disk_usage(output).free
    needed_memory = max(0, forecast["additional_memory_bytes"] - allocated_bytes) + forecast["memory_reserve_bytes"]
    needed_disk = forecast["remaining_disk_bytes"] + forecast["disk_reserve_bytes"]
    if memory < needed_memory or disk < needed_disk:
        raise ValueError("measured memory/disk headroom below declared incremental forecast")
    return {"host_and_cgroup_raw_headroom_bytes": memory, "output_disk_free_bytes": disk,
            "required_memory_bytes": needed_memory, "required_disk_bytes": needed_disk}


def _check(protocol_path: Path, sha: str, *, root: Path, reserved: bool = False,
           allocated_bytes: int = 0) -> tuple[dict, dict]:
    path = raw._evaluation_path(root, protocol_path)
    if path.parent != root / "experiments" or raw._digest(path) != raw._sha(sha):
        raise ValueError("separately frozen evaluation protocol SHA mismatch")
    p = raw._json(path.read_bytes())
    if (set(p) != {"format", "purpose", "evaluation_source_sha256", "raw_evaluator_sha256", "source",
                   "baseline", "cells", "environment", "modes", "repeats", "seed", "max_steps",
                   "planner", "resources", "output_dir"}
            or p["format"] != FORMAT or p["purpose"] != "consumed-TRAIN-development"
            or p["source"] != SOURCE or p["baseline"] != BASELINE
            or p["raw_evaluator_sha256"] != RAW_EVALUATOR_SHA
            or p["cells"] != raw.CELLS or p["environment"] != raw.ENVIRONMENT
            or p["modes"] != ["mppi"] or type(p["repeats"]) is not int or p["repeats"] != 2
            or type(p["seed"]) is not int or p["seed"] != 20260928
            or type(p["max_steps"]) is not int or p["max_steps"] != 2000
            or p["planner"] != PLANNER or any(type(p["planner"].get(k)) is not type(v) for k, v in PLANNER.items())):
        raise ValueError("not the isolated frozen RAW100k H3 training-noise evaluation")
    forecast = p["resources"]
    if (not isinstance(forecast, dict) or set(forecast) != RESOURCE_KEYS
            or any(type(value) is not int or value < (1 if key in (
                "measured_peak_rss_bytes", "additional_memory_bytes", "remaining_disk_bytes", "max_wall_seconds") else 0)
                   for key, value in forecast.items())
            or forecast["max_wall_seconds"] > 172800):
        raise ValueError("resources need measured RSS and explicit incremental memory/disk/wall forecasts")
    if (raw._digest(raw._path(root, "scripts/evaluate_tdmpc2_h3_noise_train.py")) != raw._sha(p["evaluation_source_sha256"])
            or raw._digest(raw._path(root, "scripts/evaluate_tdmpc2_full_train.py")) != RAW_EVALUATOR_SHA):
        raise ValueError("H3 noise or RAW evaluator source SHA mismatch")
    output = raw._path(root, p["output_dir"], existing=False)
    if (output.parent != root / "runs" or not output.name.startswith("tdmpc2-h3-noise-full-train-")
            or not output.parent.is_dir() or (not output.is_dir() if reserved else output.exists())):
        raise ValueError("output requires a new exclusive runs/tdmpc2-h3-noise-full-train-* directory")
    if reserved and (output / "result.json").exists():
        raise ValueError("completed output cannot be resumed")
    if torch.version.cuda is not None or str(torch.__version__) != "2.1.0+cpu":
        raise ValueError("H3 noise evaluation requires isolated Torch 2.1.0+cpu")
    resources = _resources(output if reserved else output.parent, forecast, allocated_bytes=allocated_bytes)
    _, pin = _source(root)
    _baseline(root, pin)
    return p, {"status": "preflight_only", "environment_resets": 0, "torch_load_calls": 0,
               "reused_train_only": True, "generalization_claim": False, "official_score": False,
               "protocol_sha256": sha, "evaluation_source_sha256": p["evaluation_source_sha256"],
               "baseline_full_eval_result_sha256": BASELINE["full_eval_result"]["sha256"],
               "source_protocol_sha256": SOURCE["protocol"]["sha256"],
               "source_result_sha256": SOURCE["result"]["sha256"], "checkpoint": pin,
               "cells": raw.CELLS, "planner": PLANNER, "planned_episodes": 8, "max_steps": 2000,
               "output_dir": p["output_dir"], "resources": resources}


def preflight(protocol_path: Path, protocol_sha256: str, *, root: Path = ROOT) -> dict:
    """Read-only, source-bound, no environment import, model loading or reset."""
    return _check(protocol_path, protocol_sha256, root=root.resolve(strict=True))[1]


def _episode(env, model, planner, cell: dict, *, repeat: int, seed: int, budget,
             checkpoint_target: int = 100000,
             clock=time.perf_counter) -> dict:
    """RAW evaluator's one-reset action/metric semantics, changing only MPPI eval_mode."""
    from haic.algorithms.tdmpc2.haic_env import environment_action, episode_boundary, model_observation

    raw._seed(seed)
    planner.reset()
    obs, _ = env.reset(seed=cell["geometry_seed"], options={"track_id": cell["track_id"]})
    if (getattr(env.unwrapped, "track_id", None) != cell["track_id"]
            or getattr(env.unwrapped, "track_seed", None) != cell["geometry_seed"]):
        raise ValueError("environment reset did not honor consumed TRAIN road")
    pixels = model_observation(obs)
    total = latency_sum = latency_max = first_latency = 0.0
    actions, native_actions = hashlib.sha256(), hashlib.sha256()
    for decision in range(1, 2001):
        budget()
        started = clock()
        with torch.inference_mode():
            tensor = torch.as_tensor(pixels[None], device="cpu")
            action = planner.plan(tensor, t0=(decision == 1), eval_mode=False)
            action = action.detach().cpu().numpy().astype(np.float32)
        latency = clock() - started
        if not math.isfinite(latency) or not 0 <= latency <= 5.0:
            raise TimeoutError("CPU action latency exceeded 5 seconds or is invalid")
        native = environment_action(action)
        latency_sum += latency
        latency_max = max(latency_max, latency)
        actions.update(action.tobytes())
        native_actions.update(native.tobytes())
        obs, reward, terminated, truncated, info = env.step(native)
        if (getattr(env.unwrapped, "track_id", None) != cell["track_id"]
                or getattr(env.unwrapped, "track_seed", None) != cell["geometry_seed"]):
            raise ValueError("environment road changed during episode")
        if (type(terminated) is not bool or type(truncated) is not bool or not isinstance(info, dict)
                or type(info.get("finished")) is not bool or type(reward) not in (int, float)
                or not math.isfinite(reward)
                or any(type(info.get(k)) not in (int, float) or not math.isfinite(info[k])
                       for k in ("progress", "damage"))):
            raise ValueError("invalid environment flags or raw metrics")
        total += reward
        if not math.isfinite(total):
            raise ValueError("nonfinite raw episode return")
        done, terminal = episode_boundary(terminated, truncated, info)
        if decision == 1:
            first_latency = latency
        if done:
            finished = info["finished"]
            if (finished and not truncated or truncated and not finished and not terminated and decision != 2000):
                raise ValueError("finish/truncation conflicts with full-episode semantics")
            if (info.get("finish_time_s") is not None
                    and (type(info["finish_time_s"]) not in (int, float) or not math.isfinite(info["finish_time_s"]))):
                raise ValueError("nonfinite finish time")
            return {"event": "episode", "target": checkpoint_target, "mode": "mppi", "repeat": repeat,
                    "episode_seed": seed, **cell, "decisions": decision, "max_steps": 2000,
                    "raw_return": total, "progress": float(info["progress"]), "damage": float(info["damage"]),
                    "finished": finished, "terminated": terminated, "truncated": truncated,
                    "terminal": terminal, "censored": bool(truncated and not finished and not terminated),
                    "finish_time_s": info.get("finish_time_s"),
                    "action_trace_sha256": actions.hexdigest(),
                    "native_action_trace_sha256": native_actions.hexdigest(),
                    "action_latency_first_s": first_latency,
                    "action_latency_mean_s": latency_sum / decision,
                    "action_latency_max_s": latency_max, "action_latency_total_s": latency_sum}
        pixels = model_observation(obs)
    raise ValueError("environment did not end its full 2000-decision episode")


def _summary(rows: list[dict], pin: dict) -> dict:
    schedule = [(repeat, cell) for repeat in range(2) for cell in raw.CELLS]
    if (len(rows) != 8 or any(row.get("event") != "episode" or row.get("target") != pin["target"]
            or row.get("mode") != "mppi" or row.get("repeat") != repeat
            or row.get("episode_seed") != 20260928 + index
            or any(row.get(k) != v for k, v in cell.items()) or row.get("max_steps") != 2000
            or type(row.get("decisions")) is not int or not 1 <= row["decisions"] <= 2000
            or "training_return" in row or "training_reward" in row
            for index, (row, (repeat, cell)) in enumerate(zip(rows, schedule)))):
        raise ValueError("incomplete or altered eight-episode RAW H3 noise denominator")
    roads = []
    for cell in raw.CELLS:
        cohort = [r for r in rows if r["geometry_seed"] == cell["geometry_seed"]]
        decisions = sum(r["decisions"] for r in cohort)
        roads.append({**cell, "episodes": 2, "finishes": sum(r["finished"] for r in cohort),
                      "censored": sum(r["censored"] for r in cohort),
                      "uncensored": sum(not r["censored"] for r in cohort), "decisions": decisions,
                      "mean_progress": sum(r["progress"] for r in cohort) / 2,
                      "mean_raw_return": sum(r["raw_return"] for r in cohort) / 2,
                      "mean_damage": sum(r["damage"] for r in cohort) / 2,
                      "mean_action_latency_s": sum(r["action_latency_total_s"] for r in cohort) / decisions,
                      "max_action_latency_s": max(r["action_latency_max_s"] for r in cohort)})
    decisions = sum(r["decisions"] for r in rows)
    return {"checkpoint_sha256": pin["sha256"], "mode": "mppi", "episodes": 8,
            "finishes": sum(r["finished"] for r in rows), "censored": sum(r["censored"] for r in rows),
            "uncensored": sum(not r["censored"] for r in rows), "decisions": decisions,
            "mean_progress": sum(r["progress"] for r in rows) / 8,
            "mean_raw_return": sum(r["raw_return"] for r in rows) / 8,
            "mean_damage": sum(r["damage"] for r in rows) / 8,
            "mean_action_latency_s": sum(r["action_latency_total_s"] for r in rows) / decisions,
            "max_action_latency_s": max(r["action_latency_max_s"] for r in rows), "roads": roads}


def _write_json(path: Path, record: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _run_core(root: Path, p: dict, protocol_path: Path, sha: str, pin: dict, model, planner,
              *, env_factory, clock=time.perf_counter, initial_peak_rss_bytes: int | None = None) -> dict:
    """Private synthetic-test seam; public execute has no model/env injection."""
    from haic.algorithms.tdmpc2.planner import PlannerConfig

    config = PlannerConfig(action_dim=3, discount=.995, horizon=3, episodic=True,
                           num_samples=512, num_pi_trajs=24, iterations=6, num_elites=64)
    if (getattr(planner, "config", None) != config or p.get("planner") != PLANNER
            or pin.get("sha256") != SOURCE["checkpoints"][0]["sha256"]):
        raise ValueError("private core requires unchanged RAW100k H3 MPPI")
    output = raw._path(root, p["output_dir"], existing=False)
    if output.parent != root / "runs" or not output.name.startswith("tdmpc2-h3-noise-full-train-"):
        raise ValueError("invalid exclusive output path")
    output.mkdir(mode=0o700, exist_ok=False)
    ledger = output / "episodes.jsonl"
    rows = []
    resets = 0
    env = None
    started = clock()
    peak_at_start = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                     if initial_peak_rss_bytes is None else initial_peak_rss_bytes)

    def allocated() -> int:
        return max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - peak_at_start)

    def budget() -> None:
        if clock() - started > p["resources"]["max_wall_seconds"]:
            raise TimeoutError("declared H3 noise wall-time budget exceeded")
        _resources(output, p["resources"], allocated_bytes=allocated())

    try:
        for repeat in range(2):
            for index, cell in enumerate(raw.CELLS):
                budget()
                _, checked = _check(protocol_path, sha, root=root, reserved=True, allocated_bytes=allocated())
                if checked["checkpoint"] != pin or p != raw._json(raw._evaluation_path(root, protocol_path).read_bytes()):
                    raise ValueError("H3 noise source changed before reset")
                seed = 20260928 + repeat * 4 + index
                resets += 1
                raw._journal(ledger, {"event": "reset_intent", "target": pin["target"], "mode": "mppi",
                                      "repeat": repeat, "episode_seed": seed, **cell})
                if env is None:
                    env = env_factory(2000)
                budget()
                _, checked = _check(protocol_path, sha, root=root, reserved=True, allocated_bytes=allocated())
                if checked["checkpoint"] != pin:
                    raise ValueError("H3 noise source changed immediately before reset")
                row = _episode(env, model, planner, cell, repeat=repeat, seed=seed, budget=budget,
                               checkpoint_target=pin["target"], clock=clock)
                raw._journal(ledger, row)
                rows.append(row)
        budget()
        if _check(protocol_path, sha, root=root, reserved=True,
                  allocated_bytes=allocated())[1]["checkpoint"] != pin:
            raise ValueError("H3 noise source changed before final result")
        result = _summary(rows, pin)
        distinct_finished = sum(road["finishes"] > 0 for road in result["roads"])
        report = {"status": "complete", "scope": "reused_consumed_TRAIN_development_only",
                  "reused_train_only": True, "generalization_claim": False, "official_score": False,
                  "protocol_sha256": sha, "evaluation_source_sha256": p["evaluation_source_sha256"],
                  "source_protocol_sha256": SOURCE["protocol"]["sha256"],
                  "source_result_sha256": SOURCE["result"]["sha256"],
                  "baseline_full_eval_result_sha256": BASELINE["full_eval_result"]["sha256"],
                  "source_model_sha256": pin["sha256"], "source_model_trained_horizon": 3,
                  "planner": PLANNER, "cpu_only_torch": str(torch.__version__),
                  "evaluation_reward": "raw_environment_only",
                  "full_episode_finish_comparison_valid": result["censored"] == 0,
                  "pairing": "same consumed TRAIN road/reset seeds as historical MPPI; different stochastic actions and trajectories",
                  "denominators": {"distinct_training_roads": 4, "repeats_per_road": 2,
                                   "planned_episodes": 8, "completed_episodes": len(rows), "max_steps": 2000},
                  "episodes_sha256": raw._digest(ledger), "primary_mppi": result,
                  "baseline_h3_eval_mode_true": {"episodes": 8, "finishes": 0},
                  "primary_local_target": {"min_finishes": 4, "min_distinct_finished_roads": 2,
                                           "observed_finishes": result["finishes"],
                                           "observed_distinct_finished_roads": distinct_finished,
                                           "met_on_reused_train": result["finishes"] >= 4 and distinct_finished >= 2
                                           and result["censored"] == 0,
                                           "fresh_generalization_claim": False},
                  "resources": p["resources"], "peak_process_rss_bytes": resource.getrusage(
                      resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "elapsed_seconds": clock() - started}
        if env is not None:
            env.close()
            env = None
        _write_json(output / "result.json", report)
        return report
    except BaseException as exc:
        try:
            raw._journal(ledger, {"event": "partial", "reason": type(exc).__name__,
                                  "complete_episodes": len(rows), "reset_intents": resets,
                                  "resume_supported": False})
        except BaseException:
            try:
                _write_json(output / "failure.json", {
                    "event": "failure", "reason": type(exc).__name__, "protocol_sha256": sha,
                    "source_result_sha256": SOURCE["result"]["sha256"],
                    "episode_ledger_sha256": raw._digest(ledger) if ledger.is_file() else None,
                    "complete_episodes": len(rows), "reset_intents": resets,
                    "environment_resets": None if resets else 0, "resume_supported": False})
            except BaseException as preserve_error:
                raise RuntimeError("cannot durably preserve partial evaluation; exposure unknown") from preserve_error
        raise
    finally:
        if env is not None:
            env.close()


def execute(protocol_path: Path, protocol_sha256: str, *, root: Path = ROOT) -> dict:
    """Production-only executor. No public test factories or action-mode switches."""
    root = root.resolve(strict=True)
    p, checked = _check(protocol_path, protocol_sha256, root=root)
    _, latest = _check(protocol_path, protocol_sha256, root=root)
    if latest["checkpoint"] != checked["checkpoint"] or p != raw._json(raw._evaluation_path(root, protocol_path).read_bytes()):
        raise ValueError("frozen H3 noise protocol or checkpoint changed")
    from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner
    from haic.algorithms.tdmpc2.haic_env import make_training_env

    pin = checked["checkpoint"]
    initial_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    model = raw._model({**pin, "path": str(raw._path(root, pin["path"]))}, {
        "format": "haic-tdmpc2-long-train-v1",
        "protocol_sha256": SOURCE["protocol"]["sha256"],
        "source_sha256": _source(root)[0]["source_sha256"]})
    planner = TDMPC2Planner(model, PlannerConfig(action_dim=3, discount=.995, horizon=3,
                                                episodic=True, num_samples=512, num_pi_trajs=24,
                                                iterations=6, num_elites=64))
    allocated = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - initial_peak)
    if _check(protocol_path, protocol_sha256, root=root, allocated_bytes=allocated)[1]["checkpoint"] != pin:
        raise ValueError("RAW100k source changed after trusted model load")
    return _run_core(root, p, protocol_path, protocol_sha256, pin, model, planner,
                     env_factory=make_training_env, initial_peak_rss_bytes=initial_peak)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True, help="separately frozen experiments/*.json")
    parser.add_argument("--protocol-sha256", required=True, help="external SHA-256 of evaluation protocol")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="default: read-only, zero resets")
    mode.add_argument("--execute", action="store_true", help="eight full-episode reused TRAIN resets")
    args = parser.parse_args()
    result = execute(args.protocol, args.protocol_sha256) if args.execute else preflight(args.protocol, args.protocol_sha256)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
