"""Development-only evaluator. This file is never placed in a submission ZIP."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANDATORY_CELLS = ((1, 516237), (2, 644062), (3, 1007), (4, 18800))
# A geometry is reserved across all four IDs. Unrelated sealed studies are not read.
DEVELOPMENT_SEEDS = (2867319041, 359018627, 1764402399, 4029571806)
HOLDOUT_SEEDS = (3249018572, 1097358264, 2376840915, 798412603)
ENV_FILES = ("env_wrapper.py", "damage.py", "core/track_variables.py",
             "core/vendor/car_racing.py", "core/vendor/car_dynamics.py",
             "core/finish_line.py", "core/obstacle_contacts.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_action(action):
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("action must contain three finite numbers")
    if np.any(value < (-1, 0, 0)) or np.any(value > (1, 1, 1)):
        raise ValueError("action outside official bounds")
    return value


def pace_profile(consecutive_rejections):
    if type(consecutive_rejections) is not int or consecutive_rejections < 0:
        raise ValueError("rejections must be a nonnegative integer")
    return (13, 15, 18)[min(2, consecutive_rejections // 3)]


def assess(mandatory, extra, *, limit_seconds):
    all_rows = list(mandatory) + list(extra)
    coords = [(r["track_id"], r["seed"]) for r in all_rows]
    if len(coords) != len(set(coords)):
        raise ValueError("duplicate evaluation cell")
    if set((r["track_id"], r["seed"]) for r in mandatory) != set(MANDATORY_CELLS):
        raise ValueError("missing or unexpected mandatory cells")
    if not extra or {r["track_id"] for r in extra} != {1, 2, 3, 4}:
        raise ValueError("extra evaluation must cover all four tracks")

    def fast(r, limit):
        return (r["finished"] is True and isinstance(r.get("lap_time_ms"), (int, float))
                and 0 < r["lap_time_ms"] <= 1000 * limit
                and r.get("progress", 0) >= .95 and not r.get("error"))

    mandatory_finished = sum(bool(r["finished"]) for r in mandatory)
    extra_finished = sum(bool(r["finished"]) for r in extra)
    per_track = {}
    for track in range(1, 5):
        rows = [r for r in extra if r["track_id"] == track]
        per_track[str(track)] = {"finishes": sum(bool(r["finished"]) for r in rows),
                                 "cells": len(rows),
                                 "fast_fraction": sum(fast(r, limit_seconds) for r in rows) / len(rows)}
    extra_fast = sum(fast(r, limit_seconds) for r in extra) / len(extra)
    passed = (all(fast(r, limit_seconds) for r in mandatory) and extra_fast >= .90
              and all(r["fast_fraction"] >= .75 for r in per_track.values()))
    original_target_met = (all(fast(r, 13) for r in mandatory)
                           and sum(fast(r, 13) for r in extra) / len(extra) >= .90
                           and all(sum(fast(r, 13) for r in extra if r["track_id"] == t)
                                   / sum(r["track_id"] == t for r in extra) >= .75
                                   for t in range(1, 5)))
    return {"passed": passed, "original_target_met": original_target_met,
            "pace_limit_seconds": limit_seconds, "mandatory_finishes": mandatory_finished,
            "mandatory_cells": len(mandatory), "extra_finishes": extra_finished,
            "extra_cells": len(extra), "extra_fast_fraction": extra_fast,
            "per_track": per_track}


def run_cell(source, params, track, seed, *, max_steps=2000, trace_path=None):
    # Only this evaluator touches simulator state; Agent receives observation only.
    from local_simulator.environment import create_environment, reset_environment
    from local_simulator.schema import MapSpec
    from tools.evaluate_bare_generalization import _peak_worker_rss_mib

    source = Path(source).resolve()
    source_hash = digest(source)
    started = time.perf_counter()
    module_spec = importlib.util.spec_from_file_location("apex_submission", source)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    agent = module.Agent(**params)
    init_ms = (time.perf_counter() - started) * 1000
    if init_ms > 10000:
        raise RuntimeError("initialization exceeded official 10-second limit")
    spec = MapSpec(track, seed, "official", (), max_steps, 4)
    environment, raw = create_environment(spec, render_mode=None)
    traces, contacts, offroad, partial = [], 0, 0, 0
    action_hash = hashlib.sha256()
    max_act_ms = 0.0
    info = {}
    try:
        observation, _ = reset_environment(environment, spec)
        start_time = float(raw.t)
        began = time.perf_counter()
        agent.reset(observation)
        reset_ms = 1000 * (time.perf_counter() - began)
        if reset_ms > 5000:
            raise RuntimeError("reset exceeded official 5-second limit")
        for step in range(max_steps):
            began = time.perf_counter()
            action = validate_action(agent.act(observation))
            latency_ms = 1000 * (time.perf_counter() - began)
            if latency_ms > 5000:
                raise RuntimeError("action exceeded official 5-second limit")
            max_act_ms = max(max_act_ms, latency_ms)
            action_hash.update(action.tobytes())
            observation, _, terminated, truncated, info = environment.step(action)
            contacts += int(info.get("collision", False))
            wheel_contacts = [bool(w.tiles) for w in raw.car.wheels]
            offroad += int(not any(wheel_contacts))
            partial += int(not all(wheel_contacts))
            if trace_path:
                traces.append({"step": step, "sim_time_s": raw.t - start_time,
                               "action": action.tolist(), "progress": info.get("progress", 0),
                               "speed": float(np.linalg.norm(raw.car.hull.linearVelocity)),
                               "position": list(raw.car.hull.position), "angle": raw.car.hull.angle,
                               "collision": bool(info.get("collision", False)),
                               "debug": getattr(agent, "debug", {})})
            if terminated or truncated:
                break
        finish = raw.finish_time_s
        row = {"track_id": track, "seed": seed, "finished": finish is not None,
               "lap_time_ms": round(1000 * (finish - start_time)) if finish is not None else None,
               "progress": float(info.get("progress", 0)), "collision_count": contacts,
               "damage": float(info.get("damage", 0)),
               "retire_reason": None if finish is not None else (info.get("retire_reason") or "max_steps"),
               "steps": step + 1, "initialization_ms": init_ms, "reset_ms": reset_ms,
               "action_latency_max_ms": max_act_ms, "peak_worker_rss_mib": _peak_worker_rss_mib(),
               "offtrack_samples": offroad, "partial_offtrack_samples": partial,
               "source_sha256": source_hash, "parameters": params,
               "action_trace_sha256": action_hash.hexdigest(), "error": None}
    finally:
        environment.close()
    if row["peak_worker_rss_mib"] > 1024:
        raise RuntimeError("worker exceeded official 1024 MiB memory limit")
    if digest(source) != source_hash:
        raise RuntimeError("source changed during evaluation")
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        Path(trace_path).write_text(json.dumps(traces), encoding="utf-8")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("agent.py"))
    parser.add_argument("--params", default="{}")
    parser.add_argument("--suite", choices=("mandatory", "development", "holdout"), default="mandatory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run_cell(**json.loads(args.worker))))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        parser.error("output exists; use a new path to preserve prior receipts")
    params = json.loads(args.params)
    if not isinstance(params, dict) or args.workers < 1 or args.max_steps < 1:
        parser.error("invalid parameters, workers, or max-steps")
    source = args.source.resolve()
    source_hash = digest(source)
    env_hashes = {p: digest(ROOT / p) for p in ENV_FILES}
    seeds = DEVELOPMENT_SEEDS if args.suite == "development" else HOLDOUT_SEEDS
    cells = list(MANDATORY_CELLS)
    if args.suite != "mandatory":
        cells += [(track, seed) for seed in seeds for track in range(1, 5)]
    freeze = {"source_sha256": source_hash, "parameters": params, "suite": args.suite,
              "environment_sha256": env_hashes, "cells": cells, "max_steps": args.max_steps}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    def run(pair):
        track, seed = pair
        payload = {"source": str(source), "params": params, "track": track, "seed": seed,
                   "max_steps": args.max_steps}
        if args.trace_dir:
            payload["trace_path"] = str(args.trace_dir / f"track-{track}-seed-{seed}.json")
        command = [sys.executable, "-m", "agents.apex_2026.evaluate", "--worker", json.dumps(payload)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
        if result.returncode:
            raise RuntimeError(f"worker {pair} failed: {result.stderr[-3000:]}")
        row = json.loads(result.stdout)
        if row["source_sha256"] != source_hash or digest(source) != source_hash:
            raise RuntimeError("source changed after freeze")
        print(json.dumps({k: row[k] for k in ("track_id", "seed", "finished", "lap_time_ms", "progress", "collision_count")}), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run, cells))
    if env_hashes != {p: digest(ROOT / p) for p in ENV_FILES}:
        raise RuntimeError("official environment changed during evaluation")
    report = {"freeze": freeze, "rows": rows}
    if args.suite != "mandatory":
        report["profiles"] = {str(limit): assess(rows[:4], rows[4:], limit_seconds=limit)
                              for limit in (13, 15, 18)}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"receipt: {args.output}")


if __name__ == "__main__":
    main()
