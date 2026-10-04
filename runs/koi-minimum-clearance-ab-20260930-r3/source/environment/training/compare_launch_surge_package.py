"""Compare frozen launch and selected-control ZIPs on registered local cells."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import zipfile

from training.confirm_stable_package import ROOT, RESULT_KEYS, check_archive, summarize as base_summarize
from training.confirm_speed_coupled_package import decide


CONTROL_SHA256 = "5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40"
CHILD = r'''
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
track = int(sys.argv[2])
seed = int(sys.argv[3])
started = time.monotonic()
import agent
driver = agent.Agent()
import_create_s = time.monotonic() - started
if not str(Path(agent.__file__).resolve()).startswith(str(Path.cwd())):
    raise RuntimeError("packaged agent was not imported")
started = time.monotonic()
driver.reset(np.zeros((4, 84, 84), dtype=np.float32))
reset_s = time.monotonic() - started
sys.path.append(str(root))
from training.evaluate_closed_loop import run_episode
raw = run_episode(mode="packaged_launch_comparison", track_id=track, seed=seed,
                  agent=driver, max_decisions=2000, plan_budget_seconds=4.5,
                  capture_trace=True, fail_on_invalid_action=True)
keys = json.loads(sys.argv[4])
row = {key: raw[key] for key in keys}
trace = raw["decision_trace"] or []
initial = next((index + 1 for index, item in enumerate(trace)
                if item["progress"] is not None and item["progress"] >= 0.10), None)
early = trace[:initial] if initial is not None else trace
row["first_tenth_ms"] = initial * 80 if initial is not None else None
row["first_tenth_mean_gas"] = (sum(item["gas"] for item in early) / len(early)) if early else None
row["first_tenth_mean_speed"] = (sum(item["speed"] for item in early) / len(early)) if early else None
row["peak_speed"] = max((item["speed"] for item in trace), default=None)
row["decision_trace"] = trace
row["import_create_s"] = import_create_s
row["reset_s"] = reset_s
row["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
row["launch_count"] = getattr(driver._controller, "launch_count", 0)
row["launch_exit_reason"] = getattr(driver._controller, "exit_reason", None)
row["risk_clear_decisions"] = 0
print(json.dumps(row), flush=True)
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cells_for(seed_start):
    if seed_start < 1:
        raise ValueError("seed start must be positive")
    return tuple((track, seed) for track in (1, 2, 3)
                 for seed in range(seed_start, seed_start + 4))


def summarize(rows):
    summary = base_summarize(rows)
    summary.pop("risk_clear_decisions")
    first_tenth = [row["first_tenth_ms"] for row in rows if row["first_tenth_ms"] is not None]
    early_gas = [row["first_tenth_mean_gas"] for row in rows if row["first_tenth_mean_gas"] is not None]
    early_speed = [row["first_tenth_mean_speed"] for row in rows if row["first_tenth_mean_speed"] is not None]
    summary["median_first_tenth_ms"] = statistics.median(first_tenth) if first_tenth else None
    summary["mean_first_tenth_gas"] = statistics.mean(early_gas) if early_gas else None
    summary["mean_first_tenth_speed"] = statistics.mean(early_speed) if early_speed else None
    summary["launch_count"] = sum(row["launch_count"] for row in rows)
    return summary


def compare(candidate, control, expected_candidate_sha256, seed_start, output):
    if len(expected_candidate_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_candidate_sha256):
        raise ValueError("expected candidate digest must be lowercase SHA-256")
    archives = {"candidate": candidate, "control": control}
    expected = {"candidate": expected_candidate_sha256, "control": CONTROL_SHA256}
    for arm, path in archives.items():
        if digest(path) != expected[arm]:
            raise ValueError(f"{arm} archive differs from frozen package")
    checks = {arm: check_archive(path) for arm, path in archives.items()}
    rows = {arm: [] for arm in archives}
    cells = cells_for(seed_start)
    with tempfile.TemporaryDirectory() as directory:
        roots = {}
        for arm, path in archives.items():
            target = Path(directory) / arm
            target.mkdir()
            with zipfile.ZipFile(path) as archive:
                archive.extractall(target)
            roots[arm] = target
        for index, (track, seed) in enumerate(cells):
            order = ("control", "candidate") if index % 2 == 0 else ("candidate", "control")
            for arm in order:
                result = subprocess.run(
                    [sys.executable, "-c", CHILD, str(ROOT), str(track), str(seed), json.dumps(RESULT_KEYS)],
                    cwd=roots[arm], text=True, capture_output=True, timeout=120, check=False,
                )
                if result.returncode:
                    raise RuntimeError(f"{arm} {track}/{seed}: {result.stderr[-2000:]}")
                row = json.loads(result.stdout.strip().splitlines()[-1])
                rows[arm].append(row)
                print(json.dumps({"arm": arm, "track": track, "seed": seed,
                                  "completed": row["completed"], "lapTimeMs": row["lapTimeMs"],
                                  "first_tenth_ms": row["first_tenth_ms"]}), flush=True)
    if any(digest(path) != expected[arm] for arm, path in archives.items()):
        raise RuntimeError("package input changed during comparison")
    summaries = {arm: summarize(values) for arm, values in rows.items()}
    report = {"candidate_sha256": expected_candidate_sha256, "control_sha256": CONTROL_SHA256,
              "cells": [{"track_id": track, "seed": seed} for track, seed in cells],
              "checks": checks, "summaries": summaries, "episodes": rows,
              "decision": decide(summaries["candidate"], summaries["control"])}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--control-archive", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.candidate_archive, args.control_archive, args.candidate_sha256,
                     args.seed_start, args.output)
    print(json.dumps({"summaries": result["summaries"], "decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
