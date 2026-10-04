"""Compare exact frozen controller ZIPs on fresh local finish-line cells."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import zipfile

from training.package_submission import static_violations


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_SHA256 = "fd837b6cfeb41bcf7011991b929a25c77ba22bfd6b631d3e46803921121e4169"
CONTROL_SHA256 = "5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40"
CELLS = tuple((track, seed) for track in (1, 2, 3) for seed in (336, 337, 338, 339))
RESULT_KEYS = ("track_id", "seed", "completed", "lapTimeMs", "progress", "collisions", "damage",
               "retire_reason", "error", "invalid_actions", "steps", "act_p50_ms", "act_p95_ms",
               "act_max_ms", "wall_time_s")
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
raw = run_episode(mode="packaged_confirmation", track_id=track, seed=seed,
                  agent=driver, max_decisions=2000, plan_budget_seconds=4.5,
                  fail_on_invalid_action=True)
keys = json.loads(sys.argv[4])
row = {key: raw[key] for key in keys}
row["import_create_s"] = import_create_s
row["reset_s"] = reset_s
row["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
row["risk_clear_decisions"] = getattr(driver._controller, "risk_clear_decisions", None)
print(json.dumps(row), flush=True)
'''


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_archive(path: Path) -> dict:
    if path.stat().st_size > 500_000_000:
        raise ValueError("ZIP exceeds size limit")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if "agent.py" not in names or len(names) != len(set(names)) or len(names) > 1000:
            raise ValueError("invalid archive inventory")
        total = 0
        for info in archive.infolist():
            name = info.filename
            if info.is_dir() or name.startswith("/") or ".." in Path(name).parts or "\\" in name:
                raise ValueError("invalid archive path")
            if Path(name).suffix.lower() in {".com", ".dll", ".dylib", ".exe", ".msi", ".scr", ".so"}:
                raise ValueError("forbidden executable in archive")
            if info.file_size > 500_000_000 or info.file_size / max(info.compress_size, 1) > 100:
                raise ValueError("ZIP member exceeds size or compression ratio limit")
            total += info.file_size
            if name.endswith(".py"):
                violations = static_violations(archive.read(name).decode("utf-8"), name)
                if violations:
                    raise ValueError(f"{name}: {violations}")
        if total > 2_000_000_000:
            raise ValueError("extracted archive exceeds size limit")
    return {"files": len(names), "archive_bytes": path.stat().st_size, "uncompressed_bytes": total}


def summarize(rows: list[dict]) -> dict:
    finished = sorted(row["lapTimeMs"] for row in rows if row["completed"])
    incomplete = [row["progress"] for row in rows if not row["completed"]]
    return {
        "completed": len(finished), "denominator": len(rows),
        "median_finished_lap_ms": statistics.median(finished) if finished else None,
        "mean_incomplete_progress": statistics.mean(incomplete) if incomplete else None,
        "p90_finished_lap_ms": finished[min(len(finished) - 1, int(0.9 * (len(finished) - 1)))] if finished else None,
        "collisions": sum(row["collisions"] or 0 for row in rows),
        "damage": sum(row["damage"] or 0 for row in rows),
        "invalid_actions": sum(row["invalid_actions"] for row in rows),
        "max_import_create_s": max(row["import_create_s"] for row in rows),
        "max_reset_s": max(row["reset_s"] for row in rows),
        "max_peak_rss_bytes": max(row["peak_rss_bytes"] for row in rows),
        "max_act_ms": max(row["act_max_ms"] for row in rows),
        "p95_act_ms": max(row["act_p95_ms"] for row in rows),
        "risk_clear_decisions": sum(row["risk_clear_decisions"] or 0 for row in rows),
    }


def compare(candidate: Path, control: Path, output: Path) -> dict:
    archives = {"candidate": candidate, "control": control}
    expected = {"candidate": CANDIDATE_SHA256, "control": CONTROL_SHA256}
    for arm, path in archives.items():
        if digest(path) != expected[arm]:
            raise ValueError(f"{arm} archive differs from frozen package")
    checks = {arm: check_archive(path) for arm, path in archives.items()}
    rows = {arm: [] for arm in archives}
    with tempfile.TemporaryDirectory() as directory:
        roots = {}
        for arm, path in archives.items():
            target = Path(directory) / arm
            target.mkdir()
            with zipfile.ZipFile(path) as archive:
                archive.extractall(target)
            roots[arm] = target
        for index, (track, seed) in enumerate(CELLS):
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
                                  "completed": row["completed"], "lapTimeMs": row["lapTimeMs"]}), flush=True)
    if any(digest(path) != expected[arm] for arm, path in archives.items()):
        raise RuntimeError("package input changed during comparison")
    summaries = {arm: summarize(values) for arm, values in rows.items()}
    report = {"candidate_sha256": CANDIDATE_SHA256, "control_sha256": CONTROL_SHA256,
              "cells": [{"track_id": t, "seed": s} for t, s in CELLS],
              "checks": checks, "summaries": summaries, "episodes": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--control-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.candidate_archive, args.control_archive, args.output)
    print(json.dumps(result["summaries"], sort_keys=True))


if __name__ == "__main__":
    main()
