"""Compare exact speed-budget and selected-control ZIPs on untouched confirmation cells."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from training.confirm_stable_package import ROOT, RESULT_KEYS, check_archive, summarize as _base_summarize
from training.confirm_speed_coupled_package import decide


CANDIDATE_SHA256 = "6b6c675e4e30dc51af061b7c42a9c73058222cf1873adc7ed6530774f7f8cda9"
CONTROL_SHA256 = "5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40"
CELLS = tuple((track, seed) for track in (1, 2, 3) for seed in (2020, 2021, 2022, 2023))


def cells_for(seed_start: int) -> tuple[tuple[int, int], ...]:
    if seed_start < 1:
        raise ValueError("seed start must be positive")
    return tuple((track, seed) for track in (1, 2, 3) for seed in range(seed_start, seed_start + 4))
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
row["activation_count"] = getattr(driver._controller, "activation_count", 0)
print(json.dumps(row), flush=True)
'''


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict]) -> dict:
    base = _base_summarize([dict(row, risk_clear_decisions=row.get("risk_clear_decisions", 0)) for row in rows])
    base.pop("risk_clear_decisions")
    base["activation_count"] = sum(int(row.get("activation_count", 0)) for row in rows)
    return base


def compare(candidate: Path, control: Path, output: Path,
            cells: tuple[tuple[int, int], ...] = CELLS) -> dict:
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
                                  "completed": row["completed"], "lapTimeMs": row["lapTimeMs"]}), flush=True)
    if any(digest(path) != expected[arm] for arm, path in archives.items()):
        raise RuntimeError("package input changed during comparison")
    summaries = {arm: summarize(values) for arm, values in rows.items()}
    report = {"candidate_sha256": CANDIDATE_SHA256, "control_sha256": CONTROL_SHA256,
              "cells": [{"track_id": t, "seed": s} for t, s in cells],
              "checks": checks, "summaries": summaries, "episodes": rows,
              "decision": decide(summaries["candidate"], summaries["control"])}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--control-archive", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.candidate_archive, args.control_archive, args.output,
                     cells_for(args.seed_start))
    print(json.dumps({"summaries": result["summaries"], "decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
