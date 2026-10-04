"""Compare frozen speed-coupled and selected-control ZIPs on untouched held-out cells."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from training.confirm_stable_package import CHILD, RESULT_KEYS, ROOT, check_archive, summarize


CANDIDATE_SHA256 = "54c117287390704acc52505d680c31574fdcf53ac062cbb0ce5258a57058c324"
CONTROL_SHA256 = "5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40"
CELLS = tuple((track, seed) for track in (1, 2, 3) for seed in (2004, 2005, 2006, 2007))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide(candidate: dict, control: dict) -> str:
    """Apply preregistered completion-first comparison; road contact is diagnostic."""
    if candidate["invalid_actions"]:
        return "REJECT"
    if candidate["completed"] != control["completed"]:
        return "ADVANCE" if candidate["completed"] > control["completed"] else "REJECT"
    candidate_lap = candidate["median_finished_lap_ms"]
    control_lap = control["median_finished_lap_ms"]
    if candidate_lap is not None and (control_lap is None or candidate_lap < control_lap):
        return "ADVANCE"
    return "REJECT"


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.candidate_archive, args.control_archive, args.output)
    print(json.dumps({"summaries": result["summaries"], "decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
