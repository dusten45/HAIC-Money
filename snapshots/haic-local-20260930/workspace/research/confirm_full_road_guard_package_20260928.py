"""Run a frozen extracted archive on unused Linux confirmation cells."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "artifacts/haic-research-v2/full-road-guard-candidate-20260928/submission-full-road-guard.zip"
RUN = ROOT / "runs/haic-research-v2/score-linux-full-road-guard-package-confirmation-20260928"
CELLS = tuple((track, seed) for track in (1, 2) for seed in (192, 193, 194, 195))

CHILD = r'''
import json
import statistics
import sys
from pathlib import Path

import agent

root = Path(sys.argv[1])
run = Path(sys.argv[2])
cells = json.loads(sys.argv[3])
sys.path.append(str(root))
from training.evaluate_closed_loop import run_episode

driver = agent.Agent()
rows = []
keys = ("track_id", "seed", "completed", "lapTimeMs", "progress", "collisions",
        "damage", "retire_reason", "error", "invalid_actions", "steps",
        "act_p50_ms", "act_p95_ms", "act_max_ms", "wall_time_s")
with (run / "events.jsonl").open("x", encoding="utf-8") as events:
    for track, seed in cells:
        raw = run_episode(mode="packaged_confirmation", track_id=track, seed=seed,
                          agent=driver, max_decisions=2000, plan_budget_seconds=4.5)
        row = {key: raw[key] for key in keys}
        rows.append(row)
        events.write(json.dumps({"type": "episode", **row}) + "\n")
        events.flush()
        print(track, seed, row["completed"], row["lapTimeMs"], row["progress"], flush=True)
finished = [row["lapTimeMs"] for row in rows if row["completed"]]
summary = {"completed": len(finished), "denominator": len(rows),
           "median_finished_lap_ms": statistics.median(finished) if finished else None,
           "mean_progress": statistics.mean(row["progress"] for row in rows),
           "collisions": sum(row["collisions"] or 0 for row in rows),
           "damage": sum(row["damage"] or 0 for row in rows),
           "invalid_actions": sum(row["invalid_actions"] for row in rows),
           "act_p95_ms_max": max(row["act_p95_ms"] for row in rows),
           "act_max_ms": max(row["act_max_ms"] for row in rows)}
(run / "integration_report.json").write_text(json.dumps({
    "run_id": run.name, "decision": "CONFIRMATION_ONLY", "summary": summary,
    "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "UNKNOWN",
              "competitive_or_product_outcome": "UNKNOWN"}
}, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary), flush=True)
'''


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "frozen package confirmation on unused cells",
        "split": "confirmation", "cells": [{"track_id": t, "seed": s} for t, s in CELLS],
        "max_decisions": 2000, "archive_sha256": digest(ARCHIVE),
        "source_sha256": digest(Path(__file__).resolve()), "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory)
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extractall(destination)
        completed = subprocess.run(
            [sys.executable, "-c", CHILD, str(ROOT), str(RUN), json.dumps(CELLS)],
            cwd=destination, text=True, capture_output=True, timeout=300, check=False,
        )
        print(completed.stdout, end="", flush=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr + completed.stdout)


if __name__ == "__main__":
    main()
