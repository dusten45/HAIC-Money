"""Diagnose coupled-controller tune failures without opening reserved cells."""

import json
from pathlib import Path
import time

from haic_agent.speed_coupled_corridor_runtime import SpeedCoupledCorridorAgent
from research.score_probe_20260928 import ROOT, compact, sha256
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/speed-coupled-failure-trace-20260928"
CELLS = ((1, 228), (3, 228), (1, 230))


def summarize(result):
    trace = result["decision_trace"] or []
    progress = [item["progress"] for item in trace]
    last_advance = max((i + 1 for i in range(1, len(progress))
                        if progress[i] > progress[i - 1]), default=None)
    return {
        **compact(result),
        "max_speed": max((x["speed"] for x in trace), default=None),
        "last_progress_advance_step": last_advance,
        "last_10": [{key: item[key] for key in ("step", "progress", "speed", "steer", "gas", "brake")}
                    for item in trace[-10:]],
    }


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "failure-cause diagnostic on consumed tune only",
        "split": "consumed_tune", "cells": [{"track_id": t, "seed": s} for t, s in CELLS],
        "source_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_sha256": sha256(ROOT / "haic_agent/speed_coupled_corridor_runtime.py"),
        "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    rows = []
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for track, seed in CELLS:
            row = summarize(run_episode(mode="speed_coupled_failure_trace", track_id=track,
                seed=seed, agent=SpeedCoupledCorridorAgent(), max_decisions=2000,
                plan_budget_seconds=4.5, capture_trace=True))
            rows.append(row)
            events.write(json.dumps({"type": "trace_summary", **row}) + "\n")
            print(json.dumps(row), flush=True)
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY", "rows": rows,
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "PASS",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
