"""Check pixel-only corridor driving on a consumed failure cell."""

import json
from pathlib import Path
import time

from haic_agent.corridor_agent import VisionCorridorAgent
from research.score_probe_20260928 import ROOT, compact, sha256
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/corridor-reliability-probe-20260928"


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "pixel-only corridor mechanism probe",
        "split": "tune", "cells": [{"track_id": 1, "seed": 185}],
        "source_sha256": sha256(Path(__file__).resolve()),
        "corridor_source_sha256": sha256(ROOT / "haic_agent/corridor_agent.py"),
        "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    agent = VisionCorridorAgent()
    row = compact(run_episode(mode="corridor_probe", track_id=1, seed=185,
        agent=agent, max_decisions=2000, plan_budget_seconds=4.5))
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        events.write(json.dumps({"type": "episode", **row}) + "\n")
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY", "result": row,
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "UNKNOWN",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"}
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
