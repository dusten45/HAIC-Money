"""Expand fresh tune coverage for the frozen side-evidence candidate."""

import json
from pathlib import Path
import time

import torch

from haic_agent.obstacle_side_evidence_runtime import ObstacleSideEvidenceAgent
from research.score_probe_20260928 import CHECKPOINT, ROOT, compact, make_agent, sha256, summary
from training.evaluate_closed_loop import run_episode


CELLS = tuple((track, seed) for track in (1, 2) for seed in range(176, 184))
RUN = ROOT / "runs/haic-research-v2/score-linux-side-evidence-expanded-tune-20260928"


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "fresh tune coverage for failure discovery",
        "split": "tune", "cells": [{"track_id": t, "seed": s} for t, s in CELLS],
        "max_decisions": 2000,
        "source_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_sha256": sha256(ROOT / "haic_agent/obstacle_side_evidence_runtime.py"),
        "checkpoint_sha256": sha256(CHECKPOINT), "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    agent = ObstacleSideEvidenceAgent(make_agent(saved, 1.0))
    rows = []
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for track, seed in CELLS:
            row = compact(run_episode(mode="side_evidence", track_id=track, seed=seed,
                agent=agent, max_decisions=2000, plan_budget_seconds=4.5))
            row["early_evidence_count"] = agent.early_evidence_count
            row["road_guard_count"] = agent.road_guard_count
            rows.append(row)
            events.write(json.dumps({"type": "episode", **row}) + "\n")
            events.flush()
            print(track, seed, row["completed"], row["lapTimeMs"], row["progress"], row["retire_reason"], flush=True)
    result = summary(rows)
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY", "summary": result,
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "UNKNOWN",
                  "competitive_or_product_outcome": "UNKNOWN"}
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
