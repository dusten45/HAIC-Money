"""Evaluate one fixed obstacle-hybrid candidate on the registered tune cells."""

import json
from pathlib import Path
import time

import torch

from research.hybrid_actor import ObstacleHybridAgent
from research.score_probe_20260928 import CHECKPOINT, CELLS, ROOT, compact, make_agent, sha256, summary
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/score-probe-hybrid-20260928"


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    hybrid_source = ROOT / "research/hybrid_actor.py"
    manifest = {
        "run_id": RUN.name,
        "purpose": "fixed pixel-obstacle hybrid tune diagnostic; not SOTA evidence",
        "source_sha256": sha256(source),
        "hybrid_source_sha256": sha256(hybrid_source),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "cells": [{"track_id": track, "seed": seed} for track, seed in CELLS],
        "split": "tune",
        "max_decisions": 2000,
        "throttle_expansion": 3.5,
        "obstacle_override_y_min": 28.0,
        "created_unix": time.time(),
    }
    (RUN / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    agent = ObstacleHybridAgent(make_agent(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), 3.5))
    rows = []
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for track, seed in CELLS:
            row = compact(run_episode(
                mode="hybrid", track_id=track, seed=seed, agent=agent,
                max_decisions=2000, plan_budget_seconds=4.5,
            ))
            row["obstacle_override_count"] = agent.override_count
            rows.append(row)
            events.write(json.dumps({"type": "episode", **row}) + "\n")
            events.flush()
            print(track, seed, row["completed"], row["lapTimeMs"], row["progress"], row["obstacle_override_count"], flush=True)
    report = {
        "run_id": RUN.name,
        "summary": summary(rows),
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "PASS", "competitive_or_product_outcome": "UNKNOWN"},
        "decision": "DIAGNOSTIC_ONLY",
    }
    (RUN / "integration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
