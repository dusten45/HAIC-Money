"""Regression check of full-road confidence on expanded tune cells."""

import json
from pathlib import Path
import time

import torch

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from research.score_probe_20260928 import CHECKPOINT, ROOT, compact, make_agent, sha256, summary
from training.evaluate_closed_loop import run_episode


CELLS = tuple((track, seed) for track in (1, 2) for seed in range(176, 184))
RUN = ROOT / "runs/haic-research-v2/score-linux-full-road-guard-expanded-tune-20260928"


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "regression check on consumed expanded tune cells",
        "split": "tune", "cells": [{"track_id": t, "seed": s} for t, s in CELLS],
        "source_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_sha256": sha256(ROOT / "haic_agent/obstacle_full_road_guard_runtime.py"),
        "checkpoint_sha256": sha256(CHECKPOINT), "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    agent = ObstacleFullRoadGuardAgent(make_agent(saved, 1.0))
    rows = []
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for track, seed in CELLS:
            row = compact(run_episode(mode="full_road_guard", track_id=track, seed=seed,
                agent=agent, max_decisions=2000, plan_budget_seconds=4.5))
            row["full_road_guard_count"] = agent.full_road_guard_count
            rows.append(row)
            events.write(json.dumps({"type": "episode", **row}) + "\n")
            events.flush()
            print(track, seed, row["completed"], row["lapTimeMs"], row["progress"], row["full_road_guard_count"], flush=True)
    result = summary(rows)
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY", "summary": result,
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "UNKNOWN",
                  "competitive_or_product_outcome": "UNKNOWN"}
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
