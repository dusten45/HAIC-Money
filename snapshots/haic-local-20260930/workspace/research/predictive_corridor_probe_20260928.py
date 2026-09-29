"""Screen pixel preview/steering-memory mechanism on consumed tune cells."""

import json
from pathlib import Path
import time

import torch

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.predictive_corridor_runtime import PredictiveCorridorAgent
from research.score_probe_20260928 import CHECKPOINT, ROOT, compact, make_agent, sha256, summary
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/predictive-corridor-activation-20260928"
CELLS = ((1, 228), (3, 228), (1, 230), (1, 184))


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name,
        "purpose": "direction 1 activation screen, not an independent outcome comparison",
        "plan": "docs/plans/2026-09-28-top-three-performance-rebuild.md",
        "split": "consumed_tune",
        "cells": [{"track_id": t, "seed": s} for t, s in CELLS],
        "max_decisions": 2000,
        "source_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_sha256": sha256(ROOT / "haic_agent/predictive_corridor_runtime.py"),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    rows = {"control": [], "candidate": []}
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for track, seed in CELLS:
            for name in ("control", "candidate"):
                agent = (ObstacleFullRoadGuardAgent(make_agent(saved, 1.0))
                         if name == "control" else PredictiveCorridorAgent())
                result = run_episode(mode=name, track_id=track, seed=seed,
                    agent=agent, max_decisions=2000, plan_budget_seconds=4.5)
                row = compact(result)
                row["preview_count"] = getattr(agent, "preview_count", None)
                row["dropout_count"] = getattr(agent, "dropout_count", None)
                row["mean_speed"] = result.get("mean_speed")
                row["max_speed"] = result.get("max_speed")
                rows[name].append(row)
                events.write(json.dumps({"type": "episode", "arm": name, **row}) + "\n")
                events.flush()
                print(name, track, seed, row["completed"], row["lapTimeMs"],
                      row["progress"], row["preview_count"], row["dropout_count"], flush=True)
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name,
        "decision": "DIAGNOSTIC_ONLY",
        "summaries": {name: summary(values) for name, values in rows.items()},
        "gates": {
            "rule_compliance": "UNKNOWN",
            "mechanism_activation": "PASS" if any(r["dropout_count"] for r in rows["candidate"]) else "UNKNOWN",
            "competitive_or_product_outcome": "NOT_APPLICABLE",
        },
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
