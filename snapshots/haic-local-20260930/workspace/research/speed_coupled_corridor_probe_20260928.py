"""One consumed-tune comparison of coupled speed and steering control."""

import json
from pathlib import Path
import time

import torch

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.speed_coupled_corridor_runtime import SpeedCoupledCorridorAgent
from research.score_probe_20260928 import CHECKPOINT, ROOT, compact, make_agent, sha256, summary
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/speed-coupled-corridor-probe-20260928"


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "coupled steering and speed activation probe",
        "split": "consumed_tune", "cells": [{"track_id": 1, "seed": 184}],
        "max_decisions": 2000,
        "source_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_sha256": sha256(ROOT / "haic_agent/speed_coupled_corridor_runtime.py"),
        "corridor_source_sha256": sha256(ROOT / "haic_agent/corridor_agent.py"),
        "baseline_source_sha256": sha256(ROOT / "haic_agent/obstacle_full_road_guard_runtime.py"),
        "checkpoint_sha256": sha256(CHECKPOINT), "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    agents = {"full_road_guard": ObstacleFullRoadGuardAgent(make_agent(saved, 1.0)),
              "speed_coupled_corridor": SpeedCoupledCorridorAgent()}
    rows = {name: [] for name in agents}
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for name, agent in agents.items():
            result = run_episode(mode=name, track_id=1, seed=184,
                agent=agent, max_decisions=2000, plan_budget_seconds=4.5)
            row = compact(result)
            row["max_speed"] = result.get("max_speed")
            row["mean_speed"] = result.get("mean_speed")
            rows[name].append(row)
            events.write(json.dumps({"type": "episode", "arm": name, **row}) + "\n")
            print(name, json.dumps(row), flush=True)
    summaries = {name: summary(values) for name, values in rows.items()}
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY", "summaries": summaries,
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "UNKNOWN",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
