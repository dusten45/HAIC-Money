"""Compare motion and final progress of margin pedal versus baseline on consumed tune."""

import json
from pathlib import Path
import statistics
import time

import torch

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.road_margin_throttle_runtime import RoadMarginThrottleAgent
from research.score_probe_20260928 import CHECKPOINT, ROOT, compact, make_agent, sha256
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/road-margin-throttle-trace-20260928"


def summarize(name, result):
    trace = result["decision_trace"] or []
    speeds = [item["speed"] for item in trace]
    progress = [item["progress"] for item in trace]
    last_advance = max((i + 1 for i in range(1, len(progress))
                        if progress[i] > progress[i - 1]), default=None)
    return {
        "arm": name, **compact(result),
        "max_speed": max(speeds) if speeds else None,
        "median_speed": statistics.median(speeds) if speeds else None,
        "gas_full_steps": sum(item["gas"] >= 0.99 for item in trace),
        "last_progress_advance_step": last_advance,
        "last_10": [{key: item[key] for key in ("step", "progress", "speed", "gas", "brake")}
                    for item in trace[-10:]],
    }


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "mechanism failure trace; not an outcome comparison",
        "split": "consumed_tune", "cells": [{"track_id": 1, "seed": 184}],
        "source_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_sha256": sha256(ROOT / "haic_agent/road_margin_throttle_runtime.py"),
        "checkpoint_sha256": sha256(CHECKPOINT), "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    arms = {"full_road_guard": ObstacleFullRoadGuardAgent(make_agent(saved, 1.0)),
            "road_margin_throttle": RoadMarginThrottleAgent(make_agent(saved, 1.0))}
    rows = []
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for name, agent in arms.items():
            row = summarize(name, run_episode(mode=name, track_id=1, seed=184,
                agent=agent, max_decisions=2000, plan_budget_seconds=4.5, capture_trace=True))
            rows.append(row)
            events.write(json.dumps({"type": "trace_summary", **row}) + "\n")
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY", "rows": rows,
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "PASS",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"},
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
