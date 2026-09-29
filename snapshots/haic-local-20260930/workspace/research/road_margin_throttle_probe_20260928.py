"""Activation probe for grass-tolerant pedal override on consumed tune cell."""

import json
from pathlib import Path
import time

import torch

from haic_agent.road_margin_throttle_runtime import RoadMarginThrottleAgent
from research.score_probe_20260928 import CHECKPOINT, ROOT, compact, make_agent, sha256
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/road-margin-throttle-probe-20260928"


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "activation and finish diagnostic only",
        "split": "consumed_tune", "cells": [{"track_id": 1, "seed": 184}],
        "max_decisions": 2000,
        "source_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_sha256": sha256(ROOT / "haic_agent/road_margin_throttle_runtime.py"),
        "checkpoint_sha256": sha256(CHECKPOINT), "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    agent = RoadMarginThrottleAgent(make_agent(saved, 1.0))
    row = compact(run_episode(mode="road_margin_throttle_probe", track_id=1,
        seed=184, agent=agent, max_decisions=2000, plan_budget_seconds=4.5))
    row["margin_throttle_count"] = agent.margin_throttle_count
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        events.write(json.dumps({"type": "episode", **row}) + "\n")
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY", "result": row,
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "PASS" if agent.margin_throttle_count else "FAIL",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"},
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
