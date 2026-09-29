"""Probe far-road steering preview on consumed tune cells."""

import json
from pathlib import Path
import time

import torch

from haic_agent.lookahead_corridor_runtime import LookaheadFallbackAgent
from research.score_probe_20260928 import CHECKPOINT, ROOT, compact, make_agent, sha256
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/lookahead-fallback-probe-20260928"
CELLS = ((3, 228), (1, 228), (1, 184), (1, 230))


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "far-road steering preview activation diagnostic",
        "split": "consumed_tune", "cells": [{"track_id": t, "seed": s} for t, s in CELLS],
        "max_decisions": 2000,
        "source_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_sha256": sha256(ROOT / "haic_agent/lookahead_corridor_runtime.py"),
        "checkpoint_sha256": sha256(CHECKPOINT), "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    rows = []
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for track, seed in CELLS:
            agent = LookaheadFallbackAgent(make_agent(saved, 1.0))
            result = run_episode(mode="lookahead_fallback", track_id=track,
                seed=seed, agent=agent, max_decisions=2000, plan_budget_seconds=4.5)
            row = compact(result)
            row["max_speed"] = result.get("max_speed")
            row["coupled_count"] = agent.coupled_count
            row["fallback_count"] = agent.fallback_count
            row["lookahead_count"] = agent.corridor.lookahead_count
            rows.append(row)
            events.write(json.dumps({"type": "episode", **row}) + "\n")
            events.flush()
            print(json.dumps(row), flush=True)
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY", "rows": rows,
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "PASS"
                  if any(row["lookahead_count"] for row in rows) else "FAIL",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
