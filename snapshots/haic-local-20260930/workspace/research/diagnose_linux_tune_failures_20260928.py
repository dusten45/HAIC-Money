"""Trace fixed candidate failures on already consumed tune cells in Linux."""

import json
from pathlib import Path
import time

import torch

from haic_agent.hybrid_runtime import ObstacleHybridAgent
from research.score_probe_20260928 import CHECKPOINT, ROOT, make_agent, sha256
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/diagnose-linux-tune-failures-20260928"
CELLS = ((2, 132), (2, 135))


class DiagnosticHybrid(ObstacleHybridAgent):
    def last_step_diagnostics(self):
        values = self.corridor.last_step_diagnostics()
        return {**values, "override": values["obstacle_y"] is not None and values["obstacle_y"] >= 28.0}


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "read-only failure tracing on consumed tune cells",
        "cells": [{"track_id": track, "seed": seed} for track, seed in CELLS],
        "source_sha256": sha256(Path(__file__).resolve()),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "split": "tune", "max_decisions": 2000, "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for track, seed in CELLS:
            agent = DiagnosticHybrid(make_agent(saved, 1.0))
            result = run_episode(mode="hybrid_trace", track_id=track, seed=seed,
                                 agent=agent, max_decisions=2000, plan_budget_seconds=4.5,
                                 capture_trace=True)
            events.write(json.dumps({"type": "trace", "track_id": track, "seed": seed,
                "completed": result["completed"], "progress": result["progress"],
                "retire_reason": result["retire_reason"], "trace": result["decision_trace"]}) + "\n")
            events.flush()
            print(track, seed, result["completed"], result["progress"], result["retire_reason"], result["steps"], flush=True)
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY",
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "UNKNOWN",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"}
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
