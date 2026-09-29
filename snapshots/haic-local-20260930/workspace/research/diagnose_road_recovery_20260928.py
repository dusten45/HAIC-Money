"""Inspect pixel road geometry during a consumed tune failure."""

import json
from pathlib import Path
import time

import torch

from haic_agent.obstacle_commit_late_runtime import ObstacleCommitLateAgent
from haic_agent.pixel_features import current_frame, road_centers
from research.score_probe_20260928 import CHECKPOINT, ROOT, make_agent, sha256
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/diagnose-road-recovery-20260928"


class DiagnosticRoadRecovery(ObstacleCommitLateAgent):
    def act(self, observation):
        action = super().act(observation)
        centers = road_centers(current_frame(observation))
        self._road_diag = {
            "road_centers": centers,
            "corridor_steer_base": 0.022 * (centers.get(42, 42.0) - 42.0)
                + 0.018 * (centers.get(42, 42.0) - centers.get(54, 42.0)),
        }
        return action

    def last_step_diagnostics(self):
        return {**self.corridor.last_step_diagnostics(), **self._road_diag,
                "committed_side": self._committed_side}


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "diagnostic pixel geometry on consumed tune cell",
        "split": "tune", "cells": [{"track_id": 2, "seed": 135}],
        "source_sha256": sha256(Path(__file__).resolve()),
        "checkpoint_sha256": sha256(CHECKPOINT), "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    agent = DiagnosticRoadRecovery(make_agent(saved, 1.0))
    result = run_episode(mode="diagnostic", track_id=2, seed=135, agent=agent,
                         max_decisions=2000, plan_budget_seconds=4.5, capture_trace=True)
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        events.write(json.dumps({"type": "trace", "track_id": 2, "seed": 135,
            "completed": result["completed"], "retire_reason": result["retire_reason"],
            "trace": result["decision_trace"]}) + "\n")
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY",
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "UNKNOWN",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"}
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
