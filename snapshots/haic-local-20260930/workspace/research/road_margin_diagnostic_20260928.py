"""Measure how often the strict sprint gate rejects visible road margin.

Offline diagnostic on an already consumed tune cell. Simulator state is not
used by the candidate; this script only summarizes its pixel observations.
"""

import json
from pathlib import Path
import time

import numpy as np
import torch

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.pixel_features import ROAD_HIGH, ROAD_LOW, current_frame, road_centers
from research.score_probe_20260928 import CHECKPOINT, ROOT, compact, make_agent, sha256
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/road-margin-diagnostic-20260928"


class MarginDiagnosticAgent(ObstacleFullRoadGuardAgent):
    def __init__(self, base):
        super().__init__(base)
        self.samples = []

    def reset(self, observation):
        super().reset(observation)
        self.samples = []

    def act(self, observation):
        action = super().act(observation)
        frame = current_frame(observation)
        centers = road_centers(frame)
        diag = self.corridor.last_step_diagnostics()
        if len(centers) == 7 and diag["obstacle_y"] is None:
            values = list(centers.values())
            asphalt = (frame[54] >= ROAD_LOW) & (frame[54] <= ROAD_HIGH)
            self.samples.append({
                "max_center_offset": max(abs(value - 42.0) for value in values),
                "center_range": max(values) - min(values),
                "near_road_center": centers[54],
                "center_pixel_asphalt": bool(asphalt[42]),
                "near_asphalt_count": int(np.count_nonzero(asphalt[25:60])),
                "pixel_speed": float(diag["pixel_speed"]),
            })
        return action


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name,
        "purpose": "read-only pixel margin diagnostic; no candidate comparison",
        "split": "consumed_tune", "cells": [{"track_id": 1, "seed": 184}],
        "source_sha256": sha256(Path(__file__).resolve()),
        "baseline_source_sha256": sha256(ROOT / "haic_agent/obstacle_full_road_guard_runtime.py"),
        "checkpoint_sha256": sha256(CHECKPOINT), "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    agent = MarginDiagnosticAgent(make_agent(saved, 1.0))
    episode = compact(run_episode(mode="road_margin_diagnostic", track_id=1,
        seed=184, agent=agent, max_decisions=2000, plan_budget_seconds=4.5))
    samples = agent.samples
    strict_rejects = [x for x in samples if x["max_center_offset"] > 2.5]
    centered_road_rejects = [x for x in strict_rejects if x["center_pixel_asphalt"]]
    result = {
        "episode": episode,
        "full_road_no_obstacle_frames": len(samples),
        "strict_offset_rejected_frames": len(strict_rejects),
        "strict_offset_rejected_with_asphalt_at_screen_center": len(centered_road_rejects),
        "median_near_asphalt_pixels": float(np.median([x["near_asphalt_count"] for x in samples])) if samples else None,
        "median_rejected_center_offset": float(np.median([x["max_center_offset"] for x in strict_rejects])) if strict_rejects else None,
    }
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        events.write(json.dumps({"type": "diagnostic", **result}) + "\n")
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY", "result": result,
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "NOT_APPLICABLE",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"},
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
