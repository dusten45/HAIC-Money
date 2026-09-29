"""Measure pixel-estimated and simulator speed at existing preview events."""

import hashlib
import io
import json
from pathlib import Path
import statistics
import time
import zipfile

import torch

from haic_agent.anticipatory_bend_runtime import AnticipatoryBendAgent
from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.pixel_features import current_frame, estimate_observation_speed, road_centers
from research.score_probe_20260928 import make_agent
from training.evaluate_closed_loop import run_episode


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts/haic-research-v2/anticipatory-bend-candidate-20260928/submission-anticipatory-bend.zip"
RUN = ROOT / "runs/haic-research-v2/anticipatory-preview-speed-observation-20260928"
CELLS = ((2, 1700), (3, 1701))


class Observer:
    def __init__(self, controller):
        self.controller = controller
        self.preview_events = []
        self.step = 0

    def reset(self, observation):
        self.controller.reset(observation)
        self.preview_events = []
        self.step = 0

    def act(self, observation):
        before = self.controller.preview_count
        action = self.controller.act(observation)
        self.step += 1
        if self.controller.preview_count > before:
            frame = current_frame(observation)
            centers = road_centers(frame)
            self.preview_events.append({
                "step": self.step, "pixel_speed": estimate_observation_speed(observation, frame),
                "steer": float(action[0]), "gas": float(action[1]),
                "bend": centers[30] - centers[54], "near": centers[54],
            })
        return action


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(PACKAGE) as z:
        saved = torch.load(io.BytesIO(z.read("policy.pt")), map_location="cpu", weights_only=True)
    manifest = {"run_id": RUN.name, "split": "consumed_tune_diagnostic", "cells": list(CELLS), "package_sha256": sha(PACKAGE), "probe_sha256": sha(Path(__file__)), "created_unix": time.time()}
    (RUN / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows = []
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for track, seed in CELLS:
            driver = Observer(AnticipatoryBendAgent(ObstacleFullRoadGuardAgent(make_agent(saved, 1.0))))
            raw = run_episode(mode="preview_observation", track_id=track, seed=seed, agent=driver, max_decisions=2000, plan_budget_seconds=4.5)
            row = {"track_id": track, "seed": seed, "completed": raw["completed"], "lapTimeMs": raw["lapTimeMs"], "preview_events": driver.preview_events}
            rows.append(row)
            events.write(json.dumps({"type": "episode", **row}) + "\n")
            events.flush()
    speeds = [event["pixel_speed"] for row in rows for event in row["preview_events"]]
    report = {"run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY", "preview_events": len(speeds), "min_pixel_speed": min(speeds), "median_pixel_speed": statistics.median(speeds), "max_pixel_speed": max(speeds), "at_or_below_34": sum(speed <= 34 for speed in speeds)}
    (RUN / "integration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
