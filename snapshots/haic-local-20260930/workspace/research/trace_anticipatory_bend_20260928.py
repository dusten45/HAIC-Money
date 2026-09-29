"""Record pixel cues and action changes on a consumed tune failure."""

import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from haic_agent.anticipatory_bend_runtime import AnticipatoryBendAgent
from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.pixel_features import current_frame, estimate_observation_speed, road_centers
from research.score_probe_20260928 import compact, make_agent
from training.evaluate_closed_loop import run_episode


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts/haic-research-v2/full-road-guard-candidate-20260928/submission-full-road-guard.zip"
RUN = ROOT / "runs/haic-research-v2/anticipatory-bend-trace-20260928"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BaseSpy:
    def __init__(self, base):
        self.base = base
        self.corridor = base.corridor
        self.last_action = None

    def reset(self, observation):
        self.base.reset(observation)
        self.last_action = None

    def act(self, observation):
        self.last_action = np.asarray(self.base.act(observation), dtype=np.float32).copy()
        return self.last_action


class TracingAgent:
    def __init__(self, candidate, spy, events):
        self.candidate = candidate
        self.spy = spy
        self.events = events
        self.step = 0

    def reset(self, observation):
        self.candidate.reset(observation)
        self.step = 0

    def act(self, observation):
        frame = current_frame(observation)
        centers = road_centers(frame)
        speed = estimate_observation_speed(observation, frame)
        count_before = self.candidate.preview_count
        action = self.candidate.act(observation)
        row = {
            "type": "step", "step": self.step,
            "road_centers": len(centers),
            "near": centers.get(54), "middle": centers.get(42), "far": centers.get(30),
            "pixel_speed": speed,
            "obstacle_y": self.spy.corridor.last_step_diagnostics()["obstacle_y"],
            "base_action": self.spy.last_action.tolist(),
            "candidate_action": action.tolist(),
            "preview_active": self.candidate.preview_count > count_before,
        }
        self.events.write(json.dumps(row) + "\n")
        self.events.flush()
        self.step += 1
        return action


def main():
    with zipfile.ZipFile(PACKAGE) as archive:
        saved = torch.load(io.BytesIO(archive.read("policy.pt")), map_location="cpu", weights_only=True)
    RUN.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": RUN.name,
        "purpose": "trace anticipatory bend mechanism on a previously consumed failure cell",
        "split": "consumed_tune", "track_id": 3, "seed": 228,
        "max_decisions": 2000,
        "control_package_sha256": _sha256(PACKAGE),
        "candidate_source_sha256": _sha256(ROOT / "haic_agent/anticipatory_bend_runtime.py"),
        "trace_source_sha256": _sha256(Path(__file__)),
        "created_unix": time.time(),
    }
    (RUN / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        spy = BaseSpy(ObstacleFullRoadGuardAgent(make_agent(saved, 1.0)))
        candidate = AnticipatoryBendAgent(spy)
        agent = TracingAgent(candidate, spy, events)
        result = run_episode(mode="diagnostic", track_id=3, seed=228, agent=agent,
                             max_decisions=2000, plan_budget_seconds=4.5)
        events.write(json.dumps({"type": "episode", **compact(result)}) + "\n")
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY",
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "PASS" if candidate.preview_count else "FAIL",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"},
        "preview_count": candidate.preview_count,
        "episode": compact(result),
    }, indent=2) + "\n", encoding="utf-8")
    print(compact(result), "preview_count", candidate.preview_count)


if __name__ == "__main__":
    main()
