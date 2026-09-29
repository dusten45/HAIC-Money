"""Pixel-action trace for the consumed tune near-finish failure."""

import json
from pathlib import Path
import time

from haic_agent.fast_policy_safety_runtime import FastPolicySafetyAgent
from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.pixel_features import current_frame, estimate_observation_speed, road_centers
from research.historical_speed_policy_nested_probe_20260928 import ARMS, ROOT, load_historical_agent
from research.score_probe_20260928 import compact, sha256
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/fast-policy-safety-finish-trace-20260928"


class PixelTraceAgent(FastPolicySafetyAgent):
    def __init__(self, fast, stable):
        super().__init__(fast, stable)
        self.pixel_steps = []

    def reset(self, observation):
        super().reset(observation)
        self.pixel_steps = []

    def act(self, observation):
        frame = current_frame(observation)
        centers = road_centers(frame)
        speed = estimate_observation_speed(observation, frame)
        action = super().act(observation)
        self.pixel_steps.append({
            "step": len(self.pixel_steps) + 1,
            "speed_pixel": speed,
            "road_rows": len(centers),
            "near_offset": centers.get(54, 42.0) - 42.0 if 54 in centers else None,
            "far_offset": centers.get(30, 42.0) - 42.0 if 30 in centers else None,
            "protecting": self.protecting,
            "steer": float(action[0]), "gas": float(action[1]), "brake": float(action[2]),
        })
        return action


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "pixel-only diagnosis of consumed tune near-finish DNF",
        "split": "consumed_tune", "cell": {"track_id": 3, "seed": 262},
        "source_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_sha256": sha256(ROOT / "haic_agent/fast_policy_safety_runtime.py"),
        "fast_checkpoint_sha256": sha256(ARMS["speed_seed8104"]),
        "stable_checkpoint_sha256": sha256(ARMS["selected_control"]),
        "max_decisions": 2000, "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    agent = PixelTraceAgent(
        ObstacleFullRoadGuardAgent(load_historical_agent(ARMS["speed_seed8104"])),
        ObstacleFullRoadGuardAgent(load_historical_agent(ARMS["selected_control"])),
    )
    result = run_episode(mode="pixel_trace", track_id=3, seed=262,
        agent=agent, max_decisions=2000, plan_budget_seconds=4.5, capture_trace=True)
    trace = result["decision_trace"]
    for pixel, item in zip(agent.pixel_steps, trace):
        pixel["progress"] = item["progress"]
    progress = [item["progress"] for item in agent.pixel_steps]
    last_advance = max((index for index in range(1, len(progress))
                        if progress[index] > progress[index - 1]), default=None)
    summary = {"result": compact(result), "last_advance_step": None if last_advance is None else last_advance + 1,
               "last_80_pixel_steps": agent.pixel_steps[-80:]}
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        events.write(json.dumps({"type": "pixel_trace_summary", **summary}) + "\n")
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY",
        "result": summary["result"], "last_advance_step": summary["last_advance_step"],
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "PASS",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"},
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": summary["result"], "last_advance_step": summary["last_advance_step"],
                      "last_20_pixel_steps": agent.pixel_steps[-20:]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
