"""Inspect the historic faster actor's late failure on a consumed tune cell."""

import json
from pathlib import Path
import time

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from research.historical_speed_policy_nested_probe_20260928 import ARMS, ROOT, load_historical_agent
from research.score_probe_20260928 import sha256
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/historical-speed-seed8104-failure-trace-20260928"


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "consumed tune failure trace; diagnostic only",
        "split": "consumed_tune", "cell": {"track_id": 1, "seed": 230},
        "checkpoint_sha256": sha256(ARMS["speed_seed8104"]),
        "source_sha256": sha256(Path(__file__).resolve()),
        "max_decisions": 2000, "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    agent = ObstacleFullRoadGuardAgent(load_historical_agent(ARMS["speed_seed8104"]))
    result = run_episode(mode="historical_speed_trace", track_id=1, seed=230,
        agent=agent, max_decisions=2000, plan_budget_seconds=4.5, capture_trace=True)
    trace = result["decision_trace"]
    progress = [item["progress"] for item in trace]
    last_advance = max((index for index in range(1, len(progress))
                        if progress[index] > progress[index - 1]), default=None)
    first_stall = None if last_advance is None else last_advance + 1
    excerpt = [{key: item[key] for key in ("step", "progress", "speed", "steer", "gas", "brake",
                                            "car_x", "car_y", "car_yaw")}
               for item in trace[max(0, (first_stall or 0) - 8):min(len(trace), (first_stall or 0) + 12)]]
    summary = {"completed": result["completed"], "progress": result["progress"],
               "steps": result["steps"], "last_advance_step": first_stall,
               "last_12": [{key: item[key] for key in ("step", "progress", "speed", "steer", "gas", "brake")}
                           for item in trace[-12:]], "stall_excerpt": excerpt}
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        events.write(json.dumps({"type": "trace_summary", **summary}) + "\n")
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY", **summary,
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "UNKNOWN",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"},
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
