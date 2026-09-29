"""Trace a failed TRAIN-only teacher route to identify steering geometry errors."""

import json
from pathlib import Path
import time

from research.score_probe_20260928 import ROOT, sha256
from training.env_factory import create_training_environment
from training.evaluate_closed_loop import run_episode
from training.fast_route_teacher import FastRouteTeacher


RUN = ROOT / "runs/haic-research-v2/fast-route-teacher-train-trace-20260928"


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "train-only teacher steering diagnosis",
        "split": "train", "cell": {"track_id": 1, "seed": 43},
        "max_decisions": 2000,
        "source_sha256": sha256(Path(__file__).resolve()),
        "teacher_source_sha256": sha256(ROOT / "training/fast_route_teacher.py"),
        "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    holder = {}

    def factory(**kwargs):
        holder["environment"] = create_training_environment(**kwargs)
        return holder["environment"]

    teacher = FastRouteTeacher(lambda: holder["environment"])
    result = run_episode(mode="train_teacher_trace", track_id=1, seed=43,
        agent=teacher, max_decisions=2000, plan_budget_seconds=4.5,
        environment_factory=factory, capture_trace=True)
    track = holder["environment"].unwrapped.track
    trace = result["decision_trace"]
    rows = []
    for item in trace:
        x, y = item["car_x"], item["car_y"]
        nearest = min(range(len(track)), key=lambda i: (track[i][2] - x) ** 2 + (track[i][3] - y) ** 2)
        distance = ((track[nearest][2] - x) ** 2 + (track[nearest][3] - y) ** 2) ** 0.5
        if item["step"] <= 20 or item["step"] % 10 == 0 or item["step"] > len(trace) - 12:
            rows.append({key: item[key] for key in ("step", "progress", "speed", "steer", "gas", "brake", "car_yaw")}
                        | {"nearest_index": nearest, "road_distance": round(distance, 2),
                           "road_beta": round(float(track[nearest][1]), 3)})
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        events.write(json.dumps({"type": "teacher_trace", "completed": result["completed"],
                                 "progress": result["progress"], "rows": rows}) + "\n")
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY", "completed": result["completed"],
        "progress": result["progress"], "rows": rows,
        "gates": {"rule_compliance": "NOT_APPLICABLE", "mechanism_activation": "PASS",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"},
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows[:20] + rows[-15:], indent=2), flush=True)


if __name__ == "__main__":
    main()
