"""Measure one TRAIN-only privileged teacher lap; not candidate evidence."""

import json
from pathlib import Path
import time

from research.score_probe_20260928 import ROOT, compact, sha256
from training.env_factory import create_training_environment
from training.evaluate_closed_loop import run_episode
from training.fast_route_teacher import FastRouteTeacher


RUN = ROOT / "runs/haic-research-v2/fast-route-teacher-train-probe-20260928"
CELLS = ((1, 43), (2, 102))


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name,
        "purpose": "train-only privileged teacher feasibility; never submission-candidate rate",
        "split": "train",
        "cells": [{"track_id": t, "seed": s} for t, s in CELLS],
        "max_decisions": 2000,
        "source_sha256": sha256(Path(__file__).resolve()),
        "teacher_source_sha256": sha256(ROOT / "training/fast_route_teacher.py"),
        "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    rows = []
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for track, seed in CELLS:
            holder = {}

            def factory(**kwargs):
                holder["environment"] = create_training_environment(**kwargs)
                return holder["environment"]

            teacher = FastRouteTeacher(lambda: holder["environment"])
            result = run_episode(mode="train_teacher", track_id=track, seed=seed,
                agent=teacher, max_decisions=2000, plan_budget_seconds=4.5,
                environment_factory=factory, capture_trace=False)
            row = compact(result)
            row.update(max_speed=teacher.max_speed, brake_decisions=teacher.brake_decisions,
                       obstacle_decisions=teacher.obstacle_decisions)
            rows.append(row)
            events.write(json.dumps({"type": "teacher_episode", **row}) + "\n")
            events.flush()
            print(json.dumps(row), flush=True)
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "TEACHER_FEASIBILITY_ONLY", "rows": rows,
        "gates": {"rule_compliance": "NOT_APPLICABLE", "mechanism_activation": "PASS",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
