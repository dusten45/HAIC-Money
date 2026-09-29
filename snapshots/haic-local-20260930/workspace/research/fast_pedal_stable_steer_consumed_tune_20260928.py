"""Mechanism screen for stable steering with learned fast pedal on consumed tune."""

import json
from pathlib import Path
import time

from haic_agent.fast_pedal_stable_steer_runtime import FastPedalStableSteerAgent
from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from research.historical_speed_policy_nested_probe_20260928 import ARMS, ROOT, load_historical_agent
from research.score_probe_20260928 import compact, sha256, summary
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/fast-pedal-stable-steer-consumed-tune-20260928"
CELLS = ((1, 184), (1, 228), (3, 228), (1, 230))


def guarded(path):
    return ObstacleFullRoadGuardAgent(load_historical_agent(path))


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "mechanism screen on already consumed tune cells",
        "split": "consumed_tune", "cells": [{"track_id": t, "seed": s} for t, s in CELLS],
        "control_checkpoint_sha256": sha256(ARMS["selected_control"]),
        "fast_checkpoint_sha256": sha256(ARMS["speed_seed8104"]),
        "source_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_sha256": sha256(ROOT / "haic_agent/fast_pedal_stable_steer_runtime.py"),
        "max_decisions": 2000, "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    rows = {"selected_control": [], "fast_pedal_stable_steer": []}
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for index, (track, seed) in enumerate(CELLS):
            order = tuple(rows) if index % 2 == 0 else tuple(reversed(tuple(rows)))
            for name in order:
                agent = (guarded(ARMS["selected_control"]) if name == "selected_control" else
                         FastPedalStableSteerAgent(guarded(ARMS["speed_seed8104"]),
                                                    guarded(ARMS["selected_control"])))
                result = run_episode(mode=name, track_id=track, seed=seed,
                    agent=agent, max_decisions=2000, plan_budget_seconds=4.5)
                row = compact(result)
                row["mean_speed"] = result.get("mean_speed")
                row["max_speed"] = result.get("max_speed")
                row["fast_pedal_decisions"] = getattr(agent, "fast_pedal_decisions", None)
                row["stable_decisions"] = getattr(agent, "stable_decisions", None)
                rows[name].append(row)
                events.write(json.dumps({"type": "episode", "arm": name, **row}) + "\n")
                events.flush()
                print(name, track, seed, row["completed"], row["lapTimeMs"], row["progress"],
                      row["fast_pedal_decisions"], flush=True)
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "MECHANISM_SCREEN",
        "summaries": {name: summary(values) for name, values in rows.items()},
        "gates": {"rule_compliance": "UNKNOWN",
                  "mechanism_activation": "PASS" if any(r["fast_pedal_decisions"] for r in rows["fast_pedal_stable_steer"]) else "FAIL",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
