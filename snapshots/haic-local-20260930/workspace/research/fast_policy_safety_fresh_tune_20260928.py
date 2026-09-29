"""Matched three-track tune comparison of a fixed fast/stable pixel policy."""

import json
from pathlib import Path
import time

from haic_agent.fast_policy_safety_runtime import FastPolicySafetyAgent
from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from research.historical_speed_policy_nested_probe_20260928 import ARMS, ROOT, load_historical_agent
from research.score_probe_20260928 import compact, sha256, summary
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/score-linux-fast-policy-safety-fresh-tune-20260928"
CELLS = tuple((track, seed) for track in (1, 2, 3) for seed in (260, 261, 262, 263))


def guarded(path):
    return ObstacleFullRoadGuardAgent(load_historical_agent(path))


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "matched fresh three-track fast-policy safety comparison",
        "split": "tune",
        "cells": [{"track_id": t, "seed": s} for t, s in CELLS],
        "control_checkpoint_sha256": sha256(ARMS["selected_control"]),
        "fast_checkpoint_sha256": sha256(ARMS["speed_seed8104"]),
        "source_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_sha256": sha256(ROOT / "haic_agent/fast_policy_safety_runtime.py"),
        "max_decisions": 2000, "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    rows = {"selected_control": [], "safety_switch": []}
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for index, (track, seed) in enumerate(CELLS):
            order = tuple(rows) if index % 2 == 0 else tuple(reversed(tuple(rows)))
            for name in order:
                if name == "selected_control":
                    agent = guarded(ARMS["selected_control"])
                else:
                    agent = FastPolicySafetyAgent(guarded(ARMS["speed_seed8104"]),
                                                  guarded(ARMS["selected_control"]))
                result = run_episode(mode=name, track_id=track, seed=seed,
                    agent=agent, max_decisions=2000, plan_budget_seconds=4.5)
                row = compact(result)
                row["mean_speed"] = result.get("mean_speed")
                row["max_speed"] = result.get("max_speed")
                row["fast_decisions"] = getattr(agent, "fast_decisions", None)
                row["protected_decisions"] = getattr(agent, "protected_decisions", None)
                row["protection_entries"] = getattr(agent, "protection_entries", None)
                rows[name].append(row)
                events.write(json.dumps({"type": "episode", "arm": name, **row}) + "\n")
                events.flush()
                print(name, track, seed, row["completed"], row["lapTimeMs"],
                      row["progress"], row["fast_decisions"], row["protected_decisions"], flush=True)
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "TUNE_SCREEN",
        "summaries": {name: summary(values) for name, values in rows.items()},
        "gates": {"rule_compliance": "UNKNOWN",
                  "mechanism_activation": "PASS" if any(r["protected_decisions"] for r in rows["safety_switch"]) else "UNKNOWN",
                  "competitive_or_product_outcome": "UNKNOWN"},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
