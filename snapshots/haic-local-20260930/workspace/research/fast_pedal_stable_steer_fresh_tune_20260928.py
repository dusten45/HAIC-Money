"""Preregistered matched three-track tune of a fast-pedal/stable-steering policy."""

import json
from pathlib import Path
import time

from haic_agent.fast_pedal_stable_steer_runtime import FastPedalStableSteerAgent
from research.fast_pedal_stable_steer_consumed_tune_20260928 import guarded
from research.historical_speed_policy_nested_probe_20260928 import ARMS, ROOT
from research.score_probe_20260928 import compact, sha256, summary
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/score-linux-fast-pedal-stable-steer-fresh-tune-20260928"
CELLS = tuple((track, seed) for track in (1, 2, 3) for seed in (272, 273, 274, 275))


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "matched fresh three-track fast-pedal stable-steering tune",
        "split": "tune", "cells": [{"track_id": t, "seed": s} for t, s in CELLS],
        "control_checkpoint_sha256": sha256(ARMS["selected_control"]),
        "fast_checkpoint_sha256": sha256(ARMS["speed_seed8104"]),
        "source_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_sha256": sha256(ROOT / "haic_agent/fast_pedal_stable_steer_runtime.py"),
        "protocol": "Linux Python 3.11 CPU; 2000 decisions; finish_time_s completion; alternating arm order",
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
                print(name, track, seed, row["completed"], row["lapTimeMs"],
                      row["progress"], row["fast_pedal_decisions"], flush=True)
    control = summary(rows["selected_control"])
    candidate = summary(rows["fast_pedal_stable_steer"])
    activated = any(r["fast_pedal_decisions"] for r in rows["fast_pedal_stable_steer"])
    no_invalid = all(r["invalid_actions"] == 0 for values in rows.values() for r in values)
    advances = activated and no_invalid and candidate["completed"] >= control["completed"]
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name, "decision": "ADVANCE_TO_HELD_OUT" if advances else "REJECT_ON_TUNE",
        "summaries": {"selected_control": control, "fast_pedal_stable_steer": candidate},
        "gates": {"rule_compliance": "UNKNOWN",
                  "mechanism_activation": "PASS" if activated else "FAIL",
                  "competitive_or_product_outcome": "PASS" if advances else "FAIL"},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
