"""Matched three-track screen of coupled speed and steering control."""

import json
from pathlib import Path
import sys
import time

import torch

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.speed_coupled_corridor_runtime import SpeedCoupledCorridorAgent
from research.score_probe_20260928 import CHECKPOINT, ROOT, compact, make_agent, sha256, summary
from training.evaluate_closed_loop import run_episode


PROTOCOLS = {
    "tune": (tuple((track, seed) for track in (1, 2, 3) for seed in (228, 229, 230, 231)),
             "score-linux-speed-coupled-corridor-tune-20260928"),
    "held_out": (tuple((track, seed) for track in (1, 2, 3) for seed in (232, 233, 234, 235)),
                 "score-linux-speed-coupled-corridor-heldout-20260928"),
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in PROTOCOLS:
        raise SystemExit("usage: speed_coupled_corridor_compare_20260928 {tune|held_out}")
    split = sys.argv[1]
    cells, run_id = PROTOCOLS[split]
    run = ROOT / "runs/haic-research-v2" / run_id
    run.mkdir(parents=True, exist_ok=False)
    (run / "run_manifest.json").write_text(json.dumps({
        "run_id": run_id, "purpose": "matched three-track coupled steering-speed comparison",
        "split": split, "cells": [{"track_id": t, "seed": s} for t, s in cells],
        "max_decisions": 2000, "one_cpu_process": True,
        "source_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_sha256": sha256(ROOT / "haic_agent/speed_coupled_corridor_runtime.py"),
        "corridor_source_sha256": sha256(ROOT / "haic_agent/corridor_agent.py"),
        "baseline_source_sha256": sha256(ROOT / "haic_agent/obstacle_full_road_guard_runtime.py"),
        "checkpoint_sha256": sha256(CHECKPOINT), "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    agents = {"full_road_guard": ObstacleFullRoadGuardAgent(make_agent(saved, 1.0)),
              "speed_coupled_corridor": SpeedCoupledCorridorAgent()}
    rows = {name: [] for name in agents}
    with (run / "events.jsonl").open("x", encoding="utf-8") as events:
        for index, (track, seed) in enumerate(cells):
            order = tuple(agents) if index % 2 == 0 else tuple(reversed(tuple(agents)))
            for name in order:
                result = run_episode(mode=name, track_id=track, seed=seed,
                    agent=agents[name], max_decisions=2000, plan_budget_seconds=4.5)
                row = compact(result)
                row["max_speed"] = result.get("max_speed")
                row["mean_speed"] = result.get("mean_speed")
                rows[name].append(row)
                events.write(json.dumps({"type": "episode", "arm": name, **row}) + "\n")
                events.flush()
                print(name, track, seed, row["completed"], row["lapTimeMs"],
                      row["progress"], row["collisions"], flush=True)
    summaries = {name: summary(values) for name, values in rows.items()}
    (run / "integration_report.json").write_text(json.dumps({
        "run_id": run_id, "summaries": summaries, "decision": "DIAGNOSTIC_ONLY",
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "PASS",
                  "competitive_or_product_outcome": "UNKNOWN"},
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
