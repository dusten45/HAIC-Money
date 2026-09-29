"""Frozen matched comparison of grass-tolerant acceleration with a speed cap."""

import json
from pathlib import Path
import sys
import time

import torch

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.road_margin_speed_cap_runtime import RoadMarginSpeedCapAgent
from research.score_probe_20260928 import CHECKPOINT, ROOT, compact, make_agent, sha256, summary
from training.evaluate_closed_loop import run_episode


PROTOCOLS = {
    "consumed_tune": (((1, 184),), "score-linux-road-margin-speed-cap-consumed-tune-20260928"),
    "tune": (tuple((track, seed) for track in (1, 2, 3) for seed in (228, 229, 230, 231)),
             "score-linux-road-margin-speed-cap-tune-20260928"),
    "held_out": (tuple((track, seed) for track in (1, 2, 3) for seed in (232, 233, 234, 235)),
                 "score-linux-road-margin-speed-cap-heldout-20260928"),
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in PROTOCOLS:
        raise SystemExit("usage: road_margin_speed_cap_compare_20260928 {consumed_tune|tune|held_out}")
    split = sys.argv[1]
    cells, run_id = PROTOCOLS[split]
    run = ROOT / "runs/haic-research-v2" / run_id
    run.mkdir(parents=True, exist_ok=False)
    (run / "run_manifest.json").write_text(json.dumps({
        "run_id": run_id, "purpose": "matched grass-margin acceleration with early speed cap",
        "split": split, "cells": [{"track_id": t, "seed": s} for t, s in cells],
        "max_decisions": 2000, "one_cpu_process": True,
        "source_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_sha256": sha256(ROOT / "haic_agent/road_margin_speed_cap_runtime.py"),
        "margin_source_sha256": sha256(ROOT / "haic_agent/road_margin_throttle_runtime.py"),
        "baseline_source_sha256": sha256(ROOT / "haic_agent/obstacle_full_road_guard_runtime.py"),
        "checkpoint_sha256": sha256(CHECKPOINT), "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    agents = {"full_road_guard": ObstacleFullRoadGuardAgent(make_agent(saved, 1.0)),
              "road_margin_speed_cap": RoadMarginSpeedCapAgent(make_agent(saved, 1.0))}
    rows = {name: [] for name in agents}
    with (run / "events.jsonl").open("x", encoding="utf-8") as events:
        for index, (track, seed) in enumerate(cells):
            order = tuple(agents) if index % 2 == 0 else tuple(reversed(tuple(agents)))
            for name in order:
                agent = agents[name]
                row = compact(run_episode(mode=name, track_id=track, seed=seed,
                    agent=agent, max_decisions=2000, plan_budget_seconds=4.5))
                if name == "road_margin_speed_cap":
                    row["boost_count"] = agent.margin_throttle_count
                rows[name].append(row)
                events.write(json.dumps({"type": "episode", "arm": name, **row}) + "\n")
                events.flush()
                print(name, track, seed, row["completed"], row["lapTimeMs"],
                      row["progress"], row.get("boost_count"), flush=True)
    summaries = {name: summary(values) for name, values in rows.items()}
    (run / "integration_report.json").write_text(json.dumps({
        "run_id": run_id, "summaries": summaries, "decision": "DIAGNOSTIC_ONLY",
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "PASS"
                  if any(row.get("boost_count", 0) for row in rows["road_margin_speed_cap"]) else "FAIL",
                  "competitive_or_product_outcome": "UNKNOWN"},
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
