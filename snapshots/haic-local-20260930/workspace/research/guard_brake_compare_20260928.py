"""Fixed comparisons for high-speed obstacle braking."""

import json
from pathlib import Path
import sys
import time

import torch

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.obstacle_guard_brake_runtime import ObstacleGuardBrakeAgent
from research.score_probe_20260928 import CHECKPOINT, ROOT, compact, make_agent, sha256, summary
from training.evaluate_closed_loop import run_episode


PROTOCOLS = {
    "consumed_tune": ("tune", tuple((t, s) for t in (1, 2) for s in (184, 185, 186, 187)),
                      "score-linux-guard-brake-consumed-tune-20260928"),
    "fresh_tune": ("tune", tuple((t, s) for t in (1, 2) for s in (196, 197, 198, 199)),
                   "score-linux-guard-brake-fresh-tune-20260928"),
    "held_out": ("held_out", tuple((t, s) for t in (1, 2) for s in (200, 201, 202, 203)),
                 "score-linux-guard-brake-fresh-heldout-20260928"),
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in PROTOCOLS:
        raise SystemExit("usage: guard_brake_compare_20260928 {consumed_tune|fresh_tune|held_out}")
    protocol = sys.argv[1]
    split, cells, run_id = PROTOCOLS[protocol]
    run = ROOT / "runs/haic-research-v2" / run_id
    run.mkdir(parents=True, exist_ok=False)
    (run / "run_manifest.json").write_text(json.dumps({
        "run_id": run_id, "purpose": "fixed matched guard-brake comparison",
        "split": split, "cells": [{"track_id": t, "seed": s} for t, s in cells],
        "max_decisions": 2000,
        "source_sha256": sha256(Path(__file__).resolve()),
        "guard_brake_source_sha256": sha256(ROOT / "haic_agent/obstacle_guard_brake_runtime.py"),
        "full_road_guard_source_sha256": sha256(ROOT / "haic_agent/obstacle_full_road_guard_runtime.py"),
        "checkpoint_sha256": sha256(CHECKPOINT), "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    agents = {"full_road_guard": ObstacleFullRoadGuardAgent(make_agent(saved, 1.0)),
              "guard_brake": ObstacleGuardBrakeAgent(make_agent(saved, 1.0))}
    rows = {name: [] for name in agents}
    with (run / "events.jsonl").open("x", encoding="utf-8") as events:
        for index, (track, seed) in enumerate(cells):
            order = tuple(agents) if index % 2 == 0 else tuple(reversed(tuple(agents)))
            for name in order:
                agent = agents[name]
                row = compact(run_episode(mode=name, track_id=track, seed=seed,
                    agent=agent, max_decisions=2000, plan_budget_seconds=4.5))
                if name == "guard_brake":
                    row["guard_brake_count"] = agent.guard_brake_count
                rows[name].append(row)
                events.write(json.dumps({"type": "episode", "arm": name, **row}) + "\n")
                events.flush()
                print(name, track, seed, row["completed"], row["lapTimeMs"], row["progress"], row.get("guard_brake_count"), flush=True)
    summaries = {name: summary(values) for name, values in rows.items()}
    (run / "integration_report.json").write_text(json.dumps({
        "run_id": run_id, "summaries": summaries, "decision": "DIAGNOSTIC_ONLY",
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "UNKNOWN",
                  "competitive_or_product_outcome": "UNKNOWN"}
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
