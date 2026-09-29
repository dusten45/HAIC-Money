"""Read-only checkpoint reuse diagnostic on consumed tune cells."""

import json
from pathlib import Path
import time

from agent import Agent
from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from research.score_probe_20260928 import CHECKPOINT, ROOT, compact, sha256, summary
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/historical-speed-policy-consumed-tune-20260928"
CELLS = ((1, 184), (1, 228), (3, 228), (1, 230))
OLD = ROOT / "artifacts/haic/ppo-speed-shortfall-0p6-vs-0p0-kl0p5-u8-20260924T133743Z/run"
ARMS = {
    "selected_control": CHECKPOINT,
    "speed_seed8104": OLD / "seed8104/penalty_0p6/policy.pt",
    "speed_seed8105": OLD / "seed8105/penalty_0p6/policy.pt",
}


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name,
        "purpose": "diagnose historic speed-trained actors behind common obstacle guard",
        "split": "consumed_tune",
        "cells": [{"track_id": track, "seed": seed} for track, seed in CELLS],
        "arms": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                 for name, path in ARMS.items()},
        "max_decisions": 2000,
        "source_sha256": sha256(Path(__file__).resolve()),
        "guard_source_sha256": sha256(ROOT / "haic_agent/obstacle_full_road_guard_runtime.py"),
        "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    rows = {name: [] for name in ARMS}
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for track, seed in CELLS:
            for name, path in ARMS.items():
                base = Agent(policy_checkpoint=str(path), planner_enabled=False,
                             strict_checkpoint_loading=True)
                agent = ObstacleFullRoadGuardAgent(base)
                result = run_episode(mode=name, track_id=track, seed=seed,
                    agent=agent, max_decisions=2000, plan_budget_seconds=4.5)
                row = compact(result)
                row["mean_speed"] = result.get("mean_speed")
                row["max_speed"] = result.get("max_speed")
                rows[name].append(row)
                events.write(json.dumps({"type": "episode", "arm": name, **row}) + "\n")
                events.flush()
                print(name, track, seed, row["completed"], row["lapTimeMs"],
                      row["progress"], row["mean_speed"], flush=True)
    (RUN / "integration_report.json").write_text(json.dumps({
        "run_id": RUN.name,
        "decision": "DIAGNOSTIC_ONLY",
        "summaries": {name: summary(values) for name, values in rows.items()},
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "UNKNOWN",
                  "competitive_or_product_outcome": "NOT_APPLICABLE"},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
