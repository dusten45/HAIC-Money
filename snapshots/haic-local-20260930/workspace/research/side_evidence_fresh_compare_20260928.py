"""Prospective matched comparison of side evidence and road guard."""

import json
from pathlib import Path
import sys
import time

import torch

from haic_agent.obstacle_road_guard_runtime import ObstacleRoadGuardAgent
from haic_agent.obstacle_side_evidence_runtime import ObstacleSideEvidenceAgent
from research.score_probe_20260928 import CHECKPOINT, ROOT, compact, make_agent, sha256, summary
from training.evaluate_closed_loop import run_episode


PROTOCOLS = {
    "tune": (tuple((track, seed) for track in (1, 2) for seed in (168, 169, 170, 171)),
             "score-linux-side-evidence-fresh-tune-20260928"),
    "held_out": (tuple((track, seed) for track in (1, 2) for seed in (172, 173, 174, 175)),
                 "score-linux-side-evidence-fresh-heldout-20260928"),
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in PROTOCOLS:
        raise SystemExit("usage: side_evidence_fresh_compare_20260928 {tune|held_out}")
    split = sys.argv[1]
    cells, run_id = PROTOCOLS[split]
    run = ROOT / "runs/haic-research-v2" / run_id
    run.mkdir(parents=True, exist_ok=False)
    (run / "run_manifest.json").write_text(json.dumps({
        "run_id": run_id, "purpose": "frozen matched candidate comparison",
        "split": split, "cells": [{"track_id": t, "seed": s} for t, s in cells],
        "max_decisions": 2000,
        "source_sha256": sha256(Path(__file__).resolve()),
        "side_evidence_source_sha256": sha256(ROOT / "haic_agent/obstacle_side_evidence_runtime.py"),
        "road_guard_source_sha256": sha256(ROOT / "haic_agent/obstacle_road_guard_runtime.py"),
        "commit_late_source_sha256": sha256(ROOT / "haic_agent/obstacle_commit_late_runtime.py"),
        "commit_parent_source_sha256": sha256(ROOT / "haic_agent/obstacle_commit_runtime.py"),
        "checkpoint_sha256": sha256(CHECKPOINT), "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    agents = {"road_guard": ObstacleRoadGuardAgent(make_agent(saved, 1.0)),
              "side_evidence": ObstacleSideEvidenceAgent(make_agent(saved, 1.0))}
    rows = {name: [] for name in agents}
    with (run / "events.jsonl").open("x", encoding="utf-8") as events:
        for index, (track, seed) in enumerate(cells):
            order = tuple(agents) if index % 2 == 0 else tuple(reversed(tuple(agents)))
            for name in order:
                agent = agents[name]
                row = compact(run_episode(mode=name, track_id=track, seed=seed,
                    agent=agent, max_decisions=2000, plan_budget_seconds=4.5))
                row["road_guard_count"] = agent.road_guard_count
                if name == "side_evidence":
                    row["early_evidence_count"] = agent.early_evidence_count
                rows[name].append(row)
                events.write(json.dumps({"type": "episode", "arm": name, **row}) + "\n")
                events.flush()
                print(name, track, seed, row["completed"], row["lapTimeMs"], row["progress"], row.get("early_evidence_count"), flush=True)
    summaries = {name: summary(values) for name, values in rows.items()}
    (run / "integration_report.json").write_text(json.dumps({
        "run_id": run_id, "summaries": summaries, "decision": "DIAGNOSTIC_ONLY",
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "UNKNOWN",
                  "competitive_or_product_outcome": "UNKNOWN"}
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
