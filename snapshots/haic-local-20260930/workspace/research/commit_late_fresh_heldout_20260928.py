"""Frozen candidate comparison on unused held-out seeds 156-159."""

import json
from pathlib import Path
import time

import torch

from haic_agent.hybrid_runtime import ObstacleHybridAgent
from haic_agent.obstacle_commit_late_runtime import ObstacleCommitLateAgent
from research.score_probe_20260928 import CHECKPOINT, ROOT, compact, make_agent, sha256, summary
from training.evaluate_closed_loop import run_episode


CELLS = tuple((track, seed) for track in (1, 2) for seed in (156, 157, 158, 159))
RUN = ROOT / "runs/haic-research-v2/score-linux-commit-late-fresh-heldout-20260928"


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": RUN.name, "purpose": "frozen matched candidate comparison on fresh held-out cells",
        "split": "held_out", "cells": [{"track_id": t, "seed": s} for t, s in CELLS],
        "max_decisions": 2000,
        "source_sha256": sha256(Path(__file__).resolve()),
        "hybrid_source_sha256": sha256(ROOT / "haic_agent/hybrid_runtime.py"),
        "commit_late_source_sha256": sha256(ROOT / "haic_agent/obstacle_commit_late_runtime.py"),
        "commit_parent_source_sha256": sha256(ROOT / "haic_agent/obstacle_commit_runtime.py"),
        "checkpoint_sha256": sha256(CHECKPOINT), "created_unix": time.time(),
    }
    (RUN / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    agents = {"prior_hybrid": ObstacleHybridAgent(make_agent(saved, 1.0)),
              "side_commit_late": ObstacleCommitLateAgent(make_agent(saved, 1.0))}
    rows = {name: [] for name in agents}
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for index, (track, seed) in enumerate(CELLS):
            order = tuple(agents) if index % 2 == 0 else tuple(reversed(tuple(agents)))
            for name in order:
                agent = agents[name]
                row = compact(run_episode(mode=name, track_id=track, seed=seed,
                    agent=agent, max_decisions=2000, plan_budget_seconds=4.5))
                row["override_count"] = agent.override_count
                if name == "side_commit_late":
                    row["side_correction_count"] = agent.side_correction_count
                rows[name].append(row)
                events.write(json.dumps({"type": "episode", "arm": name, **row}) + "\n")
                events.flush()
                print(name, track, seed, row["completed"], row["lapTimeMs"], row["progress"], row.get("side_correction_count"), flush=True)
    report = {"run_id": RUN.name, "summaries": {name: summary(values) for name, values in rows.items()},
              "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "UNKNOWN",
                        "competitive_or_product_outcome": "UNKNOWN"},
              "decision": "DIAGNOSTIC_ONLY"}
    (RUN / "integration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
