"""One-time matched held-out comparison of the frozen obstacle hybrid."""

import json
from pathlib import Path
import time

import torch

from research.hybrid_actor import ObstacleHybridAgent
from research.score_probe_20260928 import CHECKPOINT, ROOT, compact, make_agent, sha256, summary
from training.evaluate_closed_loop import run_episode


RUN = ROOT / "runs/haic-research-v2/score-confirm-hybrid-20260928"
CELLS = tuple((track, seed) for track in (1, 2) for seed in (136, 137, 138, 139))


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    manifest = {
        "run_id": RUN.name,
        "purpose": "one-time matched held-out comparison; frozen candidate selected from tune",
        "source_sha256": sha256(source),
        "hybrid_source_sha256": sha256(ROOT / "research/hybrid_actor.py"),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "cells": [{"track_id": track, "seed": seed} for track, seed in CELLS],
        "split": "held_out",
        "max_decisions": 2000,
        "arms": ["original_ppo", "obstacle_hybrid"],
        "throttle_expansion": 1.0,
        "obstacle_override_y_min": 28.0,
        "tune_reference": "runs/haic-research-v2/score-probe-hybrid-control-20260928/integration_report.json",
        "created_unix": time.time(),
    }
    (RUN / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    agents = {
        "original_ppo": make_agent(saved, 1.0),
        "obstacle_hybrid": ObstacleHybridAgent(make_agent(saved, 1.0)),
    }
    rows = {name: [] for name in agents}
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for index, (track, seed) in enumerate(CELLS):
            order = tuple(agents) if index % 2 == 0 else tuple(reversed(tuple(agents)))
            for name in order:
                row = compact(run_episode(
                    mode=name, track_id=track, seed=seed, agent=agents[name],
                    max_decisions=2000, plan_budget_seconds=4.5,
                ))
                if name == "obstacle_hybrid":
                    row["obstacle_override_count"] = agents[name].override_count
                rows[name].append(row)
                events.write(json.dumps({"type": "episode", "arm": name, **row}) + "\n")
                events.flush()
                print(name, track, seed, row["completed"], row["lapTimeMs"], row["progress"], flush=True)
    report = {
        "run_id": RUN.name,
        "summaries": {name: summary(values) for name, values in rows.items()},
        "gates": {
            "rule_compliance": "UNKNOWN",
            "mechanism_activation": "PASS",
            "competitive_or_product_outcome": "UNKNOWN",
        },
        "decision": "PENDING_REVIEW",
    }
    (RUN / "integration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
