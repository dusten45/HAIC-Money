"""Fixed, matched actor-only throttle diagnostic on previously registered tune cells."""

import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from agent import Agent
from haic_agent.networks import VisualActorCritic
from training.evaluate_closed_loop import run_episode


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/policy.pt"
RUN = ROOT / "runs/haic-research-v2/score-probe-throttle-20260928"
CELLS = tuple((track, seed) for track in (1, 2) for seed in (132, 133, 134, 135))
ARMS = (("control", 1.0), ("throttle_3p5", 3.5))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_agent(saved: dict, throttle_expansion: float) -> Agent:
    metadata = saved["metadata"]
    actor = VisualActorCritic(
        use_hud=metadata.get("use_hud", True),
        use_visual_features=metadata.get("use_visual_features", False),
        use_temporal_features=metadata.get("use_temporal_features", False),
        throttle_expansion=throttle_expansion,
        brake_expansion=1.0,
    )
    actor.load_state_dict(saved["model_state"], strict=True)
    actor.eval()
    return Agent(policy=actor, planner_enabled=False, strict_checkpoint_loading=True)


def compact(row: dict) -> dict:
    keys = (
        "track_id", "seed", "completed", "lapTimeMs", "progress", "collisions",
        "damage", "retire_reason", "error", "invalid_actions", "steps",
        "act_p50_ms", "act_p95_ms", "act_max_ms", "wall_time_s",
    )
    return {key: row[key] for key in keys}


def summary(rows: list[dict]) -> dict:
    completed = [row["lapTimeMs"] for row in rows if row["completed"]]
    return {
        "completed": len(completed),
        "denominator": len(rows),
        "median_finished_lap_ms": statistics.median(completed) if completed else None,
        "mean_progress": statistics.mean(row["progress"] for row in rows),
        "collisions": sum(row["collisions"] or 0 for row in rows),
        "damage": sum(row["damage"] or 0 for row in rows),
    }


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    checkpoint_hash = sha256(CHECKPOINT)
    manifest = {
        "run_id": RUN.name,
        "purpose": "fixed throttle activation and matched tune diagnostic; not SOTA evidence",
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "cells": [{"track_id": track, "seed": seed} for track, seed in CELLS],
        "split": "tune",
        "max_decisions": 2000,
        "arms": [{"id": name, "throttle_expansion": value, "brake_expansion": 1.0} for name, value in ARMS],
        "created_unix": time.time(),
    }
    (RUN / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    agents = {name: make_agent(saved, value) for name, value in ARMS}
    rows = {name: [] for name, _ in ARMS}
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for index, (track, seed) in enumerate(CELLS):
            for name, _ in (ARMS if index % 2 == 0 else reversed(ARMS)):
                result = compact(run_episode(
                    mode=name, track_id=track, seed=seed, agent=agents[name],
                    max_decisions=2000, plan_budget_seconds=4.5,
                ))
                rows[name].append(result)
                events.write(json.dumps({"type": "episode", "arm": name, **result}) + "\n")
                events.flush()
                print(name, track, seed, result["completed"], result["lapTimeMs"], result["progress"], flush=True)
    report = {
        "run_id": RUN.name,
        "summaries": {name: summary(values) for name, values in rows.items()},
        "gates": {
            "rule_compliance": "UNKNOWN",
            "mechanism_activation": "UNKNOWN",
            "competitive_or_product_outcome": "UNKNOWN",
        },
        "decision": "DIAGNOSTIC_ONLY",
    }
    (RUN / "integration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
