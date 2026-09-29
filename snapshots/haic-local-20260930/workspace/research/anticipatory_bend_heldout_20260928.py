"""Matched three-track held-out check of fixed anticipatory bend steering."""

import hashlib
import io
import json
from pathlib import Path
import statistics
import time
import zipfile

import torch

from haic_agent.anticipatory_bend_runtime import AnticipatoryBendAgent
from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from research.score_probe_20260928 import compact, make_agent
from training.evaluate_closed_loop import run_episode


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts/haic-research-v2/full-road-guard-candidate-20260928/submission-full-road-guard.zip"
RUN = ROOT / "runs/haic-research-v2/anticipatory-bend-heldout-20260928"
PLAN = ROOT / "docs/plans/2026-09-28-anticipatory-bend-heldout.md"
CELLS = tuple((track, seed) for track in (1, 2, 3) for seed in (304, 305, 306, 307))
EXPECTED_CANDIDATE_SHA = "27459733edf6d845b61296f68fa82eb17af9672e2a6c88337ff8ab6d9a2f91d2"
EXPECTED_CONTROL_SHA = "5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40"
SOURCE_PATHS = (
    "agent.py", "env_wrapper.py", "core/vendor/car_racing.py",
    "haic_agent/anticipatory_bend_runtime.py", "haic_agent/pixel_features.py",
    "haic_agent/obstacle_full_road_guard_runtime.py",
    "haic_agent/obstacle_side_evidence_runtime.py",
    "haic_agent/obstacle_road_guard_runtime.py",
    "haic_agent/obstacle_commit_late_runtime.py",
    "haic_agent/obstacle_commit_runtime.py", "haic_agent/corridor_agent.py",
    "research/score_probe_20260928.py", "training/evaluate_closed_loop.py",
    "training/env_factory.py", "training/site_maps.py",
    "research/anticipatory_bend_heldout_20260928.py",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(rows):
    finished = [row["lapTimeMs"] for row in rows if row["completed"]]
    return {
        "completed": len(finished),
        "denominator": len(rows),
        "median_finished_lap_ms": statistics.median(finished) if finished else None,
        "mean_progress": statistics.mean(row["progress"] for row in rows),
        "collisions": sum(row["collisions"] or 0 for row in rows),
        "damage": sum(row["damage"] or 0 for row in rows),
    }


def main():
    if _sha256(ROOT / "haic_agent/anticipatory_bend_runtime.py") != EXPECTED_CANDIDATE_SHA:
        raise RuntimeError("candidate source changed after tune")
    if _sha256(PACKAGE) != EXPECTED_CONTROL_SHA:
        raise RuntimeError("control package changed after tune")
    with zipfile.ZipFile(PACKAGE) as archive:
        saved = torch.load(io.BytesIO(archive.read("policy.pt")), map_location="cpu", weights_only=True)
    RUN.mkdir(parents=True, exist_ok=False)
    source_hashes = {relative: _sha256(ROOT / relative) for relative in SOURCE_PATHS}
    manifest = {
        "run_id": RUN.name,
        "purpose": "unchanged anticipatory bend candidate on a matched three-track held-out set",
        "plan": str(PLAN.relative_to(ROOT)),
        "plan_sha256": _sha256(PLAN),
        "split": "held_out",
        "cells": [{"track_id": track, "seed": seed} for track, seed in CELLS],
        "max_decisions": 2000,
        "control_package": str(PACKAGE.relative_to(ROOT)),
        "control_package_sha256": _sha256(PACKAGE),
        "candidate_source_sha256": _sha256(ROOT / "haic_agent/anticipatory_bend_runtime.py"),
        "probe_source_sha256": _sha256(Path(__file__)),
        "evaluator_source_sha256": _sha256(ROOT / "training/evaluate_closed_loop.py"),
        "source_sha256": source_hashes,
        "created_unix": time.time(),
    }
    (RUN / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows = {"control": [], "candidate": []}
    with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
        for index, (track, seed) in enumerate(CELLS):
            order = ("control", "candidate") if index % 2 == 0 else ("candidate", "control")
            for arm in order:
                baseline = ObstacleFullRoadGuardAgent(make_agent(saved, 1.0))
                agent = baseline if arm == "control" else AnticipatoryBendAgent(baseline)
                result = run_episode(
                    mode=arm, track_id=track, seed=seed, agent=agent,
                    max_decisions=2000, plan_budget_seconds=4.5,
                )
                row = compact(result)
                row["preview_count"] = getattr(agent, "preview_count", None)
                row["mean_speed"] = result.get("mean_speed")
                row["max_speed"] = result.get("max_speed")
                rows[arm].append(row)
                events.write(json.dumps({"type": "episode", "arm": arm, **row}) + "\n")
                events.flush()
                print(arm, track, seed, row["completed"], row["lapTimeMs"],
                      row["progress"], row["preview_count"], flush=True)
    report = {
        "run_id": RUN.name,
        "decision": "DIAGNOSTIC_ONLY",
        "infrastructure_valid": source_hashes == {
            relative: _sha256(ROOT / relative) for relative in SOURCE_PATHS
        },
        "summaries": {arm: _summary(episodes) for arm, episodes in rows.items()},
        "gates": {
            "rule_compliance": "UNKNOWN",
            "mechanism_activation": "PASS" if any(row["preview_count"] for row in rows["candidate"]) else "FAIL",
            "competitive_or_product_outcome": "NOT_APPLICABLE",
        },
    }
    (RUN / "integration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
