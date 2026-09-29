"""Matched, diagnostic-only activation check on already consumed tune cells."""

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
RUN = ROOT / "runs/haic-research-v2/anticipatory-bend-steer-only-20260928"
CELLS = ((3, 228), (1, 228))


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
    with zipfile.ZipFile(PACKAGE) as archive:
        saved = torch.load(io.BytesIO(archive.read("policy.pt")), map_location="cpu", weights_only=True)
    RUN.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": RUN.name,
        "purpose": "isolate early bounded steering without pedal change; consumed tune only, not promotion evidence",
        "plan": "docs/plans/2026-09-28-top-three-performance-rebuild.md",
        "split": "consumed_tune",
        "cells": [{"track_id": track, "seed": seed} for track, seed in CELLS],
        "max_decisions": 2000,
        "control_package": str(PACKAGE.relative_to(ROOT)),
        "control_package_sha256": _sha256(PACKAGE),
        "candidate_source_sha256": _sha256(ROOT / "haic_agent/anticipatory_bend_runtime.py"),
        "probe_source_sha256": _sha256(Path(__file__)),
        "evaluator_source_sha256": _sha256(ROOT / "training/evaluate_closed_loop.py"),
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
