"""Held-out comparison runner with evaluation-only wheel-road exposure metrics."""

import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/plans/2026-09-29-speed-coupled-preview-steering.md"
PLAN_SHA = "7a005c0b029b0221d549903f875997f2903ca5a332256ce44f5f321027952cf8"
RUN = ROOT / "runs/haic-research-v2/speed-coupled-preview-tune-20260929"
PACKAGES = {
    "preview": ROOT / "artifacts/haic-research-v2/anticipatory-bend-candidate-20260928/submission-anticipatory-bend.zip",
    "preview_speed": ROOT / "artifacts/haic-research-v2/speed-coupled-preview-candidate-20260929/submission-speed-coupled-preview.zip",
}
EXPECTED = {
    "preview": "ce28ab744956bc4898d102c41ff6184ee4ac5a819a8559a656a6aedda4008f66",
    "preview_speed": "54c117287390704acc52505d680c31574fdcf53ac062cbb0ce5258a57058c324",
}
CELLS = tuple((track, seed) for track in (1, 2, 3) for seed in (2000, 2001, 2002, 2003))
SOURCES = ("training/evaluate_closed_loop.py", "training/env_factory.py", "env_wrapper.py", "core/vendor/car_racing.py", "research/speed_coupled_preview_tune_20260929.py")

CHILD = r'''
import json
import sys
import time
import resource
from pathlib import Path

root, track, seed = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
started = time.monotonic()
import agent
driver = agent.Agent()
init_s = time.monotonic() - started
if not str(Path(agent.__file__).resolve()).startswith(str(Path.cwd())):
    raise RuntimeError("agent was not loaded from extracted package")
if not str(Path(sys.modules["haic_agent"].__file__).resolve()).startswith(str(Path.cwd())):
    raise RuntimeError("haic_agent was not loaded from extracted package")
sys.path.append(str(root))
from training.evaluate_closed_loop import run_episode
from training.env_factory import create_training_environment

class RoadMeter:
    def __init__(self, environment):
        self.environment_object = environment
        self.steps = 0
        self.any_off = 0
        self.all_off = 0
        self.all_off_streak = 0
        self.max_all_off_streak = 0
        self.max_negative_streak = 0

    def __getattr__(self, name):
        return getattr(self.environment_object, name)

    def step(self, action):
        result = self.environment_object.step(action)
        self.steps += 1
        wheels = self.environment_object.unwrapped.car.wheels
        off_count = sum(not bool(wheel.tiles) for wheel in wheels)
        self.any_off += int(off_count > 0)
        self.all_off += int(off_count == 4)
        self.all_off_streak = self.all_off_streak + 1 if off_count == 4 else 0
        self.max_all_off_streak = max(self.max_all_off_streak, self.all_off_streak)
        self.max_negative_streak = max(self.max_negative_streak, self.environment_object.environment.off_track_counter)
        return result

meter = []
def factory(**kwargs):
    wrapped = RoadMeter(create_training_environment(**kwargs))
    meter.append(wrapped)
    return wrapped

raw = run_episode(mode="steering_speed_tune", track_id=track, seed=seed,
                  agent=driver, max_decisions=2000, plan_budget_seconds=4.5,
                  fail_on_invalid_action=True, environment_factory=factory)
keys = ("track_id", "seed", "completed", "lapTimeMs", "progress", "collisions", "damage",
        "retire_reason", "error", "invalid_actions", "steps", "mean_speed", "max_speed",
        "act_p50_ms", "act_p95_ms", "act_max_ms", "wall_time_s")
row = {key: raw[key] for key in keys}
controller = driver._controller
row["preview_count"] = getattr(controller, "preview_count", getattr(getattr(controller, "base", None), "preview_count", None))
row["boost_count"] = getattr(controller, "boost_count", 0)
row["early_steer_count"] = getattr(controller, "early_steer_count", 0)
row["bend_brake_count"] = getattr(controller, "bend_brake_count", 0)
row["hazard_brake_count"] = getattr(controller, "hazard_brake_count", 0)
row["boost_during_preview_count"] = getattr(controller, "boost_during_preview_count", 0)
road = meter[0]
row.update(any_wheel_off_count=road.any_off, all_wheels_off_count=road.all_off,
           all_wheels_off_max_streak=road.max_all_off_streak,
           max_negative_reward_streak=road.max_negative_streak,
           any_wheel_off_fraction=road.any_off/max(road.steps,1),
           import_create_s=init_s,
           peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
print(json.dumps(row), flush=True)
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def other_claims():
    claims = []
    targets = {f"{track}:{seed}" for track, seed in CELLS}
    for path in (ROOT / "runs/haic-research-v2").glob("*/run_manifest.json"):
        if path.parent == RUN:
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        found = set(manifest.get("seed_ids", [])) & targets
        for cell in manifest.get("cells", []):
            if isinstance(cell, dict):
                identity = f"{cell.get('track_id')}:{cell.get('seed')}"
                if identity in targets:
                    found.add(identity)
        if found:
            claims.append({"run": path.parent.name, "cells": sorted(found)})
    return claims


def summarize(rows):
    laps = [row["lapTimeMs"] for row in rows if row["completed"]]
    incomplete = [row["progress"] for row in rows if not row["completed"]]
    return {
        "completed": len(laps), "denominator": len(rows),
        "median_finished_lap_ms": statistics.median(laps) if laps else None,
        "mean_incomplete_progress": statistics.mean(incomplete) if incomplete else None,
        "collisions": sum(row["collisions"] or 0 for row in rows),
        "damage": sum(row["damage"] or 0 for row in rows),
        "invalid_actions": sum(row["invalid_actions"] for row in rows),
        "preview_count": sum(row["preview_count"] or 0 for row in rows),
        "boost_count": sum(row["boost_count"] for row in rows),
        "early_steer_count": sum(row["early_steer_count"] for row in rows),
        "bend_brake_count": sum(row["bend_brake_count"] for row in rows),
        "hazard_brake_count": sum(row["hazard_brake_count"] for row in rows),
        "boost_during_preview_count": sum(row["boost_during_preview_count"] for row in rows),
        "all_wheels_off_count": sum(row["all_wheels_off_count"] for row in rows),
        "any_wheel_off_fraction": sum(row["any_wheel_off_count"] for row in rows) / sum(row["steps"] for row in rows),
        "max_all_wheels_off_streak": max(row["all_wheels_off_max_streak"] for row in rows),
        "max_negative_reward_streak": max(row["max_negative_reward_streak"] for row in rows),
        "mean_speed": statistics.mean(row["mean_speed"] for row in rows if row["mean_speed"] is not None),
    }


def main():
    if digest(PLAN) != PLAN_SHA:
        raise RuntimeError("plan changed")
    for arm, path in PACKAGES.items():
        if digest(path) != EXPECTED[arm]:
            raise RuntimeError(f"{arm} package drift")
    claims = other_claims()
    if claims:
        raise RuntimeError(f"split already consumed: {claims}")
    RUN.mkdir(parents=True, exist_ok=False)
    sources = {name: digest(ROOT / name) for name in SOURCES}
    manifest = {
        "run_id": RUN.name, "split": "held_out", "plan": str(PLAN.relative_to(ROOT)),
        "plan_sha256": PLAN_SHA, "cells": [{"track_id": t, "seed": s} for t, s in CELLS],
        "max_decisions": 2000, "packages": {arm: {"path": str(path.relative_to(ROOT)), "sha256": EXPECTED[arm]} for arm, path in PACKAGES.items()},
        "source_sha256": sources, "created_unix": time.time(),
    }
    (RUN / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows = {arm: [] for arm in PACKAGES}
    with tempfile.TemporaryDirectory() as directory:
        extracted = {}
        for arm, path in PACKAGES.items():
            target = Path(directory) / arm
            target.mkdir()
            with zipfile.ZipFile(path) as archive:
                archive.extractall(target)
            extracted[arm] = target
        with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
            for index, (track, seed) in enumerate(CELLS):
                for arm in (("preview", "preview_speed") if index % 2 == 0 else ("preview_speed", "preview")):
                    result = subprocess.run(
                        [sys.executable, "-c", CHILD, str(ROOT), str(track), str(seed)],
                        cwd=extracted[arm], text=True, capture_output=True, timeout=120, check=False,
                    )
                    if result.returncode:
                        events.write(json.dumps({"type": "infrastructure_error", "arm": arm, "track_id": track, "seed": seed, "stderr": result.stderr[-2000:]}) + "\n")
                        events.flush()
                        raise RuntimeError(result.stderr[-2000:])
                    row = json.loads(result.stdout.strip().splitlines()[-1])
                    rows[arm].append(row)
                    events.write(json.dumps({"type": "episode", "arm": arm, **row}) + "\n")
                    events.flush()
                    print(arm, track, seed, row["completed"], row["lapTimeMs"], row["retire_reason"], row["all_wheels_off_max_streak"], row["boost_count"], flush=True)
    drift = sources != {name: digest(ROOT / name) for name in SOURCES} or any(digest(path) != EXPECTED[arm] for arm, path in PACKAGES.items())
    claims_after = other_claims()
    sums = {arm: summarize(values) for arm, values in rows.items()}
    candidate_only_failures = [list(cell) for cell, base, speed in zip(CELLS, rows["preview"], rows["preview_speed"]) if base["completed"] and not speed["completed"]]
    exposure_failures = [list(cell) for cell, base, speed in zip(CELLS, rows["preview"], rows["preview_speed"]) if speed["all_wheels_off_max_streak"] > base["all_wheels_off_max_streak"] + 10 or speed["any_wheel_off_fraction"] > base["any_wheel_off_fraction"] + 0.02]
    valid = not drift and not claims_after
    decision = "DIAGNOSTIC_ONLY" if not valid else "REJECT" if candidate_only_failures or exposure_failures or sums["preview_speed"]["completed"] < sums["preview"]["completed"] else "ADVANCE" if sums["preview_speed"]["completed"] > sums["preview"]["completed"] or sums["preview_speed"]["median_finished_lap_ms"] < sums["preview"]["median_finished_lap_ms"] else "REVISE"
    report = {
        "run_id": RUN.name, "decision": decision, "infrastructure_valid": valid,
        "split_collisions": claims_after, "candidate_only_failures": candidate_only_failures,
        "material_road_exposure_cells": exposure_failures, "summaries": sums,
        "gates": {"rule_compliance": "UNKNOWN", "mechanism_activation": "PASS" if sums["preview_speed"]["boost_count"] > 0 else "FAIL", "competitive_or_product_outcome": "PASS" if decision == "ADVANCE" else "FAIL"},
    }
    (RUN / "integration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
