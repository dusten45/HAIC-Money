"""Trace a consumed tune failure from extracted preview and fast-speed ZIPs."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

from research.anticipatory_steer_speed_tune_20260928 import CHILD, PACKAGES, ROOT


RUN = ROOT / "runs/haic-research-v2/anticipatory-speed-road-cue-trace-20260928"
CELLS = ((2, 1700),)
TRACE_CHILD = CHILD.replace(
    "from training.env_factory import create_training_environment",
    "from training.env_factory import create_training_environment\nfrom haic_agent.pixel_features import current_frame, road_centers, estimate_observation_speed",
).replace(
    "self.max_negative_streak = 0",
    "self.max_negative_streak = 0\n        self.road_samples = []",
).replace(
    "self.max_negative_streak = max(self.max_negative_streak, self.environment_object.environment.off_track_counter)",
    "self.max_negative_streak = max(self.max_negative_streak, self.environment_object.environment.off_track_counter)\n"
    "        self.road_samples.append({'off_wheels': off_count, 'negative_streak': self.environment_object.environment.off_track_counter, 'centers': road_centers(current_frame(result[0])), 'pixel_speed': estimate_observation_speed(result[0], current_frame(result[0]))})",
).replace(
    "fail_on_invalid_action=True, environment_factory=factory)",
    "fail_on_invalid_action=True, environment_factory=factory, capture_trace=True)",
).replace(
    "row.update(any_wheel_off_count=road.any_off",
    "row['decision_trace'] = raw['decision_trace']\nrow['road_samples'] = road.road_samples\nrow.update(any_wheel_off_count=road.any_off",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": RUN.name, "split": "consumed_tune_diagnostic", "cells": list(CELLS),
        "package_sha256": {arm: digest(path) for arm, path in PACKAGES.items()},
        "source_sha256": digest(Path(__file__)), "created_unix": time.time(),
    }
    (RUN / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory() as directory:
        extracted = {}
        for arm, path in PACKAGES.items():
            target = Path(directory) / arm
            target.mkdir()
            with zipfile.ZipFile(path) as archive:
                archive.extractall(target)
            extracted[arm] = target
        rows = {}
        with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
            for track, seed in CELLS:
                for arm in PACKAGES:
                    result = subprocess.run(
                        [sys.executable, "-c", TRACE_CHILD, str(ROOT), str(track), str(seed)],
                        cwd=extracted[arm], text=True, capture_output=True, timeout=120,
                    )
                    if result.returncode:
                        raise RuntimeError(result.stderr[-2000:])
                    row = json.loads(result.stdout.strip().splitlines()[-1])
                    rows[arm] = row
                    events.write(json.dumps({"type": "episode", "arm": arm, **row}) + "\n")
                    events.flush()
    report = {"run_id": RUN.name, "decision": "DIAGNOSTIC_ONLY", "episodes": {
        arm: {key: row[key] for key in ("completed", "lapTimeMs", "progress", "steps", "boost_count", "all_wheels_off_max_streak")}
        for arm, row in rows.items()
    }}
    (RUN / "integration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
