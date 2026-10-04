"""Trace wrapper-boundary actions for one already consumed local failure cell."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from training.confirm_stable_package import RESULT_KEYS, check_archive


CONTROL_SHA256 = "5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40"
ROOT = Path(__file__).resolve().parents[1]
CHILD = r'''
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
track = int(sys.argv[2])
seed = int(sys.argv[3])
keys = json.loads(sys.argv[4])
arm = sys.argv[5]
started = time.monotonic()
import agent
driver = agent.Agent()
import_create_s = time.monotonic() - started
if not str(Path(agent.__file__).resolve()).startswith(str(Path.cwd())):
    raise RuntimeError("packaged agent was not imported")
started = time.monotonic()
driver.reset(np.zeros((4, 84, 84), dtype=np.float32))
reset_s = time.monotonic() - started
instrumentation = []
if arm == "candidate":
    from haic_agent.pixel_features import current_frame, estimate_observation_speed, road_centers
    controller = driver._controller
    preview = controller.preview
    launch = controller.launch
    base_actions = []
    original_launch_act = launch.act
    original_preview_act = preview.act

    def trace_launch(observation):
        action = original_launch_act(observation)
        base_actions.append(np.asarray(action, dtype=np.float32).tolist())
        return action

    def trace_preview(observation):
        before = preview.preview_count
        action = original_preview_act(observation)
        frame = current_frame(observation)
        centers = {int(key): float(value) for key, value in road_centers(frame).items()}
        previous = np.asarray(observation, dtype=np.float32)[-2]
        previous_centers = {int(key): float(value) for key, value in road_centers(previous).items()}
        obstacle = launch.corridor.last_step_diagnostics()["obstacle_y"]
        instrumentation.append({
            "step": len(instrumentation) + 1,
            "preview_activated": preview.preview_count > before,
            "base_action": base_actions[-1],
            "final_action": np.asarray(action, dtype=np.float32).tolist(),
            "pixel_speed": float(estimate_observation_speed(observation, frame)),
            "road_centers": centers,
            "previous_road_centers": previous_centers,
            "obstacle_y": None if obstacle is None else float(obstacle),
            "launch_active": bool(launch.launch_active),
        })
        return action

    launch.act = trace_launch
    preview.act = trace_preview

sys.path.append(str(root))
from training.evaluate_closed_loop import run_episode
raw = run_episode(mode="packaged_preview_boundary_trace", track_id=track, seed=seed,
                  agent=driver, max_decisions=2000, plan_budget_seconds=4.5,
                  capture_trace=True, fail_on_invalid_action=True)
row = {key: raw[key] for key in keys}
row["decision_trace"] = raw["decision_trace"] or []
row["instrumentation"] = instrumentation
row["import_create_s"] = import_create_s
row["reset_s"] = reset_s
row["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
print(json.dumps(row), flush=True)
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace(candidate, control, candidate_sha256, track, seed, output):
    if (len(candidate_sha256) != 64
            or any(char not in "0123456789abcdef" for char in candidate_sha256)):
        raise ValueError("candidate SHA-256 must be lowercase hex")
    if not 1 <= track <= 3 or seed < 1:
        raise ValueError("invalid track or seed")
    archives = {"candidate": candidate, "control": control}
    expected = {"candidate": candidate_sha256, "control": CONTROL_SHA256}
    for arm, path in archives.items():
        if digest(path) != expected[arm]:
            raise ValueError(f"{arm} ZIP differs from frozen package")
    checks = {arm: check_archive(path) for arm, path in archives.items()}
    rows = {}
    with tempfile.TemporaryDirectory() as directory:
        for arm, path in archives.items():
            destination = Path(directory) / arm
            destination.mkdir()
            with zipfile.ZipFile(path) as archive:
                archive.extractall(destination)
            result = subprocess.run(
                [sys.executable, "-c", CHILD, str(ROOT), str(track), str(seed),
                 json.dumps(RESULT_KEYS), arm],
                cwd=destination, text=True, capture_output=True, timeout=180, check=False,
            )
            if result.returncode:
                raise RuntimeError(f"{arm} trace failed: {result.stderr[-2000:]}")
            rows[arm] = json.loads(result.stdout.strip().splitlines()[-1])
    if any(digest(path) != expected[arm] for arm, path in archives.items()):
        raise RuntimeError("package changed during trace")
    report = {"candidate_sha256": candidate_sha256, "control_sha256": CONTROL_SHA256,
              "track": track, "seed": seed, "checks": checks, "episodes": rows,
              "diagnostic_only": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--control-archive", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = trace(args.candidate_archive, args.control_archive, args.candidate_sha256,
                   args.track_id, args.seed, args.output)
    print(json.dumps({"track": report["track"], "seed": report["seed"],
                      "candidate_completed": report["episodes"]["candidate"]["completed"],
                      "control_completed": report["episodes"]["control"]["completed"],
                      "preview_events": sum(item["preview_activated"] for item in
                                            report["episodes"]["candidate"]["instrumentation"])}))


if __name__ == "__main__":
    main()
