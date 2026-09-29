"""Is a sub-15s lap physically available on the official tracks?

Measures, per official track/seed:
  * the generated track length from the simulator's own tile centres,
  * the mean speed a 15.0s lap would require,
  * the top speed the car actually reaches under sustained full throttle on
    that track's opening straight, and the speed it holds with the throttle
    ceilings the actor is currently allowed.

Diagnostic only; it drives with fixed pedal commands, not with the policy.
"""

import json
import math
from pathlib import Path

import numpy as np

from training.env_factory import create_training_environment

TARGET_LAP_S = 15.0
DECISION_SECONDS = 4 / 50  # frame skip 4 at 50 FPS
EPISODES = ((1, 42), (1, 48), (2, 101), (2, 107))
THROTTLES = (0.12, 0.24, 0.42, 1.0)


def track_length(simulator) -> float:
    points = [(float(entry[2]), float(entry[3])) for entry in simulator.track]
    return sum(
        math.dist(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )


def straight_line_speed(track_id: int, seed: int, gas: float, *, decisions: int = 120) -> dict:
    environment = create_training_environment(
        track_id=track_id, seed=seed, max_decisions=decisions + 5
    )
    speeds = []
    try:
        environment.reset()
        action = np.array([0.0, gas, 0.0], dtype=np.float32)
        for _ in range(decisions):
            transition = environment.step_transition(action)
            speeds.append(float(transition.labels.speed))
            if transition.terminated:
                break
    finally:
        environment.close()
    return {
        "gas": gas,
        "decisions_driven": len(speeds),
        "peak_speed": max(speeds) if speeds else 0.0,
        "speed_at_40_decisions": speeds[39] if len(speeds) > 39 else None,
    }


def main() -> None:
    report = {"target_lap_s": TARGET_LAP_S, "tracks": {}}
    for track_id, seed in EPISODES:
        environment = create_training_environment(track_id=track_id, seed=seed, max_decisions=2)
        try:
            environment.reset()
            simulator = environment.environment.unwrapped
            length = track_length(simulator)
            tiles = len(simulator.track)
        finally:
            environment.close()
        entry = {
            "tiles": tiles,
            "track_length": length,
            "required_mean_speed_for_target": length / TARGET_LAP_S,
            "decisions_available_at_target": TARGET_LAP_S / DECISION_SECONDS,
            "throttle_probe": [],
        }
        for gas in THROTTLES:
            entry["throttle_probe"].append(straight_line_speed(track_id, seed, gas))
        report["tracks"][f"track{track_id}-seed{seed}"] = entry
        print(json.dumps({f"track{track_id}-seed{seed}": entry}, indent=2), flush=True)

    Path("tmp").mkdir(exist_ok=True)
    Path("tmp/lap-time-ceiling.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
