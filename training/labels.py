"""Collector-only simulator labels for auxiliary training targets."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from haic_agent.observation import HUDFeatures, extract_hud_features


@dataclass(frozen=True)
class TrainingLabels:
    speed: float
    wheel_omega: tuple[float, float, float, float]
    steering_angle: float
    yaw_rate: float
    tile_progress: float
    collision: bool
    damage: float
    off_track: bool
    finished: bool


def _simulator(environment: Any) -> Any:
    return getattr(environment, "unwrapped", environment)


def collect_labels(environment: Any, info: dict[str, Any]) -> TrainingLabels:
    """Read the collector's state at the completed decision boundary only."""
    simulator = _simulator(environment)
    car = simulator.car
    velocity = car.hull.linearVelocity
    speed = float(np.hypot(float(velocity[0]), float(velocity[1])))
    wheels = tuple(float(wheel.omega) for wheel in car.wheels)
    if len(wheels) != 4:
        raise ValueError("simulator must provide four wheel rotation signals")
    steering = float(np.mean([car.wheels[0].joint.angle, car.wheels[1].joint.angle]))
    total_tiles = len(simulator.track)
    progress = float(simulator.tile_visited_count / total_tiles) if total_tiles else 0.0
    return TrainingLabels(
        speed=speed,
        wheel_omega=wheels,
        steering_angle=steering,
        yaw_rate=float(car.hull.angularVelocity),
        tile_progress=float(np.clip(progress, 0.0, 1.0)),
        collision=bool(info.get("collision", False)),
        damage=float(
            info["damage"]
            if "damage" in info
            else getattr(getattr(environment, "damage", None), "damage", 0.0)
        ),
        off_track=bool(info.get("retire_reason") == "off_track"),
        finished=bool(info.get("finished", False)),
    )


def labels_with_hud(
    environment: Any, info: dict[str, Any], observation: np.ndarray
) -> tuple[TrainingLabels, HUDFeatures]:
    """Pair train-only state labels with HUD crops from the same pixels."""
    return collect_labels(environment, info), extract_hud_features(observation)
