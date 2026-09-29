"""Collector-only simulator labels for auxiliary training targets."""

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from haic_agent.observation import HUDFeatures, extract_hud_features
from core.vendor.car_racing import TRACK_WIDTH


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
    lateral_error: float = 0.0
    road_half_width: float = TRACK_WIDTH
    heading_error: float = 0.0


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
    site_map = getattr(simulator, "site_map", None)
    geometry = getattr(site_map, "geometry", None)
    road_half_width = float(getattr(geometry, "width", TRACK_WIDTH))
    lateral_error = 0.0
    heading_error = 0.0
    position = getattr(car.hull, "position", None)
    hull_angle = getattr(car.hull, "angle", None)
    if position is not None and hull_angle is not None and simulator.track:
        x, y = float(position[0]), float(position[1])
        valid_indices = [
            index
            for index, entry in enumerate(simulator.track)
            if entry is not None and len(entry) >= 4
        ]
        contact_indices = {
            int(tile.idx)
            for wheel in car.wheels
            for tile in getattr(wheel, "tiles", ())
            if hasattr(tile, "idx")
        }
        candidates = [
            index for index in valid_indices if index in contact_indices
        ] or valid_indices
        if candidates:
            nearest_index = min(
                candidates,
                key=lambda index: (simulator.track[index][2] - x) ** 2
                + (simulator.track[index][3] - y) ** 2,
            )
            _track_progress, beta, center_x, center_y = simulator.track[nearest_index]
            lateral_error = (x - center_x) * math.cos(beta) + (y - center_y) * math.sin(beta)
            # Car's local +Y axis is forward. The hull and road use the same
            # body-angle convention, so their rotation difference is direct.
            angle_difference = float(hull_angle) - beta
            heading_error = math.atan2(math.sin(angle_difference), math.cos(angle_difference))
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
        lateral_error=float(lateral_error),
        road_half_width=road_half_width,
        heading_error=float(heading_error),
    )


def labels_with_hud(
    environment: Any, info: dict[str, Any], observation: np.ndarray
) -> tuple[TrainingLabels, HUDFeatures]:
    """Pair train-only state labels with HUD crops from the same pixels."""
    return collect_labels(environment, info), extract_hud_features(observation)
