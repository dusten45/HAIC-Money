from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


SCHEMA_VERSION = 1
OBSTACLE_MODES = frozenset(
    {"official", "custom_only", "official_plus_custom"}
)
MAX_SEED = 0xFFFFFFFF
MAX_CUSTOM_OBSTACLES = 64
MIN_OBSTACLE_RADIUS = 0.2
MAX_OBSTACLE_RADIUS = 4.0
MAX_STEPS = 10_000
MAX_FRAME_SKIP = 16


def _require_int(name: str, value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return int(value)


def _require_float(
    name: str,
    value: object,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


@dataclass(frozen=True)
class CustomObstacle:
    progress: float
    lateral: float
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "progress",
            _require_float("progress", self.progress, 0.0, 1.0),
        )
        object.__setattr__(
            self,
            "lateral",
            _require_float("lateral", self.lateral, -1.0, 1.0),
        )
        object.__setattr__(
            self,
            "radius",
            _require_float(
                "radius",
                self.radius,
                MIN_OBSTACLE_RADIUS,
                MAX_OBSTACLE_RADIUS,
            ),
        )


@dataclass(frozen=True)
class MapSpec:
    track_id: int
    seed: int
    obstacle_mode: str
    obstacles: tuple[CustomObstacle, ...]
    max_steps: int
    frame_skip: int
    schema_version: int = SCHEMA_VERSION
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "track_id",
            _require_int("track_id", self.track_id, 1, 2**63 - 1),
        )
        object.__setattr__(
            self,
            "seed",
            _require_int("seed", self.seed, 0, MAX_SEED),
        )
        if self.obstacle_mode not in OBSTACLE_MODES:
            raise ValueError(
                "obstacle_mode must be one of "
                f"{sorted(OBSTACLE_MODES)}"
            )
        if not isinstance(self.obstacles, tuple):
            raise ValueError("obstacles must be a tuple")
        if len(self.obstacles) > MAX_CUSTOM_OBSTACLES:
            raise ValueError(
                f"at most {MAX_CUSTOM_OBSTACLES} custom obstacles are allowed"
            )
        if not all(isinstance(obstacle, CustomObstacle) for obstacle in self.obstacles):
            raise ValueError("obstacles must contain CustomObstacle values")
        object.__setattr__(
            self,
            "max_steps",
            _require_int("max_steps", self.max_steps, 1, MAX_STEPS),
        )
        object.__setattr__(
            self,
            "frame_skip",
            _require_int("frame_skip", self.frame_skip, 1, MAX_FRAME_SKIP),
        )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if not isinstance(self.metadata, tuple):
            raise ValueError("metadata must be a tuple of key/value pairs")


def validate_map_payload(payload: object) -> MapSpec:
    if not isinstance(payload, Mapping):
        raise ValueError("map payload must be an object")

    schema_version = payload.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version}")

    raw_obstacles = payload.get("obstacles", [])
    if not isinstance(raw_obstacles, (list, tuple)):
        raise ValueError("obstacles must be an array")
    if len(raw_obstacles) > MAX_CUSTOM_OBSTACLES:
        raise ValueError(
            f"at most {MAX_CUSTOM_OBSTACLES} custom obstacles are allowed"
        )

    obstacles: list[CustomObstacle] = []
    for index, raw_obstacle in enumerate(raw_obstacles):
        if not isinstance(raw_obstacle, Mapping):
            raise ValueError(f"obstacles[{index}] must be an object")
        try:
            obstacles.append(
                CustomObstacle(
                    progress=raw_obstacle["progress"],
                    lateral=raw_obstacle["lateral"],
                    radius=raw_obstacle["radius"],
                )
            )
        except KeyError as error:
            raise ValueError(
                f"obstacles[{index}] is missing {error.args[0]}"
            ) from error

    raw_metadata = payload.get("metadata", {})
    if raw_metadata is None:
        raw_metadata = {}
    if not isinstance(raw_metadata, Mapping):
        raise ValueError("metadata must be an object")

    return MapSpec(
        track_id=payload.get("track_id"),
        seed=payload.get("seed"),
        obstacle_mode=payload.get("obstacle_mode", "official"),
        obstacles=tuple(obstacles),
        max_steps=payload.get("max_steps", 2000),
        frame_skip=payload.get("frame_skip", 4),
        schema_version=schema_version,
        metadata=tuple(sorted(raw_metadata.items())),
    )


def map_to_dict(spec: MapSpec) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": spec.schema_version,
        "track_id": spec.track_id,
        "seed": spec.seed,
        "obstacle_mode": spec.obstacle_mode,
        "obstacles": [
            {
                "progress": obstacle.progress,
                "lateral": obstacle.lateral,
                "radius": obstacle.radius,
            }
            for obstacle in spec.obstacles
        ],
        "max_steps": spec.max_steps,
        "frame_skip": spec.frame_skip,
    }
    if spec.metadata:
        result["metadata"] = dict(spec.metadata)
    return result


def map_from_dict(payload: object) -> MapSpec:
    return validate_map_payload(payload)
