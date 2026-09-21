from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any


MAX_SEED = 0xFFFFFFFF
MAX_CUSTOM_OBSTACLES = 64
MIN_OBSTACLE_RADIUS = 0.2
MAX_OBSTACLE_RADIUS = 4.0
MAX_STEPS = 10_000
MAX_FRAME_SKIP = 16
CUSTOM_MAP_ID_PATTERN = re.compile(r"^custom-track-[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


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
class CustomTrackGeometry:
    centerline: tuple[tuple[float, float], ...]
    width: float
    start_index: int = 0
    direction: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.centerline, tuple) or len(self.centerline) < 3:
            raise ValueError("centerline must contain at least three points")
        normalized: list[tuple[float, float]] = []
        for index, point in enumerate(self.centerline):
            if not isinstance(point, tuple) or len(point) != 2:
                raise ValueError(f"centerline[{index}] must contain two values")
            x = _require_float(f"centerline[{index}].x", point[0], -100_000.0, 100_000.0)
            y = _require_float(f"centerline[{index}].y", point[1], -100_000.0, 100_000.0)
            normalized.append((x, y))
        object.__setattr__(self, "centerline", tuple(normalized))
        object.__setattr__(self, "width", _require_float("width", self.width, 0.5, 100.0))
        object.__setattr__(
            self,
            "start_index",
            _require_int("start_index", self.start_index, 0, len(normalized) - 1),
        )
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")

    def ordered_centerline(self) -> tuple[tuple[float, float], ...]:
        point_count = len(self.centerline)
        return tuple(
            self.centerline[(self.start_index + self.direction * offset) % point_count]
            for offset in range(point_count)
        )


@dataclass(frozen=True)
class CustomMapSpec:
    map_id: str
    geometry: CustomTrackGeometry
    obstacles: tuple[CustomObstacle, ...]
    max_steps: int
    frame_skip: int
    generator: tuple[tuple[str, Any], ...] = ()
    metadata: tuple[tuple[str, Any], ...] = ()
    schema_version: int = 2

    map_kind = "custom"

    def __post_init__(self) -> None:
        if not isinstance(self.map_id, str) or not CUSTOM_MAP_ID_PATTERN.fullmatch(self.map_id):
            raise ValueError(
                "map_id must match custom-track-[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
            )
        if not isinstance(self.geometry, CustomTrackGeometry):
            raise ValueError("geometry must be a CustomTrackGeometry")
        if not isinstance(self.obstacles, tuple):
            raise ValueError("obstacles must be a tuple")
        if len(self.obstacles) > MAX_CUSTOM_OBSTACLES:
            raise ValueError(f"at most {MAX_CUSTOM_OBSTACLES} custom obstacles are allowed")
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
        if self.schema_version != 2:
            raise ValueError(f"unsupported custom map schema_version: {self.schema_version}")
        if not isinstance(self.generator, tuple):
            raise ValueError("generator must be a tuple of key/value pairs")
        if not isinstance(self.metadata, tuple):
            raise ValueError("metadata must be a tuple of key/value pairs")
        for name, values in (("generator", self.generator), ("metadata", self.metadata)):
            if not all(
                isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], str)
                for pair in values
            ):
                raise ValueError(f"{name} must contain string key/value pairs")
        object.__setattr__(self, "generator", tuple(sorted(self.generator)))
        object.__setattr__(self, "metadata", tuple(sorted(self.metadata)))

    @property
    def obstacle_mode(self) -> str:
        return "custom_only"


__all__ = [
    "CUSTOM_MAP_ID_PATTERN",
    "CustomMapSpec",
    "CustomObstacle",
    "CustomTrackGeometry",
    "MAX_CUSTOM_OBSTACLES",
    "MAX_FRAME_SKIP",
    "MAX_OBSTACLE_RADIUS",
    "MAX_SEED",
    "MAX_STEPS",
    "MIN_OBSTACLE_RADIUS",
]
