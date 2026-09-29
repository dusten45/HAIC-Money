from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping

from .track_model import (
    MAX_CUSTOM_OBSTACLES,
    MAX_FRAME_SKIP,
    MAX_OBSTACLE_RADIUS,
    MAX_SEED,
    MAX_STEPS,
    MIN_OBSTACLE_RADIUS,
    CustomMapSpec,
    CustomObstacle,
    CustomTrackGeometry,
    _require_float,
    _require_int,
)


LEGACY_SCHEMA_VERSION = 1
MAP_SCHEMA_VERSION = 2
RUN_SCHEMA_VERSION = 2
# Kept as a compatibility alias for the run-log module until its schema is migrated.
SCHEMA_VERSION = LEGACY_SCHEMA_VERSION
OBSTACLE_MODES = frozenset(
    {"official", "custom_only", "official_plus_custom"}
)


@dataclass(frozen=True)
class MapSpec:
    track_id: int
    seed: int
    obstacle_mode: str
    obstacles: tuple[CustomObstacle, ...]
    max_steps: int
    frame_skip: int
    schema_version: int = MAP_SCHEMA_VERSION
    metadata: tuple[tuple[str, Any], ...] = ()

    map_kind = "official"

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
        if self.schema_version not in (LEGACY_SCHEMA_VERSION, MAP_SCHEMA_VERSION):
            raise ValueError(f"unsupported map schema_version: {self.schema_version}")
        if not isinstance(self.metadata, tuple):
            raise ValueError("metadata must be a tuple of key/value pairs")

    @property
    def map_id(self) -> str:
        return f"official-track-{self.track_id}-seed-{self.seed}"


MapDocument = MapSpec | CustomMapSpec


def _obstacles_from_payload(raw_obstacles: object) -> tuple[CustomObstacle, ...]:
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
    return tuple(obstacles)


def _metadata_from_payload(payload: Mapping[str, Any], name: str) -> tuple[tuple[str, Any], ...]:
    raw_metadata = payload.get(name, {})
    if raw_metadata is None:
        raw_metadata = {}
    if not isinstance(raw_metadata, Mapping):
        raise ValueError(f"{name} must be an object")
    return tuple(sorted(raw_metadata.items()))


def _custom_geometry_from_payload(payload: Mapping[str, Any]) -> CustomTrackGeometry:
    raw_geometry = payload.get("geometry")
    if not isinstance(raw_geometry, Mapping):
        raise ValueError("geometry must be an object for custom maps")
    raw_centerline = raw_geometry.get("centerline")
    if not isinstance(raw_centerline, (list, tuple)):
        raise ValueError("geometry.centerline must be an array")
    centerline: list[tuple[float, float]] = []
    for index, raw_point in enumerate(raw_centerline):
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
            raise ValueError(f"geometry.centerline[{index}] must contain two values")
        centerline.append(
            (
                _require_float(
                    f"geometry.centerline[{index}].x",
                    raw_point[0],
                    -100_000.0,
                    100_000.0,
                ),
                _require_float(
                    f"geometry.centerline[{index}].y",
                    raw_point[1],
                    -100_000.0,
                    100_000.0,
                ),
            )
        )
    return CustomTrackGeometry(
        centerline=tuple(centerline),
        width=raw_geometry.get("width"),
        start_index=raw_geometry.get("start_index", 0),
        direction=raw_geometry.get("direction", 1),
    )


def _generator_from_payload(payload: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    raw_generator = payload.get("generator", {})
    if raw_generator is None:
        raw_generator = {}
    if not isinstance(raw_generator, Mapping):
        raise ValueError("generator must be an object")
    return tuple(sorted(raw_generator.items()))


def validate_map_payload(payload: object) -> MapDocument:
    if not isinstance(payload, Mapping):
        raise ValueError("map payload must be an object")

    raw_schema_version = payload.get("schema_version", LEGACY_SCHEMA_VERSION)
    if raw_schema_version not in (LEGACY_SCHEMA_VERSION, MAP_SCHEMA_VERSION):
        raise ValueError(f"unsupported map schema_version: {raw_schema_version}")

    map_kind = payload.get("map_kind", "official")
    if map_kind == "custom":
        if raw_schema_version != MAP_SCHEMA_VERSION:
            raise ValueError("custom maps require schema_version 2")
        return CustomMapSpec(
            map_id=payload.get("map_id"),
            geometry=_custom_geometry_from_payload(payload),
            obstacles=_obstacles_from_payload(payload.get("obstacles", [])),
            max_steps=payload.get("max_steps", 2000),
            frame_skip=payload.get("frame_skip", 4),
            generator=_generator_from_payload(payload),
            metadata=_metadata_from_payload(payload, "metadata"),
        )
    if map_kind != "official":
        raise ValueError("map_kind must be official or custom")

    return MapSpec(
        track_id=payload.get("track_id"),
        seed=payload.get("seed"),
        obstacle_mode=payload.get("obstacle_mode", "official"),
        obstacles=_obstacles_from_payload(payload.get("obstacles", [])),
        max_steps=payload.get("max_steps", 2000),
        frame_skip=payload.get("frame_skip", 4),
        schema_version=MAP_SCHEMA_VERSION,
        metadata=_metadata_from_payload(payload, "metadata"),
    )


def _obstacles_to_dict(obstacles: tuple[CustomObstacle, ...]) -> list[dict[str, float]]:
    return [
        {
            "progress": obstacle.progress,
            "lateral": obstacle.lateral,
            "radius": obstacle.radius,
        }
        for obstacle in obstacles
    ]


def map_to_dict(document: MapDocument) -> dict[str, Any]:
    if isinstance(document, CustomMapSpec):
        result: dict[str, Any] = {
            "schema_version": MAP_SCHEMA_VERSION,
            "map_id": document.map_id,
            "map_kind": "custom",
            "geometry": {
                "centerline": [list(point) for point in document.geometry.centerline],
                "width": document.geometry.width,
                "start_index": document.geometry.start_index,
                "direction": document.geometry.direction,
            },
            "obstacle_mode": "custom_only",
            "obstacles": _obstacles_to_dict(document.obstacles),
            "max_steps": document.max_steps,
            "frame_skip": document.frame_skip,
        }
        if document.generator:
            result["generator"] = dict(document.generator)
        if document.metadata:
            result["metadata"] = dict(document.metadata)
        return result

    if not isinstance(document, MapSpec):
        raise TypeError("document must be a MapSpec or CustomMapSpec")
    result = {
        "schema_version": MAP_SCHEMA_VERSION,
        "map_id": document.map_id,
        "map_kind": "official",
        "track_id": document.track_id,
        "seed": document.seed,
        "obstacle_mode": document.obstacle_mode,
        "obstacles": _obstacles_to_dict(document.obstacles),
        "max_steps": document.max_steps,
        "frame_skip": document.frame_skip,
    }
    if document.metadata:
        result["metadata"] = dict(document.metadata)
    return result


def map_from_dict(payload: object) -> MapDocument:
    return validate_map_payload(payload)


def map_fingerprint(document: MapDocument) -> str:
    encoded = json.dumps(
        map_to_dict(document),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(encoded.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "CustomMapSpec",
    "CustomObstacle",
    "CustomTrackGeometry",
    "LEGACY_SCHEMA_VERSION",
    "MAP_SCHEMA_VERSION",
    "MapDocument",
    "MapSpec",
    "OBSTACLE_MODES",
    "RUN_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "map_from_dict",
    "map_fingerprint",
    "map_to_dict",
    "validate_map_payload",
]
