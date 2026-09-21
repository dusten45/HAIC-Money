from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np

DEFAULT_GRASS_FRICTION_MULTIPLIER = 0.6

OBSTACLE_COUNT = 6
OBSTACLE_RADIUS = 1.2
LATERAL_OFFSET_RATIO = 0.6
EXCLUDE_START_RATIO = 0.1
EXCLUDE_FINISH_RATIO = 0.05
MIN_GAP_INDICES = 20

OBSTACLE_RNG_STREAM = 0x4F425354
_MAX_SEED_COMPONENT = 0xFFFFFFFF

TrackPoint = tuple[float, float, float, float]


@dataclass(frozen=True)
class ObstacleSpec:
    position: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class TrackVariables:
    grass_friction_multiplier: float
    obstacles: tuple[ObstacleSpec, ...] = ()


def validate_track_id(track_id: int) -> int:
    if isinstance(track_id, bool) or not isinstance(track_id, int):
        raise TypeError("track_id must be an integer")
    result = int(track_id)
    if result <= 0:
        raise ValueError("track_id must be a positive integer")
    return result


def build_track_variables(track_id: int, seed: int) -> TrackVariables:
    validate_track_id(track_id)
    _seed_component("seed", seed)
    return TrackVariables(grass_friction_multiplier=DEFAULT_GRASS_FRICTION_MULTIPLIER)


def _seed_component(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= _MAX_SEED_COMPONENT:
        raise ValueError(f"{name} must be between 0 and {_MAX_SEED_COMPONENT}")
    return result


def _obstacle_rng(track_id: int, seed: int) -> np.random.Generator:
    track_id = validate_track_id(track_id)
    track_components = []
    while track_id:
        track_components.append(track_id & _MAX_SEED_COMPONENT)
        track_id >>= 32
    seed_sequence = np.random.SeedSequence([
        *track_components,
        _seed_component("seed", seed),
        OBSTACLE_RNG_STREAM,
    ])
    return np.random.default_rng(seed_sequence)


def _place_obstacles(
    track_id: int,
    seed: int,
    track: Sequence[TrackPoint],
    track_width: float,
    count: int = OBSTACLE_COUNT,
) -> tuple[ObstacleSpec, ...]:
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be non-negative")
    if not isinstance(track_width, (int, float, np.integer, np.floating)):
        raise TypeError("track_width must be a number")
    track_width = float(track_width)
    if not math.isfinite(track_width) or track_width <= OBSTACLE_RADIUS:
        raise ValueError("track_width must be finite and greater than OBSTACLE_RADIUS")
    validate_track_id(track_id)
    _seed_component("seed", seed)
    if count == 0:
        return ()

    track_points = tuple(track)
    lo = math.ceil(len(track_points) * EXCLUDE_START_RATIO)
    hi = math.floor(len(track_points) * (1.0 - EXCLUDE_FINISH_RATIO))
    eligible_count = max(0, hi - lo)

    compressed_count = eligible_count - (count - 1) * (MIN_GAP_INDICES - 1)
    if compressed_count < count:
        raise ValueError(
            "track is too short to place the requested obstacles with the minimum gap"
        )

    rng = _obstacle_rng(track_id, seed)
    compressed = np.sort(rng.choice(compressed_count, size=count, replace=False))
    indices = [
        lo + int(value) + order * (MIN_GAP_INDICES - 1)
        for order, value in enumerate(compressed)
    ]

    max_offset = min(
        track_width * LATERAL_OFFSET_RATIO,
        track_width - OBSTACLE_RADIUS,
    )
    specs: list[ObstacleSpec] = []
    for index in indices:
        try:
            _alpha, beta, x, y = track_points[index]
            beta, x, y = float(beta), float(x), float(y)
        except (TypeError, ValueError) as error:
            raise ValueError("each track point must contain numeric (alpha, beta, x, y)") from error
        if not all(math.isfinite(value) for value in (beta, x, y)):
            raise ValueError("track point beta, x, and y must be finite")

        offset = float(rng.uniform(-max_offset, max_offset))
        specs.append(ObstacleSpec(
            position=(
                x + offset * math.cos(beta),
                y + offset * math.sin(beta),
            ),
            radius=OBSTACLE_RADIUS,
        ))
    return tuple(specs)


def with_obstacles(
    variables: TrackVariables,
    track_id: int,
    seed: int,
    track: Sequence[TrackPoint],
    track_width: float,
) -> TrackVariables:
    return replace(
        variables,
        obstacles=_place_obstacles(track_id, seed, track, track_width),
    )
