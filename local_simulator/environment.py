from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import gymnasium as gym

from core.track_variables import ObstacleSpec
from core.vendor.car_racing import CarRacing, TRACK_WIDTH
from env_wrapper import CarEnvironment
from .schema import MapSpec


@dataclass(frozen=True)
class TrackSnapshot:
    points: tuple[tuple[float, float, float, float], ...]
    width: float


@dataclass(frozen=True)
class MapBundle:
    spec: MapSpec
    track: TrackSnapshot
    official_obstacles: tuple[tuple[float, float], ...]
    custom_obstacles: tuple[ObstacleSpec, ...]


def create_environment(
    spec: MapSpec,
    render_mode: str | None = "rgb_array",
) -> tuple[CarEnvironment, CarRacing]:
    raw_environment = CarRacing(
        continuous=True,
        render_mode=render_mode,
    )
    limited_environment = gym.wrappers.TimeLimit(
        raw_environment,
        max_episode_steps=spec.max_steps * spec.frame_skip + 200,
    )
    return (
        CarEnvironment(
            limited_environment,
            skip_frames=spec.frame_skip,
        ),
        raw_environment,
    )


def reset_environment(
    environment: CarEnvironment,
    spec: MapSpec,
) -> tuple[Any, dict[str, Any]]:
    options: dict[str, Any] = {}
    if spec.obstacle_mode != "custom_only":
        options["track_id"] = spec.track_id
    observation, info = environment.reset(seed=spec.seed, options=options)
    custom_obstacles = map_custom_obstacles(environment, spec)
    attach_custom_obstacles(environment, custom_obstacles)
    return observation, info


def snapshot_track(environment: CarEnvironment) -> TrackSnapshot:
    track = getattr(environment.unwrapped, "track", None)
    if not track:
        raise RuntimeError("the environment must be reset before taking a snapshot")
    points = tuple(
        (
            float(alpha),
            float(beta),
            float(x),
            float(y),
        )
        for alpha, beta, x, y in track
    )
    return TrackSnapshot(points=points, width=float(TRACK_WIDTH))


def _custom_obstacle_position(
    environment: CarEnvironment,
    progress: float,
    lateral: float,
    radius: float,
) -> tuple[float, float]:
    track = environment.unwrapped.track
    index = min(len(track) - 1, max(0, round(progress * (len(track) - 1))))
    _alpha, beta, x, y = track[index]
    max_offset = min(TRACK_WIDTH * 0.6, TRACK_WIDTH - radius)
    offset = lateral * max_offset
    return (
        float(x + offset * math.cos(beta)),
        float(y + offset * math.sin(beta)),
    )


def map_custom_obstacles(
    environment: CarEnvironment,
    spec: MapSpec,
) -> tuple[ObstacleSpec, ...]:
    if spec.obstacle_mode == "official":
        return ()
    if not getattr(environment.unwrapped, "track", None):
        raise RuntimeError(
            "the environment must be reset before mapping custom obstacles"
        )
    return tuple(
        ObstacleSpec(
            position=_custom_obstacle_position(
                environment,
                obstacle.progress,
                obstacle.lateral,
                obstacle.radius,
            ),
            radius=obstacle.radius,
        )
        for obstacle in spec.obstacles
    )


def attach_custom_obstacles(
    environment: CarEnvironment,
    obstacles: tuple[ObstacleSpec, ...],
) -> None:
    if not obstacles:
        return
    if not getattr(environment.unwrapped, "track", None):
        raise RuntimeError(
            "the environment must be reset before attaching obstacles"
        )
    environment.unwrapped._create_obstacles(obstacles)


def official_obstacle_positions(
    environment: CarEnvironment,
) -> list[tuple[float, float]]:
    return [
        (float(body.position.x), float(body.position.y))
        for body in environment.unwrapped.obstacles
    ]


def build_map_bundle(spec: MapSpec) -> MapBundle:
    environment, _raw_environment = create_environment(spec, render_mode=None)
    try:
        reset_environment(environment, spec)
        custom_obstacles = map_custom_obstacles(environment, spec)
        official_count = 6 if spec.obstacle_mode != "custom_only" else 0
        all_obstacles = official_obstacle_positions(environment)
        official_obstacles = tuple(all_obstacles[:official_count])
        return MapBundle(
            spec=spec,
            track=snapshot_track(environment),
            official_obstacles=official_obstacles,
            custom_obstacles=custom_obstacles,
        )
    finally:
        environment.close()
