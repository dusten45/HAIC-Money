"""Track Lab geometry and obstacle placement for local training episodes."""

from __future__ import annotations

import math
from typing import Any

from core.finish_line import FinishLineTracker
from core.track_variables import ObstacleSpec
from core.vendor.car_racing import CarRacing, TRACK_DETAIL_STEP, TRACK_WIDTH
from training.site_maps import SiteMapSpec


class SiteCustomCarRacing(CarRacing):
    """Build a CarRacing road from a Track Lab centerline."""

    def __init__(self, site_map: SiteMapSpec, render_mode: str | None = None) -> None:
        if site_map.map_kind != "custom" or site_map.geometry is None:
            raise ValueError("SiteCustomCarRacing requires a custom site map")
        super().__init__(continuous=True, render_mode=render_mode)
        self.site_map = site_map
        self.custom_geometry = site_map.geometry

    def _ordered_centerline(self) -> tuple[tuple[float, float], ...]:
        geometry = self.custom_geometry
        count = len(geometry.centerline)
        return tuple(
            geometry.centerline[(geometry.start_index + geometry.direction * offset) % count]
            for offset in range(count)
        )

    def _normal_angles(self, points: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
        angles: list[float] = []
        for index, point in enumerate(points):
            previous = points[(index - 1) % len(points)]
            following = points[(index + 1) % len(points)]
            tangent = math.atan2(following[1] - previous[1], following[0] - previous[0])
            angles.append(tangent - math.pi / 2.0)
        return tuple(angles)

    def _create_track(self) -> bool:
        points = self._ordered_centerline()
        normals = self._normal_angles(points)
        half_width = self.custom_geometry.width
        self.road = []
        self.road_poly = []
        track: list[tuple[float, float, float, float]] = []

        for index, ((x, y), beta) in enumerate(zip(points, normals)):
            previous_index = (index - 1) % len(points)
            previous_x, previous_y = points[previous_index]
            previous_beta = normals[previous_index]
            road1_l = (
                x - half_width * math.cos(beta),
                y - half_width * math.sin(beta),
            )
            road1_r = (
                x + half_width * math.cos(beta),
                y + half_width * math.sin(beta),
            )
            road2_l = (
                previous_x - half_width * math.cos(previous_beta),
                previous_y - half_width * math.sin(previous_beta),
            )
            road2_r = (
                previous_x + half_width * math.cos(previous_beta),
                previous_y + half_width * math.sin(previous_beta),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            tile = self.world.CreateStaticBody(fixtures=self.fd_tile)
            tile.userData = tile
            tile.color = self.road_color + 0.01 * (index % 3) * 255
            tile.road_visited = False
            tile.road_friction = 1.0
            tile.idx = index
            tile.fixtures[0].sensor = True
            self.road.append(tile)
            self.road_poly.append((vertices, tile.color))
            track.append((index / len(points), beta, x, y))

        self.track = track
        return True

    def _replace_finish_line_for_custom_width(self) -> None:
        _progress, beta, x, y = self.track[0]
        self.finish_line_tracker = FinishLineTracker(
            center=(float(x), float(y)),
            forward=(-math.sin(beta), math.cos(beta)),
            half_width=self.custom_geometry.width,
            half_depth=TRACK_DETAIL_STEP * 0.25,
            qualification_ratio=self.lap_complete_percent,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = super().reset(seed=seed, options=options)
        self._replace_finish_line_for_custom_width()
        return observation, info


def site_obstacle_specs(site_map: SiteMapSpec, track: list[tuple[float, float, float, float]]) -> tuple[ObstacleSpec, ...]:
    """Convert Track Lab progress/lateral coordinates to physical road positions."""
    if site_map.obstacle_mode == "official" or not site_map.obstacles:
        return ()
    if not track:
        raise RuntimeError("the environment must be reset before mapping site obstacles")
    width = site_map.geometry.width if site_map.geometry is not None else TRACK_WIDTH
    specs: list[ObstacleSpec] = []
    for obstacle in site_map.obstacles:
        max_offset = min(width * 0.6, width - obstacle.radius)
        if max_offset < 0:
            raise ValueError(
                f"obstacle radius {obstacle.radius} exceeds site map road width {width}"
            )
        index = min(len(track) - 1, max(0, round(obstacle.progress * (len(track) - 1))))
        _progress, beta, x, y = track[index]
        offset = obstacle.lateral * max_offset
        specs.append(
            ObstacleSpec(
                position=(
                    float(x + offset * math.cos(beta)),
                    float(y + offset * math.sin(beta)),
                ),
                radius=obstacle.radius,
            )
        )
    return tuple(specs)


def attach_site_obstacles(environment: Any, site_map: SiteMapSpec) -> int:
    """Attach map-authored obstacles once after the wrapper's warmup period."""
    raw = getattr(environment, "unwrapped", environment)
    specs = site_obstacle_specs(site_map, raw.track)
    if specs:
        raw._create_obstacles(specs)
    return len(specs)


__all__ = ["SiteCustomCarRacing", "attach_site_obstacles", "site_obstacle_specs"]
