from __future__ import annotations

import math

from core.finish_line import FinishLineTracker
from core.vendor.car_racing import (
    CarRacing,
    TRACK_DETAIL_STEP,
)

from .track_model import CustomTrackGeometry


class CustomCarRacing(CarRacing):
    """CarRacing backend whose road tiles come from a stored centerline."""

    def __init__(
        self,
        geometry: CustomTrackGeometry,
        render_mode: str | None = None,
    ) -> None:
        super().__init__(render_mode=render_mode, continuous=True)
        self.custom_geometry = geometry

    def _normal_angles(
        self, points: tuple[tuple[float, float], ...]
    ) -> tuple[float, ...]:
        angles: list[float] = []
        for index, point in enumerate(points):
            previous = points[(index - 1) % len(points)]
            following = points[(index + 1) % len(points)]
            tangent = math.atan2(
                following[1] - previous[1],
                following[0] - previous[0],
            )
            angles.append(tangent - math.pi / 2.0)
        return tuple(angles)

    def _create_track(self) -> bool:
        points = self.custom_geometry.ordered_centerline()
        normals = self._normal_angles(points)
        half_width = self.custom_geometry.width
        self.road = []
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
        _alpha, beta, x, y = self.track[0]
        self.finish_line_tracker = FinishLineTracker(
            center=(float(x), float(y)),
            forward=(-math.sin(beta), math.cos(beta)),
            half_width=self.custom_geometry.width,
            half_depth=TRACK_DETAIL_STEP * 0.25,
            qualification_ratio=self.lap_complete_percent,
        )

    def reset(self, *, seed=None, options=None):
        del options
        observation, info = super().reset(seed=seed, options={})
        self._replace_finish_line_for_custom_width()
        return observation, info


__all__ = ["CustomCarRacing"]
