from __future__ import annotations

import math
from typing import Any

from .environment import MapBundle, TrackSnapshot, build_map_bundle
from .schema import CustomMapSpec, MapDocument


def custom_track_snapshot(spec: CustomMapSpec) -> TrackSnapshot:
    points = spec.geometry.ordered_centerline()
    projected: list[tuple[float, float, float, float]] = []
    for index, (x, y) in enumerate(points):
        previous = points[(index - 1) % len(points)]
        following = points[(index + 1) % len(points)]
        tangent = math.atan2(following[1] - previous[1], following[0] - previous[0])
        normal_angle = tangent - math.pi / 2.0
        projected.append((index / len(points), normal_angle, x, y))
    return TrackSnapshot(points=tuple(projected), width=spec.geometry.width)


def _custom_obstacle_position(
    spec: CustomMapSpec,
    progress: float,
    lateral: float,
    radius: float,
) -> tuple[float, float]:
    track = custom_track_snapshot(spec)
    index = min(len(track.points) - 1, max(0, round(progress * (len(track.points) - 1))))
    _alpha, beta, x, y = track.points[index]
    if radius > spec.geometry.width:
        raise ValueError(
            f"obstacle radius {radius} exceeds custom road width {spec.geometry.width}"
        )
    max_offset = min(spec.geometry.width * 0.6, spec.geometry.width - radius)
    offset = lateral * max_offset
    return (
        float(x + offset * math.cos(beta)),
        float(y + offset * math.sin(beta)),
    )


def _bundle_preview(bundle: MapBundle) -> dict[str, Any]:
    return {
        "track": {
            "points": [list(point) for point in bundle.track.points],
            "width": bundle.track.width,
        },
        "official_obstacles": [list(position) for position in bundle.official_obstacles],
        "custom_obstacles": [
            {
                "position": list(obstacle.position),
                "radius": obstacle.radius,
            }
            for obstacle in bundle.custom_obstacles
        ],
    }


def _custom_preview(spec: CustomMapSpec) -> dict[str, Any]:
    track = custom_track_snapshot(spec)
    return {
        "track": {
            "points": [list(point) for point in track.points],
            "width": track.width,
        },
        "official_obstacles": [],
        "custom_obstacles": [
            {
                "position": list(
                    _custom_obstacle_position(
                        spec,
                        obstacle.progress,
                        obstacle.lateral,
                        obstacle.radius,
                    )
                ),
                "radius": obstacle.radius,
            }
            for obstacle in spec.obstacles
        ],
    }


def build_map_preview(document: MapDocument) -> dict[str, Any]:
    if isinstance(document, CustomMapSpec):
        return _custom_preview(document)
    return _bundle_preview(build_map_bundle(document))


__all__ = ["build_map_preview", "custom_track_snapshot"]
