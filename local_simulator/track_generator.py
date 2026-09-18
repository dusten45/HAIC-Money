from __future__ import annotations

from hashlib import sha256
import json
import math
import random
from typing import Iterable

from .track_model import (
    MAX_SEED,
    CustomMapSpec,
    CustomTrackGeometry,
    _require_float,
    _require_int,
)


TEMPLATES = frozenset({"oval", "s_curve", "hairpin", "chicane"})
MIN_CENTERLINE_POINTS = 12


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _cross(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _on_segment(
    first: tuple[float, float],
    second: tuple[float, float],
    point: tuple[float, float],
) -> bool:
    return (
        min(first[0], second[0]) - 1e-9 <= point[0] <= max(first[0], second[0]) + 1e-9
        and min(first[1], second[1]) - 1e-9 <= point[1] <= max(first[1], second[1]) + 1e-9
    )


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    first_left = _cross(first_start, first_end, second_start)
    first_right = _cross(first_start, first_end, second_end)
    second_left = _cross(second_start, second_end, first_start)
    second_right = _cross(second_start, second_end, first_end)
    epsilon = 1e-9
    if (
        ((first_left > epsilon and first_right < -epsilon) or (first_left < -epsilon and first_right > epsilon))
        and ((second_left > epsilon and second_right < -epsilon) or (second_left < -epsilon and second_right > epsilon))
    ):
        return True
    if abs(first_left) <= epsilon and _on_segment(first_start, first_end, second_start):
        return True
    if abs(first_right) <= epsilon and _on_segment(first_start, first_end, second_end):
        return True
    if abs(second_left) <= epsilon and _on_segment(second_start, second_end, first_start):
        return True
    if abs(second_right) <= epsilon and _on_segment(second_start, second_end, first_end):
        return True
    return False


def _are_adjacent(first_index: int, second_index: int, count: int) -> bool:
    if first_index == second_index:
        return True
    if (first_index + 1) % count == second_index:
        return True
    if (second_index + 1) % count == first_index:
        return True
    return False


def validate_custom_geometry(geometry: CustomTrackGeometry) -> None:
    points = geometry.centerline
    if len(points) < MIN_CENTERLINE_POINTS:
        raise ValueError(
            f"centerline must contain at least {MIN_CENTERLINE_POINTS} points"
        )
    if geometry.width <= 0 or not math.isfinite(geometry.width):
        raise ValueError("width must be positive and finite")

    segment_lengths = [
        _distance(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    ]
    if any(length <= 1e-6 for length in segment_lengths):
        raise ValueError("centerline contains a zero-length segment")
    if sum(segment_lengths) < geometry.width * 8.0:
        raise ValueError("centerline is too short for the selected width")

    for first_index in range(len(points)):
        first_start = points[first_index]
        first_end = points[(first_index + 1) % len(points)]
        for second_index in range(first_index + 1, len(points)):
            if _are_adjacent(first_index, second_index, len(points)):
                continue
            second_start = points[second_index]
            second_end = points[(second_index + 1) % len(points)]
            if _segments_intersect(first_start, first_end, second_start, second_end):
                raise ValueError(
                    f"centerline self-intersection between segments {first_index} and {second_index}"
                )

    for index, point in enumerate(points):
        previous = points[(index - 1) % len(points)]
        following = points[(index + 1) % len(points)]
        incoming = (point[0] - previous[0], point[1] - previous[1])
        outgoing = (following[0] - point[0], following[1] - point[1])
        incoming_length = math.hypot(*incoming)
        outgoing_length = math.hypot(*outgoing)
        cosine = (
            (incoming[0] * outgoing[0] + incoming[1] * outgoing[1])
            / (incoming_length * outgoing_length)
        )
        if cosine < -0.995:
            raise ValueError(f"centerline turn is too sharp at point {index}")


def custom_geometry_fingerprint(geometry: CustomTrackGeometry) -> str:
    payload = {
        "centerline": [list(point) for point in geometry.centerline],
        "width": geometry.width,
        "start_index": geometry.start_index,
        "direction": geometry.direction,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _radial_loop(
    count: int,
    radius_x: float,
    radius_y: float,
    harmonic: int,
    amplitude: float,
    rng: random.Random,
) -> tuple[tuple[float, float], ...]:
    phase = rng.uniform(-0.18, 0.18)
    points: list[tuple[float, float]] = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        modulation = 1.0 + amplitude * math.sin(harmonic * angle + phase)
        points.append(
            (
                radius_x * modulation * math.cos(angle),
                radius_y * modulation * math.sin(angle),
            )
        )
    return tuple(points)


def _hairpin_loop(rng: random.Random) -> tuple[tuple[float, float], ...]:
    radius = 22.0 + rng.uniform(-2.0, 2.0)
    length = 92.0 + rng.uniform(-5.0, 5.0)
    line_count = 12
    arc_count = 12
    points: list[tuple[float, float]] = []
    for index in range(line_count):
        ratio = index / line_count
        points.append((-length / 2.0 + length * ratio, radius))
    for index in range(arc_count):
        angle = math.pi / 2.0 - math.pi * index / arc_count
        points.append((length / 2.0 + radius * math.cos(angle), radius * math.sin(angle)))
    for index in range(line_count):
        ratio = index / line_count
        points.append((length / 2.0 - length * ratio, -radius))
    for index in range(arc_count):
        angle = -math.pi / 2.0 - math.pi * index / arc_count
        points.append((-length / 2.0 + radius * math.cos(angle), radius * math.sin(angle)))
    return tuple(points)


def _generate_geometry(template: str, design_seed: int, width: float) -> CustomTrackGeometry:
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {sorted(TEMPLATES)}")
    design_seed = _require_int("design_seed", design_seed, 0, MAX_SEED)
    width = _require_float("width", width, 0.5, 100.0)
    rng = random.Random(design_seed)
    count = 48
    if template == "oval":
        points = _radial_loop(
            count,
            64.0 + rng.uniform(-6.0, 6.0),
            38.0 + rng.uniform(-4.0, 4.0),
            2,
            0.025,
            rng,
        )
    elif template == "s_curve":
        points = _radial_loop(
            count,
            62.0 + rng.uniform(-5.0, 5.0),
            42.0 + rng.uniform(-4.0, 4.0),
            1,
            0.07,
            rng,
        )
    elif template == "chicane":
        points = _radial_loop(
            count,
            60.0 + rng.uniform(-5.0, 5.0),
            40.0 + rng.uniform(-4.0, 4.0),
            3,
            0.055,
            rng,
        )
    else:
        points = _hairpin_loop(rng)
    geometry = CustomTrackGeometry(centerline=points, width=width)
    validate_custom_geometry(geometry)
    return geometry


def generate_custom_map(
    map_id: str,
    design_seed: int,
    template: str,
    width: float = 8.0,
    max_steps: int = 2000,
    frame_skip: int = 4,
) -> CustomMapSpec:
    geometry = _generate_geometry(template, design_seed, width)
    return CustomMapSpec(
        map_id=map_id,
        geometry=geometry,
        obstacles=(),
        max_steps=max_steps,
        frame_skip=frame_skip,
        generator=(("design_seed", design_seed), ("template", template)),
    )


def generate_custom_maps(
    start_seed: int,
    count: int,
    template: str,
    prefix: str = "custom-track",
) -> tuple[CustomMapSpec, ...]:
    start_seed = _require_int("start_seed", start_seed, 0, MAX_SEED)
    count = _require_int("count", count, 0, 64)
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("prefix must be a non-empty string")
    return tuple(
        generate_custom_map(
            map_id=f"{prefix}-{seed:04d}",
            design_seed=seed,
            template=template,
        )
        for seed in range(start_seed, start_seed + count)
    )


__all__ = [
    "TEMPLATES",
    "custom_geometry_fingerprint",
    "generate_custom_map",
    "generate_custom_maps",
    "validate_custom_geometry",
]
