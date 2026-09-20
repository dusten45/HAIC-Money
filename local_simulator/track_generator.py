from __future__ import annotations

from hashlib import sha256
import json
import math

from .track_model import (
    MAX_SEED,
    CustomMapSpec,
    CustomTrackGeometry,
    _require_float,
    _require_int,
)


TEMPLATES = frozenset({"oval", "s_curve", "hairpin", "chicane", "technical"})
GENERATOR_VERSION = 2
CORNER_COUNT_RANGES = {
    "oval": (4, 4),
    "s_curve": (6, 8),
    "hairpin": (6, 9),
    "chicane": (7, 10),
    "technical": (9, 12),
}
MIN_CENTERLINE_POINTS = 12
MAX_CENTERLINE_POINTS = 4096


class _TrackRng:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 4294967296.0

    def uniform(self, minimum: float, maximum: float) -> float:
        return minimum + (maximum - minimum) * self.random()

    def index(self, length: int) -> int:
        return min(length - 1, int(self.random() * length))


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


def _corner_sequence(template: str, rng: _TrackRng) -> tuple[str, ...]:
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {sorted(TEMPLATES)}")
    minimum, maximum = CORNER_COUNT_RANGES[template]
    count = minimum + rng.index(maximum - minimum + 1)

    if template == "oval":
        return ("left:wide",) * count

    if template == "s_curve":
        direction = "left" if rng.random() < 0.5 else "right"
        classes = ("wide", "medium", "tight")
        return tuple(
            f"{direction if index % 2 == 0 else ('right' if direction == 'left' else 'left')}:{classes[rng.index(len(classes))]}"
            for index in range(count)
        )

    if template == "hairpin":
        direction = "left" if rng.random() < 0.5 else "right"
        sequence = [
            f"{direction}:{('wide' if rng.random() < 0.4 else 'medium')}"
            for _ in range(count)
        ]
        sequence[count // 3] = f"{direction}:hairpin"
        other_direction = "right" if direction == "left" else "left"
        sequence[(2 * count) // 3] = f"{other_direction}:hairpin"
        return tuple(sequence)

    if template == "chicane":
        direction = "left" if rng.random() < 0.5 else "right"
        return tuple(
            f"{direction if index % 2 == 0 else ('right' if direction == 'left' else 'left')}:{'tight' if rng.random() < 0.65 else 'medium'}"
            for index in range(count)
        )

    classes = ["wide", "medium", "tight"]
    classes.extend(classes[rng.index(len(classes))] for _ in range(count - len(classes)))
    for index in range(len(classes) - 1, 0, -1):
        swap_index = rng.index(index + 1)
        classes[index], classes[swap_index] = classes[swap_index], classes[index]
    return tuple(
        f"{'left' if rng.random() < 0.5 else 'right'}:{corner_class}"
        for corner_class in classes
    )


def _round_coordinate(value: float) -> float:
    magnitude = math.floor(abs(value) * 100_000.0 + 0.5) / 100_000.0
    rounded = math.copysign(magnitude, value)
    return 0.0 if rounded == 0.0 else rounded


def _interpolate(
    start: tuple[float, float],
    end: tuple[float, float],
    ratio: float,
) -> tuple[float, float]:
    return (
        start[0] + (end[0] - start[0]) * ratio,
        start[1] + (end[1] - start[1]) * ratio,
    )


def _quadratic_bezier(
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
    ratio: float,
) -> tuple[float, float]:
    inverse = 1.0 - ratio
    return (
        inverse * inverse * start[0] + 2.0 * inverse * ratio * control[0] + ratio * ratio * end[0],
        inverse * inverse * start[1] + 2.0 * inverse * ratio * control[1] + ratio * ratio * end[1],
    )


def _angle_gaps(rng: _TrackRng, count: int) -> tuple[float, ...]:
    weights = tuple(0.5 + rng.random() for _ in range(count))
    weight_total = sum(weights)
    raw_gaps = tuple(2.0 * math.pi * weight / weight_total for weight in weights)
    minimum = math.radians(20.0)
    maximum = math.radians(100.0)
    base = 2.0 * math.pi / count

    # Pull the random gaps toward equal spacing only as much as needed to
    # satisfy the geometric bounds. This preserves seeded variation while
    # avoiding seeds that fail after a finite number of random retries.
    factor = 1.0
    for gap in raw_gaps:
        deviation = gap - base
        if deviation > 0.0:
            factor = min(factor, (maximum - base) / deviation)
        elif deviation < 0.0:
            factor = min(factor, (base - minimum) / -deviation)
    factor = max(0.0, min(1.0, factor * 0.999))
    return tuple(base + factor * (gap - base) for gap in raw_gaps)


def _generate_geometry(
    template: str,
    design_seed: int,
    width: float,
) -> tuple[CustomTrackGeometry, tuple[str, ...]]:
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {sorted(TEMPLATES)}")
    design_seed = _require_int("design_seed", design_seed, 0, MAX_SEED)
    width = _require_float("width", width, 0.5, 100.0)
    rng = _TrackRng(design_seed)
    sequence = _corner_sequence(template, rng)
    count = len(sequence)
    gaps = _angle_gaps(rng, count)

    scale = max(1.0, width / 8.0)
    radius_x = (64.0 + rng.uniform(-6.0, 6.0)) * scale
    radius_y = (40.0 + rng.uniform(-4.0, 4.0)) * scale
    phase = rng.uniform(0.0, 2.0 * math.pi)
    template_amplitude = {
        "oval": 0.015,
        "s_curve": 0.055,
        "hairpin": 0.08,
        "chicane": 0.085,
        "technical": 0.11,
    }[template]
    class_amplitude = {"wide": 0.035, "medium": 0.085, "tight": 0.14, "hairpin": 0.22}

    anchors: list[tuple[float, float]] = []
    angle = rng.uniform(0.0, 2.0 * math.pi)
    for index, token in enumerate(sequence):
        direction, corner_class = token.split(":", 1)
        signed_class_offset = class_amplitude[corner_class] * (1.0 if direction == "left" else -1.0)
        harmonic = 1 + (index % 3)
        radial_modulation = template_amplitude * math.sin(harmonic * angle + phase)
        radial_jitter = rng.uniform(-0.025, 0.025)
        radial_scale = 1.0 + signed_class_offset + radial_modulation + radial_jitter
        anchors.append(
            (
                radius_x * radial_scale * math.cos(angle),
                radius_y * radial_scale * math.sin(angle),
            )
        )
        angle += gaps[index]

    trim_ratios = {"wide": 0.30, "medium": 0.25, "tight": 0.21, "hairpin": 0.19}
    incoming_points: list[tuple[float, float]] = []
    outgoing_points: list[tuple[float, float]] = []
    for index, anchor in enumerate(anchors):
        previous = anchors[(index - 1) % count]
        following = anchors[(index + 1) % count]
        incoming_length = _distance(anchor, previous)
        outgoing_length = _distance(anchor, following)
        corner_class = sequence[index].split(":", 1)[1]
        trim = min(
            trim_ratios[corner_class] * min(incoming_length, outgoing_length),
            0.35 * min(incoming_length, outgoing_length),
        )
        incoming_points.append(_interpolate(anchor, previous, trim / incoming_length))
        outgoing_points.append(_interpolate(anchor, following, trim / outgoing_length))

    points: list[tuple[float, float]] = [incoming_points[0]]
    for index, anchor in enumerate(anchors):
        outgoing = outgoing_points[index]
        control_length = max(_distance(anchor, incoming_points[index]), _distance(anchor, outgoing))
        curve_steps = max(1, math.ceil(2.0 * control_length / 4.0))
        for step in range(1, curve_steps + 1):
            points.append(
                _quadratic_bezier(
                    incoming_points[index],
                    anchor,
                    outgoing,
                    step / curve_steps,
                )
            )

        next_index = (index + 1) % count
        straight_start = outgoing_points[index]
        straight_end = incoming_points[next_index]
        straight_steps = max(1, math.ceil(_distance(straight_start, straight_end) / 4.0))
        last_step = straight_steps if next_index != 0 else straight_steps - 1
        for step in range(1, last_step + 1):
            points.append(_interpolate(straight_start, straight_end, step / straight_steps))

    quantized = tuple(
        (_round_coordinate(point[0]), _round_coordinate(point[1]))
        for point in points
    )
    if len(quantized) > MAX_CENTERLINE_POINTS:
        raise ValueError(f"generated centerline exceeds {MAX_CENTERLINE_POINTS} points")
    geometry = CustomTrackGeometry(centerline=quantized, width=width)
    validate_custom_geometry(geometry)
    return geometry, sequence


def generate_custom_map(
    map_id: str,
    design_seed: int,
    template: str,
    width: float = 8.0,
    max_steps: int = 2000,
    frame_skip: int = 4,
) -> CustomMapSpec:
    geometry, corner_sequence = _generate_geometry(template, design_seed, width)
    return CustomMapSpec(
        map_id=map_id,
        geometry=geometry,
        obstacles=(),
        max_steps=max_steps,
        frame_skip=frame_skip,
        generator=(
            ("design_seed", design_seed),
            ("generator_version", GENERATOR_VERSION),
            ("corner_count", len(corner_sequence)),
            ("corner_sequence", corner_sequence),
            ("template", template),
        ),
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
