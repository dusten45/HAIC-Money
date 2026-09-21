from __future__ import annotations

from dataclasses import dataclass
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


TEMPLATES = frozenset({"oval", "s_curve", "hairpin", "chicane", "technical", "extreme_technical"})
GENERATOR_VERSION = 5
CORNER_COUNT_RANGES = {
    "oval": (4, 4),
    "s_curve": (6, 8),
    "hairpin": (6, 9),
    "chicane": (7, 10),
    "technical": (9, 12),
    "extreme_technical": (12, 16),
}
CORNER_RADIUS_WIDTH_RANGES = {
    "wide": (1.45, 3.2),
    "medium": (1.1, 2.7),
    "tight": (1.45, 2.0),
    "hairpin": (1.45, 1.9),
}
MAX_GENERATED_TRACK_WIDTH = 9.0
MIN_CENTERLINE_POINTS = 12
MAX_CENTERLINE_POINTS = 4096


@dataclass(frozen=True)
class _ExtremeRoute:
    vertices: tuple[tuple[float, float], ...]
    corner_sequence: tuple[str, ...]
    s_section_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _ExtremeGeometryResult:
    geometry: CustomTrackGeometry
    corner_sequence: tuple[str, ...]
    corner_turn_degrees: tuple[float, ...]
    corner_radius_widths: tuple[float, ...]
    s_section_pairs: tuple[tuple[int, int], ...]
    s_section_connector_lengths: tuple[float, ...]
    near_90_corner_count: int


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


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-18:
        return _distance(point, start)
    ratio = max(
        0.0,
        min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared),
    )
    projection = (start[0] + ratio * dx, start[1] + ratio * dy)
    return _distance(point, projection)


def _segments_distance(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> float:
    if _segments_intersect(first_start, first_end, second_start, second_end):
        return 0.0
    return min(
        _point_segment_distance(first_start, second_start, second_end),
        _point_segment_distance(first_end, second_start, second_end),
        _point_segment_distance(second_start, first_start, first_end),
        _point_segment_distance(second_end, first_start, first_end),
    )


def _are_adjacent(first_index: int, second_index: int, count: int) -> bool:
    if first_index == second_index:
        return True
    if (first_index + 1) % count == second_index:
        return True
    if (second_index + 1) % count == first_index:
        return True
    return False


def _road_boundaries(
    geometry: CustomTrackGeometry,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    points = geometry.centerline
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        previous = points[(index - 1) % len(points)]
        following = points[(index + 1) % len(points)]
        incoming_length = _distance(previous, point)
        outgoing_length = _distance(point, following)
        incoming = (
            (point[0] - previous[0]) / incoming_length,
            (point[1] - previous[1]) / incoming_length,
        )
        outgoing = (
            (following[0] - point[0]) / outgoing_length,
            (following[1] - point[1]) / outgoing_length,
        )
        incoming_normal = (-incoming[1], incoming[0])
        outgoing_normal = (-outgoing[1], outgoing[0])
        normal_x = incoming_normal[0] + outgoing_normal[0]
        normal_y = incoming_normal[1] + outgoing_normal[1]
        normal_length = math.hypot(normal_x, normal_y)
        if normal_length <= 1e-9:
            normal_x, normal_y = outgoing_normal
            normal_length = 1.0
        normal = (normal_x / normal_length, normal_y / normal_length)
        denominator = abs(normal[0] * incoming_normal[0] + normal[1] * incoming_normal[1])
        miter_length = min(geometry.width / max(denominator, 1e-6), geometry.width * 4.0)
        offset = (normal[0] * miter_length, normal[1] * miter_length)
        left.append((point[0] + offset[0], point[1] + offset[1]))
        right.append((point[0] - offset[0], point[1] - offset[1]))
    return tuple(left), tuple(right)


def _road_edges_intersect(geometry: CustomTrackGeometry) -> bool:
    points = geometry.centerline
    count = len(points)
    boundaries = _road_boundaries(geometry)
    for boundary in boundaries:
        for first_index in range(count):
            first_start = boundary[first_index]
            first_end = boundary[(first_index + 1) % count]
            for second_index in range(first_index + 1, count):
                if _are_adjacent(first_index, second_index, count):
                    continue
                second_start = boundary[second_index]
                second_end = boundary[(second_index + 1) % count]
                if (
                    max(first_start[0], first_end[0]) + 1e-9
                    < min(second_start[0], second_end[0])
                    or max(second_start[0], second_end[0]) + 1e-9
                    < min(first_start[0], first_end[0])
                    or max(first_start[1], first_end[1]) + 1e-9
                    < min(second_start[1], second_end[1])
                    or max(second_start[1], second_end[1]) + 1e-9
                    < min(first_start[1], first_end[1])
                ):
                    continue
                if _segments_intersect(first_start, first_end, second_start, second_end):
                    return True

    segment_lengths = [
        _distance(points[index], points[(index + 1) % count])
        for index in range(count)
    ]
    total_length = sum(segment_lengths)
    midpoint_distances: list[float] = []
    cumulative = 0.0
    for length in segment_lengths:
        midpoint_distances.append(cumulative + length * 0.5)
        cumulative += length
    # Ignore nearby samples along the same smooth bend; the 8-unit floor is
    # twice the generator's maximum segment spacing.
    required_arc_separation = max(geometry.width * 4.0, 8.0)
    minimum_centerline_distance = geometry.width * 2.0
    for first_index in range(count):
        first_midpoint = midpoint_distances[first_index]
        first_start = points[first_index]
        first_end = points[(first_index + 1) % count]
        for second_index in range(first_index + 1, count):
            if _are_adjacent(first_index, second_index, count):
                continue
            separation = abs(midpoint_distances[second_index] - first_midpoint)
            separation = min(separation, total_length - separation)
            if separation <= required_arc_separation:
                continue
            second_start = points[second_index]
            second_end = points[(second_index + 1) % count]
            if _segments_distance(first_start, first_end, second_start, second_end) < minimum_centerline_distance:
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

        side_a = incoming_length
        side_b = outgoing_length
        side_c = _distance(previous, following)
        doubled_area = abs(_cross(previous, point, following))
        if doubled_area > 1e-9:
            radius = side_a * side_b * side_c / (2.0 * doubled_area)
            # CustomCarRacing offsets each edge by geometry.width, so this
            # field is the road half-width in world units.
            minimum_radius = geometry.width * 1.05
            if radius < minimum_radius:
                raise ValueError(
                    f"turn radius {radius:.6g} is below minimum {minimum_radius:.6g} at point {index}"
                )

    if _road_edges_intersect(geometry):
        raise ValueError("road boundaries intersect or overlap")


def custom_geometry_fingerprint(geometry: CustomTrackGeometry) -> str:
    payload = {
        "centerline": [list(point) for point in geometry.centerline],
        "width": geometry.width,
        "start_index": geometry.start_index,
        "direction": geometry.direction,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _corner_sequence(
    template: str,
    rng: _TrackRng,
) -> tuple[str, ...]:
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {sorted(TEMPLATES)}")
    if template == "extreme_technical":
        rng.random()
        count = 12 + 2 * rng.index(3)
    else:
        minimum, maximum = CORNER_COUNT_RANGES[template]
        count = minimum + rng.index(maximum - minimum + 1)

    if template == "oval":
        return ("left:wide",) * count

    if template == "s_curve":
        directions = ["left", "right", "left", "left", "right", "left"]
        while len(directions) < count:
            directions.append("right" if directions[-1] == "left" else "left")
        classes = ("wide", "medium", "tight")
        return tuple(
            f"{directions[index]}:{classes[rng.index(len(classes))]}"
            for index in range(count)
        )

    if template == "hairpin":
        sequence = [
            f"left:{('wide' if rng.random() < 0.4 else 'medium')}"
            for _ in range(count)
        ]
        hairpin_indices = (count // 3, (2 * count) // 3)
        right_hairpin = hairpin_indices[rng.index(2)]
        for index in hairpin_indices:
            direction = "right" if index == right_hairpin else "left"
            sequence[index] = f"{direction}:hairpin"
        return tuple(sequence)

    if template == "chicane":
        directions = ["left" if index % 2 == 0 else "right" for index in range(count)]
        if count % 2 == 0:
            directions[-1] = "left"
        return tuple(
            f"{directions[index]}:{'tight' if rng.random() < 0.65 else 'medium'}"
            for index in range(count)
        )

    if template == "extreme_technical":
        sequence = []
        hairpin_count = count // 4
        hairpin_indices = {
            1 + index * count // hairpin_count
            for index in range(hairpin_count)
        }
        for index in range(count):
            direction = "right" if index in hairpin_indices else "left"
            corner_class = "hairpin" if direction == "right" else "medium"
            sequence.append(f"{direction}:{corner_class}")
        return tuple(sequence)

    classes = ["wide", "medium", "tight"]
    classes.extend(classes[rng.index(len(classes))] for _ in range(count - len(classes)))
    for index in range(len(classes) - 1, 0, -1):
        swap_index = rng.index(index + 1)
        classes[index], classes[swap_index] = classes[swap_index], classes[index]
    directions = ["left" if index % 2 == 0 else "right" for index in range(count)]
    if count % 2 == 0:
        directions[-1] = "left"
    return tuple(f"{directions[index]}:{corner_class}" for index, corner_class in enumerate(classes))


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


def _profile_angle_gaps(
    gaps: tuple[float, ...],
    sequence: tuple[str, ...],
    *,
    compact_s_turns: bool = False,
) -> tuple[float, ...]:
    constrained_corners = [
        index
        for index, token in enumerate(sequence)
        if token.startswith("right:") or token.endswith(":hairpin")
    ]
    if not constrained_corners:
        return gaps

    count = len(sequence)
    constrained_gaps: set[int] = set()
    for index in constrained_corners:
        adjacent = {(index - 1) % count, index}
        if constrained_gaps.intersection(adjacent):
            raise ValueError("corner profile contains adjacent concave turns")
        constrained_gaps.update(adjacent)

    remaining_indices = [index for index in range(count) if index not in constrained_gaps]
    if not remaining_indices:
        raise ValueError("corner profile has no unconstrained straight intervals")
    minimum_pair_total = max(
        40.0,
        (360.0 - 100.0 * len(remaining_indices)) / len(constrained_corners),
    )
    maximum_pair_total = min(
        200.0,
        (360.0 - 20.0 * len(remaining_indices)) / len(constrained_corners),
    )
    if maximum_pair_total < minimum_pair_total:
        raise ValueError("corner profile cannot fit within the angle-gap bounds")
    preferred_pair_total = 60.0 if compact_s_turns else 80.0
    pair_total_degrees = max(minimum_pair_total, min(preferred_pair_total, maximum_pair_total))
    remaining_total_degrees = 360.0 - pair_total_degrees * len(constrained_corners)
    remaining_base = remaining_total_degrees / len(remaining_indices)
    if not 20.0 <= remaining_base <= 100.0:
        raise ValueError("corner profile cannot fit within the angle-gap bounds")

    raw_degrees = tuple(math.degrees(gaps[index]) for index in remaining_indices)
    raw_mean = sum(raw_degrees) / len(raw_degrees)
    factor = 1.0
    for gap in raw_degrees:
        deviation = gap - raw_mean
        if deviation > 0.0:
            factor = min(factor, (100.0 - remaining_base) / deviation)
        elif deviation < 0.0:
            factor = min(factor, (remaining_base - 20.0) / -deviation)
    factor = max(0.0, min(1.0, factor * 0.999))

    adjusted = [math.radians(pair_total_degrees * 0.5)] * count
    for index, gap in zip(remaining_indices, raw_degrees):
        adjusted[index] = math.radians(remaining_base + factor * (gap - raw_mean))
    return tuple(adjusted)


def _round_metric(value: float, places: int) -> float:
    scale = 10.0 ** places
    magnitude = math.floor(abs(value) * scale + 0.5) / scale
    return math.copysign(magnitude, value)


def _measure_corner_profiles(
    points: tuple[tuple[float, float], ...],
    ranges: tuple[tuple[int, int], ...],
    width: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    turns: list[float] = []
    radii: list[float] = []
    count = len(points)
    for start, end in ranges:
        total_turn = 0.0
        local_radii: list[float] = []
        for index in range(start, end + 1):
            previous = points[(index - 1) % count]
            point = points[index % count]
            following = points[(index + 1) % count]
            incoming = (point[0] - previous[0], point[1] - previous[1])
            outgoing = (following[0] - point[0], following[1] - point[1])
            cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
            turn = math.atan2(cross, dot)
            total_turn += turn
            side_a = math.hypot(*incoming)
            side_b = math.hypot(*outgoing)
            side_c = _distance(previous, following)
            doubled_area = abs(cross)
            if (
                start < index < end
                and doubled_area > 1e-9
                and abs(turn) > math.radians(0.1)
            ):
                local_radii.append(side_a * side_b * side_c / (2.0 * doubled_area))
        turns.append(_round_metric(math.degrees(total_turn), 2))
        radii.append(_round_metric(min(local_radii) / width if local_radii else 1e6, 3))
    return tuple(turns), tuple(radii)


def _build_extreme_route(
    design_seed: int, width: float, attempt_index: int
) -> _ExtremeRoute:
    count = len(_corner_sequence("extreme_technical", _TrackRng(design_seed)))
    if count not in {12, 14, 16}:
        raise ValueError(f"unsupported extreme corner count {count}")
    rng = _TrackRng((design_seed + attempt_index * 0x9E3779B9) & 0xFFFFFFFF)
    radius_x = 150.0 + (width - 8.0) + rng.uniform(-6.0, 6.0)
    radius_y = 93.75 + 0.625 * (width - 8.0) + rng.uniform(-4.0, 4.0)
    corners = [
        (-radius_x, -radius_y),
        (radius_x, -radius_y),
        (radius_x, radius_y),
        (-radius_x, radius_y),
    ]
    if rng.index(2):
        corners = [(x, -y) for x, y in reversed(corners)]
    shift = rng.index(4)
    corners = corners[shift:] + corners[:shift]
    vectors = []
    for index, start in enumerate(corners):
        end = corners[(index + 1) % 4]
        length = _distance(start, end)
        vectors.append(((end[0] - start[0]) / length, (end[1] - start[1]) / length))
    # Inward is the left normal because this is a counter-clockwise loop.
    roundness = max(0.0, min(1.0, (width - 8.0) / 92.0))
    tangent = 1.7 * width * (1.0 + 0.4 * roundness) * math.sqrt(2.0)
    if count == 12:
        selected_sides = (0, 2) if rng.index(2) == 0 else (1, 3)
    elif count == 14:
        dogleg_corner = rng.index(4)
        selected_sides = tuple(
            side for side in range(4)
            if side not in {dogleg_corner, (dogleg_corner - 1) % 4}
        )
    else:
        side_lengths = tuple(_distance(corners[index], corners[(index + 1) % 4]) for index in range(4))
        longest = max(side_lengths)
        long_sides = tuple(index for index, length in enumerate(side_lengths) if length == longest)
        short_sides = tuple(index for index in range(4) if index not in long_sides)
        selected_sides = (*long_sides, short_sides[rng.index(len(short_sides))])

    feature_points: list[tuple[tuple[float, float], tuple[float, float]]] = []
    side_events: dict[int, tuple[float, float, float]] = {}
    for side in selected_sides:
        start, end = corners[side], corners[(side + 1) % 4]
        length = _distance(start, end)
        u = vectors[side]
        inward = (-u[1], u[0])
        s_gap = rng.uniform(0.6 * width, 2.2 * width)
        p_gap = rng.uniform(0.6 * width, 1.4 * width)
        depth = 2.0 * tangent + s_gap
        leg = 2.0 * tangent + p_gap
        usable = length - 12.0 * width - leg
        if usable < 0.0:
            raise ValueError("extreme notch does not fit its side")
        if count == 14:
            near_start = side == (dogleg_corner + 1) % 4
            end_jitter = rng.uniform(0.0, min(2.0 * width, usable))
            position = 6.0 * width + end_jitter if near_start else length - 6.0 * width - leg - end_jitter
        elif count == 16 and side in selected_sides[:2]:
            short_side = selected_sides[2]
            near_start = (side + 1) % 4 == short_side
            end_jitter = rng.uniform(0.0, min(2.0 * width, usable))
            position = (
                6.0 * width + end_jitter
                if near_start
                else length - 6.0 * width - leg - end_jitter
            )
        else:
            position = 6.0 * width + rng.random() * usable
        side_events[side] = (position, depth, leg)

    vertices: list[tuple[float, float]] = []
    for side, start in enumerate(corners):
        if not vertices:
            vertices.append(start)
        if side in side_events:
            position, depth, leg = side_events[side]
            u = vectors[side]
            inward = (-u[1], u[0])
            first = (start[0] + u[0] * position, start[1] + u[1] * position)
            second = (first[0] + inward[0] * depth, first[1] + inward[1] * depth)
            third = (second[0] + u[0] * leg, second[1] + u[1] * leg)
            fourth = (third[0] - inward[0] * depth, third[1] - inward[1] * depth)
            vertices.extend((first, second, third, fourth))
            feature_points.append((first, second))
            feature_points.append((third, fourth))
        vertices.append(corners[(side + 1) % 4])

    # Do not keep the repeated closing corner while replacing an outer turn.
    if vertices[-1] == vertices[0]:
        vertices.pop()
    if count == 14:
        u, v = vectors[dogleg_corner - 1], vectors[dogleg_corner]
        a = 2.0 * tangent + rng.uniform(0.6 * width, 2.2 * width)
        b = 2.0 * tangent + rng.uniform(0.6 * width, 2.2 * width)
        corner = corners[dogleg_corner]
        entry = (corner[0] - a * u[0], corner[1] - a * u[1])
        elbow = (entry[0] + b * v[0], entry[1] + b * v[1])
        exit_point = (corner[0] + b * v[0], corner[1] + b * v[1])
        index = vertices.index(corner)
        vertices[index:index + 1] = [entry, elbow, exit_point]
        feature_points.append((entry, elbow))

    indices = {point: index for index, point in enumerate(vertices)}
    pairs = tuple((indices[first], indices[second]) for first, second in feature_points)
    turns = []
    for index, point in enumerate(vertices):
        previous = vertices[index - 1]
        following = vertices[(index + 1) % len(vertices)]
        incoming = (point[0] - previous[0], point[1] - previous[1])
        outgoing = (following[0] - point[0], following[1] - point[1])
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        turns.append("left:tight" if cross > 0 else "right:tight")
    return _ExtremeRoute(tuple(vertices), tuple(turns), pairs)


def _round_extreme_route(route: _ExtremeRoute, width: float) -> _ExtremeGeometryResult:
    anchors = route.vertices
    count = len(anchors)
    incoming_points = []
    outgoing_points = []
    tangents = []
    roundness = max(0.0, min(1.0, (width - 8.0) / 92.0))
    for index, anchor in enumerate(anchors):
        previous, following = anchors[index - 1], anchors[(index + 1) % count]
        incoming_length, outgoing_length = _distance(anchor, previous), _distance(anchor, following)
        u = ((previous[0] - anchor[0]) / incoming_length, (previous[1] - anchor[1]) / incoming_length)
        v = ((following[0] - anchor[0]) / outgoing_length, (following[1] - anchor[1]) / outgoing_length)
        cosine = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
        half = (math.pi - math.acos(cosine)) / 2.0
        radius = 1.7 * width * (1.0 + 0.4 * roundness)
        trim = radius * math.sin(half) / max(math.cos(half) ** 2, 1e-9)
        trim = min(trim, 0.45 * incoming_length, 0.45 * outgoing_length)
        incoming_points.append(_interpolate(anchor, previous, trim / incoming_length))
        outgoing_points.append(_interpolate(anchor, following, trim / outgoing_length))
        tangents.append(trim)

    points = [incoming_points[0]]
    ranges = []
    for index, anchor in enumerate(anchors):
        start = len(points) - 1
        control_length = max(_distance(anchor, incoming_points[index]), _distance(anchor, outgoing_points[index]))
        steps = max(4, math.ceil(2.0 * control_length / 4.0))
        for step in range(1, steps + 1):
            points.append(_quadratic_bezier(incoming_points[index], anchor, outgoing_points[index], step / steps))
        ranges.append((start, len(points) - 1))
        next_index = (index + 1) % count
        edge_length = _distance(outgoing_points[index], incoming_points[next_index])
        steps = max(1, math.ceil(edge_length / 4.0))
        last = steps if next_index else steps - 1
        for step in range(1, last + 1):
            points.append(_interpolate(outgoing_points[index], incoming_points[next_index], step / steps))
    quantized = tuple((_round_coordinate(x), _round_coordinate(y)) for x, y in points)
    if len(quantized) > MAX_CENTERLINE_POINTS:
        raise ValueError(f"generated centerline exceeds {MAX_CENTERLINE_POINTS} points")
    geometry = CustomTrackGeometry(centerline=quantized, width=width)
    measured_turns, measured_radii = _measure_corner_profiles(quantized, tuple(ranges), width)
    connectors = tuple(
        _distance(anchors[first], anchors[second]) - tangents[first] - tangents[second]
        for first, second in route.s_section_pairs
    )
    near_90 = sum(75.0 <= abs(turn) <= 105.0 for turn in measured_turns)
    return _ExtremeGeometryResult(
        geometry,
        route.corner_sequence,
        measured_turns,
        measured_radii,
        route.s_section_pairs,
        connectors,
        near_90,
    )


def _validate_extreme_profile(candidate: _ExtremeGeometryResult) -> None:
    count = len(candidate.corner_sequence)
    if count not in {12, 14, 16}:
        raise ValueError(f"extreme corner count {count} is unsupported")
    if len(candidate.s_section_pairs) < 3:
        raise ValueError("extreme route needs at least three S sections")
    if not (len(candidate.corner_turn_degrees) == len(candidate.corner_radius_widths) == count):
        raise ValueError("extreme corner metadata lengths do not match the route")
    if len(candidate.s_section_connector_lengths) != len(candidate.s_section_pairs):
        raise ValueError("extreme S-section metadata lengths do not match the route")
    if any(not 0.6 * candidate.geometry.width - 1e-5 <= length <= 2.2 * candidate.geometry.width + 1e-5
           for length in candidate.s_section_connector_lengths):
        raise ValueError("extreme S-section connector must be 0.6–2.2 track widths")
    for token, turn, radius in zip(candidate.corner_sequence, candidate.corner_turn_degrees, candidate.corner_radius_widths):
        direction, corner_class = token.split(":", 1)
        if (1.0 if direction == "left" else -1.0) * turn < 20.0:
            raise ValueError(f"measured corner turn does not match {token}")
        low, high = CORNER_RADIUS_WIDTH_RANGES[corner_class]
        if not low <= radius <= high:
            raise ValueError(f"measured corner radius does not match {token}")
    measured = sum(75.0 <= abs(turn) <= 105.0 for turn in candidate.corner_turn_degrees)
    if measured < 3 or measured != candidate.near_90_corner_count:
        raise ValueError("extreme route needs at least three measured near-90-degree corners")


def _generate_extreme_geometry(design_seed: int, width: float) -> _ExtremeGeometryResult:
    design_seed = _require_int("design_seed", design_seed, 0, MAX_SEED)
    width = _require_float("width", width, 0.5, MAX_GENERATED_TRACK_WIDTH)
    last_error = None
    for attempt_index in range(64):
        try:
            candidate = _round_extreme_route(_build_extreme_route(design_seed, width, attempt_index), width)
            _validate_extreme_profile(candidate)
            validate_custom_geometry(candidate.geometry)
        except ValueError as error:
            last_error = error
            continue
        return candidate
    raise ValueError(
        f"could not generate a valid extreme_technical track for seed {design_seed} "
        f"and width {width}: {last_error}"
    )


def _build_geometry_candidate(
    template: str,
    design_seed: int,
    width: float,
) -> tuple[CustomTrackGeometry, tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    rng = _TrackRng(design_seed)
    sequence = _corner_sequence(template, rng)
    count = len(sequence)
    gaps = _profile_angle_gaps(
        _angle_gaps(rng, count),
        sequence,
        compact_s_turns=template == "extreme_technical",
    )

    radius_x = 150.0 + (width - 8.0) + rng.uniform(-6.0, 6.0)
    radius_y = 93.75 + (width - 8.0) * 0.625 + rng.uniform(-4.0, 4.0)
    roundness = max(0.0, min(1.0, (width - 8.0) / 92.0))
    target_axis_ratio = 1.0 if template != "oval" else 1.6 - 0.45 * roundness
    radius_y = max(radius_y, radius_x / target_axis_ratio)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    template_amplitude = {
        "oval": 0.015,
        "s_curve": 0.02,
        "hairpin": 0.02,
        "chicane": 0.03,
        "technical": 0.04,
        "extreme_technical": 0.055,
    }[template]
    class_adjustment = {"wide": 0.01, "medium": 0.0, "tight": -0.01, "hairpin": 0.0}

    anchors: list[tuple[float, float]] = []
    directions = tuple(token.split(":", 1)[0] for token in sequence)
    wide_track_ratio = max(0.0, min(1.0, (width - 8.0) / 92.0))
    regular_left_radius = 1.15 - 0.1 * wide_track_ratio
    regular_right_radius = 0.5 + 0.25 * wide_track_ratio
    nested_left_radius = 0.9 + 0.15 * wide_track_ratio
    angle = rng.uniform(0.0, 2.0 * math.pi)
    for index, token in enumerate(sequence):
        direction, corner_class = token.split(":", 1)
        direction_sign = 1.0 if direction == "left" else -1.0
        harmonic = 1 + (index % 3)
        radial_modulation = template_amplitude * math.sin(harmonic * angle + phase)
        radial_jitter = rng.uniform(-0.025, 0.025)
        if corner_class == "hairpin":
            direction_radius = (
                1.8 - 0.5 * wide_track_ratio
                if direction == "left"
                else 0.04 + 0.7 * wide_track_ratio
            )
            radial_scale = direction_radius + radial_jitter
        else:
            if (
                direction == "left"
                and directions[index - 1] == "right"
                and directions[(index + 1) % count] == "right"
            ):
                direction_radius = nested_left_radius
            else:
                direction_radius = regular_left_radius if direction == "left" else regular_right_radius
            radial_scale = (
                direction_radius
                + direction_sign * class_adjustment[corner_class]
                + radial_modulation
                + radial_jitter
            )
        anchors.append(
            (
                radius_x * radial_scale * math.cos(angle),
                radius_y * radial_scale * math.sin(angle),
            )
        )
        angle += gaps[index]

    incoming_points: list[tuple[float, float]] = []
    outgoing_points: list[tuple[float, float]] = []
    for index, anchor in enumerate(anchors):
        previous = anchors[(index - 1) % count]
        following = anchors[(index + 1) % count]
        incoming_length = _distance(anchor, previous)
        outgoing_length = _distance(anchor, following)
        corner_class = sequence[index].split(":", 1)[1]
        previous_ray = (
            (previous[0] - anchor[0]) / incoming_length,
            (previous[1] - anchor[1]) / incoming_length,
        )
        following_ray = (
            (following[0] - anchor[0]) / outgoing_length,
            (following[1] - anchor[1]) / outgoing_length,
        )
        interior_cosine = max(
            -1.0,
            min(1.0, previous_ray[0] * following_ray[0] + previous_ray[1] * following_ray[1]),
        )
        deflection = math.pi - math.acos(interior_cosine)
        half_deflection = deflection * 0.5
        radius_target = {
            "wide": 2.6,
            "medium": 2.2,
            "tight": 1.7,
            "hairpin": 1.6,
        }[corner_class] * width * (1.0 + 0.4 * max(0.0, min(1.0, (width - 8.0) / 92.0)))
        quadratic_radius_factor = math.sin(half_deflection) / max(
            math.cos(half_deflection) ** 2,
            1e-9,
        )
        trim = min(
            radius_target * quadratic_radius_factor,
            0.45 * min(incoming_length, outgoing_length),
        )
        incoming_points.append(_interpolate(anchor, previous, trim / incoming_length))
        outgoing_points.append(_interpolate(anchor, following, trim / outgoing_length))

    points: list[tuple[float, float]] = [incoming_points[0]]
    corner_ranges: list[tuple[int, int]] = []
    for index, anchor in enumerate(anchors):
        corner_start = len(points) - 1
        outgoing = outgoing_points[index]
        control_length = max(_distance(anchor, incoming_points[index]), _distance(anchor, outgoing))
        curve_steps = max(4, math.ceil(2.0 * control_length / 4.0))
        for step in range(1, curve_steps + 1):
            points.append(
                _quadratic_bezier(
                    incoming_points[index],
                    anchor,
                    outgoing,
                    step / curve_steps,
                )
            )
        corner_ranges.append((corner_start, len(points) - 1))

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
    measured_turns, measured_radii = _measure_corner_profiles(
        quantized,
        tuple(corner_ranges),
        width,
    )
    return geometry, sequence, measured_turns, measured_radii


def _generate_geometry(
    template: str,
    design_seed: int,
    width: float,
) -> tuple[CustomTrackGeometry, tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {sorted(TEMPLATES)}")
    design_seed = _require_int("design_seed", design_seed, 0, MAX_SEED)
    width = _require_float("width", width, 0.5, MAX_GENERATED_TRACK_WIDTH)
    last_error: ValueError | None = None
    for attempt in range(64):
        attempt_seed = (design_seed + attempt * 0x9E3779B9) & 0xFFFFFFFF
        try:
            geometry, sequence, measured_turns, measured_radii = _build_geometry_candidate(
                template,
                attempt_seed,
                width,
            )
            for token, turn_degrees, radius_widths in zip(
                sequence,
                measured_turns,
                measured_radii,
            ):
                expected_sign = 1.0 if token.startswith("left:") else -1.0
                minimum_turn = 90.0 if token.endswith(":hairpin") else 20.0
                if expected_sign * turn_degrees < minimum_turn:
                    raise ValueError(f"measured corner turn does not match {token}")
                corner_class = token.split(":", 1)[1]
                minimum_radius, maximum_radius = CORNER_RADIUS_WIDTH_RANGES[corner_class]
                if not minimum_radius <= radius_widths <= maximum_radius:
                    raise ValueError(f"measured corner radius does not match {token}")
            validate_custom_geometry(geometry)
        except ValueError as error:
            last_error = error
            continue
        return geometry, sequence, measured_turns, measured_radii
    raise ValueError(f"could not generate a valid {template} track: {last_error}")


def generate_custom_map(
    map_id: str,
    design_seed: int,
    template: str,
    width: float = 8.0,
    max_steps: int = 2000,
    frame_skip: int = 4,
) -> CustomMapSpec:
    extreme_result = None
    if template == "extreme_technical":
        extreme_result = _generate_extreme_geometry(design_seed, width)
        geometry = extreme_result.geometry
        corner_sequence = extreme_result.corner_sequence
        corner_turns = extreme_result.corner_turn_degrees
        corner_radii = extreme_result.corner_radius_widths
    else:
        geometry, corner_sequence, corner_turns, corner_radii = _generate_geometry(
            template,
            design_seed,
            width,
        )
    generator_metadata = [
        ("design_seed", design_seed),
        ("generator_version", GENERATOR_VERSION),
        ("corner_count", len(corner_sequence)),
        ("corner_sequence", corner_sequence),
        ("corner_turn_degrees", corner_turns),
        ("corner_radius_widths", corner_radii),
        ("template", template),
    ]
    if extreme_result is not None:
        generator_metadata.extend((
            ("s_section_count", len(extreme_result.s_section_pairs)),
            ("near_90_corner_count", extreme_result.near_90_corner_count),
        ))
    return CustomMapSpec(
        map_id=map_id,
        geometry=geometry,
        obstacles=(),
        max_steps=max_steps,
        frame_skip=frame_skip,
        generator=tuple(generator_metadata),
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
    "MAX_GENERATED_TRACK_WIDTH",
    "TEMPLATES",
    "custom_geometry_fingerprint",
    "generate_custom_map",
    "generate_custom_maps",
    "validate_custom_geometry",
]
