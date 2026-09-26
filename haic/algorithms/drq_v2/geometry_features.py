"""Training-only shape descriptors for unmodified seeded CarRacing roads.

Nothing in this module modifies or imports the official environment. Blind roads
must never be passed to these functions; reserved blind IDs are exclusions only.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np


def _centerline(track: Any) -> np.ndarray:
    rows = np.asarray(track, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 4 or len(rows) < 20:
        raise ValueError("official track must contain >=20 (alpha,beta,x,y) rows")
    if not np.isfinite(rows).all():
        raise ValueError("track centerline contains non-finite coordinates")
    points = rows[:, 2:4]
    if np.linalg.norm(points[0] - points[-1]) < 1e-7:
        points = points[:-1]
    if len(points) < 20:
        raise ValueError("track has fewer than 20 distinct centerline points")
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths < 0.25) or np.any(lengths > 40.0):
        raise ValueError("track has a collapsed or discontinuous centerline segment")
    return points


def validate_track_structure(track: Any, *, half_width: float = 40.0 / 6.0
                             ) -> dict[str, Any]:
    """Detect broken seams and nonadjacent centerline crossings without driving.

    The official generator deliberately removes one tile at its seam, leaving a
    roughly 7m final-to-first edge rather than its usual 3.5m edge. A static
    check cannot prove that a policy can reach the 95%-progress finish trigger.
    """
    if not math.isfinite(half_width) or half_width <= 0:
        raise ValueError("road half-width must be positive and finite")
    points = _centerline(track)
    n = len(points)
    edges = np.roll(points, -1, axis=0) - points
    edge_lengths = np.linalg.norm(edges, axis=1)
    heading = np.arctan2(edges[:, 1], edges[:, 0])
    seam_turn = float(np.arctan2(np.sin(heading[0] - heading[-1]),
                                      np.cos(heading[0] - heading[-1])))
    seam_length = float(edge_lengths[-1])
    usual_step = float(np.median(edge_lengths[:-1]))

    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j >= n - 2))
    i, j = i[keep], j[keep]
    a, b = points[i], edges[i]
    c, d = points[j], edges[j]
    difference = c - a
    cross = lambda u, v: u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    denominator = cross(b, d)
    nonparallel = np.abs(denominator) > 1e-8
    t = np.zeros_like(denominator)
    u = np.zeros_like(denominator)
    t[nonparallel] = cross(difference[nonparallel], d[nonparallel]) / denominator[nonparallel]
    u[nonparallel] = cross(difference[nonparallel], b[nonparallel]) / denominator[nonparallel]
    crossing = nonparallel & (t >= -1e-6) & (t <= 1 + 1e-6) & (u >= -1e-6) & (u <= 1 + 1e-6)
    crossing_count = int(np.count_nonzero(crossing))

    # This is a conservative tile-overlap warning, not a malformed-road veto:
    # valid tight turns can bring nonconsecutive official road tiles close.
    away = (j - i > 7) & (j - i < n - 7)
    nearby = away & (np.linalg.norm(points[i] - points[j], axis=1) < 2 * half_width)
    return {
        "centerline_self_intersections": crossing_count,
        "seam_edge_m": round(seam_length, 5),
        "median_segment_m": round(usual_step, 5),
        "seam_turn_rad": round(seam_turn, 6),
        "nonadjacent_road_overlap_warning_pairs": int(np.count_nonzero(nearby)),
        "road_half_width_m": round(half_width, 4),
        "centerline_connected": bool(
            seam_length <= 2.5 * usual_step
            and abs(seam_turn) <= 0.55
            and crossing_count == 0
        ),
        "finish_static_only": True,
    }


def _road_signature(points: np.ndarray, widths: np.ndarray, bins: int) -> dict[str, Any]:
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    turn = np.arctan2(np.sin(angles - np.roll(angles, 1)),
                      np.cos(angles - np.roll(angles, 1)))
    perimeter = float(lengths.sum())
    arclength = np.cumsum(lengths) - lengths
    fractional_bin = arclength / perimeter * bins
    lower = np.floor(fractional_bin).astype(int) % bins
    fraction = fractional_bin - np.floor(fractional_bin)
    sampled_turn = np.zeros(bins, dtype=np.float64)
    sampled_width = np.zeros(bins, dtype=np.float64)
    sampled_length = np.zeros(bins, dtype=np.float64)
    np.add.at(sampled_turn, lower, turn * (1.0 - fraction))
    np.add.at(sampled_turn, (lower + 1) % bins, turn * fraction)
    np.add.at(sampled_width, lower, widths * (1.0 - fraction))
    np.add.at(sampled_width, (lower + 1) % bins, widths * fraction)
    np.add.at(sampled_length, lower, (1.0 - fraction))
    np.add.at(sampled_length, (lower + 1) % bins, fraction)
    sampled_width = sampled_width / np.maximum(sampled_length, 1e-6)
    # Smoothing over one neighboring bin reduces tile-to-bin aliasing when the
    # same closed road is described with a different starting tile.
    sampled_turn = (
        np.roll(sampled_turn, -1) + 2.0 * sampled_turn + np.roll(sampled_turn, 1)
    ) / 4.0
    # Turning angle per sample, not absolute compass heading, is translation and
    # rotation invariant. Keep the sign so left/right symmetry stays auditable.
    return {
        "turn_sequence": [round(float(value), 5) for value in sampled_turn],
        "relative_width_sequence": [
            round(float(value / np.median(widths)), 4) for value in sampled_width
        ],
        "perimeter_m": round(perimeter, 3),
        "width_median_m": round(float(np.median(widths)), 3),
    }


def _bend_events(lengths: np.ndarray, curvature: np.ndarray) -> list[dict[str, Any]]:
    n = len(lengths)
    cumulative = np.cumsum(lengths) - lengths
    perimeter = float(lengths.sum())
    signs = np.where(curvature >= 0.025, 1, np.where(curvature <= -0.025, -1, 0))
    events: list[dict[str, Any]] = []
    start: int | None = None
    direction = 0
    for index, current in enumerate(np.r_[signs, 0]):
        if index < n and current != 0 and start is None:
            start, direction = index, int(current)
        if start is not None and (current != direction or index == n):
            stop = index
            # A one-tile curvature spike is not a sustained turn requiring
            # anticipation; keep only bends with a significant heading change.
            angle = float(np.sum(curvature[start:stop] * lengths[start:stop]))
            if abs(angle) >= 0.3:
                previous_straight = 0.0
                for offset in range(1, n):
                    previous = (start - offset) % n
                    if abs(curvature[previous]) >= 0.010:
                        break
                    previous_straight += float(lengths[previous])
                events.append({
                    "start_fraction": round(float(cumulative[start] / perimeter), 5),
                    "end_fraction": round(float(cumulative[min(stop, n - 1)] / perimeter), 5),
                    "start_m": round(float(cumulative[start]), 3),
                    "end_m": round(float(cumulative[min(stop, n - 1)]), 3),
                    "direction": "left" if direction > 0 else "right",
                    "turn_angle_rad": round(angle, 5),
                    "length_m": round(float(lengths[start:stop].sum()), 3),
                    "preceding_straight_m": round(previous_straight, 3),
                    "entry_straight_since_spawn_m": round(
                        min(previous_straight, float(cumulative[start])), 3
                    ),
                })
            start = index if index < n and current != 0 else None
            direction = int(current) if start is not None else 0
    return events


def describe_track(track: Any, *, half_width: float = 40.0 / 6.0,
                   bins: int = 64) -> dict[str, Any]:
    """Measure existing road curvature and straight-to-turn transitions.

    This is a descriptor, not a parameterization: the official generator exposes
    only the geometry seed. ``half_width`` is the fixed upstream TRACK_WIDTH.
    """
    if not isinstance(bins, int) or bins < 32 or bins > 256:
        raise ValueError("signature bins must be an integer in [32,256]")
    if not math.isfinite(half_width) or half_width <= 0:
        raise ValueError("road half-width must be positive and finite")
    points = _centerline(track)
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    headings = np.arctan2(edges[:, 1], edges[:, 0])
    turn = np.arctan2(np.sin(headings - np.roll(headings, 1)),
                      np.cos(headings - np.roll(headings, 1)))
    local_scale = (lengths + np.roll(lengths, 1)) / 2.0
    curvature = turn / local_scale
    abs_curvature = np.abs(curvature)
    straight = abs_curvature < 0.010
    sharp = abs_curvature > 0.045
    # Scan one full cycle backwards from each sharp entry without creating fake
    # straight length at the circular start/finish seam.
    previous_straight = np.zeros(len(points), dtype=np.float64)
    for index in range(len(points)):
        if not sharp[index]:
            continue
        distance = 0.0
        for offset in range(1, len(points)):
            previous = (index - offset) % len(points)
            if not straight[previous]:
                break
            distance += float(lengths[previous])
        previous_straight[index] = distance

    sharp_entries = np.flatnonzero(sharp & ~np.roll(sharp, 1))
    bend_events = _bend_events(lengths, curvature)
    sign = np.sign(curvature)
    left_count = int(np.count_nonzero(sharp & (sign > 0)))
    right_count = int(np.count_nonzero(sharp & (sign < 0)))
    rapid_reversals = 0
    consecutive_turns = 0
    for before, after in zip(bend_events, bend_events[1:]):
        gap = after["start_m"] - before["end_m"]
        if 0 <= gap <= 65:
            if before["direction"] != after["direction"]:
                rapid_reversals += 1
            else:
                consecutive_turns += 1

    early = [event for event in bend_events if event["start_fraction"] < 0.20]
    mid = [event for event in bend_events if 0.20 <= event["start_fraction"] < 0.80]
    late = [event for event in bend_events if event["start_fraction"] >= 0.80]
    strongest_early = max(early, key=lambda event: abs(event["turn_angle_rad"]), default=None)
    strongest_mid = max(mid, key=lambda event: abs(event["turn_angle_rad"]), default=None)
    strongest_late = max(late, key=lambda event: abs(event["turn_angle_rad"]), default=None)
    mid_reversals = 0
    mid_same_sign = 0
    for before, after in zip(mid, mid[1:]):
        if 0 <= after["start_m"] - before["end_m"] <= 65:
            if before["direction"] != after["direction"]:
                mid_reversals += 1
            else:
                mid_same_sign += 1

    signature = _road_signature(points, np.full(len(points), 2 * half_width), bins)
    canon = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature_pairs = list(zip(signature["turn_sequence"], signature["relative_width_sequence"]))
    canonical_pairs = min(
        signature_pairs[shift:] + signature_pairs[:shift]
        for shift in range(len(signature_pairs))
    )
    canonical = json.dumps({
        "signed_turn_and_relative_width": canonical_pairs,
        "perimeter_m": signature["perimeter_m"],
        "width_median_m": signature["width_median_m"],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "point_count": int(len(points)),
        "perimeter_m": signature["perimeter_m"],
        "fixed_half_width_m": round(half_width, 4),
        "abs_curvature_p50_per_m": round(float(np.quantile(abs_curvature, 0.50)), 6),
        "abs_curvature_p90_per_m": round(float(np.quantile(abs_curvature, 0.90)), 6),
        "abs_curvature_p99_per_m": round(float(np.quantile(abs_curvature, 0.99)), 6),
        "sharp_left_points": left_count,
        "sharp_right_points": right_count,
        "sharp_entry_count": int(len(sharp_entries)),
        "longest_straight_before_sharp_m": round(float(previous_straight.max()), 3),
        "rapid_reversals_65m": rapid_reversals,
        "consecutive_same_sign_65m": consecutive_turns,
        "early_bend_count": len(early),
        "mid_bend_count": len(mid),
        "late_bend_count": len(late),
        "early_strongest_bend": strongest_early,
        "mid_strongest_bend": strongest_mid,
        "late_strongest_bend": strongest_late,
        "mid_reversals_65m": mid_reversals,
        "mid_consecutive_same_sign_65m": mid_same_sign,
        "bend_events": bend_events,
        "signature": signature,
        "signature_sha256": hashlib.sha256(canon).hexdigest(),
        "cyclic_signature_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def signature_distance(first: dict[str, Any], second: dict[str, Any],
                       *, allow_mirror: bool = False) -> float:
    """Minimum cyclic signed-turn/width distance; a smaller value is more alike."""
    first_turn = np.asarray(first["turn_sequence"], dtype=np.float64)
    second_turn = np.asarray(second["turn_sequence"], dtype=np.float64)
    first_width = np.asarray(first["relative_width_sequence"], dtype=np.float64)
    second_width = np.asarray(second["relative_width_sequence"], dtype=np.float64)
    if (first_turn.shape != second_turn.shape or first_turn.ndim != 1
            or first_width.shape != second_width.shape or first_width.shape != first_turn.shape
            or not np.isfinite(first_turn).all() or not np.isfinite(second_turn).all()
            or not np.isfinite(first_width).all() or not np.isfinite(second_width).all()):
        raise ValueError("signature arrays must have matching finite one-dimensional shapes")
    length_penalty = abs(math.log(max(float(first["perimeter_m"]), 1e-9)
                                  / max(float(second["perimeter_m"]), 1e-9)))
    width_penalty = abs(math.log(max(float(first["width_median_m"]), 1e-9)
                                 / max(float(second["width_median_m"]), 1e-9)))
    result = float("inf")
    index = (np.arange(len(first_turn))[None, :] - np.arange(len(first_turn))[:, None]) % len(first_turn)
    fractions = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=np.float64)[:, None, None]
    rotated_width = second_width[index]
    aligned_width = ((1.0 - fractions) * rotated_width[None, :, :]
                     + fractions * np.roll(rotated_width, 1, axis=1)[None, :, :])
    width_distance = np.sqrt(np.mean((first_width[None, None, :] - aligned_width) ** 2, axis=2))
    for orientation in (-1.0, 1.0) if allow_mirror else (1.0,):
        rotated = (second_turn * orientation)[index]
        aligned = ((1.0 - fractions) * rotated[None, :, :]
                   + fractions * np.roll(rotated, 1, axis=1)[None, :, :])
        curvature_distance = np.sqrt(np.mean((first_turn[None, None, :] - aligned) ** 2, axis=2))
        result = min(result, float(np.min(
            curvature_distance + 0.1 * width_distance + 0.03 * (length_penalty + width_penalty)
        )))
    return result
