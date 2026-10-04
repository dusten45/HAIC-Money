"""Deterministic perception features derived only from the 84x84 pixel view.

These values describe HUD and road objects. They never select or modify an
action; the learned policy remains responsible for driving decisions.
"""

from __future__ import annotations

import numpy as np


ROAD_LOW = 0.24
ROAD_HIGH = 0.52
OBSTACLE_LOW = 0.54
SPEED_ROI = (77, 83, 10, 13)
SPEED_BASELINE = 0.27
SPEED_PER_UNIT = 0.085
FEATURE_NAMES = (
    "speed_fraction",
    "near_road_center_offset",
    "far_road_center_offset",
    "road_curve_magnitude",
    "obstacle_present",
    "obstacle_lateral_offset",
    "obstacle_urgency",
)
VISUAL_FEATURE_SIZE = len(FEATURE_NAMES)
TEMPORAL_FEATURE_NAMES = ("previous_obstacle_lateral_offset",)
TEMPORAL_FEATURE_SIZE = len(TEMPORAL_FEATURE_NAMES)


def current_frame(observation: np.ndarray) -> np.ndarray:
    """Validate an observation and return its newest normalized grayscale frame."""
    try:
        pixels = np.asarray(observation, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("observation must contain normalized float pixels") from error
    if pixels.shape == (4, 84, 84):
        frame = pixels[-1]
    elif pixels.shape == (84, 84):
        frame = pixels
    else:
        raise ValueError("observation must have shape (84, 84) or (4, 84, 84)")
    if not np.all(np.isfinite(frame)) or float(frame.min()) < 0.0 or float(frame.max()) > 1.0:
        raise ValueError("observation pixels must be finite and normalized to [0, 1]")
    return frame


def road_centers(frame: np.ndarray) -> dict[int, float]:
    """Track the visible asphalt center from near to far image rows."""
    asphalt = (frame >= ROAD_LOW) & (frame <= ROAD_HIGH)
    horizontal = np.arange(frame.shape[1], dtype=np.float32)
    centers: dict[int, float] = {}
    previous = 42.0
    for row in (54, 50, 46, 42, 38, 34, 30):
        selected = asphalt[row] & (np.abs(horizontal - previous) <= 17.0)
        locations = np.flatnonzero(selected)
        if len(locations) >= 4:
            previous = float(locations.mean())
            centers[row] = previous
    return centers


def center_at(row: float, centers: dict[int, float]) -> float:
    """Interpolate a road center for a fractional image row."""
    known_rows = sorted(centers)
    if not known_rows:
        return 42.0
    return float(
        np.interp(
            row,
            np.asarray(known_rows, dtype=np.float32),
            np.asarray([centers[y] for y in known_rows], dtype=np.float32),
        )
    )


def nearest_bright_object(
    frame: np.ndarray, centers: dict[int, float]
) -> tuple[float, float, float] | None:
    """Find the nearest small bright component inside the visible road corridor."""
    if len(centers) < 3:
        return None
    bright = frame >= OBSTACLE_LOW
    bright[:22, :] = False
    bright[62:, :] = False
    visited = np.zeros(bright.shape, dtype=np.bool_)
    candidates: list[tuple[float, float, float]] = []
    height, width = bright.shape

    for start_y in range(22, 62):
        for start_x in range(width):
            if not bright[start_y, start_x] or visited[start_y, start_x]:
                continue
            stack = [(start_x, start_y)]
            visited[start_y, start_x] = True
            min_x = max_x = start_x
            min_y = max_y = start_y
            sum_x = sum_y = area = 0
            while stack:
                x, y = stack.pop()
                area += 1
                sum_x += x
                sum_y += y
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for neighbor_y in range(max(22, y - 1), min(62, y + 2)):
                    for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                        if bright[neighbor_y, neighbor_x] and not visited[neighbor_y, neighbor_x]:
                            visited[neighbor_y, neighbor_x] = True
                            stack.append((neighbor_x, neighbor_y))

            box_width = max_x - min_x + 1
            box_height = max_y - min_y + 1
            if not (4 <= area <= 80 and 2 <= box_width <= 9 and 2 <= box_height <= 10):
                continue
            center_x = sum_x / area
            center_y = sum_y / area
            road_center = center_at(center_y, centers)
            if abs(center_x - road_center) <= 12.0:
                candidates.append((center_y, center_x, road_center))

    return max(candidates, key=lambda item: item[0]) if candidates else None


def estimate_speed(frame: np.ndarray) -> float:
    """Decode the rendered speed bar using its calibrated grayscale mass."""
    top, bottom, left, right = SPEED_ROI
    mass = float(np.asarray(frame, dtype=np.float32)[top:bottom, left:right].sum())
    return float(np.clip((mass - SPEED_BASELINE) / SPEED_PER_UNIT, 0.0, 80.0))


def estimate_observation_speed(observation: np.ndarray, frame: np.ndarray) -> float:
    """Average the last two rendered speed readings in the frame stack."""
    pixels = np.asarray(observation, dtype=np.float32)
    stack = pixels[-2:] if pixels.shape == (4, 84, 84) else pixels[np.newaxis, ...]
    return float(np.mean([estimate_speed(item) for item in stack]))


def road_sweep(centers: dict[int, float]) -> float:
    """Estimate bend strength from visible near/far road-center changes."""
    return max(
        (
            abs(centers.get(row, 42.0) - centers.get(row + 24, 42.0))
            for row in (30, 34, 38, 42)
        ),
        default=0.0,
    )


def extract_visual_features(observation: np.ndarray) -> np.ndarray:
    """Encode HUD speed, road alignment, bend, and nearest obstacle from pixels."""
    frame = current_frame(observation)
    centers = road_centers(frame)
    obstacle = nearest_bright_object(frame, centers)
    if obstacle is None:
        obstacle_present = 0.0
        obstacle_lateral = 0.0
        obstacle_urgency = 0.0
    else:
        obstacle_y, obstacle_x, obstacle_road_center = obstacle
        obstacle_present = 1.0
        obstacle_lateral = float(
            np.clip((obstacle_x - obstacle_road_center) / 12.0, -1.0, 1.0)
        )
        obstacle_urgency = float(np.clip((obstacle_y - 22.0) / 18.0, 0.0, 1.0))
    return np.asarray(
        (
            estimate_observation_speed(observation, frame) / 80.0,
            np.clip((centers.get(54, 42.0) - 42.0) / 42.0, -1.0, 1.0),
            np.clip((centers.get(42, 42.0) - 42.0) / 42.0, -1.0, 1.0),
            np.clip(road_sweep(centers) / 42.0, 0.0, 1.0),
            obstacle_present,
            obstacle_lateral,
            obstacle_urgency,
        ),
        dtype=np.float32,
    )


def extract_visual_features_batch(observations: np.ndarray) -> np.ndarray:
    """Extract a row per observation for a batch of ``(B, 4, 84, 84)`` pixels."""
    pixels = np.asarray(observations, dtype=np.float32)
    if pixels.ndim != 4 or pixels.shape[1:] != (4, 84, 84):
        raise ValueError("observations must have shape (B, 4, 84, 84)")
    if not np.all(np.isfinite(pixels)) or float(pixels.min()) < 0.0 or float(pixels.max()) > 1.0:
        raise ValueError("observation pixels must be finite and normalized to [0, 1]")
    return np.stack([extract_visual_features(item) for item in pixels], axis=0)


def extract_temporal_features(observation: np.ndarray) -> np.ndarray:
    """Read a prior-frame obstacle side only when the current view is centered."""
    pixels = np.asarray(observation, dtype=np.float32)
    if pixels.shape != (4, 84, 84):
        raise ValueError("observation must have shape (4, 84, 84)")
    current = pixels[-1]
    previous = pixels[-2]
    current_centers = road_centers(current)
    current_obstacle = nearest_bright_object(current, current_centers)
    if current_obstacle is None:
        return np.zeros(TEMPORAL_FEATURE_SIZE, dtype=np.float32)
    _, current_x, current_road_center = current_obstacle
    current_lateral = float((current_x - current_road_center) / 12.0)
    if abs(current_lateral) >= 0.2:
        return np.zeros(TEMPORAL_FEATURE_SIZE, dtype=np.float32)
    previous_centers = road_centers(previous)
    previous_obstacle = nearest_bright_object(previous, previous_centers)
    if previous_obstacle is None:
        return np.zeros(TEMPORAL_FEATURE_SIZE, dtype=np.float32)
    _, previous_x, previous_road_center = previous_obstacle
    previous_lateral = float(
        np.clip((previous_x - previous_road_center) / 12.0, -1.0, 1.0)
    )
    if abs(previous_lateral) < 0.3:
        return np.zeros(TEMPORAL_FEATURE_SIZE, dtype=np.float32)
    return np.asarray((previous_lateral,), dtype=np.float32)


def extract_temporal_features_batch(observations: np.ndarray) -> np.ndarray:
    """Extract one temporal side feature per ``(B, 4, 84, 84)`` observation."""
    pixels = np.asarray(observations, dtype=np.float32)
    if pixels.ndim != 4 or pixels.shape[1:] != (4, 84, 84):
        raise ValueError("observations must have shape (B, 4, 84, 84)")
    if not np.all(np.isfinite(pixels)) or float(pixels.min()) < 0.0 or float(pixels.max()) > 1.0:
        raise ValueError("observations pixels must be finite and normalized to [0, 1]")
    return np.stack([extract_temporal_features(item) for item in pixels], axis=0)


__all__ = [
    "FEATURE_NAMES",
    "TEMPORAL_FEATURE_NAMES",
    "TEMPORAL_FEATURE_SIZE",
    "VISUAL_FEATURE_SIZE",
    "center_at",
    "current_frame",
    "estimate_observation_speed",
    "estimate_speed",
    "extract_visual_features",
    "extract_visual_features_batch",
    "extract_temporal_features",
    "extract_temporal_features_batch",
    "nearest_bright_object",
    "road_centers",
    "road_sweep",
]
