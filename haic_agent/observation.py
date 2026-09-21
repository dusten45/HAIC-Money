"""Inference-safe pixel observation helpers.

This module deliberately accepts only the processed 84 by 84 pixel view.  It
does not import the simulator or accept any simulator state.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


# Coordinates are in the 84x84 grayscale image after env_wrapper preprocessing.
# They cover the indicators rendered in the simulator's lower HUD strip.
HUD_ROIS = {
    "speed": (72, 84, 9, 14),
    "wheel_0": (72, 84, 14, 18),
    "wheel_1": (72, 84, 16, 20),
    "wheel_2": (72, 84, 18, 22),
    "wheel_3": (72, 84, 20, 24),
    "steering": (74, 82, 40, 58),
    "yaw": (74, 82, 61, 79),
}


@dataclass(frozen=True)
class HUDFeatures:
    """The current processed frame and requested fixed-position HUD crops."""

    full_frame: np.ndarray
    regions: Mapping[str, np.ndarray]


def _current_frame(observation: np.ndarray) -> np.ndarray:
    pixels = np.asarray(observation)
    if pixels.ndim == 3 and pixels.shape[1:] == (84, 84):
        pixels = pixels[-1]
    if pixels.shape != (84, 84):
        raise ValueError("HUD features require a processed (84, 84) pixel frame")
    if pixels.dtype != np.float32:
        raise TypeError("HUD features require float32 processed pixels")
    return pixels


def extract_hud_features(
    observation: np.ndarray, channels: Sequence[str] | None = None
) -> HUDFeatures:
    """Extract fixed HUD ROIs from a processed pixel frame or frame stack.

    ``channels`` makes unreliable gauges removable without changing the full
    frame input used by the visual policy.
    """
    frame = _current_frame(observation)
    names = tuple(HUD_ROIS) if channels is None else tuple(channels)
    unknown = set(names) - set(HUD_ROIS)
    if unknown:
        raise ValueError(f"unknown HUD channels: {sorted(unknown)}")
    regions = {
        name: frame[top:bottom, left:right].copy()
        for name, (top, bottom, left, right) in HUD_ROIS.items()
        if name in names
    }
    return HUDFeatures(full_frame=frame.copy(), regions=regions)
