"""Pixel-only, threshold-free motion statistic for a four-frame RLPD input."""

import numpy as np


def visual_motion_score(stack: np.ndarray) -> float:
    """Mean absolute adjacent-frame difference over 3 pairs and 84x84 pixels.

    Accept only CHW float32 pixels in [0, 1] or an exact uint8 stack. No action,
    vehicle telemetry, or episode history is an input to this statistic.
    """
    pixels = np.asarray(stack)
    if pixels.shape != (4, 84, 84):
        raise ValueError("visual motion requires exactly four 84x84 CHW frames")
    if pixels.dtype == np.uint8:
        pixels = pixels.astype(np.float32) / 255.0
    elif pixels.dtype != np.float32 or not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels > 1)):
        raise ValueError("visual motion requires uint8 or finite float32 pixels in [0, 1]")
    return float(np.abs(pixels[1:] - pixels[:-1]).mean(dtype=np.float64))
