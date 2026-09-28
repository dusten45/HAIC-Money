"""Exclusive gas/brake adapter for a separately trained two-action TD-MPC2 model."""

import numpy as np


def environment_action(action: np.ndarray) -> np.ndarray:
    """Convert [steer, signed longitudinal] to official [steer, gas, brake]."""
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (2,) or not np.isfinite(value).all() or (np.abs(value) > 1).any():
        raise ValueError("2D action must be a finite [-1,1] steering/longitudinal pair")
    return np.array([value[0], max(0, value[1]), max(0, -value[1])], dtype=np.float32)


def model_action(action: np.ndarray) -> np.ndarray:
    """Invert only actions on the exclusive-pedal manifold, never clip silently."""
    value = np.asarray(action, dtype=np.float32)
    if (value.shape != (3,) or not np.isfinite(value).all()
            or not -1 <= value[0] <= 1 or (value[1:] < 0).any()
            or (value[1:] > 1).any() or value[1] * value[2] != 0):
        raise ValueError("native action must be bounded and have mutually exclusive pedals")
    return np.array([value[0], value[1] - value[2]], dtype=np.float32)
