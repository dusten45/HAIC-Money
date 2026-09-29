"""Inference-only, brake-only speed candidate for a frozen DrQ-v2 actor."""

import numpy as np


def release_light_brake(official_action):
    """Reduce competing light brake on near-straights; never alter steering or gas.

    The input is already mapped to the official action box. It must not be
    converted from native DrQ pedal coordinates a second time.
    """
    action = np.asarray(official_action, dtype=np.float32)
    if (action.shape != (3,) or not np.isfinite(action).all()
            or not -1.0 <= action[0] <= 1.0
            or np.any(action[1:] < 0.0) or np.any(action[1:] > 1.0)):
        raise ValueError("expected a finite official [steer, gas, brake] action")
    result = action.copy()
    if (abs(result[0]) <= np.float32(0.20) and result[1] >= np.float32(0.80)
            and 0.0 < result[2] <= np.float32(0.30)):
        result[2] *= np.float32(0.25)
    return result
