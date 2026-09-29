"""Small, explicit transforms for the PPO obstacle-risk experiment."""

from __future__ import annotations

import math


def rescale_quadratic_hazard_risk(
    quadratic_risk: float,
    urgency: float,
    urgency_exponent: float,
) -> float:
    """Convert a risk value containing ``urgency**2`` to another power.

    The obstacle presence and lane-overlap factors already included in
    ``quadratic_risk`` are preserved.  At urgency zero, risk remains zero.
    """
    risk = float(quadratic_risk)
    urgency_value = float(urgency)
    exponent = float(urgency_exponent)
    if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
        raise ValueError("quadratic_risk must be finite and in [0, 1]")
    if not math.isfinite(urgency_value) or not 0.0 <= urgency_value <= 1.0:
        raise ValueError("urgency must be finite and in [0, 1]")
    if not math.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("urgency_exponent must be finite and positive")
    if risk == 0.0 or urgency_value == 0.0:
        return 0.0
    return risk * urgency_value ** (exponent - 2.0)
