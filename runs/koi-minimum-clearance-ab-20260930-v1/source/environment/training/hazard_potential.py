"""Potential-based reward shaping for visible hazard risk."""

from __future__ import annotations

import math


def _finite_float(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error

    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def hazard_potential(risk: float) -> float:
    """Return the potential Φ(s) = -risk(s) for risk in [0, 1]."""
    value = _finite_float("risk", risk)
    if not 0.0 <= value <= 1.0:
        raise ValueError("risk must be between 0 and 1")
    return -value


def hazard_potential_shaping_reward(
    current_risk: float,
    next_risk: float,
    gamma: float,
    scale: float,
    terminated: bool = False,
    truncated: bool = False,
) -> float:
    """Compute scale * (gamma * Φ(s') - Φ(s)).

    Terminal and truncated transitions use zero for the next-state potential.
    The supplied risks must be finite values in [0, 1], gamma in [0, 1],
    and scale a finite nonnegative value.
    """
    current_potential = hazard_potential(current_risk)
    next_potential = hazard_potential(next_risk)
    discount = _finite_float("gamma", gamma)
    if not 0.0 <= discount <= 1.0:
        raise ValueError("gamma must be between 0 and 1")

    shaping_scale = _finite_float("scale", scale)
    if shaping_scale < 0.0:
        raise ValueError("scale must be nonnegative")

    if terminated or truncated:
        next_potential = 0.0

    return shaping_scale * (discount * next_potential - current_potential)
