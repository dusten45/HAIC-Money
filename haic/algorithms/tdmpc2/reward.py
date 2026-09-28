"""TRAIN-target-only TD-MPC2 damage-increment reward; never modify HAIC reward."""

import math
from numbers import Real

from damage import MAX_DAMAGE


DAMAGE_COST = 5.0
NEGATIVE_DELTA_TOLERANCE = 1e-6


class DamageIncrementReward:
    def __init__(self) -> None:
        self._previous_damage: float | None = None

    def reset(self) -> None:
        """Start a new environment episode with zero cumulative damage."""
        self._previous_damage = 0.0

    def step(self, raw_reward: object, cumulative_damage: object) -> tuple[float, float]:
        """Return (training reward, new damage); reject corrupt episode telemetry."""
        if self._previous_damage is None:
            raise RuntimeError("damage reward requires an episode reset")
        if (isinstance(raw_reward, bool) or not isinstance(raw_reward, Real)
                or isinstance(cumulative_damage, bool) or not isinstance(cumulative_damage, Real)):
            raise ValueError("reward and cumulative damage must be real scalars")
        raw, current = float(raw_reward), float(cumulative_damage)
        if not math.isfinite(raw) or not math.isfinite(current) or not 0 <= current <= MAX_DAMAGE:
            raise ValueError("nonfinite or out-of-bounds reward/damage telemetry")
        change = current - self._previous_damage
        if change < -NEGATIVE_DELTA_TOLERANCE:
            raise ValueError("cumulative damage decreased within episode")
        damage_delta = max(0.0, change)
        shaped = raw - DAMAGE_COST * damage_delta
        if not math.isfinite(shaped):
            raise ValueError("nonfinite training reward")
        self._previous_damage = max(self._previous_damage, current)
        return shaped, damage_delta
