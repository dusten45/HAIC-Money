"""Pure reward-target checks; no HAIC environment or optimizer."""

import math

import numpy as np
import pytest

from haic.algorithms.tdmpc2.reward import DAMAGE_COST, DamageIncrementReward


def test_zero_event_repeated_increment_saturation_and_episode_reset():
    shaper = DamageIncrementReward()
    with pytest.raises(RuntimeError, match="requires an episode reset"):
        shaper.step(1.0, 0.0)
    shaper.reset()
    assert shaper.step(0.25, 0.0) == (0.25, 0.0)
    train, delta = shaper.step(0.5, 0.2)
    assert delta == pytest.approx(0.2)
    assert train == pytest.approx(0.5 - DAMAGE_COST * 0.2)
    assert shaper.step(0.5, 0.2) == (0.5, 0.0)
    assert shaper.step(-1.0, 1.0) == pytest.approx((-5.0, 0.8))
    assert shaper.step(0.5, 1.0) == (0.5, 0.0)
    shaper.reset()
    assert shaper.step(0.5, np.float32(0.2)) == pytest.approx((-0.5, 0.2))


@pytest.mark.parametrize("reward,damage", [
    (float("nan"), 0.0), (float("inf"), 0.0), ("1.0", 0.0),
    (True, 0.0), (0.0, float("nan")), (0.0, float("-inf")),
    (0.0, -0.01), (0.0, 1.01), (0.0, None), (0.0, True),
])
def test_invalid_reward_or_damage_does_not_advance_episode(reward, damage):
    shaper = DamageIncrementReward()
    shaper.reset()
    shaper.step(0.0, 0.2)
    with pytest.raises(ValueError):
        shaper.step(reward, damage)
    assert shaper.step(1.0, 0.4) == pytest.approx((0.0, 0.2))


def test_material_negative_reset_within_episode_fails_but_floating_dust_is_ignored():
    shaper = DamageIncrementReward()
    shaper.reset()
    shaper.step(0.0, 0.4)
    assert shaper.step(1.0, 0.4 - 1e-7) == (1.0, 0.0)
    with pytest.raises(ValueError, match="decreased within episode"):
        shaper.step(1.0, 0.2)
    assert shaper.step(1.0, 0.4) == (1.0, 0.0)
    shaper.reset()
    assert shaper.step(1.0, 0.0) == (1.0, 0.0)


def test_very_large_finite_raw_reward_remains_finite():
    shaper = DamageIncrementReward()
    shaper.reset()
    assert math.isfinite(shaper.step(-1e308, 0.2)[0])
