"""Static differential coverage for the visible compound brake onset."""

from unittest.mock import patch

import numpy as np
import pytest

from agent import (
    _CompoundBrakeOnsetController,
    _CompoundClearingBrakeCarryController,
)


def _pair():
    return _CompoundClearingBrakeCarryController(), _CompoundBrakeOnsetController()


def _act_pair(pair, *, speed, obstacle=(32.0, 35.0, 42.0), sweep=6.0,
              direction=1.0, strong=False):
    centers = {row: 42.0 for row in (30, 34, 38, 42, 46, 50, 54, 58, 62)}
    centers[30] += direction * sweep
    if obstacle is not None and direction < 0:
        obstacle = (obstacle[0], 84.0 - obstacle[1], 42.0)
    actions = []
    for controller in pair:
        with patch.object(controller, "_road_centers", return_value=centers), \
             patch.object(controller, "_nearest_obstacle", return_value=obstacle), \
             patch.object(controller, "_estimate_speed", return_value=speed), \
             patch.object(controller, "_adjust_obstacle_steering",
                          wraps=controller._adjust_obstacle_steering) as steering:
            if strong:
                steering.return_value = direction * 0.281
            actions.append(controller.act(np.zeros((4, 84, 84), dtype=np.float32)))
    return actions


def _assert_shared_state(pair):
    assert vars(pair[0]) == vars(pair[1])


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_visible_compound_onset_speed_boundaries(direction):
    # Values straddle both the supplemental onset and inherited base onset.
    for delta in (-1.0, 0.0, 0.5, 0.500001, 0.75, 1.0, 1.000001,
                  1.5, 2.0, 2.499999, 2.5, 8.0, 42.0):
        pair = _pair()
        control, candidate = _act_pair(pair, speed=38.0 + delta, direction=direction)
        assert pair[0]._pace_command_target == 38.0
        np.testing.assert_array_equal(candidate[:2], control[:2])
        _assert_shared_state(pair)
        assert np.isfinite(candidate).all()
        assert not (candidate[1] > 0 and candidate[2] > 0)
        assert 0 <= float(control[2] - candidate[2]) <= 0.040001
        if delta <= 0.5 or delta >= 2.5:
            np.testing.assert_array_equal(candidate, control)
        else:
            excess = delta - 0.5
            base = np.clip(delta * 0.012, 0.04, 0.28) if delta > 1 else 0.0
            expected = max(base, (0.04 + 0.012 * excess) * excess / 2.0)
            assert float(candidate[2]) == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("kwargs", [
    {"obstacle": (44.0, 35.0, 42.0)},
    {"obstacle": (55.0, 35.0, 42.0)},
    {"sweep": 5.999}, {"sweep": 12.0}, {"sweep": 16.0},
    {"strong": True}, {"obstacle": None},
])
def test_ineligible_geometry_remains_exact(kwargs):
    for speed in (30.75, 38.75, 40.75, 47.75, 70.0):
        pair = _pair()
        actions = _act_pair(pair, speed=speed, **kwargs)
        np.testing.assert_array_equal(*actions)
        _assert_shared_state(pair)


def test_misses_reacquisition_clearing_and_reset_preserve_shared_state():
    pair = _pair()
    for present in (True, False, True, False, False, False, False, False, False,
                    True, False, True):
        actions = _act_pair(pair, speed=38.75,
                            obstacle=(32.0, 35.0, 42.0) if present else None)
        np.testing.assert_array_equal(actions[0][:2], actions[1][:2])
        _assert_shared_state(pair)
        if not present:
            np.testing.assert_array_equal(*actions)
        else:
            assert actions[1][2] < actions[0][2]
    for controller in pair:
        controller.reset()
    _assert_shared_state(pair)
    for invalid in (None, np.full((4, 84, 84), np.nan), np.zeros((4, 84, 84))):
        np.testing.assert_array_equal(*(controller.act(invalid) for controller in pair))
        _assert_shared_state(pair)
    _act_pair(pair, speed=38.75)
    # Lost road after acquiring a valid road follows the existing recovery path.
    np.testing.assert_array_equal(*(controller.act(None) for controller in pair))
    _assert_shared_state(pair)


def test_direct_envelope_preserves_floor_and_is_bounded_randomly():
    rng = np.random.default_rng(47913)
    for _ in range(400):
        pair = _pair()
        for controller in pair:
            controller._obstacle_side = 1.0
            controller._obstacle_missing = 0
            controller._pace_latched_target = 30.0
            controller._pace_command_target = 35.0
            controller._pace_sweep = 8.0
        excess = float(rng.uniform(0.0, 30.0))
        base = float(rng.uniform(0.0, 0.28))
        control, candidate = (controller._curve_brake_envelope(
            excess=excess, base_brake=base) for controller in pair)
        assert base <= candidate <= control
        assert control - candidate <= 0.04 + 1e-12
        if excess >= 2:
            assert candidate == control
        for attribute, excluded in (("_obstacle_side", 0.0),
                                    ("_obstacle_missing", 1),
                                    ("_pace_latched_target", 40.0),
                                    ("_pace_command_target", 30.0),
                                    ("_pace_command_target", 38.001),
                                    ("_pace_sweep", 5.999),
                                    ("_pace_sweep", 12.0)):
            previous = getattr(pair[0], attribute)
            for controller in pair:
                setattr(controller, attribute, excluded)
            assert pair[0]._curve_brake_envelope(excess=excess, base_brake=base) == \
                pair[1]._curve_brake_envelope(excess=excess, base_brake=base)
            for controller in pair:
                setattr(controller, attribute, previous)


def _synthetic_camera(speed):
    frame = np.full((84, 84), 0.1, dtype=np.float32)
    for row in range(20, 63):
        center = 42.0 + 0.5 * max(42 - row, 0)
        frame[row, int(round(center - 11)):int(round(center + 11))] = 0.4
    frame[30:34, 34:37] = 0.68
    frame[77:83, 10:13] = (0.27 + 0.085 * speed) / 18.0
    return np.repeat(frame[None, :, :], 4, axis=0)


def test_real_camera_detection_reaches_qualified_onset():
    probe = _CompoundClearingBrakeCarryController()
    probe.act(_synthetic_camera(0.0))
    target = probe._pace_command_target
    assert 30.0 < target <= 38.0
    pair = _pair()
    actions = [controller.act(_synthetic_camera(target + 0.75)) for controller in pair]
    _assert_shared_state(pair)
    np.testing.assert_array_equal(actions[0][:2], actions[1][:2])
    assert 0.0 < actions[1][2] < actions[0][2]
    assert actions[0][2] - actions[1][2] <= 0.04
