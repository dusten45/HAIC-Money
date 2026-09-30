"""Differential contract for visible compound base-brake diagnostics."""

from unittest.mock import patch

import numpy as np
import pytest

from agent import (
    _CompoundClearingBrakeCarryController,
    _VisibleCompoundBaseBrakeController,
)


def _pair():
    return _CompoundClearingBrakeCarryController(), _VisibleCompoundBaseBrakeController()


def _act(pair, *, speed, obstacle=(46.0, 35.0, 42.0), sweep=6.0,
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
    assert vars(pair[0]) == vars(pair[1])
    np.testing.assert_array_equal(actions[0][:2], actions[1][:2])
    assert np.isfinite(actions[1]).all()
    assert not (actions[1][1] > 0 and actions[1][2] > 0)
    return actions


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("kwargs", [
    {}, {"obstacle": (44.0, 35.0, 42.0)},
    {"obstacle": (32.0, 35.0, 42.0), "sweep": 12.0},
    {"obstacle": (32.0, 35.0, 42.0), "sweep": 16.0},
    {"obstacle": (32.0, 35.0, 42.0), "strong": True},
])
def test_current_near_extreme_and_strong_compounds_use_base(direction, kwargs):
    for speed in (28.0, 30.0, 30.5, 30.500001, 31.0, 31.000001,
                  32.0, 35.0, 40.0, 45.0, 60.0, 80.0):
        pair = _pair()
        control, candidate = _act(pair, speed=speed, direction=direction, **kwargs)
        assert pair[0]._pace_command_target == 30.0
        base = np.clip((speed - 30.0) * 0.012, 0.04, 0.28) if speed > 31 else 0.0
        assert float(candidate[2]) == pytest.approx(base, abs=1e-7)
        assert 0 <= float(control[2] - candidate[2]) <= 0.046001
        if speed > 31:
            assert float(control[2] - candidate[2]) <= 0.034001
        if speed <= 30.5 or speed >= 45:
            np.testing.assert_array_equal(candidate, control)


def test_registered_numeric_examples():
    for speed, expected in ((35.0, (0.094, 0.060)),
                            (40.0, (0.154, 0.120)),
                            (45.0, (0.180, 0.180))):
        actions = _act(_pair(), speed=speed)
        assert [float(action[2]) for action in actions] == pytest.approx(expected)


def test_misses_clear_and_adaptive_reacquisition_remain_exact():
    pair = _pair()
    control, candidate = _act(pair, speed=35.0)
    assert candidate[2] < control[2]
    for _ in range(6):
        np.testing.assert_array_equal(*_act(pair, speed=35.0, obstacle=None))
    # A qualified adaptive obstacle does not receive the candidate envelope.
    np.testing.assert_array_equal(*_act(pair, speed=38.75,
                                      obstacle=(32.0, 35.0, 42.0)))
    for _ in range(6):
        np.testing.assert_array_equal(*_act(pair, speed=40.0, obstacle=None))
    # Reacquisition qualifies from the current detection, without extra memory.
    control, candidate = _act(pair, speed=35.0)
    assert candidate[2] < control[2]
    for controller in pair:
        controller.reset()
    assert vars(pair[0]) == vars(pair[1])
    for observation in (None, np.full((4, 84, 84), np.nan)):
        np.testing.assert_array_equal(*(controller.act(observation) for controller in pair))
    _act(pair, speed=35.0)
    np.testing.assert_array_equal(*(controller.act(None) for controller in pair))
    assert vars(pair[0]) == vars(pair[1])


@pytest.mark.parametrize("kwargs", [
    {"sweep": 5.999}, {"sweep": 0.0}, {"obstacle": None},
    {"obstacle": (32.0, 35.0, 42.0)},
    {"obstacle": (43.999, 35.0, 42.0)},
])
def test_nonqualifying_actions_are_exact(kwargs):
    for speed in (30.75, 35.0, 38.75, 40.75, 47.75, 75.0):
        np.testing.assert_array_equal(*_act(_pair(), speed=speed, **kwargs))


def test_envelope_state_gates_and_base_floor():
    rng = np.random.default_rng(55211)
    for _ in range(250):
        pair = _pair()
        for controller in pair:
            controller._obstacle_side = float(rng.choice((-1, 1)))
            controller._obstacle_missing = 0
            controller._pace_latched_target = 30.0
            controller._pace_command_target = 30.0
            controller._pace_sweep = 8.0
        excess = float(rng.uniform(0.0, 50.0))
        # Use a physically reachable inherited brake for this overspeed.
        delta = excess + 0.5
        base = float(np.clip(delta * 0.012, 0.04, 0.28)) if delta > 1 else 0.0
        control, candidate = [controller._curve_brake_envelope(
            excess=excess, base_brake=base) for controller in pair]
        assert candidate == base
        assert 0 <= control - candidate <= 0.046 + 1e-12
        if delta > 1:
            assert control - candidate <= 0.034 + 1e-12
        for attribute, excluded in (("_obstacle_side", 0.0),
                                    ("_obstacle_missing", 1),
                                    ("_pace_latched_target", 40.0),
                                    ("_pace_command_target", 35.0),
                                    ("_pace_command_target", None),
                                    ("_pace_sweep", 5.999)):
            previous = [getattr(controller, attribute) for controller in pair]
            for controller in pair:
                setattr(controller, attribute, excluded)
            assert pair[0]._curve_brake_envelope(excess=excess, base_brake=base) == \
                pair[1]._curve_brake_envelope(excess=excess, base_brake=base)
            for controller, value in zip(pair, previous):
                setattr(controller, attribute, value)


def test_unmocked_camera_near_compound_detection():
    frame = np.full((84, 84), 0.1, dtype=np.float32)
    for row in range(20, 63):
        center = 42.0 + 0.5 * max(42 - row, 0)
        frame[row, int(round(center - 11)):int(round(center + 11))] = 0.4
    frame[46:50, 34:37] = 0.68
    frame[77:83, 10:13] = (0.27 + 0.085 * 35.0) / 18.0
    observation = np.repeat(frame[None, :, :], 4, axis=0)
    pair = _pair()
    actions = [controller.act(observation) for controller in pair]
    assert pair[0]._pace_command_target == 30.0
    assert vars(pair[0]) == vars(pair[1])
    np.testing.assert_array_equal(actions[0][:2], actions[1][:2])
    assert float(actions[1][2]) == pytest.approx(0.06, abs=1e-6)
    assert float(actions[0][2]) == pytest.approx(0.094, abs=1e-6)
