"""Physical camera tests for the isolated faster pursuit candidate."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "fast_preview_agent.py"


def _module():
    spec = importlib.util.spec_from_file_location("apex_fast_preview_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _camera(*, radius=None, side=1, center=42, speed=70, obstacle=False):
    frame = np.full((84, 84), .63, dtype=np.float32)
    for row in range(73):
        ahead = (63 - row) / 1.701
        if radius is None:
            middle, halfwidth = float(center), 9.0
        elif abs(ahead) < radius:
            middle = center + side * 1.3608 * (radius - np.sqrt(radius**2 - ahead**2))
            halfwidth = 1.3608 * (20 / 3) / np.sqrt(1 - (ahead / radius)**2)
        else:
            continue
        lo, hi = max(0, round(middle - halfwidth)), min(84, round(middle + halfwidth) + 1)
        frame[row, lo:hi] = .40
    if obstacle:
        frame[42:46, 40:44] = .67
    frame[73:] = 0.0
    top, bottom = 81.9 - .042 * speed, 81.9
    for row in range(73, 84):
        for col in range(10, 14):
            width = max(0.0, min(col + 1.0, 12.6) - max(float(col), 10.5))
            height = max(0.0, min(row + 1.0, bottom) - max(float(row), top))
            frame[row, col] = width * height
    return np.repeat(frame[None], 4, axis=0)


def test_quantized_aligned_circle_accelerates_below_pursuit_grip_capacity():
    agent = _module().Agent()
    action = agent.act(_camera(radius=60, speed=70))
    assert agent.mode == "metric"
    assert agent.last_target > 95.0
    assert action[1] > .25
    assert action[2] == 0.0
    assert action[0] == pytest.approx(np.arctan(3.24 / 60), rel=.20)


@pytest.mark.parametrize("side", [-1, 1])
def test_pursuit_retains_sharp_corner_braking_and_physical_steering_bound(side):
    metric = _module().Agent()._metric
    action = metric.act(_camera(radius=25, side=side, speed=110))
    assert action[0] * side > 0
    assert 0 < abs(action[0]) <= np.arctan(3.24 * metric.lateral_accel / metric.last_speed**2) + 1e-6
    assert metric.last_target < metric.last_speed - 2
    assert action[1] == 0.0 and action[2] > 0.0


def test_shorter_preview_corrects_visible_straight_offset_promptly():
    agent = _module().Agent()
    action = agent.act(_camera(center=46, speed=50))
    assert agent.mode == "metric"
    assert .12 < action[0] < .24
    assert action[1] > .25 and action[2] == 0.0


def test_fast_small_steering_leaves_rear_wheel_force_for_lateral_grip():
    # Explicit future defaults isolate the missing rear-grip budget from the
    # existing candidate's separate conservative parameter defaults.
    agent = _module().Agent(cruise_speed=130, lateral_accel=190,
        preview_time=.12, robust_cruise_speed=100, propulsion=1,
        corner_boost=.5, unstable_sweep=40, low_gas_gate=.90, high_gas_gate=.98)
    action = agent.act(_camera(center=43, speed=100))
    assert agent.mode == "metric"
    assert .005 < action[0] < .03
    assert .5 < action[1] <= .80
    assert action[2] == 0.0


def test_fast_straight_keeps_full_acceleration():
    agent = _module().Agent()
    action = agent.act(_camera(speed=90))
    assert agent.mode == "metric"
    assert abs(action[0]) < .001
    assert action[1] > .8 and action[2] == 0.0


def test_accelerating_tighter_bend_caps_gas_before_rear_wheel_spin():
    # Uniform-asphalt characterization found the instantaneous tire-circle
    # budget could spin .06-rad steering while accelerating from70m/s. Fixed
    # gas .3 remained stable from70 and100; .5 and above did not.
    agent = _module().Agent()
    action = agent.act(_camera(radius=60, speed=70))
    assert .04 < action[0] < .08
    assert agent.last_target > 95
    assert .25 < action[1] <= .30 + 1e-6
    assert action[2] == 0.0


def test_near_force_limit_keeps_declared_gas_gate():
    agent = _module().Agent()
    action = agent.act(_camera(center=47, speed=90))
    demand = agent.last_speed**2 * abs(np.tan(float(action[0]))) / (3.24 * agent._metric.lateral_accel)
    assert demand > agent._metric.high_gas_gate
    assert action[1] <= .16 + 1e-6


def test_real_camera_obstacle_retains_guarded_robust_pedals():
    agent = _module().Agent()
    action = agent.act(_camera(speed=70, obstacle=True))
    assert agent.mode in ("robust", "guided")
    assert agent._robust._corridor_bbox is not None
    assert agent.last_target <= agent._robust._pace_effective_target
    if action[2] > 0:
        assert action[1] == 0.0
    assert action[1] <= .3


def test_reset_and_invalid_camera_preserve_finite_bounded_actions():
    agent = _module().Agent()
    observation = _camera(radius=60, speed=70)
    first = agent.act(observation)
    for _ in range(6):
        action = agent.act(np.full((4, 84, 84), np.nan, dtype=np.float32))
        assert action.dtype == np.float32 and action.shape == (3,)
        assert np.isfinite(action).all()
        assert np.all(action >= [-1, 0, 0]) and np.all(action <= [1, 1, 1])
    agent.reset()
    np.testing.assert_array_equal(agent.act(observation), first)


@pytest.mark.parametrize("parameters", [{"preview_time":float("nan")}, {"lateral_accel":0}, {"high_gas_gate":2}])
def test_invalid_physical_controller_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        _module().Agent(**parameters)
