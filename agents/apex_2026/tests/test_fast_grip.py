"""Avoid a front-joint request exceeding the measured high-speed grip budget."""
import importlib.util
from pathlib import Path

import numpy as np

from agents.apex_2026.tests.test_fast_envelope import camera


def agent_type():
    root = Path(__file__).resolve().parents[1]
    source = root / "fast_grip_agent.py"
    if not source.exists():
        source = root / "fast_path_agent.py"
    spec = importlib.util.spec_from_file_location("fast_grip_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def test_high_speed_lane_error_cannot_request_unachievable_front_curvature():
    controller = agent_type()()
    action = controller.act(camera(speed=100, center=54))
    requested_lateral = controller.last_speed**2*abs(np.tan(action[0]))/3.24
    assert requested_lateral <= 190.01
    assert controller.last_target < 100
    assert controller.last_steer == float(action[0])


def test_braking_and_steering_share_the_working_tire_force_budget():
    controller = agent_type()()
    action = controller.act(camera(speed=90, curvature=.025))
    assert action[2] > .2 and action[1] == 0
    requested_lateral = controller.last_speed**2*abs(np.tan(action[0]))/3.24
    requested_longitudinal = 309*action[2]
    assert np.hypot(requested_lateral, requested_longitudinal) <= 210.01


def test_straight_launch_keeps_useful_propulsion():
    action = agent_type()().act(camera(speed=15))
    assert action[1] > .9 and action[2] == 0


def test_invalid_recovery_is_finite_and_reset_forgets_clipped_steering():
    controller = agent_type()()
    controller.act(camera(speed=100, center=54))
    action = controller.act(np.full((4, 84, 84), np.nan, dtype=np.float32))
    assert np.isfinite(action).all()
    assert np.all(action >= [-1, 0, 0]) and np.all(action <= 1)
    controller.reset()
    np.testing.assert_array_equal(controller.act(camera()), agent_type()().act(camera()))
