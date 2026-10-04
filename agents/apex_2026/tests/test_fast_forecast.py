"""Projection should reduce torque before acceleration saturates a tight bend."""
import importlib.util
from pathlib import Path

import numpy as np

from agents.apex_2026.tests.test_fast_envelope import camera


def agent_type():
    root = Path(__file__).resolve().parents[1]
    source = root / "fast_forecast_agent.py"
    if not source.exists():
        source = root / "fast_path_agent.py"
    spec = importlib.util.spec_from_file_location("fast_forecast_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def test_accelerating_turn_reserves_rear_grip_for_projected_cruise_load():
    controller = agent_type()()
    action = controller.act(camera(curvature=.015, speed=70))
    assert abs(action[0]) > .04
    assert 0 < action[1] <= .300001
    assert action[2] == 0


def test_mild_bend_retains_available_high_acceleration():
    action = agent_type()().act(camera(curvature=.003, speed=60))
    assert action[1] > .9 and action[2] == 0


def test_straight_launch_and_invalid_recovery_remain_bounded():
    controller = agent_type()()
    assert controller.act(camera(speed=15))[1] > .9
    action = controller.act(np.full((4, 84, 84), np.nan, dtype=np.float32))
    assert np.isfinite(action).all()
    assert np.all(action >= [-1, 0, 0]) and np.all(action <= 1)
    controller.reset()
    np.testing.assert_array_equal(controller.act(camera()), agent_type()().act(camera()))
