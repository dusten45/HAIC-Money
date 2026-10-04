"""Earlier deceleration when the observed speed envelope is falling."""
import importlib.util
from pathlib import Path

import numpy as np

from agents.apex_2026.tests.test_fast_envelope import camera


def agent_type():
    root = Path(__file__).resolve().parents[1]
    source = root / "fast_braking_agent.py"
    if not source.exists():
        source = root / "fast_path_agent.py"
    spec = importlib.util.spec_from_file_location("fast_braking_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def test_falling_corner_envelope_brakes_without_waiting_for_large_overspeed():
    controller = agent_type()()
    controller.act(camera(speed=90))
    action = controller.act(camera(curvature=.012, speed=90))
    assert action[1] == 0
    assert action[2] > .2
    assert action[2] < .9


def test_rising_envelope_does_not_add_braking_below_target():
    controller = agent_type()()
    controller.act(camera(curvature=.02, speed=70))
    action = controller.act(camera(speed=70))
    assert action[1] > 0
    assert action[2] == 0


def test_reset_clears_previous_deceleration_and_invalid_pixels_are_finite():
    controller = agent_type()()
    controller.act(camera(speed=90))
    controller.act(camera(curvature=.015, speed=90))
    action = controller.act(np.full((4, 84, 84), np.nan, dtype=np.float32))
    assert np.isfinite(action).all()
    assert np.all(action >= [-1, 0, 0]) and np.all(action <= 1)
    controller.reset()
    np.testing.assert_array_equal(controller.act(camera()), agent_type()().act(camera()))
