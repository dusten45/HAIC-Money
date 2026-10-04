"""Useful propulsion and geometric passing for the new camera candidate."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def agent_type():
    source = ROOT / "fast_envelope_agent.py"
    if not source.exists():
        source = ROOT / "agent.py"
    spec = importlib.util.spec_from_file_location("fast_envelope_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def camera(curvature=0.0, speed=0.0, center=42.0, obstacle=None):
    frame = np.full((84, 84), .63, dtype=np.float32)
    for row in range(73):
        ahead = (63.0 - row) / 1.701
        middle = center + 1.3608 * .5 * curvature * max(0.0, ahead) ** 2
        lo, hi = int(round(middle - 9)), int(round(middle + 9))
        frame[row, max(0, lo):min(84, hi + 1)] = .40
    if obstacle is not None:
        col, row = obstacle
        frame[row - 2:row + 3, col - 2:col + 3] = .686
    frame[73:] = 0
    top, bottom = 81.9 - .042 * speed, 81.9
    for row in range(73, 84):
        for col in range(10, 14):
            width = max(0., min(col + 1., 12.6) - max(float(col), 10.5))
            height = max(0., min(row + 1., bottom) - max(float(row), top))
            frame[row, col] = width * height
    return np.repeat(frame[None], 4, axis=0)


def test_mild_clear_bend_accelerates_below_available_lateral_force():
    controller = agent_type()()
    action = controller.act(camera(curvature=.015, speed=45))
    assert action[0] > .015
    assert action[1] > .32
    assert action[2] == 0


def test_distant_side_obstacle_preserves_available_approach_speed():
    controller = agent_type()()
    controller.act(camera(speed=70, obstacle=(47, 20)))
    assert controller.last_target > 90


def test_close_obstacle_selects_opposite_side():
    action = agent_type()().act(camera(speed=40, obstacle=(45, 53)))
    assert action[0] < -.03


def test_large_lateral_correction_at_high_speed_brakes():
    action = agent_type()().act(camera(center=58, speed=95))
    assert action[1] == 0
    assert action[2] > 0


def test_finite_recovery_reset_and_input_preservation():
    controller = agent_type()()
    observation = camera(curvature=.01, speed=40)
    original = observation.copy()
    controller.act(observation)
    np.testing.assert_array_equal(observation, original)
    for _ in range(7):
        action = controller.act(np.full((4, 84, 84), np.nan, dtype=np.float32))
        assert action.dtype == np.float32 and action.shape == (3,)
        assert np.isfinite(action).all()
        assert np.all(action >= [-1, 0, 0]) and np.all(action <= 1)
    controller.reset()
    np.testing.assert_array_equal(controller.act(camera()), agent_type()().act(camera()))
