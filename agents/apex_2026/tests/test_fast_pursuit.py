"""Measured pursuit demand instead of duplicate quantized curve fits."""
import importlib.util
from pathlib import Path

import numpy as np

from agents.apex_2026.tests.test_fast_envelope import camera


def agent_type():
    root = Path(__file__).resolve().parents[1]
    source = root / "fast_pursuit_agent.py"
    if not source.exists():
        source = root / "fast_path_agent.py"
    spec = importlib.util.spec_from_file_location("fast_pursuit_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def test_visible_gentle_curve_does_not_receive_duplicate_fitted_speed_minimum():
    controller = agent_type()()
    action = controller.act(camera(curvature=.012, speed=70))
    assert controller.last_target > 95
    assert action[1] > .3
    assert action[2] == 0


def test_sharp_near_correction_keeps_pursuit_speed_limit():
    controller = agent_type()()
    action = controller.act(camera(center=60, speed=90))
    assert action[1] == 0
    assert action[2] > 0


def test_missing_camera_and_reset_remain_bounded():
    controller = agent_type()()
    controller.act(camera(curvature=.015, speed=40))
    for _ in range(4):
        action = controller.act(np.full((4, 84, 84), np.nan, dtype=np.float32))
        assert np.isfinite(action).all()
        assert np.all(action >= [-1, 0, 0]) and np.all(action <= 1)
    controller.reset()
    np.testing.assert_array_equal(controller.act(camera()), agent_type()().act(camera()))
