"""Improve supported bend references while preserving tested hazard/recovery."""
import importlib.util
from pathlib import Path

import numpy as np

from agents.apex_2026.fast_confidence_agent import Agent as Reference
from agents.apex_2026.tests.test_fast_envelope import camera


def agent_type():
    root = Path(__file__).resolve().parents[1]
    source = root / "fast_clear_ridge_agent.py"
    if not source.exists():
        source = root / "fast_confidence_agent.py"
    spec = importlib.util.spec_from_file_location("clear_ridge_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def fixture(name):
    frame = np.load(Path(__file__).parent / "fixtures" / name)["frame"]
    return np.repeat(frame[None], 4, axis=0)


def test_supported_nearly_horizontal_bend_selects_parametric_reference():
    controller = agent_type()()
    observation = fixture("ridge-track4-step65.npz")
    action = controller.act(observation)
    assert controller.mode == "ridge"
    assert np.isfinite(action).all() and action[0] < 0


def test_visible_obstacle_keeps_exact_reference_decision():
    observation = camera(speed=60, obstacle=(44, 27))
    np.testing.assert_array_equal(agent_type()().act(observation), Reference().act(observation))


def test_unsupported_ego_keeps_exact_reference_recovery():
    observation = fixture("confidence-track4-step70.npz")
    controller, baseline = agent_type()(), Reference()
    controller.last_center = baseline.last_center = 30.2475
    action = controller.act(observation)
    np.testing.assert_array_equal(action, baseline.act(observation))
    assert action[1] == 0 and action[2] >= .34


def test_new_mode_preserves_pixels_reset_and_finite_action_contract():
    controller = agent_type()()
    observation = camera(curvature=.003, speed=60)
    original = observation.copy()
    action = controller.act(observation)
    assert action.dtype == np.float32 and action.shape == (3,)
    np.testing.assert_array_equal(observation, original)
    invalid = controller.act(np.full((4, 84, 84), np.nan, dtype=np.float32))
    assert np.isfinite(invalid).all()
    assert np.all(invalid >= [-1, 0, 0]) and np.all(invalid <= 1)
    controller.reset()
    np.testing.assert_array_equal(controller.act(observation), agent_type()().act(observation))
