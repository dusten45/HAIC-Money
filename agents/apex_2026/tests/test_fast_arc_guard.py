"""Actual-camera regression for the base controller's near turning support."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from agents.apex_2026.fast_early_circle_agent import Agent as Reference
from agents.apex_2026.tests.test_fast_envelope import camera


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "speed-20261005"


def candidate_type():
    source = ROOT / "fast_arc_guard_agent.py"
    if not source.exists():
        return Reference
    spec = importlib.util.spec_from_file_location("fast_arc_guard_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def saved_controller(controller_type, step):
    controller = controller_type()
    rows = json.loads((RESULTS / "slow-phase-track3-v2.json").read_text())["rows"]
    for name, value in rows[step - 1]["policy"].items():
        setattr(controller, name, value)
    observation = np.load(RESULTS / "slow-phase-track3-cameras.npz")[f"observation_step{step}"]
    return controller, observation


def emitted_envelope_depth(controller, observation, steer):
    k = np.tan(float(steer)) / controller.WHEELBASE
    s = np.linspace(0., 8., 33)
    phase = k * s
    if abs(k) < 1e-8:
        center = np.column_stack((np.zeros_like(s), s))
    else:
        center = np.column_stack(((1. - np.cos(phase)) / k, np.sin(phase) / k))
    path = np.concatenate([center + np.column_stack((x * np.cos(phase) + y * np.sin(phase),
                                                    -x * np.sin(phase) + y * np.cos(phase)))
                           for x, y in ((-1.6, -2.4), (-1.6, 2.6), (1.6, -2.4), (1.6, 2.6))])
    field = controller._distance_field(observation[-1])
    return float(np.min(controller._sample_distance(field, path)))


@pytest.mark.parametrize("step", [77, 78])
def test_actual_bend_camera_keeps_emitted_near_envelope_on_supported_road(step):
    controller, observation = saved_controller(candidate_type(), step)
    reference, _ = saved_controller(Reference, step)
    original = reference.act(observation)
    action = controller.act(observation)
    assert emitted_envelope_depth(reference, observation, original[0]) < 1.9
    assert emitted_envelope_depth(controller, observation, action[0]) >= 1.9
    assert action[1] == 0 and action[2] >= original[2]
    actual_curve = abs(np.tan(float(action[0])) / controller.WHEELBASE)
    assert controller.last_target <= np.sqrt(controller.lateral_accel / actual_curve) + 1e-6
    previous = json.loads((RESULTS / "slow-phase-track3-v2.json").read_text())["rows"][step - 1]["policy"]["last_steer"]
    assert abs(float(action[0]) - previous) <= .240001


def test_supported_clear_cameras_preserve_existing_actions():
    controller, reference = candidate_type()(), Reference()
    for observation in (camera(speed=40), camera(curvature=.004, speed=45), camera(speed=50)):
        np.testing.assert_array_equal(controller.act(observation), reference.act(observation))


def test_no_supported_short_turn_preserves_existing_late_edge_action():
    controller, observation = saved_controller(candidate_type(), 81)
    reference, _ = saved_controller(Reference, 81)
    np.testing.assert_array_equal(controller.act(observation), reference.act(observation))


def test_invalid_pixels_and_reset_keep_finite_existing_recovery():
    controller, reference = candidate_type()(), Reference()
    invalid = np.full((4, 84, 84), np.nan, np.float32)
    for _ in range(3):
        np.testing.assert_array_equal(controller.act(invalid), reference.act(invalid))
    controller.reset()
    reference.reset()
    np.testing.assert_array_equal(controller.act(camera()), reference.act(camera()))
