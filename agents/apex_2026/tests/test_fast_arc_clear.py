"""Keep hazardous turn plans while retaining a supported clear near turn."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from agents.apex_2026.fast_arc_guard_agent import Agent as ArcReference
from agents.apex_2026.fast_early_circle_agent import Agent as HazardReference
from agents.apex_2026.tests.test_fast_arc_guard import emitted_envelope_depth, saved_controller
from agents.apex_2026.tests.test_fast_envelope import camera


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "speed-20261005"


def candidate_type():
    source = ROOT / "fast_arc_clear_agent.py"
    if not source.exists():
        return ArcReference
    spec = importlib.util.spec_from_file_location("fast_arc_clear_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


@pytest.mark.parametrize("track", [3, 4])
def test_saved_post_pass_camera_keeps_reference_turn_direction_and_action(track):
    fixture = np.load(RESULTS / "arc-clear-v2-hazard-fixtures.npz")
    observation = fixture[f"track{track}_observation"]
    state = json.loads(str(fixture[f"track{track}_prestate"]))
    candidate, reference = candidate_type()(), HazardReference()
    for controller in (candidate, reference):
        for name, value in state.items():
            setattr(controller, name, value)
    np.testing.assert_array_equal(candidate.act(observation), reference.act(observation))


@pytest.mark.parametrize("step", [77, 78])
def test_clear_bend_still_keeps_supported_near_vehicle_envelope(step):
    controller, observation = saved_controller(candidate_type(), step)
    action = controller.act(observation)
    assert emitted_envelope_depth(controller, observation, action[0]) >= 1.9
    assert action[0] < -.3 and action[1] == 0
    curve = abs(np.tan(float(action[0])) / controller.WHEELBASE)
    assert controller.last_target <= np.sqrt(controller.lateral_accel / curve) + 1e-6


def test_clear_straight_invalid_input_and_reset_preserve_finite_actions():
    candidate, reference = candidate_type()(), ArcReference()
    for observation in (camera(speed=50), np.full((4, 84, 84), np.nan, np.float32)):
        np.testing.assert_array_equal(candidate.act(observation), reference.act(observation))
    candidate.reset()
    reference.reset()
    np.testing.assert_array_equal(candidate.act(camera()), reference.act(camera()))
