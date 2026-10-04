"""Release post-pass phase only after confirmed camera rear clearance."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from agents.apex_2026.fast_arc_clear_agent import Agent as Reference
from agents.apex_2026.tests.test_fast_envelope import camera


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "speed-20261005"


def candidate_type():
    source = ROOT / "fast_rear_clear_agent.py"
    if not source.exists():
        return Reference
    spec = importlib.util.spec_from_file_location("fast_rear_clear_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def saved_controller(kind, step):
    fixture = np.load(RESULTS / "postpass-track4-cameras.npz")
    agent = kind()
    for key, value in json.loads(str(fixture[f"prestate_step{step}"])).items():
        setattr(agent, key, value)
    return agent, fixture[f"observation_step{step}"]


def test_actual_rear_clear_camera_releases_supported_ridge_phase():
    controller, observation = saved_controller(candidate_type(), 101)
    reference, _ = saved_controller(Reference, 101)
    original = reference.act(observation)
    action = controller.act(observation)
    assert reference.mode == "confidence" and reference.circle_cooldown == 11
    assert controller.pass_side == 0 and controller.pass_missing <= 4
    assert controller.mode == "ridge" and controller.circle_cooldown == 0
    assert action[0] < original[0]
    assert np.isfinite(action).all()


def test_missing_object_expiry_retains_parent_phase_and_action():
    controller, observation = saved_controller(candidate_type(), 101)
    reference, _ = saved_controller(Reference, 101)
    controller.pass_missing = reference.pass_missing = 4
    np.testing.assert_array_equal(controller.act(observation), reference.act(observation))
    assert controller.pass_missing == 5
    assert controller.mode == "confidence" and controller.circle_cooldown == 11


def test_camera_before_rear_clearance_retains_parent_phase_and_action():
    controller, observation = saved_controller(candidate_type(), 100)
    reference, _ = saved_controller(Reference, 100)
    np.testing.assert_array_equal(controller.act(observation), reference.act(observation))
    assert controller.pass_side and controller.circle_cooldown == 12


def route_inputs():
    ahead = np.linspace(-4., 30., 36)
    return ahead, np.zeros_like(ahead)


def active_pass(kind):
    agent = kind()
    agent.pass_side, agent.pass_x, agent.pass_y = -1, 10., -1.
    agent.circle_cooldown = 12
    return agent


def test_camera_transport_includes_yaw_rotation_before_rear_release():
    agent = active_pass(candidate_type())
    agent._route(*route_inputs(), [], 0., -8.)
    assert agent.pass_side == 0 and agent.pass_missing == 1
    assert agent.circle_cooldown == 0


def test_new_current_circle_retains_active_hazard_phase():
    agent = active_pass(candidate_type())
    agent._route(*route_inputs(), [(20., 0.)], 0., -8.)
    assert agent.pass_side != 0 and agent.circle_cooldown == 12


def test_invalid_camera_does_not_release_pass_and_reset_preserves_contract():
    controller, _ = saved_controller(candidate_type(), 101)
    reference, _ = saved_controller(Reference, 101)
    invalid = np.full((4, 84, 84), np.nan, np.float32)
    np.testing.assert_array_equal(controller.act(invalid), reference.act(invalid))
    assert controller.pass_side and controller.circle_cooldown == 12
    controller.reset()
    reference.reset()
    observation = camera()
    before = observation.copy()
    np.testing.assert_array_equal(controller.act(observation), reference.act(observation))
    np.testing.assert_array_equal(observation, before)
