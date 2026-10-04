"""Parametric camera hazard distance must retain a stoppable known prefix."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from agents.apex_2026.fast_rear_clear_agent import Agent as Reference
from agents.apex_2026.tests.test_fast_envelope import camera


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results'/'speed-20261005'


def candidate_type():
    source = ROOT/'fast_arc_hazard_agent.py'
    if not source.exists():
        return Reference
    spec = importlib.util.spec_from_file_location('fast_arc_hazard_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def saved_controller(kind, track, step):
    fixture = np.load(RESULTS/'arc-hazard-v1-cameras.npz')
    agent = kind()
    for key, value in json.loads(str(fixture[f'track{track}_prestate_step{step}'])).items():
        setattr(agent, key, value)
    return agent, fixture[f'track{track}_observation_step{step}'].copy()


def segment_gap(path, point):
    delta = np.diff(path, axis=0)
    phase = np.clip(((point-path[:-1])*delta).sum(axis=1)/(delta*delta).sum(axis=1), 0., 1.)
    nearest = path[:-1]+phase[:, None]*delta
    return np.min(np.linalg.norm(nearest-point, axis=1))


@pytest.mark.parametrize('step', (132, 133))
def test_actual_far_along_bend_circle_uses_stoppable_trimmed_ridge(step):
    controller, observation = saved_controller(candidate_type(), 4, step)
    reference, _ = saved_controller(Reference, 4, step)
    reference.act(observation)
    action = controller.act(observation)
    assert reference.mode == 'confidence'
    assert controller.mode == 'ridge'
    assert controller.hazard_deferred and not controller.pass_side
    path = controller.hazard_prefix
    assert len(path) >= 5
    length = controller._arc(path)[-1]
    speed = controller.last_speed
    assert length >= speed**2/(2*controller.braking_accel)+.08*speed+2.6
    for y, x in controller.hazard_original_circles:
        assert segment_gap(path, np.asarray([x, y])) >= 3.7
    stop_speed = np.sqrt(2*controller.braking_accel*max(0., length-2.6-.08*speed))
    assert controller.last_target <= stop_speed
    assert np.isfinite(action).all()
    assert np.all(action >= [-1., 0., 0.]) and np.all(action <= 1.)
    # Deferral is local to this action. The unmodified detector still sees
    # the real future circle after act; another consumer cannot inherit [] .
    frame = controller._frame(observation)
    road = reference._road(frame)
    assert controller._circles(frame, road) == reference._circles(frame, road)
    assert controller._circles(frame, road)


@pytest.mark.parametrize('track,step', ((1, 59), (1, 60), (4, 128), (4, 134), (4, 135), (4, 139)))
def test_actual_immediate_or_active_pass_camera_preserves_exact_parent(track, step):
    controller, observation = saved_controller(candidate_type(), track, step)
    reference, _ = saved_controller(Reference, track, step)
    np.testing.assert_array_equal(controller.act(observation), reference.act(observation))
    assert not getattr(controller, 'hazard_deferred', False)
    assert controller.mode == reference.mode
    assert controller.pass_side == reference.pass_side


def test_actual_near_circle_cannot_defer_even_without_prior_pass_memory():
    controller, observation = saved_controller(candidate_type(), 4, 139)
    reference, _ = saved_controller(Reference, 4, 139)
    for agent in (controller, reference):
        agent.pass_side, agent.pass_missing, agent.pass_y = 0, 0, None
    np.testing.assert_array_equal(controller.act(observation), reference.act(observation))
    assert not getattr(controller, 'hazard_deferred', False)
    assert controller.mode == 'confidence'


def test_unknown_circle_beyond_supported_endpoint_cannot_be_deferred():
    controller = candidate_type()()
    path = np.column_stack((np.zeros(31), np.arange(3., 65., 2.)))
    assert controller._hazard_trim(path, [(68., 0.)], 40.) is None


def test_invalid_reset_normal_camera_and_input_contract_preserve_parent():
    controller, observation = saved_controller(candidate_type(), 4, 132)
    before = observation.copy()
    controller.act(observation)
    np.testing.assert_array_equal(observation, before)
    invalid = np.full((4, 84, 84), np.nan, np.float32)
    reference = Reference()
    controller.reset()
    np.testing.assert_array_equal(controller.act(invalid), reference.act(invalid))
    controller.reset()
    reference.reset()
    assert not getattr(controller, 'hazard_deferred', False)
    for observation in (camera(), camera(speed=40, curvature=.01)):
        before = observation.copy()
        np.testing.assert_array_equal(controller.act(observation), reference.act(observation))
        np.testing.assert_array_equal(observation, before)
