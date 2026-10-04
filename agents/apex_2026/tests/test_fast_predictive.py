"""Causal standalone prediction preserves real actions and visible support."""
import ast
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from agents.apex_2026.fast_rear_clear_agent import Agent as RearClear
from agents.apex_2026.research.speed_20261005.physics_four_tire_model import predict_step as reference_step
from agents.apex_2026.research.speed_20261005.physics_camera_observer import (
    CameraObserver as ReferenceObserver, calibration_from_receipts,
)
from agents.apex_2026.tests.test_fast_envelope import camera
from agents.apex_2026.tests.test_predictive_control import legal_state

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE/'fast_predictive_agent.py'
BUILDER = BASE/'research/speed_20261005/physics_build_predictive.py'


def module():
    path = SOURCE if SOURCE.exists() else BASE/'fast_rear_clear_agent.py'
    spec = importlib.util.spec_from_file_location('fast_predictive_integration_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_public_physics_export_matches_validated_model_without_private_imports():
    candidate = module()
    assert hasattr(candidate, 'predict_step'), 'pure validated vehicle predictor is not exported yet'
    left, right = legal_state(70.), legal_state(70.)
    for command in ([.06, .3, 0.], [-.03, .6, 0.], [-.03, 0., 1.]):
        for _ in range(4):
            left, diagnostics = candidate.predict_step(left, command)
            right, expected = reference_step(right, command)
            for key in right:
                if isinstance(right[key], np.ndarray):
                    np.testing.assert_array_equal(left[key], right[key])
                else:
                    assert left[key] == right[key]
            assert diagnostics == expected
    tree = ast.parse(SOURCE.read_text())
    imported = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)}
    assert imported <= {'numpy', 'math', 'copy', 'time'}
    assert not any(isinstance(n, ast.ImportFrom) for n in ast.walk(tree))


def test_reproducible_builder_preserves_exact_rearclear_prefix():
    assert BUILDER.exists(), 'offline source builder has not been implemented'
    assert SOURCE.exists()
    spec = importlib.util.spec_from_file_location('predictive_builder_test', BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert builder.build_text(BASE.parents[1]) == SOURCE.read_text()
    assert SOURCE.read_bytes().startswith((BASE/'fast_rear_clear_agent.py').read_bytes())


def test_exported_observer_matches_causal_camera_history():
    candidate = module()
    assert hasattr(candidate, 'CameraObserver'), 'camera-only dynamics observer is not exported yet'
    exported = candidate.CameraObserver(candidate._fixed_camera_calibration())
    reference = ReferenceObserver(calibration_from_receipts())
    for speed, action in ((0., [0., 1., 0.]), (5., [.06, .3, 0.]), (10., [-.03, 0., .25])):
        frame = camera(speed=speed)[-1]
        exported.observe(frame)
        reference.observe(frame)
        left, _ = exported.advance(action)
        right, _ = reference.advance(action)
        for key in right:
            if isinstance(right[key], np.ndarray):
                np.testing.assert_array_equal(left[key], right[key])
            else:
                assert left[key] == right[key]


def test_parent_called_once_and_observer_advances_only_the_emitted_plan(monkeypatch):
    candidate = module()
    assert hasattr(candidate, 'FixedControlPlanner'), 'joint predictive planner is missing'
    chosen = np.array([.03, .6, 0.], np.float32)
    calls = []
    original = candidate._BaseRearClearAgent.act

    def parent_once(self, observation):
        calls.append('parent')
        return original(self, observation)

    monkeypatch.setattr(candidate._BaseRearClearAgent, 'act', parent_once)
    monkeypatch.setattr(candidate.FixedControlPlanner, 'plan', lambda self, states, previous:
                        {'feasible': True, 'action': chosen.copy(), 'reason': 'feasible', 'model_steps': 0})
    agent = candidate.Agent()
    advances = []
    advance = agent._observer.advance

    def emitted_once(action):
        advances.append(np.asarray(action).copy())
        return advance(action)

    monkeypatch.setattr(agent._observer, 'advance', emitted_once)
    observation = camera(speed=40.)
    before = observation.copy()
    result = agent.act(observation)
    np.testing.assert_array_equal(result, chosen)
    assert calls == ['parent'] and len(advances) == 1
    np.testing.assert_array_equal(advances[0], result)
    assert agent._observer.actions_seen == 1
    assert agent.last_steer == float(result[0])
    np.testing.assert_array_equal(observation, before)


def test_conservative_field_preserves_grass_islands_but_fills_known_car():
    candidate = module()
    agent = candidate.Agent()
    assert hasattr(agent, '_predictive_field'), 'a separate conservative camera field is missing'
    frame = np.full((84, 84), .4, np.float32)
    frame[73:] = 0.
    frame[60:66, 41:44] = .1  # Known rendered car footprint.
    frame[40:45, 42:47] = .63  # An enclosed grass island, not an occlusion.
    field, grass = agent._predictive_field(frame, [])
    assert field[63, 42] > 1.9
    assert field[42, 44] == 0. and grass[42, 44]
    assert field[0, 42] == 0. and field[72, 42] == 0.


def test_hull_rotation_progress_uses_hull_chord_not_stationary_com():
    candidate = module()
    assert hasattr(candidate, '_HullCameraGeometry'), 'CM-to-hull geometry adapter is missing'
    geometry = candidate._HullCameraGeometry(np.array([[-3., 0.], [-5., 0.]]), np.full((73, 84), 10.))
    center = np.array([0., -.0804591735470409])
    delta = center-candidate.rotate(center, .16)
    result = geometry.evaluate(np.array([[0., 0.], delta]), [0., .16], np.zeros((2, 2)), .02)
    assert result['progress_m'][1] == pytest.approx(-delta[0])


def test_representative_uncertainty_states_are_bounded_and_detached():
    candidate = module()
    agent = candidate.Agent()
    assert hasattr(agent, '_predictive_scenarios'), 'bounded state scenarios are missing'
    state = legal_state(70.)
    states = agent._predictive_scenarios(state)
    assert len(states) == 3
    assert states[1]['velocity'][0] < 0. < states[2]['velocity'][0]
    assert all(abs(s['yaw']) <= .15 for s in states)
    assert all(np.max(abs(s['joint'])) <= .006 for s in states)
    states[0]['velocity'][0] = 100.
    assert state['velocity'][0] == 0. and states[1]['velocity'][0] != 100.


def test_low_speed_and_reset_preserve_real_fallback_action():
    candidate = module()
    agent = candidate.Agent()
    assert hasattr(agent, '_observer'), 'predictive state history is missing'
    observation = camera(curvature=.01, speed=10.)
    expected = RearClear().act(observation)
    np.testing.assert_array_equal(agent.act(observation), expected)
    assert agent._observer.actions_seen == 1
    agent.reset()
    assert agent._observer.actions_seen == 0 and np.max(abs(agent._observer.state['omega'])) == 0.
    np.testing.assert_array_equal(agent.act(observation), expected)


def test_planning_deadline_falls_back_and_still_records_one_real_action():
    candidate = module()
    assert hasattr(candidate, '_PlanningBudgetExceeded'), 'bounded planning callback is missing'
    observation = camera(speed=40.)
    agent = candidate.Agent(planning_budget_s=0.)
    np.testing.assert_array_equal(agent.act(observation), RearClear().act(observation))
    assert agent.predictive_status == 'fallback_timeout'
    assert agent._observer.actions_seen == 1


def test_invalid_observation_returns_finite_action_and_retains_control_history():
    candidate = module()
    agent = candidate.Agent()
    assert hasattr(agent, '_observer'), 'predictive control history is missing'
    for observation in (camera(), np.full((4, 84, 84), np.nan, np.float32)):
        result = agent.act(observation)
        assert result.shape == (3,) and result.dtype == np.float32 and np.isfinite(result).all()
        assert np.all(result >= [-1., 0., 0.]) and np.all(result <= 1.)
    assert agent._observer.actions_seen == 2
