"""Batching changes computation while preserving the frozen control policy."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from agents.apex_2026.research.speed_20261005.physics_four_tire_model import predict_step
from agents.apex_2026.research.speed_20261005.predictive_batch_model import predict_batch
from agents.apex_2026.research.speed_20261005.predictive_control import FixedControlPlanner
from agents.apex_2026.tests.test_predictive_control import legal_state, straight_geometry


SOURCE = Path(__file__).resolve().parents[1]/'research'/'speed_20261005'/'predictive_batch_control.py'


def planner_type():
    assert SOURCE.exists(), 'independent batch control helper is not implemented'
    spec = importlib.util.spec_from_file_location('predictive_batch_control_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FixedBatchControlPlanner


def compare(left, right):
    for key in ('feasible', 'reason', 'candidate_count', 'feasible_count', 'scenario_count', 'model_steps'):
        assert left[key] == right[key], key
    if right['action'] is None:
        assert left['action'] is None
    else:
        np.testing.assert_array_equal(left['action'], right['action'])
        for key in ('score', 'worst_progress_m', 'predicted_terminal_speed_mps',
                    'backup_start_s', 'backup_end_progress_m', 'backup_duration_s'):
            assert left[key] == pytest.approx(right[key], abs=1e-10, rel=1e-12), key
    assert left['rejected_max_violation_m'] == pytest.approx(right['rejected_max_violation_m'], abs=1e-10)


def test_candidates_and_float32_pedals_preserve_frozen_order():
    batch = planner_type()(predict_batch, straight_geometry())
    scalar = FixedControlPlanner(predict_step, straight_geometry())
    for previous in ([0., 0., 0.], [.37, .6, 0.]):
        np.testing.assert_array_equal(batch.candidate_actions(previous), scalar.candidate_actions(previous))


@pytest.mark.parametrize('speed,count', ((0., 1), (100., 3)))
def test_synthetic_launch_and_fast_clear_control_match_scalar(speed, count):
    states = [legal_state(speed) for _ in range(count)]
    if count == 3:
        states[1]['yaw'], states[2]['yaw'] = -.15, .15
        states[1]['joint'][:2], states[2]['joint'][:2] = -.006, .006
    previous = np.zeros(3, np.float32)
    geometry = straight_geometry()
    before = [state['velocity'].copy() for state in states]
    result = planner_type()(predict_batch, geometry).plan(states, previous)
    reference = FixedControlPlanner(predict_step, geometry).plan(states, previous)
    compare(result, reference)
    assert result['feasible']
    for state, value in zip(states, before):
        np.testing.assert_array_equal(state['velocity'], value)


def test_unavoidable_hazard_returns_same_none_and_logical_work():
    states = [legal_state(70.) for _ in range(3)]
    geometry = straight_geometry(half_width=4.5, circles=[[0., 15.]])
    result = planner_type()(predict_batch, geometry).plan(states, [0., 0., 0.])
    reference = FixedControlPlanner(predict_step, geometry).plan(states, [0., 0., 0.])
    compare(result, reference)
    assert result['action'] is None


def test_stopped_backup_rows_freeze_before_tick_and_keep_unpadded_geometry():
    states = [legal_state(0.), legal_state(70.)]
    planner = planner_type()(predict_batch, straight_geometry())
    actions = np.asarray([[0., 0., 0.], [0., 0., 1.]], np.float32)
    rows = planner._rollout_matrix(states, actions)
    # The first action's cold scenario is already stopped at its first block,
    # while its rolling scenario needs a nonzero tail. No padded future poses
    # may be supplied as a real trajectory to geometry for the stopped row.
    backup = planner._backup_matrix(rows, actions, np.ones(4, bool))
    assert backup['duration_s'][0] == 0. and backup['stopped'][0]
    assert backup['duration_s'][1] > 0.
    assert backup['pose_counts'][0] == 5
    assert backup['pose_counts'][1] == 5+round(backup['duration_s'][1]/.02)


def test_batch_model_exception_and_invalid_state_keep_none_fallback():
    def broken_model(state, commands):
        raise RuntimeError('batch model failed')

    planner = planner_type()(broken_model, straight_geometry())
    result = planner.plan([legal_state(70.)], [0., 0., 0.])
    assert result['action'] is None and result['reason'] == 'planning_error'
    invalid = legal_state(70.)
    invalid['wheel_velocity_override'] = np.zeros((4, 2))
    assert planner.plan([invalid], [0., 0., 0.])['action'] is None


@pytest.mark.parametrize('step', (20, 40))
def test_actual_causal_camera_states_and_strict_hull_geometry_match_scalar(step):
    base = Path(__file__).resolve().parents[1]
    fixture = np.load(base/'results'/'speed-20261005'/'predictive-batch-control-v1-cameras.npz')
    spec = importlib.util.spec_from_file_location('batch_camera_geometry_reference', base/'fast_predictive_agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    states = json.loads(str(fixture[f'states_step{step}']))
    for state in states:
        for key in FixedControlPlanner.ARRAY_SHAPES:
            state[key] = np.asarray(state[key], float)
    geometry = module._HullCameraGeometry(fixture[f'path_step{step}'], fixture[f'field_step{step}'],
                                        fixture[f'circles_step{step}'], fixture[f'grass_step{step}'])
    previous = fixture[f'previous_step{step}']
    result = planner_type()(predict_batch, geometry).plan(states, previous)
    reference = FixedControlPlanner(predict_step, geometry).plan(states, previous)
    compare(result, reference)
    assert result['feasible'] and result['feasible_count'] > 0
