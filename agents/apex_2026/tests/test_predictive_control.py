"""A supplied legal state/model jointly plans the executed camera action."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from agents.apex_2026.research.speed_20261005.camera_mpc_reference import CameraGeometry
from agents.apex_2026.research.speed_20261005.physics_four_tire_model import predict_step


SOURCE = Path(__file__).resolve().parents[1]/'research'/'speed_20261005'/'predictive_control.py'


def planner_type():
    assert SOURCE.exists(), 'NumPy-only supplied-state joint planner is not implemented yet'
    spec = importlib.util.spec_from_file_location('predictive_control_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FixedControlPlanner


def legal_state(speed):
    # Public mechanical constants; no simulator initialization or labels.
    center = np.array([0., -.0804591735470409])
    return {'angle': 0., 'velocity': np.array([0., float(speed)]), 'yaw': 0.,
            'mass': 7.3019199669361115, 'inertia': 19.170645977936854,
            'local_center': center, 'wheel_offsets': np.array(
                [[-1.1, 1.6], [1.1, 1.6], [-1.1, -1.64], [1.1, -1.64]])-center,
            'joint': np.zeros(4), 'omega': np.full(4, speed/.54),
            'gas': np.zeros(4), 'wheel_velocity_override': None}


def straight_geometry(half_width=6.6666666667, end=35., circles=()):
    x = (np.arange(84)-42.)/1.3608
    field = np.tile(np.maximum(0., half_width-abs(x)), (73, 1))
    path = np.column_stack((np.zeros(17), np.linspace(3., end, 17)))
    return CameraGeometry(path, field, circles)


def test_control_candidates_obey_slew_motor_command_and_separate_pedals():
    planner = planner_type()(predict_step, straight_geometry())
    actions = planner.candidate_actions(np.array([.37, .6, 0.]))
    assert 1 <= len(actions) <= 72
    assert actions.shape[1] == 3 and actions.dtype == np.float32
    assert np.isfinite(actions).all()
    assert np.all(abs(actions[:, 0]) <= .400001)
    assert np.all(abs(actions[:, 0]-.37) <= .240001)
    assert np.all(actions[:, 1]*actions[:, 2] == 0.)


def test_legal_right_steer_predicts_right_motion_with_actual_joint_lag():
    planner = planner_type()(predict_step, straight_geometry())
    state = legal_state(70.)
    before = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in state.items()}
    rollout = planner.rollout(state, [.06, .16, 0.])
    assert rollout['positions'][0, 0] == rollout['positions'][0, 1] == 0.
    assert rollout['positions'][-1, 0] > 0. and rollout['angles'][-1] < 0.
    assert rollout['states'][1]['joint'][0] == pytest.approx(-.06)
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(state[key], value)
        else:
            assert state[key] == value


def test_full_brake_is_float32_exact_and_locks_both_plan_and_backup():
    commands = []

    def recording_model(state, command):
        commands.append(command)
        return predict_step(state, command)

    planner = planner_type()(recording_model, straight_geometry())
    actions = planner.candidate_actions([0., 0., 0.])
    full_brake = actions[int(np.argmax(actions[:, 2]))]
    assert float(full_brake[2]) == 1., 'float32(.9) is below the physical Python >=.9 lock threshold'
    assert full_brake[1] == 0.
    rollout = planner.rollout(legal_state(70.), full_brake)
    # A locked wheel gets only the subsequent tire-force omega response,
    # rather than retaining the initial rolling omega minus a partial brake.
    assert np.max(abs(rollout['states'][1]['omega'])) <= 3.
    planner.plan([legal_state(70.)], [0., 0., 0.])
    assert any(command[2] == 1. for command in commands)
    assert all(not .89 < command[2] < .99 for command in commands)


def test_clear_launch_chooses_real_propulsion_and_finite_legal_action():
    planner = planner_type()(predict_step, straight_geometry())
    result = planner.plan([legal_state(0.)], [0., 0., 0.])
    assert result['feasible']
    action = result['action']
    assert action.shape == (3,) and action.dtype == np.float32 and np.isfinite(action).all()
    assert action[1] > 0. and action[2] == 0.
    assert result['worst_progress_m'] > 0.


def test_first_executed_block_backup_allows_100_mps_on_35_m_visible_straight():
    planner = planner_type()(predict_step, straight_geometry())
    result = planner.plan([legal_state(100.)], [0., 0., 0.])
    assert result['feasible']
    assert result['worst_progress_m'] > 30.
    assert result['predicted_terminal_speed_mps'] > 95.
    assert result['backup_start_s'] == .08
    assert result['backup_end_progress_m'] < 32.4


def test_unavoidable_near_circle_returns_none_for_existing_agent_fallback():
    planner = planner_type()(predict_step, straight_geometry(half_width=4.5, circles=[[0., 15.]]))
    result = planner.plan([legal_state(70.)], [0., 0., 0.])
    assert not result['feasible'] and result['action'] is None
    assert result['reason'] == 'no_feasible_candidate'


def test_uncertain_lateral_state_is_checked_without_nominal_only_acceptance():
    planner = planner_type()(predict_step, straight_geometry())
    nominal, adverse = legal_state(0.), legal_state(0.)
    adverse['local_center'] = np.array([0., -.0804591735470409])
    adverse['velocity'] = np.array([60., 0.])
    assert planner.plan([nominal], [0., 0., 0.])['feasible']
    result = planner.plan([nominal, adverse], [0., 0., 0.])
    assert not result['feasible'] and result['action'] is None


def test_privileged_override_invalid_state_and_predict_error_return_none():
    planner = planner_type()(predict_step, straight_geometry())
    invalid = legal_state(40.)
    invalid['wheel_velocity_override'] = np.zeros((4, 2))
    result = planner.plan([invalid], [0., 0., 0.])
    assert result['action'] is None and not result['feasible']
    bad = legal_state(40.)
    bad['velocity'][0] = np.nan
    assert planner.plan([bad], [0., 0., 0.])['action'] is None

    def broken_model(state, action):
        raise RuntimeError('supplied model failed')

    broken = planner_type()(broken_model, straight_geometry())
    result = broken.plan([legal_state(40.)], [0., 0., 0.])
    assert result['action'] is None and result['reason'] == 'planning_error'
