"""Vectorized legal-state asphalt prediction retains the scalar model order."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from agents.apex_2026.research.speed_20261005.physics_four_tire_model import predict_step, rotate


SOURCE = Path(__file__).resolve().parents[1]/'research'/'speed_20261005'/'predictive_batch_model.py'


def batch_module():
    assert SOURCE.exists(), 'vectorized supplied-state model is not implemented'
    spec = importlib.util.spec_from_file_location('predictive_batch_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legal_state(speed):
    center = np.array([0., -.0804591735470409])
    return {'angle': 0., 'velocity': np.array([0., float(speed)]), 'yaw': 0.,
            'mass': 7.3019199669361115, 'inertia': 19.170645977936854,
            'local_center': center, 'wheel_offsets': np.array(
                [[-1.1, 1.6], [1.1, 1.6], [-1.1, -1.64], [1.1, -1.64]])-center,
            'joint': np.zeros(4), 'omega': np.full(4, speed/.54),
            'gas': np.zeros(4), 'wheel_velocity_override': None}


def fixed_cases():
    states = [legal_state(speed) for speed in (70., 20., 100., 60., 100., 0., 100., 70.)]
    states[1]['gas'] = np.array([.2, .1, .4, .8])
    states[1]['omega'] = np.array([-100., -.1, 0., 100.])
    states[2]['angle'], states[2]['yaw'] = .3, 1.2
    states[2]['velocity'] = np.array([95., 33.])
    states[2]['wheel_offsets'][0] += [.1, -.02]
    states[3]['joint'] = np.array([.39, -.39, .6, -.6])
    states[3]['omega'] = np.array([100., -30., 0., 15.])
    states[3]['gas'][2:] = [.2, .6]
    states[6]['angle'], states[6]['yaw'] = -.8, -3.5
    states[6]['velocity'] = np.array([101., -4.])
    states[7]['angle'], states[7]['yaw'] = -.2, -.4
    states[7]['velocity'] = np.array([-9., 69.])
    states[7]['joint'][:2] = [.03, -.03]
    commands = np.array([[0., 0., 0.], [.7, .2, .25], [.4, .3, 0.],
                         [-.7, .9, float(np.float32(.9))], [0., 0., 1.],
                         [.06, 1., 0.], [0., .8, .001], [.0301, .7, -.1]], float)
    return states, commands


def compare_row(actual, expected):
    for name in ('angle', 'velocity', 'yaw', 'mass', 'inertia', 'local_center',
                 'wheel_offsets', 'joint', 'omega', 'gas'):
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-10, rtol=1e-12, err_msg=name)
    assert actual['wheel_velocity_override'] is expected['wheel_velocity_override'] is None


def test_mixed_branch_batch_matches_scalar_one_step_and_preserves_inputs():
    module = batch_module()
    states, commands = fixed_cases()
    packed = module.pack_states(states)
    before = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in packed.items()}
    result, diagnostics = module.predict_batch(packed, commands)
    for index, (state, command) in enumerate(zip(states, commands)):
        expected, row = predict_step(state, [float(value) for value in command])
        compare_row(module.unpack_state(result, index), expected)
        for name in ('wheel_forward_mps', 'wheel_lateral_mps', 'tire_force_local_N',
                     'unsaturated_tire_demand_ratio'):
            np.testing.assert_allclose(diagnostics[name][index], row[name], atol=1e-10, rtol=1e-12)
    for name, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(packed[name], value)
            assert not np.shares_memory(result[name], packed[name])
        else:
            assert packed[name] is value


def test_16_and_32_step_recurrences_match_scalar_not_just_initial_forces():
    module = batch_module()
    states, commands = fixed_cases()
    packed = module.pack_states(states)
    for tick in range(1, 33):
        packed, diagnostics = module.predict_batch(packed, commands)
        states = [predict_step(state, [float(value) for value in command])[0]
                  for state, command in zip(states, commands)]
        if tick in (16, 32):
            for index, state in enumerate(states):
                compare_row(module.unpack_state(packed, index), state)
        assert np.max(np.linalg.norm(diagnostics['tire_force_local_N'], axis=-1)) <= 400.+1e-10
        assert np.max(np.linalg.norm(packed['velocity'], axis=-1)) <= 100.+1e-10


def test_raw_gas_and_joint_update_order_and_body_wheel_caps():
    module = batch_module()
    states, commands = fixed_cases()
    result, diagnostic = module.predict_batch(module.pack_states(states), commands)
    np.testing.assert_allclose(result['gas'][1], [.2, .1, .2, .2])
    np.testing.assert_allclose(result['gas'][3, 2:], [.3, .7])
    np.testing.assert_allclose(result['joint'][3], [.33, -.4, .4, -.4])
    # Before clipping, body+angular wheel velocities may exceed100. The tire
    # inputs reconstruct the old-step capped velocity in their local frame.
    np.testing.assert_array_less(np.hypot(diagnostic['wheel_forward_mps'], diagnostic['wheel_lateral_mps']), 100.+1e-10)


def test_float32_point9_stays_partial_while_exact1_locks_before_tire_response():
    module = batch_module()
    states = [legal_state(70.), legal_state(70.)]
    commands = np.array([[0., 0., .9], [0., 0., 1.]], np.float32)
    result, diagnostics = module.predict_batch(module.pack_states(states), commands)
    assert result['omega'][0, 0] > 100.
    np.testing.assert_allclose(result['omega'][1], -.02*diagnostics['tire_force_local_N'][1, :, 1]*.54/1.6)
    assert np.max(abs(result['omega'][1])) <= 3.


def test_negative_brake_is_identical_to_zero_and_rolling_baseline_stays_fixed():
    module = batch_module()
    state = legal_state(70.)
    result, _ = module.predict_batch(module.pack_states([state, state]), [[0., 0., -.1], [0., 0., 0.]])
    compare_row(module.unpack_state(result, 0), module.unpack_state(result, 1))
    compare_row(module.unpack_state(result, 1), state)


def test_batch_row_permutation_and_duplication_do_not_couple_dynamics():
    module = batch_module()
    states, commands = fixed_cases()
    base, _ = module.predict_batch(module.pack_states(states), commands)
    order = [7, 0, 3, 3, 1, 5]
    reordered, _ = module.predict_batch(module.pack_states([states[index] for index in order]), commands[order])
    for index, original in enumerate(order):
        compare_row(module.unpack_state(reordered, index), module.unpack_state(base, original))


def test_global_coordinate_rotation_retains_local_tire_diagnostics_and_torque():
    module = batch_module()
    states, commands = fixed_cases()
    state = states[2]
    rotated = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in state.items()}
    phi = .7
    rotated['angle'] += phi
    rotated['velocity'] = rotate(rotated['velocity'], phi)
    result, diagnostics = module.predict_batch(module.pack_states([state, rotated]), np.repeat(commands[2][None], 2, axis=0))
    np.testing.assert_allclose(result['velocity'][1], rotate(result['velocity'][0], phi), atol=1e-10)
    assert result['angle'][1]-result['angle'][0] == pytest.approx(phi)
    for name in diagnostics:
        np.testing.assert_allclose(diagnostics[name][0], diagnostics[name][1], atol=1e-10)


def test_override_nonfinite_and_misaligned_batches_are_rejected():
    module = batch_module()
    state = legal_state(40.)
    state['wheel_velocity_override'] = np.zeros((4, 2))
    with pytest.raises(ValueError):
        module.pack_states([state])
    state = legal_state(40.)
    state['velocity'][0] = np.nan
    with pytest.raises(ValueError):
        module.pack_states([state])
    packed = module.pack_states([legal_state(40.)])
    with pytest.raises(ValueError):
        module.predict_batch(packed, np.zeros((2, 3)))
