"""PRIVILEGED four-tire forward-model validation, not a legal Agent.

Four fixed uniform-asphalt cases, exact initial wheel/body state and .08-.32s
prediction horizons. No maps, timed laps, candidate edits or parameter grid.
The analytical rigid-body approximation omits Box2D joint/limit solving; this
experiment measures those errors instead of treating the model as certified.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from agents.apex_2026.research.speed_20261005.physics_dynamics import (
    DT, OFFICIAL_FILES, ROOT, digest, make_car, tick,
)

CASES = (
    dict(name='moderate_acceleration', speed=70., warmup_steps=50,
         warmup=[.03, .3, 0.], command=[.03, .65, 0.]),
    dict(name='tight_acceleration_oversupply', speed=70., warmup_steps=60,
         warmup=[.06, .5, 0.], command=[.06, .5, 0.]),
    dict(name='settled_tight_turn_braking', speed=100., warmup_steps=30,
         warmup=[.06, .16, 0.], command=[.06, 0., .3]),
    dict(name='settled_tight_turn_reversal', speed=100., warmup_steps=30,
         warmup=[.06, .16, 0.], command=[-.06, .16, 0.]),
)
HORIZONS = (4, 8, 12, 16)
ACCEPTANCE = {'at08': {'speed_error_mps': 2., 'yaw_error_radps': .1,
                        'rear_longitudinal_slip_error_mps': .5,
                        'max_per_tire_force_vector_error_N': 50.},
              'at32': {'speed_error_mps': 3., 'yaw_error_radps': .2,
                        'rear_longitudinal_slip_error_mps': 1.,
                        'max_per_tire_force_vector_error_N': 100.}}


def rotate(vector, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c*vector[0]-s*vector[1], s*vector[0]+c*vector[1]])


def cross(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def body_state(car):
    hull = car.hull
    angle = float(hull.angle)
    masses = np.array([hull.mass, *[w.mass for w in car.wheels]], float)
    centers = np.array([list(hull.worldCenter), *[list(w.worldCenter) for w in car.wheels]], float)
    compound_center = np.sum(masses[:, None]*centers, axis=0)/masses.sum()
    local_center = rotate(compound_center-np.asarray(hull.position), -angle)
    wheel_local = np.array([list(hull.GetLocalPoint(w.position)) for w in car.wheels], float)
    hull_local_center = np.array(hull.localCenter, float)
    inertia = float(hull.inertia-hull.mass*np.sum(hull_local_center**2))
    inertia += hull.mass*float(np.sum((hull_local_center-local_center)**2))
    for i, wheel in enumerate(car.wheels):
        inertia += wheel.inertia+wheel.mass*float(np.sum((wheel_local[i]-local_center)**2))
    velocity = np.sum(masses[:, None]*np.array([list(hull.linearVelocity),
                         *[list(w.linearVelocity) for w in car.wheels]], float), axis=0)/masses.sum()
    return {'angle': angle, 'velocity': velocity, 'yaw': float(hull.angularVelocity),
            'mass': float(masses.sum()), 'inertia': inertia, 'local_center': local_center,
            'wheel_offsets': wheel_local-local_center,
            'joint': np.array([w.joint.angle for w in car.wheels], float),
            'omega': np.array([w.omega for w in car.wheels], float),
            'gas': np.array([w.gas for w in car.wheels], float),
            'wheel_velocity_override': np.array([list(w.linearVelocity) for w in car.wheels], float)}


def tire_inputs(state):
    front, side = [], []
    for i, offset in enumerate(state['wheel_offsets']):
        if state.get('wheel_velocity_override') is not None:
            world_velocity = state['wheel_velocity_override'][i]
        else:
            arm = rotate(offset, state['angle'])
            world_velocity = state['velocity']+state['yaw']*np.array([-arm[1], arm[0]])
            # Box2D also limits each wheel's translation to2m/.02s.
            world_velocity *= min(1., 100./max(100., float(np.linalg.norm(world_velocity))))
        local = rotate(world_velocity, -(state['angle']+state['joint'][i]))
        side.append(float(local[0])); front.append(float(local[1]))
    return np.asarray(front), np.asarray(side)


def tire_forces(omega, forward, sideways):
    requested = 82.*np.column_stack((-sideways, .54*omega-forward))
    ratios = np.linalg.norm(requested, axis=1)/400.
    applied = requested/np.maximum(1., ratios)[:, None]
    return applied, ratios


def predict_step(state, command):
    joint_target, requested_gas, brake = command
    state = copy.deepcopy(state)
    state['gas'][2:] += np.minimum(requested_gas-state['gas'][2:], .1)
    omega = state['omega']+DT*40000.*state['gas']/1.6/(abs(state['omega'])+5.)
    if brake >= .9:
        omega[:] = 0.
    elif brake > 0.:
        omega -= np.sign(omega)*np.minimum(15.*brake, abs(omega))
    forward, sideways = tire_inputs(state)
    forces, ratios = tire_forces(omega, forward, sideways)
    state['omega'] = omega-DT*forces[:, 1]*.54/1.6
    body_forces = np.array([rotate(force, state['joint'][i]) for i, force in enumerate(forces)])
    world_force = rotate(body_forces.sum(axis=0), state['angle'])
    torque = sum(cross(offset, force) for offset, force in zip(state['wheel_offsets'], body_forces))
    state['velocity'] += DT*world_force/state['mass']
    state['velocity'] *= min(1., 100./max(100., float(np.linalg.norm(state['velocity']))))
    state['yaw'] += DT*torque/state['inertia']
    state['angle'] += DT*state['yaw']
    state['joint'][:2] += DT*np.clip(50.*(joint_target-state['joint'][:2]), -3., 3.)
    state['joint'] = np.clip(state['joint'], -.4, .4)
    state['wheel_velocity_override'] = None
    diagnostics = {'wheel_forward_mps': forward.tolist(), 'wheel_lateral_mps': sideways.tolist(),
                   'tire_force_local_N': forces.tolist(), 'unsaturated_tire_demand_ratio': ratios.tolist()}
    return state, diagnostics


def summarize(state):
    local = rotate(state['velocity'], -state['angle'])
    forward, sideways = tire_inputs(state)
    forces, ratios = tire_forces(state['omega'], forward, sideways)
    return {'speed_mps': float(np.linalg.norm(state['velocity'])), 'body_yaw_radps': state['yaw'],
            'body_sideslip_deg': math.degrees(math.atan2(local[0], local[1])),
            'front_joint_rad': state['joint'][:2].tolist(),
            'rear_rolling_mps': (.54*state['omega'][2:]).tolist(),
            'rear_longitudinal_slip_mps': (.54*state['omega'][2:]-forward[2:]).tolist(),
            'wheel_lateral_mps': sideways.tolist(), 'wheel_forward_mps': forward.tolist(),
            'instant_tire_force_local_N': forces.tolist(), 'instant_tire_demand_ratio': ratios.tolist()}


def serializable_state(state):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in state.items()}


def error_row(predicted, actual):
    return {'speed_error_mps': abs(predicted['speed_mps']-actual['speed_mps']),
            'yaw_error_radps': abs(predicted['body_yaw_radps']-actual['body_yaw_radps']),
            'body_sideslip_error_deg': abs(predicted['body_sideslip_deg']-actual['body_sideslip_deg']),
            'front_joint_error_rad': float(np.max(abs(np.array(predicted['front_joint_rad'])-
                                                       np.array(actual['front_joint_rad'])))),
            'rear_longitudinal_slip_error_mps': float(np.max(abs(np.array(predicted['rear_longitudinal_slip_mps'])-
                                                                       np.array(actual['rear_longitudinal_slip_mps'])))),
            'max_per_tire_force_vector_error_N': float(np.max(np.linalg.norm(
                np.array(predicted['instant_tire_force_local_N'])-np.array(actual['instant_tire_force_local_N']), axis=1)))}


def run_case(config):
    world, car = make_car(config['speed'])
    for _ in range(config['warmup_steps']):
        tick(world, car, *config['warmup'])
    initial = body_state(car)
    predicted = copy.deepcopy(initial)
    rows = []
    for step in range(1, 17):
        predicted, forces = predict_step(predicted, config['command'])
        tick(world, car, *config['command'])
        actual = summarize(body_state(car))
        forecast = summarize(predicted)
        if step in HORIZONS:
            rows.append({'horizon_s': step*DT, 'predicted': forecast, 'actual': actual,
                         'absolute_error': error_row(forecast, actual), 'predicted_step_tire_forces': forces})
    passes = {}
    for name, horizon in (('at08', .08), ('at32', .32)):
        row = next(r for r in rows if abs(r['horizon_s']-horizon) < 1e-9)
        passes[name] = all(row['absolute_error'][metric] <= bound for metric, bound in ACCEPTANCE[name].items())
    return {'configuration': config, 'initial_privileged_state': serializable_state(initial),
            'initial_summary': summarize(initial), 'compound_mass_kg': initial['mass'],
            'compound_yaw_inertia_kgm2': initial['inertia'], 'rows': rows,
            'acceptance_passed': passes}


def behavioral_tests():
    world, car = make_car(70.)
    state = body_state(car)
    for _ in range(16):
        state, _ = predict_step(state, [0., 0., 0.])
    straight = summarize(state)
    assert abs(straight['speed_mps']-70.) < 1e-10 and abs(straight['body_yaw_radps']) < 1e-10
    assert np.max(abs(np.array(straight['rear_longitudinal_slip_mps']))) < 1e-8
    state = body_state(car)
    reversed_state, _ = predict_step(state, [0., 0., .9])
    # Brake lock is applied before tire force. Friction then changes wheelomega.
    locked_force, _ = tire_forces(np.zeros(4), *tire_inputs(state))
    np.testing.assert_allclose(reversed_state['omega'], -DT*locked_force[:, 1]*.54/1.6, atol=1e-10)
    for case in CASES:
        state = body_state(car)
        _, diagnostics = predict_step(state, case['command'])
        assert np.max(np.linalg.norm(diagnostics['tire_force_local_N'], axis=1)) <= 400.+1e-9
    return ['Unforced straight70m/s rolling state stays constant for.32s.',
            'Locked-brake update is applied before tire-force wheel response.',
            'All fixed-case modeled tire force vectors remain within400N.']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists')
    official = {name: digest(ROOT/name) for name in OFFICIAL_FILES}
    source_hash = digest(__file__)
    helper = ROOT/'agents/apex_2026/research/speed_20261005/physics_dynamics.py'
    helper_hash = digest(helper)
    checks = behavioral_tests()
    rows = [run_case(config) for config in CASES]
    assert source_hash == digest(__file__) and helper_hash == digest(helper)
    assert official == {name: digest(ROOT/name) for name in OFFICIAL_FILES}
    report = {'classification': 'privileged_fixed_four_tire_forward_model_validation',
              'is_legal_camera_candidate': False, 'is_lap_benchmark': False, 'new_driving_episodes': 0,
              'uniform_asphalt_cases': 4, 'source_sha256': source_hash, 'helper_sha256': helper_hash,
              'official_sha256': official, 'acceptance_thresholds': ACCEPTANCE,
              'behavioral_tests_passed': checks, 'rows': rows,
              'all_cases_meet_fixed_prediction_acceptance': all(all(row['acceptance_passed'].values()) for row in rows),
              'limits': ['Exact privileged initial body/wheel state; no camera observer error included.',
                         'Uniform asphalt only, with no contacts, edges, damage or road geometry.',
                         'Rigid-body approximation omits iterative Box2D joint and translation-limit solving.',
                         'Instantaneous force diagnostic uses end-state wheel slips, not forces integrated over the prior step.',
                         'Prediction validation is not a torque controller or a legal agent performance test.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
    print(json.dumps({'all_accepted': report['all_cases_meet_fixed_prediction_acceptance'],
                      'cases': [{'name': row['configuration']['name'], 'accepted': row['acceptance_passed'],
                                 'error_at08': row['rows'][0]['absolute_error'],
                                 'error_at32': row['rows'][-1]['absolute_error']} for row in rows]}, indent=2))


if __name__ == '__main__':
    main()
