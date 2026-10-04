"""PRIVILEGED fixed full-brake tail validation, not a driving controller.

Four handselected uniform-asphalt initial conditions test a held joint and
gas0/brake.9 for .72s. Exact simulator state initializes this diagnostic only.
No road maps, camera Agent edits, lap episodes or parameter search are used.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from agents.apex_2026.research.speed_20261005.physics_dynamics import (
    DT, OFFICIAL_FILES, ROOT, digest, make_car, tick,
)
from agents.apex_2026.research.speed_20261005.physics_four_tire_model import (
    body_state, error_row, predict_step, rotate, summarize,
)

CASES = (
    {'name': 'straight70', 'nominal_initial_speed_mps': 70., 'joint': 0., 'warmup_steps': 0},
    {'name': 'straight100', 'nominal_initial_speed_mps': 100., 'joint': 0., 'warmup_steps': 0},
    {'name': 'settled_turn_from70', 'nominal_initial_speed_mps': 70., 'joint': .06, 'warmup_steps': 50},
    {'name': 'settled_turn100', 'nominal_initial_speed_mps': 100., 'joint': .06, 'warmup_steps': 30},
)
ACCEPTANCE = {'max_hull_position_error_m': .5, 'max_speed_error_mps': 3.,
              'max_yaw_error_radps': .2, 'max_front_joint_error_rad': .01,
              'stop_tick_difference_s': .04}


def run_case(config):
    world, car = make_car(config['nominal_initial_speed_mps'])
    for _ in range(config['warmup_steps']):
        tick(world, car, config['joint'], .16, 0.)
    initial = body_state(car)
    state = copy.deepcopy(initial)
    start_hull = np.array(car.hull.position, float)
    initial_angle = initial['angle']
    center_shift = rotate(initial['local_center'], initial_angle)
    cm_displacement = np.zeros(2)
    command = [float(initial['joint'][0]), 0., .9]
    rows = []
    actual_stop, predicted_stop = None, None
    for step in range(1, 37):
        state, _ = predict_step(state, command)
        cm_displacement += DT*state['velocity']
        predicted_hull = cm_displacement+center_shift-rotate(state['local_center'], state['angle'])
        tick(world, car, *command)
        actual = summarize(body_state(car))
        predicted = summarize(state)
        actual_hull = np.array(car.hull.position, float)-start_hull
        error = error_row(predicted, actual)
        error['hull_position_error_m'] = float(np.linalg.norm(predicted_hull-actual_hull))
        error['hull_angle_error_rad'] = abs(state['angle']-float(car.hull.angle))
        if predicted_stop is None and predicted['speed_mps'] <= .5 and abs(predicted['body_yaw_radps']) <= .1:
            predicted_stop = step*DT
        if actual_stop is None and actual['speed_mps'] <= .5 and abs(actual['body_yaw_radps']) <= .1:
            actual_stop = step*DT
        rows.append({'time_s': step*DT,
                     'predicted_hull_xy_relative_initial_body_m': rotate(predicted_hull, -initial_angle).tolist(),
                     'actual_hull_xy_relative_initial_body_m': rotate(actual_hull, -initial_angle).tolist(),
                     'predicted': predicted, 'actual': actual, 'absolute_error': error})
    maximum = {name: max(r['absolute_error'][metric] for r in rows) for name, metric in
               (('max_hull_position_error_m', 'hull_position_error_m'),
                ('max_speed_error_mps', 'speed_error_mps'),
                ('max_yaw_error_radps', 'yaw_error_radps'),
                ('max_front_joint_error_rad', 'front_joint_error_rad'))}
    maximum['stop_tick_difference_s'] = abs(actual_stop-predicted_stop) if actual_stop is not None and predicted_stop is not None else None
    passed = {name: maximum[name] is not None and maximum[name] <= bound for name, bound in ACCEPTANCE.items()}
    return {'configuration': config, 'actual_initial_summary': summarize(initial),
            'tail_command_world_joint_gas_brake': command,
            'predicted_stop_speed_half_yaw_tenth_s': predicted_stop,
            'actual_stop_speed_half_yaw_tenth_s': actual_stop,
            'maximum_error_over_all36_raw_steps': maximum, 'acceptance_passed': passed,
            'checkpoints': [r for r in rows if r['time_s'] in (.32, .64, .72)],
            'full_raw_step_pose_and_state_rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve prior receipts')
    model = ROOT/'agents/apex_2026/research/speed_20261005/physics_four_tire_model.py'
    helper = ROOT/'agents/apex_2026/research/speed_20261005/physics_dynamics.py'
    hashes = {'source_sha256': digest(__file__), 'model_sha256': digest(model), 'helper_sha256': digest(helper)}
    official = {p: digest(ROOT/p) for p in OFFICIAL_FILES}
    rows = [run_case(c) for c in CASES]
    assert hashes == {'source_sha256': digest(__file__), 'model_sha256': digest(model), 'helper_sha256': digest(helper)}
    assert official == {p: digest(ROOT/p) for p in OFFICIAL_FILES}
    report = {'classification': 'privileged_fixed_uniform_asphalt_full_brake_tail_model_validation',
              'is_legal_camera_candidate': False, 'is_lap_benchmark': False, 'new_driving_episodes': 0,
              'new_holdout_opened': False, 'uniform_fixed_cases': 4, 'horizon_s': .72,
              **hashes, 'official_sha256': official, 'acceptance_thresholds': ACCEPTANCE,
              'all_case_acceptance_passed': all(all(r['acceptance_passed'].values()) for r in rows), 'rows': rows,
              'limits': ['Exact privileged initial state; no camera estimation uncertainty.',
                         'Uniform asphalt without contacts, grass, damage or camera road boundaries.',
                         'Settled-turn initial speed is the measured value after fixed low-gas warmup, not its nominal starting speed.',
                         'Brake.9 resets axle spin before each tire update; this models a locked-wheel stop, not brake grip reservation.',
                         'A matched fixed tail does not certify geometry feasibility or safe legal predictive controls.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
    print(json.dumps({'all_accepted': report['all_case_acceptance_passed'],
                      'rows': [{'name': r['configuration']['name'], 'actual_initial_speed_mps': r['actual_initial_summary']['speed_mps'],
                                'actual_stop_s': r['actual_stop_speed_half_yaw_tenth_s'],
                                'predicted_stop_s': r['predicted_stop_speed_half_yaw_tenth_s'],
                                'maximum_error': r['maximum_error_over_all36_raw_steps'],
                                'accepted': r['acceptance_passed']} for r in rows]}, indent=2))


if __name__ == '__main__':
    main()
