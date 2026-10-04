"""Fixed supplied-state planner profiles; no worlds, episodes or lap claims."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from agents.apex_2026.fast_rear_clear_agent import Agent
from agents.apex_2026.research.speed_20261005.camera_mpc_reference import CameraGeometry
from agents.apex_2026.research.speed_20261005.physics_four_tire_model import predict_step
from agents.apex_2026.research.speed_20261005.predictive_control import FixedControlPlanner


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_at(speed):
    center = np.array([0., -.0804591735470409])
    return {'angle': 0., 'velocity': np.array([0., float(speed)]), 'yaw': 0.,
            'mass': 7.3019199669361115, 'inertia': 19.170645977936854,
            'local_center': center, 'wheel_offsets': np.array(
                [[-1.1, 1.6], [1.1, 1.6], [-1.1, -1.64], [1.1, -1.64]])-center,
            'joint': np.zeros(4), 'omega': np.full(4, speed/.54),
            'gas': np.zeros(4), 'wheel_velocity_override': None}


def ensemble(speed, count):
    states = [state_at(speed)]
    for quantity, amplitude in (('side', .5), ('yaw', .15), ('joint', .006)):
        for sign in (-1, 1):
            value = state_at(speed)
            if quantity == 'side':
                value['velocity'][0] = sign*amplitude
                value['velocity'][1] = np.sqrt(max(0., speed**2-amplitude**2))
            elif quantity == 'yaw':
                value['yaw'] = sign*amplitude
            else:
                value['joint'][:2] = sign*amplitude
            states.append(value)
    return states[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    assert not output.exists()
    paths = {'planner': Path(__file__).with_name('predictive_control.py'),
             'model': Path(__file__).with_name('physics_four_tire_model.py'),
             'geometry': Path(__file__).with_name('camera_mpc_reference.py'),
             'camera_perception': Path('agents/apex_2026/fast_rear_clear_agent.py'),
             'profile': Path(__file__)}
    before = {name: digest(path) for name, path in paths.items()}
    rows = []
    for speed, count in ((0., 1), (70., 3), (100., 3), (70., 7)):
        states = ensemble(speed, count)
        started = time.perf_counter()
        frame = np.full((84, 84), .63, np.float32)
        frame[:73, 33:52] = .40
        frame[73:] = 0.
        perception = Agent()
        field = perception._distance_field(frame)
        path = perception._ridge(frame)
        assert path is not None
        geometry = CameraGeometry(path, field)
        field_ms = 1000*(time.perf_counter()-started)
        planner = FixedControlPlanner(predict_step, geometry)
        result = planner.plan(states, [0., 0., 0.])
        total_ms = 1000*(time.perf_counter()-started)
        action = result.pop('action')
        result['action'] = None if action is None else action.tolist()
        rows.append({'speed_mps': speed, 'scenarios': count, 'field_and_ridge_ms': field_ms,
                     'total_ms_including_camera_field': total_ms, 'plan': result})
        assert total_ms < 5000., 'local bounded prototype exceeded official per-action allowance'
    assert before == {name: digest(path) for name, path in paths.items()}
    report = {'classification': 'fixed_synthetic_legal_state_local_profile_no_worlds_or_laps',
              'source_sha256': before, 'all_sources_unchanged': True,
              'world_steps': 0, 'new_holdout_opened': False, 'rows': rows,
              'limits': ['Profiles measure this host only, not an official worst-case timing guarantee.',
                         'Three/seven scenarios are limited separate perturbations, not a certified error box.',
                         'Body speed and rear-wheel raster uncertainty are not perturbed in this profile.',
                         'Projection bounds use maximum prior/post speed to match semi-implicit model integration.',
                         'No observer integration or completed-lap result is established here.']}
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
