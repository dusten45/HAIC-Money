"""Fixed scalar/batch policy timing, including field/geometry; no new laps."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from agents.apex_2026.research.speed_20261005.physics_four_tire_model import predict_step
from agents.apex_2026.research.speed_20261005.predictive_batch_model import predict_batch
from agents.apex_2026.research.speed_20261005.predictive_batch_control import FixedBatchControlPlanner
from agents.apex_2026.research.speed_20261005.predictive_control import FixedControlPlanner
from agents.apex_2026.tests.test_predictive_control import legal_state, straight_geometry


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    return value


def equal_policy(batch, scalar):
    for key, value in scalar.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(batch[key], value)
        elif isinstance(value, float):
            np.testing.assert_allclose(batch[key], value, atol=1e-10, rtol=1e-12)
        else:
            assert batch[key] == value, (key, batch[key], value)


def timed_case(name, states, previous, geometry_factory):
    results, timings = {}, {}
    for label, planner_type, predictor in (
            ('scalar', FixedControlPlanner, predict_step),
            ('batch', FixedBatchControlPlanner, predict_batch)):
        started = time.perf_counter()
        geometry = geometry_factory()
        after_geometry = time.perf_counter()
        results[label] = planner_type(predictor, geometry).plan(states, previous)
        ended = time.perf_counter()
        timings[label] = {'field_and_geometry_setup_ms': 1000*(after_geometry-started),
                          'planner_with_geometry_evaluation_ms': 1000*(ended-after_geometry),
                          'total_ms': 1000*(ended-started)}
    equal_policy(results['batch'], results['scalar'])
    return {'name': name, 'timings': timings,
            'total_speedup': timings['scalar']['total_ms']/timings['batch']['total_ms'],
            'scalar_policy_result': serial(results['scalar']),
            'batch_policy_result': serial(results['batch']),
            'every_scalar_result_field_matched': True}


def main():
    base = Path('agents/apex_2026')
    output = base/'results'/'speed-20261005'/'predictive-batch-control-v1-profile.json'
    assert not output.exists()
    source = base/'fast_predictive_agent.py'
    expected = 'd2bd8a0855876f5cc4d281bd1183646ad29bd27e7f9231a4af6d917e901b844e'
    assert digest(source) == expected
    sources = {'standalone_reference': source,
               'scalar_model': Path(__file__).with_name('physics_four_tire_model.py'),
               'batch_model': Path(__file__).with_name('predictive_batch_model.py'),
               'scalar_planner': Path(__file__).with_name('predictive_control.py'),
               'batch_planner': Path(__file__).with_name('predictive_batch_control.py'),
               'geometry': Path(__file__).with_name('camera_mpc_reference.py'),
               'profile': Path(__file__)}
    before = {name: digest(path) for name, path in sources.items()}
    assert before['scalar_planner'] == '341c61575651774cd5b582c644e47da021fc27737f43cb606b1a57fa561caf99'
    spec = importlib.util.spec_from_file_location('batch_profile_frozen_camera', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    field_builder = module.Agent()
    fixture_path = base/'results'/'speed-20261005'/'predictive-batch-control-v1-cameras.npz'
    fixture = np.load(fixture_path)
    cases = []
    for step in (20, 40):
        states = json.loads(str(fixture[f'states_step{step}']))
        for state in states:
            for key in FixedControlPlanner.ARRAY_SHAPES:
                state[key] = np.asarray(state[key], float)

        def geometry_factory(step=step):
            observation = fixture[f'observation_step{step}']
            frame = field_builder._frame(observation)
            circles = fixture[f'circles_step{step}']
            field, grass = field_builder._predictive_field(frame, circles)
            np.testing.assert_array_equal(field, fixture[f'field_step{step}'])
            np.testing.assert_array_equal(grass, fixture[f'grass_step{step}'])
            return module._HullCameraGeometry(fixture[f'path_step{step}'], field, circles, grass)

        cases.append(timed_case(f'existing_camera_step{step}_three_scenarios', states,
                                fixture[f'previous_step{step}'], geometry_factory))
    states = [legal_state(100.) for _ in range(3)]
    states[1]['yaw'], states[2]['yaw'] = -.15, .15
    states[1]['joint'][:2], states[2]['joint'][:2] = -.006, .006
    cases.append(timed_case('fixed_straight_100mps_three_scenarios', states,
                            np.zeros(3, np.float32), straight_geometry))
    assert before == {name: digest(path) for name, path in sources.items()}
    report = {'classification': 'fixed_existing_camera_and_supplied_state_policy_profile_not_lap',
              'source_sha256': before, 'fixture_sha256': digest(fixture_path),
              'all_sources_unchanged': True, 'world_steps': 0, 'new_holdout_opened': False,
              'policy_changed': False, 'cases': cases,
              'limits': ['Single fixed measurement per case on this shared host, no worst-case timing proof.',
                         'Actual camera cases include rebuilding their distance field and all rollout/backup geometry.',
                         'Camera decoding, observer update, ridge extraction and scenario construction are outside timers.',
                         'Synthetic case constructs a supplied field, not camera perception.',
                         'Pure finite predictor/geometry callbacks are required; eager batch errors may differ on scalar-unvisited rows.',
                         'No standalone agent changed, no episodes run, no pace or safety certificate.']}
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
