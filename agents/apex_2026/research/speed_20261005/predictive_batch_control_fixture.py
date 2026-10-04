"""Export causal saved camera states/geometry, without planning or worlds."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = Path('agents/apex_2026/results/speed-20261005/predictive-batch-control-v1-cameras.npz')
    receipt = output.with_suffix('.json')
    assert not output.exists() and not receipt.exists()
    source = Path('agents/apex_2026/fast_predictive_agent.py')
    pin = 'd2bd8a0855876f5cc4d281bd1183646ad29bd27e7f9231a4af6d917e901b844e'
    assert digest(source) == pin
    spec = importlib.util.spec_from_file_location('batch_camera_source_reference', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = Path('.haic-artifacts/apex-speed-20261005/slow-phase-track3-v2')
    labels = json.loads((raw/'inspection.json').read_text())
    cameras = np.load(raw/'cameras.npz')
    assert digest(raw/'cameras.npz') == labels['camera_npz_sha256']
    assert labels['all_actions_and_post_step_speeds_exact']
    agent = module.Agent()
    previous = np.zeros(3, np.float32)
    arrays, cases = {}, []
    for step, row in enumerate(labels['rows'][:41]):
        assert row['step'] == step
        observation, frame = cameras['observations'][step], cameras['frames'][step]
        # One camera-only parent update; observer receives only the camera
        # and past logged legal actions. No label initializes/corrects it.
        module._BaseRearClearAgent.act(agent, observation)
        state, sensor = agent._observer.observe(frame)
        if step in (20, 40):
            geometry = agent._predictive_geometry(frame, state)
            assert geometry is not None
            scenarios = agent._predictive_scenarios(state)
            serial = [{key: value.tolist() if isinstance(value, np.ndarray) else value
                       for key, value in item.items()} for item in scenarios]
            for key, value in (('observation', observation), ('field', geometry.field),
                               ('path', geometry.path), ('circles', geometry.circles),
                               ('grass', geometry.grass_mask), ('previous', previous)):
                arrays[f'{key}_step{step}'] = np.asarray(value).copy()
            arrays[f'states_step{step}'] = np.asarray(json.dumps(serial))
            cases.append({'step': step, 'past_actions_seen': agent._observer.actions_seen,
                          'sensor': sensor, 'scenario_count': len(scenarios)})
        action = np.asarray(row['action'], np.float32)
        agent._observer.advance(action)
        agent.last_steer = float(action[0])
        previous = action.copy()
    assert digest(source) == pin
    np.savez_compressed(output, **arrays)
    result = {'classification': 'causal_existing_camera_state_geometry_export_not_lap',
              'source_sha256': pin, 'script_sha256': digest(__file__),
              'input_inspection_sha256': digest(raw/'inspection.json'),
              'input_camera_sha256': digest(raw/'cameras.npz'), 'fixture_sha256': digest(output),
              'truth_for_initialization': False, 'past_actions_are_logged_legal_prefix': True,
              'world_steps': 0, 'new_holdout_opened': False, 'cases': cases}
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
