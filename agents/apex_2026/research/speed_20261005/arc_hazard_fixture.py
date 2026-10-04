"""Export existing exact camera prefixes; no simulator episode is executed."""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    raw = Path('.haic-artifacts/apex-speed-20261005/rear-clear-bottleneck')
    result = Path('agents/apex_2026/results/speed-20261005')
    fixture = result/'arc-hazard-v1-cameras.npz'
    receipt = result/'arc-hazard-v1-cameras.json'
    assert not fixture.exists() and not receipt.exists()
    source = Path('agents/apex_2026/fast_rear_clear_agent.py')
    expected = '093aaa77a0123138e52f51f576d54e6ae95b36b38929056f8a6a6a22215740dc'
    assert digest(source) == expected
    prior = json.loads((result/'rear-clear-bottleneck.json').read_text())
    assert prior['source_sha256'] == expected
    assert prior['all_actions_and_post_step_speeds_exact']
    arrays, cases = {}, []
    for track, steps in ((1, (59, 60)), (4, (128, 132, 133, 134, 135, 139))):
        inspection = raw/f'track{track}-inspection.json'
        rows = json.loads(inspection.read_text())
        observations = np.load(raw/f'track{track}-all-cameras.npz')['observations']
        pin = next(case for case in prior['cases'] if case['track_id'] == track)
        assert digest(inspection) == pin['raw_inspection_sha256']
        for step in steps:
            row = rows[step]
            assert row['step'] == step
            arrays[f'track{track}_observation_step{step}'] = observations[step].copy()
            arrays[f'track{track}_prestate_step{step}'] = np.asarray(json.dumps(row['previous']))
            cases.append({'track_id': track, 'step': step, 'prestate': row['previous'],
                          'reference_action': row['action'],
                          'original_exact_prefix_action_sha256': pin['action_prefix_sha256']})
    np.savez_compressed(fixture, **arrays)
    receipt.write_text(json.dumps({
        'classification': 'export_existing_exact_open_camera_fixture_no_new_episode',
        'parent_source_sha256': expected, 'script_sha256': digest(__file__),
        'input_diagnostic_sha256': digest(result/'rear-clear-bottleneck.json'),
        'fixture_sha256': digest(fixture), 'new_holdout_opened': False, 'cases': cases,
    }, indent=2)+'\n')
    print(receipt)


if __name__ == '__main__':
    main()
