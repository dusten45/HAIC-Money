"""Export only old camera observations and past legal controls, no worlds."""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    raw = Path('.haic-artifacts/apex-speed-20261005/slow-phase-track3-v2')
    base = Path('agents/apex_2026')
    output = base/'tests/fixtures/predictive-batch-history.npz'
    receipt = base/'results/speed-20261005/predictive-batch-integration-v1-history.json'
    assert not output.exists() and not receipt.exists()
    camera_pin = 'dbb7a5c1adc6ea2624fcefc1cf33507f48ed40cc7d28c69c251f22b10f63088b'
    label_pin = 'd1732aaebf0faff8775175d6946eea2c5a533d3e3affdbbcae871d8c0311f431'
    assert digest(raw/'cameras.npz') == camera_pin
    assert digest(raw/'inspection.json') == label_pin
    rows = json.loads((raw/'inspection.json').read_text())['rows']
    cameras = np.load(raw/'cameras.npz')
    # Future truth/progress/contact labels are not exported. The target-frame
    # logged command is not needed or exported; only its preceding actions.
    observations = cameras['observations'][:61].copy()
    actions = np.asarray([row['action'] for row in rows[:60]], np.float32)
    assert observations.shape == (61, 4, 84, 84) and actions.shape == (60, 3)
    assert np.isfinite(observations).all() and np.isfinite(actions).all()
    np.savez_compressed(output, observations=observations, actions=actions)
    result = {'classification': 'existing_camera_legal_history_export_not_new_episode',
              'source_camera_sha256': camera_pin, 'source_inspection_sha256': label_pin,
              'script_sha256': digest(__file__), 'fixture_path': str(output),
              'fixture_sha256': digest(output), 'fixture_bytes': output.stat().st_size,
              'frames': 61, 'past_legal_actions': 60, 'fixed_target_frames': [20, 40, 60],
              'truth_for_initialization': False, 'world_steps': 0, 'new_holdout_opened': False,
              'old_candidate_prefix_not_a_new_predictive_trajectory': True}
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
