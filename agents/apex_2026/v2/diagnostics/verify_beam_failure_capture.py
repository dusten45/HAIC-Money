"""Read-only validation of retained Beam R3 replay evidence; never resets an environment."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CAPTURE = Path('/tmp/apex-v2-beam-r3/failure_capture')
TRACE_SHA = 'a876261132ff4f9d745acf00668f0c508fdcebb78b8d3fdb05df904b243ef988'
BEAM_SHA = 'aef90410f8ad7f3f022a7c5a07c9ba87b997304964a63e5ca0601d8e60e7e8fa'
SHADOW_SHA = '8d5c21ce445bda2f192e4815739253ebf46d1099f635d966c12fe9e877990a56'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    run = CAPTURE.parent
    trace_path = run / 't3-s4111953688.jsonl'
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    original = json.loads((run / 't3-s4111953688.json').read_text())
    provenance = original['provenance']
    receipt = json.loads((CAPTURE / 'receipt.json').read_text())
    truth = json.loads((CAPTURE / 'truth.json').read_text())
    require(sha(trace_path) == TRACE_SHA == receipt['trace_sha256'], 'Trace hash mismatch')
    require(provenance['agent_sha256'] == BEAM_SHA == sha(run / 'source.py'), 'Beam source mismatch')
    require(sha(ROOT / 'agents/apex_2026/v2/shadow_physics.py') == SHADOW_SHA, 'Shadow source mismatch')
    require(sha(ROOT / 'agents/apex_2026/evaluate.py') == provenance['evaluator_sha256'], 'Evaluator source mismatch')
    for name, expected in provenance['official_source_sha256'].items():
        require(sha(ROOT / name) == expected, f'Official source mismatch: {name}')
    require(receipt['steps'] == 160 and receipt['captured_steps'] == [120, 160], 'Capture scope mismatch')
    require([entry['step'] for entry in truth] == list(range(120, 161)), 'Missing capture states')
    with np.load(CAPTURE / 'observations.npz', allow_pickle=False) as archive:
        obs = archive['observations']
        require(obs.shape == (41, 4, 84, 84) and obs.dtype == np.float32, 'Observation schema mismatch')
        require(np.isfinite(obs).all() and obs.min() >= 0 and obs.max() <= 1, 'Observation value range mismatch')
        bounds = [float(obs.min()), float(obs.max())]
    for entry in truth:
        step = entry['step']
        require(entry['before'] == rows[step - 1]['before'], f'Before-state mismatch at {step}')
        require(entry['before'] == rows[step - 2]['after'], f'Previous after-state mismatch at {step - 1}')
    for step, row in enumerate(rows[:160], 1):
        require(np.array(row['action'], dtype=np.float32).tolist() == row['action'], f'Action conversion changes step {step}')
    files = [trace_path, run / 't3-s4111953688.json', run / 'source.py', run / 'capture_failure.py',
             CAPTURE / 'observations.npz', CAPTURE / 'truth.json', CAPTURE / 'receipt.json']
    return {
        'schema_version': 1,
        'scope': 'Read-only verification of existing diagnostic replay evidence; no new evaluation or reset',
        'verification_resets': 0,
        'capture_reset_count_reported': receipt['reset_count'],
        'capture_owner': 'shadow_physics',
        'track_id': original['track_id'], 'seed': original['seed'],
        'original_episode_steps': len(rows),
        'source_sha256': {'beam': BEAM_SHA, 'shadow': SHADOW_SHA, 'evaluator': provenance['evaluator_sha256'],
                          'official': provenance['official_source_sha256']},
        'files_sha256': {str(path): sha(path) for path in files},
        'verifier_sha256': sha(__file__),
        'verified': {
            'exact_saved_before_states': {'count': 41, 'steps_inclusive': [120, 160]},
            'exact_previous_after_states': {'count': 41, 'steps_inclusive': [119, 159]},
            'float32_action_conversion_lossless': {'count': 160, 'steps_inclusive': [1, 160]},
            'observation_shape': [41, 4, 84, 84], 'observation_dtype': 'float32', 'observation_range': bounds,
            'observation_index': 'decision_step - 120; stack BEFORE action',
            'current_evaluator_and_official_sources_match_original': True,
        },
        'capture_script_reported_checks': {'after_position_steps_inclusive': [1, 160], 'absolute_position_l2_tolerance': 1e-4},
        'limitations': [
            'No separately retained replay actions; script directly casts recorded actions to float32, verified lossless.',
            'After-position values for all 160 steps were not saved; completion receipt reports tolerance checks, not bitwise equality.',
            'Full-state equality is established only for the 41 saved dictionaries, not the whole 160-step replay or 249-step original episode.',
            'Public observation stacks were not saved in the original run, so their original-run pixel parity cannot be independently checked.',
            'Additional velocity, wheel, RPM and gas telemetry is diagnostic-only and has no original-trace counterpart for exact parity.',
            'Source checks reflect current files; capture script did not snapshot source hashes at replay start.',
        ],
    }


if __name__ == '__main__':
    result = verify()
    destination = ROOT / 'agents/apex_2026/v2/results/beam-failure-capture-verification.json'
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'result': str(destination), 'exact_saved_states': 41, 'new_resets': 0}))
