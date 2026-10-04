"""Source-bound mandatory replay diagnostics; this is never submission code.

Agent sees only the original camera stack. Diagnostic attributes are copied
after its sole act call. Cached cameras are before that call; simulator truth
is recorded separately after it and never supplied to inference.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from agents.apex_2026.evaluate import (
    ENV_FILES, MANDATORY_CELLS, end_reason, validate_action,
)


ROOT = Path(__file__).resolve().parents[4]
SEMANTIC_FIELDS = ('track_id', 'seed', 'finished', 'lap_time_ms', 'progress',
                   'collision_count', 'damage', 'retire_reason', 'steps',
                   'offtrack_samples', 'partial_offtrack_samples',
                   'source_sha256', 'parameters', 'action_trace_sha256', 'error')
DIAGNOSTIC_FIELDS = ('predictive_status', 'predictive_result', 'predictive_sensor',
                     'predictive_latency_s', 'predictive_model_calls')
DEBUG_FIELDS = ('last_speed', 'last_target', 'last_steer', 'last_yaw',
                'lost_frames', 'pass_side', 'pass_missing', 'pass_x', 'pass_y',
                'corridor_road', 'corridor_phase')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_binding(source, receipt, track, seed):
    """Validate existing SCREEN inputs before loading source or creating a world."""
    if (track, seed) not in MANDATORY_CELLS:
        raise ValueError('only the four mandatory measured cells may be replayed')
    source, receipt = Path(source).resolve(), Path(receipt).resolve()
    source_sha, receipt_sha = digest(source), digest(receipt)
    document = json.loads(receipt.read_text())
    freeze, rows = document['freeze'], document['rows']
    if freeze['source_sha256'] != source_sha:
        raise ValueError('source hash does not match original receipt')
    expected = [list(cell) for cell in MANDATORY_CELLS]
    cells = [(row['track_id'], row['seed']) for row in rows]
    if (freeze['suite'] != 'mandatory' or freeze['cells'] != expected or
            len(cells) != 4 or set(cells) != set(MANDATORY_CELLS)):
        raise ValueError('original SCREEN must contain exactly four mandatory cells')
    params = freeze['parameters']
    if not isinstance(params, dict):
        raise ValueError('frozen parameters must be an object')
    for row in rows:
        if any(key not in row for key in SEMANTIC_FIELDS):
            raise ValueError('original row is missing semantic fields')
        if row['source_sha256'] != source_sha:
            raise ValueError('row source hash disagrees with freeze')
        if row['parameters'] != params:
            raise ValueError('row parameters disagree with freeze')
        if row.get('error') is not None:
            raise ValueError('an original worker error cannot authorize exact replay')
    environment = freeze['environment_sha256']
    if set(environment) != set(ENV_FILES):
        raise ValueError('environment hash coverage differs from official evaluator')
    for name, sha in environment.items():
        if digest(ROOT/name) != sha:
            raise ValueError('environment hash changed: '+name)
    row = next(row for row in rows if (row['track_id'], row['seed']) == (track, seed))
    if (type(row.get('steps')) is not int or row['steps'] < 1 or
            type(freeze['max_steps']) is not int or row['steps'] > freeze['max_steps']):
        raise ValueError('original steps are invalid')
    action_sha = row.get('action_trace_sha256', '')
    if (not isinstance(action_sha, str) or len(action_sha) != 64 or
            any(char not in '0123456789abcdef' for char in action_sha)):
        raise ValueError('original action hash is invalid')
    return dict(source_path=source, receipt_path=receipt, source_sha256=source_sha,
                receipt_sha256=receipt_sha, parameters=params, row=row,
                freeze=freeze, environment_sha256=environment)


def require_current_hashes(binding):
    for key, hash_key in (('source_path', 'source_sha256'),
                          ('receipt_path', 'receipt_sha256')):
        if digest(binding[key]) != binding[hash_key]:
            raise ValueError(key.split('_')[0]+' changed after binding')
    for name, sha in binding['environment_sha256'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('environment changed after binding: '+name)


def load_bound_agent(binding):
    require_current_hashes(binding)
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location('apex_submission', binding['source_path'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    agent = module.Agent(**binding['parameters'])
    latency_ms = 1000*(time.perf_counter()-started)
    require_current_hashes(binding)
    return agent, latency_ms


class ActionTrace:
    def __init__(self):
        self._hash = hashlib.sha256()
        self.count = 0

    def add(self, action):
        value = validate_action(action)
        self._hash.update(value.tobytes())
        self.count += 1
        return value

    def sha256(self):
        return self._hash.hexdigest()

    def verify(self, expected_sha, expected_steps):
        if self.count != expected_steps:
            raise ValueError(f'action steps differ: {self.count} != {expected_steps}')
        if self.sha256() != expected_sha:
            raise ValueError('action hash differs from original receipt')
        return True


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': 'nan' if math.isnan(value) else '+inf' if value > 0 else '-inf'}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {'unserializable_type': type(value).__name__}


def capture_diagnostics(agent):
    result = {key: _json_safe(getattr(agent, key)) for key in DIAGNOSTIC_FIELDS
              if hasattr(agent, key)}
    result['missing_fields'] = [key for key in DIAGNOSTIC_FIELDS if not hasattr(agent, key)]
    observer = getattr(agent, '_observer', None)
    result['observer_actions_seen'] = _json_safe(getattr(observer, 'actions_seen', None))
    result['observer_state_after_advance'] = _json_safe(getattr(observer, 'state', None))
    result['previous_emitted_action'] = _json_safe(getattr(agent, '_predictive_previous_action', None))
    result['debug'] = {key: _json_safe(getattr(agent, key)) for key in DEBUG_FIELDS
                       if hasattr(agent, key)}
    return result


def verify_outcome(actual, expected):
    changed = [key for key in SEMANTIC_FIELDS if actual.get(key) != expected.get(key)]
    if changed:
        raise ValueError('nonruntime outcome differs: '+', '.join(changed))
    return True


def summarize(rows):
    counts = Counter(row['diagnostics'].get('predictive_status', '<missing>') for row in rows)
    latency = np.array([row['action_latency_ms'] for row in rows], float)
    calls = [row['diagnostics'].get('predictive_model_calls') for row in rows]
    calls = [value for value in calls if isinstance(value, (int, float)) and math.isfinite(value)]
    return dict(steps=len(rows), status_counts=dict(counts),
                predictive_active_steps=counts['predictive'],
                predictive_active_fraction=counts['predictive']/len(rows) if rows else 0.,
                action_latency_max_ms=float(latency.max()) if len(rows) else None,
                action_latency_p95_ms=float(np.quantile(latency, .95)) if len(rows) else None,
                model_calls_max=max(calls) if calls else None,
                model_calls_sum=sum(calls) if calls else None)


def run_replay(source, receipt, track, seed, output, cameras):
    binding = validate_binding(source, receipt, track, seed)
    output, cameras = Path(output).resolve(), Path(cameras).resolve()
    if output == cameras or output.exists() or cameras.exists():
        raise ValueError('use separate new diagnostic output and camera paths')
    if not cameras.is_relative_to(ROOT/'.haic-artifacts'):
        raise ValueError('large camera arrays must stay in ignored .haic-artifacts')
    agent, initialization_ms = load_bound_agent(binding)
    # Imports and world creation happen only after an existing receipt is bound.
    from local_simulator.environment import create_environment, reset_environment
    from local_simulator.schema import MapSpec
    spec = MapSpec(track, seed, 'official', (), binding['freeze']['max_steps'], 4)
    environment, raw = create_environment(spec, render_mode=None)
    rows, frames, contacts, offroad, partial = [], [], 0, 0, 0
    actions, camera_hash = ActionTrace(), hashlib.sha256()
    try:
        observation, _ = reset_environment(environment, spec)
        start_time = float(raw.t)
        started = time.perf_counter()
        agent.reset(observation)
        reset_ms = 1000*(time.perf_counter()-started)
        for step in range(binding['freeze']['max_steps']):
            # Copy before inference, then call act exactly once. No diagnostic
            # method is invoked on Agent or its observer/planner.
            frame = np.asarray(observation, dtype=np.float32).copy()
            frames.append(frame)
            camera_hash.update(frame.tobytes())
            started = time.perf_counter()
            action = actions.add(agent.act(observation))
            latency_ms = 1000*(time.perf_counter()-started)
            diagnostics = capture_diagnostics(agent)
            truth = dict(speed=float(np.linalg.norm(raw.car.hull.linearVelocity)),
                         position=list(raw.car.hull.position), angle=float(raw.car.hull.angle),
                         world_yaw_radps=float(raw.car.hull.angularVelocity))
            observation, _, terminated, truncated, info = environment.step(action)
            contacts += int(info.get('collision', False))
            wheels = [bool(wheel.tiles) for wheel in raw.car.wheels]
            offroad += int(not any(wheels))
            partial += int(not all(wheels))
            rows.append(dict(step=step, action=action.tolist(), action_latency_ms=latency_ms,
                             diagnostics=diagnostics, pre_step_truth=truth,
                             sim_time_s=float(raw.t-start_time), progress=info.get('progress', 0.),
                             collision=bool(info.get('collision', False))))
            if terminated or truncated:
                break
        finish = raw.finish_time_s
        actual = dict(track_id=track, seed=seed, finished=finish is not None,
                      lap_time_ms=round(1000*(finish-start_time)) if finish is not None else None,
                      progress=float(info.get('progress', 0.)), collision_count=contacts,
                      damage=float(info.get('damage', 0.)),
                      retire_reason=end_reason(info, finished=finish is not None,
                                               terminated=terminated, truncated=truncated),
                      steps=actions.count, offtrack_samples=offroad, partial_offtrack_samples=partial,
                      source_sha256=binding['source_sha256'], parameters=binding['parameters'],
                      action_trace_sha256=actions.sha256(), error=None)
    finally:
        environment.close()
    require_current_hashes(binding)
    failures = []
    for check in (lambda: actions.verify(binding['row']['action_trace_sha256'], binding['row']['steps']),
                  lambda: verify_outcome(actual, binding['row'])):
        try:
            check()
        except ValueError as error:
            failures.append(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    cameras.parent.mkdir(parents=True, exist_ok=True)
    with cameras.open('xb') as stream:
        np.savez_compressed(stream, observations=np.asarray(frames, dtype=np.float32))
    report = dict(classification='exact-source diagnostic replay; not a new candidate gate',
                  source_sha256=binding['source_sha256'], receipt_sha256=binding['receipt_sha256'],
                  source_path=str(binding['source_path']), receipt_path=str(binding['receipt_path']),
                  harness_sha256=digest(__file__), environment_sha256=binding['environment_sha256'],
                  parameters=binding['parameters'], track_id=track, seed=seed,
                  exact_repeat=not failures, integrity_failures=failures,
                  original_row=binding['row'], replay_row=actual,
                  camera_path=str(cameras), camera_file_sha256=digest(cameras),
                  camera_stack_sha256=camera_hash.hexdigest(), camera_phase='before act, stacked CHW float32',
                  truth_enters_inference=False, new_holdout_opened=False,
                  summary=summarize(rows), initialization_ms=initialization_ms, reset_ms=reset_ms,
                  rows=rows,
                  limits=['Replay latencies include diagnostic host conditions and are not official certification.',
                          'Wall-clock planning budgets may cause action divergence; divergence is an integrity failure.',
                          'Diagnostic statuses are observed in this replay; exact actions do not prove identical original planning branches.',
                          'Missing or nonfinite diagnostic fields are explicitly represented, not imputed.',
                          'Camera arrays precede act; observer state follows the single emitted-action advance.'])
    with output.open('x') as stream:
        json.dump(_json_safe(report), stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'receipt', 'output', 'cameras'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--track', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    report = run_replay(args.source, args.receipt, args.track, args.seed, args.output, args.cameras)
    print(json.dumps(dict(exact_repeat=report['exact_repeat'], summary=report['summary'])))
    if not report['exact_repeat']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
