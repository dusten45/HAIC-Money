"""Fixed six-cell consumed-TRAIN shield A/B; only the explicit run command resets.

Freeze and validate are zero-environment operations. The frozen crossing ZIP is
the only comparator; neither recovery nor steering-release is a runtime input.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import zipfile

from scripts import evaluate_koi_collision_priority as legacy


ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / 'runs/koi-collision-recovery-v1'
WRAPPER = ROOT / 'haic/algorithms/koi/collision_shield.py'
ARMS = ('crossing_projection', 'collision_shield')
CELLS = ((1, 3184000002), (2, 3184000006), (2, 3184000001),
         (3, 3184000002), (1, 3184000013), (1, 3184000015))
SUBGROUPS = {f'{t}:{s}': ('corner_conflict_challenges' if i < 4 else 'ordinary_controls')
             for i, (t, s) in enumerate(CELLS)}
SCHEMA = 'koi-collision-shield-v1'
GATE: dict = dict(max_actions_per_encounter=6, rearm_threat_free_decisions=3,
            max_extra_offroad_ticks=24, max_lateral_increase_m=3.16,
            max_heading_error_increase_rad=.25, max_kept_path_ratio=1.01,
            max_kept_lap_increase_ms=20, minimum_damage_benefit=.2,
            minimum_collision_benefit=1, no_lost_finish=True,
            no_per_cell_damage_or_collision_increase=True, no_new_clean_object_hit=True,
            require_complete_matched_evidence=True, require_exact_noop_and_pedals=True,
            benefit='gained finish OR damage >=0.2 AND collision decisions >=1 lower',
            tuning=False)
sha, read_json, save = legacy.sha, legacy.read_json, legacy.save


def scheduled_slots():
    rows = []
    for track, seed in CELLS:
        arms = ARMS if (track + seed) % 2 else ARMS[::-1]
        rows.extend(dict(track_id=track, seed=seed, mode=arm, repeat=0,
                         file=f'{track}-{seed}-r0-{arm}.json', status='unrun') for arm in arms)
    return rows


def local_path(directory, name):
    path = (Path(directory) / name).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise ValueError(f'path outside study: {name}')
    return path


def valid_episode(episode):
    return (episode.get('error') is None and episode.get('invalid_actions') == 0
            and isinstance(episode.get('completed'), bool)
            and (episode['completed'] or episode.get('retire_reason') in ('off_track', 'crash'))
            and episode.get('damage') is not None and episode.get('collisions') is not None
            and bool(episode.get('decision_trace')))


def consumed_evidence() -> dict:
    """Verify only these already-consumed TRAIN receipts, not a freshness claim."""
    parent = read_json(RECOVERY / 'protocol.json')
    report = read_json(RECOVERY / 'episode-report.json')
    if (parent['cells'] != [list(c) for c in CELLS] or parent['repeat'] != 1
            or parent['comparator_zip_sha256'] != legacy.CROSSING_SHA
            or parent['baseline'] != ARMS[0] or report['operator_error'] is not None):
        raise ValueError('recovery selection or crossing identity mismatch')
    pins = {str(RECOVERY / name): sha(RECOVERY / name)
            for name in ('protocol.json', 'episode-report.json')}
    rows, seen = [], set()
    for row in report['rows']:
        key = (row['track_id'], row['seed'], row['mode'])
        if (key in seen or key[:2] not in CELLS or row['repeat'] != 0
                or row['mode'] not in (ARMS[0], 'collision_recovery') or row['status'] != 'completed'):
            raise ValueError('incomplete or duplicate consumed recovery receipt')
        seen.add(key)
        path = local_path(RECOVERY, row['file'])
        if sha(path) != row['sha256']:
            raise ValueError('consumed episode hash mismatch')
        episode = read_json(path)
        if not valid_episode(episode) or (episode['track_id'], episode['seed'], episode['mode']) != key:
            raise ValueError('invalid consumed episode')
        pins[str(path)] = sha(path)
        for field in ('raw_trace', 'decision_stream'):
            evidence = local_path(RECOVERY, episode[field + '_file'])
            if sha(evidence) != episode[field + '_sha256']:
                raise ValueError('consumed stream hash mismatch')
            pins[str(evidence)] = sha(evidence)
        if row['mode'] == ARMS[0]:
            rows.append(dict(track_id=key[0], seed=key[1], finished=episode['completed'],
                             damage=episode['damage'], collisions=episode['collisions'],
                             subgroup=SUBGROUPS[f'{key[0]}:{key[1]}']))
    if len(seen) != 12:
        raise ValueError('all twelve consumed recovery episodes required')
    # This selected-cell validator rejects old24, Track4, protected and unrun cells.
    legacy.validate_cells(list(CELLS), baseline=ARMS[0])
    return dict(source_sha256=pins, baseline_outcomes=rows,
                selection='Same six outcome-selected recovery cells, not fresh or randomly sampled.',
                subgroup_definition='First four prior conflict/recovery challenges; last two clean baseline-finish controls. Roles are not claims that control roads contain no corners.')


def freeze_protocol(output):
    """Create a separate immutable protocol, without importing Agent/environment."""
    evidence = consumed_evidence()
    comparator = legacy.BASELINES[ARMS[0]][0]
    if sha(comparator) != legacy.CROSSING_SHA:
        raise ValueError('crossing ZIP identity mismatch')
    if not WRAPPER.is_file():
        raise ValueError('collision shield runtime is not ready')
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    names = ('__init__.py', 'evaluate_koi_collision_shield.py', 'analyze_koi_collision_shield.py',
             'evaluate_koi_collision_priority.py', 'analyze_koi_collision_priority.py',
             'evaluate_koi_minimum_clearance_ab.py', 'evaluate_koi_adaptive_ab.py',
             'package_koi_adaptive_avoidance.py')
    sources = {ROOT / 'scripts' / name: 'scripts/' + name for name in names}
    sources[WRAPPER] = 'collision_shield.py'
    source_hashes = {}
    for source, relative in sources.items():
        copied = output / 'source' / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copied)
        source_hashes[str(copied)] = sha(copied)
    copied_zip = output / comparator.name
    shutil.copyfile(comparator, copied_zip)
    if sha(copied_zip) != legacy.CROSSING_SHA:
        raise ValueError('copied crossing ZIP mismatch')
    with zipfile.ZipFile(copied_zip) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names)):
            raise ValueError('unsafe or duplicate ZIP members')
        archive.extractall(output / 'model')
    model_hashes = {str(output / 'model' / n): sha(output / 'model' / n) for n in names if not n.endswith('/')}
    environment_hashes = read_json(RECOVERY / 'protocol.json')['environment_sha256']
    for path, expected in environment_hashes.items():
        if sha(path) != expected:
            raise ValueError(f'original environment source changed: {path}')
    preserved = [ROOT / 'agent.py', legacy.V2_ZIP,
                 ROOT / 'submissions/koi-steering-release-v2.manifest.json',
                 ROOT / 'submissions/koi-steering-release-v2-submission.zip',
                 ROOT / 'haic/algorithms/koi/collision_recovery.py',
                 ROOT / 'haic/algorithms/koi/steering_release.py']
    if sha(legacy.V2_ZIP) != legacy.V2_SHA:
        raise ValueError('preserved v2 ZIP identity mismatch')
    protocol = dict(schema=SCHEMA, scope=legacy.SCOPE, fresh=False, official_action=False,
                    baseline=ARMS[0], candidate_name=ARMS[1], class_name='CollisionShieldAgent',
                    arms=list(ARMS), cells=[list(c) for c in CELLS], repeat=1,
                    subgroups=SUBGROUPS, gate=GATE, schedule=scheduled_slots(),
                    comparator_zip=str(copied_zip), comparator_zip_sha256=legacy.CROSSING_SHA,
                    wrapper_file='source/collision_shield.py', wrapper_sha256=sha(output / 'source/collision_shield.py'),
                    source_sha256=source_hashes, environment_sha256=environment_hashes,
                    model_source_sha256=model_hashes, consumed_evidence=evidence,
                    preservation_sha256={str(p): sha(p) for p in preserved},
                    runtime_executable_sha256=sha(legacy.PYTHON.resolve()),
                     python=str(legacy.PYTHON), snapshot=str(legacy.SNAPSHOT), max_decisions=1200,
                     child_timeout_s=180, frame_skip=4, warmup_ticks=50, raw_fps=50,
                     resource_plan=dict(execution='serial CPU21; one child; no GPU',
                     historical_episode_wall_s=20.63518266240135,
                     historical_receipt='runs/koi-collision-recovery-v1/1-3184000002-r0-crossing_projection.process.json',
                     wall_budget_note='180s per episode allows >8x observed baseline wall; twelve slots; preserve any timeout, no automatic retry.',
                     host_available_gib_observed=104, disk_available_gib_observed=111,
                     cgroup_path='/', cgroup_max_bytes=129567293440,
                     cgroup_current_bytes_observed=91952590848,
                     cgroup_oom_events_observed=0, cgroup_pressure_avg10_observed=0.,
                     resource_gate_note='Measured headroom, not official quotas or universal floors; no new memory/disk hard threshold. Compact serial observer workload, no learner/checkpoints.'),
                     telemetry_observer_only=True, metric_definitions=dict(legacy.METRIC_DEFINITIONS,
                     shield='active means changed action; baseline action independently tapped before shield; no-op/pedals exact; encounter budget reconstructed from observed-clear diagnostics, not trusting reported encounter counters. Unresolved threat dropout inhibits rearm until reset, never retains control.',
                    duration='Actual post-pre decision time, not assumed four ticks at terminal.',
                    state='Full logged observer state: physical pose, velocities, wheel actuator/dynamics, road contacts, reward/damage counters; not a serialization of Box2D solver internals.',
                    efficiency='Whole-episode metrics include DNFs; only mutually finished paths/laps support efficiency comparisons.'))
    save(output / 'protocol.json', protocol)
    return sha(output / 'protocol.json')


def validate_frozen(output, protocol_sha256):
    output = Path(output).resolve()
    if sha(output / 'protocol.json') != protocol_sha256:
        raise ValueError('protocol hash mismatch')
    protocol = read_json(output / 'protocol.json')
    if (protocol['schema'] != SCHEMA or protocol['cells'] != [list(c) for c in CELLS]
            or protocol['arms'] != list(ARMS) or protocol['schedule'] != scheduled_slots()
            or protocol['subgroups'] != SUBGROUPS or protocol['gate'] != GATE
            or protocol['comparator_zip_sha256'] != legacy.CROSSING_SHA
            or protocol['fresh'] is not False or protocol['official_action'] is not False):
        raise ValueError('fixed shield protocol contract changed')
    if sha(protocol['comparator_zip']) != legacy.CROSSING_SHA:
        raise ValueError('frozen comparator ZIP changed')
    if sha(Path(protocol['python']).resolve()) != protocol['runtime_executable_sha256']:
        raise ValueError('CPU21 executable changed')
    for group in ('source_sha256', 'environment_sha256', 'model_source_sha256', 'preservation_sha256'):
        if not protocol[group]:
            raise ValueError('empty source inventory')
        for path, expected in protocol[group].items():
            if sha(path) != expected:
                raise ValueError(f'frozen source changed: {path}')
    for path, expected in protocol['consumed_evidence']['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'consumed evidence changed: {path}')
    if sha(output / protocol['wrapper_file']) != protocol['wrapper_sha256']:
        raise ValueError('wrapper hash mismatch')
    return protocol


class BaselineProbe:
    """Observer-only tap, so no-op parity does not trust candidate self-reporting."""

    def __init__(self, driver):
        self._driver = driver
        self.action = None
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self._driver, name)

    def act(self, observation):
        action = self._driver.act(observation)
        self.action = [float(value) for value in action]
        self.calls += 1
        return action


def make_observer(*args):
    observer: Any = legacy.make_observer(*args)
    original_state = observer.state

    def state():
        value: dict = original_state()
        raw, env = observer.unwrapped, observer.environment
        bodies = [raw.car.hull, *raw.car.wheels]
        value['dynamic_state'] = [dict(position=list(map(float, b.position)), angle=float(b.angle),
                                      velocity=list(map(float, b.linearVelocity)), angular_velocity=float(b.angularVelocity),
                                      awake=bool(b.awake)) for b in bodies]
        value['wheel_state'] = [dict(gas=float(w.gas), brake=float(w.brake), steer=float(w.steer),
                                    omega=float(w.omega), phase=float(w.phase),
                                    joint_angle=float(w.joint.angle), joint_speed=float(w.joint.speed),
                                    motor_speed=float(w.joint.motorSpeed)) for w in raw.car.wheels]
        value['environment_state'] = dict(off_track_counter=int(env.off_track_counter),
                                          damage=float(env.damage.damage), reward=float(raw.reward),
                                          prev_reward=float(raw.prev_reward), tile_visited_count=int(raw.tile_visited_count),
                                          fuel_spent=float(raw.car.fuel_spent))
        return value

    observer.state = state
    original_step = observer.step

    def step(action):
        result = original_step(action)
        observer.last = dict(observer.last, post_observation_sha256=hashlib.sha256(result[0].tobytes()).hexdigest())
        return result

    observer.step = step
    return observer


def worker(output, index, protocol_sha256, *, import_only=False):
    """Isolated CPU21 process. Observer telemetry never enters model.act."""
    from importlib import import_module
    output = Path(output).resolve()
    protocol = validate_frozen(output, protocol_sha256)
    if not 0 <= index < len(protocol['schedule']):
        raise ValueError('worker slot outside fixed schedule')
    slot = protocol['schedule'][index]
    sys.path.insert(0, str(output / 'model'))
    entry = import_module('agent')
    if not Path(str(entry.__file__)).resolve().is_relative_to(output / 'model'):
        raise ValueError('baseline imported outside frozen ZIP')
    model = entry.Agent()
    probe = None
    if slot['mode'] == ARMS[1]:
        spec = importlib.util.spec_from_file_location('collision_shield_frozen', output / protocol['wrapper_file'])
        if spec is None or spec.loader is None:
            raise ValueError('cannot load frozen shield')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if module.MAX_INTERVENTION_ACTIONS != 6 or module.REARM_CLEAR_ACTIONS != 3:
            raise ValueError('shield encounter budget differs from protocol')
        probe = BaselineProbe(model.driver)
        model = module.CollisionShieldAgent(probe)
    import haic_agent
    haic_agent.__path__.append(str(legacy.SNAPSHOT / 'haic_agent'))
    sys.path.insert(0, str(legacy.SNAPSHOT))
    from scripts.evaluate_koi_minimum_clearance_ab import footprint_measurements
    import cv2
    import numpy as np
    import torch
    from training.env_factory import create_training_environment
    run_episode = import_module('training.evaluate_closed_loop').run_episode
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(slot['seed'])
    if not torch.__version__.startswith('2.1.') or '+cpu' not in torch.__version__:
        raise ValueError('CPU21 runtime required')
    if import_only:
        print(json.dumps(dict(import_only=True, environment_resets=0, model_type=type(model).__name__)))
        return
    holder = {}
    episode_path = output / slot['file']
    raw_path, decision_path = episode_path.with_suffix('.raw.jsonl'), episode_path.with_suffix('.decisions.jsonl')

    class MeasuredAgent:
        def reset(self, observation):
            return model.reset(observation)

        def act(self, observation):
            self.observation_sha256 = hashlib.sha256(observation.tobytes()).hexdigest()
            before = probe.calls if probe else 0
            self.action = np.asarray(model.act(observation)).tolist()
            if probe and probe.calls != before + 1:
                raise ValueError('shield must call frozen baseline exactly once per decision')
            return self.action

        def last_step_diagnostics(self):
            info = dict(shield=copy.deepcopy(model.last_shield) if probe else None,
                        baseline_probe_action=copy.deepcopy(probe.action) if probe else self.action,
                        evaluation_only=dict(holder['environment'].last,
                                             observation_sha256=self.observation_sha256))
            decision_stream.write(json.dumps(dict(step=holder['environment'].step_number,
                                                  action=self.action, controller=info), allow_nan=False) + '\n')
            return info

    with raw_path.open('x', buffering=1) as raw_stream, decision_path.open('x', buffering=1) as decision_stream:
        def factory(**kwargs):
            observer = make_observer(create_training_environment(**kwargs), raw_stream,
                                     output / 'reset-ledger.jsonl', slot, protocol_sha256, footprint_measurements)
            holder['environment'] = observer
            return observer
        result = run_episode(mode=slot['mode'], track_id=slot['track_id'], seed=slot['seed'],
                             agent=MeasuredAgent(), max_decisions=protocol['max_decisions'],
                             plan_budget_seconds=4.5, capture_trace=True, fail_on_invalid_action=True,
                             environment_factory=factory)
    observer = holder.get('environment')
    initial = getattr(observer, 'initial', None)
    states = ([initial] if initial else []) + [json.loads(line) for line in raw_path.read_text().splitlines()]
    result.update(repeat=0, scope=legacy.SCOPE, protocol_sha256=protocol_sha256,
                  catalog=getattr(observer, 'catalog', None), initial_state=initial,
                  geometry_sha256=getattr(observer, 'geometry_sha256', None),
                  initial_observation_sha256=getattr(observer, 'initial_observation_sha256', None),
                  raw_trace_file=raw_path.name, raw_trace_sha256=sha(raw_path),
                  decision_stream_file=decision_path.name, decision_stream_sha256=sha(decision_path),
                  off_track_count_max=getattr(observer, 'max_counter', None),
                  physical=legacy.physical_metrics(states, getattr(observer, 'obstacles', [])),
                  versions=dict(python=sys.version, numpy=np.__version__, cv2=getattr(cv2, '__version__'), torch=torch.__version__))
    save(episode_path, result)
    legacy.append_jsonl(output / 'reset-ledger.jsonl', dict(slot, event='episode_end',
                       protocol_sha256=protocol_sha256, retire_reason=result['retire_reason'], error=result['error']))


def worker_command(output, index, protocol_sha256, python, *, import_only=False):
    return [python, '-I', '-B', '-c',
            'import sys; sys.path.insert(0,sys.argv[1]); from scripts.evaluate_koi_collision_shield import worker; worker(sys.argv[2],int(sys.argv[3]),sys.argv[4],import_only=sys.argv[5]=="True")',
            str(Path(output) / 'source'), str(output), str(index), protocol_sha256, str(import_only)]


def preflight(output, protocol_sha256):
    """Import/construct both frozen arms in isolated processes, with no env factory."""
    output = Path(output).resolve()
    protocol = validate_frozen(output, protocol_sha256)
    receipts = []
    for index in (0, 1):
        command = worker_command(output, index, protocol_sha256, protocol['python'], import_only=True)
        child = subprocess.run(command, cwd=output / 'model', env=legacy.ENV,
                               capture_output=True, text=True, timeout=30)
        if child.returncode:
            raise RuntimeError(f'import-only preflight failed: {child.stderr}')
        receipts.append(json.loads(child.stdout))
    if any(r.get('environment_resets') != 0 or r.get('import_only') is not True for r in receipts):
        raise ValueError('invalid zero-reset preflight receipt')
    return dict(protocol_sha256=protocol_sha256, environment_resets=0, arms=receipts)


def run(output, protocol_sha256):
    output = Path(output).resolve()
    protocol = validate_frozen(output, protocol_sha256)
    if sha(Path(__file__)) != sha(output / 'source/scripts/evaluate_koi_collision_shield.py'):
        raise ValueError('run operator differs from frozen source')
    for slot in protocol['schedule']:
        path = output / slot['file']
        if any(path.with_suffix(suffix).exists() for suffix in ('.json', '.raw.jsonl', '.decisions.jsonl', '.process.json')):
            raise ValueError('existing slot evidence; never resume or overwrite')
    if (output / 'reset-ledger.jsonl').exists() or (output / 'episode-report.json').exists():
        raise ValueError('existing study evidence; never resume')
    save(output / 'run-start.json', dict(protocol_sha256=protocol_sha256, time_ns=time.time_ns()))
    rows, error = copy.deepcopy(protocol['schedule']), None
    try:
        for index, row in enumerate(rows):
            row['status'] = 'started'
            command = worker_command(output, index, protocol_sha256, protocol['python'])
            started, child_error, stdout, stderr = time.monotonic(), None, '', ''
            path = output / row['file']
            try:
                child = subprocess.run(command, cwd=output / 'model', env=legacy.ENV,
                                       capture_output=True, text=True, timeout=protocol['child_timeout_s'])
                stdout, stderr = child.stdout, child.stderr
                if child.returncode:
                    child_error = f'child_exit_{child.returncode}'
            except subprocess.TimeoutExpired as failure:
                stdout, stderr = failure.stdout or '', failure.stderr or ''
                child_error = 'child_timeout'
            except BaseException as failure:
                child_error = f'{type(failure).__name__}: {failure}'
                raise
            finally:
                for suffix, text in (('.stdout.txt', stdout), ('.stderr.txt', stderr)):
                    with path.with_suffix(suffix).open('x') as stream:
                        stream.write(text.decode(errors='replace') if isinstance(text, bytes) else text)
                process = path.with_suffix('.process.json')
                save(process, dict(command=command, protocol_sha256=protocol_sha256,
                                   wall_time_s=time.monotonic() - started, error=child_error))
                row['process_sha256'] = sha(process)
            if child_error:
                raise RuntimeError(child_error)
            episode = read_json(path)
            if not valid_episode(episode):
                raise ValueError('invalid/capped episode; retain partial evidence and stop')
            row.update(status='completed', sha256=sha(path))
            print(json.dumps(dict(row, finished=episode['completed'], damage=episode['damage'],
                                  collisions=episode['collisions'])), flush=True)
    except BaseException as failure:
        error = f'{type(failure).__name__}: {failure}'
        for row in rows:
            if row['status'] == 'started':
                row.update(status='operator_error', error=error)
        raise
    finally:
        save(output / 'episode-report.json', dict(protocol_sha256=protocol_sha256,
                                                  scope=legacy.SCOPE, rows=rows, operator_error=error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'validate', 'preflight', 'run'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if args.command == 'freeze':
        print(freeze_protocol(args.output))
    elif not args.protocol_sha256:
        parser.error('--protocol-sha256 is required for validate/run')
    elif args.command == 'validate':
        validate_frozen(args.output, args.protocol_sha256)
        print(json.dumps(dict(valid=True, environment_resets=0)))
    elif args.command == 'preflight':
        print(json.dumps(preflight(args.output, args.protocol_sha256)))
    else:
        run(args.output, args.protocol_sha256)


if __name__ == '__main__':
    main()
