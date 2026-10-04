"""Separate consumed-TRAIN nominal A/B. Only `run` constructs/reset environments."""

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import zipfile

from scripts import evaluate_koi_collision_shield as shield


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'submissions/20261002-crossing-projection-collision-shield-v1-baseline'
BASELINE_SHA = 'c9e376a049805a8669af9c3959e09f3f96565530a5a4feb631977237ed64f801'
SHIELD_SHA = 'ad772bde9a9c3f4596fdfc742e7cac33f1b605ff52a2b3f76fbd4b75d6361d96'
SCHEMA = 'koi-nominal-trajectory-v1'
ARMS = ('frozen_shield', 'nominal_trajectory')
CELLS = (*shield.CELLS, (2, 3184000015), (3, 3184000015))
SUBGROUPS = {f'{t}:{s}': ('safety_regressions' if i < 4 else 'ordinary_controls')
             for i, (t, s) in enumerate(CELLS)}
GATE = dict(max_kept_lap_increase_ms=20, mean_lap_reduction_ms=20,
            ordinary_max_lateral_fraction=-.03, ordinary_path_fraction=-.001,
            ordinary_steering_integral_fraction=-.03,
            improving_ordinary_geometry_seeds=2, no_lost_finish=True,
            no_per_cell_damage_collision_increase=True, no_new_hit=True,
            complete_matched_evidence=True, baseline_window_return_coverage=True,
            steering_variation_nonincrease=True, return_censor_bound_nonincrease=True)
METRICS = dict(window='station-25 to station+25; interpolated XY/lateral endpoints; continuous forward/no seam; all baseline-eligible windows retained',
               steering='Final issued command held over actual pre/post interval, including partial terminal holds; integral command-s; variation at command jumps strictly inside window, no artificial endpoint zero jumps',
               return_definition='Valid physical rear-clear; abs lateral<=1m sustained .24s, confirmed before min(clear+2s,next greater-station obstacle entry,episode end); paired common horizon; retain censors/pending runs; bound is descriptive, not KM/RMST',
               weighting='Per-object relative changes averaged within each ordinary cell, then equal cells; directions also require both ordinary geometry means negative; zero baseline gets no reduction credit',
               safety='All six objects/cell, including unpassed; touching contact OR sampled clearance<=0; unassociated collision evidence rejects; all natural DNFs retained',
               scope='Eight outcome-selected consumed TRAIN layouts/five roads; ordinary four layouts/two roads; not fresh/confirmation/blind/official')
HELPERS = tuple('scripts/' + name for name in (
    '__init__.py', 'evaluate_koi_nominal_trajectory.py', 'analyze_koi_nominal_trajectory.py',
    'evaluate_koi_collision_shield.py', 'analyze_koi_collision_shield.py',
    'evaluate_koi_collision_priority.py', 'analyze_koi_collision_priority.py',
    'evaluate_koi_minimum_clearance_ab.py', 'evaluate_koi_adaptive_ab.py',
    'package_koi_adaptive_avoidance.py', 'analyze_koi_minimum_clearance_ab.py',
    'analyze_koi_adaptive_ab.py', 'analyze_koi_steering_release_ab.py')) + (
    'haic/__init__.py', 'haic/algorithms/__init__.py', 'haic/algorithms/koi/__init__.py',
    'haic/algorithms/koi/nominal_trajectory.py', 'haic/algorithms/koi/minimum_clearance.py',
    'haic/algorithms/koi/collision_shield.py', 'haic/algorithms/koi/trajectory_rollout.py')
sha, read_json, save, local_path = shield.sha, shield.read_json, shield.save, shield.local_path


def scheduled_slots():
    rows = []
    for track, seed in CELLS:
        arms = ARMS if (track + seed) % 2 else ARMS[::-1]
        rows.extend(dict(track_id=track, seed=seed, mode=arm, repeat=0,
                         file=f'{track}-{seed}-r0-{arm}.json', status='unrun') for arm in arms)
    return rows


def valid_episode(episode):
    return (shield.valid_episode(episode) and episode.get('damage_telemetry_valid') is True
            and episode.get('collision_telemetry_valid') is True
            and type(episode.get('damage')) in (int, float) and episode['damage'] >= 0
            and type(episode.get('collisions')) is int and episode['collisions'] >= 0
            and episode.get('steps') == len(episode['decision_trace']))


def consumed_evidence():
    rows = shield.legacy.validate_cells(list(CELLS), baseline='crossing_projection')
    pins = {row['file']: row['sha256'] for row in rows}
    prior = shield.legacy.PRIOR
    for name in ('protocol.json', 'episode-report.json'):
        pins[str(prior / name)] = sha(prior / name)
    directory = ROOT / 'runs/koi-collision-shield-v1'
    report = read_json(directory / 'episode-report.json')
    if report['operator_error'] is not None or report['protocol_sha256'] != sha(directory / 'protocol.json'):
        raise ValueError('completed prior shield report required')
    found = set()
    for row in report['rows']:
        cell = row['track_id'], row['seed']
        if row['mode'] != 'collision_shield':
            continue
        if cell not in shield.CELLS or cell in found or row['status'] != 'completed':
            raise ValueError('prior shield cell coverage differs')
        found.add(cell)
        path = local_path(directory, row['file'])
        episode = read_json(path)
        if sha(path) != row['sha256'] or not valid_episode(episode):
            raise ValueError('invalid consumed shield episode')
        pins[str(path)] = row['sha256']
    if found != set(shield.CELLS):
        raise ValueError('six consumed shield episodes required')
    for name in ('protocol.json', 'episode-report.json'):
        pins[str(directory / name)] = sha(directory / name)
    return pins


def freeze_protocol(output):
    output = Path(output).resolve()
    evidence = consumed_evidence()
    manifest = read_json(BUNDLE / 'manifest.json')
    if manifest['zip_sha256'] != BASELINE_SHA or sha(BUNDLE / 'submission.zip') != BASELINE_SHA:
        raise ValueError('exact frozen submission bundle required')
    if sha(ROOT / 'haic/algorithms/koi/collision_shield.py') != SHIELD_SHA:
        raise ValueError('unchanged v1 geometry helper required')
    environment = read_json(ROOT / 'runs/koi-collision-shield-v1/protocol.json')['environment_sha256']
    if len(environment) != 142 or any(sha(p) != h for p, h in environment.items()):
        raise ValueError('frozen environment142 differs')
    output.mkdir(exist_ok=False)
    sources = {}
    for relative in HELPERS:
        source, copied = ROOT / relative, output / 'source' / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copied)
        sources[relative] = dict(original=str(source), sha256=sha(copied))
    shutil.copyfile(BUNDLE / 'submission.zip', output / 'baseline.zip')
    model = {}
    with zipfile.ZipFile(output / 'baseline.zip') as archive:
        names = archive.namelist()
        if (len(names) != 11 or len(set(names)) != 11 or set(names) != set(manifest['submission_members'])
                or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names)):
            raise ValueError('unsafe or changed frozen bundle members')
        for name in names:
            expected = manifest['files_sha256']['source/' + name]
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('frozen bundle member differs')
            model[name] = expected
        archive.extractall(output / 'model')
    if model['haic_agent/collision_shield_runtime.py'] != SHIELD_SHA:
        raise ValueError('frozen shield source differs')
    protocol: dict[str, Any] = dict(schema=SCHEMA, scope=shield.legacy.SCOPE, fresh=False, official_action=False,
                    cells=[list(c) for c in CELLS], arms=list(ARMS), subgroups=SUBGROUPS,
                    gate=GATE, metric_definitions=METRICS, schedule=scheduled_slots(),
                    baseline_zip_sha256=BASELINE_SHA, shield_sha256=SHIELD_SHA,
                    source_inventory=sources, model_source_sha256=model,
                    environment_sha256=environment, consumed_evidence_sha256=evidence,
                    preservation_sha256={str(ROOT / 'agent.py'): sha(ROOT / 'agent.py')},
                    python=str(shield.legacy.PYTHON), snapshot=str(shield.legacy.SNAPSHOT),
                    runtime_executable_sha256=sha(shield.legacy.PYTHON.resolve()),
                    max_decisions=1200, child_timeout_s=180, frame_skip=4, warmup_ticks=50,
                     raw_fps=50, execution='serial isolated CPU21; one child; no GPU; no automatic retry')
    protocol['resource_plan'] = dict(
        measured_at_utc='2026-10-02T02:16:23Z',
        historical_12_episode_wall_s=281.381628, forecast_16_episode_base_wall_s=384.58,
        pure_planner_cpu21_calls=20, pure_planner_mean_s=.06592448791489006,
        pure_planner_max_s=.08582854783162475,
        candidate_cost_note='Pure supplied-geometry timing, not driving throughput;180s child timeout preserves expensive/failed episodes without retry.',
        comparable_12_episode_directory_bytes=94164075,
        forecast_remaining_output_bytes=502208400,
        output_forecast_note='4x comparable per-episode bytes x16/12, including candidate/observer growth; no weights/checkpoints/replay.',
        disk_available_gib_observed=110, host_available_gib_observed=104,
        cgroup_path='/', cgroup_max_bytes=129567293440,
        cgroup_current_bytes_observed=98356015104,
        raw_cgroup_headroom_bytes=31211278336, cgroup_oom_events_observed=0,
        cgroup_pressure_avg10_observed=0., gpu='unused',
        resource_gate_note='One compact CPU child; no universal RAM/disk floor or official local-training quota. Recheck headroom before run; preflight records worker import RSS.')
    save(output / 'protocol.json', protocol)
    return sha(output / 'protocol.json')


def validate_frozen(output, protocol_sha256):
    output = Path(output).resolve()
    if sha(output / 'protocol.json') != protocol_sha256:
        raise ValueError('protocol hash mismatch')
    protocol = read_json(output / 'protocol.json')
    contract = dict(schema=SCHEMA, cells=[list(c) for c in CELLS], arms=list(ARMS),
                    subgroups=SUBGROUPS, gate=GATE, metric_definitions=METRICS,
                    schedule=scheduled_slots(), baseline_zip_sha256=BASELINE_SHA,
                    shield_sha256=SHIELD_SHA, fresh=False, official_action=False,
                    max_decisions=1200, child_timeout_s=180, frame_skip=4, warmup_ticks=50, raw_fps=50,
                    python=str(shield.legacy.PYTHON), snapshot=str(shield.legacy.SNAPSHOT))
    if any(protocol.get(k) != v for k, v in contract.items()):
        raise ValueError('fixed nominal protocol contract differs')
    if set(protocol['source_inventory']) != set(HELPERS) or len(protocol['environment_sha256']) != 142:
        raise ValueError('source closure differs')
    for relative, pin in protocol['source_inventory'].items():
        if sha(pin['original']) != pin['sha256'] or sha(local_path(output / 'source', relative)) != pin['sha256']:
            raise ValueError('imported helper source differs: ' + relative)
    if protocol['source_inventory']['haic/algorithms/koi/collision_shield.py']['sha256'] != SHIELD_SHA:
        raise ValueError('geometry shield helper changed')
    for group in ('environment_sha256', 'consumed_evidence_sha256', 'preservation_sha256'):
        if not protocol[group] or any(sha(p) != h for p, h in protocol[group].items()):
            raise ValueError('frozen inventory differs: ' + group)
    if (sha(output / 'baseline.zip') != BASELINE_SHA
            or sha(Path(protocol['python']).resolve()) != protocol['runtime_executable_sha256']):
        raise ValueError('frozen model/runtime differs')
    with zipfile.ZipFile(output / 'baseline.zip') as archive:
        names = archive.namelist()
        if len(names) != 11 or set(names) != set(protocol['model_source_sha256']):
            raise ValueError('model inventory differs')
        for name, expected in protocol['model_source_sha256'].items():
            if (hashlib.sha256(archive.read(name)).hexdigest() != expected
                    or sha(local_path(output / 'model', name)) != expected):
                raise ValueError('model member differs')
    return protocol


class ActionProbe:
    """One independent nominal tap; never let v1 unwrap this adapter."""

    def __init__(self, nominal):
        self.nominal, self.action, self.calls = nominal, None, 0

    def __getattr__(self, name):
        if name == 'driver':
            raise AttributeError(name)
        return getattr(self.nominal, name)

    def act(self, observation):
        returned = self.nominal.act(observation)
        self.action = [float(v) for v in returned]
        self.calls += 1
        return returned

    def reset(self, observation=None):
        self.action, self.calls = None, 0
        return self.nominal.reset(observation)


def compose(model, mode, candidate_class=None):
    if mode not in ARMS or (mode == ARMS[1] and candidate_class is None):
        raise ValueError('known arm and candidate constructor required')
    original = ActionProbe(model.driver.driver)
    candidate = candidate_class(original) if mode == ARMS[1] and candidate_class is not None else None
    nominal = ActionProbe(candidate) if candidate is not None else original
    model.driver.driver = nominal
    return model, nominal, original, candidate


def measured_act(model, nominal, original, candidate, observation):
    before = nominal.calls, original.calls
    action = model.act(observation)
    if (nominal.calls, original.calls) != (before[0] + 1, before[1] + 1):
        raise ValueError('each nominal must be called exactly once')
    if candidate is not None:
        candidate.observe_executed_action(action)
    return action


def worker(output, index, protocol_sha256, *, import_only=False):
    output = Path(output).resolve()
    protocol = validate_frozen(output, protocol_sha256)
    if not 0 <= index < len(protocol['schedule']):
        raise ValueError('worker outside frozen schedule')
    slot = protocol['schedule'][index]
    sys.path.insert(0, str(output / 'model'))
    entry = importlib.import_module('agent')
    if Path(str(entry.__file__)).resolve() != output / 'model/agent.py':
        raise ValueError('Agent imported outside frozen bundle')
    from haic.algorithms.koi.nominal_trajectory import NominalTrajectoryAgent
    model, nominal, original, candidate = compose(entry.Agent(), slot['mode'], NominalTrajectoryAgent)
    import haic_agent
    haic_agent.__path__.append(str(Path(protocol['snapshot']) / 'haic_agent'))
    sys.path.insert(0, protocol['snapshot'])
    import cv2
    import numpy as np
    import torch
    from scripts.evaluate_koi_minimum_clearance_ab import footprint_measurements
    from training.env_factory import create_training_environment
    run_episode: Any = importlib.import_module('training.evaluate_closed_loop').run_episode
    imported_pins = dict(protocol['environment_sha256'])
    imported_pins.update({str(output / 'source' / p): pin['sha256'] for p, pin in protocol['source_inventory'].items()})
    imported_pins.update({str(output / 'model' / p): h for p, h in protocol['model_source_sha256'].items()})
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(output) or path.is_relative_to(protocol['snapshot']):
                if str(path) not in imported_pins or sha(path) != imported_pins[str(path)]:
                    raise ValueError('imported project/environment module outside frozen closure: ' + str(path))
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(slot['seed'])
    if torch.__version__ != '2.1.0+cpu' or np.__version__ != '1.26.0' or getattr(cv2, '__version__') != '4.8.1':
        raise ValueError('pinned CPU21 dependency versions required')
    if import_only:
        import resource
        print(json.dumps(dict(import_only=True, environment_resets=0, mode=slot['mode'],
                             peak_import_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)))
        return
    path = output / slot['file']
    raw_path, decisions_path = path.with_suffix('.raw.jsonl'), path.with_suffix('.decisions.jsonl')
    holder = {}

    class MeasuredAgent:
        def reset(self, observation):
            self.feedback_calls = 0
            return model.reset(observation)

        def act(self, observation):
            self.pixels = hashlib.sha256(observation.tobytes()).hexdigest()
            self.action = np.asarray(measured_act(model, nominal, original, candidate, observation)).tolist()
            self.feedback_calls += int(candidate is not None)
            return self.action

        def last_step_diagnostics(self):
            info = dict(nominal.nominal.last_step_diagnostics())
            info.update(shield=copy.deepcopy(model.driver.last_shield),
                        baseline_probe_action=copy.deepcopy(nominal.action),
                        crossing_probe_action=copy.deepcopy(original.action),
                        nominal_probe_calls=nominal.calls, crossing_probe_calls=original.calls,
                        executed_action_observations=self.feedback_calls,
                        evaluation_only=dict(holder['environment'].last, observation_sha256=self.pixels))
            decisions.write(json.dumps(dict(step=holder['environment'].step_number,
                                           action=self.action, controller=info), allow_nan=False) + '\n')
            return info

    with raw_path.open('x', buffering=1) as raw, decisions_path.open('x', buffering=1) as decisions:
        def factory(**kwargs):
            observer = shield.make_observer(create_training_environment(**kwargs), raw,
                output / 'reset-ledger.jsonl', slot, protocol_sha256, footprint_measurements)
            holder['environment'] = observer
            return observer
        result = run_episode(mode=slot['mode'], track_id=slot['track_id'], seed=slot['seed'],
                             agent=MeasuredAgent(), max_decisions=1200, plan_budget_seconds=4.5,
                             capture_trace=True, fail_on_invalid_action=True, environment_factory=factory)
    observer = holder['environment']
    states = [observer.initial, *[json.loads(line) for line in raw_path.read_text().splitlines()]]
    result.update(repeat=0, scope=protocol['scope'], protocol_sha256=protocol_sha256,
                  catalog=observer.catalog, initial_state=observer.initial, geometry_sha256=observer.geometry_sha256,
                  initial_observation_sha256=observer.initial_observation_sha256,
                  raw_trace_file=raw_path.name, raw_trace_sha256=sha(raw_path),
                  decision_stream_file=decisions_path.name, decision_stream_sha256=sha(decisions_path),
                  off_track_count_max=observer.max_counter,
                  physical=shield.legacy.physical_metrics(states, observer.obstacles),
                  versions=dict(python=sys.version, numpy=np.__version__, cv2=getattr(cv2, '__version__'), torch=torch.__version__))
    save(path, result)
    shield.legacy.append_jsonl(output / 'reset-ledger.jsonl', dict(slot, event='episode_end',
        protocol_sha256=protocol_sha256, retire_reason=result['retire_reason'], error=result['error']))


def worker_command(output, index, protocol_sha256, python, *, import_only=False):
    return [python, '-I', '-B', '-c',
            'import sys; sys.path.insert(0,sys.argv[1]); from scripts.evaluate_koi_nominal_trajectory import worker; worker(sys.argv[2],int(sys.argv[3]),sys.argv[4],import_only=sys.argv[5]=="True")',
            str(Path(output) / 'source'), str(output), str(index), protocol_sha256, str(import_only)]


def preflight(output, protocol_sha256):
    output = Path(output).resolve()
    protocol = validate_frozen(output, protocol_sha256)
    receipts = []
    for index in (0, 1):
        child = subprocess.run(worker_command(output, index, protocol_sha256, protocol['python'], import_only=True),
                               cwd=output / 'model', env=shield.legacy.ENV, capture_output=True, text=True, timeout=30)
        if child.returncode:
            raise RuntimeError(child.stderr)
        receipt = json.loads(child.stdout)
        if receipt.get('import_only') is not True or receipt.get('environment_resets') != 0:
            raise ValueError('zero-reset preflight receipt differs')
        receipts.append(receipt)
    return dict(protocol_sha256=protocol_sha256, environment_resets=0, arms=receipts)


def run(output, protocol_sha256):
    output = Path(output).resolve()
    protocol = validate_frozen(output, protocol_sha256)
    if sha(__file__) != protocol['source_inventory']['scripts/evaluate_koi_nominal_trajectory.py']['sha256']:
        raise ValueError('operator differs from frozen source')
    if (output / 'run-start.json').exists() or (output / 'episode-report.json').exists() or (output / 'reset-ledger.jsonl').exists():
        raise ValueError('existing study evidence; never restart')
    if any(list(output.glob(Path(row['file']).stem + '.*')) for row in protocol['schedule']):
        raise ValueError('existing slot evidence; never overwrite')
    save(output / 'run-start.json', dict(protocol_sha256=protocol_sha256, time_ns=time.time_ns()))
    rows, error = copy.deepcopy(protocol['schedule']), None
    try:
        for index, row in enumerate(rows):
            row['status'] = 'started'
            path = output / row['file']
            command = worker_command(output, index, protocol_sha256, protocol['python'])
            started, child_error, stdout, stderr = time.monotonic(), None, '', ''
            try:
                child = subprocess.run(command, cwd=output / 'model', env=shield.legacy.ENV,
                                       capture_output=True, text=True, timeout=180)
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
                    path.with_suffix(suffix).write_bytes(text if isinstance(text, bytes) else text.encode())
                process = path.with_suffix('.process.json')
                save(process, dict(command=command, protocol_sha256=protocol_sha256,
                                   wall_time_s=time.monotonic() - started, error=child_error))
                row['process_sha256'] = sha(process)
            if child_error:
                raise RuntimeError(child_error)
            if not valid_episode(read_json(path)):
                raise ValueError('invalid/capped episode; retain evidence and stop')
            row.update(status='completed', sha256=sha(path))
            print(json.dumps(row), flush=True)
    except BaseException as failure:
        error = f'{type(failure).__name__}: {failure}'
        for row in rows:
            if row['status'] == 'started':
                row.update(status='operator_error', error=error,
                           partial_artifacts=[dict(file=p.name, sha256=sha(p), bytes=p.stat().st_size)
                                              for p in sorted(output.glob(Path(row['file']).stem + '.*'))])
        raise
    finally:
        save(output / 'episode-report.json', dict(protocol_sha256=protocol_sha256, rows=rows, operator_error=error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'validate', 'preflight', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if args.command == 'freeze':
        print(freeze_protocol(args.output))
    elif not args.protocol_sha256:
        parser.error('--protocol-sha256 required')
    elif args.command == 'validate':
        validate_frozen(args.output, args.protocol_sha256)
        print(json.dumps(dict(valid=True, environment_resets=0)))
    elif args.command == 'preflight':
        print(json.dumps(preflight(args.output, args.protocol_sha256)))
    else:
        run(args.output, args.protocol_sha256)


if __name__ == '__main__':
    main()
