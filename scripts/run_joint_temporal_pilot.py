"""Exclusive six-episode consumed-TRAIN pilot. Only ``execute`` can reset.

Freeze is file-only; preflight imports isolated frozen sources without constructing
an environment. Analysis never calls a policy, predictor, or simulator. Physical
telemetry is written by the existing audit observer, never passed into the policy.
"""

import argparse
import ast
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path('/tmp/kilo/koi-baseline-c4e224d/snapshots/haic-local-20260930/workspace')
PYTHON = Path('/tmp/kilo/haic-cpu21/bin/python')
BUNDLE = ROOT / 'submissions/20261002-crossing-projection-collision-shield-v1-baseline'
CHAMPION_SHA = 'c9e376a049805a8669af9c3959e09f3f96565530a5a4feb631977237ed64f801'
SCHEMA = 'haic-joint-temporal-pilot-v1'
OPERATOR = 'scripts/run_joint_temporal_pilot.py'
SUCCESSOR = 'haic/algorithms/joint_control/successor.py'
CELLS = ((1, 3184000013), (3, 3184000002), (2, 3184000006))
ARMS = ('baseline', 'successor')
LIMITS = dict(max_resets=6, max_decisions=1200, total_seconds=3600,
              child_seconds=600, rss_bytes=1024**3, address_space_bytes=4 * 1024**3,
              memory_reserve_bytes=256 * 1024**2, output_bytes=1024**3,
              disk_reserve_bytes=128 * 1024**2)
THREAD_ENV = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                  SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy',
                  PYGAME_HIDE_SUPPORT_PROMPT='1', PYTHONDONTWRITEBYTECODE='1')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def json_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def local_path(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('path escapes frozen directory')
    return path


def check_pins(pins):
    if not isinstance(pins, dict) or not pins:
        raise ValueError('nonempty targeted source/evidence pins required')
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('source/evidence hash differs: ' + path)


def schedule() -> list[dict[str, Any]]:
    return [dict(track_id=t, seed=s, mode=a, repeat=0, file=f'{t}-{s}-{a}.json')
            for t, s in CELLS for a in ARMS]


def contract():
    return dict(schema=SCHEMA, cells=[list(c) for c in CELLS], arms=list(ARMS),
                schedule=schedule(), limits=LIMITS, fresh=False, partition='TRAIN',
                official_action=False, champion_zip_sha256=CHAMPION_SHA,
                python=str(PYTHON), snapshot=str(SNAPSHOT),
                environment=dict(frame_skip=4, frame_stack=4, warmup_ticks=50, raw_fps=50,
                                 reset='unchanged seeded track reset', physics='unchanged',
                                 factory='training.env_factory.create_training_environment'),
                execution='serial full natural episodes, no retry/replacement; accidents do not stop cohort',
                memory_justification='Prior temporal collectors peaked below400MB; 1GiB RSS/4GiB AS with 256MiB additional reserve.',
                forecast_qualification='Exact actual issued H4 sequence and 16 raw ticks required; continuation mismatch is NOT model range failure.')


def source_closure(root, entries, *, packages=None, exclude=()):
    """Static local Python import closure only, not a repository inventory audit."""
    root = Path(root).resolve()
    pending, found = [Path(p) for p in entries], set()

    def enqueue(module):
        parts = module.split('.') if module else []
        if not parts or parts[0] in exclude or (packages is not None and parts[0] not in packages):
            return
        for n in range(1, len(parts) + 1):
            package = Path(*parts[:n]) / '__init__.py'
            if (root / package).is_file() and package not in found:
                pending.append(package)
        for candidate in (Path(*parts).with_suffix('.py') if parts else Path('__init__.py'),
                          Path(*parts) / '__init__.py'):
            if (root / candidate).is_file() and candidate not in found:
                pending.append(candidate)

    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        path = local_path(root, relative)
        tree = ast.parse(path.read_text(), filename=str(path))
        found.add(relative)
        package = relative.parent.parts
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    enqueue(name.name)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    base = '.'.join((*package[:len(package) - node.level + 1], base)).rstrip('.')
                enqueue(base)
                for name in node.names:
                    if name.name != '*':
                        enqueue('.'.join(filter(None, (base, name.name))))
        for n in range(1, len(package) + 1):
            init = Path(*package[:n]) / '__init__.py'
            if (root / init).is_file() and init not in found:
                pending.append(init)
    return {str(p): sha(root / p) for p in sorted(found)}


def champion_members(bundle=BUNDLE):
    bundle = Path(bundle)
    if sha(bundle / 'submission.zip') != CHAMPION_SHA:
        raise ValueError('exact champion ZIP required')
    with zipfile.ZipFile(bundle / 'submission.zip') as archive:
        names = archive.namelist()
        if len(names) != 11 or len(set(names)) != 11:
            raise ValueError('exact eleven champion members required')
        members = {}
        for name in names:
            source = local_path(bundle / 'source', name)
            data = archive.read(name)
            if source.read_bytes() != data:
                raise ValueError('frozen champion source differs from ZIP: ' + name)
            members[name] = data
    return members


def validate_claim(claim, output, calibration_sha256, *, root=ROOT):
    expected = dict(schema='haic-joint-temporal-pilot-consumed-train-v1', partition='TRAIN',
                    fresh=False, cells=[list(c) for c in CELLS], max_resets=6,
                    run_directory=str(Path(output).resolve()), calibration_sha256=calibration_sha256)
    if any(claim.get(k) != v for k, v in expected.items()):
        raise ValueError('exact immutable consumed-TRAIN reuse claim required')
    for key in ('source_sha256', 'evidence_sha256'):
        check_pins(claim.get(key))
    if not {str(Path(root) / OPERATOR), str(Path(root) / SUCCESSOR)} <= set(claim['source_sha256']):
        raise ValueError('claim must bind operator and successor sources')


def validate_calibration(value, root):
    if value.get('schema') != 'haic-joint-temporal-interval-calibration-v1':
        raise ValueError('source-bound interval calibration required')
    pins = value.get('source_pins')
    check_pins(pins)
    required = {str(Path(root) / 'haic/algorithms/joint_control' / (name + '.py'))
                for name in ('hud', 'motion', 'physics', 'observer', 'comparison', 'interval_comparison')}
    if not required <= set(pins):
        raise ValueError('calibration must bind the exact runtime predictor/scorer sources')


def freeze(output, calibration, claim, *, environment_sources=()):
    output, calibration, claim = map(lambda p: Path(p).resolve(), (output, calibration, claim))
    if output.exists() or not output.parent.is_dir():
        raise ValueError('new output directory in an existing parent required')
    calibration_data = read_json(calibration)
    validate_calibration(calibration_data, ROOT)
    validate_claim(read_json(claim), output, sha(calibration))
    members = champion_members()
    sources = source_closure(ROOT, [OPERATOR, SUCCESSOR,
        'scripts/evaluate_koi_collision_shield.py', 'scripts/evaluate_koi_minimum_clearance_ab.py'],
        packages=('scripts', 'haic'))
    environment = source_closure(SNAPSHOT, ['training/evaluate_closed_loop.py',
        'training/env_factory.py', *environment_sources], exclude=('agent',))
    output.mkdir()
    for name, expected in sources.items():
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if sha(target) != expected:
            raise ValueError('source changed while freezing')
    for name, data in members.items():
        target = local_path(output / 'model', name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    for source, name in ((calibration, 'calibration.json'), (claim, 'claim.json'),
                         (BUNDLE / 'submission.zip', 'champion.zip')):
        shutil.copyfile(source, output / name)
    protocol = dict(contract(), repository_root=str(ROOT), source_sha256=sources,
        environment_sha256={str(SNAPSHOT / p): h for p, h in environment.items()},
        model_sha256={n: hashlib.sha256(b).hexdigest() for n, b in members.items()},
        preserved_sha256={str(BUNDLE / 'source' / n): hashlib.sha256(b).hexdigest() for n, b in members.items()},
        champion_zip=str(BUNDLE / 'submission.zip'), calibration_path=str(calibration),
        calibration_sha256=sha(calibration), claim_path=str(claim), claim_sha256=sha(claim),
        runtime_sha256=sha(PYTHON.resolve()))
    save(output / 'protocol.json', protocol)
    digest = sha(output / 'protocol.json')
    validate_frozen(output, digest)
    return digest


def validate_frozen(output, digest):
    output = Path(output).resolve()
    if sha(output / 'protocol.json') != digest:
        raise ValueError('protocol hash differs')
    p = read_json(output / 'protocol.json')
    if any(p.get(k) != v for k, v in contract().items()):
        raise ValueError('fixed six-episode protocol differs')
    for name, expected in p['source_sha256'].items():
        for base in (output / 'source', Path(p['repository_root'])):
            if sha(local_path(base, name)) != expected:
                raise ValueError('targeted source differs: ' + name)
    check_pins(p['environment_sha256'])
    check_pins(p['preserved_sha256'])
    for name, expected in p['model_sha256'].items():
        if sha(local_path(output / 'model', name)) != expected:
            raise ValueError('extracted champion source differs')
    for name, original, expected in (('champion.zip', p['champion_zip'], CHAMPION_SHA),
            ('calibration.json', p['calibration_path'], p['calibration_sha256']),
            ('claim.json', p['claim_path'], p['claim_sha256'])):
        if sha(output / name) != expected or sha(original) != expected:
            raise ValueError('bound ' + name + ' differs')
    validate_claim(read_json(output / 'claim.json'), output, p['calibration_sha256'], root=p['repository_root'])
    validate_calibration(read_json(output / 'calibration.json'), p['repository_root'])
    if sha(PYTHON.resolve()) != p['runtime_sha256']:
        raise ValueError('CPU21 executable differs')
    return p


def validate_admission(value, digest, protocol, *, now=None, starting=False):
    now = time.time() if now is None else now
    required = dict(schema='haic-joint-temporal-pilot-admission-v1', admitted=True,
                    protocol_sha256=digest, claim_sha256=protocol['claim_sha256'],
                    calibration_sha256=protocol['calibration_sha256'])
    if any(value.get(k) != v for k, v in required.items()):
        raise ValueError('current source-bound run admission required')
    created, expires = value.get('created_unix_s'), value.get('expires_unix_s')
    if (type(created) not in (int, float) or type(expires) not in (int, float)
            or not math.isfinite(created) or not math.isfinite(expires)
            or not created <= now < expires <= created + LIMITS['total_seconds']
            or (starting and now - created > 300)):
        raise ValueError('run admission expired or not current')


def memory_sample():
    status = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            name, value, _ = line.split()
            status[name[:-1]] = int(value) * 1024
    if set(status) != {'VmRSS', 'VmHWM'}:
        raise RuntimeError('process memory telemetry missing')
    if max(status.values()) > LIMITS['rss_bytes']:
        raise MemoryError('1GiB process RSS cap exceeded')
    return status


def resource_admission(output, *, child=False):
    from scripts.evaluate_koi_minimum_clearance_ab import resource_measurements
    measure: dict[str, Any] = resource_measurements(output)
    process = memory_sample()
    additional = max(0, LIMITS['rss_bytes'] - (process['VmRSS'] if child else 0))
    needed = additional + LIMITS['memory_reserve_bytes']
    headrooms = [measure['host_mem_available_bytes']]
    headrooms += [r['raw_headroom_bytes'] for r in measure['visible_cgroup_ancestors']
                  if r['raw_headroom_bytes'] is not None]
    used = sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file())
    remaining = max(0, LIMITS['output_bytes'] - used) + LIMITS['disk_reserve_bytes']
    if min(headrooms) < needed or min(d['free_bytes'] for d in measure['disk']) < remaining:
        raise RuntimeError('insufficient measured incremental memory/disk headroom')
    if used > LIMITS['output_bytes']:
        raise RuntimeError('1GiB pilot output cap exceeded')
    return dict(measurement=measure, process=process, needed_memory_bytes=needed, output_bytes=used)


def worker_command(output, index, digest, *, import_only=False):
    return [str(PYTHON), '-I', '-B', '-c',
        'import sys; sys.path.insert(0,sys.argv[1]); '
        'from scripts.run_joint_temporal_pilot import worker; '
        'worker(sys.argv[2],int(sys.argv[3]),sys.argv[4],import_only=sys.argv[5]=="True")',
        str(Path(output).resolve() / 'source'), str(Path(output).resolve()), str(index), digest, str(import_only)]


def load_runtime(output, protocol, mode, seed):
    if any(n in sys.modules for n in ('agent', 'haic_agent', 'training')):
        raise ValueError('fresh isolated worker module namespace required')
    sys.path.insert(0, str(output / 'model'))
    champion = importlib.import_module('agent').Agent()
    if Path(str(sys.modules['agent'].__file__)).resolve() != output / 'model/agent.py':
        raise ValueError('champion imported outside frozen package')
    import haic_agent
    haic_agent.__path__.append(str(SNAPSHOT / 'haic_agent'))
    sys.path.insert(0, str(SNAPSHOT))
    # Keep new policy/reused audit helpers in the frozen source namespace.
    sys.path.insert(0, str(output / 'source'))
    import cv2
    import numpy as np
    import torch
    from training.env_factory import create_training_environment
    evaluator = importlib.import_module('training.evaluate_closed_loop')
    if (Path(str(evaluator.__file__)).resolve() != SNAPSHOT / 'training/evaluate_closed_loop.py'
            or 'fail_on_invalid_action' not in inspect.signature(evaluator.run_episode).parameters):
        raise ValueError('exact external strict evaluator required, not root evaluator')
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    if (torch.__version__, np.__version__, getattr(cv2, '__version__')) != ('2.1.0+cpu', '1.26.0', '4.8.1'):
        raise ValueError('pinned CPU21 dependency versions required')
    if mode == 'successor':
        from haic.algorithms.joint_control.successor import TemporalSuccessor
        model = TemporalSuccessor(champion, read_json(output / 'calibration.json'))
    else:
        model = champion
    from scripts import evaluate_koi_collision_shield as audit
    from scripts.evaluate_koi_minimum_clearance_ab import footprint_measurements
    pins = dict(protocol['environment_sha256'])
    pins.update({str(output / 'source' / n): h for n, h in protocol['source_sha256'].items()})
    pins.update({str(output / 'model' / n): h for n, h in protocol['model_sha256'].items()})
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(SNAPSHOT) or path.is_relative_to(output):
                if str(path) not in pins or sha(path) != pins[str(path)]:
                    raise ValueError('import outside targeted frozen closure: ' + str(path))
    return model, evaluator.run_episode, create_training_environment, audit, footprint_measurements


class MeasuredAgent:
    """The authoritative clocks surround only the complete model.act call."""

    def __init__(self, model, stream, holder, mode):
        self.model, self.stream, self.holder, self.mode = model, stream, holder, mode
        self.calls = 0
        self.prefix = True

    def reset(self, observation):
        return self.model.reset(observation)

    def emit(self, value):
        self.stream.write(json.dumps(value, allow_nan=False) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def act(self, observation):
        self.calls += 1
        self.pixels = hashlib.sha256(observation.tobytes()).hexdigest()
        reference = self.holder.get('reference_decisions')
        if self.prefix and reference is not None and self.calls <= len(reference):
            prior = reference[self.calls - 1]
            if (prior['observation_sha256'] != self.pixels
                    or prior['evaluation_only']['pre'] != self.holder['environment'].state()):
                raise ValueError('pre-divergence state/image mismatch')
        self.emit(dict(event='act_intent', step=self.calls))
        cpu, wall = time.process_time(), time.perf_counter()
        error = None
        try:
            action = self.model.act(observation)
        except BaseException as failure:
            error = f'{type(failure).__name__}: {failure}'
            raise
        finally:
            # Capture both clocks before hashing, serialization, audit, or display work.
            elapsed_wall, elapsed_cpu = time.perf_counter() - wall, time.process_time() - cpu
            self.timing = dict(cpu_seconds=elapsed_cpu, wall_seconds=elapsed_wall)
            self.emit(dict(event='act', step=self.calls, **self.timing, error=error))
        memory_sample()
        self.action = [float(v) for v in action]
        if self.prefix and reference is not None:
            self.prefix = self.calls <= len(reference) and self.action == reference[self.calls - 1]['action']
        return action

    def last_step_diagnostics(self):
        policy = copy.deepcopy(self.model.last_diagnostics) if self.mode == 'successor' else {}
        telemetry = copy.deepcopy(self.holder['environment'].last)
        reference = self.holder.get('reference_decisions')
        if self.prefix and reference is not None and telemetry != reference[self.calls - 1]['evaluation_only']:
            raise ValueError('unchanged action prefix physical/image mismatch')
        row = dict(event='decision', step=self.calls, action=self.action,
                   observation_sha256=self.pixels, timing=self.timing,
                   policy=policy, evaluation_only=telemetry)
        self.emit(row)
        # Avoid duplicating large forecast arrays in the evaluator's episode trace.
        return dict(intervention=bool(policy.get('intervention')), eligible=bool(policy.get('eligible')))


def natural_episode(result):
    return (result.get('error') is None and result.get('invalid_actions') == 0
            and isinstance(result.get('completed'), bool)
            and (result['completed'] or result.get('retire_reason') in ('off_track', 'crash', 'max_steps'))
            and result.get('damage') is not None and result.get('collisions') is not None
            and not (result.get('retire_reason') == 'max_steps' and result.get('steps') == LIMITS['max_decisions'])
            and 0 < result.get('steps', 0) <= LIMITS['max_decisions'])


def initial_record(observer):
    # The simulator's track uses tuples; the baseline receipt necessarily uses
    # JSON arrays. Compare the same lossless serialization on both arms.
    value = dict(catalog=observer.catalog, geometry_sha256=observer.geometry_sha256,
                 initial_state=observer.initial, initial_observation_sha256=observer.initial_observation_sha256)
    return json.loads(json.dumps(value, allow_nan=False))


def worker(output, index, digest, *, import_only=False):
    output = Path(output).resolve()
    p = validate_frozen(output, digest)
    if not 0 <= index < 6:
        raise ValueError('worker outside six-slot schedule')
    slot = p['schedule'][index]
    resource.setrlimit(resource.RLIMIT_AS, (LIMITS['address_space_bytes'], LIMITS['address_space_bytes']))
    signal.alarm(LIMITS['child_seconds'])
    model, run_episode, create_environment, audit, footprint = load_runtime(output, p, slot['mode'], slot['seed'])
    if import_only:
        print(json.dumps(dict(mode=slot['mode'], import_only=True, environment_resets=0,
                             evaluator_file=str(SNAPSHOT / 'training/evaluate_closed_loop.py'),
                             memory=memory_sample())))
        return
    start = read_json(output / 'run-start.json')
    if start.get('protocol_sha256') != digest or start.get('preflight_sha256') != sha(output / 'preflight.json'):
        raise ValueError('bound preflight/run-start required')
    if start.get('admission_sha256') != sha(output / 'admission.json'):
        raise ValueError('admission changed')
    validate_preflight(read_json(output / 'preflight.json'), digest)
    validate_admission(read_json(output / 'admission.json'), digest, p)
    ledger = output / 'reset-ledger.jsonl'
    if ledger.exists() and any(r['file'] == slot['file'] for r in json_rows(ledger)):
        raise ValueError('existing reset evidence; never retry')
    path = output / slot['file']
    raw_path, decision_path = path.with_suffix('.raw.jsonl'), path.with_suffix('.decisions.jsonl')
    holder = {}
    baseline = None
    if slot['mode'] == 'successor':
        baseline = read_json(output / p['schedule'][index - 1]['file'])
        if not natural_episode(baseline) or baseline.get('protocol_sha256') != digest:
            raise ValueError('completed bound baseline required before successor')
        holder['reference_decisions'] = decisions(output, baseline)
    with raw_path.open('x', buffering=1) as raw, decision_path.open('x', buffering=1) as decision_stream:
        def factory(**kwargs):
            if kwargs != dict(track_id=slot['track_id'], seed=slot['seed'], max_decisions=1200) or 'environment' in holder:
                raise ValueError('one exact unchanged environment per child required')
            validate_frozen(output, digest)
            validate_admission(read_json(output / 'admission.json'), digest, p)
            append(output / 'resources.jsonl', dict(slot=index, phase='before_environment', **resource_admission(output, child=True)))
            observer = audit.make_observer(create_environment(**kwargs), raw, ledger, slot, digest, footprint)
            raw_reset = observer.unwrapped.reset

            def guarded_reset(*args, **reset_kwargs):
                if observer.reset_count:
                    raise ValueError('one reset per child; never retry')
                validate_frozen(output, digest)
                validate_admission(read_json(output / 'admission.json'), digest, p)
                append(output / 'resources.jsonl', dict(slot=index, phase='before_reset', **resource_admission(output, child=True)))
                return raw_reset(*args, **reset_kwargs)

            observer.unwrapped.reset = guarded_reset
            reset = observer.reset

            def checked_initial_reset():
                result = reset()
                initial = initial_record(observer)
                save(path.with_suffix('.initial.json'), dict(protocol_sha256=digest, **initial))
                if baseline is not None and any(baseline.get(k) != v for k, v in initial.items()):
                    raise ValueError('static track/initial state/image parity differs')
                return result

            observer.reset = checked_initial_reset
            holder['environment'] = observer
            return observer

        result = run_episode(mode=slot['mode'], track_id=slot['track_id'], seed=slot['seed'],
            agent=MeasuredAgent(model, decision_stream, holder, slot['mode']), max_decisions=1200,
            plan_budget_seconds=4.5, capture_trace=True, fail_on_invalid_action=True, environment_factory=factory)
    observer = holder.get('environment')
    initial = getattr(observer, 'initial', None)
    states = ([initial] if initial is not None else []) + json_rows(raw_path)
    result.update(protocol_sha256=digest, scope='consumed TRAIN, not official or fresh',
        catalog=getattr(observer, 'catalog', None), initial_state=initial,
        geometry_sha256=getattr(observer, 'geometry_sha256', None),
        initial_observation_sha256=getattr(observer, 'initial_observation_sha256', None),
        raw_trace_file=raw_path.name, raw_trace_sha256=sha(raw_path),
        decision_stream_file=decision_path.name, decision_stream_sha256=sha(decision_path),
        off_track_count_max=getattr(observer, 'max_counter', None),
        physical=audit.legacy.physical_metrics(states, getattr(observer, 'obstacles', [])),
        memory=memory_sample(), natural_episode=natural_episode(result))
    save(path, result)
    append(ledger, dict(slot, event='episode_end', protocol_sha256=digest,
                       natural_episode=result['natural_episode'], error=result['error']))


def validate_preflight(receipt, digest):
    rows = receipt.get('arms', [])
    if (receipt.get('protocol_sha256') != digest or receipt.get('environment_resets') != 0
            or len(rows) != 2 or {r.get('mode') for r in rows} != set(ARMS)
            or any(r.get('import_only') is not True or r.get('environment_resets') != 0
                   or r.get('evaluator_file') != str(SNAPSHOT / 'training/evaluate_closed_loop.py') for r in rows)):
        raise ValueError('two bound zero-reset arm preflights required')


def preflight(output, digest):
    output = Path(output).resolve()
    validate_frozen(output, digest)
    if (output / 'preflight.json').exists():
        raise ValueError('preflight already exists')
    rows = []
    for index in (0, 1):
        child = subprocess.run(worker_command(output, index, digest, import_only=True),
            cwd=output / 'model', env={**os.environ, **THREAD_ENV}, capture_output=True, text=True, timeout=60)
        if child.returncode:
            raise RuntimeError(child.stderr)
        rows.append(json.loads(child.stdout))
    result = dict(protocol_sha256=digest, environment_resets=0, arms=rows)
    validate_preflight(result, digest)
    save(output / 'preflight.json', result)
    return result


def execute(output, digest, admission):
    output = Path(output).resolve()
    p = validate_frozen(output, digest)
    validate_preflight(read_json(output / 'preflight.json'), digest)
    value = read_json(admission)
    validate_admission(value, digest, p, starting=True)
    for name in ('run-start.json', 'episode-report.json', 'reset-ledger.jsonl', 'admission.json'):
        if (output / name).exists():
            raise ValueError('existing execution evidence; never restart')
    if any(list(output.glob(Path(r['file']).stem + '.*')) for r in schedule()):
        raise ValueError('existing slot evidence; never overwrite')
    save(output / 'admission.json', value)
    save(output / 'run-start.json', dict(protocol_sha256=digest, time_ns=time.time_ns(),
        preflight_sha256=sha(output / 'preflight.json'), admission_sha256=sha(output / 'admission.json')))
    rows = [dict(r, status='unrun') for r in schedule()]
    started, error = time.monotonic(), None
    try:
        for index, row in enumerate(rows):
            validate_frozen(output, digest)
            validate_admission(value, digest, p)
            append(output / 'resources.jsonl', dict(slot=index, phase='before_child', **resource_admission(output)))
            remaining = min(LIMITS['total_seconds'] - (time.monotonic() - started), value['expires_unix_s'] - time.time())
            if remaining <= 0:
                raise TimeoutError('fixed total wall deadline')
            row['status'] = 'started'
            path = output / row['file']
            child_start, child_error, returncode = time.monotonic(), None, None
            try:
                with path.with_suffix('.stdout.log').open('x') as stdout, path.with_suffix('.stderr.log').open('x') as stderr:
                    child = subprocess.run(worker_command(output, index, digest), cwd=output / 'model',
                        env={**os.environ, **THREAD_ENV}, stdout=stdout, stderr=stderr,
                        timeout=min(LIMITS['child_seconds'], remaining))
                returncode = child.returncode
                if returncode:
                    raise RuntimeError(f'worker exited {returncode}')
                episode = read_json(path)
                row['sha256'] = sha(path)
                if not natural_episode(episode):
                    raise RuntimeError('invalid/resource/execution episode: ' + str(episode.get('retire_reason')))
                row['status'] = 'completed'
                # Pair parity is a provenance requirement, not a performance gate.
                if row['mode'] == 'successor':
                    left = read_json(output / rows[index - 1]['file'])
                    parity = pair_parity(left, episode, output)
                    if not parity['exact_prefix']:
                        raise RuntimeError('static/initial/raw/decision prefix provenance mismatch')
                    row['parity'] = parity
            except BaseException as failure:
                row['status'] = 'failed'
                child_error = f'{type(failure).__name__}: {failure}'
                raise
            finally:
                receipt = path.with_suffix('.process.json')
                save(receipt, dict(protocol_sha256=digest, slot=index, returncode=returncode,
                                   wall_seconds=time.monotonic() - child_start, error=child_error))
                row.update(process_file=receipt.name, process_sha256=sha(receipt))
    except BaseException as failure:
        error = f'{type(failure).__name__}: {failure}'
    finally:
        artifacts = {path.name: sha(path) for row in rows
                     for path in output.glob(Path(row['file']).stem + '.*') if path.is_file()}
        report = dict(protocol_sha256=digest, rows=rows, operator_error=error,
                      wall_seconds=time.monotonic() - started,
                      artifacts_sha256=artifacts,
                      reset_ledger_sha256=sha(output / 'reset-ledger.jsonl') if (output / 'reset-ledger.jsonl').exists() else None)
        save(output / 'episode-report.json', report)
    return report


def decisions(output, episode):
    path = local_path(output, episode['decision_stream_file'])
    if sha(path) != episode['decision_stream_sha256']:
        raise ValueError('decision stream differs')
    return [r for r in json_rows(path) if r['event'] == 'decision']


def raw_states(output, episode):
    path = local_path(output, episode['raw_trace_file'])
    if sha(path) != episode['raw_trace_sha256']:
        raise ValueError('raw stream differs')
    return json_rows(path)


def pair_parity(left, right, output):
    a, b = decisions(output, left), decisions(output, right)
    shared = min(len(a), len(b))
    first = next((i for i in range(shared) if a[i]['action'] != b[i]['action']), None)
    prefix = shared if first is None else first
    before_equal = lambda i: (a[i]['observation_sha256'] == b[i]['observation_sha256']
        and a[i]['evaluation_only']['pre'] == b[i]['evaluation_only']['pre'])
    state_equal = all(before_equal(i) and a[i]['evaluation_only'] == b[i]['evaluation_only'] for i in range(prefix))
    raw_a = [r for r in raw_states(output, left) if r['step'] <= prefix]
    raw_b = [r for r in raw_states(output, right) if r['step'] <= prefix]
    initial = all(left.get(k) is not None and left[k] == right.get(k) for k in
                  ('catalog', 'geometry_sha256', 'initial_state', 'initial_observation_sha256'))
    divergent_pre = first is None or before_equal(first)
    return dict(exact_prefix=initial and state_equal and divergent_pre and raw_a == raw_b,
                static_and_initial_equal=initial, prefix_decisions=prefix,
                first_changed_step=None if first is None else a[first]['step'],
                first_changed_pre_equal=divergent_pre, raw_prefix_equal=raw_a == raw_b,
                all_actions_equal=first is None and len(a) == len(b))


def distribution(values) -> dict[str, Any]:
    ordered = sorted(values)
    if any(not math.isfinite(v) or v < 0 for v in ordered):
        raise ValueError('invalid timing value')
    def percentile(q):
        if not ordered:
            return None
        index = (len(ordered) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return dict(n=len(ordered), p50=percentile(.5), p95=percentile(.95), p99=percentile(.99),
                max=ordered[-1] if ordered else None, over_1s_count=sum(v > 1 for v in ordered))


def rate(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                rate=numerator / denominator if denominator else None)


def forecast_ranges(rows, raw) -> dict[str, Any]:
    """Forecast misses only for exact planned/issued H4 continuations, never accidents."""
    counts = dict(no_forecast=0, censored=0, continuation_mismatch=0, qualified=0, range_miss=0)
    events = []
    by_step = {}
    for state in raw:
        by_step.setdefault(state['step'], []).append(state)
    for index, row in enumerate(rows):
        forecast = row.get('policy', {}).get('forecast')
        if not forecast:
            counts['no_forecast'] += 1
            continue
        future = rows[index:index + 4]
        if len(future) < 4:
            counts['censored'] += 1
            continue
        if [r['action'] for r in future] != forecast['actions']:
            counts['continuation_mismatch'] += 1
            events.append(dict(step=row['step'], status='continuation_mismatch'))
            continue
        states = [s for r in future for s in by_step.get(r['step'], [])]
        if len(states) != 16 or any(len(by_step.get(r['step'], [])) != 4 for r in future):
            counts['censored'] += 1
            continue
        start = row['evaluation_only']['pre']
        if any(abs(s['t'] - start['t'] - .02 * (i + 1)) > 1e-7 for i, s in enumerate(states)):
            counts['censored'] += 1
            continue
        c, s = math.cos(start['yaw']), math.sin(start['yaw'])
        actual = [[0., 0., 0.]] + [[c * (p['x'] - start['x']) + s * (p['y'] - start['y']),
            -s * (p['x'] - start['x']) + c * (p['y'] - start['y']),
            (p['yaw'] - start['yaw'] + math.pi) % (2 * math.pi) - math.pi] for p in states]
        poses, pos_bound, yaw_bound = forecast['poses'], forecast['position_residual'], forecast['yaw_residual']
        if (not poses or len(pos_bound) != 17 or len(yaw_bound) != 17
                or any(len(scene) != 17 or any(len(p) != 3 for p in scene) for scene in poses)
                or any(not math.isfinite(v) or v < 0 for v in pos_bound + yaw_bound)
                or any(not math.isfinite(v) for scene in poses for p in scene for v in p)):
            raise ValueError('invalid forecast range evidence')
        contained = any(all(math.hypot(p[0] - a[0], p[1] - a[1]) <= pb
            and abs((p[2] - a[2] + math.pi) % (2 * math.pi) - math.pi) <= yb
            for p, a, pb, yb in zip(scene, actual, pos_bound, yaw_bound)) for scene in poses)
        counts['qualified'] += 1
        counts['range_miss'] += int(not contained)
        events.append(dict(step=row['step'], t=start['t'], status='contained' if contained else 'range_miss'))
    return dict(counts=counts, range_miss_rate=rate(counts['range_miss'], counts['qualified']), events=events,
                interpretation='Same scenario for entire H4; exact issued continuation only. Range misses are not accidents. No offline predictor replay.')


def failure_events(rows, raw, episode) -> dict[str, Any]:
    interventions = [r for r in rows if r.get('policy', {}).get('intervention')]
    initial = episode.get('initial_state') or {}
    events, contacts, offroad, collision = [], set(initial.get('contacts', [])), False, False
    offroad = bool(initial) and not any(initial.get('wheel_road_contacts', []))
    for kind, present in (('contact_at_reset', bool(contacts)), ('all_wheels_offroad_at_reset', offroad)):
        if present:
            events.append(dict(kind=kind, detail=sorted(contacts) if kind == 'contact_at_reset' else None,
                               step=0, t=initial['t'], left_censored=True))
    for state in raw:
        touched = set(state['contacts'])
        kinds: list[tuple[str, Any]] = [('contact', sorted(touched - contacts))] if touched - contacts else []
        now_offroad = not any(state['wheel_road_contacts'])
        if now_offroad and not offroad:
            kinds.append(('all_wheels_offroad', None))
        now_collision = bool(state.get('collision'))
        if now_collision and not collision:
            kinds.append(('collision_onset', None))
        for kind, detail in kinds:
            events.append(dict(kind=kind, detail=detail, step=state['step'], t=state['t']))
        contacts, offroad, collision = touched, now_offroad, now_collision
    previous_damage = initial.get('environment_state', {}).get('damage', 0.)
    for row in rows:
        post = row['evaluation_only']['post']
        damage = post.get('environment_state', {}).get('damage', previous_damage)
        if damage > previous_damage:
            events.append(dict(kind='damage_increase', detail=damage - previous_damage,
                               step=row['step'], t=post['t']))
        previous_damage = damage
    if episode.get('completed') is False and raw:
        events.append(dict(kind='dnf', detail=episode.get('retire_reason'), step=raw[-1]['step'], t=raw[-1]['t']))
    events.sort(key=lambda e: (e['t'], e['step'], e['kind']))
    for event in events:
        before = [r for r in interventions if r['step'] <= event['step']
                  and r['evaluation_only']['pre']['t'] <= event['t']]
        last = before[-1] if before else None
        event.update(after_intervention=last is not None,
            last_intervention_step=last['step'] if last else None,
            decision_distance=event['step'] - last['step'] if last else None,
            seconds_after_intervention=event['t'] - last['evaluation_only']['pre']['t'] if last else None)
    return dict(events=events, after_intervention=rate(sum(e['after_intervention'] for e in events), len(events)),
                interpretation='Temporal association only; not proof that an intervention caused a failure.')


def analyze(output, digest):
    output = Path(output).resolve()
    p = validate_frozen(output, digest)
    report = read_json(output / 'episode-report.json')
    if report.get('protocol_sha256') != digest or len(report['rows']) != 6:
        raise ValueError('bound complete six-slot report required, including unrun/failed slots')
    for name, expected in report['artifacts_sha256'].items():
        if sha(local_path(output, name)) != expected:
            raise ValueError('sealed complete/partial evidence differs: ' + name)
    ledger_path = output / 'reset-ledger.jsonl'
    ledger = json_rows(ledger_path) if ledger_path.exists() else []
    if (sha(ledger_path) if ledger_path.exists() else None) != report['reset_ledger_sha256']:
        raise ValueError('reset ledger differs')
    intents = [r for r in ledger if r['event'] == 'reset_intent']
    if len(intents) > 6 or len({r['file'] for r in intents}) != len(intents):
        raise ValueError('duplicate/excess reset evidence')
    for entry in ledger:
        if (entry.get('protocol_sha256') != digest or entry.get('file') not in {s['file'] for s in schedule()}
                or entry.get('event') not in ('reset_intent', 'episode_end')):
            raise ValueError('unexpected reset ledger identity/event')
    records, loaded, latency = [], {}, {a: [] for a in ARMS}
    for slot, row in zip(p['schedule'], report['rows']):
        if any(row.get(k) != v for k, v in slot.items()):
            raise ValueError('report schedule differs')
        path = output / row['file']
        if row.get('process_file'):
            receipt_path = local_path(output, row['process_file'])
            if sha(receipt_path) != row['process_sha256'] or read_json(receipt_path)['protocol_sha256'] != digest:
                raise ValueError('process receipt differs')
        if row['status'] == 'completed' and (
                sum(r['file'] == row['file'] for r in intents) != 1
                or sum(r['file'] == row['file'] and r['event'] == 'episode_end' for r in ledger) != 1):
            raise ValueError('completed episode requires exactly one reset/end')
        if not path.exists():
            partial_rows, truncated_tail = [], False
            stream = path.with_suffix('.decisions.jsonl')
            if stream.exists():
                lines = stream.read_text().splitlines()
                for index, line in enumerate(lines):
                    try:
                        partial_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        if index != len(lines) - 1:
                            raise ValueError('malformed nonterminal partial decision record')
                        truncated_tail = True
            calls = [r for r in partial_rows if r['event'] == 'act']
            ends = [r for r in partial_rows if r['event'] == 'decision']
            intervention = sum(bool(r.get('policy', {}).get('intervention')) for r in ends)
            eligible = sum(bool(r.get('policy', {}).get('eligible')) for r in ends)
            latency[row['mode']].extend(dict(r, file=row['file']) for r in calls)
            records.append(dict(slot=slot, status=row['status'], partial=True,
                logged_act_calls=len(calls), logged_decision_ends=len(ends), decisions=len(ends),
                intervention=rate(intervention, len(ends)), eligible=rate(eligible, len(ends)),
                abstain=rate(len(ends) - intervention, len(ends)) if row['mode'] == 'successor' else None,
                act_timing_censored=len({r['step'] for r in partial_rows if r['event'] == 'act_intent'}
                    - {r['step'] for r in calls}),
                truncated_decision_tail=truncated_tail))
            continue
        episode = read_json(path)
        if row.get('sha256') is not None and sha(path) != row['sha256']:
            raise ValueError('episode hash differs')
        if episode.get('protocol_sha256') != digest:
            raise ValueError('episode protocol differs')
        loaded[row['file']] = episode
        rows, raw = decisions(output, episode), raw_states(output, episode)
        calls = [r for r in json_rows(output / episode['decision_stream_file']) if r['event'] == 'act']
        latency[row['mode']].extend(dict(r, file=row['file']) for r in calls)
        policies = [r['policy'] for r in rows]
        n = len(rows)
        if natural_episode(episode) and n != episode['steps']:
            raise ValueError('natural episode/decision denominator differs')
        intervention = sum(bool(r.get('intervention')) for r in policies)
        eligible = sum(bool(r.get('eligible')) for r in policies)
        records.append(dict(slot=slot, status=row['status'], partial=not natural_episode(episode),
            completed=episode['completed'], lap_time_ms=episode['lapTimeMs'], progress=episode['progress'],
            retire_reason=episode['retire_reason'], damage=episode['damage'],
            collision_positive_decisions=episode['collisions'], physical=episode['physical'],
            decisions=n, intervention=rate(intervention, n), eligible=rate(eligible, n),
            abstain=rate(n - intervention, n) if row['mode'] == 'successor' else None,
            abstain_reasons={reason: rate(sum(reason in r.get('reasons', []) for r in policies), n)
                            for reason in sorted({reason for r in policies for reason in r.get('reasons', [])})},
            intervention_given_eligible=rate(intervention, eligible),
            forecast=forecast_ranges(rows, raw), failures=failure_events(rows, raw, episode), memory=episode['memory']))
    pairs = []
    for index in (0, 2, 4):
        left, right = (loaded.get(p['schedule'][i]['file']) for i in (index, index + 1))
        if left is None or right is None:
            pairs.append(dict(cell=list(CELLS[index // 2]), matched=False))
            continue
        both = left['completed'] and right['completed']
        parity = pair_parity(left, right, output)
        matched = (natural_episode(left) and natural_episode(right) and parity['exact_prefix']
                   and all(report['rows'][i]['status'] == 'completed' for i in (index, index + 1)))
        changed = sum(bool(r.get('policy', {}).get('intervention')) for r in decisions(output, right))
        pairs.append(dict(cell=list(CELLS[index // 2]), matched=matched,
            successor_effect_evaluated=matched and changed > 0,
            parity=parity, baseline_finished=left['completed'],
            successor_finished=right['completed'], both_finished=both,
            lap_delta_ms=right['lapTimeMs'] - left['lapTimeMs'] if both and matched else None))
    timings = {arm: dict(cpu_seconds=distribution([r['cpu_seconds'] for r in calls]),
        wall_seconds=distribution([r['wall_seconds'] for r in calls]),
        all_over_1s=[r for r in calls if r['cpu_seconds'] > 1 or r['wall_seconds'] > 1]) for arm, calls in latency.items()}
    by_arm = {}
    for arm in ARMS:
        observed = [r for r in records if r['slot']['mode'] == arm and 'completed' in r]
        logged = [r for r in records if r['slot']['mode'] == arm and 'decisions' in r]
        finished = sum(r['completed'] for r in observed)
        by_arm[arm] = dict(allocated_episodes=3, observed_episodes=len(observed),
            natural_episodes=sum(not r['partial'] for r in observed), completion=rate(finished, len(observed)),
            missing_episodes=3 - len(observed),
            finished_lap_times_ms=[r['lap_time_ms'] for r in observed if r['completed']],
            progress=[r['progress'] for r in observed],
            logged_completed_decision_ends=sum(r['decisions'] for r in logged),
            intervention=rate(sum(r['intervention']['numerator'] for r in logged), sum(r['decisions'] for r in logged)),
            abstain=rate(sum(r['abstain']['numerator'] for r in logged if r['abstain'] is not None),
                         sum(r['decisions'] for r in logged)) if arm == 'successor' else None,
            damage_sum=sum(r['damage'] for r in observed if r['damage'] is not None),
            collision_positive_decisions_sum=sum(r['collision_positive_decisions'] for r in observed if r['collision_positive_decisions'] is not None),
            physical_contact_events_sum=sum(r['physical'].get('physical_contact_events', 0) for r in observed))
    result = dict(schema=SCHEMA + '-analysis', protocol_sha256=digest,
        episode_report_sha256=sha(output / 'episode-report.json'), reset_intents=len(intents),
        complete_six_natural_episodes=all(r['status'] == 'completed' for r in report['rows'])
            and len(loaded) == 6 and all(natural_episode(e) for e in loaded.values()),
        successor_effect_evaluated=by_arm['successor']['intervention']['numerator'] > 0,
        operator_error=report['operator_error'], episodes=records, pairs=pairs, by_arm=by_arm, integrated_act=timings,
        qualification='Three predeclared consumed TRAIN cells, not fresh/official/generalization evidence; no promotion gate. All accidents and DNFs retained.')
    save(output / 'summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('freeze', 'preflight', 'execute', 'analyze'):
        sub = commands.add_parser(command)
        sub.add_argument('--output', type=Path, required=True)
        if command == 'freeze':
            sub.add_argument('--calibration', type=Path, required=True)
            sub.add_argument('--claim', type=Path, required=True)
            sub.add_argument('--environment-source', action='append', default=[])
        else:
            sub.add_argument('--protocol-sha256', required=True)
        if command == 'execute':
            sub.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        result = dict(protocol_sha256=freeze(args.output, args.calibration, args.claim,
                                             environment_sources=args.environment_source), environment_resets=0)
    elif args.command == 'execute':
        result = execute(args.output, args.protocol_sha256, args.admission)
    else:
        result = globals()[args.command](args.output, args.protocol_sha256)
    print(json.dumps(result, allow_nan=False, indent=2))
    if args.command == 'execute' and result['operator_error'] is not None:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
