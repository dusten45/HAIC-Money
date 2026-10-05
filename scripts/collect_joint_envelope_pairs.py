"""Three bounded consumed-TRAIN pairs; only execute may construct/reset an env.

The first causal, cost-supported anchor is selected without consulting its sign,
absolute veto, or physical labels. The contemporary champion prefix is the replay
reference. Historical episodes bind consumption/catalog/initial pixels ONLY.
"""

import argparse
import ast
from collections import deque
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
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
import struct
import subprocess
import sys
import time
from types import FunctionType, MethodType
from typing import Any
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path('/tmp/kilo/koi-baseline-c4e224d/snapshots/haic-local-20260930/workspace')
PYTHON = Path('/tmp/kilo/haic-cpu21/bin/python')
BUNDLE = ROOT / 'submissions/20261002-crossing-projection-collision-shield-v1-baseline'
CHAMPION_SHA = 'c9e376a049805a8669af9c3959e09f3f96565530a5a4feb631977237ed64f801'
PRIOR_CALIBRATION = ROOT / 'runs/joint-temporal-interval-v1/calibration.json'
PRIOR_CALIBRATION_SHA = '5c8fe665e21688a88e286ded8771577d51b1c1849db2a744c91f1905cf3bf1b0'
SCHEMA = 'haic-joint-envelope-pairs-v1'
CALIBRATION_SCHEMA = 'haic-joint-temporal-paired-envelope-calibration-v1'
OPERATOR = 'scripts/collect_joint_envelope_pairs.py'
COMPARATOR = 'haic/algorithms/joint_control/paired_residual.py'
CELLS = ((1, 3184000003), (2, 3184000004), (3, 3184000005))
ARMS = ('baseline', 'repeat', 'alternative')
FIRST, LAST, CAP, HORIZON, SKIP, WARMUP = 18, 60, 64, 4, 4, 51
LIMITS = dict(max_resets=9, max_decisions=576, max_raw_ticks=2763,
              total_seconds=2400, child_seconds=600, rss_bytes=1024**3,
              address_space_bytes=4 * 1024**3, output_bytes=1024**3,
              memory_reserve_bytes=256 * 1024**2, disk_reserve_bytes=128 * 1024**2)
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


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def canonical_sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    """Exclusive, durable artifacts; counters alone use the old atomic journal."""
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(encoded(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def jsonable(value) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def local_path(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('path escapes artifact directory')
    return path


def check_pins(pins):
    if not isinstance(pins, dict) or not pins:
        raise ValueError('nonempty targeted source/evidence pins required')
    for name, digest in pins.items():
        path = Path(name)
        if not path.is_absolute() or path != path.resolve() or sha(path) != digest:
            raise ValueError('source/evidence differs: ' + name)


def source_closure(root, entries, *, packages=None, exclude=()):
    """Targeted AST import closure, adapted from run_joint_temporal_pilot."""
    root, pending, found = Path(root).resolve(), list(map(Path, entries)), set()

    def enqueue(module):
        parts = module.split('.') if module else []
        if not parts or parts[0] in exclude or (packages is not None and parts[0] not in packages):
            return
        for n in range(1, len(parts) + 1):
            for candidate in (Path(*parts[:n]) / '__init__.py', Path(*parts[:n]).with_suffix('.py')):
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


def schedule() -> list[dict[str, Any]]:
    return [dict(track_id=t, seed=s, arm=a, slot=f'{t}-{s}-{a}', decisions=CAP,
                 raw_ticks=WARMUP + SKIP * CAP) for t, s in CELLS for a in ARMS]


def contract() -> dict[str, Any]:
    return dict(schema=SCHEMA, cells=[list(c) for c in CELLS], arms=list(ARMS), schedule=schedule(),
        limits=LIMITS, partition='TRAIN', fresh=False, new_paired_starts=True, official_action=False,
        search_completed_prefix=[FIRST, LAST], environment_max_decisions=CAP,
        horizon_decisions=HORIZON, frame_skip=SKIP, warmup_raw=WARMUP,
        champion_zip_sha256=CHAMPION_SHA, python=str(PYTHON), snapshot=str(SNAPSHOT),
        selection='FIRST causal old-precondition + common full H4 cost support; no rank/veto/future filter',
        continuation='baseline/repeat 4*a0; alternative a1 then3*a0; float32; no suffix Agent queries',
        not_found='stop at completed prefix60, retain prefix, skip both replay arms; no replacement',
        parity='full accessible raw/images, NOT hidden Box2D solver equivalence',
        full_pilot='report-only conditional gate; never automatically run full episodes')


def validate_calibration(value, root=ROOT):
    prior = Path(root) / 'runs/joint-temporal-interval-v1/calibration.json'
    if sha(prior) != PRIOR_CALIBRATION_SHA:
        raise ValueError('original absolute calibration pin differs')
    if (value.get('schema') != CALIBRATION_SCHEMA or value.get('calibration_complete') is not True
            or value.get('absolute_calibration') != read_json(prior)):
        raise ValueError('new performance calibration must preserve ALL original absolute fields')
    absolute = value['absolute_calibration']
    check_pins(absolute.get('source_pins'))
    check_pins(value.get('source_pins'))
    check_pins(value.get('evidence_pins'))
    required = {str(Path(root) / 'haic/algorithms/joint_control' / (n + '.py'))
                for n in ('hud', 'motion', 'physics', 'observer', 'comparison', 'interval_comparison')}
    if not required <= set(absolute['source_pins']) or str(Path(root) / COMPARATOR) not in value['source_pins']:
        raise ValueError('calibration lacks exact physical/comparator source pins')
    for key, floor in (('position_residual', .5), ('yaw_residual', .05)):
        values = absolute[key]
        if (len(values) != 17 or values[0] != 0 or
                any(type(v) not in (int, float) or not math.isfinite(v) or v < floor for v in values[1:])):
            raise ValueError('invalid unchanged physical envelope')
    if absolute['paired_cost_residual'] != .44961874671412616:
        raise ValueError('old paired residual differs')
    for key in ('lower', 'upper'):
        v = value['performance'][key]
        if type(v) not in (int, float) or not math.isfinite(v) or v < .05:
            raise ValueError('new residual requires declared .05 engineering floor')
    fitted = set(absolute['CAL']) | set(absolute['HELD_OUT_FROM_FIT'])
    if fitted & {s for _, s in CELLS}:
        raise ValueError('new paired roads overlap prior joint fitting/evaluation split')


def validate_claim(claim, output, calibration_sha256, *, root=ROOT):
    expected = dict(schema='haic-joint-envelope-pairs-consumed-train-v1', partition='TRAIN',
        fresh=False, new_paired_starts=True, cells=[list(c) for c in CELLS], max_resets=9,
        max_decisions=576, max_raw_ticks=2763, run_directory=str(Path(output).resolve()),
        calibration_sha256=calibration_sha256)
    if any(claim.get(k) != v for k, v in expected.items()):
        raise ValueError('exact immutable consumed-TRAIN claim required')
    check_pins(claim.get('source_sha256'))
    check_pins(claim.get('evidence_sha256'))
    if not {str(Path(root) / OPERATOR), str(Path(root) / COMPARATOR)} <= set(claim['source_sha256']):
        raise ValueError('claim must bind operator and new comparator')
    references = claim.get('references', [])
    if [(r['track_id'], r['seed']) for r in references] != list(CELLS):
        raise ValueError('three ordered historical consumption references required')
    for r in references:
        if claim['evidence_sha256'].get(r['episode_path']) != r['episode_sha256']:
            raise ValueError('historical reference not pinned as consumption evidence')
        episode = read_json(r['episode_path'])
        if ((episode['track_id'], episode['seed']) != (r['track_id'], r['seed'])
                or episode.get('map_id') is not None or episode.get('obstacle_mode') != 'official'
                or episode.get('obstacle_count') != 6 or len(episode['catalog']['obstacles']) != 6
                or not episode.get('initial_observation_sha256')):
            raise ValueError('historical TRAIN profile/catalog unavailable')


def champion_members(bundle=BUNDLE):
    bundle = Path(bundle)
    if sha(bundle / 'submission.zip') != CHAMPION_SHA:
        raise ValueError('exact unchanged champion ZIP required')
    with zipfile.ZipFile(bundle / 'submission.zip') as archive:
        names = archive.namelist()
        if len(names) != 11 or len(set(names)) != 11:
            raise ValueError('exact eleven champion members required')
        members = {name: archive.read(name) for name in names}
    for name, data in members.items():
        if local_path(bundle / 'source', name).read_bytes() != data:
            raise ValueError('champion source differs from ZIP')
    return members


def freeze(output, calibration, claim, *, environment_sources=()):
    output, calibration, claim = [Path(p).resolve() for p in (output, calibration, claim)]
    if output.exists() or not output.parent.is_dir():
        raise ValueError('exclusive new output directory with existing parent required')
    validate_calibration(read_json(calibration))
    validate_claim(read_json(claim), output, sha(calibration))
    members = champion_members()
    # The legacy helper is dynamically compiled by the low-level capture module.
    # Its inventories/template/validate/collect/load_runtime are never called.
    sources = source_closure(ROOT, [OPERATOR, COMPARATOR, 'scripts/collect_joint_prediction_probe.py'],
                             packages=('scripts', 'haic'))
    environment = source_closure(SNAPSHOT, ['training/env_factory.py', 'training/evaluate_closed_loop.py',
                                           *environment_sources], exclude=('agent',))
    output.mkdir()
    for name, digest in sources.items():
        target = local_path(output / 'source', name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if sha(target) != digest:
            raise ValueError('source changed during freeze')
    for name, data in members.items():
        target = local_path(output / 'model', name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    for source, name in ((calibration, 'calibration.json'), (claim, 'claim.json'),
                         (BUNDLE / 'submission.zip', 'champion.zip')):
        shutil.copyfile(source, output / name)
    p = dict(contract(), repository_root=str(ROOT), output_directory=str(output),
        source_sha256=sources, environment_sha256={str(SNAPSHOT / n): h for n, h in environment.items()},
        model_sha256={n: hashlib.sha256(v).hexdigest() for n, v in members.items()},
        preserved_sha256={str(BUNDLE / 'source' / n): hashlib.sha256(v).hexdigest() for n, v in members.items()},
        calibration_path=str(calibration), calibration_sha256=sha(calibration),
        claim_path=str(claim), claim_sha256=sha(claim), champion_zip=str(BUNDLE / 'submission.zip'),
        runtime_sha256=sha(PYTHON.resolve()), cpu_affinity=sorted(os.sched_getaffinity(0)))
    save(output / 'protocol.json', p)
    digest = sha(output / 'protocol.json')
    validate_frozen(output, digest)
    return digest


def validate_frozen(output, digest):
    output = Path(output).resolve()
    if sha(output / 'protocol.json') != digest:
        raise ValueError('protocol hash differs')
    p = read_json(output / 'protocol.json')
    if any(p.get(k) != v for k, v in contract().items()) or p['output_directory'] != str(output):
        raise ValueError('fixed three-pair protocol differs')
    for name, expected in p['source_sha256'].items():
        for root in (output / 'source', Path(p['repository_root'])):
            if sha(local_path(root, name)) != expected:
                raise ValueError('targeted source differs: ' + name)
    check_pins(p['environment_sha256'])
    check_pins(p['preserved_sha256'])
    if len(p['model_sha256']) != 11:
        raise ValueError('eleven champion members required')
    for name, expected in p['model_sha256'].items():
        if sha(local_path(output / 'model', name)) != expected:
            raise ValueError('frozen champion member differs')
    for name, original, expected in (('champion.zip', p['champion_zip'], CHAMPION_SHA),
            ('calibration.json', p['calibration_path'], p['calibration_sha256']),
            ('claim.json', p['claim_path'], p['claim_sha256'])):
        if sha(output / name) != expected or sha(original) != expected:
            raise ValueError('bound artifact differs: ' + name)
    validate_calibration(read_json(output / 'calibration.json'), p['repository_root'])
    validate_claim(read_json(output / 'claim.json'), output, p['calibration_sha256'], root=p['repository_root'])
    if sha(PYTHON.resolve()) != p['runtime_sha256']:
        raise ValueError('CPU21 interpreter differs')
    return p


def validate_admission(value, digest, p, *, starting=False, now=None):
    now = time.time() if now is None else now
    expected = dict(schema='haic-joint-envelope-pairs-admission-v1', admitted=True,
        protocol_sha256=digest, claim_sha256=p['claim_sha256'], calibration_sha256=p['calibration_sha256'])
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('source-bound current admission required')
    created, expires = value.get('created_unix_s'), value.get('expires_unix_s')
    if (type(created) not in (int, float) or type(expires) not in (int, float)
            or not math.isfinite(created) or not math.isfinite(expires)
            or not created <= now < expires <= created + LIMITS['total_seconds']
            or (starting and now - created > 300)):
        raise ValueError('admission not current or expired')


def resource_admission(output):
    """Call in a fresh exec child, never use inherited getrusage high-water marks."""
    memory = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            key, value, _ = line.split()
            memory[key[:-1]] = int(value) * 1024
    if set(memory) != {'VmRSS', 'VmHWM'} or max(memory.values()) > LIMITS['rss_bytes']:
        raise MemoryError('actual process RSS/HWM exceeds 1GiB or is unavailable')
    info = {k: int(v.split()[0]) * 1024 for k, v in
            (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    membership = Path('/proc/self/cgroup').read_text().splitlines()
    relative = next(row.split('::', 1)[1] for row in membership if row.startswith('0::'))
    mount = Path('/sys/fs/cgroup')
    # A cgroup namespace can expose '/' even when the host membership is deeper.
    leaf = mount / relative.lstrip('/')
    if not (leaf / 'memory.max').is_file():
        leaf = mount
    groups, headroom = [], [info['MemAvailable']]
    for path in (leaf, *leaf.parents):
        if not path.is_relative_to(mount):
            break
        limit, current = (path / 'memory.max').read_text().strip(), int((path / 'memory.current').read_text())
        free = None if limit == 'max' else int(limit) - current
        groups.append(dict(path=str(path), maximum=limit, current=current, headroom=free))
        if free is not None:
            headroom.append(free)
    needed = max(0, LIMITS['rss_bytes'] - memory['VmRSS']) + LIMITS['memory_reserve_bytes']
    used = sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file())
    # Leave bounded room for an in-flight raw block and its partial receipt.
    if used + 8 * 1024**2 > LIMITS['output_bytes']:
        raise RuntimeError('output cap serialization reserve exhausted')
    disk_needed = LIMITS['output_bytes'] - used + LIMITS['disk_reserve_bytes']
    if min(headroom) < needed or shutil.disk_usage(output).free < disk_needed:
        raise RuntimeError('insufficient measured memory/disk headroom')
    return dict(process=memory, host_available=info['MemAvailable'], cgroups=groups,
                needed_memory=needed, output_bytes=used, disk_free=shutil.disk_usage(output).free)


def capture_tools():
    from scripts.collect_joint_temporal_pairs import PhysicalCapture, Journal, BlockBudget, empty_counters
    from scripts.collect_joint_temporal_pairs import verify_geometry
    return PhysicalCapture, Journal, BlockBudget, empty_counters, verify_geometry


def verify_imports(output, p):
    pins = dict(p['environment_sha256'])
    pins.update({str(output / 'source' / n): h for n, h in p['source_sha256'].items()})
    pins.update({str(output / 'model' / n): h for n, h in p['model_sha256'].items()})
    origins = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.is_relative_to(SNAPSHOT) or path.is_relative_to(output):
            if pins.get(str(path)) != sha(path):
                raise ValueError('import outside targeted closure: ' + str(path))
            origins[name] = str(path)
        elif path.is_relative_to(Path(p['repository_root'])):
            raise ValueError('unfrozen repository import: ' + str(path))
    return origins


def load_runtime(output, p, arm, seed):
    if any(name in sys.modules for name in ('agent', 'haic_agent', 'training')):
        raise ValueError('fresh isolated module namespace required')
    if (Path(sys.executable).absolute() != PYTHON or sys.version_info[:2] != (3, 11)
            or any(os.environ.get(k) != v for k, v in THREAD_ENV.items())):
        raise ValueError('exact single-thread CPU21 process environment required')
    sys.path.insert(0, str(output / 'model'))
    import haic_agent
    if Path(haic_agent.__file__).resolve() != output / 'model/haic_agent/__init__.py':
        raise ValueError('champion package imported outside frozen sources')
    # The strict evaluator imports Agent at module scope, even for replay. Import
    # its exact MODULE now; never instantiate it on either replay arm.
    entry = importlib.import_module('agent')
    if Path(entry.__file__).resolve() != output / 'model/agent.py':
        raise ValueError('champion import outside frozen package')
    haic_agent.__path__.append(str(SNAPSHOT / 'haic_agent'))
    sys.path.insert(0, str(SNAPSHOT))
    sys.path.insert(0, str(output / 'source'))
    import cv2
    import numpy as np
    import torch
    from training.env_factory import create_training_environment
    evaluator = importlib.import_module('training.evaluate_closed_loop')
    if (Path(str(evaluator.__file__)).resolve() != SNAPSHOT / 'training/evaluate_closed_loop.py'
            or Path(inspect.getfile(create_training_environment)).resolve() != SNAPSHOT / 'training/env_factory.py'
            or 'fail_on_invalid_action' not in inspect.signature(evaluator.run_episode).parameters):
        raise ValueError('exact external strict stack required, not root weak evaluator')
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    if (torch.__version__, np.__version__, cv2.__version__) != ('2.1.0+cpu', '1.26.0', '4.8.1'):
        raise ValueError('pinned CPU21 dependency versions required')
    model = None
    if arm == 'baseline':
        model = entry.Agent()
    elif arm not in ARMS:
        raise ValueError('unknown arm')
    capture_tools()
    from haic.algorithms.joint_control import observer, interval_comparison, paired_residual, physics
    verify_imports(output, p)
    return np, create_training_environment, model


def action32(action):
    import numpy as np
    value = np.asarray(action)
    if (value.shape != (3,) or not np.isfinite(value).all() or abs(value[0]) > 1
            or np.any((value[1:] < 0) | (value[1:] > 1))):
        raise ValueError('finite bounded action triple required')
    result = value.astype(np.float32)
    if not np.array_equal(value, result):
        raise ValueError('recorded action must be exact float32, not silently requantized')
    return result.copy()


def joint_actions(a0):
    import numpy as np
    a0 = action32(a0)
    steer, gas, brake = map(float, a0)
    a1 = a0.copy()
    a1[0] = math.copysign(max(0., abs(steer) - .04), steer) if steer else .04
    a1[1:] = [min(1., gas + .05), max(0., brake - .05)]
    return np.stack((a0, a1))


def tail_actions(actions, arm):
    import numpy as np
    if arm not in ARMS:
        raise ValueError('unknown arm')
    a0, a1 = [action32(a) for a in actions]
    return np.stack(([a1] if arm == 'alternative' else [a0]) + [a0] * 3)


def calibrated_motions(snapshots, absolute):
    """Detached per-link widening, identical to the old successor._motions rule."""
    calibrated, latest = [], ['mapping_history_missing']
    support = absolute['mapping']['by_provenance']
    for snapshot in list(snapshots)[1:]:
        motion, reasons = deepcopy(snapshot['mapping_motion']), []
        bound = support.get(motion.get('provenance', 'unsupported'), {})
        if not motion.get('valid', False):
            reasons.append('mapping_invalid')
        try:
            values = [float(motion[k]) for k in ('position_uncertainty', 'yaw_uncertainty')]
            residual = [float(bound[k]) for k in ('position_residual', 'yaw_residual')]
            if (bound.get('supported') is not True or bound.get('n_cal', 0) <= 0
                    or any(not math.isfinite(v) or v < 0 for v in values + residual)):
                raise ValueError('unsupported provenance')
            motion['position_uncertainty'], motion['yaw_uncertainty'] = map(max, zip(values, residual))
        except (KeyError, TypeError, ValueError):
            motion['valid'] = False
            reasons.append('mapping_calibration_unsupported:' + str(motion.get('provenance')))
        motion['calibration_reasons'] = reasons
        calibrated.append(motion)
        latest = reasons
    return calibrated, latest


def eligibility(snapshot, proposal, info, shield_info, witness, mapping_reasons):
    """Inputs are causal observer/policy diagnostics only, never capture labels."""
    import numpy as np
    reasons = list(mapping_reasons)
    if not snapshot['valid']:
        reasons.append('observer_invalid')
    if not float(snapshot['state'].forward_speed) > 5:
        reasons.append('not_supported_forward_motion')
    if not 40 <= float(info.get('pixel_speed', math.nan)) <= 80:
        reasons.append('hud_speed_outside_pilot_envelope')
    if not (np.all(proposal >= np.array([-.35, 0., .02], np.float32))
            and np.all(proposal <= np.array([.35, .1, .5], np.float32))):
        reasons.append('action_outside_pilot_envelope')
    if (shield_info.get('active', True) or shield_info.get('baseline_threat', True)
            or shield_info.get('rearm_blocked', True) or witness['encounter_open'] or witness['box_threatened']):
        reasons.append('shield_active_threat_or_blocked')
    if shield_info.get('reason') not in ('baseline_clear', 'no_visible_obstacle'):
        reasons.append('shield_projection_unsupported')
    if witness['pre_recovery'] or any(info.get(k, False) for k in (
            'recovery_changed', 'recovery_steps_remaining', 'impact_proxy_trigger',
            'impact_steps_remaining', 'contact_proxy', 'contact_remaining', 'contact_active')):
        reasons.append('nominal_recovery_or_contact')
    if (witness['captures'] != 1 or not witness['brake_history']
            or not math.isfinite(witness['pre_wheel']) or shield_info.get('projection_speed') is None):
        reasons.append('feedback_capture_unavailable')
    return reasons


def query_champion(model, observation):
    """Capture the old feedback witness without changing actions or committing candidates."""
    import numpy as np
    shield, captured = model.driver, []
    nominal = shield.driver
    witness = dict(pre_wheel=float(shield.wheel_angle),
        pre_recovery=any(getattr(nominal, k, 0) > 0 for k in ('impact_left', 'recovery_left', 'contact_left')))
    original = shield.act.__func__
    namespace = dict(original.__globals__)
    advance = namespace['advance_boxes']

    def capture(boxes, pose):
        captured.append(True)
        return advance(boxes, pose)

    namespace['advance_boxes'] = capture
    method = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    method.__kwdefaults__ = original.__kwdefaults__
    had_act, previous = 'act' in vars(shield), vars(shield).get('act')
    try:
        shield.act = MethodType(method, shield)
        action = model.act(observation.copy())
    finally:
        if had_act:
            shield.act = previous
        else:
            del shield.act
    if np.asarray(action).dtype != np.float32:
        raise ValueError('champion must issue float32')
    witness.update(captures=len(captured), brake_history=bool(nominal.brake_history),
                   encounter_open=bool(shield.encounter_open), box_threatened=bool(np.asarray(shield.box_threatened).any()))
    return action32(action), deepcopy(shield.last_step_diagnostics()), deepcopy(shield.last_shield), witness


def physical_predictions(hypotheses, actions) -> dict[str, Any]:
    import numpy as np
    from haic.algorithms.joint_control.physics import PhysicsState, PhysicsParameters, predict
    scenarios = len(np.atleast_1d(hypotheses.forward_speed))
    tiled = {}
    for field in fields(hypotheses):
        shape = (scenarios, 4) if field.name in ('friction', 'wheel_omega') else (scenarios,)
        value = np.broadcast_to(np.asarray(getattr(hypotheses, field.name), float), shape)
        tiled[field.name] = np.tile(value, (2, 1)) if len(shape) == 2 else np.tile(value, 2)
    controls = np.broadcast_to(actions[0], (2, scenarios, 16, 3)).copy()
    controls[:, :, :4] = actions[:, None, None]
    result = predict(PhysicsState(**tiled), controls.reshape(-1, 16, 3), .02,
                     parameters=PhysicsParameters(1., 1., 1.))
    return {k: v.reshape(2, scenarios, *v.shape[1:]) for k, v in result.items() if k != 'final_state'}


class PrefixSelector:
    """No environment reference: reset0, then one actual commit/observe per hold."""

    def __init__(self, model, calibration, observation):
        from haic.algorithms.joint_control.observer import TemporalObserver
        self.model, self.calibration = model, calibration
        self.observer = TemporalObserver(decision_dt=.08)
        self.frames, self.snapshots = deque(maxlen=4), deque(maxlen=4)
        self.frames.append(observation[-1].copy())
        self.snapshots.append(self.observer.reset(observation))
        self.calls = 0
        model.reset(observation.copy())

    def propose(self, completed, observation) -> tuple[Any, dict[str, Any] | None, dict[str, Any]]:
        import numpy as np
        from haic.algorithms.joint_control import interval_comparison as old, paired_residual as new
        if completed != self.calls or completed > LAST:
            raise ValueError('nonsequential or post-anchor champion query')
        a0, info, shield_info, witness = query_champion(self.model, observation)
        self.calls += 1
        note = dict(completed_prefix=completed, action=a0, comparison_computed=False, reasons=[])
        if not FIRST <= completed <= LAST:
            return a0, None, jsonable(note)
        snapshot = self.snapshots[-1]
        motions, mapping_reasons = calibrated_motions(self.snapshots, self.calibration['absolute_calibration'])
        reasons = eligibility(snapshot, a0, info, shield_info, witness, mapping_reasons)
        note['reasons'] = reasons
        if reasons:
            return a0, None, jsonable(note)
        actions = joint_actions(a0)
        scene = old.extract_scene(np.stack(self.frames), motions=motions)
        hypotheses = self.observer.shared_hypotheses()
        before = canonical_sha(jsonable(dict(snapshot=self.observer.snapshot(), hypotheses=hypotheses)))
        absolute = self.calibration['absolute_calibration']
        original = old.compare_candidates(scene, hypotheses, actions, observer_valid=True,
            position_residual=absolute['position_residual'], yaw_residual=absolute['yaw_residual'],
            paired_cost_residual=absolute['paired_cost_residual'])
        corrected = new.compare_candidates(scene, hypotheses, actions, observer_valid=True, calibration=self.calibration)
        if before != canonical_sha(jsonable(dict(snapshot=self.observer.snapshot(), hypotheses=hypotheses))):
            raise ValueError('comparison mutated causal state')
        for key in ('poses', 'reference_costs', 'absolute_supported', 'veto',
                    'absolute_road_clearance', 'absolute_obstacle_clearance'):
            if not np.array_equal(original[key], corrected[key], equal_nan=True):
                raise ValueError('new performance comparator changed physical/cost semantics: ' + key)
        common = bool(corrected['common_support'] and np.asarray(corrected['cost_supported']).all())
        note.update(comparison_computed=True, common_full_h4_cost_support=common,
                    original=original, corrected=corrected)
        if not common:
            return a0, None, jsonable(note)
        prediction = physical_predictions(hypotheses, actions)
        poses = np.stack([prediction[k] for k in ('relative_x', 'relative_y', 'yaw_delta')], axis=-1)
        if corrected['poses'].shape != (2, 19, 17, 3) or not np.array_equal(poses, corrected['poses'][:, :, 1:]):
            raise ValueError('physical prediction and saved shared poses differ')
        anchor = dict(schema=SCHEMA + '-anchor', completed_prefix=completed, actions=actions,
            actions_float32_hex=actions.tobytes().hex(), snapshot=snapshot, hypotheses=hypotheses,
            motions=motions, scene=scene, original=original, corrected=corrected, physical_predictions=prediction,
            champion_diagnostics=info, shield_diagnostics=shield_info, feedback_witness=witness,
            selector='first causal preeligible+cost-supported point, NOT rank/veto/future-selected')
        return a0, jsonable(anchor), jsonable(note)

    def commit(self, action):
        self.observer.commit_action(action32(action))

    def observe(self, observation):
        self.frames.append(observation[-1].copy())
        self.snapshots.append(self.observer.observe(observation, elapsed_dt=.08))


def collect_prefix_and_tail(arm, observation, step, save_anchor, record_query, *, selector=None, baseline=None) -> dict[str, Any]:
    """Small injectable execution core. Replay cannot query or instantiate an Agent."""
    if arm not in ARMS or (arm == 'baseline') != (selector is not None):
        raise ValueError('selector belongs exclusively to baseline')
    actions, anchor = [], None
    if arm == 'baseline':
        for completed in range(LAST + 1):
            action, anchor, note = selector.propose(completed, observation)
            record_query(note)
            if anchor is not None:
                if not FIRST <= completed <= LAST or anchor['completed_prefix'] != completed:
                    raise ValueError('anchor off-by-one or outside search window')
                anchor['prefix_actions'] = deepcopy(actions)
                save_anchor(anchor)  # Must be durable BEFORE the first suffix call.
                break
            if completed == LAST:
                return dict(status='NOT_FOUND', anchor=None, actions=actions, agent_calls=selector.calls)
            selector.commit(action)
            observation = step(action, completed)
            selector.observe(observation)
            actions.append(action32(action).tolist())
    else:
        if baseline is None or baseline['status'] != 'COMPLETE':
            raise ValueError('complete contemporary baseline required for replay')
        anchor = deepcopy(baseline['anchor'])
        prefix = anchor['prefix_actions']
        if len(prefix) != anchor['completed_prefix'] or not FIRST <= len(prefix) <= LAST:
            raise ValueError('invalid recorded anchor prefix')
        for completed, action in enumerate(prefix):
            observation = step(action32(action), completed)
            actions.append(action32(action).tolist())
    start = anchor['completed_prefix']
    for offset, action in enumerate(tail_actions(anchor['actions'], arm)):
        if selector is not None:
            selector.commit(action)
        observation = step(action, start + offset)
        if selector is not None:
            selector.observe(observation)
        actions.append(action.tolist())
    return dict(status='COMPLETE', anchor=anchor, actions=actions, agent_calls=selector.calls if selector else 0)


def json_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_arm(directory, digest) -> dict[str, Any]:
    directory = Path(directory)
    receipt = read_json(directory / 'receipt.json')
    if receipt['protocol_sha256'] != digest or receipt['status'] not in ('COMPLETE', 'NOT_FOUND'):
        raise ValueError('complete bound arm required')
    for name, expected in receipt['artifacts_sha256'].items():
        if sha(local_path(directory, name)) != expected:
            raise ValueError('sealed arm artifact differs')
    state = read_json(directory / 'state.json')
    boundaries, raw = json_rows(directory / 'boundaries.jsonl'), json_rows(directory / 'raw.jsonl')
    if (len(boundaries) != len(state['actions']) + 1
            or len(raw) != 2 * (WARMUP + SKIP * len(state['actions']))):
        raise ValueError('incomplete raw/endpoint stream')
    with (directory / 'observations.zlib-stream').open('rb') as stream:
        for row in boundaries:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError('missing saved observation')
            size, = struct.unpack('>Q', header)
            if size > 4 * 84 * 84 * 4 + 1024:
                raise ValueError('invalid observation chunk size')
            blob = zlib.decompress(stream.read(size))
            if len(blob) != 4 * 84 * 84 * 4 or hashlib.sha256(blob).hexdigest() != row['image_sha256']:
                raise ValueError('saved observation differs from endpoint digest')
        if stream.read(1):
            raise ValueError('extra saved observations')
    return dict(state=state, endpoints=boundaries, raw=raw, static=read_json(directory / 'static.json'),
                receipt_sha256=sha(directory / 'receipt.json'))


def collect_arm(output, p, slot, digest, deadline, np, factory, model) -> dict[str, Any]:
    if (slot['arm'] == 'baseline') != (model is not None):
        raise ValueError('only the baseline may own an Agent instance')
    PhysicalCapture, Journal, BlockBudget, _, verify_geometry = capture_tools()
    arm_dir = output / slot['slot']
    arm_dir.mkdir(exist_ok=False)
    budget, journal = BlockBudget(output, slot, deadline), Journal(arm_dir)
    data: dict[str, Any] = dict(schema=SCHEMA + '-arm', protocol_sha256=digest, slot=slot, status='PARTIAL',
                actions=[], anchor=None, error=None, resource_samples=[], agent_calls=0)
    baseline = None
    environment, capture, decision, warmup, boundaries, returned = None, None, -1, True, [], []
    try:
        if slot['arm'] != 'baseline':
            baseline = load_arm(output / f"{slot['track_id']}-{slot['seed']}-baseline", digest)
            if baseline['state']['status'] != 'COMPLETE':
                raise ValueError('NOT_FOUND baseline must skip replay without reset')
            data['baseline_receipt_sha256'] = baseline['receipt_sha256']
        data['resource_samples'].append(resource_admission(output))
        # CAP MUST NOT become anchor+4: time_limit.maximum participates in parity.
        environment = factory(track_id=slot['track_id'], seed=slot['seed'], max_decisions=CAP, render_mode=None)
        raw = environment.unwrapped
        original_reset, original_step = raw.reset, raw.step

        def observed_reset(*args, **kwargs):
            budget.begin('resets')  # Blocks wrapper warmup retry before delegation.
            value = original_reset(*args, **kwargs)
            budget.complete('resets')
            return value

        def observed_step(action):
            nonlocal capture
            if capture is None:
                capture = PhysicalCapture(environment, np)
                save(arm_dir / 'static.json', capture.static)
                if baseline and encoded(capture.static) != encoded(baseline['static']):
                    raise ValueError('static fixture parity differs')
            issued = None if action is None else action32(action).tolist()
            index = budget.arm['raw_actual_started'] + 1
            if warmup and ((index == 1 and issued is not None) or (index != 1 and issued != [0., 0., 0.])):
                raise ValueError('unexpected warmup action/order')
            parity = baseline is not None and (slot['arm'] == 'repeat' or
                index <= WARMUP + SKIP * baseline['state']['anchor']['completed_prefix'])
            before = capture.dynamic()
            pre = dict(event='PRE', raw_index=index, decision=decision, warmup=warmup, action=issued, state=before)
            journal.record('raw.jsonl', pre)
            if parity and encoded(pre) != encoded(baseline['raw'][2 * (index - 1)]):
                raise ValueError('raw PRE parity differs')
            budget.begin('raw')
            value = original_step(action)
            budget.complete('raw')
            result = dict(reward=float(value[1]), terminated=bool(value[2]), truncated=bool(value[3]),
                          info=jsonable(value[4]), image_sha256=hashlib.sha256(np.ascontiguousarray(value[0]).tobytes()).hexdigest())
            returned.append(dict(raw_index=index, **result))
            post = dict(event='POST', raw_index=index, decision=decision, warmup=warmup, action=issued,
                        state=capture.dynamic(), **result)
            journal.record('raw.jsonl', post)
            if parity and encoded(post) != encoded(baseline['raw'][2 * (index - 1) + 1]):
                raise ValueError('raw POST parity differs')
            return value

        raw.reset, raw.step = observed_reset, observed_step

        def endpoint(observation, info, reward=None, terminated=False, truncated=False):
            arr = np.asarray(observation)
            if (arr.dtype != np.float32 or arr.shape != (4, 84, 84) or not np.isfinite(arr).all()
                    or arr.min() < 0 or arr.max() > 1 or not np.array_equal(arr, environment.environment.stack_state)):
                raise ValueError('exact normalized decision observation required')
            row = dict(index=len(boundaries), state=capture.full(), info=jsonable(info), reward=reward,
                terminated=bool(terminated), truncated=bool(truncated), image_sha256=hashlib.sha256(arr.tobytes()).hexdigest())
            boundaries.append(row)
            journal.record('boundaries.jsonl', row)
            journal.observation(arr.tobytes())
            if baseline and (slot['arm'] == 'repeat' or row['index'] <= baseline['state']['anchor']['completed_prefix']):
                if encoded(row) != encoded(baseline['endpoints'][row['index']]):
                    raise ValueError('full endpoint/image parity differs')
            return row

        validate_frozen(output, digest)
        validate_admission(read_json(output / 'admission.json'), digest, p)
        data['resource_samples'].append(resource_admission(output))
        budget.reserve(warmup=True)
        observation, info = environment.reset()
        warmup = False
        initial = endpoint(observation, info)
        reference = next(r for r in read_json(output / 'claim.json')['references']
                         if (r['track_id'], r['seed']) == (slot['track_id'], slot['seed']))
        historical = read_json(reference['episode_path'])
        data['geometry'] = verify_geometry(environment, historical, np)
        if initial['image_sha256'] != historical['initial_observation_sha256']:
            raise ValueError('consumed historical initial pixels differ')
        journal.flush()
        budget.seal()
        selector = PrefixSelector(model, read_json(output / 'calibration.json'), observation) if model is not None else None

        def step(action, index):
            nonlocal decision
            decision = index
            before = boundaries[-1]['state']['t']
            budget.reserve()
            journal.record('decisions.jsonl', dict(event='INTENT', index=index, action=action32(action).tolist()))
            journal.flush()
            budget.begin('decisions')
            obs, reward, terminated, truncated, info = environment.step(action32(action))
            budget.complete('decisions')
            data['actions'].append(action32(action).tolist())
            after = endpoint(obs, info, float(reward), terminated, truncated)
            journal.record('decisions.jsonl', dict(event='COMPLETE', index=index, action=action32(action).tolist()))
            if not math.isclose(after['state']['t'] - before, .08, abs_tol=1e-12, rel_tol=0):
                raise ValueError('shortened hold retained; no replacement')
            journal.flush()
            budget.seal()
            data['resource_samples'].append(resource_admission(output))
            anchor = data['anchor'] if baseline is None else baseline['state']['anchor']
            final_suffix = anchor is not None and index == anchor['completed_prefix'] + HORIZON - 1
            if (terminated or truncated) and not final_suffix:
                raise ValueError('terminal prefix/suffix retained as partial; no replacement')
            return obs

        def anchor_save(anchor):
            anchor.update(protocol_sha256=digest, calibration_sha256=p['calibration_sha256'],
                          source_sha256=p['source_sha256'], initial_prefix_endpoint_sha256=canonical_sha(boundaries[-1]))
            save(arm_dir / 'anchor.json', anchor)
            data['anchor'] = anchor

        def query_note(note):
            data['agent_calls'] += 1
            journal.record('decisions.jsonl', dict(event='QUERY', **note))

        result = collect_prefix_and_tail(slot['arm'], observation, step, anchor_save, query_note,
                                         selector=selector, baseline=baseline['state'] if baseline else None)
        if data['actions'] != result['actions'] or data['agent_calls'] != result['agent_calls']:
            raise ValueError('actual action/query counts differ')
        data.update(result)
        expected = LAST if data['status'] == 'NOT_FOUND' else data['anchor']['completed_prefix'] + HORIZON
        if (budget.arm['decisions_completed'] != expected or budget.arm['raw_completed'] != WARMUP + SKIP * expected
                or data['agent_calls'] != (LAST + 1 if data['status'] == 'NOT_FOUND' else
                    data['anchor']['completed_prefix'] + 1) * (slot['arm'] == 'baseline')):
            raise ValueError('complete arm length/query budget differs')
        verify_imports(output, p)
        validate_frozen(output, digest)
    except BaseException as failure:
        data['status'], data['error'] = 'PARTIAL', f'{type(failure).__name__}: {failure}'
        budget.fail()
        data['returned_raw_evidence'] = returned
        raise
    finally:
        try:
            journal.close()
            data['counters'] = deepcopy(budget.arm)
            save(arm_dir / 'state.json', jsonable(data))
            save(arm_dir / 'receipt.json', dict(protocol_sha256=digest, status=data['status'],
                artifacts_sha256={p.name: sha(p) for p in arm_dir.iterdir() if p.is_file() and p.name != 'receipt.json'}))
        finally:
            if environment is not None:
                environment.close()
    return data


def worker_command(output, digest, index=0, mode='collect'):
    return [str(PYTHON), '-I', '-B', '-c',
        'import sys;sys.path.insert(0,sys.argv[1]);from scripts.collect_joint_envelope_pairs import worker;'
        'worker(sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5])',
        str(Path(output).resolve() / 'source'), str(Path(output).resolve()), digest, str(index), mode]


def worker(output, digest, index, mode):
    output = Path(output).resolve()
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        raise ValueError('fresh isolated bytecode-disabled worker required')
    p = validate_frozen(output, digest)
    os.sched_setaffinity(0, p['cpu_affinity'])
    resource.setrlimit(resource.RLIMIT_AS, (LIMITS['address_space_bytes'],) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (LIMITS['output_bytes'],) * 2)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('child wall deadline')))
    signal.alarm(LIMITS['child_seconds'])
    if mode == 'resources':
        print(json.dumps(resource_admission(output)))
        return
    if mode == 'analyze':
        analyze_saved(output, digest)
        return
    if mode not in ('preflight', 'collect') or not 0 <= index < len(schedule()):
        raise ValueError('invalid worker operation/slot')
    slot = schedule()[index]
    resource_admission(output)
    np, factory, model = load_runtime(output, p, slot['arm'], slot['seed'])
    if mode == 'preflight':
        print(json.dumps(dict(arm=slot['arm'], import_only=True, environment_resets=0,
            agent_instances=int(model is not None), evaluator_file=str(SNAPSHOT / 'training/evaluate_closed_loop.py'),
            cpu_affinity=sorted(os.sched_getaffinity(0)), resources=resource_admission(output))))
        return
    start = read_json(output / 'run-start.json')
    if (start['protocol_sha256'] != digest or start['parent_pid'] != os.getppid()
            or start['admission_sha256'] != sha(output / 'admission.json')
            or start['preflight_sha256'] != sha(output / 'preflight.json')):
        raise ValueError('live authorized parent and immutable admission/preflight required')
    validate_preflight(read_json(output / 'preflight.json'), digest, p)
    validate_admission(read_json(output / 'admission.json'), digest, p)
    deadline = min(start['deadline_monotonic'], time.monotonic() + LIMITS['child_seconds'])
    collect_arm(output, p, slot, digest, deadline, np, factory, model)


def run_json_child(output, digest, *, index=0, mode='resources'):
    child = subprocess.run(worker_command(output, digest, index, mode), cwd=Path(output) / 'model',
        env={**os.environ, **THREAD_ENV}, capture_output=True, text=True, timeout=LIMITS['child_seconds'])
    if child.returncode:
        raise RuntimeError(child.stderr or 'isolated child failed')
    return json.loads(child.stdout)


def validate_preflight(value, digest, p):
    arms = value.get('arms', [])
    if (value.get('protocol_sha256') != digest or value.get('environment_resets') != 0
            or len(arms) != 3 or [r.get('arm') for r in arms] != list(ARMS)
            or any(r.get('environment_resets') != 0 or r.get('import_only') is not True
                   or r.get('agent_instances') != int(r['arm'] == 'baseline')
                   or r.get('cpu_affinity') != p['cpu_affinity']
                   or r.get('evaluator_file') != str(SNAPSHOT / 'training/evaluate_closed_loop.py') for r in arms)):
        raise ValueError('three source-bound zero-reset strict-stack preflights required')


def preflight(output, digest):
    output = Path(output).resolve()
    p = validate_frozen(output, digest)
    if (output / 'preflight.json').exists():
        raise ValueError('exclusive preflight already exists')
    rows = [run_json_child(output, digest, index=i, mode='preflight') for i in range(3)]
    result = dict(schema=SCHEMA + '-preflight', protocol_sha256=digest, environment_resets=0, arms=rows)
    validate_preflight(result, digest, p)
    save(output / 'preflight.json', result)
    return result


def execute(output, digest, admission) -> dict[str, Any]:
    output = Path(output).resolve()
    p = validate_frozen(output, digest)
    validate_preflight(read_json(output / 'preflight.json'), digest, p)
    value = read_json(admission)
    validate_admission(value, digest, p, starting=True)
    for name in ('admission.json', 'run-start.json', 'counters.json', 'report.json', *(s['slot'] for s in schedule())):
        if (output / name).exists():
            raise ValueError('existing execution evidence; never restart/retry')
    # Pre-exec admission runs in a new stdlib process, independent of caller RSS history.
    initial_resources = run_json_child(output, digest)
    _, _, _, empty_counters, _ = capture_tools()
    save(output / 'admission.json', value)
    started = time.monotonic()
    save(output / 'run-start.json', dict(protocol_sha256=digest, parent_pid=os.getpid(),
        deadline_monotonic=started + LIMITS['total_seconds'], initial_resources=initial_resources,
        admission_sha256=sha(output / 'admission.json'), preflight_sha256=sha(output / 'preflight.json')))
    save(output / 'counters.json', empty_counters(schedule()))
    rows, error = [dict(s, status='UNRUN') for s in schedule()], None
    try:
        for index, row in enumerate(rows):
            if row['arm'] != 'baseline' and rows[index - index % 3]['status'] == 'NOT_FOUND':
                row['status'] = 'SKIPPED_NOT_FOUND'
                continue
            validate_frozen(output, digest)
            validate_admission(value, digest, p)
            remaining = min(LIMITS['total_seconds'] - (time.monotonic() - started), value['expires_unix_s'] - time.time())
            if remaining <= 0:
                raise TimeoutError('total wall deadline')
            row['status'] = 'STARTED'
            begin, returncode, child_error = time.monotonic(), None, None
            try:
                with (output / (row['slot'] + '.stdout.log')).open('x') as stdout, (output / (row['slot'] + '.stderr.log')).open('x') as stderr:
                    child = subprocess.run(worker_command(output, digest, index), cwd=output / 'model',
                        env={**os.environ, **THREAD_ENV}, stdout=stdout, stderr=stderr,
                        timeout=min(LIMITS['child_seconds'], remaining))
                returncode = child.returncode
                if returncode:
                    raise RuntimeError(f'worker exited {returncode}')
                arm = load_arm(output / row['slot'], digest)
                row.update(status=arm['state']['status'], receipt_sha256=arm['receipt_sha256'])
            except BaseException as failure:
                row['status'] = 'PARTIAL'
                child_error = f'{type(failure).__name__}: {failure}'
                raise
            finally:
                save(output / (row['slot'] + '.process.json'), dict(protocol_sha256=digest, slot=row['slot'],
                    returncode=returncode, error=child_error, wall_seconds=time.monotonic() - begin))
    except BaseException as failure:
        error = f'{type(failure).__name__}: {failure}'
    report = dict(schema=SCHEMA + '-report', protocol_sha256=digest, rows=rows, operator_error=error,
        wall_seconds=time.monotonic() - started,
        artifacts_sha256={str(path.relative_to(output)): sha(path) for path in output.rglob('*')
            if path.is_file() and not path.is_relative_to(output / 'source') and not path.is_relative_to(output / 'model')})
    save(output / 'report.json', report)
    return report


def rate(n, d):
    return dict(numerator=n, denominator=d, rate=n / d if d else None)


def predicted_selects(comparison):
    """The unchanged absolute guard is orthogonal to performance ranking."""
    interval = comparison.get('delta_interval', [None, None])[1]
    return bool(comparison.get('common_support') and all(comparison['cost_supported'])
        and interval is not None and interval[1] is not None and math.isfinite(interval[1]) and interval[1] < -.05
        and all(comparison['absolute_supported'][1]) and not any(comparison['veto'][1]))


def potential_pilot_gate(rows) -> dict[str, Any]:
    complete = [r for r in rows if r.get('status') == 'COMPLETE' and r.get('parity', {}).get('passed')]
    selected = [r for r in complete if r.get('new_would_select')]
    evaluable = [r for r in selected if r.get('actual_cost_supported') is True
                 and r.get('actual_cost_interval') is not None and r.get('actual_safe') is not None
                 and r.get('contained') is not None]
    blockers = [r['cell'] for r in selected if r.get('actual_safe') is False or r.get('contained') is False
                or (r.get('actual_cost_supported') is True and r.get('actual_cost_interval') is not None
                    and r['actual_cost_interval'][1] >= -.05)]
    wins = [r for r in evaluable if r['actual_cost_interval'][1] < -.05 and r['actual_safe'] and r['contained']]
    unknown = len(selected) != len(evaluable) or any(r.get('status') not in ('COMPLETE', 'NOT_FOUND') for r in rows)
    outcome = 'BLOCKED' if blockers else 'N/A' if unknown or not complete else 'PASS' if wins else 'NOT_MET'
    return dict(outcome=outcome, potential_full_pilot=outcome == 'PASS', allocated_pairs=len(rows),
        complete_pairs=len(complete), not_found_pairs=sum(r.get('status') == 'NOT_FOUND' for r in rows),
        selected_pairs=len(selected), selected_evaluable_pairs=len(evaluable),
        supported_selected_benefit=rate(len(wins), len(evaluable)) if evaluable else
            dict(numerator=None, denominator=0, rate=None), blockers=blockers,
        qualification='Report only; main decides next stage. Unsupported unrelated regions do not globally block. N/A is not zero wins.')


def pair_parity(base, repeat, alternative) -> dict[str, Any]:
    n = base['state']['anchor']['completed_prefix']
    full = (base['static'] == repeat['static'] and base['endpoints'] == repeat['endpoints']
            and base['raw'] == repeat['raw'] and base['state']['actions'] == repeat['state']['actions'])
    prefix = (base['static'] == alternative['static'] and base['endpoints'][:n + 1] == alternative['endpoints'][:n + 1]
        and base['raw'][:2 * (WARMUP + SKIP * n)] == alternative['raw'][:2 * (WARMUP + SKIP * n)]
        and base['state']['actions'][:n] == alternative['state']['actions'][:n])
    return dict(passed=full and prefix, baseline_repeat_full=full, alternative_through_anchor=prefix,
                prefix_decisions=n, prefix_raw_including_warmup=WARMUP + SKIP * n)


def actual_trajectory(arm, anchor) -> dict[str, Any]:
    import numpy as np
    n = anchor['completed_prefix']
    initial = arm['endpoints'][n]['state']
    posts = [arm['raw'][2 * i + 1] for i in range(WARMUP + SKIP * n, WARMUP + SKIP * (n + HORIZON))]
    states = [initial] + [r['state'] for r in posts]
    start = initial['hull']
    if any(not math.isclose(s['t'] - initial['t'], .02 * i, rel_tol=0, abs_tol=1e-10) for i, s in enumerate(states)):
        raise ValueError('actual suffix cadence differs')
    xy = np.asarray([s['hull']['position'] for s in states]) - start['position']
    c, s = math.cos(start['angle']), math.sin(start['angle'])
    yaw = np.unwrap([s['hull']['angle'] for s in states]) - start['angle']
    poses = np.column_stack((xy[:, 0] * c + xy[:, 1] * s, -xy[:, 0] * s + xy[:, 1] * c, yaw))
    return dict(poses=poses, speed=np.asarray([np.linalg.norm(s['hull']['linearVelocity']) for s in states]),
                states=states, boundary_states=[r['state'] for r in arm['endpoints'][n:n + HORIZON + 1]],
                controls=np.asarray([p['action'] for p in posts]), reward=sum(p['reward'] for p in posts))


def containment(actual, predicted, position, yaw) -> dict[str, Any]:
    import numpy as np
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if predicted.shape != (19, 17, 3) or actual.shape != (17, 3):
        raise ValueError('complete fixed nineteen-scenario H4 paths required')
    distance = np.linalg.norm(predicted[..., :2] - actual[None, :, :2], axis=-1)
    angle = np.abs((predicted[..., 2] - actual[None, :, 2] + np.pi) % (2 * np.pi) - np.pi)
    whole = ((distance <= np.asarray(position)) & (angle <= np.asarray(yaw))).all(axis=1)
    return dict(contained=bool(whole.any()), whole_path_scenarios=np.flatnonzero(whole).tolist(),
                position_errors=distance, yaw_errors=angle, interpretation='ONE fixed scenario for all17 endpoints, joint XY+yaw')


def physical_labels(trajectory) -> dict[str, Any]:
    states, boundaries = trajectory['states'], trajectory['boundary_states']
    if any(len(s['wheels']) != 4 for s in states) or not boundaries or any(
            not s['damage_telemetry_valid'] for s in boundaries):
        raise ValueError('complete wheel/damage labels required')
    wheels = [sum(bool(w['tiles']) for w in s['wheels']) for s in states]
    contacts = [bool(s['collision']) or any(c['touching'] and 'obstacle' in (c['a'][0], c['b'][0])
        and {'hull', 'wheel'}.intersection((c['a'][0], c['b'][0])) for c in s['contacts']) for s in states]
    # CollisionDamage updates AFTER the raw hold returns. Its final update exists
    # at the final decision boundary, not in that hold's last raw POST record.
    damage = max(s['damage'] for s in states + boundaries) - boundaries[0]['damage']
    return dict(obstacle_contact=any(contacts), contact_ticks=sum(contacts), wheel_road_counts=wheels,
                partial_wheel_loss=any(n < 4 for n in wheels), all_wheels_lost=any(n == 0 for n in wheels),
                damage_increase=damage, safe=not any(contacts) and damage <= 0 and all(wheels))


def progress_proxy(trajectory, static):
    import numpy as np
    road = np.asarray(static['track'], float)[:, 2:4]
    direction = np.roll(road, -1, axis=0) - road
    length = np.linalg.norm(direction, axis=1)
    if np.any(length <= 0):
        raise ValueError('degenerate GT centerline')
    station = np.r_[0., np.cumsum(length[:-1])]
    xy = np.asarray([s['hull']['position'] for s in trajectory['states']])
    f = np.clip(np.sum((xy[:, None] - road) * direction, axis=-1) / length**2, 0, 1)
    offset = xy[:, None] - (road + f[..., None] * direction)
    nearest = np.argmin(np.sum(offset**2, axis=-1), axis=1)
    arc = station[nearest] + f[np.arange(len(xy)), nearest] * length[nearest]
    signed = np.unwrap(arc / length.sum() * 2 * np.pi) * length.sum() / (2 * np.pi)
    return dict(signed_centerline_progress=(signed - signed[0]).tolist(), raw_reward=trajectory['reward'],
                qualification='Privileged GT nearest-segment/reward proxies only; never runtime scene or official ranking score')


def analyze_pair(base, repeat, alternative, calibration):
    import numpy as np
    from haic.algorithms.joint_control.interval_comparison import IntervalScene, score_trajectories
    from haic.algorithms.joint_control.paired_residual import envelope_excess
    anchor = base['state']['anchor']
    parity = pair_parity(base, repeat, alternative)
    if not parity['passed']:
        raise ValueError('strict baseline-repeat/prefix parity failed')
    actions = np.asarray(anchor['actions'], np.float32)
    if actions.tobytes().hex() != anchor['actions_float32_hex'] or not np.array_equal(actions, joint_actions(actions[0])):
        raise ValueError('saved anchor actions differ from exact float32 rule')
    n = anchor['completed_prefix']
    for name, arm in (('baseline', base), ('repeat', repeat), ('alternative', alternative)):
        if not np.array_equal(np.asarray(arm['state']['actions'][n:]), tail_actions(actions, name)):
            raise ValueError('actual continuation mismatch, not a model range miss')
    actual = [actual_trajectory(a, anchor) for a in (base, alternative)]
    controls = np.asarray([a['controls'] for a in actual])
    expected = np.repeat(np.stack((tail_actions(actions, 'baseline'), tail_actions(actions, 'alternative'))), 4, axis=1)
    if not np.array_equal(controls, expected):
        raise ValueError('raw actual controls differ from fixed tails')
    raw_scene = anchor['scene']
    bool_fields = {'road', 'known', 'uncertain_boundary', 'unobserved'}
    scene = IntervalScene(**{f.name: (raw_scene[f.name] if f.name in ('history_used', 'motion_provenance')
        else np.asarray(raw_scene[f.name], float).reshape(-1, 4) if f.name == 'boxes'
        else np.asarray(raw_scene[f.name], dtype=bool if f.name in bool_fields else float)) for f in fields(IntervalScene)})
    poses = np.asarray([a['poses'] for a in actual])
    scored = score_trajectories(scene, poses, controls)
    costs = np.asarray(scored['reference_costs'])
    supported = bool(scored['common_support'] and costs.shape == (2, 5) and np.isfinite(costs).all())
    delta = costs[1] - costs[0] if supported else None
    interval = [float(delta.min()), float(delta.max())] if supported else None
    excess = envelope_excess(anchor['corrected']['reference_delta'][1], delta) if supported else None
    order = lambda bounds: ('N/A' if bounds is None or any(v is None for v in bounds) else
                           'alternative' if bounds[1] < -.05 else 'baseline' if bounds[0] > .05 else 'ambiguous')
    absolute = calibration['absolute_calibration']
    tubes = [containment(a['poses'], anchor['corrected']['poses'][i], absolute['position_residual'], absolute['yaw_residual'])
             for i, a in enumerate(actual)]
    labels = [physical_labels(a) for a in actual]
    effects = {}
    predicted = np.asarray(anchor['corrected']['poses'])
    speed = np.asarray(anchor['physical_predictions']['speed'])
    for horizon in (1, 4):
        tick = horizon * 4
        real = np.asarray([list(a['poses'][tick]) + [a['speed'][tick] - a['speed'][0]] for a in actual])
        forecast = np.concatenate((predicted[:, :, tick], speed[:, :, tick - 1, None]), axis=-1)
        actual_delta, predicted_delta = real[1] - real[0], forecast[1] - forecast[0]
        actual_delta[2] = (actual_delta[2] + np.pi) % (2 * np.pi) - np.pi
        predicted_delta[:, 2] = (predicted_delta[:, 2] + np.pi) % (2 * np.pi) - np.pi
        effects[str(horizon)] = dict(components=['right', 'forward', 'yaw', 'speed_change'], actual=actual_delta,
            predicted_center=predicted_delta[0], predicted_shared=predicted_delta,
            center_error=predicted_delta[0] - actual_delta)
    return jsonable(dict(status='COMPLETE', parity=parity, anchor_completed_prefix=n,
        new_would_select=predicted_selects(anchor['corrected']), old_would_select=predicted_selects(anchor['original']),
        original_predicted_interval=anchor['original']['delta_interval'][1],
        new_predicted_interval=anchor['corrected']['delta_interval'][1], actual_cost_supported=supported,
        original_predicted_order=order(anchor['original']['delta_interval'][1]),
        new_predicted_order=order(anchor['corrected']['delta_interval'][1]),
        paired_envelope_excess=excess, paired_envelope_contained=None if excess is None else
            excess['lower'] <= calibration['performance']['lower'] and excess['upper'] <= calibration['performance']['upper'],
        actual_cost_interval=interval, actual_reference_deltas=delta, actual_score=scored, effects=effects,
        actual_safe=labels[1]['safe'], contained=all(t['contained'] for t in tubes), containment=tubes,
        physical_labels=labels, actual_trajectories=[{k: a[k] for k in ('poses', 'speed', 'controls')} for a in actual],
        privileged_gt_proxies=[progress_proxy(a, base['static']) for a in actual],
        actual_cost_order=order(interval), continuation='exact_fixed_H4', new_data_fit=False))


def analyze_saved(output, digest):
    output = Path(output).resolve()
    p = validate_frozen(output, digest)
    report = read_json(output / 'report.json')
    if report['protocol_sha256'] != digest or len(report['rows']) != 9:
        raise ValueError('bound nine-slot report required, including skipped/unrun/partial')
    for name, expected in report['artifacts_sha256'].items():
        if sha(local_path(output, name)) != expected:
            raise ValueError('sealed complete/partial evidence differs')
    counters = read_json(output / 'counters.json')
    for key, ceiling in (('resets_started', 9), ('decisions_started', 576), ('raw_started', 2763)):
        if counters['totals'][key] > ceiling:
            raise ValueError('overbudget reservations')
    rows = []
    for cell_index, cell in enumerate(CELLS):
        records = report['rows'][3 * cell_index:3 * cell_index + 3]
        for slot, record in zip(schedule()[3 * cell_index:3 * cell_index + 3], records):
            if any(record.get(k) != v for k, v in slot.items()):
                raise ValueError('report schedule differs')
        row = dict(cell=list(cell), status='PARTIAL', actual_cost_order='N/A')
        if records[0]['status'] == 'NOT_FOUND':
            arm = load_arm(output / records[0]['slot'], digest)
            if (arm['state']['status'] != 'NOT_FOUND' or len(arm['state']['actions']) != LAST
                    or [r['status'] for r in records[1:]] != ['SKIPPED_NOT_FOUND'] * 2):
                raise ValueError('invalid NOT_FOUND/skip evidence')
            row.update(status='NOT_FOUND', prefix_decisions=LAST)
        elif all(r['status'] == 'COMPLETE' for r in records):
            arms = [load_arm(output / r['slot'], digest) for r in records]
            row.update(analyze_pair(*arms, read_json(output / 'calibration.json')))
        else:
            row['arms'] = [{**r, 'reserved_counters': counters['arms'].get(r['slot'])} for r in records]
        rows.append(row)
    result = dict(schema=SCHEMA + '-analysis', protocol_sha256=digest, report_sha256=sha(output / 'report.json'),
        rows=rows, gate=potential_pilot_gate(rows), counters=counters, operator_error=report['operator_error'],
        qualification='New paired starts on three consumed TRAIN roads, excluded from fitting; no official, population, or safety guarantee.')
    save(output / 'analysis.json', result)
    return result


def analyze(output, digest):
    output = Path(output).resolve()
    validate_frozen(output, digest)
    if (output / 'analysis.json').exists():
        raise ValueError('analysis output is exclusive')
    child = subprocess.run(worker_command(output, digest, mode='analyze'), cwd=output / 'model',
        env={**os.environ, **THREAD_ENV}, capture_output=True, text=True, timeout=LIMITS['child_seconds'])
    if child.returncode:
        raise RuntimeError(child.stderr)
    return read_json(output / 'analysis.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('freeze', 'preflight', 'execute', 'analyze'):
        sub = commands.add_parser(name)
        sub.add_argument('--output', type=Path, required=True)
        if name == 'freeze':
            sub.add_argument('--calibration', type=Path, required=True)
            sub.add_argument('--claim', type=Path, required=True)
            sub.add_argument('--environment-source', action='append', default=[])
        else:
            sub.add_argument('--protocol-sha256', required=True)
        if name == 'execute':
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
    return int(args.command == 'execute' and result['operator_error'] is not None)


if __name__ == '__main__':
    raise SystemExit(main())
