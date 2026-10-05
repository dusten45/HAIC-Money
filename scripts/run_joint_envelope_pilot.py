"""One gated baseline/successor pair; only execute can construct/reset a simulator.

The previous pilot remains immutable. Its stateless I/O, capture, timing and
analysis primitives are reused directly, never patched or given fabricated data.
"""

import argparse
import hashlib
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

from scripts import run_joint_temporal_pilot as old


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'haic-joint-envelope-pilot-v1'
OPERATOR = 'scripts/run_joint_envelope_pilot.py'
SUCCESSOR = 'haic/algorithms/joint_control/envelope_successor.py'
COMPARATOR = 'haic/algorithms/joint_control/paired_residual.py'
CELL = (3, 3184000005)
PAIRED_CELLS = ((1, 3184000003), (2, 3184000004), CELL)
ARMS = ('baseline', 'successor')
CALIBRATION_SHA = 'a0ceed5036b116491375c84dca4ea5b781119d0cdd5fa7c8ebc8ddf77b1a9f96'
ABSOLUTE_SHA = '5c8fe665e21688a88e286ded8771577d51b1c1849db2a744c91f1905cf3bf1b0'
LIMITS = dict(old.LIMITS, max_resets=2, total_seconds=1200)
sha, read_json, save, append = old.sha, old.read_json, old.save, old.append
local_path, rate = old.local_path, old.rate


def schedule() -> list[dict[str, Any]]:
    return [dict(track_id=CELL[0], seed=CELL[1], mode=arm, repeat=0,
                 file=f'{CELL[0]}-{CELL[1]}-{arm}.json') for arm in ARMS]


def contract() -> dict[str, Any]:
    return dict(schema=SCHEMA, cells=[list(CELL)], arms=list(ARMS), schedule=schedule(),
        limits=LIMITS, max_interventions=40, partition='TRAIN', fresh=False, official_action=False,
        python=str(old.PYTHON), snapshot=str(old.SNAPSHOT), champion_zip_sha256=old.CHAMPION_SHA,
        calibration_sha256=CALIBRATION_SHA, absolute_calibration_sha256=ABSOLUTE_SHA,
        environment=dict(max_decisions=1200, frame_skip=4, frame_stack=4, warmup_raw=51,
            raw_fps=50, factory='training.env_factory.create_training_environment', physics='unchanged'),
        selection='first qualifying pair in declared order; any selected counterexample blocks',
        resource_basis='Prior collector peaks below400MB; serial1GiB RSS/4GiB AS,256MiB memory reserve.',
        execution='two full natural episodes; accidents retained; no retry, replacement, or extra episode')


def interval(value):
    return (isinstance(value, list) and len(value) == 2 and all(type(v) in (int, float)
            and math.isfinite(v) for v in value) and value[0] <= value[1])


def first_qualifying(analysis):
    """Check saved labels, not a new scorer replay or a trusted PASS string alone."""
    rows = analysis.get('rows', [])
    if (analysis.get('schema') != 'haic-joint-envelope-pairs-v1-analysis'
            or analysis.get('operator_error') is not None
            or [r.get('cell') for r in rows] != [list(c) for c in PAIRED_CELLS]):
        raise ValueError('exact ordered three-pair analysis required')
    qualifying, selected = [], 0
    for row in rows:
        if row.get('status') == 'NOT_FOUND':
            if row.get('new_would_select'):
                raise ValueError('missing pair cannot be selected')
            continue
        if row.get('status') != 'COMPLETE' or row.get('parity', {}).get('passed') is not True:
            raise ValueError('partial or provenance-invalid pair blocks full episodes')
        if type(row.get('new_would_select')) is not bool:
            raise ValueError('explicit prospective selection label required')
        if not row['new_would_select']:
            continue
        selected += 1
        actual, predicted = row.get('actual_cost_interval'), row.get('new_predicted_interval')
        labels, tubes = row.get('physical_labels', []), row.get('containment', [])
        if (not interval(predicted) or predicted[1] >= -.05
                or row.get('actual_cost_supported') is not True or not interval(actual) or actual[1] >= -.05
                or row.get('actual_safe') is not True or row.get('contained') is not True
                or row.get('continuation') != 'exact_fixed_H4' or row.get('new_data_fit') is not False
                or len(labels) != 2 or labels[1].get('safe') is not True
                or labels[1].get('obstacle_contact') is not False or labels[1].get('all_wheels_lost') is not False
                or type(labels[1].get('damage_increase')) not in (int, float)
                or not math.isfinite(labels[1]['damage_increase']) or labels[1]['damage_increase'] > 0
                or len(tubes) != 2 or any(t.get('contained') is not True for t in tubes)):
            raise ValueError('selected false benefit, harm, range miss, or missing labels blocks full episodes')
        qualifying.append(row['cell'])
    gate = analysis.get('gate', {})
    if (not qualifying or gate.get('outcome') != 'PASS' or gate.get('potential_full_pilot') is not True
            or gate.get('blockers') != [] or gate.get('selected_pairs') != selected
            or gate.get('selected_evaluable_pairs') != selected
            or gate.get('supported_selected_benefit', {}).get('numerator') != len(qualifying)):
        raise ValueError('conditional full-episode gate not satisfied')
    if qualifying[0] != list(CELL):
        raise ValueError('first qualifying cell differs from the only authorized cell')
    return qualifying[0]


def saved_prediction_selects(comparison):
    """Collector predicate from saved masks only; no scorer or physics replay."""
    cost = comparison.get('cost_supported')
    bounds = comparison.get('delta_interval')
    if (type(comparison.get('common_support')) is not bool
            or not isinstance(cost, list) or len(cost) != 2 or any(type(v) is not bool for v in cost)
            or not isinstance(bounds, list) or len(bounds) != 2
            or (bounds[1] is not None and (not isinstance(bounds[1], list) or len(bounds[1]) != 2))):
        raise ValueError('complete saved common/cost/interval evidence required')
    for key in ('absolute_supported', 'veto'):
        mask = comparison.get(key)
        if (not isinstance(mask, list) or len(mask) != 2 or any(not isinstance(row, list)
                or len(row) != 19 or any(type(v) is not bool for v in row) for row in mask)):
            raise ValueError('complete saved two-candidate/nineteen-scenario physical masks required')
    upper: Any = bounds[1][1] if bounds[1] is not None else None
    return bool(comparison['common_support'] and all(cost)
        and type(upper) in (int, float) and math.isfinite(upper) and upper < -.05
        and all(comparison['absolute_supported'][1]) and not any(comparison['veto'][1]))


def gate_binding(analysis_path, calibration_sha256):
    analysis_path = Path(analysis_path).resolve()
    paths = dict(analysis=analysis_path, protocol=analysis_path.parent / 'protocol.json',
                 report=analysis_path.parent / 'report.json')
    hashes = {k: sha(p) for k, p in paths.items()}
    analysis, protocol, report = (read_json(paths[k]) for k in ('analysis', 'protocol', 'report'))
    if (analysis.get('protocol_sha256') != hashes['protocol'] or analysis.get('report_sha256') != hashes['report']
            or report.get('protocol_sha256') != hashes['protocol'] or report.get('operator_error') is not None
            or protocol.get('schema') != 'haic-joint-envelope-pairs-v1'
            or protocol.get('calibration_sha256') != calibration_sha256
            or protocol.get('cells') != [list(c) for c in PAIRED_CELLS]):
        raise ValueError('gate analysis/protocol/report/calibration binding differs')
    expected = [(t, s, a) for t, s in PAIRED_CELLS for a in ('baseline', 'repeat', 'alternative')]
    if [(r.get('track_id'), r.get('seed'), r.get('arm')) for r in report.get('rows', [])] != expected:
        raise ValueError('gate report allocation differs')
    if [r.get('cell') for r in analysis.get('rows', [])] != [list(c) for c in PAIRED_CELLS]:
        raise ValueError('gate analysis allocation differs')
    anchor_hashes = {}
    for index, row in enumerate(analysis['rows']):
        statuses = [r.get('status') for r in report['rows'][index * 3:index * 3 + 3]]
        required = ['COMPLETE'] * 3 if row['status'] == 'COMPLETE' else ['NOT_FOUND', 'SKIPPED_NOT_FOUND', 'SKIPPED_NOT_FOUND']
        if statuses != required:
            raise ValueError('gate row/report completion differs')
        track, seed = PAIRED_CELLS[index]
        anchor_path = analysis_path.parent / f'{track}-{seed}-baseline/anchor.json'
        name = str(anchor_path.relative_to(analysis_path.parent))
        if row['status'] == 'NOT_FOUND':
            if anchor_path.exists() or name in report['artifacts_sha256']:
                raise ValueError('NOT_FOUND pair must have no saved anchor')
            continue
        digest = sha(anchor_path)
        if report['artifacts_sha256'].get(name) != digest:
            raise ValueError('baseline anchor is not sealed in report')
        anchor = read_json(anchor_path)
        if (anchor.get('schema') != 'haic-joint-envelope-pairs-v1-anchor'
                or anchor.get('protocol_sha256') != hashes['protocol']
                or anchor.get('calibration_sha256') != calibration_sha256):
            raise ValueError('baseline anchor protocol/calibration binding differs')
        corrected = anchor.get('corrected')
        if not isinstance(corrected, dict):
            raise ValueError('saved corrected comparison missing')
        predicted = saved_prediction_selects(corrected)
        if (row.get('new_would_select') is not predicted
                or row.get('new_predicted_interval') != corrected['delta_interval'][1]):
            raise ValueError('analysis selection/interval differs from sealed anchor prediction')
        anchor_hashes[str(anchor_path)] = digest
    selected = first_qualifying(analysis)
    # Bind only the selected initial catalog/image, not all previous raw streams.
    directory = analysis_path.parent / f'{CELL[0]}-{CELL[1]}-baseline'
    static_path, boundaries_path = directory / 'static.json', directory / 'boundaries.jsonl'
    for path in (static_path, boundaries_path):
        if report['artifacts_sha256'].get(str(path.relative_to(analysis_path.parent))) != sha(path):
            raise ValueError('selected baseline initial evidence is not sealed in report')
    static = read_json(static_path)
    with boundaries_path.open() as stream:
        initial = json.loads(stream.readline())
    if initial['index'] != 0:
        raise ValueError('selected paired baseline lacks reset endpoint')
    reference = dict(track=static['track'], obstacles=[dict(x=b['position'][0], y=b['position'][1],
        radius=b['fixtures'][0]['radius']) for b in static['bodies']['obstacle']],
        initial_observation_sha256=initial['image_sha256'])
    return dict(paths={k: str(v) for k, v in paths.items()}, sha256=hashes, selected_cell=selected,
                anchor_source_sha256=anchor_hashes, initial_reference=reference,
                initial_source_sha256={str(p): sha(p) for p in (static_path, boundaries_path)})


def validate_calibration(value, root):
    prior = Path(root) / 'runs/joint-temporal-interval-v1/calibration.json'
    if (sha(prior) != ABSOLUTE_SHA or value.get('absolute_calibration') != read_json(prior)
            or value.get('schema') != 'haic-joint-temporal-paired-envelope-calibration-v1'
            or value.get('calibration_complete') is not True
            or any(value.get('performance', {}).get(k) != .05 for k in ('lower', 'upper', 'floor'))):
        raise ValueError('exact new performance calibration and unchanged nested absolute calibration required')
    old.validate_calibration(value['absolute_calibration'], root)
    old.check_pins(value.get('source_pins'))
    old.check_pins(value.get('evidence_pins'))
    if str(Path(root) / COMPARATOR) not in value['source_pins']:
        raise ValueError('new paired-envelope source pin missing')


def validate_claim(claim, output, gate_hashes, *, root=ROOT):
    expected = dict(schema='haic-joint-envelope-pilot-consumed-train-v1', partition='TRAIN', fresh=False,
        cells=[list(CELL)], max_resets=2, run_directory=str(Path(output).resolve()),
        calibration_sha256=CALIBRATION_SHA, gate_sha256=gate_hashes)
    if any(claim.get(k) != v for k, v in expected.items()):
        raise ValueError('exact two-reset consumed-TRAIN claim required')
    old.check_pins(claim.get('source_sha256'))
    old.check_pins(claim.get('evidence_sha256'))
    if not {str(Path(root) / n) for n in (OPERATOR, SUCCESSOR, COMPARATOR, old.OPERATOR)} <= set(claim['source_sha256']):
        raise ValueError('claim must bind new runner/adapter/comparator and immutable reused runner')


def freeze(output, calibration, claim, gate_analysis):
    output, calibration, claim = [Path(p).resolve() for p in (output, calibration, claim)]
    if output.exists() or not output.parent.is_dir() or sha(calibration) != CALIBRATION_SHA:
        raise ValueError('exclusive output and exact new calibration required')
    validate_calibration(read_json(calibration), ROOT)
    gate = gate_binding(gate_analysis, CALIBRATION_SHA)
    validate_claim(read_json(claim), output, gate['sha256'])
    sources = old.source_closure(ROOT, [OPERATOR, SUCCESSOR], packages=('scripts', 'haic'))
    environment = old.source_closure(old.SNAPSHOT, ['training/evaluate_closed_loop.py', 'training/env_factory.py'], exclude=('agent',))
    members = old.champion_members()
    output.mkdir()
    for name, digest in sources.items():
        path = local_path(output / 'source', name)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
        if sha(path) != digest:
            raise ValueError('source changed during freeze')
    for name, blob in members.items():
        path = local_path(output / 'model', name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(blob)
    for origin, name in ((calibration, 'calibration.json'), (claim, 'claim.json'), (old.BUNDLE / 'submission.zip', 'champion.zip')):
        shutil.copyfile(origin, output / name)
    protocol = dict(contract(), repository_root=str(ROOT), output_directory=str(output), gate_binding=gate,
        source_sha256=sources, environment_sha256={str(old.SNAPSHOT / n): h for n, h in environment.items()},
        model_sha256={n: hashlib.sha256(b).hexdigest() for n, b in members.items()},
        preserved_sha256={str(old.BUNDLE / 'source' / n): hashlib.sha256(b).hexdigest() for n, b in members.items()},
        champion_zip=str(old.BUNDLE / 'submission.zip'), calibration_path=str(calibration), claim_path=str(claim),
        claim_sha256=sha(claim), runtime_sha256=sha(old.PYTHON.resolve()))
    save(output / 'protocol.json', protocol)
    digest = sha(output / 'protocol.json')
    validate_frozen(output, digest)
    return digest


def validate_frozen(output, digest):
    output = Path(output).resolve()
    if sha(output / 'protocol.json') != digest:
        raise ValueError('protocol hash differs')
    p = read_json(output / 'protocol.json')
    if any(p.get(k) != v for k, v in contract().items()) or p.get('output_directory') != str(output):
        raise ValueError('fixed two-episode contract differs')
    for name, expected in p['source_sha256'].items():
        for root in (output / 'source', Path(p['repository_root'])):
            if sha(local_path(root, name)) != expected:
                raise ValueError('source hash differs: ' + name)
    old.check_pins(p['environment_sha256'])
    old.check_pins(p['preserved_sha256'])
    for name, expected in p['model_sha256'].items():
        if sha(local_path(output / 'model', name)) != expected:
            raise ValueError('champion member differs')
    for filename, origin, expected in (('champion.zip', p['champion_zip'], old.CHAMPION_SHA),
            ('calibration.json', p['calibration_path'], CALIBRATION_SHA), ('claim.json', p['claim_path'], p['claim_sha256'])):
        if sha(output / filename) != expected or sha(origin) != expected:
            raise ValueError('bound artifact differs: ' + filename)
    gate = gate_binding(p['gate_binding']['paths']['analysis'], CALIBRATION_SHA)
    if gate != p['gate_binding']:
        raise ValueError('conditional gate evidence changed')
    validate_claim(read_json(output / 'claim.json'), output, gate['sha256'], root=p['repository_root'])
    validate_calibration(read_json(output / 'calibration.json'), p['repository_root'])
    if sha(old.PYTHON.resolve()) != p['runtime_sha256']:
        raise ValueError('CPU21 executable differs')
    return p


def validate_admission(value, digest, p, *, now=None, starting=False):
    now = time.time() if now is None else now
    expected = dict(schema='haic-joint-envelope-pilot-admission-v1', admitted=True,
        protocol_sha256=digest, claim_sha256=p['claim_sha256'], calibration_sha256=CALIBRATION_SHA)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('current two-reset run admission required')
    created, expires = value.get('created_unix_s'), value.get('expires_unix_s')
    if (type(created) not in (int, float) or type(expires) not in (int, float)
            or not math.isfinite(created) or not math.isfinite(expires)
            or not created <= now < expires <= created + 1200 or (starting and now - created > 300)):
        raise ValueError('admission expired or not current')


def worker_command(output, index, digest, *, import_only=False):
    if index not in (0, 1):
        raise ValueError('only two serial slots exist')
    return [str(old.PYTHON), '-I', '-B', '-c',
        'import sys; sys.path.insert(0,sys.argv[1]); from scripts.run_joint_envelope_pilot import worker; '
        'worker(sys.argv[2],int(sys.argv[3]),sys.argv[4],import_only=sys.argv[5]=="True")',
        str(Path(output).resolve() / 'source'), str(Path(output).resolve()), str(index), digest, str(import_only)]


def load_runtime(output, p, mode):
    # Load only the untouched champion through the old strict external loader.
    # Passing successor here would feed nested new calibration into the old agent.
    champion, evaluator, factory, audit, footprint = old.load_runtime(output, p, 'baseline', CELL[1])
    if mode == 'successor':
        from haic.algorithms.joint_control.envelope_successor import EnvelopeSuccessor
        champion = EnvelopeSuccessor(champion, read_json(output / 'calibration.json'))
    elif mode != 'baseline':
        raise ValueError('unknown arm')
    pins = {str(output / 'source' / n): h for n, h in p['source_sha256'].items()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).resolve().is_relative_to(output / 'source'):
            path = str(Path(filename).resolve())
            if pins.get(path) != sha(path):
                raise ValueError('new adapter imported outside frozen closure')
    return champion, evaluator, factory, audit, footprint


def check_initial(reference, actual):
    catalog = actual['catalog']
    obstacles = [{k: o[k] for k in ('x', 'y', 'radius')} for o in catalog['obstacles']]
    if (catalog['track'] != reference['track'] or obstacles != reference['obstacles']
            or actual['initial_observation_sha256'] != reference['initial_observation_sha256']):
        raise ValueError('qualifying paired catalog/initial-image differs; dynamic time-limit state is not compared')


def worker(output, index, digest, *, import_only=False):
    output = Path(output).resolve()
    p = validate_frozen(output, digest)
    if index not in (0, 1):
        raise ValueError('only two serial slots exist')
    slot, path = schedule()[index], output / schedule()[index]['file']
    resource.setrlimit(resource.RLIMIT_AS, (LIMITS['address_space_bytes'], LIMITS['address_space_bytes']))
    signal.alarm(600)
    model, evaluator, factory, audit, footprint = load_runtime(output, p, slot['mode'])
    if import_only:
        print(json.dumps(dict(mode=slot['mode'], import_only=True, environment_resets=0,
            evaluator_file=str(old.SNAPSHOT / 'training/evaluate_closed_loop.py'), memory=old.memory_sample())))
        return
    start = read_json(output / 'run-start.json')
    if (start.get('protocol_sha256') != digest or start.get('preflight_sha256') != sha(output / 'preflight.json')
            or start.get('admission_sha256') != sha(output / 'admission.json')):
        raise ValueError('bound one-use run-start required')
    old.validate_preflight(read_json(output / 'preflight.json'), digest)
    ledger = output / 'reset-ledger.jsonl'
    if ledger.exists() and any(r['file'] == slot['file'] for r in old.json_rows(ledger)):
        raise ValueError('existing reset intent; never retry')
    baseline = read_json(output / schedule()[0]['file']) if index else None
    if baseline is not None and (not old.natural_episode(baseline) or baseline.get('protocol_sha256') != digest):
        raise ValueError('completed bound baseline required')
    holder = {'reference_decisions': old.decisions(output, baseline)} if baseline else {}
    raw_path, decision_path = path.with_suffix('.raw.jsonl'), path.with_suffix('.decisions.jsonl')

    def admit(phase):
        validate_frozen(output, digest)
        validate_admission(read_json(output / 'admission.json'), digest, p)
        append(output / 'resources.jsonl', dict(slot=index, phase=phase, **old.resource_admission(output, child=True)))

    with raw_path.open('x', buffering=1) as raw, decision_path.open('x', buffering=1) as stream:
        def environment_factory(**kwargs):
            if kwargs != dict(track_id=CELL[0], seed=CELL[1], max_decisions=1200) or 'environment' in holder:
                raise ValueError('one unchanged1200-decision environment required')
            admit('before_environment')
            observer = audit.make_observer(factory(**kwargs), raw, ledger, slot, digest, footprint)
            reset, raw_reset, raw_step = observer.reset, observer.unwrapped.reset, observer.unwrapped.step
            observer.warmup_raw = 0

            def guarded_raw_step(action):
                if not observer.active:
                    if observer.warmup_raw >= 51:
                        raise ValueError('extra warmup raw tick')
                    observer.warmup_raw += 1
                return raw_step(action)

            def guarded_raw_reset(*args, **kwargs):
                if observer.reset_count:
                    raise ValueError('second reset forbidden')
                admit('before_reset')
                return raw_reset(*args, **kwargs)  # Existing audit writes reset intent BEFORE delegation.

            def checked_reset():
                result = reset()
                initial = old.initial_record(observer)
                save(path.with_suffix('.initial.json'), dict(protocol_sha256=digest, warmup_raw=observer.warmup_raw, **initial))
                if observer.warmup_raw != 51:
                    raise ValueError('unchanged reset must include exactly51 raw ticks')
                check_initial(p['gate_binding']['initial_reference'], initial)
                if baseline is not None and any(baseline.get(k) != v for k, v in initial.items()):
                    raise ValueError('full-episode static/initial/image parity differs')
                return result

            observer.unwrapped.step, observer.unwrapped.reset, observer.reset = guarded_raw_step, guarded_raw_reset, checked_reset
            holder['environment'] = observer
            return observer

        result = evaluator(mode=slot['mode'], track_id=CELL[0], seed=CELL[1],
            agent=old.MeasuredAgent(model, stream, holder, slot['mode']), max_decisions=1200,
            plan_budget_seconds=4.5, capture_trace=True, fail_on_invalid_action=True, environment_factory=environment_factory)
    observer = holder.get('environment')
    initial = getattr(observer, 'initial', None)
    states = ([initial] if initial else []) + old.json_rows(raw_path)
    result.update(protocol_sha256=digest, scope='one consumed TRAIN road; not replicated/official',
        catalog=getattr(observer, 'catalog', None), geometry_sha256=getattr(observer, 'geometry_sha256', None),
        initial_state=initial, initial_observation_sha256=getattr(observer, 'initial_observation_sha256', None),
        warmup_raw=getattr(observer, 'warmup_raw', None), raw_trace_file=raw_path.name, raw_trace_sha256=sha(raw_path),
        decision_stream_file=decision_path.name, decision_stream_sha256=sha(decision_path),
        physical=audit.legacy.physical_metrics(states, getattr(observer, 'obstacles', [])),
        memory=old.memory_sample(), natural_episode=old.natural_episode(result))
    save(path, result)
    append(ledger, dict(slot, event='episode_end', protocol_sha256=digest,
                       natural_episode=result['natural_episode'], error=result['error']))


def preflight(output, digest):
    output = Path(output).resolve()
    validate_frozen(output, digest)
    if (output / 'preflight.json').exists():
        raise ValueError('preflight already exists')
    rows = []
    for index in (0, 1):
        child = subprocess.run(worker_command(output, index, digest, import_only=True), cwd=output / 'model',
            env={**os.environ, **old.THREAD_ENV}, capture_output=True, text=True, timeout=60)
        if child.returncode:
            raise RuntimeError(child.stderr)
        rows.append(json.loads(child.stdout))
    result = dict(protocol_sha256=digest, environment_resets=0, arms=rows)
    old.validate_preflight(result, digest)
    save(output / 'preflight.json', result)
    return result


def execute(output, digest, admission):
    output = Path(output).resolve()
    p, value = validate_frozen(output, digest), read_json(admission)
    old.validate_preflight(read_json(output / 'preflight.json'), digest)
    validate_admission(value, digest, p, starting=True)
    if any((output / n).exists() for n in ('run-start.json', 'admission.json', 'episode-report.json', 'reset-ledger.jsonl')) \
            or any(list(output.glob(Path(s['file']).stem + '.*')) for s in schedule()):
        raise ValueError('existing execution evidence; no retry or overwrite')
    save(output / 'admission.json', value)
    save(output / 'run-start.json', dict(protocol_sha256=digest, time_ns=time.time_ns(),
        preflight_sha256=sha(output / 'preflight.json'), admission_sha256=sha(output / 'admission.json')))
    rows, started, error = [dict(s, status='unrun') for s in schedule()], time.monotonic(), None
    try:
        for index, row in enumerate(rows):
            validate_frozen(output, digest)
            validate_admission(value, digest, p)
            append(output / 'resources.jsonl', dict(slot=index, phase='before_child', **old.resource_admission(output)))
            remaining = min(1200 - (time.monotonic() - started), value['expires_unix_s'] - time.time())
            if remaining <= 0:
                raise TimeoutError('fixed1200s total deadline')
            row['status'] = 'started'
            path, child_error, returncode, began = output / row['file'], None, None, time.monotonic()
            try:
                with path.with_suffix('.stdout.log').open('x') as stdout, path.with_suffix('.stderr.log').open('x') as stderr:
                    child = subprocess.run(worker_command(output, index, digest), cwd=output / 'model',
                        env={**os.environ, **old.THREAD_ENV}, stdout=stdout, stderr=stderr, timeout=min(600, remaining))
                returncode = child.returncode
                if returncode:
                    raise RuntimeError(f'worker exited {returncode}')
                episode = read_json(path)
                row['sha256'] = sha(path)
                if not old.natural_episode(episode):
                    raise RuntimeError('execution/action/resource failure: ' + str(episode.get('retire_reason')))
                if index:
                    row['parity'] = old.pair_parity(read_json(output / rows[0]['file']), episode, output)
                    if not row['parity']['exact_prefix']:
                        raise ValueError('pre-divergence provenance failure')
                row['status'] = 'completed'
            except BaseException as failure:
                row['status'], child_error = 'failed', f'{type(failure).__name__}: {failure}'
                raise
            finally:
                receipt = path.with_suffix('.process.json')
                save(receipt, dict(protocol_sha256=digest, slot=index, returncode=returncode,
                    error=child_error, wall_seconds=time.monotonic() - began))
                row.update(process_file=receipt.name, process_sha256=sha(receipt))
    except BaseException as failure:
        error = f'{type(failure).__name__}: {failure}'
    finally:
        ledger = output / 'reset-ledger.jsonl'
        result = dict(protocol_sha256=digest, rows=rows, operator_error=error, wall_seconds=time.monotonic() - started,
            artifacts_sha256={p.name: sha(p) for r in rows for p in output.glob(Path(r['file']).stem + '.*') if p.is_file()},
            reset_ledger_sha256=sha(ledger) if ledger.exists() else None)
        save(output / 'episode-report.json', result)
    return result


def partial_stream(path):
    rows, truncated = [], False
    if Path(path).exists():
        lines = Path(path).read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise ValueError('malformed nonfinal partial log record')
                truncated = True
    return rows, truncated


def policy_metrics(rows, mode) -> dict[str, Any]:
    n = len(rows)
    changed = sum(r['policy'].get('intervention') is True for r in rows)
    eligible = sum(r['policy'].get('eligible') is True for r in rows)
    compared = [r for r in rows if r['policy'].get('comparison') is not None]
    negative = positive = ambiguous = unsupported = 0
    for row in compared:
        bounds = row['policy']['comparison'].get('delta_interval', [None, None])[1]
        if not interval(bounds):
            unsupported += 1
        elif bounds[1] < -.05:
            negative += 1
        elif bounds[0] > .05:
            positive += 1
        else:
            ambiguous += 1
    if changed > 40:
        raise ValueError('successor exceeds fixed40 intervention budget')
    actual_differences = sum(r['action'] != r['policy'].get('proposal_action', r['action']) for r in rows)
    if mode == 'successor' and changed != actual_differences:
        raise ValueError('intervention labels differ from actual issued/proposed actions')
    return dict(decisions=n, eligible=rate(eligible, n), interventions=rate(changed, n),
        abstain=rate(n - changed, n) if mode == 'successor' else None,
        comparisons=rate(len(compared), n), candidate_orders=dict(alternative=negative, baseline=positive,
            ambiguous=ambiguous, unsupported=unsupported, denominator=len(compared)),
        actual_proposal_differences=actual_differences,
        reasons={reason: rate(sum(reason in r['policy'].get('reasons', []) for r in rows), n)
                 for reason in sorted({f for r in rows for f in r['policy'].get('reasons', [])})})


def analyze(output, digest):
    output = Path(output).resolve()
    p = validate_frozen(output, digest)
    report = read_json(output / 'episode-report.json')
    if report.get('protocol_sha256') != digest or len(report.get('rows', [])) != 2:
        raise ValueError('bound two-slot report required')
    for name, expected in report['artifacts_sha256'].items():
        if sha(local_path(output, name)) != expected:
            raise ValueError('complete/partial artifact differs')
    ledger_path = output / 'reset-ledger.jsonl'
    if (sha(ledger_path) if ledger_path.exists() else None) != report['reset_ledger_sha256']:
        raise ValueError('reset ledger differs')
    ledger = old.json_rows(ledger_path) if ledger_path.exists() else []
    intents = [r for r in ledger if r.get('event') == 'reset_intent']
    if len(intents) > 2 or len({r['file'] for r in intents}) != len(intents):
        raise ValueError('duplicate/excess reset evidence')
    for entry in ledger:
        if (entry.get('protocol_sha256') != digest or entry.get('file') not in {s['file'] for s in schedule()}
                or entry.get('event') not in ('reset_intent', 'episode_end')):
            raise ValueError('unknown reset ledger identity')
    results, episodes = [], []
    absolute = read_json(output / 'calibration.json')['absolute_calibration']
    for slot, record in zip(schedule(), report['rows']):
        if any(record.get(k) != v for k, v in slot.items()):
            raise ValueError('report schedule differs')
        path = output / slot['file']
        if record.get('process_file') and sha(local_path(output, record['process_file'])) != record['process_sha256']:
            raise ValueError('process receipt differs')
        stream, truncated = partial_stream(path.with_suffix('.decisions.jsonl'))
        rows, calls = [r for r in stream if r['event'] == 'decision'], [r for r in stream if r['event'] == 'act']
        metrics = policy_metrics(rows, slot['mode'])
        result: dict[str, Any] = dict(slot=slot, status=record['status'], partial=True, truncated_decision_tail=truncated, **metrics,
            act_timing_censored=len({r['step'] for r in stream if r['event'] == 'act_intent'} - {r['step'] for r in calls}),
            integrated_act=dict(cpu_seconds=old.distribution([r['cpu_seconds'] for r in calls]),
                wall_seconds=old.distribution([r['wall_seconds'] for r in calls]),
                all_over_1s=[r for r in calls if r['cpu_seconds'] > 1 or r['wall_seconds'] > 1]))
        try:
            episode = read_json(path) if path.exists() else None
        except json.JSONDecodeError:
            if record['status'] == 'completed':
                raise ValueError('completed episode JSON is malformed')
            episode = None
            result['partial_episode_json'] = True
        episodes.append(episode)
        if episode is not None:
            if episode.get('protocol_sha256') != digest or (record.get('sha256') and sha(path) != record['sha256']):
                raise ValueError('episode binding differs')
            if rows != old.decisions(output, episode) or truncated:
                raise ValueError('completed decision stream differs')
            if old.natural_episode(episode) and len(rows) != episode['steps']:
                raise ValueError('episode/decision denominator differs')
            if record['status'] == 'completed' and any(sum(r['file'] == slot['file'] and r['event'] == event for r in ledger) != 1
                                                      for event in ('reset_intent', 'episode_end')):
                raise ValueError('completed slot requires one reset and one end')
            raw = old.raw_states(output, episode)
            for row in rows:
                forecast = row['policy'].get('forecast')
                if forecast and any(forecast[key] != absolute[key] for key in ('position_residual', 'yaw_residual')):
                    raise ValueError('forecast changed unchanged absolute calibration')
            failures = old.failure_events(rows, raw, episode)
            failures['intervention_exposure'] = metrics['interventions']['numerator']
            if not failures['intervention_exposure']:
                failures['after_intervention'] = dict(numerator=None, denominator=0, rate=None)
            result.update(partial=not old.natural_episode(episode), completed=episode['completed'],
                lap_time_ms=episode['lapTimeMs'] if episode['completed'] else None,
                finished_lap_mean_ms=episode['lapTimeMs'] if episode['completed'] else None,
                completion=rate(int(episode['completed']), 1), progress=episode['progress'],
                retire_reason=episode['retire_reason'], damage=episode['damage'],
                collision_positive_decisions=episode['collisions'], physical=episode['physical'], memory=episode['memory'],
                forecast=old.forecast_ranges(rows, raw), failures=failures)
        results.append(result)
    pair: dict[str, Any] = dict(cell=list(CELL), matched=False, lap_delta_ms=None)
    if all(e is not None for e in episodes):
        left, right = episodes
        parity = old.pair_parity(left, right, output)
        matched = (all(r['status'] == 'completed' for r in report['rows'])
                   and all(old.natural_episode(e) for e in episodes) and parity['exact_prefix'])
        both = left['completed'] and right['completed']
        pair.update(matched=matched, parity=parity, both_finished=both,
                    lap_delta_ms=right['lapTimeMs'] - left['lapTimeMs'] if matched and both else None)
    changed = results[1]['actual_proposal_differences']
    evaluated = pair['matched'] and changed > 0
    result = dict(schema=SCHEMA + '-analysis', protocol_sha256=digest,
        episode_report_sha256=sha(output / 'episode-report.json'), gate_sha256=p['gate_binding']['sha256'],
        reset_intents=len(intents), complete_two_natural_episodes=pair['matched'], episodes=results, pair=pair,
        successor_effect_evaluated=evaluated, effect_status='EVALUATED_ONE_ROAD' if evaluated else 'NOT_EVALUATED',
        operator_error=report['operator_error'], qualification='One consumed TRAIN road only, not replicated, official, or safety proof. Range mismatch is not an accident; post-intervention association is not causality.')
    save(output / 'summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('freeze', 'preflight', 'execute', 'analyze'):
        sub = commands.add_parser(name)
        sub.add_argument('--output', required=True, type=Path)
        if name == 'freeze':
            for arg in ('calibration', 'claim', 'gate-analysis'):
                sub.add_argument('--' + arg, required=True, type=Path)
        else:
            sub.add_argument('--protocol-sha256', required=True)
        if name == 'execute':
            sub.add_argument('--admission', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        result = dict(protocol_sha256=freeze(args.output, args.calibration, args.claim, args.gate_analysis), environment_resets=0)
    elif args.command == 'execute':
        result = execute(args.output, args.protocol_sha256, args.admission)
    else:
        result = globals()[args.command](args.output, args.protocol_sha256)
    print(json.dumps(result, allow_nan=False, indent=2))
    return int(args.command == 'execute' and result['operator_error'] is not None)


if __name__ == '__main__':
    raise SystemExit(main())
