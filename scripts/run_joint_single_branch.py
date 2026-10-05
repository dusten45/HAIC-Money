"""One new B reset; A/C and the fixed-H4 pair are explicitly REUSED evidence.

Only execute can construct/reset an environment. Private function copies reuse
immutable freeze/validation plumbing, never change module globals or episode IDs.
The inherited protocol field gate_binding.paths.analysis names the reuse analysis
certificate, not a newly run A/C episode. All reference paths retain old identities.
"""

import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import time
from types import FunctionType
from typing import Any

from scripts import run_joint_envelope_pilot as envelope
from scripts import run_joint_temporal_pilot as old

ROOT = Path(__file__).resolve().parents[1]
OPERATOR = 'scripts/run_joint_single_branch.py'
SUCCESSOR = 'haic/algorithms/joint_control/single_intervention.py'
SCHEMA = 'haic-joint-single-branch-v1'
CELL, STEP = (3, 3184000005), 34
SUMMARY_SHA = '72dad2216d6f8d772c60f72e96206ab4838f5967589889da3fae3c62fd4f204f'
REFERENCE_PROTOCOL_SHA = 'a2f07616ea1830d82322d813c0c2c1da45691da27a947d1da6cbc88e2f47504e'
LIMITS = dict(old.LIMITS, max_resets=1, total_seconds=600)
sha, read_json, save, append = old.sha, old.read_json, old.save, old.append
CHECKS = ('source_calibration_environment', 'initial_static', 'first33_and_pre34',
          'fixed_h4_same_anchor', 'fixed_h4_float32_actions', 'first_h1_physical_intersection')


def slot() -> dict[str, Any]:
    return dict(track_id=CELL[0], seed=CELL[1], mode='successor', branch='B', repeat=0,
                file='3-3184000005-single.json')


def contract() -> dict[str, Any]:
    return dict(schema=SCHEMA, cells=[list(CELL)], schedule=[slot()], limits=LIMITS,
        intervention_step=STEP, max_interventions=1, partition='TRAIN', fresh=False, official_action=False,
        python=str(old.PYTHON), snapshot=str(old.SNAPSHOT), champion_zip_sha256=old.CHAMPION_SHA,
        calibration_sha256=envelope.CALIBRATION_SHA, absolute_calibration_sha256=envelope.ABSOLUTE_SHA,
        environment=envelope.contract()['environment'], reference_roles=dict(A='REUSED', B='NEW', C='REUSED'),
        resource_basis='Prior matched episodes below400MB;1GiB RSS/4GiB AS,256MiB reserve,one600s child.',
        continuation='C34 exactly once; all other actions are fresh champion proposals on actual B observations')


def action_bytes(action):
    if len(action) != 3 or any(not math.isfinite(v) for v in action):
        raise ValueError('finite three-component float32 action required')
    return struct.pack('<3f', *action)


def decision_prefix(path, count=STEP):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['event'] == 'decision':
                rows.append(row)
                if len(rows) == count:
                    break
    if [r['step'] for r in rows] != list(range(1, count + 1)):
        raise ValueError('complete ordered decision prefix required')
    return rows


def reference_binding(certificate_path, calibration_sha256):
    certificate_path = Path(certificate_path).resolve()
    cert = read_json(certificate_path)
    if (cert.get('schema') != 'haic-joint-single-branch-reuse-v1' or cert.get('reusable') is not True
            or cert.get('cell') != list(CELL) or cert.get('intervention_step') != STEP
            or cert.get('calibration_sha256') != calibration_sha256
            or any(cert.get('checks', {}).get(k) is not True for k in CHECKS)):
        raise ValueError('completed source-bound A/C/fixed-H4 reuse certificate required')
    root = Path(cert['references_directory']).resolve()
    paths = {k: root / name for k, name in (('protocol', 'protocol.json'), ('summary', 'summary.json'), ('report', 'episode-report.json'))}
    hashes = {k: sha(p) for k, p in paths.items()}
    hashes['certificate'] = sha(certificate_path)
    if hashes['summary'] != SUMMARY_SHA or hashes['protocol'] != REFERENCE_PROTOCOL_SHA:
        raise ValueError('only the fixed existing A/C study can be reused')
    pins = cert['evidence_sha256']
    old.check_pins(pins)
    report, protocol = read_json(paths['report']), read_json(paths['protocol'])
    if (report['protocol_sha256'] != hashes['protocol'] or report['operator_error'] is not None
            or protocol['calibration_sha256'] != calibration_sha256
            or protocol['environment'] != contract()['environment'] or protocol.get('cells') != [list(CELL)]
            or protocol.get('champion_zip_sha256') != old.CHAMPION_SHA):
        raise ValueError('reference source/calibration/environment identity differs')
    required = list(paths.values())
    refs, prefix, episodes = {}, {}, {}
    for label, mode in (('A', 'baseline'), ('C', 'successor')):
        name = f'3-3184000005-{mode}.json'
        path, record = root / name, next(r for r in report['rows'] if r['file'] == name)
        episode = read_json(path)
        if (record['status'] != 'completed' or record['sha256'] != sha(path)
                or episode['protocol_sha256'] != hashes['protocol'] or not old.natural_episode(episode)):
            raise ValueError('natural report-bound A/C reference required')
        files = [path, old.local_path(root, episode['decision_stream_file']), old.local_path(root, episode['raw_trace_file'])]
        required.extend(files)
        if (sha(files[1]) != episode['decision_stream_sha256'] or sha(files[2]) != episode['raw_trace_sha256']):
            raise ValueError('reference raw/decision receipt differs')
        refs[label] = dict(provenance='REUSED', root=str(root), episode_path=str(path), episode_sha256=sha(path))
        episodes[label] = episode
        prefix[label] = decision_prefix(files[1])
    if any(episodes['A'].get(k) != episodes['C'].get(k) for k in
           ('catalog', 'initial_state', 'geometry_sha256', 'initial_observation_sha256')):
        raise ValueError('A/C full initial/static/image identity differs')
    for a, c in zip(prefix['A'], prefix['C']):
        if a['observation_sha256'] != c['observation_sha256'] or a['evaluation_only']['pre'] != c['evaluation_only']['pre']:
            raise ValueError('A/C pre34 image/state identity differs')
        if a['step'] < STEP and (action_bytes(a['action']) != action_bytes(c['action']) or a['evaluation_only'] != c['evaluation_only']):
            raise ValueError('A/C first33 action/physical prefix differs')
    if [r['step'] for r in prefix['C'] if r['policy'].get('intervention')] != [STEP]:
        raise ValueError('C first intervention must be decision34')
    anchor_path = Path(cert['fixed_h4']['anchor_path']).resolve()
    required.append(anchor_path)
    if any(pins.get(str(path)) != sha(path) for path in required):
        raise ValueError('certificate does not bind every required reference')
    anchor = read_json(anchor_path)
    if (anchor['completed_prefix'] != STEP - 1 or anchor['calibration_sha256'] != calibration_sha256
            or action_bytes(anchor['actions'][0]) != action_bytes(prefix['A'][-1]['action'])
            or action_bytes(anchor['actions'][1]) != action_bytes(prefix['C'][-1]['action'])):
        raise ValueError('certified fixed-H4 anchor/action does not match online34')
    return dict(paths=dict(analysis=str(certificate_path)), sha256=hashes, references=refs,
        expected_action=prefix['C'][-1]['action'], fixed_h4=dict(provenance='REUSED', anchor_path=str(anchor_path),
            baseline_tail=[anchor['actions'][0]] * 4, alternative_tail=[anchor['actions'][1]] + [anchor['actions'][0]] * 3))


def validate_claim(claim, output, gate_hashes, *, root=ROOT):
    expected = dict(schema='haic-joint-single-branch-consumed-train-v1', partition='TRAIN', fresh=False,
        cells=[list(CELL)], intervention_step=STEP, max_resets=1, run_directory=str(Path(output).resolve()),
        calibration_sha256=envelope.CALIBRATION_SHA, gate_sha256=gate_hashes)
    if any(claim.get(k) != v for k, v in expected.items()):
        raise ValueError('exact one-reset B claim required')
    old.check_pins(claim.get('source_sha256'))
    old.check_pins(claim.get('evidence_sha256'))
    if not {str(Path(root) / n) for n in (OPERATOR, SUCCESSOR, envelope.OPERATOR, envelope.SUCCESSOR, old.OPERATOR)} <= set(claim['source_sha256']):
        raise ValueError('claim lacks new/immutable reused source pins')


def frozen_operation(function):
    """Only dependency substitution; original code/globals/defaults stay immutable."""
    namespace = dict(function.__globals__, ROOT=ROOT, OPERATOR=OPERATOR, SUCCESSOR=SUCCESSOR,
        contract=contract, gate_binding=reference_binding, validate_claim=validate_claim, validate_frozen=validate_frozen)
    return FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)


def freeze(output, calibration, claim, references, reuse_certificate):
    if Path(read_json(reuse_certificate)['references_directory']).resolve() != Path(references).resolve():
        raise ValueError('references directory differs from certificate')
    return frozen_operation(envelope.freeze)(output, calibration, claim, reuse_certificate)


def validate_frozen(output, digest):
    p = frozen_operation(envelope.validate_frozen)(output, digest)
    prior = read_json(Path(p['gate_binding']['references']['A']['root']) / 'protocol.json')
    if (any(p[key] != prior[key] for key in ('environment_sha256', 'model_sha256', 'runtime_sha256'))
            or any(p['source_sha256'][name] != expected for name, expected in prior['source_sha256'].items()
                   if name in p['source_sha256'])):
        raise ValueError('reused source/environment/runtime differs from A/C protocol')
    return p


def validate_admission(value, digest, p, *, starting=False):
    expected = dict(schema='haic-joint-single-branch-admission-v1', admitted=True, protocol_sha256=digest,
                    claim_sha256=p['claim_sha256'], calibration_sha256=envelope.CALIBRATION_SHA)
    created, expires, now = value.get('created_unix_s'), value.get('expires_unix_s'), time.time()
    if (any(value.get(k) != v for k, v in expected.items()) or type(created) not in (int, float)
            or type(expires) not in (int, float) or not math.isfinite(created) or not math.isfinite(expires)
            or not created <= now < expires <= created + 600 or (starting and now - created > 300)):
        raise ValueError('current <=600s one-reset admission required')


def worker_command(output, digest, *, import_only=False):
    return [str(old.PYTHON), '-I', '-B', '-c', 'import sys; sys.path.insert(0,sys.argv[1]); '
        'from scripts.run_joint_single_branch import worker; worker(sys.argv[2],sys.argv[3],import_only=sys.argv[4]=="True")',
        str(Path(output).resolve() / 'source'), str(Path(output).resolve()), digest, str(import_only)]


def validate_preflight(receipt, digest):
    if (receipt.get('protocol_sha256') != digest or receipt.get('branch') != 'B'
            or receipt.get('import_only') is not True or receipt.get('environment_resets') != 0
            or receipt.get('evaluator_file') != str(old.SNAPSHOT / 'training/evaluate_closed_loop.py')):
        raise ValueError('one bound zero-reset B import preflight required')


class MeasuredBranch(old.MeasuredAgent):
    def act(self, observation):
        action = super().act(observation)
        if self.calls < STEP and action_bytes(action) != action_bytes(
                self.holder['reference_decisions'][self.calls - 1]['action']):
            raise ValueError('B action diverged before the declared decision34 fork')
        return action

    def last_step_diagnostics(self):
        result = super().last_step_diagnostics()
        info = self.model.last_diagnostics
        required = self.holder['C34']['action'] if self.calls == STEP else info['proposal_action']
        if action_bytes(self.action) != action_bytes(required) or info.get('intervention') is not (self.calls == STEP):
            raise ValueError('B must intervene exactly34 and issue fresh proposals otherwise')
        if self.calls == STEP:
            self.holder['raw_stream'].flush()
            raw = old.json_rows(self.holder['raw_path'])
            if (self.holder['environment'].last != self.holder['C34']['evaluation_only']
                    or [r for r in raw if r['step'] == STEP] != self.holder['C34raw']):
                raise ValueError('B first post34 hold differs from C')
        return result


def worker(output, digest, *, import_only=False):
    output = Path(output).resolve()
    p = validate_frozen(output, digest)
    resource.setrlimit(resource.RLIMIT_AS, (LIMITS['address_space_bytes'], LIMITS['address_space_bytes']))
    signal.alarm(600)
    champion, evaluator, factory, audit, footprint = old.load_runtime(output, p, 'baseline', CELL[1])
    from haic.algorithms.joint_control.single_intervention import SingleInterventionSuccessor
    model = SingleInterventionSuccessor(champion, read_json(output / 'calibration.json'),
        expected_action=p['gate_binding']['expected_action'], intervention_step=STEP)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).resolve().is_relative_to(output / 'source'):
            path = Path(filename).resolve()
            if p['source_sha256'].get(str(path.relative_to(output / 'source'))) != sha(path):
                raise ValueError('single adapter import outside frozen closure')
    if import_only:
        print(json.dumps(dict(protocol_sha256=digest, branch='B', import_only=True, environment_resets=0,
            evaluator_file=str(old.SNAPSHOT / 'training/evaluate_closed_loop.py'), memory=old.memory_sample())))
        return
    start = read_json(output / 'run-start.json')
    if (start['protocol_sha256'] != digest or start['preflight_sha256'] != sha(output / 'preflight.json')
            or start['admission_sha256'] != sha(output / 'admission.json')):
        raise ValueError('bound single-use run-start required')
    validate_preflight(read_json(output / 'preflight.json'), digest)
    ledger, path = output / 'reset-ledger.jsonl', output / slot()['file']
    if ledger.exists():
        raise ValueError('existing reset evidence; never retry')
    refs = p['gate_binding']['references']
    a, c = (read_json(refs[k]['episode_path']) for k in ('A', 'C'))
    arows, crows = old.decisions(refs['A']['root'], a), old.decisions(refs['C']['root'], c)
    holder = dict(reference_decisions=arows, C34=crows[STEP - 1],
                  C34raw=[r for r in old.raw_states(refs['C']['root'], c) if r['step'] == STEP])
    raw_path, decision_path = path.with_suffix('.raw.jsonl'), path.with_suffix('.decisions.jsonl')
    def admit(phase):
        validate_frozen(output, digest)
        validate_admission(read_json(output / 'admission.json'), digest, p)
        append(output / 'resources.jsonl', dict(branch='B', phase=phase, **old.resource_admission(output, child=True)))
    with raw_path.open('x', buffering=1) as raw, decision_path.open('x', buffering=1) as stream:
        holder.update(raw_stream=raw, raw_path=raw_path)
        def environment_factory(**kwargs):
            if kwargs != dict(track_id=CELL[0], seed=CELL[1], max_decisions=1200) or 'environment' in holder:
                raise ValueError('only one B1200 environment may be constructed')
            admit('before_environment')
            observer = audit.make_observer(factory(**kwargs), raw, ledger, slot(), digest, footprint)
            reset, raw_reset, raw_step = observer.reset, observer.unwrapped.reset, observer.unwrapped.step
            observer.warmup_raw = 0
            def guarded_step(action):
                if not observer.active:
                    if observer.warmup_raw >= 51:
                        raise ValueError('extra warmup tick')
                    observer.warmup_raw += 1
                return raw_step(action)
            def guarded_reset(*args, **kwargs):
                if observer.reset_count:
                    raise ValueError('second reset forbidden')
                admit('before_reset')
                return raw_reset(*args, **kwargs)
            def checked_reset():
                result = reset()
                initial = old.initial_record(observer)
                save(path.with_suffix('.initial.json'), dict(protocol_sha256=digest, warmup_raw=observer.warmup_raw, **initial))
                if observer.warmup_raw != 51 or any(e.get(k) != v for e in (a, c) for k, v in initial.items()):
                    raise ValueError('B full1200 initial/static/image identity differs from A/C')
                return result
            observer.unwrapped.step, observer.unwrapped.reset, observer.reset = guarded_step, guarded_reset, checked_reset
            holder['environment'] = observer
            return observer
        result = evaluator(mode='single_intervention', track_id=CELL[0], seed=CELL[1],
            agent=MeasuredBranch(model, stream, holder, 'successor'), max_decisions=1200,
            plan_budget_seconds=4.5, capture_trace=True, fail_on_invalid_action=True, environment_factory=environment_factory)
    observer = holder.get('environment')
    initial = getattr(observer, 'initial', None)
    result.update(protocol_sha256=digest, branch='B', provenance='NEW',
        catalog=getattr(observer, 'catalog', None), initial_state=initial,
        geometry_sha256=getattr(observer, 'geometry_sha256', None),
        initial_observation_sha256=getattr(observer, 'initial_observation_sha256', None),
        warmup_raw=getattr(observer, 'warmup_raw', None), raw_trace_file=raw_path.name, raw_trace_sha256=sha(raw_path),
        decision_stream_file=decision_path.name, decision_stream_sha256=sha(decision_path),
        physical=audit.legacy.physical_metrics(([initial] if initial else []) + old.json_rows(raw_path), getattr(observer, 'obstacles', [])),
        memory=old.memory_sample(), natural_episode=old.natural_episode(result))
    save(path, result)
    append(ledger, dict(slot(), event='episode_end', protocol_sha256=digest, natural_episode=result['natural_episode']))


def preflight(output, digest):
    output = Path(output).resolve()
    validate_frozen(output, digest)
    if (output / 'preflight.json').exists():
        raise ValueError('preflight already exists')
    child = subprocess.run(worker_command(output, digest, import_only=True), cwd=output / 'model',
        env={**os.environ, **old.THREAD_ENV}, capture_output=True, text=True, timeout=60)
    if child.returncode:
        raise RuntimeError(child.stderr)
    result = json.loads(child.stdout)
    validate_preflight(result, digest)
    save(output / 'preflight.json', result)
    return result


def execute(output, digest, admission) -> dict[str, Any]:
    output = Path(output).resolve()
    p, value = validate_frozen(output, digest), read_json(admission)
    validate_preflight(read_json(output / 'preflight.json'), digest)
    validate_admission(value, digest, p, starting=True)
    if any((output / n).exists() for n in ('run-start.json', 'admission.json', 'episode-report.json', 'reset-ledger.jsonl')) \
            or list(output.glob(Path(slot()['file']).stem + '.*')):
        raise ValueError('existing B attempt; no retry/overwrite')
    save(output / 'admission.json', value)
    save(output / 'run-start.json', dict(protocol_sha256=digest, preflight_sha256=sha(output / 'preflight.json'),
                                        admission_sha256=sha(output / 'admission.json')))
    began, row, error, returncode = time.monotonic(), dict(slot(), status='started'), None, None
    path = output / slot()['file']
    try:
        append(output / 'resources.jsonl', dict(phase='before_B', **old.resource_admission(output)))
        with path.with_suffix('.stdout.log').open('x') as stdout, path.with_suffix('.stderr.log').open('x') as stderr:
            child = subprocess.run(worker_command(output, digest), cwd=output / 'model', env={**os.environ, **old.THREAD_ENV},
                stdout=stdout, stderr=stderr, timeout=min(600 - (time.monotonic() - began), value['expires_unix_s'] - time.time()))
        returncode = child.returncode
        if returncode:
            raise RuntimeError(f'B worker exited {returncode}')
        episode = read_json(path)
        row['sha256'] = sha(path)
        if not old.natural_episode(episode):
            raise RuntimeError('B execution/action/resource failure')
        row['status'] = 'completed'
    except BaseException as failure:
        error, row['status'] = f'{type(failure).__name__}: {failure}', 'failed'
    receipt = path.with_suffix('.process.json')
    save(receipt, dict(protocol_sha256=digest, returncode=returncode, error=error, wall_seconds=time.monotonic() - began))
    row.update(process_file=receipt.name, process_sha256=sha(receipt))
    ledger = output / 'reset-ledger.jsonl'
    report = dict(protocol_sha256=digest, rows=[row], operator_error=error,
        artifacts_sha256={p.name: sha(p) for p in output.glob(path.stem + '.*') if p.is_file()},
        reset_ledger_sha256=sha(ledger) if ledger.exists() else None)
    save(output / 'episode-report.json', report)
    return report


def branch_parity(a, b, c):
    if any(len(x['rows']) < STEP for x in (a, b, c)):
        return False
    for reference in (a, c):
        if any(b['episode'].get(k) != reference['episode'].get(k) for k in
               ('catalog', 'initial_state', 'geometry_sha256', 'initial_observation_sha256')):
            return False
        for br, rr in zip(b['rows'][:STEP], reference['rows'][:STEP]):
            if br['observation_sha256'] != rr['observation_sha256'] or br['evaluation_only']['pre'] != rr['evaluation_only']['pre']:
                return False
            if br['step'] < STEP and (action_bytes(br['action']) != action_bytes(rr['action']) or br['evaluation_only'] != rr['evaluation_only']):
                return False
        if [r for r in b['raw'] if r['step'] < STEP] != [r for r in reference['raw'] if r['step'] < STEP]:
            return False
    return (action_bytes(b['rows'][STEP - 1]['action']) == action_bytes(c['rows'][STEP - 1]['action'])
        and b['rows'][STEP - 1]['evaluation_only'] == c['rows'][STEP - 1]['evaluation_only']
        and [r for r in b['raw'] if r['step'] == STEP] == [r for r in c['raw'] if r['step'] == STEP])


def physical_windows(episode, rows, raw) -> dict[str, Any]:
    if len(rows) < STEP:
        return dict(horizons={}, milestones={})
    origin = rows[STEP - 1]['evaluation_only']['pre']
    track = episode['catalog']['track']
    length = sum(math.hypot(b[2] - a[2], b[3] - a[3]) for a, b in zip(track, track[1:] + track[:1]))
    posts = {round(r['evaluation_only']['post']['t'], 10): r['evaluation_only']['post'] for r in rows}
    samples, progress, previous, start_progress = [], 0., episode['initial_state']['station'], None
    controls, integrals = {r['step']: r['action'] for r in rows}, [0., 0., 0.]
    variation, prior_steer = 0., controls[STEP - 1][0]
    for s in raw:
        if s['step'] >= STEP and start_progress is None:
            start_progress = progress
        progress += (s['station'] - previous + length / 2) % length - length / 2
        previous = s['station']
        if s['step'] < STEP:
            continue
        integrals = [total + .02 * value for total, value in zip(integrals, controls[s['step']])]
        variation += abs(controls[s['step']][0] - prior_steer)
        prior_steer = controls[s['step']][0]
        post = posts.get(round(s['t'], 10), s)
        samples.append(dict(seconds=s['t'] - origin['t'], progress=progress - start_progress, lap_progress=progress / length, speed=s['speed'],
            x=s['x'], y=s['y'], lateral=s['lateral'], heading_error=s['heading_error'],
            steer_integral=integrals[0], gas_integral=integrals[1], brake_integral=integrals[2],
            steering_variation=variation,
            damage=post.get('environment_state', {}).get('damage'), road_wheels=sum(bool(v) for v in s['wheel_road_contacts'])))
    horizons = {name: samples[n - 1] if len(samples) >= n else None for name, n in (('H1', 4), ('H4', 16), ('1s', 50), ('2s', 100))}
    milestones = {}
    for f in (.25, .50, .75, .95):
        state = next((s for s in [episode['initial_state']] + raw
                      if s.get('environment_state', {}).get('tile_visited_count', -1) / len(track) >= f), None)
        milestones[str(f)] = None if state is None else dict(seconds=state['t'] - origin['t'],
            episode_seconds=state['t'] - episode['initial_state']['t'], speed=state['speed'], station=state['station'],
            tile_progress=state['environment_state']['tile_visited_count'] / len(track))
    return dict(horizons=horizons, milestones=milestones,
        qualification='Milestones are common tile-visited fractions at50Hz; horizons use signed nearest-segment GT progress. Missing targets are censored; no causal proof.')


def analyze(output, digest):
    output = Path(output).resolve()
    p, report = validate_frozen(output, digest), read_json(output / 'episode-report.json')
    if report['protocol_sha256'] != digest or len(report['rows']) != 1 or any(report['rows'][0].get(k) != v for k, v in slot().items()):
        raise ValueError('one bound B report required')
    for name, expected in report['artifacts_sha256'].items():
        if sha(old.local_path(output, name)) != expected:
            raise ValueError('sealed B evidence differs')
    ledger = output / 'reset-ledger.jsonl'
    if (sha(ledger) if ledger.exists() else None) != report['reset_ledger_sha256']:
        raise ValueError('B reset ledger differs')
    entries = old.json_rows(ledger) if ledger.exists() else []
    if (sum(r['event'] == 'reset_intent' for r in entries) > 1 or any(r['file'] != slot()['file'] or r['protocol_sha256'] != digest for r in entries)):
        raise ValueError('extra/unbound reset')
    record = report['rows'][0]
    if record['status'] == 'completed' and any(sum(r['event'] == event for r in entries) != 1 for event in ('reset_intent', 'episode_end')):
        raise ValueError('completed B requires one reset and one episode end')
    if record.get('process_file'):
        receipt = old.local_path(output, record['process_file'])
        if sha(receipt) != record['process_sha256'] or read_json(receipt)['protocol_sha256'] != digest:
            raise ValueError('B process receipt differs')
    data, summary = {}, {}
    for label in ('A', 'B', 'C'):
        ref = p['gate_binding']['references'].get(label)
        root, path = (Path(ref['root']), Path(ref['episode_path'])) if ref else (output, output / slot()['file'])
        try:
            episode = read_json(path) if path.exists() else None
        except json.JSONDecodeError:
            episode = None
        stream, truncated = envelope.partial_stream(path.with_suffix('.decisions.jsonl'))
        rows, calls = [r for r in stream if r['event'] == 'decision'], [r for r in stream if r['event'] == 'act']
        value: dict[str, Any] = dict(provenance='NEW' if label == 'B' else 'REUSED', partial=episode is None,
            policy=envelope.policy_metrics(rows, 'baseline' if label == 'A' else 'successor'),
            interventions=[dict(step=r['step'], t=r['evaluation_only']['pre']['t']) for r in rows if r['policy'].get('intervention')],
            actions34_37=[r['action'] for r in rows[STEP - 1:STEP + 3]],
            feedback34_onward=[dict(step=r['step'], action=r['action'], proposal=r['policy'].get('proposal_action'),
                recovery=r['policy'].get('champion_diagnostics', {})) for r in rows[STEP - 1:]],
            integrated_act=dict(cpu_seconds=old.distribution([r['cpu_seconds'] for r in calls]),
                wall_seconds=old.distribution([r['wall_seconds'] for r in calls]),
                all_over_1s=[r for r in calls if r['cpu_seconds'] > 1 or r['wall_seconds'] > 1]),
            truncated_tail=truncated, censored_act_calls=len({r['step'] for r in stream if r['event'] == 'act_intent'} - {r['step'] for r in calls}))
        if episode is not None:
            if label == 'B' and episode['protocol_sha256'] != digest:
                raise ValueError('B episode protocol differs')
            if rows != old.decisions(root, episode) or truncated:
                raise ValueError('decision receipt differs')
            if old.natural_episode(episode) and (len(rows) != episode['steps']
                    or [r['step'] for r in rows] != list(range(1, len(rows) + 1))):
                raise ValueError('natural episode/decision denominator differs')
            raw = old.raw_states(root, episode)
            data[label] = dict(episode=episode, rows=rows, raw=raw)
            value.update(partial=not old.natural_episode(episode), completed=episode['completed'],
                lap_time_ms=episode['lapTimeMs'] if episode['completed'] else None, progress=episode['progress'],
                damage=episode['damage'], collisions=episode['collisions'], physical=episode['physical'], memory=episode['memory'],
                all_wheel_loss_ticks=sum(not any(r['wheel_road_contacts']) for r in raw),
                partial_wheel_loss_ticks=sum(0 < sum(bool(v) for v in r['wheel_road_contacts']) < 4 for r in raw),
                failures=old.failure_events(rows, raw, episode), forecast=old.forecast_ranges(rows, raw),
                windows=physical_windows(episode, rows, raw))
        summary[label] = value
        value['fixed_tail_match'] = {name: len(value['actions34_37']) == 4 and all(
            action_bytes(a) == action_bytes(b) for a, b in zip(value['actions34_37'], tail))
            for name, tail in p['gate_binding']['fixed_h4'].items() if name.endswith('_tail')}
    b_rows = data.get('B', {}).get('rows', [])
    policy_valid = bool(b_rows) and [r['step'] for r in b_rows if r['policy'].get('intervention')] == [STEP]
    policy_valid = policy_valid and all(action_bytes(r['action']) == action_bytes(
        p['gate_binding']['expected_action'] if r['step'] == STEP else r['policy']['proposal_action']) for r in b_rows)
    matched = (set(data) == {'A', 'B', 'C'} and policy_valid and report['rows'][0]['status'] == 'completed'
        and report['operator_error'] is None
        and all(old.natural_episode(v['episode']) for v in data.values()) and branch_parity(data['A'], data['B'], data['C']))
    deltas = {}
    for right, left in (('B', 'A'), ('C', 'A'), ('C', 'B')):
        valid = matched and all(summary[k].get('completed') for k in (right, left))
        deltas[f'{right}-{left}'] = dict(lap_delta_ms=summary[right]['lap_time_ms'] - summary[left]['lap_time_ms'] if valid else None)
        for kind in ('horizons', 'milestones'):
            before = summary[left].get('windows', {}).get(kind, {})
            after = summary[right].get('windows', {}).get(kind, {})
            deltas[f'{right}-{left}'][kind] = {key: {metric: after[key][metric] - before[key][metric]
                for metric in before[key] if type(before[key][metric]) in (int, float) and type(after[key][metric]) in (int, float)}
                if matched and before.get(key) is not None and after.get(key) is not None else None
                for key in before.keys() | after.keys()}
    result = dict(schema=SCHEMA + '-analysis', protocol_sha256=digest, reuse=p['gate_binding'],
        new_reset_intents=sum(r['event'] == 'reset_intent' for r in entries), branches=summary,
        matched_ABC=matched, policy_valid=policy_valid, deltas=deltas, operator_error=report['operator_error'],
        fixed_h4=p['gate_binding']['fixed_h4'],
        qualification='One consumed TRAIN road. A/C and fixed tails REUSED, only B NEW; no fitting, promotion, official score, or causal/safety proof.')
    save(output / 'summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('freeze', 'preflight', 'execute', 'analyze'):
        sub = commands.add_parser(command)
        sub.add_argument('--output', required=True, type=Path)
        if command == 'freeze':
            for name in ('calibration', 'claim', 'references', 'reuse-certificate'):
                sub.add_argument('--' + name, required=True, type=Path)
        else:
            sub.add_argument('--protocol-sha256', required=True)
        if command == 'execute':
            sub.add_argument('--admission', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        result = dict(protocol_sha256=freeze(args.output, args.calibration, args.claim, args.references, args.reuse_certificate), environment_resets=0)
    elif args.command == 'execute':
        result = execute(args.output, args.protocol_sha256, args.admission)
    else:
        result = globals()[args.command](args.output, args.protocol_sha256)
    print(json.dumps(result, allow_nan=False, indent=2))
    return int(args.command == 'execute' and result['operator_error'] is not None)


if __name__ == '__main__':
    raise SystemExit(main())
