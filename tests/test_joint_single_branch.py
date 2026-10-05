"""Synthetic one-reset branch tests; never construct or reset a real environment."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

from scripts import run_joint_single_branch as pilot


def physical(tick=0, label='A') -> dict[str, Any]:
    offset = max(0, tick - 136) * {'A': 0., 'B': .0005, 'C': .001}[label]
    return dict(t=tick * .02, x=0., y=tick * .1 - offset, yaw=0., speed=5.,
        station=tick * .1 - offset, lateral=offset, heading_error=0., contacts=[],
        wheel_road_contacts=[1, 1, 1, 1], environment_state=dict(damage=0., tile_visited_count=tick // 60))


def write_episode(root, label, digest, *, finish=True, steps=60) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    mode = {'A': 'baseline', 'B': 'single', 'C': 'successor'}[label]
    path = Path(root) / f'3-3184000005-{mode}.json'
    raw = [dict(physical(i, label), step=(i - 1) // 4 + 1, collision=False) for i in range(1, 4 * steps + 1)]
    rows = []
    for step in range(1, steps + 1):
        changed = step == 34 and label != 'A' or label == 'C' and step in (35, 40)
        proposal = [0., 0., .1] if step <= 34 else [0., .01 * (label == 'B'), .1]
        action = [-0., .05, .05] if changed else proposal.copy()
        pre, post = physical((step - 1) * 4, label), physical(step * 4, label)
        rows.append(dict(event='decision', step=step, action=action,
            observation_sha256=hashlib.sha256(json.dumps(pre).encode()).hexdigest(),
            evaluation_only=dict(pre=pre, post=post), policy=dict(intervention=changed,
                proposal_action=proposal, eligible=changed, forecast=None, comparison=None)))
    for row in rows:
        pilot.append(path.with_suffix('.decisions.jsonl'), dict(event='act_intent', step=row['step']))
        pilot.append(path.with_suffix('.decisions.jsonl'), dict(event='act', step=row['step'], cpu_seconds=.01, wall_seconds=.02))
        pilot.append(path.with_suffix('.decisions.jsonl'), row)
    for row in raw:
        pilot.append(path.with_suffix('.raw.jsonl'), row)
    episode = dict(protocol_sha256=digest, completed=finish, lapTimeMs={'A': 5000, 'B': 5050, 'C': 5200}[label],
        progress=1., damage=0., collisions=0, invalid_actions=0, error=None, steps=steps,
        retire_reason=None if finish else 'off_track',
        catalog=dict(track=[[0., 0., 0., 0.], [0., 0., 10., 0.], [0., 0., 10., 10.], [0., 0., 0., 10.]], obstacles=[]),
        geometry_sha256='geometry', initial_state=physical(), initial_observation_sha256='pixels',
        raw_trace_file=path.with_suffix('.raw.jsonl').name, raw_trace_sha256=pilot.sha(path.with_suffix('.raw.jsonl')),
        decision_stream_file=path.with_suffix('.decisions.jsonl').name, decision_stream_sha256=pilot.sha(path.with_suffix('.decisions.jsonl')),
        physical=dict(physical_contact_events=0), memory=dict(VmRSS=100, VmHWM=100))
    pilot.save(path, episode)
    return dict(root=str(Path(root)), episode_path=str(path), episode_sha256=pilot.sha(path), provenance='REUSED'), episode, rows, raw


def setup(root):
    root = Path(root)
    refs, output = root / 'references', root / 'new'
    refs.mkdir()
    output.mkdir()
    reference_protocol = dict(pilot.envelope.contract())
    pilot.save(refs / 'protocol.json', reference_protocol)
    digest = pilot.sha(refs / 'protocol.json')
    records = {}
    for label in ('A', 'C'):
        records[label] = write_episode(refs, label, digest)
    pilot.save(refs / 'summary.json', dict(frozen_reference=True))
    pilot.save(refs / 'episode-report.json', dict(protocol_sha256=digest, operator_error=None,
        rows=[dict(file=Path(v[0]['episode_path']).name, status='completed', sha256=v[0]['episode_sha256']) for v in records.values()]))
    anchor = root / 'anchor.json'
    pilot.save(anchor, dict(completed_prefix=33, calibration_sha256=pilot.envelope.CALIBRATION_SHA,
                           actions=[records['A'][2][33]['action'], records['C'][2][33]['action']]))
    certificate = root / 'reuse.json'
    pins = {str(p): pilot.sha(p) for p in refs.iterdir() if p.is_file()}
    pins[str(anchor)] = pilot.sha(anchor)
    pilot.save(certificate, dict(schema='haic-joint-single-branch-reuse-v1', reusable=True,
        cell=list(pilot.CELL), intervention_step=34, references_directory=str(refs),
        calibration_sha256=pilot.envelope.CALIBRATION_SHA, checks={k: True for k in pilot.CHECKS},
        fixed_h4=dict(anchor_path=str(anchor)), evidence_sha256=pins))
    binding = dict(paths=dict(analysis=str(certificate)), references={k: v[0] for k, v in records.items()},
        sha256=dict(certificate=pilot.sha(certificate), protocol=digest, summary=pilot.sha(refs / 'summary.json'),
                    report=pilot.sha(refs / 'episode-report.json')),
        expected_action=records['C'][2][33]['action'], fixed_h4=dict(provenance='REUSED', anchor_path=str(anchor),
            baseline_tail=[records['A'][2][33]['action']] * 4,
            alternative_tail=[records['C'][2][33]['action']] + [records['A'][2][33]['action']] * 3))
    p: dict[str, Any] = dict(pilot.contract(), claim_sha256='claim', gate_binding=binding, source_sha256={})
    receipt = dict(protocol_sha256='new', branch='B', import_only=True, environment_resets=0,
                   evaluator_file=str(pilot.old.SNAPSHOT / 'training/evaluate_closed_loop.py'))
    pilot.save(output / 'preflight.json', receipt)
    pilot.save(output / 'caller-admission.json', dict(schema='haic-joint-single-branch-admission-v1', admitted=True,
        protocol_sha256='new', claim_sha256='claim', calibration_sha256=pilot.envelope.CALIBRATION_SHA,
        created_unix_s=100., expires_unix_s=700.))
    return output, refs, certificate, p, records


class ReuseTests(unittest.TestCase):
    def test_new_frozen_dependency_maps_must_match_reused_A_C(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior = dict(environment_sha256={'env.py': 'env'}, model_sha256={'agent.py': 'agent'},
                         runtime_sha256='python', source_sha256={'old.py': 'old'})
            pilot.save(root / 'protocol.json', prior)
            current: dict[str, Any] = dict(copy.deepcopy(prior), gate_binding=dict(references={'A': {'root': str(root)}}))
            current['source_sha256']['new.py'] = 'new'
            with patch.object(pilot, 'frozen_operation', return_value=lambda *args: current):
                self.assertIs(pilot.validate_frozen(root, 'digest'), current)
                current['source_sha256']['old.py'] = 'modified'
                with self.assertRaisesRegex(ValueError, 'differs from A/C'):
                    pilot.validate_frozen(root, 'digest')

    def test_private_freeze_clones_retain_original_code_not_module_globals(self):
        before = dict(pilot.envelope.freeze.__globals__)
        function = pilot.frozen_operation(pilot.envelope.freeze)
        self.assertIs(function.__code__, pilot.envelope.freeze.__code__)
        self.assertIsNot(function.__globals__, pilot.envelope.freeze.__globals__)
        self.assertIs(function.__globals__['gate_binding'], pilot.reference_binding)
        self.assertEqual(function.__globals__['contract']()['limits']['max_resets'], 1)
        self.assertEqual(pilot.envelope.contract()['limits']['max_resets'], 2)
        self.assertEqual(before, pilot.envelope.freeze.__globals__)
        function = pilot.frozen_operation(pilot.envelope.validate_frozen)
        self.assertIs(function.__code__, pilot.envelope.validate_frozen.__code__)

    def test_reference_certificate_pins_and_exact_signed_zero_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, refs, cert, p, _ = setup(temporary)
            with patch.object(pilot, 'SUMMARY_SHA', p['gate_binding']['sha256']['summary']), \
                    patch.object(pilot, 'REFERENCE_PROTOCOL_SHA', p['gate_binding']['sha256']['protocol']):
                binding = pilot.reference_binding(cert, pilot.envelope.CALIBRATION_SHA)
                self.assertEqual(binding, p['gate_binding'])
                self.assertEqual(pilot.action_bytes(binding['expected_action'])[:4], bytes.fromhex('00000080'))
                value = pilot.read_json(cert)
                value['checks']['first33_and_pre34'] = False
                cert.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    pilot.reference_binding(cert, pilot.envelope.CALIBRATION_SHA)
            self.assertFalse((output / '3-3184000005-baseline.json').exists())

    def test_expected_action_comes_from_bound_C_not_certificate_arbitrary_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, refs, cert, p, _ = setup(temporary)
            path = refs / '3-3184000005-successor.decisions.jsonl'
            with path.open('a') as stream:
                stream.write('{}\n')
            with patch.object(pilot, 'SUMMARY_SHA', p['gate_binding']['sha256']['summary']), \
                    patch.object(pilot, 'REFERENCE_PROTOCOL_SHA', p['gate_binding']['sha256']['protocol']), self.assertRaises(ValueError):
                pilot.reference_binding(cert, pilot.envelope.CALIBRATION_SHA)

    def test_one_slot_new_frozen_entry_and_fixed_step(self):
        contract = pilot.contract()
        self.assertEqual(contract['schedule'], [pilot.slot()])
        self.assertEqual(contract['limits']['max_resets'], 1)
        self.assertEqual(contract['limits']['total_seconds'], 600)
        self.assertEqual(contract['intervention_step'], 34)
        command = pilot.worker_command('/tmp/kilo/synthetic', 'digest')
        self.assertIn('run_joint_single_branch import worker', command[4])
        self.assertNotIn('run_joint_envelope_pilot import worker', command[4])


class MeasuredTests(unittest.TestCase):
    def test_whole_act_clock_is_inherited_and_step33_cannot_intervene(self):
        self.assertIsNot(pilot.MeasuredBranch.act, pilot.old.MeasuredAgent.act)
        with tempfile.TemporaryFile(mode='w+') as stream:
            model = SimpleNamespace(last_diagnostics=dict(intervention=True, proposal_action=[0., 0., .1]))
            holder = dict(environment=SimpleNamespace(last=dict(pre=physical(), post=physical(4))))
            measured = pilot.MeasuredBranch(model, stream, holder, 'successor')
            measured.calls, measured.action, measured.pixels, measured.timing = 33, [-0., .05, .05], 'pixels', {}
            with self.assertRaisesRegex(ValueError, 'exactly34'):
                measured.last_step_diagnostics()

    def test_own_proposal_valid_but_wrong_prefix_action_rejected_before_step(self):
        class Pixels:
            def tobytes(self):
                return b'unchanged pre33 pixels'
        obs = Pixels()
        pre = physical(128)
        prior = dict(action=[0., 0., .1], observation_sha256=hashlib.sha256(obs.tobytes()).hexdigest(),
                     evaluation_only=dict(pre=pre))
        model = SimpleNamespace(act=lambda observation: [.01, 0., .1],
                                last_diagnostics=dict(intervention=False, proposal_action=[.01, 0., .1]))
        with tempfile.TemporaryFile(mode='w+') as stream, patch.object(pilot.old, 'memory_sample', return_value={}):
            measured = pilot.MeasuredBranch(model, stream,
                dict(reference_decisions=[prior] * 33, environment=SimpleNamespace(state=lambda: pre)), 'successor')
            measured.calls = 32
            env_step = Mock()
            with self.assertRaisesRegex(ValueError, 'before the declared decision34'):
                env_step(measured.act(obs))
            env_step.assert_not_called()
            stream.seek(0)
            records = [json.loads(line) for line in stream]
            self.assertEqual([r['event'] for r in records], ['act_intent', 'act'])
            self.assertGreaterEqual(records[-1]['cpu_seconds'], 0.)
            self.assertGreaterEqual(records[-1]['wall_seconds'], 0.)

    def test_post34_uses_fresh_B_observations_not_A_future_actions(self):
        seen = []
        class Pixels:
            def __init__(self, value):
                self.value = value
            def tobytes(self):
                return str(self.value).encode()
        class Model:
            def act(self, observation):
                seen.append(observation)
                proposal = [observation.value / 100, 0., .1]
                changed = len(seen) == 1
                self.last_diagnostics = dict(proposal_action=proposal, intervention=changed)
                return [-0., .05, .05] if changed else proposal
        with tempfile.TemporaryDirectory() as temporary:
            raw_path = Path(temporary) / 'raw.jsonl'
            c_raw = [dict(physical(i), step=34) for i in range(133, 137)]
            for row in c_raw:
                pilot.append(raw_path, row)
            obs34, obs35, obs36 = Pixels(1), Pixels(7), Pixels(9)
            before, after = physical(132), physical(136)
            c34 = dict(action=[-0., .05, .05], evaluation_only=dict(pre=before, post=after))
            refs: list[dict[str, Any]] = [dict(action=[.99, 0., .1]) for _ in range(40)]
            refs[33].update(observation_sha256=hashlib.sha256(obs34.tobytes()).hexdigest(), evaluation_only=dict(pre=before))
            env = SimpleNamespace(last=c34['evaluation_only'], state=lambda: before)
            with raw_path.open('a') as raw, tempfile.TemporaryFile(mode='w+') as stream, \
                    patch.object(pilot.old, 'memory_sample', return_value={}):
                holder = dict(reference_decisions=refs, C34=c34, C34raw=c_raw, raw_path=raw_path, raw_stream=raw, environment=env)
                measured = pilot.MeasuredBranch(Model(), stream, holder, 'successor')
                measured.calls = 33
                self.assertEqual(pilot.action_bytes(measured.act(obs34)), pilot.action_bytes(c34['action']))
                measured.last_step_diagnostics()
                for obs in (obs35, obs36):
                    action = measured.act(obs)
                    measured.last_step_diagnostics()
                    self.assertEqual(action[0], obs.value / 100)
                    self.assertNotEqual(action, refs[measured.calls - 1]['action'])
                self.assertEqual(seen, [obs34, obs35, obs36])
                self.assertEqual(measured.calls, 36)

    def test_post34_physical_mismatch_rejected_after_preserving_decision(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryFile(mode='w+') as stream:
            raw_path = Path(temporary) / 'raw.jsonl'
            pilot.append(raw_path, dict(step=34, wrong=True))
            with raw_path.open('a') as raw:
                model = SimpleNamespace(last_diagnostics=dict(intervention=True, proposal_action=[0., 0., .1]))
                holder = dict(environment=SimpleNamespace(last=dict(pre=physical(), post=physical(4))),
                    C34=dict(action=[-0., .05, .05], evaluation_only={}), C34raw=[], raw_stream=raw, raw_path=raw_path)
                measured = pilot.MeasuredBranch(model, stream, holder, 'successor')
                measured.calls, measured.action, measured.pixels, measured.timing = 34, [-0., .05, .05], 'pixels', {}
                with self.assertRaisesRegex(ValueError, 'first post34 hold'):
                    measured.last_step_diagnostics()
                stream.seek(0)
                self.assertEqual(json.loads(stream.readline())['step'], 34)


class LifecycleTests(unittest.TestCase):
    def execute_fake(self, output, p, *, fail=False, finish=True):
        calls = []
        def child(command, **kwargs):
            calls.append(command)
            self.assertIn('run_joint_single_branch', command[4])
            self.assertLessEqual(kwargs['timeout'], 600)
            pilot.append(output / 'reset-ledger.jsonl', dict(pilot.slot(), event='reset_intent', protocol_sha256='new'))
            if fail:
                path = output / pilot.slot()['file']
                pilot.append(path.with_suffix('.decisions.jsonl'), dict(event='act', step=1, cpu_seconds=1.2, wall_seconds=1.3))
                path.write_text('{"partial":')
                raise pilot.subprocess.TimeoutExpired(command, 600)
            write_episode(output, 'B', 'new', finish=finish)
            pilot.append(output / 'reset-ledger.jsonl', dict(pilot.slot(), event='episode_end', protocol_sha256='new'))
            return SimpleNamespace(returncode=0)
        with patch.object(pilot, 'validate_frozen', return_value=p), patch.object(pilot.time, 'time', return_value=101.), \
                patch.object(pilot.old, 'resource_admission', return_value={}), patch.object(pilot.subprocess, 'run', side_effect=child):
            report = pilot.execute(output, 'new', output / 'caller-admission.json')
        return calls, report

    def test_exact_one_new_B_reset_ABC_laps_provenance_and_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, refs, _, p, _ = setup(temporary)
            reference_hashes = {f.name: pilot.sha(f) for f in refs.iterdir()}
            calls, report = self.execute_fake(output, p)
            self.assertEqual(len(calls), 1)
            self.assertEqual(report['rows'][0]['branch'], 'B')
            self.assertIsNone(report['operator_error'])
            with patch.object(pilot, 'validate_frozen', return_value=p):
                summary = pilot.analyze(output, 'new')
            self.assertEqual(summary['new_reset_intents'], 1)
            self.assertTrue(summary['matched_ABC'])
            self.assertEqual(summary['deltas']['B-A']['lap_delta_ms'], 50)
            self.assertEqual(summary['deltas']['C-A']['lap_delta_ms'], 200)
            self.assertEqual(summary['deltas']['C-B']['lap_delta_ms'], 150)
            self.assertEqual([v['step'] for v in summary['branches']['B']['interventions']], [34])
            self.assertEqual([summary['branches'][v]['provenance'] for v in ('A', 'B', 'C')], ['REUSED', 'NEW', 'REUSED'])
            self.assertEqual(set(summary['deltas']['B-A']['horizons']), {'H1', 'H4', '1s', '2s'})
            self.assertEqual(set(summary['deltas']['B-A']['milestones']), {'0.25', '0.5', '0.75', '0.95'})
            self.assertEqual(reference_hashes, {f.name: pilot.sha(f) for f in refs.iterdir()})

    def test_natural_DNF_no_retry_no_finish_lap(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, _, _, p, _ = setup(temporary)
            calls, _ = self.execute_fake(output, p, finish=False)
            with patch.object(pilot, 'validate_frozen', return_value=p):
                summary = pilot.analyze(output, 'new')
            self.assertEqual(len(calls), 1)
            self.assertTrue(summary['matched_ABC'])
            self.assertIsNone(summary['branches']['B']['lap_time_ms'])
            self.assertIsNone(summary['deltas']['B-A']['lap_delta_ms'])
            self.assertEqual(summary['branches']['B']['progress'], 1.)

    def test_missing_or_failed_B_is_unmatched_and_partial_timing_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, _, _, p, _ = setup(temporary)
            calls, report = self.execute_fake(output, p, fail=True)
            with patch.object(pilot, 'validate_frozen', return_value=p):
                summary = pilot.analyze(output, 'new')
            self.assertFalse(summary['matched_ABC'])
            self.assertTrue(summary['branches']['B']['partial'])
            self.assertEqual(summary['branches']['B']['integrated_act']['cpu_seconds']['over_1s_count'], 1)
            self.assertTrue(all(v['lap_delta_ms'] is None for v in summary['deltas'].values()))
            with patch.object(pilot, 'validate_frozen', return_value=p), patch.object(pilot.time, 'time', return_value=101.), \
                    patch.object(pilot.subprocess, 'run') as launch:
                with self.assertRaisesRegex(ValueError, 'no retry'):
                    pilot.execute(output, 'new', output / 'caller-admission.json')
                launch.assert_not_called()

    def test_provenance_failed_B_never_matched_even_if_finishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, _, _, p, _ = setup(temporary)
            _, report = self.execute_fake(output, p)
            report['rows'][0]['status'] = 'failed'
            (output / 'episode-report.json').write_text(json.dumps(report))
            with patch.object(pilot, 'validate_frozen', return_value=p):
                summary = pilot.analyze(output, 'new')
            self.assertFalse(summary['matched_ABC'])
            self.assertIsNone(summary['deltas']['C-B']['lap_delta_ms'])

    def test_first_H1_raw_parity_and_final_boundary_damage(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, _, records = setup(temporary)
            a = dict(episode=records['A'][1], rows=records['A'][2], raw=records['A'][3])
            c = dict(episode=records['C'][1], rows=records['C'][2], raw=records['C'][3])
            b = copy.deepcopy(c)
            self.assertTrue(pilot.branch_parity(a, b, c))
            b['raw'][135]['y'] += .01
            self.assertFalse(pilot.branch_parity(a, b, c))
            b = copy.deepcopy(c)
            b['rows'][36]['evaluation_only']['post']['environment_state']['damage'] = .4
            windows = pilot.physical_windows(b['episode'], b['rows'], b['raw'])
            self.assertEqual(windows['horizons']['H4']['damage'], .4)
            failures = pilot.old.failure_events(b['rows'], b['raw'], b['episode'])
            self.assertTrue(any(r['kind'] == 'damage_increase' and r['step'] == 37 for r in failures['events']))

    def test_tile_milestones_are_not_station_fraction_and_censored_stays_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, _, records = setup(temporary)
            _, episode, rows, raw = records['A']
            windows = pilot.physical_windows(episode, rows, raw)
            self.assertEqual(windows['milestones']['0.25']['episode_seconds'], 1.2)
            self.assertEqual(windows['milestones']['0.95']['episode_seconds'], 4.8)
            self.assertNotEqual(windows['milestones']['0.95']['station'] / 40, .95)
            for r in raw:
                r['environment_state']['tile_visited_count'] = min(2, r['environment_state']['tile_visited_count'])
            windows = pilot.physical_windows(episode, rows, raw)
            self.assertIsNone(windows['milestones']['0.75'])
            self.assertIsNone(windows['milestones']['0.95'])

    def test_preflight_one_B_import_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = dict(protocol_sha256='new', branch='B', import_only=True, environment_resets=0,
                           evaluator_file=str(pilot.old.SNAPSHOT / 'training/evaluate_closed_loop.py'))
            with patch.object(pilot, 'validate_frozen', return_value={}), patch.object(pilot.subprocess, 'run',
                    return_value=SimpleNamespace(returncode=0, stdout=json.dumps(receipt))) as launch:
                pilot.preflight(root, 'new')
                self.assertEqual(launch.call_count, 1)
                self.assertEqual(launch.call_args.args[0][-1], 'True')

    def test_worker_one_B_factory1200_and51_warmup_no_second_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, _, _, p, records = setup(temporary)
            pilot.save(output / 'calibration.json', dict(synthetic=True))
            pilot.save(output / 'admission.json', pilot.read_json(output / 'caller-admission.json'))
            pilot.save(output / 'run-start.json', dict(protocol_sha256='new',
                preflight_sha256=pilot.sha(output / 'preflight.json'), admission_sha256=pilot.sha(output / 'admission.json')))
            events = []
            class FakeObserver:
                def __init__(self, environment, raw, ledger, slot, digest, footprint):
                    self.active, self.reset_count = False, 0
                    self.catalog = records['A'][1]['catalog']
                    self.initial, self.obstacles = physical(), []
                    self.geometry_sha256, self.initial_observation_sha256 = 'geometry', 'pixels'
                    def reset():
                        pilot.append(ledger, dict(slot, event='reset_intent', protocol_sha256=digest))
                        self.reset_count += 1
                        events.append(('reset', len(pilot.old.json_rows(ledger))))
                    self.unwrapped = SimpleNamespace(reset=reset, step=lambda action: events.append('raw'))
                def reset(self):
                    self.unwrapped.reset()
                    for _ in range(51):
                        self.unwrapped.step(None)
                    self.active = True
                    return object(), {}
            def evaluator(**kwargs):
                self.assertEqual(kwargs['max_decisions'], 1200)
                self.assertTrue(kwargs['fail_on_invalid_action'])
                self.assertIsInstance(kwargs['agent'], pilot.MeasuredBranch)
                env = kwargs['environment_factory'](track_id=3, seed=3184000005, max_decisions=1200)
                env.reset()
                with self.assertRaisesRegex(ValueError, 'second reset'):
                    env.reset()
                with self.assertRaises(ValueError):
                    kwargs['environment_factory'](track_id=3, seed=3184000005, max_decisions=1200)
                return dict(error=None, invalid_actions=0, completed=False, retire_reason='off_track',
                            damage=0., collisions=0, steps=1, lapTimeMs=None, progress=0.)
            module: Any = ModuleType('haic.algorithms.joint_control.single_intervention')
            module.SingleInterventionSuccessor = Mock(return_value=object())
            factory = Mock()
            audit = SimpleNamespace(make_observer=FakeObserver, legacy=SimpleNamespace(physical_metrics=lambda *args: {}))
            with patch.object(pilot, 'validate_frozen', return_value=p), patch.object(pilot, 'validate_admission'), \
                    patch.object(pilot.resource, 'setrlimit'), patch.object(pilot.signal, 'alarm'), \
                    patch.object(pilot.old, 'memory_sample', return_value={}), \
                    patch.object(pilot.old, 'resource_admission', return_value={}), \
                    patch.object(pilot.old, 'load_runtime', return_value=(object(), evaluator, factory, audit, object())) as loader, \
                    patch.dict(pilot.sys.modules, {module.__name__: module}):
                pilot.worker(output, 'new')
            self.assertEqual(loader.call_args.args[2], 'baseline')
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(factory.call_args.kwargs, dict(track_id=3, seed=3184000005, max_decisions=1200))
            self.assertEqual(events[0], ('reset', 1))
            self.assertEqual(events.count('raw'), 51)
            self.assertEqual(module.SingleInterventionSuccessor.call_args.kwargs['intervention_step'], 34)
            self.assertEqual(module.SingleInterventionSuccessor.call_args.kwargs['expected_action'], p['gate_binding']['expected_action'])
            self.assertEqual(len([r for r in pilot.old.json_rows(output / 'reset-ledger.jsonl') if r['event'] == 'reset_intent']), 1)


if __name__ == '__main__':
    unittest.main()
