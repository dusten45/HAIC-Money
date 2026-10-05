"""Synthetic/file-only runner tests. Never import or construct a simulator."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

from scripts import run_joint_temporal_pilot as pilot


def admission() -> dict[str, Any]:
    return dict(schema='haic-joint-temporal-pilot-admission-v1', admitted=True,
                protocol_sha256='protocol', claim_sha256='claim', calibration_sha256='calibration',
                created_unix_s=100., expires_unix_s=3700.)


def preflight() -> dict[str, Any]:
    return dict(protocol_sha256='protocol', environment_resets=0, arms=[
        dict(mode=arm, import_only=True, environment_resets=0,
             evaluator_file=str(pilot.SNAPSHOT / 'training/evaluate_closed_loop.py')) for arm in pilot.ARMS])


def natural(completed=False, reason: str | None = 'crash') -> dict[str, Any]:
    return dict(error=None, invalid_actions=0, completed=completed, retire_reason=reason,
                damage=.2, collisions=1, steps=4, lapTimeMs=320 if completed else None,
                protocol_sha256='protocol', progress=1.)


def state(t=0., y=0., *, step=None, contacts=(), road=(1, 1, 1, 1), damage=0.) -> dict[str, Any]:
    value: dict[str, Any] = dict(t=t, x=0., y=y, yaw=0., contacts=list(contacts),
                 wheel_road_contacts=list(road), environment_state=dict(damage=damage))
    if step is not None:
        value['step'] = step
    return value


def forecasts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    action = [0., .1, 0.]
    raw = [state(i * .02, i * .1, step=(i - 1) // 4 + 1) for i in range(1, 17)]
    rows: list[dict[str, Any]] = [dict(event='decision', step=i + 1, action=action.copy(), policy={},
                 observation_sha256='pixels', evaluation_only=dict(
                     pre=state(i * .08, i * .4), post=state((i + 1) * .08, (i + 1) * .4)))
            for i in range(4)]
    rows[0]['policy']['forecast'] = dict(poses=[[[0., i * .1, 0.] for i in range(17)]],
        actions=[action.copy() for _ in range(4)], position_residual=[0.] + [.01] * 16,
        yaw_residual=[0.] + [.01] * 16)
    return rows, raw


class ContractTests(unittest.TestCase):
    def test_exact_six_slots_and_fixed_resource_contract(self):
        self.assertEqual(pilot.CELLS, ((1, 3184000013), (3, 3184000002), (2, 3184000006)))
        self.assertEqual(len(pilot.schedule()), 6)
        self.assertEqual([(r['track_id'], r['seed'], r['mode']) for r in pilot.schedule()],
                         [(t, s, a) for t, s in pilot.CELLS for a in ('baseline', 'successor')])
        self.assertEqual(pilot.LIMITS['max_decisions'], 1200)
        self.assertEqual(pilot.LIMITS['total_seconds'], 3600)
        self.assertEqual(pilot.LIMITS['child_seconds'], 600)
        self.assertEqual(pilot.LIMITS['rss_bytes'], 1024**3)
        self.assertEqual(pilot.LIMITS['address_space_bytes'], 4 * 1024**3)

    def test_admission_binding_expiry_and_initial_freshness(self):
        p = dict(claim_sha256='claim', calibration_sha256='calibration')
        pilot.validate_admission(admission(), 'protocol', p, now=101, starting=True)
        pilot.validate_admission(admission(), 'protocol', p, now=1000, starting=False)
        for now, starting in ((99, False), (3700, False), (401, True)):
            with self.subTest(now=now), self.assertRaises(ValueError):
                pilot.validate_admission(admission(), 'protocol', p, now=now, starting=starting)
        for key, value in (('protocol_sha256', 'wrong'), ('admitted', False), ('expires_unix_s', 3701),
                           ('created_unix_s', float('nan'))):
            changed = dict(admission(), **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot.validate_admission(changed, 'protocol', p, now=101)

    def test_preflight_requires_both_arms_external_evaluator_no_resets(self):
        pilot.validate_preflight(preflight(), 'protocol')
        for key, value in (('environment_resets', 1), ('evaluator_file', '/workspace/training/evaluate_closed_loop.py'),
                           ('mode', 'baseline'), ('import_only', False)):
            receipt = preflight()
            receipt['arms'][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot.validate_preflight(receipt, 'protocol')

    def test_natural_accidents_and_time_limit_are_not_execution_failure(self):
        for reason in ('crash', 'off_track', 'max_steps'):
            self.assertTrue(pilot.natural_episode(natural(reason=reason)))
        self.assertTrue(pilot.natural_episode(natural(True, None)))
        for key, value in (('error', 'failure'), ('invalid_actions', 1), ('damage', None),
                           ('retire_reason', 'invalid_action'), ('steps', 1201)):
            self.assertFalse(pilot.natural_episode(dict(natural(), **{key: value})))
        self.assertFalse(pilot.natural_episode(dict(natural(reason='max_steps'), steps=1200)))
        # progress=1 alone must never turn a DNF into a finish.
        self.assertFalse(natural()['completed'])

    def test_worker_is_cpu21_isolated_and_thread_bounded(self):
        command = pilot.worker_command('/tmp/kilo/test', 1, 'sha', import_only=True)
        self.assertEqual(command[:3], ['/tmp/kilo/haic-cpu21/bin/python', '-I', '-B'])
        self.assertEqual(command[-1], 'True')
        self.assertEqual(pilot.THREAD_ENV['OMP_NUM_THREADS'], '1')
        self.assertEqual(pilot.THREAD_ENV['CUDA_VISIBLE_DEVICES'], '')

    def test_local_paths_reject_escape(self):
        with self.assertRaises(ValueError):
            pilot.local_path('/tmp/kilo/run', '../escape')

    def test_static_closure_does_not_import_root_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'scripts').mkdir()
            (root / 'haic').mkdir()
            (root / 'training').mkdir()
            (root / 'scripts/main.py').write_text('from haic import helper\nfrom training.env_factory import Env\n')
            (root / 'scripts/__init__.py').write_text('')
            (root / 'haic/__init__.py').write_text('')
            (root / 'haic/helper.py').write_text('from . import leaf\n')
            (root / 'haic/leaf.py').write_text('raise RuntimeError("must not import")\n')
            (root / 'training/env_factory.py').write_text('raise RuntimeError("simulator forbidden")\n')
            pins = pilot.source_closure(root, ['scripts/main.py'], packages=('scripts', 'haic'))
            self.assertEqual(set(pins), {'scripts/main.py', 'scripts/__init__.py', 'haic/__init__.py',
                                         'haic/helper.py', 'haic/leaf.py'})

    def test_claim_bound_to_exact_cells_sources_and_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = [root / pilot.OPERATOR, root / pilot.SUCCESSOR, root / 'evidence.json']
            for file in files:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text('synthetic')
            claim = dict(schema='haic-joint-temporal-pilot-consumed-train-v1', partition='TRAIN', fresh=False,
                cells=[list(c) for c in pilot.CELLS], max_resets=6, run_directory=str(root / 'run'),
                calibration_sha256='calibration', source_sha256={str(f): pilot.sha(f) for f in files[:2]},
                evidence_sha256={str(files[2]): pilot.sha(files[2])})
            pilot.validate_claim(claim, root / 'run', 'calibration', root=root)
            for key, value in (('max_resets', 7), ('fresh', True), ('calibration_sha256', 'other'),
                               ('cells', [[1, 999]])):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    pilot.validate_claim(dict(claim, **{key: value}), root / 'run', 'calibration', root=root)

    def test_calibration_requires_current_runtime_source_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'haic/algorithms/joint_control'
            directory.mkdir(parents=True)
            pins = {}
            for name in ('hud', 'motion', 'physics', 'observer', 'comparison', 'interval_comparison'):
                path = directory / (name + '.py')
                path.write_text('synthetic')
                pins[str(path)] = pilot.sha(path)
            calibration = dict(schema='haic-joint-temporal-interval-calibration-v1', source_pins=pins)
            pilot.validate_calibration(calibration, root)
            (directory / 'physics.py').write_text('modified')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                pilot.validate_calibration(calibration, root)


class MeasurementTests(unittest.TestCase):
    def test_full_act_clock_excludes_hashing_diagnostics_and_capture(self):
        events = []

        class Pixels:
            def tobytes(self):
                events.append('pixels')
                return b'pixels'

        class Model:
            last_diagnostics = dict(intervention=True)

            def act(self, observation):
                events.append('act')
                return [0., .1, 0.]

        holder = dict(environment=SimpleNamespace(last=dict(pre=state(), post=state(.08))))
        with tempfile.TemporaryFile(mode='w+') as stream:
            agent = pilot.MeasuredAgent(Model(), stream, holder, 'successor')
            def cpu():
                events.append('cpu')
                return 2. if events.count('cpu') == 1 else 2.25
            def wall():
                events.append('wall')
                return 3. if events.count('wall') == 1 else 3.5
            with patch.object(pilot.time, 'process_time', side_effect=cpu), \
                    patch.object(pilot.time, 'perf_counter', side_effect=wall), \
                    patch.object(pilot, 'memory_sample', return_value={}):
                agent.act(Pixels())
                agent.last_step_diagnostics()
            self.assertEqual(events, ['pixels', 'cpu', 'wall', 'act', 'wall', 'cpu'])
            stream.seek(0)
            rows = [json.loads(line) for line in stream]
            self.assertEqual(rows[0]['event'], 'act_intent')
            self.assertEqual(rows[1]['cpu_seconds'], .25)
            self.assertEqual(rows[1]['wall_seconds'], .5)
            self.assertNotIn('evaluation_only', rows[1])
            self.assertEqual(rows[2]['policy'], dict(intervention=True))
            self.assertIn('evaluation_only', rows[2])

    def test_failed_act_timing_is_preserved(self):
        model = SimpleNamespace(act=lambda obs: (_ for _ in ()).throw(ValueError('bad action')))
        with tempfile.TemporaryFile(mode='w+') as stream:
            agent = pilot.MeasuredAgent(model, stream, {}, 'successor')
            with self.assertRaises(ValueError):
                agent.act(SimpleNamespace(tobytes=lambda: b'pixels'))
            stream.seek(0)
            row = [json.loads(line) for line in stream][-1]
            self.assertIn('bad action', row['error'])
            self.assertGreaterEqual(row['cpu_seconds'], 0.)

    def test_external_prefix_check_happens_before_policy(self):
        model = SimpleNamespace(act=lambda obs: self.fail('policy must not run'))
        holder = dict(reference_decisions=[dict(observation_sha256='other')])
        with tempfile.TemporaryFile(mode='w+') as stream:
            agent = pilot.MeasuredAgent(model, stream, holder, 'successor')
            with self.assertRaisesRegex(ValueError, 'pre-divergence'):
                agent.act(SimpleNamespace(tobytes=lambda: b'pixels'))

    def test_percentiles_and_all_over_one_second(self):
        result = pilot.distribution([0., 1., 2., 3.])
        self.assertEqual(result['n'], 4)
        self.assertEqual(result['p50'], 1.5)
        self.assertAlmostEqual(result['p95'], 2.85)
        self.assertAlmostEqual(result['p99'], 2.97)
        self.assertEqual(result['over_1s_count'], 2)
        self.assertIsNone(pilot.distribution([])['p50'])
        self.assertEqual(pilot.rate(0, 0), dict(numerator=0, denominator=0, rate=None))


class AnalysisTests(unittest.TestCase):
    def test_initial_parity_canonicalizes_live_track_tuples(self):
        observer = SimpleNamespace(catalog=dict(track=[(0., .1, 2., 3.)], obstacles=[]),
            geometry_sha256='geometry', initial=state(), initial_observation_sha256='pixels')
        saved = json.loads(json.dumps(dict(catalog=observer.catalog, geometry_sha256='geometry',
                                          initial_state=state(), initial_observation_sha256='pixels')))
        self.assertNotEqual(saved['catalog'], observer.catalog)
        self.assertEqual(pilot.initial_record(observer), saved)

    def test_forecast_sequence_mismatch_never_counts_as_range_failure(self):
        rows, raw = forecasts()
        rows[2]['action'][1] = .2
        for pose in rows[0]['policy']['forecast']['poses'][0][1:]:
            pose[0] = 100.
        result = pilot.forecast_ranges(rows, raw)
        self.assertEqual(result['counts']['continuation_mismatch'], 1)
        self.assertEqual(result['counts']['qualified'], 0)
        self.assertEqual(result['counts']['range_miss'], 0)
        self.assertIsNone(result['range_miss_rate']['rate'])

    def test_exact_sequence_containment_and_real_range_miss(self):
        rows, raw = forecasts()
        self.assertEqual(pilot.forecast_ranges(rows, raw)['range_miss_rate'], pilot.rate(0, 1))
        raw[-1]['x'] += .1
        result = pilot.forecast_ranges(rows, raw)
        self.assertEqual(result['range_miss_rate'], pilot.rate(1, 1))
        self.assertNotIn('accidents', result)

    def test_shortened_terminal_hold_is_censored(self):
        rows, raw = forecasts()
        for current_rows, current_raw in ((rows[:-1], raw), (rows, raw[:-1])):
            result = pilot.forecast_ranges(current_rows, current_raw)
            self.assertEqual(result['counts']['censored'], 1)
            self.assertEqual(result['counts']['range_miss'], 0)

    def test_joint_same_scenario_entire_horizon_not_per_tick_switching(self):
        rows, raw = forecasts()
        first = rows[0]['policy']['forecast']['poses'][0]
        second = copy.deepcopy(first)
        first[1][0] = 1.
        second[-1][0] = 1.
        rows[0]['policy']['forecast']['poses'].append(second)
        self.assertEqual(pilot.forecast_ranges(rows, raw)['counts']['range_miss'], 1)

    def test_pose_transform_uses_anchor_body_axes_and_wraps_yaw(self):
        rows, raw = forecasts()
        angle = 3.141592653589793 / 2
        rows[0]['evaluation_only']['pre'].update(x=10., y=20., yaw=angle)
        for i, item in enumerate(raw, start=1):
            item.update(x=10. - i * .1, y=20., yaw=angle + 2 * 3.141592653589793)
        self.assertEqual(pilot.forecast_ranges(rows, raw)['counts']['range_miss'], 0)

    def test_invalid_forecast_evidence_fails_closed(self):
        rows, raw = forecasts()
        rows[0]['policy']['forecast']['position_residual'][-1] = float('nan')
        with self.assertRaises(ValueError):
            pilot.forecast_ranges(rows, raw)

    def test_failures_keep_time_distance_and_noncausal_label(self):
        rows, raw = forecasts()
        rows[1]['policy']['intervention'] = True
        raw[0]['contacts'] = [0]
        raw[8]['contacts'] = [1]
        raw[8]['wheel_road_contacts'] = [0, 0, 0, 0]
        rows[-1]['evaluation_only']['post']['environment_state']['damage'] = .2
        result = pilot.failure_events(rows, raw, dict(natural(), initial_state=state()))
        early = result['events'][0]
        self.assertFalse(early['after_intervention'])
        contact = next(e for e in result['events'] if e['kind'] == 'contact' and e['detail'] == [1])
        self.assertEqual(contact['decision_distance'], 1)
        self.assertAlmostEqual(contact['seconds_after_intervention'], .1)
        damage = next(e for e in result['events'] if e['kind'] == 'damage_increase')
        self.assertAlmostEqual(damage['t'], .32)
        self.assertIn('not proof', result['interpretation'])
        self.assertTrue(any(e['kind'] == 'dnf' for e in result['events']))

    def test_pair_parity_checks_initial_divergence_pre_and_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows, raw = forecasts()
            episodes = []
            for arm in pilot.ARMS:
                decision_path, raw_path = output / f'{arm}.decisions.jsonl', output / f'{arm}.raw.jsonl'
                for row in rows:
                    pilot.append(decision_path, row)
                for row in raw:
                    pilot.append(raw_path, row)
                episodes.append(dict(catalog={'track': [1]}, geometry_sha256='geometry', initial_state=state(),
                    initial_observation_sha256='pixels', decision_stream_file=decision_path.name,
                    decision_stream_sha256=pilot.sha(decision_path), raw_trace_file=raw_path.name,
                    raw_trace_sha256=pilot.sha(raw_path)))
            self.assertTrue(pilot.pair_parity(episodes[0], episodes[1], output)['exact_prefix'])
            changed = copy.deepcopy(rows)
            changed[1]['action'][0] = .1
            changed[2]['evaluation_only']['pre']['y'] = 90.
            path = output / 'successor.decisions.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in changed))
            episodes[1]['decision_stream_sha256'] = pilot.sha(path)
            result = pilot.pair_parity(episodes[0], episodes[1], output)
            self.assertTrue(result['exact_prefix'])
            self.assertEqual(result['prefix_decisions'], 1)
            changed[1]['observation_sha256'] = 'wrong at divergence'
            path.write_text(''.join(json.dumps(r) + '\n' for r in changed))
            episodes[1]['decision_stream_sha256'] = pilot.sha(path)
            self.assertFalse(pilot.pair_parity(episodes[0], episodes[1], output)['exact_prefix'])


class ExecutionTests(unittest.TestCase):
    def run_fake(self, directory, *, failure_index=None):
        output = Path(directory)
        pilot.save(output / 'preflight.json', preflight())
        pilot.save(output / 'operator-admission.json', admission())
        calls = []

        def fake_worker(command, **kwargs):
            index = int(command[-3])
            calls.append(index)
            slot = pilot.schedule()[index]
            pilot.append(output / 'reset-ledger.jsonl', dict(slot, event='reset_intent', protocol_sha256='protocol'))
            if index == failure_index:
                stream = (output / slot['file']).with_suffix('.decisions.jsonl')
                pilot.append(stream, dict(event='act', step=1, cpu_seconds=1.1, wall_seconds=1.2, error=None))
                with stream.open('a') as partial:
                    partial.write('{"event":')
                raise subprocess_timeout(command)
            rows, raw = forecasts()
            for row in rows:
                row['policy'] = dict(intervention=False, eligible=False, reasons=['unsupported'])
            path = output / slot['file']
            decision_path, raw_path = path.with_suffix('.decisions.jsonl'), path.with_suffix('.raw.jsonl')
            for row in rows:
                pilot.append(decision_path, dict(event='act', step=row['step'], cpu_seconds=.01, wall_seconds=.02, error=None))
                pilot.append(decision_path, row)
            for row in raw:
                pilot.append(raw_path, row)
            episode = dict(natural(reason='crash'), catalog={'track': [1]}, geometry_sha256='geometry',
                initial_state=state(), initial_observation_sha256='pixels',
                decision_stream_file=decision_path.name, decision_stream_sha256=pilot.sha(decision_path),
                raw_trace_file=raw_path.name, raw_trace_sha256=pilot.sha(raw_path),
                physical=dict(physical_contact_events=0), memory=dict(VmRSS=100, VmHWM=100))
            pilot.save(path, episode)
            pilot.append(output / 'reset-ledger.jsonl', dict(slot, event='episode_end', protocol_sha256='protocol'))
            return SimpleNamespace(returncode=0)

        p = dict(pilot.contract(), claim_sha256='claim', calibration_sha256='calibration')
        with patch.object(pilot, 'validate_frozen', return_value=p), \
                patch.object(pilot, 'resource_admission', return_value={}), \
                patch.object(pilot.time, 'time', return_value=101.), \
                patch.object(pilot.subprocess, 'run', side_effect=fake_worker), \
                patch.object(pilot, 'pair_parity', return_value=dict(exact_prefix=True)):
            report = pilot.execute(output, 'protocol', output / 'operator-admission.json')
        return report, calls

    def test_all_six_full_episodes_run_despite_ordinary_accidents(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, calls = self.run_fake(temporary)
            self.assertEqual(calls, list(range(6)))
            self.assertIsNone(report['operator_error'])
            self.assertEqual([r['status'] for r in report['rows']], ['completed'] * 6)
            self.assertEqual(len([r for r in pilot.json_rows(Path(temporary) / 'reset-ledger.jsonl')
                                  if r['event'] == 'reset_intent']), 6)
            with patch.object(pilot, 'validate_frozen', return_value=pilot.contract()):
                result = pilot.analyze(temporary, 'protocol')
            self.assertTrue(result['complete_six_natural_episodes'])
            self.assertFalse(result['successor_effect_evaluated'])
            self.assertEqual(result['by_arm']['successor']['completion'], pilot.rate(0, 3))
            self.assertEqual(result['by_arm']['successor']['abstain'], pilot.rate(12, 12))
            self.assertEqual(result['integrated_act']['successor']['cpu_seconds']['n'], 12)
            self.assertTrue(all(r['parity']['exact_prefix'] for r in result['pairs']))

    def test_timeout_preserves_partial_and_no_retry_or_later_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, calls = self.run_fake(temporary, failure_index=1)
            self.assertEqual(calls, [0, 1])
            self.assertIn('TimeoutExpired', report['operator_error'])
            self.assertEqual([r['status'] for r in report['rows']], ['completed', 'failed'] + ['unrun'] * 4)
            self.assertTrue(report['artifacts_sha256'])
            process = pilot.read_json(Path(temporary) / report['rows'][1]['process_file'])
            self.assertIn('TimeoutExpired', process['error'])
            with patch.object(pilot, 'validate_frozen', return_value=pilot.contract()):
                summary = pilot.analyze(temporary, 'protocol')
            self.assertFalse(summary['complete_six_natural_episodes'])
            self.assertEqual(summary['reset_intents'], 2)
            self.assertEqual(summary['by_arm']['successor']['missing_episodes'], 3)
            self.assertEqual(summary['integrated_act']['successor']['cpu_seconds']['n'], 1)
            self.assertEqual(summary['integrated_act']['successor']['cpu_seconds']['over_1s_count'], 1)
            self.assertTrue(summary['episodes'][1]['truncated_decision_tail'])
            with patch.object(pilot, 'validate_frozen', return_value=dict(
                    claim_sha256='claim', calibration_sha256='calibration')), \
                    patch.object(pilot.time, 'time', return_value=101.), \
                    patch.object(pilot.subprocess, 'run') as never_run:
                with self.assertRaisesRegex(ValueError, 'never restart'):
                    pilot.execute(temporary, 'protocol', Path(temporary) / 'operator-admission.json')
                never_run.assert_not_called()

    def test_natural_finished_pair_is_not_matched_after_provenance_failure(self):
        for parity_ok in (False, True):
            with self.subTest(parity_ok=parity_ok), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                report, _ = self.run_fake(temporary)
                for row in report['rows'][:2]:
                    path = output / row['file']
                    episode = pilot.read_json(path)
                    episode.update(completed=True, lapTimeMs=320, retire_reason=None)
                    path.write_text(json.dumps(episode))
                    row['sha256'] = pilot.sha(path)
                    report['artifacts_sha256'][path.name] = pilot.sha(path)
                if parity_ok:
                    report['rows'][1]['status'] = 'failed'
                (output / 'episode-report.json').write_text(json.dumps(report))
                with patch.object(pilot, 'validate_frozen', return_value=pilot.contract()), \
                        patch.object(pilot, 'pair_parity', return_value=dict(exact_prefix=parity_ok)):
                    result = pilot.analyze(output, 'protocol')
                self.assertTrue(result['pairs'][0]['both_finished'])
                self.assertFalse(result['pairs'][0]['matched'])
                self.assertIsNone(result['pairs'][0]['lap_delta_ms'])

    def test_preflight_only_passes_import_only_to_subprocess(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'model').mkdir()
            receipts = preflight()['arms']
            with patch.object(pilot, 'validate_frozen', return_value={}), \
                    patch.object(pilot.subprocess, 'run', side_effect=[
                        SimpleNamespace(returncode=0, stdout=json.dumps(r)) for r in receipts]) as runner:
                result = pilot.preflight(output, 'protocol')
            self.assertEqual(result['environment_resets'], 0)
            self.assertTrue(all(call.args[0][-1] == 'True' for call in runner.call_args_list))

    def test_worker_passes_strict_full_episode_contract_without_real_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            pilot.save(output / 'preflight.json', preflight())
            pilot.save(output / 'admission.json', admission())
            pilot.save(output / 'run-start.json', dict(protocol_sha256='protocol',
                preflight_sha256=pilot.sha(output / 'preflight.json'),
                admission_sha256=pilot.sha(output / 'admission.json')))
            fake_audit = SimpleNamespace(legacy=SimpleNamespace(physical_metrics=lambda *args: {}))
            p = dict(pilot.contract(), claim_sha256='claim', calibration_sha256='calibration')
            with patch.object(pilot, 'validate_frozen', return_value=p), \
                    patch.object(pilot, 'validate_admission'), \
                    patch.object(pilot.resource, 'setrlimit'), \
                    patch.object(pilot.signal, 'alarm'), \
                    patch.object(pilot, 'memory_sample', return_value=dict(VmRSS=1, VmHWM=1)), \
                    patch.object(pilot, 'load_runtime') as runtime:
                from unittest.mock import Mock
                evaluator, factory = Mock(return_value=natural()), Mock()
                runtime.return_value = (object(), evaluator, factory, fake_audit, object())
                pilot.worker(output, 0, 'protocol')
                factory.assert_not_called()
                kwargs = evaluator.call_args.kwargs
                self.assertTrue(kwargs['fail_on_invalid_action'])
                self.assertTrue(kwargs['capture_trace'])
                self.assertEqual(kwargs['max_decisions'], 1200)
                self.assertEqual((kwargs['track_id'], kwargs['seed']), pilot.CELLS[0])
                self.assertIsInstance(kwargs['agent'], pilot.MeasuredAgent)


def subprocess_timeout(command):
    return pilot.subprocess.TimeoutExpired(command, 600)


if __name__ == '__main__':
    unittest.main()
