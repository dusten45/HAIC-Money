"""Synthetic/file-only checks. No real environment, collection, or frozen run."""

import copy
import importlib
import inspect
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

from scripts import run_joint_envelope_pilot as pilot
from tests.test_joint_temporal_pilot import forecasts, natural, preflight, state


def gate() -> dict[str, Any]:
    rows = []
    for cell in pilot.PAIRED_CELLS:
        rows.append(dict(cell=list(cell), status='COMPLETE', parity=dict(passed=True),
            new_would_select=cell == pilot.CELL, actual_cost_supported=True,
            actual_cost_interval=[-.3, -.2], new_predicted_interval=[-.4, -.25],
            actual_safe=True, contained=True, continuation='exact_fixed_H4', new_data_fit=False,
            physical_labels=[{}, dict(safe=True, obstacle_contact=False, all_wheels_lost=False, damage_increase=0.)],
            containment=[dict(contained=True), dict(contained=True)]))
    return dict(schema='haic-joint-envelope-pairs-v1-analysis', operator_error=None, rows=rows,
        gate=dict(outcome='PASS', potential_full_pilot=True, blockers=[], selected_pairs=1,
                  selected_evaluable_pairs=1, supported_selected_benefit=dict(numerator=1)))


def calibration() -> dict[str, Any]:
    return dict(schema='haic-joint-temporal-paired-envelope-calibration-v1', calibration_complete=True,
        performance=dict(lower=.05, upper=.05, floor=.05),
        absolute_calibration=dict(schema='haic-joint-temporal-interval-calibration-v1', calibration_complete=True,
            position_residual=[0.] + [.01] * 16, yaw_residual=[0.] + [.01] * 16))


def admission() -> dict[str, Any]:
    return dict(schema='haic-joint-envelope-pilot-admission-v1', admitted=True, protocol_sha256='protocol',
        claim_sha256='claim', calibration_sha256=pilot.CALIBRATION_SHA, created_unix_s=100., expires_unix_s=1300.)


def protocol(output) -> dict[str, Any]:
    return dict(pilot.contract(), claim_sha256='claim', source_sha256={},
        gate_binding=dict(sha256=dict(analysis='a', protocol='p', report='r'), initial_reference=dict(
            track=[[0., 0., 0., 1.]], obstacles=[], initial_observation_sha256='pixels')))


def saved_comparison(selected=True) -> dict[str, Any]:
    return dict(common_support=True, cost_supported=[True, True], delta_interval=[[0., 0.], [-.4, -.25]],
                absolute_supported=[[True] * 19, [True] * 19], veto=[[False] * 19, [not selected] * 19])


def write_gate(directory):
    directory = Path(directory)
    prior = dict(schema='haic-joint-envelope-pairs-v1', calibration_sha256=pilot.CALIBRATION_SHA,
                 cells=[list(c) for c in pilot.PAIRED_CELLS])
    pilot.save(directory / 'protocol.json', prior)
    selected = directory / f'{pilot.CELL[0]}-{pilot.CELL[1]}-baseline'
    selected.mkdir()
    pilot.save(selected / 'static.json', dict(track=[[0., 0., 0., 1.]], bodies=dict(obstacle=[])))
    pilot.append(selected / 'boundaries.jsonl', dict(index=0, image_sha256='pixels'))
    for track, seed in pilot.PAIRED_CELLS:
        path = directory / f'{track}-{seed}-baseline/anchor.json'
        path.parent.mkdir(exist_ok=True)
        pilot.save(path, dict(schema='haic-joint-envelope-pairs-v1-anchor',
            protocol_sha256=pilot.sha(directory / 'protocol.json'), calibration_sha256=pilot.CALIBRATION_SHA,
            corrected=saved_comparison((track, seed) == pilot.CELL)))
    report = dict(protocol_sha256=pilot.sha(directory / 'protocol.json'), operator_error=None,
        rows=[dict(track_id=t, seed=s, arm=a, status='COMPLETE') for t, s in pilot.PAIRED_CELLS
              for a in ('baseline', 'repeat', 'alternative')],
        artifacts_sha256={str(p.relative_to(directory)): pilot.sha(p) for p in directory.glob('*-baseline/*')})
    pilot.save(directory / 'report.json', report)
    analysis = dict(gate(), protocol_sha256=pilot.sha(directory / 'protocol.json'),
                    report_sha256=pilot.sha(directory / 'report.json'))
    pilot.save(directory / 'analysis.json', analysis)
    return directory / 'analysis.json'


def reseal_gate(directory, anchors=()):
    """Synthetic mutation fixture: keep top-level pins valid to test inner guards."""
    directory = Path(directory)
    report_path = directory / 'report.json'
    report = pilot.read_json(report_path)
    for path in anchors:
        report['artifacts_sha256'][str(path.relative_to(directory))] = pilot.sha(path)
    report_path.write_text(json.dumps(report))
    analysis_path = directory / 'analysis.json'
    analysis = pilot.read_json(analysis_path)
    analysis['report_sha256'] = pilot.sha(report_path)
    analysis_path.write_text(json.dumps(analysis))


class GateTests(unittest.TestCase):
    def test_only_first_actual_qualifying_cell_not_requested_arbitrary_cell(self):
        self.assertEqual(pilot.first_qualifying(gate()), list(pilot.CELL))
        value = gate()
        value['rows'][0]['new_would_select'] = True
        value['gate'].update(selected_pairs=2, selected_evaluable_pairs=2,
                             supported_selected_benefit=dict(numerator=2))
        with self.assertRaisesRegex(ValueError, 'first qualifying'):
            pilot.first_qualifying(value)

    def test_zero_selection_or_false_pass_is_rejected(self):
        for change in ('zero', 'missing', 'not_pass', 'wrong_order'):
            value = gate()
            if change == 'zero':
                value['rows'][-1]['new_would_select'] = False
            elif change == 'missing':
                del value['rows'][-1]['new_would_select']
            elif change == 'not_pass':
                value['gate']['outcome'] = 'NOT_MET'
            else:
                value['rows'].reverse()
            with self.subTest(change=change), self.assertRaises(ValueError):
                pilot.first_qualifying(value)

    def test_selected_counterexample_or_unknown_blocks_even_with_another_win(self):
        cases = [('actual_safe', False), ('contained', False), ('actual_cost_supported', False),
                 ('actual_cost_interval', [-.2, -.05]), ('actual_cost_interval', None),
                 ('new_predicted_interval', [-.2, float('nan')]), ('status', 'PARTIAL'),
                 ('continuation', 'different_actions'), ('new_data_fit', True)]
        for key, replacement in cases:
            value = gate()
            value['rows'][0]['new_would_select'] = True
            value['rows'][0][key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot.first_qualifying(value)

    def test_selected_physical_and_range_labels_are_cross_checked(self):
        for where, key, replacement in [('physical_labels', 'damage_increase', .2),
                ('physical_labels', 'obstacle_contact', True), ('physical_labels', 'all_wheels_lost', True),
                ('containment', 'contained', False)]:
            value = gate()
            value['rows'][-1][where][1][key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot.first_qualifying(value)

    def test_unselected_unfavorable_region_is_not_a_universal_gate(self):
        value = gate()
        value['rows'][0].update(actual_cost_interval=[1., 2.], actual_safe=False, contained=False)
        self.assertEqual(pilot.first_qualifying(value), list(pilot.CELL))

    def test_gate_binds_protocol_report_calibration_and_initial_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_gate(temporary)
            binding = pilot.gate_binding(path, pilot.CALIBRATION_SHA)
            self.assertEqual(binding['selected_cell'], list(pilot.CELL))
            self.assertEqual(set(binding['sha256']), {'analysis', 'protocol', 'report'})
            self.assertEqual(binding['initial_reference']['initial_observation_sha256'], 'pixels')
            expected_anchors = {str(Path(temporary) / f'{t}-{s}-baseline/anchor.json'): pilot.sha(
                Path(temporary) / f'{t}-{s}-baseline/anchor.json') for t, s in pilot.PAIRED_CELLS}
            self.assertEqual(binding['anchor_source_sha256'], expected_anchors)
            with self.assertRaisesRegex(ValueError, 'binding differs'):
                pilot.gate_binding(path, 'changed-calibration')
            report_path = Path(temporary) / 'report.json'
            report = pilot.read_json(report_path)
            report['operator_error'] = 'partial'
            report_path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                pilot.gate_binding(path, pilot.CALIBRATION_SHA)

    def test_report_sealed_risk_veto_cannot_be_labeled_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_gate(temporary)
            anchor_path = Path(temporary) / '3-3184000005-baseline/anchor.json'
            anchor = pilot.read_json(anchor_path)
            anchor['corrected']['veto'][1][7] = True
            anchor_path.write_text(json.dumps(anchor))
            reseal_gate(temporary, [anchor_path])
            with self.assertRaisesRegex(ValueError, 'selection/interval differs'):
                pilot.gate_binding(path, pilot.CALIBRATION_SHA)

    def test_altered_interval_rejected_even_for_unselected_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_gate(temporary)
            analysis = pilot.read_json(path)
            analysis['rows'][0]['new_predicted_interval'][1] = -.24
            path.write_text(json.dumps(analysis))
            with self.assertRaisesRegex(ValueError, 'selection/interval differs'):
                pilot.gate_binding(path, pilot.CALIBRATION_SHA)

    def test_changed_or_unbound_anchor_hash_rejected(self):
        for mutation in ('changed_bytes', 'missing_pin'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                path = write_gate(temporary)
                anchor_path = Path(temporary) / '1-3184000003-baseline/anchor.json'
                if mutation == 'changed_bytes':
                    with anchor_path.open('a') as stream:
                        stream.write(' ')
                else:
                    report_path = Path(temporary) / 'report.json'
                    report = pilot.read_json(report_path)
                    del report['artifacts_sha256'][str(anchor_path.relative_to(temporary))]
                    report_path.write_text(json.dumps(report))
                    reseal_gate(temporary)
                with self.assertRaisesRegex(ValueError, 'anchor is not sealed'):
                    pilot.gate_binding(path, pilot.CALIBRATION_SHA)

    def test_earlier_false_selection_cannot_hide_selected_candidate_or_counterexample(self):
        for actual_safe in (True, False):
            with self.subTest(actual_safe=actual_safe), tempfile.TemporaryDirectory() as temporary:
                path = write_gate(temporary)
                anchor_path = Path(temporary) / '1-3184000003-baseline/anchor.json'
                anchor = pilot.read_json(anchor_path)
                anchor['corrected']['veto'][1] = [False] * 19
                anchor_path.write_text(json.dumps(anchor))
                analysis = pilot.read_json(path)
                analysis['rows'][0]['actual_safe'] = actual_safe
                path.write_text(json.dumps(analysis))
                reseal_gate(temporary, [anchor_path])
                with self.assertRaisesRegex(ValueError, 'selection/interval differs'):
                    pilot.gate_binding(path, pilot.CALIBRATION_SHA)

    def test_saved_predicate_uses_all_unchanged_guards(self):
        self.assertTrue(pilot.saved_prediction_selects(saved_comparison()))
        for key, value in (('common_support', False), ('cost_supported', [True, False]),
                ('delta_interval', [[0., 0.], [-.4, -.05]]), ('delta_interval', [[0., 0.], [None, None]]),
                ('absolute_supported', [[True] * 19, [True] * 18 + [False]]),
                ('veto', [[False] * 19, [False] * 18 + [True]])):
            comparison = saved_comparison()
            comparison[key] = value
            with self.subTest(key=key):
                self.assertFalse(pilot.saved_prediction_selects(comparison))
        for key in ('absolute_supported', 'veto'):
            comparison = saved_comparison()
            comparison[key][1] = []
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot.saved_prediction_selects(comparison)

    def test_not_found_requires_consistent_absence_of_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_gate(temporary)
            analysis = pilot.read_json(path)
            analysis['rows'][0] = dict(cell=list(pilot.PAIRED_CELLS[0]), status='NOT_FOUND')
            path.write_text(json.dumps(analysis))
            report_path = Path(temporary) / 'report.json'
            report = pilot.read_json(report_path)
            for row, status in zip(report['rows'][:3], ('NOT_FOUND', 'SKIPPED_NOT_FOUND', 'SKIPPED_NOT_FOUND')):
                row['status'] = status
            report_path.write_text(json.dumps(report))
            reseal_gate(temporary)
            with self.assertRaisesRegex(ValueError, 'NOT_FOUND pair must have no saved anchor'):
                pilot.gate_binding(path, pilot.CALIBRATION_SHA)
            anchor_path = Path(temporary) / '1-3184000003-baseline/anchor.json'
            anchor_path.unlink()
            with self.assertRaisesRegex(ValueError, 'NOT_FOUND pair must have no saved anchor'):
                pilot.gate_binding(path, pilot.CALIBRATION_SHA)
            report = pilot.read_json(report_path)
            del report['artifacts_sha256'][str(anchor_path.relative_to(temporary))]
            report_path.write_text(json.dumps(report))
            reseal_gate(temporary)
            binding = pilot.gate_binding(path, pilot.CALIBRATION_SHA)
            self.assertEqual(binding['selected_cell'], list(pilot.CELL))
            self.assertEqual(len(binding['anchor_source_sha256']), 2)


class ContractTests(unittest.TestCase):
    def test_exact_two_slots_no_six_slot_defaults(self):
        self.assertEqual(len(pilot.schedule()), 2)
        self.assertEqual(pilot.contract()['cells'], [[3, 3184000005]])
        self.assertEqual(pilot.LIMITS['max_resets'], 2)
        self.assertEqual(pilot.LIMITS['total_seconds'], 1200)
        self.assertEqual(pilot.LIMITS['child_seconds'], 600)
        self.assertEqual(pilot.contract()['environment']['max_decisions'], 1200)
        self.assertEqual(pilot.contract()['environment']['warmup_raw'], 51)
        self.assertEqual(pilot.contract()['max_interventions'], 40)
        self.assertEqual(pilot.old.LIMITS['max_resets'], 6)
        self.assertEqual(len(pilot.old.schedule()), 6)
        with self.assertRaises(ValueError):
            pilot.worker_command('/tmp/kilo/test', 2, 'protocol')
        command = pilot.worker_command('/tmp/kilo/test', 1, 'protocol')
        self.assertIn('run_joint_envelope_pilot', command[4])
        self.assertEqual(command[:3], [str(pilot.old.PYTHON), '-I', '-B'])

    def test_admission_cannot_inherit_old3600_second_budget(self):
        p = dict(claim_sha256='claim')
        pilot.validate_admission(admission(), 'protocol', p, now=101, starting=True)
        for replacement in ({'expires_unix_s': 3700.}, {'schema': 'haic-joint-temporal-pilot-admission-v1'},
                            {'calibration_sha256': 'old'}, {'admitted': False}):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                pilot.validate_admission(dict(admission(), **replacement), 'protocol', p, now=101)
        with self.assertRaises(ValueError):
            pilot.validate_admission(admission(), 'protocol', p, now=401, starting=True)

    def test_claim_requires_new_and_reused_source_plus_gate_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / n for n in (pilot.OPERATOR, pilot.SUCCESSOR, pilot.COMPARATOR, pilot.old.OPERATOR)]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic')
            evidence = root / 'evidence.json'
            evidence.write_text('{}')
            hashes = dict(analysis='a', protocol='p', report='r')
            claim = dict(schema='haic-joint-envelope-pilot-consumed-train-v1', partition='TRAIN', fresh=False,
                cells=[list(pilot.CELL)], max_resets=2, run_directory=str(root / 'run'),
                calibration_sha256=pilot.CALIBRATION_SHA, gate_sha256=hashes,
                source_sha256={str(p): pilot.sha(p) for p in paths}, evidence_sha256={str(evidence): pilot.sha(evidence)})
            pilot.validate_claim(claim, root / 'run', hashes, root=root)
            for replacement in ({'max_resets': 6}, {'cells': [[1, 3184000003]]}, {'gate_sha256': {}}, {'fresh': True}):
                with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                    pilot.validate_claim(dict(claim, **replacement), root / 'run', hashes, root=root)

    def test_nested_calibration_preserves_entire_absolute_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior = root / 'runs/joint-temporal-interval-v1/calibration.json'
            prior.parent.mkdir(parents=True)
            value = calibration()
            pilot.save(prior, value['absolute_calibration'])
            source = root / pilot.COMPARATOR
            source.parent.mkdir(parents=True)
            source.write_text('synthetic')
            value.update(source_pins={str(source): pilot.sha(source)}, evidence_pins={str(prior): pilot.sha(prior)})
            with patch.object(pilot, 'ABSOLUTE_SHA', pilot.sha(prior)), patch.object(pilot.old, 'validate_calibration') as old_validate:
                pilot.validate_calibration(value, root)
                self.assertEqual(old_validate.call_args.args[0], value['absolute_calibration'])
                changed = copy.deepcopy(value)
                changed['absolute_calibration']['position_residual'][-1] = 999.
                with self.assertRaises(ValueError):
                    pilot.validate_calibration(changed, root)
                with self.assertRaises(ValueError):
                    pilot.validate_calibration(value['absolute_calibration'], root)

    def test_loader_uses_baseline_and_passes_full_new_calibration_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            pilot.save(output / 'calibration.json', calibration())
            module: Any = ModuleType('haic.algorithms.joint_control.envelope_successor')
            module.EnvelopeSuccessor = Mock(return_value='wrapped')
            legacy_runtime = (object(), object(), object(), object(), object())
            with patch.object(pilot.old, 'load_runtime', return_value=legacy_runtime) as loader, \
                    patch.dict(pilot.sys.modules, {module.__name__: module}):
                result = pilot.load_runtime(output, dict(source_sha256={}), 'successor')
                self.assertEqual(loader.call_args.args[2:], ('baseline', 3184000005))
                self.assertEqual(result[0], 'wrapped')
                self.assertEqual(module.EnvelopeSuccessor.call_args.args, (legacy_runtime[0], calibration()))
                module.EnvelopeSuccessor.reset_mock()
                self.assertIs(pilot.load_runtime(output, dict(source_sha256={}), 'baseline')[0], legacy_runtime[0])
                module.EnvelopeSuccessor.assert_not_called()

    def test_actual_adapter_import_contract_without_policy_or_environment_call(self):
        module = importlib.import_module('haic.algorithms.joint_control.envelope_successor')
        self.assertEqual(list(inspect.signature(module.EnvelopeSuccessor).parameters),
                         ['champion', 'calibration', 'never_intervene'])
        self.assertEqual(module.EnvelopeSuccessor.__mro__[1].__name__, 'TemporalSuccessor')

    def test_paired_initial_reference_compares_catalog_image_not_dynamic_cap(self):
        reference = protocol(None)['gate_binding']['initial_reference']
        observer = SimpleNamespace(catalog=dict(track=[(0., 0., 0., 1.)], obstacles=[]),
            initial=dict(time_limit=dict(maximum=5050)), geometry_sha256='geometry', initial_observation_sha256='pixels')
        record = pilot.old.initial_record(observer)
        pilot.check_initial(reference, record)
        record['initial_state']['time_limit']['maximum'] = 506
        pilot.check_initial(reference, record)
        record['initial_observation_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            pilot.check_initial(reference, record)


class MetricsTests(unittest.TestCase):
    def test_gain_sign_and_comparison_denominators(self):
        rows, _ = forecasts()
        intervals = [[-.4, -.2], [.1, .3], [-.2, .2], None]
        for row, bounds in zip(rows, intervals):
            row['policy'] = dict(eligible=True, intervention=False, proposal_action=row['action'],
                                 comparison=dict(delta_interval=[[0., 0.], bounds]))
        metrics = pilot.policy_metrics(rows, 'successor')
        self.assertEqual(metrics['candidate_orders'], dict(alternative=1, baseline=1, ambiguous=1, unsupported=1, denominator=4))
        self.assertEqual(metrics['comparisons'], pilot.rate(4, 4))
        self.assertEqual(metrics['abstain'], pilot.rate(4, 4))
        rows[0]['policy'].update(intervention=True, proposal_action=[.2, .1, 0.])
        self.assertEqual(pilot.policy_metrics(rows, 'successor')['actual_proposal_differences'], 1)
        rows[0]['policy']['intervention'] = False
        with self.assertRaises(ValueError):
            pilot.policy_metrics(rows, 'successor')

    def test_baseline_and_empty_denominators_not_fabricated(self):
        self.assertIsNone(pilot.policy_metrics([], 'successor')['comparisons']['rate'])
        rows, _ = forecasts()
        for row in rows:
            row['policy'] = {}
        result = pilot.policy_metrics(rows, 'baseline')
        self.assertIsNone(result['abstain'])
        self.assertEqual(result['comparisons'], pilot.rate(0, 4))

    def test_range_analysis_reused_with_continuation_qualification(self):
        rows, raw = forecasts()
        self.assertEqual(pilot.old.forecast_ranges(rows, raw)['range_miss_rate'], pilot.rate(0, 1))
        rows[2]['action'][1] = .2
        result = pilot.old.forecast_ranges(rows, raw)
        self.assertEqual(result['counts']['continuation_mismatch'], 1)
        self.assertEqual(result['range_miss_rate'], pilot.rate(0, 0))

    def test_partial_record_preserves_only_complete_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'decisions.jsonl'
            path.write_text('{"event":"act","cpu_seconds":1.2}\n{"event":')
            rows, truncated = pilot.partial_stream(path)
            self.assertTrue(truncated)
            self.assertEqual(rows[0]['cpu_seconds'], 1.2)
            path.write_text('invalid\n{}\n')
            with self.assertRaises(ValueError):
                pilot.partial_stream(path)


class LifecycleTests(unittest.TestCase):
    def fake_run(self, output, *, fail=None, intervene=False, finish=False):
        output = Path(output)
        pilot.save(output / 'preflight.json', preflight())
        pilot.save(output / 'caller-admission.json', admission())
        pilot.save(output / 'calibration.json', calibration())
        calls = []

        def fake_subprocess(command, **kwargs):
            index = int(command[-3])
            calls.append(index)
            self.assertLessEqual(kwargs['timeout'], 600)
            slot = pilot.schedule()[index]
            path = output / slot['file']
            pilot.append(output / 'reset-ledger.jsonl', dict(slot, event='reset_intent', protocol_sha256='protocol'))
            if index == fail:
                pilot.append(path.with_suffix('.decisions.jsonl'), dict(event='act', step=1, cpu_seconds=1.1, wall_seconds=1.2))
                with path.with_suffix('.decisions.jsonl').open('a') as stream:
                    stream.write('{"event":')
                path.write_text('{"incomplete_episode":')
                raise pilot.subprocess.TimeoutExpired(command, 600)
            rows, raw = forecasts()
            for row in rows:
                forecast = row['policy'].get('forecast') if index else None
                row['policy'] = dict(intervention=False, eligible=False, proposal_action=row['action'].copy(), forecast=forecast)
                pilot.append(path.with_suffix('.decisions.jsonl'), dict(event='act', step=row['step'], cpu_seconds=.01, wall_seconds=.02))
            if index and intervene:
                # Diverge on the last step, never require equality afterwards.
                rows[-1]['action'] = [.2, .1, 0.]
                rows[-1]['policy'].update(intervention=True, eligible=True)
            for row in rows:
                pilot.append(path.with_suffix('.decisions.jsonl'), row)
            for row in raw:
                pilot.append(path.with_suffix('.raw.jsonl'), row)
            episode = dict(natural(completed=finish, reason=None if finish else 'crash'),
                catalog={'track': [1]}, geometry_sha256='geometry', initial_state=state(),
                initial_observation_sha256='pixels', warmup_raw=51,
                raw_trace_file=path.with_suffix('.raw.jsonl').name, raw_trace_sha256=pilot.sha(path.with_suffix('.raw.jsonl')),
                decision_stream_file=path.with_suffix('.decisions.jsonl').name,
                decision_stream_sha256=pilot.sha(path.with_suffix('.decisions.jsonl')),
                physical=dict(physical_contact_events=0, any_wheel_offroad_ticks=0), memory=dict(VmRSS=100, VmHWM=100))
            pilot.save(path, episode)
            pilot.append(output / 'reset-ledger.jsonl', dict(slot, event='episode_end', protocol_sha256='protocol'))
            return SimpleNamespace(returncode=0)

        p = protocol(output)
        with patch.object(pilot, 'validate_frozen', return_value=p), \
                patch.object(pilot.old, 'resource_admission', return_value={}), \
                patch.object(pilot.time, 'time', return_value=101.), \
                patch.object(pilot.subprocess, 'run', side_effect=fake_subprocess):
            report = pilot.execute(output, 'protocol', output / 'caller-admission.json')
        return report, calls

    def test_two_natural_accidents_complete_no_extra_resets_noop_na_lap(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, calls = self.fake_run(temporary)
            self.assertEqual(calls, [0, 1])
            self.assertIsNone(report['operator_error'])
            with patch.object(pilot, 'validate_frozen', return_value=protocol(temporary)):
                summary = pilot.analyze(temporary, 'protocol')
            self.assertEqual(summary['reset_intents'], 2)
            self.assertTrue(summary['complete_two_natural_episodes'])
            self.assertEqual(summary['effect_status'], 'NOT_EVALUATED')
            for row in summary['episodes']:
                self.assertEqual(row['progress'], 1.)
                self.assertFalse(row['completed'])
                self.assertIsNone(row['finished_lap_mean_ms'])
                self.assertIsNone(row['failures']['after_intervention']['rate'])

    def test_real_difference_qualifies_only_one_road_and_finish_only_lap(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, calls = self.fake_run(temporary, intervene=True, finish=True)
            with patch.object(pilot, 'validate_frozen', return_value=protocol(temporary)):
                summary = pilot.analyze(temporary, 'protocol')
            self.assertEqual(calls, [0, 1])
            self.assertEqual(summary['effect_status'], 'EVALUATED_ONE_ROAD')
            self.assertEqual(summary['pair']['lap_delta_ms'], 0)
            self.assertEqual(summary['pair']['parity']['first_changed_step'], 4)
            self.assertEqual(summary['episodes'][1]['interventions'], pilot.rate(1, 4))
            self.assertEqual(summary['episodes'][1]['forecast']['counts']['continuation_mismatch'], 1)
            self.assertIsNone(summary['episodes'][1]['forecast']['range_miss_rate']['rate'])

    def test_partial_timeout_retains_all_over_one_second_and_forbids_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, calls = self.fake_run(temporary, fail=1)
            self.assertEqual(calls, [0, 1])
            self.assertIn('TimeoutExpired', report['operator_error'])
            with patch.object(pilot, 'validate_frozen', return_value=protocol(temporary)):
                summary = pilot.analyze(temporary, 'protocol')
            candidate = summary['episodes'][1]
            self.assertTrue(candidate['partial'])
            self.assertTrue(candidate['truncated_decision_tail'])
            self.assertTrue(candidate['partial_episode_json'])
            self.assertEqual(candidate['integrated_act']['cpu_seconds']['over_1s_count'], 1)
            self.assertEqual(len(candidate['integrated_act']['all_over_1s']), 1)
            self.assertEqual(summary['effect_status'], 'NOT_EVALUATED')
            with patch.object(pilot, 'validate_frozen', return_value=protocol(temporary)), \
                    patch.object(pilot.time, 'time', return_value=101.), \
                    patch.object(pilot.subprocess, 'run') as never:
                with self.assertRaisesRegex(ValueError, 'no retry'):
                    pilot.execute(temporary, 'protocol', Path(temporary) / 'caller-admission.json')
                never.assert_not_called()

    def test_failed_provenance_never_produces_matched_lap_delta(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, _ = self.fake_run(temporary, finish=True, intervene=True)
            report['rows'][1]['status'] = 'failed'
            (Path(temporary) / 'episode-report.json').write_text(json.dumps(report))
            with patch.object(pilot, 'validate_frozen', return_value=protocol(temporary)):
                summary = pilot.analyze(temporary, 'protocol')
            self.assertTrue(summary['pair']['both_finished'])
            self.assertFalse(summary['pair']['matched'])
            self.assertIsNone(summary['pair']['lap_delta_ms'])
            self.assertEqual(summary['effect_status'], 'NOT_EVALUATED')

    def test_preflight_exact_two_import_only_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            responses = [SimpleNamespace(returncode=0, stdout=json.dumps(row)) for row in preflight()['arms']]
            with patch.object(pilot, 'validate_frozen', return_value=protocol(temporary)), \
                    patch.object(pilot.subprocess, 'run', side_effect=responses) as launch:
                result = pilot.preflight(temporary, 'protocol')
            self.assertEqual(result['environment_resets'], 0)
            self.assertEqual(launch.call_count, 2)
            self.assertTrue(all(call.args[0][-1] == 'True' for call in launch.call_args_list))

    def test_worker1200cap_one_writeahead_reset_exact51_warmup(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            pilot.save(output / 'preflight.json', preflight())
            pilot.save(output / 'admission.json', admission())
            pilot.save(output / 'run-start.json', dict(protocol_sha256='protocol',
                preflight_sha256=pilot.sha(output / 'preflight.json'), admission_sha256=pilot.sha(output / 'admission.json')))
            observed = []

            class FakeObserver:
                def __init__(self, environment, raw, ledger, slot, digest, footprint):
                    self.active, self.reset_count = False, 0
                    self.catalog = dict(track=[(0., 0., 0., 1.)], obstacles=[])
                    self.geometry_sha256, self.initial_observation_sha256 = 'geometry', 'pixels'
                    self.initial, self.obstacles = state(), []
                    def reset():
                        self.reset_count += 1
                        pilot.append(ledger, dict(slot, event='reset_intent', protocol_sha256=digest))
                        observed.append('physical-reset-after-intent' if pilot.old.json_rows(ledger) else 'bad')
                    self.unwrapped = SimpleNamespace(reset=reset, step=lambda action: observed.append('raw'))

                def reset(self):
                    self.unwrapped.reset()
                    for _ in range(51):
                        self.unwrapped.step(None)
                    self.active = True
                    return object(), {}

            def evaluator(**kwargs):
                self.assertEqual(kwargs['max_decisions'], 1200)
                self.assertTrue(kwargs['fail_on_invalid_action'])
                self.assertIsInstance(kwargs['agent'], pilot.old.MeasuredAgent)
                env = kwargs['environment_factory'](track_id=3, seed=3184000005, max_decisions=1200)
                env.reset()
                with self.assertRaisesRegex(ValueError, 'second reset'):
                    env.reset()
                with self.assertRaises(ValueError):
                    kwargs['environment_factory'](track_id=1, seed=3184000013, max_decisions=64)
                return natural()

            audit = SimpleNamespace(make_observer=FakeObserver, legacy=SimpleNamespace(physical_metrics=lambda *args: {}))
            with patch.object(pilot, 'validate_frozen', return_value=protocol(output)), \
                    patch.object(pilot, 'validate_admission'), patch.object(pilot.resource, 'setrlimit'), \
                    patch.object(pilot.signal, 'alarm'), patch.object(pilot.old, 'resource_admission', return_value={}), \
                    patch.object(pilot.old, 'memory_sample', return_value=dict(VmRSS=1, VmHWM=1)), \
                    patch.object(pilot, 'load_runtime', return_value=(object(), evaluator, Mock(), audit, object())):
                pilot.worker(output, 0, 'protocol')
            self.assertEqual(observed.count('raw'), 51)
            self.assertEqual(observed[0], 'physical-reset-after-intent')
            self.assertEqual(len([r for r in pilot.old.json_rows(output / 'reset-ledger.jsonl') if r['event'] == 'reset_intent']), 1)
            self.assertEqual(pilot.read_json(output / pilot.schedule()[0]['file'])['warmup_raw'], 51)


if __name__ == '__main__':
    unittest.main()
