"""Prospective speed gates and source-bound two-phase protocol integrity."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from contextlib import ExitStack

from tools import competition_camera_speed_gate_v5 as gate
from tools import compare_camera_speed_v5 as runner
from tools import evaluate_bare_generalization as fresh

def protocol_grid():
    return {'partitions': {
        'screen': {'track_ids': [1, 2, 3, 4], 'seeds': [101, 102, 103, 104],
                   'spot_check_cells': [[1, 101], [4, 104]]},
        'confirmation': {'track_ids': [1, 2, 3, 4], 'seeds': [201, 202, 203, 204],
                         'spot_check_cells': [[2, 201], [3, 204]]}}}

def rows_for(phase, ratio=0.90, changes=None):
    cells = fresh.expected_cells(protocol_grid(), phase)
    rows = []
    for track, seed, repeat in cells:
        for arm in ('control', 'candidate'):
            row = {'partition': phase, 'arm': arm, 'track_id': track, 'seed': seed,
                   'repeat': repeat, 'finished': True, 'progress': 1.0,
                   'lap_time_ms': 10000 if arm == 'control' else 10000 * ratio,
                   'retire_reason': None, 'collision_count': 0, 'damage': 0.0,
                   'steps': 100, 'offtrack_samples': 0, 'partial_offtrack_samples': 0,
                   'initialization_ms': 1.0, 'reset_ms': 1.0,
                   'action_latency_max_ms': 1.0, 'peak_worker_rss_mib': 200.0,
                   'action_trace_sha256': ('a' if arm == 'control' else 'b') * 64,
                   'error': None}
            if changes and (arm, track, seed) in changes:
                row.update(changes[arm, track, seed])
            rows.append(row)
    return rows, cells

def summary(phase, ratio=0.90, profile='strict', changes=None):
    rows, cells = rows_for(phase, ratio, changes)
    value = gate.compare_pairs(rows, cells, phase, profile)
    value.update({'protocol_sha256': '1' * 64, 'control_agent_sha256': '2' * 64,
                  'candidate_agent_sha256': '3' * 64, 'model_sha256': '4' * 64,
                  'runner_sha256': '5' * 64, 'decision_engine_sha256': '6' * 64,
                  'harness_sha256': '7' * 64, 'helper_sha256': {'a.py': '8' * 64},
                  'environment_sha256': {'b.py': '9' * 64}})
    return value

class SpeedGateTests(unittest.TestCase):
    def test_malformed_finished_progress_cannot_normalize_to_complete(self):
        rows, cells = rows_for('screen', changes={('candidate', 1, 102): {'progress': .94999}})
        value = gate.compare_pairs(rows, cells, 'screen', 'strict')
        self.assertEqual(value['decision'], 'REJECT')
        self.assertIn('official progress', ' '.join(value['reasons']))
    def test_valid_finished_progress_is_completion_aware_but_raw_is_preserved(self):
        changes = {('candidate', 1, 102): {'progress': .95}}
        rows, cells = rows_for('screen', changes=changes)
        value = gate.compare_pairs(rows, cells, 'screen', 'strict')
        self.assertEqual(value['decision'], 'RETAIN')
        self.assertEqual(value['candidate_mean_progress'], 1.0)
        self.assertLess(value['candidate_raw_mean_progress'], 1.0)
        self.assertEqual(value['paired_cells'][1]['candidate']['progress'], .95)

    def test_dnf_progress_is_not_promoted_to_completed(self):
        dnf = {'finished': False, 'progress': .95, 'lap_time_ms': None, 'retire_reason': 'off_track'}
        changes = {(arm, 1, 102): dnf for arm in ('control', 'candidate')}
        rows, cells = rows_for('screen', changes=changes)
        value = gate.compare_pairs(rows, cells, 'screen', 'strict')
        self.assertEqual(value['decision'], 'RETAIN')
        self.assertAlmostEqual(value['candidate_mean_progress'], (15 + .95) / 16)

    def test_combined_recomputes_forged_phase_retention_and_safety_totals(self):
        for mutation in ('pace', 'contacts', 'malformed'):
            summaries = {phase: summary(phase) for phase in gate.PHASES}
            candidate = summaries['screen']['paired_cells'][1]['candidate']
            if mutation == 'pace': candidate['lap_time_ms'] = 900000
            elif mutation == 'contacts': candidate['collision_count'] = 3
            else: del candidate['lap_time_ms']
            result = gate.combined_decision(summaries, 'strict')
            self.assertEqual(result['decision'], 'REJECT')

    def test_combined_rejects_geometry_reuse_between_phases(self):
        summaries = {phase: summary(phase) for phase in gate.PHASES}
        for pair in summaries['confirmation']['paired_cells']:
            pair['seed'] -= 100
        for pair in summaries['confirmation']['seed_finish_breakdown']:
            pair['seed'] -= 100
        self.assertEqual(gate.combined_decision(summaries)['decision'], 'REJECT')

    def test_strict_retains_equal_completion_and_ten_percent_faster_combined(self):
        summaries = {phase: summary(phase) for phase in gate.PHASES}
        self.assertTrue(all(row['decision'] == 'RETAIN' for row in summaries.values()))
        self.assertEqual(gate.combined_decision(summaries, 'strict')['decision'], 'RETAIN')

    def test_phase_slowdown_is_rejected_even_if_other_phase_offsets_it(self):
        summaries = {'screen': summary('screen', 1.00001),
                     'confirmation': summary('confirmation', 0.70)}
        self.assertEqual(summaries['screen']['decision'], 'REJECT')
        self.assertEqual(gate.combined_decision(summaries, 'strict')['decision'], 'REJECT')

    def test_strict_requires_meaningful_combined_speed_gain(self):
        summaries = {phase: summary(phase, 0.90001) for phase in gate.PHASES}
        result = gate.combined_decision(summaries, 'strict')
        self.assertEqual(result['decision'], 'REJECT')
        self.assertIn('time', ' '.join(result['reasons']))

    def test_fallback_accepts_no_slowdown_but_cannot_relax_safety(self):
        summaries = {phase: summary(phase, 1.00, 'fallback') for phase in gate.PHASES}
        self.assertEqual(gate.combined_decision(summaries, 'fallback')['decision'], 'RETAIN')
        rows, cells = rows_for('screen', 1.0, {('candidate', 1, 102): {'collision_count': 1}})
        self.assertEqual(gate.compare_pairs(rows, cells, 'screen', 'fallback')['decision'], 'REJECT')

    def test_each_phase_track_finish_floor_is_hard(self):
        dnf = {'finished': False, 'progress': 0.5, 'lap_time_ms': None, 'retire_reason': 'off_track'}
        changes = {('candidate', 1, 102): dnf, ('control', 2, 102): dnf}
        rows, cells = rows_for('screen', changes=changes)
        result = gate.compare_pairs(rows, cells, 'screen', 'fallback')
        self.assertEqual(result['decision'], 'REJECT')
        self.assertIn('track', ' '.join(result['reasons']))

    def test_fallback_allows_only_one_lost_finish_across_both_phases(self):
        dnf = {'finished': False, 'progress': 0.5, 'lap_time_ms': None, 'retire_reason': 'off_track'}
        summaries = {}
        for phase in gate.PHASES:
            first, second = protocol_grid()['partitions'][phase]['seeds'][1:3]
            summaries[phase] = summary(phase, 0.95, 'fallback',
                {('candidate', 1, first): dnf, ('control', 1, second): dnf})
            self.assertEqual(summaries[phase]['decision'], 'RETAIN')
        result = gate.combined_decision(summaries, 'fallback')
        self.assertEqual(result['decision'], 'REJECT')
        self.assertIn('lost', ' '.join(result['reasons']))

    def test_no_common_finish_cannot_certify_speed(self):
        dnf = {'finished': False, 'progress': 0.5, 'lap_time_ms': None, 'retire_reason': 'off_track'}
        changes = {(arm, track, seed): dnf for arm in ('control', 'candidate')
                   for track in (1, 2, 3, 4) for seed in (101, 102, 103, 104)}
        rows, cells = rows_for('screen', changes=changes)
        value = gate.compare_pairs(rows, cells, 'screen', 'fallback')
        self.assertEqual(value['decision'], 'REJECT')
        self.assertIn('shared', ' '.join(value['reasons']))

    def test_repeat_action_mismatch_and_boolean_coordinates_reject(self):
        for mutation in ({'action_trace_sha256': 'f' * 64}, {'repeat': True}):
            rows, cells = rows_for('screen')
            row = next(row for row in rows if row['arm'] == 'candidate' and row['repeat'] == 1)
            row.update(mutation)
            self.assertEqual(gate.compare_pairs(rows, cells, 'screen', 'strict')['decision'], 'REJECT')

class SourceAndProtocolTests(unittest.TestCase):
    def test_global_slots_are_profile_independent_with_total_cap_and_distinct_sources(self):
        self.assertNotEqual(runner.study_paths('fallback', 1)['protocol'], runner.study_paths('fallback', 2)['protocol'])
        self.assertNotEqual(runner.derived_seed_partitions('fallback', '0' * 32, 1),
                            runner.derived_seed_partitions('fallback', '0' * 32, 2))
        first = {'study_slot': 1, 'evaluation_profile': 'fallback', 'candidate_agent_sha256': '1' * 64}
        with patch.object(runner, '_registered_bindings', return_value=[first]):
            with self.assertRaisesRegex(ValueError, 'occupied'):
                runner.require_available_slot('strict', 1, '2' * 64)
            with self.assertRaisesRegex(ValueError, 'distinct'):
                runner.require_available_slot('fallback', 2, '1' * 64)
            runner.require_available_slot('fallback', 2, '2' * 64)
        second = {'study_slot': 2, 'evaluation_profile': 'fallback', 'candidate_agent_sha256': '2' * 64}
        with patch.object(runner, '_registered_bindings', return_value=[first, second]):
            with self.assertRaisesRegex(ValueError, 'two|budget'):
                runner.require_available_slot('strict', 1, '3' * 64)
        for slot in (0, 3, True):
            with self.assertRaises(ValueError): runner.study_paths('strict', slot)

    def test_shared_binding_lock_prevents_simultaneous_same_slot_bindings(self):
        entered, release = threading.Event(), threading.Event()
        result = []
        def held(*args, **kwargs):
            entered.set()
            release.wait(5)
            return {'fixture': 'first binding'}
        def first():
            try: result.append(runner.bind_source('fallback', 'a' * 40, 'b' * 64, [], 1))
            except BaseException as error: result.append(error)
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)), patch.object(runner, '_bind_source_under_lock', side_effect=held) as binder:
            thread = threading.Thread(target=first)
            thread.start()
            self.assertTrue(entered.wait(5))
            try:
                with self.assertRaisesRegex(ValueError, 'owns'):
                    runner.bind_source('strict', 'c' * 40, 'd' * 64, [], 1)
            finally:
                release.set()
                thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result, [{'fixture': 'first binding'}])
            self.assertEqual(binder.call_count, 1)

    def test_binding_is_one_time_and_each_source_grid_policy_pin_is_checked(self):
        control = runner.committed_bytes('agent.py', runner.CONTROL_COMMIT)
        candidate = control + b'\nclass _SpeedOptimizedController: pass\n'
        requirements = runner.committed_bytes('requirements.txt')
        candidate_commit = 'e' * 40
        candidate_blob = hashlib.sha256(candidate).hexdigest()
        candidates = {candidate_commit: candidate}
        real_template = runner.TEMPLATE_PATH.read_bytes()
        def git_bytes(name, commit='HEAD'):
            if name == 'requirements.txt': return requirements
            if name == 'agent.py': return control if commit == runner.CONTROL_COMMIT else candidates[commit]
            path = runner.ROOT / name
            if path.is_file(): return path.read_bytes()
            raise AssertionError(name)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            (root / 'experiments').mkdir()
            template_path = root / 'experiments/camera-speed-v5.template.json'
            template_path.write_bytes(real_template)
            (root / 'model.pt').write_bytes(b'model fixture')
            for name in runner.HELPERS: (root / name).write_bytes(name.encode())
            for target, value in (("ROOT", root), ("TEMPLATE_PATH", template_path)):
                stack.enter_context(patch.object(runner, target, value))
            stack.enter_context(patch.object(runner, 'committed_bytes', side_effect=git_bytes))
            stack.enter_context(patch.object(runner, '_require_committed_runtime'))
            stack.enter_context(patch.object(runner, 'require_runtime_versions'))
            stack.enter_context(patch.object(runner, '_environment_hashes', return_value={}))
            salt = stack.enter_context(patch.object(runner.secrets, 'token_hex', return_value='f' * 32))
            protocol = runner.bind_source('strict', candidate_commit, candidate_blob, [])
            runner.validate_protocol(protocol, set())
            changed = copy.deepcopy(protocol)
            changed['study_slot'] = 2
            changed['name'] = 'camera-speed-v5-strict-study-2'
            changed['source_snapshots'] = runner._source_paths('strict', 2)
            with self.assertRaisesRegex(ValueError, 'geometry|grid'):
                runner.validate_protocol(changed, set())
            paths = runner.study_paths('strict')
            self.assertEqual(paths['control'].read_bytes(), control)
            self.assertEqual(paths['candidate'].read_bytes(), runner.selected_source(candidate))
            for field, mutation in (
                    ('decision_thresholds', {'minimum_net_finish_gain': -99}),
                    ('source_construction', {}), ('candidate_agent_sha256', '0' * 64),
                    ('seed_salt_hex', '0' * 32), ('training', 0), ('schema_version', True)):
                changed = copy.deepcopy(protocol)
                changed[field] = mutation
                with self.assertRaises(ValueError): runner.validate_protocol(changed, set())
            changed = copy.deepcopy(protocol)
            changed['partitions']['screen']['track_ids'][0] = True
            with self.assertRaisesRegex(ValueError, 'integer'):
                runner.validate_protocol(changed, set())
            with self.assertRaisesRegex(ValueError, 'already exists|occupied'):
                runner.bind_source('strict', candidate_commit, candidate_blob, [])
            salt.assert_called_once()
            with self.assertRaisesRegex(ValueError, 'distinct'):
                runner.bind_source('strict', candidate_commit, candidate_blob, [], 2)
            second_commit, second_source = 'f' * 40, candidate + b'# distinct production revision\n'
            candidates[second_commit] = second_source
            second = runner.bind_source('strict', second_commit, hashlib.sha256(second_source).hexdigest(), [], 2)
            first_seeds = {seed for part in protocol['partitions'].values() for seed in part['seeds']}
            second_seeds = {seed for part in second['partitions'].values() for seed in part['seeds']}
            self.assertTrue(first_seeds.isdisjoint(second_seeds))
            self.assertEqual(len(runner._registered_bindings()), 2)
            self.assertEqual(second['prior_bound_studies'][0]['candidate_agent_sha256'], protocol['candidate_agent_sha256'])
            with self.assertRaisesRegex(ValueError, 'two-study'):
                runner.bind_source('strict', candidate_commit, candidate_blob, [], 1)
            prior_path = paths['protocol']
            prior_path.write_bytes(prior_path.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'previous.*SHA256'):
                runner.validate_bound_history(second)

    def test_fallback_proofs_require_distinct_sources_exact_committed_bytes_and_performance(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            folder = Path(directory) / 'experiments'
            folder.mkdir()
            records = []
            for index in (1, 2):
                path = folder / f'failure-{index}.json'
                source = folder / f'prototype-{index}.py'
                source.write_bytes(f'class ActualExploratoryRoute{index}: pass\n'.encode())
                source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
                pairs = copy.deepcopy(summary('screen', .95)['paired_cells'][:3])
                for pair, (track, seed) in zip(pairs, runner.EVALUATION_POLICY['development_maps']):
                    pair.update({'track_id': track, 'seed': seed})
                report = folder / f'report-{index}.json'
                report.write_bytes(fresh._canonical({'control_agent_sha256': runner.CONTROL_BLOB_SHA256,
                    'candidate_agent_sha256': source_sha, 'canonical_cells': 3, 'paired_cells': pairs}) + b'\n')
                value = {'study_family': runner.NAME, 'evaluation_profile': 'strict',
                         'decision': 'REJECT', 'failure_kind': 'performance',
                         'completion_status': 'COMPLETE', 'reasons': ['time improvement insufficient'],
                         'evaluation_scope': 'consumed-development',
                         'control_agent_sha256': runner.CONTROL_BLOB_SHA256, 'candidate_agent_sha256': source_sha,
                         'candidate_source_path': source.relative_to(Path(directory)).as_posix(),
                         'comparison_report_path': report.relative_to(Path(directory)).as_posix(),
                         'comparison_report_sha256': hashlib.sha256(report.read_bytes()).hexdigest()}
                path.write_bytes(fresh._canonical(value) + b'\n')
                records.append(path.relative_to(Path(directory)).as_posix())
            with patch.object(runner, 'committed_bytes', side_effect=lambda name: (Path(directory) / name).read_bytes()):
                self.assertEqual(len(runner.failure_evidence('fallback', records)), 2)
                path = folder / 'failure-2.json'
                for pair in pairs: pair['candidate']['lap_time_ms'] = 8000
                report.write_bytes(fresh._canonical({'control_agent_sha256': runner.CONTROL_BLOB_SHA256,
                    'candidate_agent_sha256': source_sha, 'canonical_cells': 3, 'paired_cells': pairs}) + b'\n')
                value['comparison_report_sha256'] = hashlib.sha256(report.read_bytes()).hexdigest()
                path.write_bytes(fresh._canonical(value) + b'\n')
                with self.assertRaisesRegex(ValueError, 'recomputable'):
                    runner.failure_evidence('fallback', records)
                value['failure_kind'] = 'operational'
                path.write_bytes(fresh._canonical(value) + b'\n')
                with self.assertRaisesRegex(ValueError, 'performance'):
                    runner.failure_evidence('fallback', records)
            with patch.object(runner, 'committed_bytes', return_value=b'changed'):
                with self.assertRaisesRegex(ValueError, 'committed'):
                    runner.failure_evidence('fallback', records)

    def test_operational_failure_is_persisted_and_cannot_be_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {**protocol_grid(), 'evaluation_profile': 'strict'}
            identity = {'evaluation_profile': 'strict', 'protocol_sha256': '1' * 64,
                        'source_sha256': {'control': '2' * 64, 'candidate': '3' * 64},
                        'model_sha256': '4' * 64, 'harness_sha256': '5' * 64,
                        'runner_sha256': '6' * 64, 'decision_engine_sha256': '7' * 64}
            with patch.object(runner, 'check_frozen_inputs'), patch.object(runner, 'cold_episode', return_value={'error': 'fixture worker failure'}) as cold:
                with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                    runner.run_partition(root, identity, protocol, {}, 'screen', workers=1)
                self.assertEqual(cold.call_count, 1)
                cold.reset_mock()
                with self.assertRaisesRegex(RuntimeError, 'recorded operational'):
                    runner.run_partition(root, identity, protocol, {}, 'screen', workers=1)
                cold.assert_not_called()

    def test_confirmation_cannot_read_receipts_without_retained_screen_seal(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(fresh, 'load_cell') as reader:
            with self.assertRaisesRegex(ValueError, 'seal'):
                runner.report_phase(Path(directory), {}, protocol_grid(), 'confirmation')
            reader.assert_not_called()

    def test_screen_seal_is_recomputed_before_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {**protocol_grid(), 'evaluation_profile': 'strict'}
            identity = {'evaluation_profile': 'strict', 'protocol_sha256': '1' * 64,
                        'source_sha256': {'control': '2' * 64, 'candidate': '3' * 64},
                        'model_sha256': '4' * 64, 'harness_sha256': '5' * 64,
                        'runner_sha256': '6' * 64, 'decision_engine_sha256': '7' * 64}
            rows, _ = rows_for('screen')
            for row in rows: fresh.record_cell(root, identity, row)
            runner.freeze_finalist(root, identity, protocol)
            runner.require_phase(root, identity, protocol, 'confirmation')
            path = fresh.cell_path(root, 'screen', 'candidate', 1, 102, 0)
            envelope = fresh._read_json(path)
            envelope['row']['lap_time_ms'] = 9001
            payload = {'identity': identity, 'row': envelope['row']}
            envelope['digest'] = hashlib.sha256(fresh._canonical(payload)).hexdigest()
            fresh._atomic_json(path, envelope)
            with self.assertRaisesRegex(ValueError, 'seal'):
                runner.require_phase(root, identity, protocol, 'confirmation')

    def test_template_has_no_geometry_and_declares_finite_profiles(self):
        template = json.loads(runner.TEMPLATE_PATH.read_text())
        runner.validate_template(template)
        self.assertNotIn('seed_salt_hex', template)
        self.assertNotIn('partitions', template)
        changed = copy.deepcopy(template)
        changed['evaluation_policy']['maximum_fresh_studies'] = 99
        with self.assertRaises(ValueError): runner.validate_template(changed)

    def test_source_control_is_exact_promoted_v4_blob(self):
        data = runner.committed_bytes('agent.py', runner.CONTROL_COMMIT)
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         'd77b27e4489564f98df985b138d6c8ba2d74ff9ed275aeb62cd61477f2b46c18')

    def test_two_profiles_have_disjoint_seed_derivations_and_protect_all_old_v4_seeds(self):
        strict = runner.derived_seed_partitions('strict', '0' * 32)
        fallback = runner.derived_seed_partitions('fallback', '0' * 32)
        flat = [seed for groups in (strict, fallback) for values in groups.values() for seed in values]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(len(flat), 16)
        historical = runner.historical_geometry_seeds(runner.ROOT / 'experiments',
                                                      runner.study_paths('strict')['protocol'])
        v4 = json.loads((runner.ROOT / 'experiments/camera-policy-competition-v4.json').read_text())
        self.assertTrue({seed for phase in v4['partitions'].values() for seed in phase['seeds']} <= historical)

    def test_fallback_requires_two_distinct_committed_performance_failure_proofs(self):
        with self.assertRaisesRegex(ValueError, 'two|performance'):
            runner.failure_evidence('fallback', [])

    def test_binding_missing_candidate_pins_cannot_generate_seed_salt(self):
        with patch.object(runner.secrets, 'token_hex') as salt:
            with self.assertRaisesRegex(ValueError, 'pin|commit|SHA256'):
                runner.bind_source('strict', '', '', [])
            salt.assert_not_called()

    def test_unbound_preflight_cannot_open_worker(self):
        with patch.object(runner, 'cold_episode') as cold:
            with self.assertRaisesRegex(ValueError, 'unbound'):
                runner.load_study('strict')
            cold.assert_not_called()

if __name__ == '__main__': unittest.main()
