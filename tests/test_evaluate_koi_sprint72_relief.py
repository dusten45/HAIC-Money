"""Byte inspection and fake children only: no policy or environment execution."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch
import zipfile

from scripts import evaluate_koi_sprint72_relief as evaluate


def measurements() -> dict[str, Any]:
    group = dict(path='/sys/fs/cgroup', memory_max='1000', memory_current=100,
        raw_headroom_bytes=900, memory_events='oom 0\noom_kill 0\n',
        memory_pressure='some avg10=0.00\n', memory_stat='inactive_file 50\nfile_dirty 2\nfile_writeback 3\n')
    return dict(host_mem_available_bytes=1000, visible_cgroup_ancestors=[group],
        disk=[dict(device=1, free_bytes=10000), dict(device=2, free_bytes=10000)])


def forecast():
    return dict(peak_child_bytes=100, memory_reserve_bytes=10, competing_ram_growth_bytes=10,
        per_episode_disk_bytes=100, disk_reserve_bytes=10, competing_disk_growth_bytes=10, temp_bytes=10)


def protocol() -> dict[str, Any]:
    return dict(schedule=evaluate.scheduled_slots(), python='unused', child_timeout_s=17,
        source_inventory={evaluate.OPERATOR: {'sha256': 'pin'}},
        models={a: dict(directory='models/' + a, zip_sha256='zip-' + a) for a in evaluate.ARMS},
        resource_plan=dict(forecast=forecast(), freeze_measurement=measurements()))


def preflight_receipt(frozen) -> dict[str, Any]:
    return dict(protocol_sha256='pin', environment_resets=0, arms=[
        dict(mode=a, import_only=True, environment_resets=0, executed_action_observations=0,
            current_resident_rss_bytes=100, driver_state_keys=['base', 'brake_history', 'last'],
            model_directory=frozen['models'][a]['directory'], model_zip_sha256=frozen['models'][a]['zip_sha256'])
        for a in evaluate.ARMS])


class CandidatePackageTest(unittest.TestCase):
    def test_candidate_is_exact_champion_plus_one_source_patch_and_helper(self):
        paths = [evaluate.BUNDLE / 'submission.zip', evaluate.BUNDLE / 'manifest.json',
                 evaluate.BUNDLE / 'source' / evaluate.FAR_MEMBER,
                 evaluate.ROOT / 'haic/algorithms/koi/collision_shield.py', evaluate.ROOT / 'agent.py']
        before = {str(p): evaluate.sha(p) for p in paths}
        with patch.object(evaluate.importlib, 'import_module', side_effect=AssertionError('no policy imports')):
            original, changed = evaluate.champion_members(), evaluate.candidate_members()
        self.assertEqual(len(original), 11)
        self.assertEqual(len(changed), 12)
        self.assertEqual(set(changed) - set(original), {evaluate.HELPER_MEMBER})
        self.assertEqual([n for n in original if original[n] != changed[n]], [evaluate.FAR_MEMBER])
        self.assertEqual(changed[evaluate.HELPER_MEMBER], (evaluate.ROOT / evaluate.HELPER).read_bytes())
        far = changed[evaluate.FAR_MEMBER]
        self.assertEqual(far.count(evaluate.HELPER_IMPORT), 1)
        self.assertEqual(far.count(evaluate.HELPER_CALL), 1)
        self.assertIn(b'        side = clearance = cap = None\n' + evaluate.HELPER_CALL +
                      b'        if self.steps > 10:\n', far)
        self.assertEqual(far.replace(evaluate.HELPER_IMPORT, b'').replace(evaluate.HELPER_CALL, b''),
                         original[evaluate.FAR_MEMBER])
        self.assertEqual({str(p): evaluate.sha(p) for p in paths}, before)

    def test_nonchampion_zip_and_changed_far_source_rejected(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            wrong = Path(temporary) / 'wrong.zip'
            with zipfile.ZipFile(wrong, 'x') as archive:
                archive.writestr('agent.py', b'not the champion')
            with self.assertRaisesRegex(ValueError, 'exact frozen champion'):
                evaluate.candidate_members(baseline_zip=wrong)
        source = evaluate.champion_members()[evaluate.FAR_MEMBER]
        with self.assertRaisesRegex(ValueError, 'source hash'):
            evaluate.patch_far_source(source + b'\n')

    def test_missing_or_ambiguous_patch_point_fails_even_with_matching_pin(self):
        original = evaluate.champion_members()[evaluate.FAR_MEMBER]
        for changed in (original.replace(evaluate.PATCH_POINT, b''), original + evaluate.PATCH_POINT,
                        original + evaluate.IMPORT_POINT, original.replace(evaluate.IMPORT_POINT, b'')):
            with self.subTest(source=hashlib.sha256(changed).hexdigest()), \
                    patch.object(evaluate, 'FAR_SHA', hashlib.sha256(changed).hexdigest()), \
                    self.assertRaisesRegex(ValueError, 'unique pre-arrival'):
                evaluate.patch_far_source(changed)

    def test_bad_helper_syntax_rejected_without_executing_helper(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            helper = Path(temporary) / 'helper.py'
            helper.write_text('def broken(:\n')
            with self.assertRaises(SyntaxError):
                evaluate.candidate_members(helper=helper)
            helper.write_text('raise AssertionError("must never execute while packaging")\n')
            self.assertEqual(evaluate.candidate_members(helper=helper)[evaluate.HELPER_MEMBER], helper.read_bytes())


class OperatorContractTest(unittest.TestCase):
    def test_all_eight_consumed_cells_five_geometries_and_reverse_preserved(self):
        self.assertEqual(set(evaluate.CELLS), {(1, 3184000002), (1, 3184000013), (1, 3184000015),
            (2, 3184000001), (2, 3184000006), (2, 3184000015), (3, 3184000002), (3, 3184000015)})
        self.assertEqual(evaluate.ARMS, ('frozen_shield', 'sprint72_relief'))
        self.assertEqual(evaluate.CELLS, evaluate.magnitude.nominal.CELLS)
        rows = evaluate.scheduled_slots()
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({r['file'] for r in rows}), 16)
        self.assertEqual(len({r['seed'] for r in rows}), 5)
        self.assertEqual(evaluate.REVERSE_CASE['historical_decisions'], [194, 195, 447, 448])
        self.assertFalse(evaluate.REVERSE_CASE['exclusion_allowed'])
        self.assertEqual(evaluate.GATE['max_kept_lap_increase_ms'], 20)
        self.assertEqual(evaluate.GATE['min_improving_geometry_seeds'], 2)
        for name in ('no_lost_finish', 'offroad_nonincrease', 'no_new_hit', 'cap_undershoot_reduced',
                     'repeated_brake_gas_reduced', 'baseline_window_coverage', 'common_lap_mean_negative'):
            self.assertTrue(evaluate.GATE[name])
        self.assertEqual(len(evaluate.HELPERS), len(set(evaluate.HELPERS)))
        self.assertTrue(set(evaluate.magnitude.HELPERS) <= set(evaluate.HELPERS))
        self.assertIn(evaluate.ANALYZER, evaluate.HELPERS)
        self.assertIn(evaluate.HELPER, evaluate.HELPERS)

    def test_consumed_evidence_is_passive_complete_and_includes_partial_receipts(self):
        with patch.object(evaluate.importlib, 'import_module', side_effect=AssertionError('no policy/environment import')):
            evidence = evaluate.consumed_evidence()
        self.assertFalse(evidence['fresh'])
        self.assertEqual(evidence['cells'], [list(c) for c in evaluate.CELLS])
        self.assertEqual(len(evidence['baseline_outcomes']), 8)
        self.assertEqual(evidence['historical']['episodes'], 16)
        self.assertGreater(evidence['historical']['peak_serialization_rss_bytes'], 0)
        self.assertGreater(evidence['historical']['max_child_wall_s'], 40)
        self.assertIsNotNone(evidence['prior_partial_operator_error'])
        self.assertEqual(evidence['exclusion_audit']['observations_read'], 0)
        self.assertTrue(any(p.endswith('koi-steering-generalization-v1/reset-ledger.jsonl')
                            for p in evidence['source_sha256']))
        self.assertTrue(any(p.endswith('2-3184000006-r0-frozen_shield.decisions.jsonl')
                            for p in evidence['source_sha256']))
        self.assertTrue(all(evaluate.sha(p) == h for p, h in evidence['source_sha256'].items()))
        plan = evaluate.resource_forecast(evidence)
        self.assertIn('estimated_sixteen_episode_wall_s', plan)
        self.assertNotIn('estimated_eight_episode_wall_s', plan)
        self.assertGreaterEqual(plan['child_timeout_s'], 4 * evidence['historical']['max_child_wall_s'])

    def test_freeze_requires_callable_analyzer_before_creating_output(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            output = Path(temporary) / 'not-created'
            with patch.object(evaluate.importlib, 'import_module', return_value=SimpleNamespace(METRIC_DEFINITIONS={})), \
                    self.assertRaisesRegex(ValueError, 'callable analyzer'):
                evaluate.freeze_protocol(output)
            self.assertFalse(output.exists())

    def test_modified_contract_rejected_before_importing_policy(self):
        with patch.object(evaluate, 'sha', return_value='pin'), \
                patch.object(evaluate, 'fixed_contract', return_value={'schema': evaluate.SCHEMA}), \
                patch.object(evaluate, 'read_json', return_value={'schema': 'different'}), \
                patch.object(evaluate, 'load_model', side_effect=AssertionError('no import')):
            with self.assertRaisesRegex(ValueError, 'contract'):
                evaluate.validate_frozen('/tmp/kilo/not-created', 'pin')

    def test_original_and_frozen_transitive_source_drift_rejected(self):
        output = Path('/tmp/kilo/not-created')
        frozen = dict(environment_sha256={str(i): 'pin' for i in range(142)}, source_inventory={
            n: dict(original=str(output / 'original' / n), sha256='pin') for n in evaluate.HELPERS})
        name = 'scripts/analyze_koi_adaptive_ab.py'
        for changed in (output / 'original' / name, output / 'source' / name):
            with self.subTest(path=changed), patch.object(evaluate, 'fixed_contract', return_value={}), \
                    patch.object(evaluate, 'read_json', return_value=frozen), \
                    patch.object(evaluate, 'sha', side_effect=lambda p: 'changed' if Path(p) == changed else 'pin'):
                with self.assertRaisesRegex(ValueError, 'helper source'):
                    evaluate.validate_frozen(output, 'pin')

    def test_natural_failure_valid_but_caps_nonfinite_and_bad_telemetry_rejected(self):
        episode = dict(error=None, invalid_actions=0, completed=False, retire_reason='off_track',
            damage=.2, collisions=1, decision_trace=[{}], steps=1,
            damage_telemetry_valid=True, collision_telemetry_valid=True)
        self.assertTrue(evaluate.valid_episode(episode))
        for change in ({'retire_reason': 'max_steps'}, {'damage': float('inf')}, {'collisions': True},
                       {'steps': True}, {'damage_telemetry_valid': False}, {'completed': True, 'lapTimeMs': float('nan')}):
            with self.subTest(change=change):
                self.assertFalse(evaluate.valid_episode(dict(episode, **change)))

    def test_final_nominal_tapped_once_and_shield_may_not_revert_relief_pedals(self):
        seen = []
        driver = SimpleNamespace(brake_history=[.04], act=lambda obs: seen.append(obs) or [.1, 0., .04])

        class Shield:
            def __init__(self):
                self.driver, self.last_shield = driver, {}

            def act(self, obs):
                action = self.driver.act(obs).copy()
                self.last_shield = dict(baseline_action=action.copy())
                action[0] = -.3
                return action

        safety = Shield()
        model = SimpleNamespace(driver=safety, act=safety.act)
        model, tap = evaluate.compose(model, evaluate.ARMS[1])
        obs = object()
        self.assertEqual(evaluate.measured_act(model, tap, obs), [-.3, 0., .04])
        self.assertEqual(seen, [obs])
        self.assertEqual(tap.action, [.1, 0., .04])
        self.assertFalse(hasattr(tap, 'driver'))
        self.assertEqual(tap.calls, 1)

    def test_missing_duplicate_calls_pedal_changes_and_fictional_history_fail(self):
        for count in (0, 2):
            driver = SimpleNamespace(brake_history=[.04], act=lambda obs: [0., 0., .04])
            tap = evaluate.ActionProbe(driver)

            def repeated_act(obs):
                for _ in range(count):
                    tap.act(obs)
                return [0., 0., .04]

            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'exactly once'):
                evaluate.measured_act(SimpleNamespace(act=repeated_act), tap, None)
        for field in ('pedals', 'history', 'input', 'new_state'):
            driver = SimpleNamespace(brake_history=[.15 if field == 'history' else .04],
                                     act=lambda obs: [0., 0., .04])
            tap = evaluate.ActionProbe(driver)
            shield = SimpleNamespace(last_shield={})

            def act(obs):
                action = tap.act(obs)
                shield.last_shield['baseline_action'] = [1., 0., .04] if field == 'input' else action.copy()
                if field == 'new_state':
                    driver.hysteresis = 1
                return [0., 0., .15] if field == 'pedals' else action

            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate.measured_act(SimpleNamespace(act=act, driver=shield), tap, None)

    def test_controller_snapshot_is_detached_and_excludes_diagnostic_differences(self):
        driver = SimpleNamespace(last={'sprint72_applied': True}, brake_history=[.04], impact_left=0,
            track=SimpleNamespace(tolist=lambda: [30., 42.]), base=SimpleNamespace(_last_obstacle=None))
        state = evaluate.controller_state(driver)
        self.assertNotIn('last', state)
        self.assertEqual(state['track'], [30., 42.])
        self.assertEqual(state['brake_history'], [.04])
        driver.brake_history[0] = .15
        self.assertEqual(state['brake_history'], [.04])

    def test_cached_agent_or_package_cannot_contaminate_other_arm(self):
        for name in ('agent', 'haic_agent', 'haic_agent.far_hazard_runtime'):
            with self.subTest(name=name), patch.dict(sys.modules, {name: SimpleNamespace()}), \
                    patch.object(evaluate.importlib, 'import_module') as imported, \
                    self.assertRaisesRegex(ValueError, 'fresh isolated'):
                evaluate.load_model(Path('/tmp/kilo/not-created'), evaluate.ARMS[0], protocol())
            imported.assert_not_called()


class ReceiptsTest(unittest.TestCase):
    def test_resource_facilities_reused_and_reverse_sized_forecast(self):
        self.assertIs(evaluate.resource_measurements, evaluate.magnitude.resource_measurements)
        self.assertIs(evaluate.check_resources, evaluate.magnitude.check_resources)
        initial = measurements()
        self.assertTrue(evaluate.check_resources(initial, forecast(), 16)['allowed'])
        initial['visible_cgroup_ancestors'].insert(0, dict(initial['visible_cgroup_ancestors'][0],
            path='/sys/fs/cgroup/parent', raw_headroom_bytes=50))
        with self.assertRaisesRegex(ValueError, 'ancestor'):
            evaluate.check_resources(initial, forecast(), 16)
        value = measurements()
        value['disk'][1]['free_bytes'] = 0
        with self.assertRaisesRegex(ValueError, 'separate'):
            evaluate.check_resources(value, forecast(), 16)
        value['disk'][1]['device'] = 1
        value['disk'][0]['free_bytes'] = 1625
        with self.assertRaisesRegex(ValueError, 'shared'):
            evaluate.check_resources(value, forecast(), 16)

    def test_preflight_requires_both_isolated_models_same_state_and_rss(self):
        frozen = protocol()
        receipt = preflight_receipt(frozen)
        evaluate.validate_preflight(receipt, 'pin', frozen)
        for change in ({'model_directory': 'models/frozen_shield'}, {'model_zip_sha256': 'other'},
                       {'driver_state_keys': ['new_timer']}, {'current_resident_rss_bytes': 101},
                       {'environment_resets': 1}, {'executed_action_observations': 1}):
            changed = copy.deepcopy(receipt)
            changed['arms'][1].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'preflight/model/state/RSS'):
                evaluate.validate_preflight(changed, 'pin', frozen)

    def test_preflight_two_import_only_children_with_distinct_workdirs(self):
        frozen = protocol()
        receipts = {r['mode']: r for r in preflight_receipt(frozen)['arms']}

        def child(command, **kwargs):
            index = int(command[-3])
            mode = frozen['schedule'][index]['mode']
            self.assertEqual(Path(kwargs['cwd']).name, mode)
            return SimpleNamespace(returncode=0, stdout=json.dumps(receipts[mode]))

        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary, \
                patch.object(evaluate, 'validate_frozen', return_value=frozen), \
                patch.object(evaluate.subprocess, 'run', side_effect=child) as launch:
            result = evaluate.preflight(temporary, 'pin')
            self.assertTrue((Path(temporary) / 'preflight.json').exists())
            with self.assertRaisesRegex(ValueError, 'never overwrite'):
                evaluate.preflight(temporary, 'pin')
        self.assertEqual(result['environment_resets'], 0)
        self.assertEqual(launch.call_count, 2)
        self.assertTrue(all(c.args[0][1:3] == ['-I', '-B'] and c.args[0][-1] == 'True'
                            for c in launch.call_args_list))

    def test_admission_block_keeps_measurement_without_launch_or_reset(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            output, frozen = Path(temporary), protocol()
            evaluate.save(output / 'preflight.json', preflight_receipt(frozen))
            measured = measurements()
            measured['host_mem_available_bytes'] = 0
            with patch.object(evaluate, 'validate_frozen', return_value=frozen), \
                    patch.object(evaluate, 'sha', return_value='pin'), \
                    patch.object(evaluate, 'resource_measurements', return_value=measured), \
                    patch.object(evaluate.subprocess, 'run') as child:
                with self.assertRaisesRegex(ValueError, 'RAM forecast'):
                    evaluate.run(output, 'pin')
            child.assert_not_called()
            report = evaluate.read_json(output / 'episode-report.json')
            self.assertEqual(len(report['rows']), 16)
            self.assertTrue(all(r['status'] == 'unrun' for r in report['rows']))
            receipt = json.loads((output / 'resource-receipts.jsonl').read_text())
            self.assertIn('RAM forecast', receipt['error'])
            self.assertFalse((output / 'reset-ledger.jsonl').exists())

    def test_timeout_preserves_partial_receipts_and_cannot_retry(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            output, frozen = Path(temporary), protocol()
            evaluate.save(output / 'preflight.json', preflight_receipt(frozen))

            def timeout(command, **kwargs):
                (output / frozen['schedule'][0]['file']).with_suffix('.raw.jsonl').write_text('{"partial":true}\n')
                raise subprocess.TimeoutExpired(command, 17, output=b'retained')

            with patch.object(evaluate, 'validate_frozen', return_value=frozen), \
                    patch.object(evaluate, 'sha', return_value='pin'), \
                    patch.object(evaluate, 'resource_measurements', return_value=measurements()), \
                    patch.object(evaluate.subprocess, 'run', side_effect=timeout) as child:
                with self.assertRaisesRegex(RuntimeError, 'child_timeout'):
                    evaluate.run(output, 'pin')
                with self.assertRaisesRegex(ValueError, 'never restart'):
                    evaluate.run(output, 'pin')
            report = evaluate.read_json(output / 'episode-report.json')
            self.assertEqual(child.call_count, 1)
            self.assertEqual([r['status'] for r in report['rows']], ['operator_error'] + ['unrun'] * 15)
            self.assertEqual(len(report['rows'][0]['partial_artifacts']), 4)
            self.assertEqual(len(report['operator_artifacts']), 3)
            self.assertFalse((output / 'reset-ledger.jsonl').exists())

    def test_serial_schedule_completes_all_fake_slots_without_analyzing(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            output, frozen = Path(temporary), protocol()
            evaluate.save(output / 'preflight.json', preflight_receipt(frozen))
            seen = []

            def child(command, **kwargs):
                index = int(command[-3])
                slot = frozen['schedule'][index]
                self.assertEqual(index, len(seen))
                self.assertEqual(Path(kwargs['cwd']).name, slot['mode'])
                seen.append((slot['track_id'], slot['seed'], slot['mode']))
                evaluate.save(output / slot['file'], dict(error=None, invalid_actions=0,
                    completed=False, retire_reason='off_track', damage=0., collisions=0,
                    decision_trace=[{}], steps=1, damage_telemetry_valid=True, collision_telemetry_valid=True))
                return SimpleNamespace(returncode=0, stdout='', stderr='')

            with patch.object(evaluate, 'validate_frozen', return_value=frozen), \
                    patch.object(evaluate, 'sha', return_value='pin'), \
                    patch.object(evaluate, 'resource_measurements', return_value=measurements()), \
                    patch.object(evaluate.subprocess, 'run', side_effect=child), \
                    patch('builtins.print'), \
                    patch.object(evaluate.importlib, 'import_module', side_effect=AssertionError('no auto-analysis')):
                evaluate.run(output, 'pin')
            report = evaluate.read_json(output / 'episode-report.json')
            self.assertEqual(set(seen), {(t, s, a) for t, s in evaluate.CELLS for a in evaluate.ARMS})
            self.assertEqual(len(seen), 16)
            self.assertIsNone(report['operator_error'])
            self.assertTrue(all(r['status'] == 'completed' for r in report['rows']))
            self.assertEqual(len((output / 'resource-receipts.jsonl').read_text().splitlines()), 16)
            self.assertFalse((output / 'reset-ledger.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
