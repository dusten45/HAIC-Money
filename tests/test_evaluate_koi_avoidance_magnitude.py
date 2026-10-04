"""Passive receipts and fake children only; no environment or policy execution."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

from scripts import evaluate_koi_avoidance_magnitude as evaluate


def measurements() -> dict[str, Any]:
    group = dict(path='/sys/fs/cgroup', memory_max='1000', memory_current=100,
                 raw_headroom_bytes=900, memory_events='oom 0\noom_kill 0\n',
                 memory_pressure='some avg10=0.00\n', memory_stat='inactive_file 50\nfile_dirty 2\nfile_writeback 3\n')
    return dict(host_mem_available_bytes=1000, visible_cgroup_ancestors=[group],
                disk=[dict(device=1, free_bytes=10000), dict(device=2, free_bytes=10000)])


def forecast() -> dict[str, Any]:
    return dict(peak_child_bytes=100, memory_reserve_bytes=10, competing_ram_growth_bytes=10,
                per_episode_disk_bytes=100, disk_reserve_bytes=10, competing_disk_growth_bytes=10,
                temp_bytes=10)


class OperatorTest(unittest.TestCase):
    def test_exact_four_consumed_cells_two_geometries_and_gates(self):
        self.assertEqual(evaluate.CELLS, ((1, 3184000013), (1, 3184000015), (2, 3184000015), (3, 3184000015)))
        rows = evaluate.scheduled_slots()
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({r['file'] for r in rows}), 8)
        self.assertEqual({r['mode'] for r in rows}, set(evaluate.ARMS))
        self.assertEqual(set(evaluate.SUBGROUPS.values()), {'ordinary_controls'})
        self.assertEqual(evaluate.GATE, evaluate.nominal.GATE)
        self.assertTrue(set(evaluate.nominal.HELPERS) <= set(evaluate.HELPERS))
        self.assertIn('haic/algorithms/koi/steering_terms.py', evaluate.HELPERS)
        self.assertIn('haic/algorithms/koi/avoidance_magnitude.py', evaluate.HELPERS)
        self.assertEqual(len(evaluate.HELPERS), len(set(evaluate.HELPERS)))

    def test_primary_completed_evidence_is_passive_and_preserves_partial_study(self):
        with patch.object(evaluate.importlib, 'import_module', side_effect=AssertionError('policy/environment import forbidden')):
            evidence = evaluate.consumed_evidence()
        self.assertFalse(evidence['fresh'])
        self.assertEqual(evidence['cells'], [list(c) for c in evaluate.CELLS])
        self.assertEqual(len(evidence['baseline_outcomes']), 4)
        self.assertEqual(evidence['historical']['episodes'], 8)
        self.assertGreater(evidence['historical']['peak_serialization_rss_bytes'], 0)
        self.assertIsNotNone(evidence['prior_partial_operator_error'])
        self.assertEqual(evidence['exclusion_audit']['observations_read'], 0)
        self.assertEqual(evidence['exclusion_audit']['excludes'], list(evaluate.PROTECTED))
        self.assertTrue(all(evaluate.sha(p) == h for p, h in evidence['source_sha256'].items()))
        self.assertTrue(any(p.endswith('koi-steering-generalization-v1/reset-ledger.jsonl') for p in evidence['source_sha256']))
        self.assertTrue(any(p.endswith('seed-3184000015.json') for p in evidence['source_sha256']))

    def test_original_changed_nominal_and_final_shield_taps_call_once(self):
        seen = []
        action = [.1, .6, 0.]
        original = SimpleNamespace(driver=object(), contact_mode='crossing_projection',
                                   act=lambda obs: seen.append(obs) or action)

        class Candidate:
            def __init__(self, nominal):
                self.nominal, self.last = nominal, {}

            def act(self, obs):
                changed = self.nominal.act(obs).copy()
                changed[0] = .2
                return changed

        class Shield:
            def __init__(self):
                self.driver, self.last_shield = original, {}

            def act(self, obs):
                changed = self.driver.act(obs).copy()
                self.last_shield = dict(baseline_action=changed.copy())
                changed[0] = -.3
                return changed

        safety = Shield()
        model = SimpleNamespace(driver=safety, act=safety.act)
        model, tap, crossing, candidate = evaluate.compose(model, evaluate.ARMS[1], Candidate)
        obs = object()
        result = evaluate.measured_act(model, tap, crossing, candidate, obs)
        self.assertIs(model.driver, safety)
        self.assertEqual(seen, [obs])
        self.assertEqual((tap.calls, crossing.calls), (1, 1))
        self.assertEqual(crossing.action, [.1, .6, 0.])
        self.assertEqual(tap.action, [.2, .6, 0.])
        self.assertEqual(result, [-.3, .6, 0.])
        self.assertFalse(hasattr(tap, 'driver'))
        action[0] = .9
        assert crossing.action is not None
        self.assertEqual(crossing.action[0], .1)

    def test_retained_state_and_action_feedback_rejected(self):
        for extra in ('plan', 'wheel_angle', 'recovery_state', 'observe_executed_action', 'driver'):
            candidate = SimpleNamespace(nominal=object(), last={})
            setattr(candidate, extra, lambda *args: None)
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'retained state'):
                evaluate.candidate_state(candidate)

    def test_missing_duplicate_calls_and_pedal_changes_fail_closed(self):
        for calls in (0, 2):
            probe = evaluate.ActionProbe(SimpleNamespace(act=lambda obs: [0., 1., 0.]))

            def act(obs):
                for _ in range(calls):
                    probe.act(obs)
                return [0., 1., 0.]

            with self.subTest(calls=calls), self.assertRaisesRegex(ValueError, 'exactly once'):
                evaluate.measured_act(SimpleNamespace(act=act), probe, probe, None, None)
        probe = evaluate.ActionProbe(SimpleNamespace(act=lambda obs: [0., 1., 0.]))

        def pedals(obs):
            probe.act(obs)
            return [0., .5, 0.]

        with self.assertRaisesRegex(ValueError, 'pedals'):
            evaluate.measured_act(SimpleNamespace(act=pedals), probe, probe, None, None)

    def test_natural_dnfs_valid_but_caps_bad_telemetry_and_nonfinite_invalid(self):
        episode = dict(error=None, invalid_actions=0, completed=False, retire_reason='crash',
                       damage=.2, collisions=1, decision_trace=[{}], steps=1,
                       damage_telemetry_valid=True, collision_telemetry_valid=True)
        self.assertTrue(evaluate.valid_episode(episode))
        for change in ({'retire_reason': 'max_steps'}, {'damage': float('inf')}, {'collisions': True},
                       {'steps': True}, {'damage_telemetry_valid': False}, {'completed': True, 'lapTimeMs': float('nan')}):
            with self.subTest(change=change):
                self.assertFalse(evaluate.valid_episode(dict(episode, **change)))

    def test_modified_contract_rejected_before_import(self):
        with patch.object(evaluate, 'sha', return_value='pin'), patch.object(evaluate, 'read_json', return_value={'gate': {}}):
            with self.assertRaisesRegex(ValueError, 'contract'):
                evaluate.validate_frozen('/tmp/kilo/not-created', 'pin')

    def test_transitive_source_original_and_copy_drift_rejected_before_model_import(self):
        output = Path('/tmp/kilo/not-created')
        protocol: dict[str, Any] = dict(schema=evaluate.SCHEMA, cells=[list(c) for c in evaluate.CELLS],
            arms=list(evaluate.ARMS), subgroups=evaluate.SUBGROUPS, gate=evaluate.GATE,
            metric_definitions=evaluate.METRICS, schedule=evaluate.scheduled_slots(),
            baseline_zip_sha256=evaluate.BASELINE_SHA, shield_sha256=evaluate.SHIELD_SHA,
            fresh=False, official_action=False, max_decisions=1200, frame_skip=4, warmup_ticks=50, raw_fps=50,
            python=str(evaluate.shield.legacy.PYTHON), snapshot=str(evaluate.shield.legacy.SNAPSHOT),
            candidate_state_keys=list(evaluate.STATE_KEYS), candidate_action_feedback=False,
            environment_sha256={str(i): 'pin' for i in range(142)}, source_inventory={})
        for name in evaluate.HELPERS:
            protocol['source_inventory'][name] = dict(original=str(output / 'original' / name),
                sha256=evaluate.SHIELD_SHA if name == 'haic/algorithms/koi/collision_shield.py' else 'pin')
        name = 'scripts/analyze_koi_adaptive_ab.py'
        for changed in (output / 'original' / name, output / 'source' / name):
            def checksum(path):
                path = Path(path)
                if path == changed:
                    return 'changed'
                return evaluate.SHIELD_SHA if str(path).endswith('/haic/algorithms/koi/collision_shield.py') else 'pin'
            with self.subTest(changed=changed), patch.object(evaluate, 'read_json', return_value=protocol), \
                    patch.object(evaluate, 'sha', side_effect=checksum):
                with self.assertRaisesRegex(ValueError, 'helper source'):
                    evaluate.validate_frozen(output, 'pin')

    def test_preflight_missing_resident_measurement_or_new_state_cannot_authorize_run(self):
        receipt: dict[str, Any] = dict(protocol_sha256='pin', environment_resets=0,
            arms=[dict(mode=a, import_only=True, environment_resets=0, current_resident_rss_bytes=100,
                candidate_state_keys=list(evaluate.STATE_KEYS) if a == evaluate.ARMS[1] else [],
                candidate_driver_hidden=True, candidate_feedback_method=False) for a in evaluate.ARMS])
        evaluate.validate_preflight(receipt, 'pin', forecast())
        for change in ({'current_resident_rss_bytes': None}, {'current_resident_rss_bytes': 101},
                       {'candidate_state_keys': ['nominal', 'last', 'plan']}, {'candidate_feedback_method': True}):
            changed = copy.deepcopy(receipt)
            changed['arms'][1].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'preflight/RSS'):
                evaluate.validate_preflight(changed, 'pin', forecast())

    def test_current_measurement_timestamp_and_every_ancestor_recorded(self):
        value = measurements()
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary, \
                patch.object(evaluate.observer_helpers, 'resource_measurements', return_value=value):
            result = evaluate.resource_measurements(temporary)
        self.assertGreater(result['measured_at_ns'], 0)
        self.assertIn('T', result['measured_at_utc'])
        self.assertEqual(result['minimum_raw_cgroup_headroom_bytes'], 900)
        self.assertEqual(result['visible_cgroup_ancestors'][0]['possibly_reclaimable_clean_inactive_file_bytes'], 45)
        self.assertGreater(result['current_process_rss_bytes'], 0)

    def test_parent_headroom_cache_and_distinct_filesystems_fail_closed(self):
        initial = measurements()
        self.assertTrue(evaluate.check_resources(initial, forecast(), 8)['allowed'])
        small_parent = copy.deepcopy(initial['visible_cgroup_ancestors'][0])
        small_parent.update(path='/sys/fs/cgroup/parent', memory_max='200', memory_current=150, raw_headroom_bytes=50)
        initial['visible_cgroup_ancestors'].insert(0, small_parent)
        with self.assertRaisesRegex(ValueError, 'ancestor'):
            evaluate.check_resources(initial, forecast(), 8)
        value = measurements()
        value['disk'][1]['free_bytes'] = 0
        with self.assertRaisesRegex(ValueError, 'separate'):
            evaluate.check_resources(value, forecast(), 8)
        value['disk'][1]['device'] = 1
        value['disk'][0]['free_bytes'] = 825
        with self.assertRaisesRegex(ValueError, 'shared'):
            evaluate.check_resources(value, forecast(), 8)
        value = measurements()
        value['visible_cgroup_ancestors'][0]['raw_headroom_bytes'] = 25
        self.assertTrue(evaluate.check_resources(value, forecast(), 1, allocated_child_bytes=100)['allowed'])
        previous = measurements()
        value = measurements()
        value['visible_cgroup_ancestors'][0]['memory_events'] = 'oom 1\noom_kill 0\n'
        result = evaluate.check_resources(value, forecast(), 1, previous=previous)
        self.assertTrue(result['allowed'])
        self.assertIn('OOM counters increased', result['warnings'][0])

    def test_admission_block_retains_measurement_without_child_or_reset(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            output = Path(temporary)
            protocol: dict[str, Any] = dict(schedule=evaluate.scheduled_slots(), python='unused',
                source_inventory={'scripts/evaluate_koi_avoidance_magnitude.py': {'sha256': 'pin'}},
                resource_plan=dict(forecast=forecast(), freeze_measurement=measurements()))
            evaluate.save(output / 'preflight.json', dict(protocol_sha256='pin', environment_resets=0,
                arms=[dict(mode=a, import_only=True, environment_resets=0, current_resident_rss_bytes=100,
                    candidate_state_keys=list(evaluate.STATE_KEYS) if a == evaluate.ARMS[1] else [],
                    candidate_driver_hidden=True, candidate_feedback_method=False) for a in evaluate.ARMS]))
            measured = measurements()
            measured['host_mem_available_bytes'] = 0
            with patch.object(evaluate, 'validate_frozen', return_value=protocol), \
                    patch.object(evaluate, 'sha', return_value='pin'), \
                    patch.object(evaluate, 'resource_measurements', return_value=measured), \
                    patch.object(evaluate.subprocess, 'run') as child:
                with self.assertRaisesRegex(ValueError, 'RAM forecast'):
                    evaluate.run(output, 'pin')
            child.assert_not_called()
            report = evaluate.read_json(output / 'episode-report.json')
            self.assertTrue(all(r['status'] == 'unrun' for r in report['rows']))
            evidence = json.loads((output / 'resource-receipts.jsonl').read_text())
            self.assertEqual(evidence['measurement']['host_mem_available_bytes'], 0)
            self.assertIn('RAM forecast', evidence['error'])
            self.assertFalse((output / 'reset-ledger.jsonl').exists())

    def test_preflight_is_two_isolated_import_only_children(self):
        schedule = evaluate.scheduled_slots()
        def child(command, **kwargs):
            index = int(command[-3])
            return SimpleNamespace(returncode=0, stdout=json.dumps(dict(import_only=True,
                environment_resets=0, mode=schedule[index]['mode'], current_resident_rss_bytes=100,
                candidate_state_keys=list(evaluate.STATE_KEYS) if schedule[index]['mode'] == evaluate.ARMS[1] else [],
                candidate_driver_hidden=True, candidate_feedback_method=False)))
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary, \
                patch.object(evaluate, 'validate_frozen', return_value=dict(python='unused', schedule=schedule,
                    resource_plan=dict(forecast=forecast()))), \
                patch.object(evaluate.subprocess, 'run', side_effect=child) as launch:
            result = evaluate.preflight(temporary, 'pin')
            self.assertTrue((Path(temporary) / 'preflight.json').exists())
        self.assertEqual(result['environment_resets'], 0)
        self.assertEqual(launch.call_count, 2)
        self.assertTrue(all(c.args[0][1:3] == ['-I', '-B'] and c.args[0][-1] == 'True' for c in launch.call_args_list))

    def test_timeout_retains_all_partial_artifacts_and_never_retries(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            output = Path(temporary)
            protocol: dict[str, Any] = dict(schedule=evaluate.scheduled_slots(), python='unused', child_timeout_s=17,
                source_inventory={'scripts/evaluate_koi_avoidance_magnitude.py': {'sha256': 'pin'}},
                resource_plan=dict(forecast=forecast(), freeze_measurement=measurements()))
            evaluate.save(output / 'preflight.json', dict(protocol_sha256='pin', environment_resets=0,
                arms=[dict(mode=a, import_only=True, environment_resets=0, current_resident_rss_bytes=100,
                    candidate_state_keys=list(evaluate.STATE_KEYS) if a == evaluate.ARMS[1] else [],
                    candidate_driver_hidden=True, candidate_feedback_method=False) for a in evaluate.ARMS]))
            def timeout(command, **kwargs):
                (output / protocol['schedule'][0]['file']).with_suffix('.raw.jsonl').write_text('{"partial":true}\n')
                raise subprocess.TimeoutExpired(command, 17, output=b'retained')
            with patch.object(evaluate, 'validate_frozen', return_value=protocol), \
                    patch.object(evaluate, 'sha', return_value='pin'), \
                    patch.object(evaluate, 'resource_measurements', return_value=measurements()), \
                    patch.object(evaluate.subprocess, 'run', side_effect=timeout) as child:
                with self.assertRaisesRegex(RuntimeError, 'child_timeout'):
                    evaluate.run(output, 'pin')
                with self.assertRaisesRegex(ValueError, 'never restart'):
                    evaluate.run(output, 'pin')
            report = evaluate.read_json(output / 'episode-report.json')
            self.assertEqual(child.call_count, 1)
            self.assertEqual([r['status'] for r in report['rows']], ['operator_error'] + ['unrun'] * 7)
            self.assertEqual(len(report['rows'][0]['partial_artifacts']), 4)
            self.assertEqual(len(report['operator_artifacts']), 3)
            self.assertFalse((output / 'reset-ledger.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
