"""Shield operator tests use receipts and fake children, never an environment."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import evaluate_koi_collision_shield as evaluate


class ShieldEvaluatorTest(unittest.TestCase):
    def test_fixed_schedule_has_only_same_six_consumed_cells(self):
        rows = evaluate.scheduled_slots()
        self.assertEqual(len(rows), 12)
        self.assertEqual({(r['track_id'], r['seed']) for r in rows}, set(evaluate.CELLS))
        self.assertEqual(len({r['file'] for r in rows}), 12)
        self.assertEqual({r['mode'] for r in rows}, set(evaluate.ARMS))
        self.assertEqual(len({seed for _, seed in evaluate.CELLS}), 5)
        self.assertNotIn(4, {r['track_id'] for r in rows})
        self.assertTrue(all(r['seed'] > 3184000000 for r in rows))
        self.assertEqual(list(evaluate.SUBGROUPS.values()).count('ordinary_controls'), 2)

    def test_actual_primary_consumed_receipts_are_verified_without_importing_policy(self):
        evidence = evaluate.consumed_evidence()
        self.assertEqual(len(evidence['baseline_outcomes']), 6)
        self.assertEqual(sum(row['finished'] for row in evidence['baseline_outcomes']), 3)
        self.assertAlmostEqual(sum(row['damage'] for row in evidence['baseline_outcomes']), 1.4)
        self.assertEqual(sum(row['collisions'] for row in evidence['baseline_outcomes']), 7)
        self.assertEqual(len(evidence['source_sha256']), 38)

    def test_probe_is_exact_and_does_not_expose_privileged_state(self):
        returned = [.125, .6, 0.]
        seen = []
        driver = SimpleNamespace(contact_mode='crossing_projection', act=lambda obs: seen.append(obs) or returned)
        probe = evaluate.BaselineProbe(driver)
        observation = object()
        self.assertIs(probe.act(observation), returned)
        self.assertEqual(probe.action, returned)
        self.assertEqual(probe.calls, 1)
        self.assertEqual(probe.contact_mode, 'crossing_projection')
        self.assertEqual(seen, [observation])
        returned[0] = .9
        assert probe.action is not None
        self.assertEqual(probe.action[0], .125)

    def test_paths_reject_escape(self):
        with self.assertRaises(ValueError):
            evaluate.local_path('/tmp/kilo/study', '../outside.json')

    def test_capped_invalid_and_missing_episodes_fail_closed(self):
        base = dict(error=None, invalid_actions=0, completed=False, retire_reason='crash',
                    damage=.2, collisions=1, decision_trace=[{}])
        self.assertTrue(evaluate.valid_episode(base))
        for change in ({'error': 'failed'}, {'invalid_actions': 1}, {'completed': 1},
                       {'retire_reason': 'max_decisions'}, {'damage': None}, {'decision_trace': []}):
            with self.subTest(change=change):
                self.assertFalse(evaluate.valid_episode(dict(base, **change)))

    def test_frozen_contract_rejects_changed_cells_gates_and_wrong_comparator(self):
        protocol = dict(schema=evaluate.SCHEMA, cells=[list(c) for c in evaluate.CELLS],
                        arms=list(evaluate.ARMS), schedule=evaluate.scheduled_slots(),
                        subgroups=evaluate.SUBGROUPS, gate=evaluate.GATE,
                        comparator_zip_sha256=evaluate.legacy.CROSSING_SHA, fresh=False, official_action=False)
        for key, value in (('cells', [[4, 38300]]), ('fresh', True), ('gate', {}),
                           ('arms', ['crossing_projection', 'collision_recovery']),
                           ('comparator_zip_sha256', evaluate.legacy.V2_SHA)):
            with self.subTest(key=key), patch.object(evaluate, 'sha', return_value='protocol'), \
                    patch.object(evaluate, 'read_json', return_value=dict(protocol, **{key: value})):
                with self.assertRaisesRegex(ValueError, 'contract'):
                    evaluate.validate_frozen('/tmp/kilo/not-created', 'protocol')
        with patch.object(evaluate, 'sha', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'protocol hash'):
                evaluate.validate_frozen('/tmp/kilo/not-created', 'expected')

    def test_timeout_preserves_partial_output_and_refuses_restart(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            output = Path(temporary)
            protocol: dict = dict(schedule=evaluate.scheduled_slots(), python='not-executed', child_timeout_s=180)

            def timeout(command, **kwargs):
                path = output / protocol['schedule'][0]['file']
                path.with_suffix('.raw.jsonl').write_text('{"partial":true}\n')
                raise evaluate.subprocess.TimeoutExpired(command, 180, output='retained')

            with patch.object(evaluate, 'validate_frozen', return_value=protocol), \
                    patch.object(evaluate, 'sha', return_value='hash'), \
                    patch.object(evaluate.subprocess, 'run', side_effect=timeout) as child:
                with self.assertRaisesRegex(RuntimeError, 'child_timeout'):
                    evaluate.run(output, 'hash')
                self.assertEqual(child.call_count, 1)
                with self.assertRaisesRegex(ValueError, 'existing slot'):
                    evaluate.run(output, 'hash')
            report = evaluate.read_json(output / 'episode-report.json')
            self.assertEqual(report['protocol_sha256'], 'hash')
            self.assertEqual([r['status'] for r in report['rows']], ['operator_error'] + ['unrun'] * 11)
            self.assertFalse((output / 'reset-ledger.jsonl').exists())
            path = output / report['rows'][0]['file']
            self.assertEqual(path.with_suffix('.stdout.txt').read_text(), 'retained')
            self.assertEqual(json.loads(path.with_suffix('.raw.jsonl').read_text()), {'partial': True})

    def test_preflight_only_constructs_import_only_commands(self):
        protocol = {'python': 'not-executed'}
        receipt = json.dumps(dict(import_only=True, environment_resets=0, model_type='Fake'))
        with patch.object(evaluate, 'validate_frozen', return_value=protocol), \
                patch.object(evaluate.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=receipt)) as child:
            result = evaluate.preflight('/tmp/kilo/not-created', 'protocol')
        self.assertEqual(child.call_count, 2)
        self.assertTrue(all(call.args[0][-1] == 'True' for call in child.call_args_list))
        self.assertEqual(result['environment_resets'], 0)


if __name__ == '__main__':
    unittest.main()
