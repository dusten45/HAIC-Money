"""Synthetic nominal operator tests; no environment construction or reset."""

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import evaluate_koi_nominal_trajectory as evaluate


class OperatorTest(unittest.TestCase):
    def test_fixed_eight_consumed_cells_and_source_closure(self):
        rows = evaluate.scheduled_slots()
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({r['file'] for r in rows}), 16)
        self.assertEqual({(r['track_id'], r['seed']) for r in rows}, set(evaluate.CELLS))
        self.assertEqual(len({s for _, s in evaluate.CELLS}), 5)
        ordinary = [cell for cell in evaluate.CELLS if evaluate.SUBGROUPS[f'{cell[0]}:{cell[1]}'] == 'ordinary_controls']
        self.assertEqual(len(ordinary), 4)
        self.assertEqual(len({s for _, s in ordinary}), 2)
        self.assertNotIn((3, 3184000016), evaluate.CELLS)
        self.assertTrue(all(t in (1, 2, 3) and s >= 3184000001 for t, s in evaluate.CELLS))
        for name in ('nominal_trajectory', 'minimum_clearance', 'collision_shield', 'trajectory_rollout'):
            self.assertIn(f'haic/algorithms/koi/{name}.py', evaluate.HELPERS)

    def test_probe_never_unwraps_and_taps_copy_once(self):
        action = [.1, .6, 0.]
        nominal = SimpleNamespace(driver=object(), contact_mode='crossing_projection', act=lambda obs: action)
        probe = evaluate.ActionProbe(nominal)
        self.assertFalse(hasattr(probe, 'driver'))
        self.assertIs(probe.act(object()), action)
        action[0] = .8
        self.assertEqual(probe.action, [.1, .6, 0.])
        self.assertEqual(probe.calls, 1)
        self.assertEqual(probe.contact_mode, 'crossing_projection')

    def test_postshield_feedback_exactly_once_and_original_once(self):
        original_calls, feedback = [], []
        original = SimpleNamespace(act=lambda obs: original_calls.append(obs) or [.1, .6, 0.])

        class Candidate:
            def __init__(self, nominal):
                self.nominal = nominal

            def act(self, observation):
                action = self.nominal.act(observation).copy()
                action[0] = .2
                return action

            def observe_executed_action(self, action):
                feedback.append(action.copy())

        class FakeShield:
            def __init__(self):
                self.driver = original

            def act(self, observation):
                action = self.driver.act(observation).copy()
                action[0] = -.3
                return action

        safety = FakeShield()
        model = SimpleNamespace(driver=safety, act=safety.act)
        model, nominal, crossing, candidate = evaluate.compose(model, evaluate.ARMS[1], Candidate)
        observation = object()
        actual = evaluate.measured_act(model, nominal, crossing, candidate, observation)
        self.assertIs(model.driver, safety)
        self.assertEqual(actual, [-.3, .6, 0.])
        self.assertEqual(feedback, [actual])
        self.assertEqual(original_calls, [observation])
        self.assertEqual(nominal.action, [.2, .6, 0.])
        self.assertEqual(crossing.action, [.1, .6, 0.])

    def test_baseline_keeps_exact_existing_shield(self):
        original = SimpleNamespace(act=lambda obs: [.1, 1., 0.])
        safety = SimpleNamespace(driver=original)
        model = SimpleNamespace(driver=safety, act=lambda obs: safety.driver.act(obs))
        composed, nominal, crossing, candidate = evaluate.compose(model, evaluate.ARMS[0])
        self.assertIs(composed.driver, safety)
        self.assertIs(nominal, crossing)
        self.assertIsNone(candidate)
        self.assertEqual(evaluate.measured_act(model, nominal, crossing, candidate, None), [.1, 1., 0.])

    def test_missing_and_duplicate_nominal_calls_rejected(self):
        for calls in (0, 2):
            probe = evaluate.ActionProbe(SimpleNamespace(act=lambda obs: [0., 1., 0.]))

            def act(observation):
                for _ in range(calls):
                    probe.act(observation)
                return [0., 1., 0.]

            with self.subTest(calls=calls), self.assertRaisesRegex(ValueError, 'exactly once'):
                evaluate.measured_act(SimpleNamespace(act=act), probe, probe, None, None)

    def test_capped_and_missing_telemetry_episodes_invalid(self):
        episode = dict(error=None, invalid_actions=0, completed=False, retire_reason='crash',
                       damage=.2, collisions=1, decision_trace=[{}], steps=1,
                       damage_telemetry_valid=True, collision_telemetry_valid=True)
        self.assertTrue(evaluate.valid_episode(episode))
        for change in ({'retire_reason': 'max_steps'}, {'damage_telemetry_valid': False},
                       {'collisions': True}, {'steps': 2}, {'damage': None}):
            with self.subTest(change=change):
                self.assertFalse(evaluate.valid_episode(dict(episode, **change)))

    def test_modified_gate_rejected_before_import(self):
        with patch.object(evaluate, 'sha', return_value='protocol'), \
                patch.object(evaluate, 'read_json', return_value={'schema': evaluate.SCHEMA, 'gate': {}}):
            with self.assertRaisesRegex(ValueError, 'contract'):
                evaluate.validate_frozen('/tmp/kilo/not-created', 'protocol')

    def test_preflight_commands_are_import_only(self):
        receipt = json.dumps(dict(import_only=True, environment_resets=0))
        with patch.object(evaluate, 'validate_frozen', return_value={'python': 'unused'}), \
                patch.object(evaluate.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=receipt)) as child:
            result = evaluate.preflight('/tmp/kilo/not-created', 'hash')
        self.assertEqual(result['environment_resets'], 0)
        self.assertEqual(child.call_count, 2)
        self.assertTrue(all(call.args[0][-1] == 'True' for call in child.call_args_list))
        self.assertTrue(all(call.args[0][1:3] == ['-I', '-B'] for call in child.call_args_list))

    def test_timeout_pins_partial_files_and_refuses_restart(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            output = Path(temporary)
            protocol: dict = dict(schedule=evaluate.scheduled_slots(), python='unused',
                            source_inventory={'scripts/evaluate_koi_nominal_trajectory.py': {'sha256': 'hash'}})

            def timeout(command, **kwargs):
                (output / protocol['schedule'][0]['file']).with_suffix('.raw.jsonl').write_text('{"partial":true}\n')
                raise subprocess.TimeoutExpired(command, 180, output=b'retained')

            with patch.object(evaluate, 'validate_frozen', return_value=protocol), \
                    patch.object(evaluate, 'sha', return_value='hash'), \
                    patch.object(evaluate.subprocess, 'run', side_effect=timeout) as child:
                with self.assertRaisesRegex(RuntimeError, 'child_timeout'):
                    evaluate.run(output, 'hash')
                with self.assertRaisesRegex(ValueError, 'never restart'):
                    evaluate.run(output, 'hash')
            report = json.loads((output / 'episode-report.json').read_text())
            self.assertEqual(child.call_count, 1)
            self.assertEqual([r['status'] for r in report['rows']], ['operator_error'] + ['unrun'] * 15)
            pins = report['rows'][0]['partial_artifacts']
            self.assertEqual(len(pins), 4)
            self.assertTrue(all(pin['bytes'] == (output / pin['file']).stat().st_size for pin in pins))
            self.assertFalse((output / 'reset-ledger.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
