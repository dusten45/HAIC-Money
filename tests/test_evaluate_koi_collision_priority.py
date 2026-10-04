"""Focused no-environment tests for the consumed-TRAIN collision-priority harness."""

import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import evaluate_koi_collision_priority as evaluate


def state(t, wheels, rear=-1., clearance=1., contacts=()):
    return dict(t=t, wheel_road_contacts=wheels, rear=[rear], clearance=[clearance], contacts=list(contacts))


class EvaluatorTest(unittest.TestCase):
    def test_allowlist_is_completed_execution_not_finished_only(self):
        base = dict(track_id=1, seed=3184000005, mode=evaluate.PRIOR_ARM,
                    status='completed', completed=False, retire_reason='crash', error=None, invalid_actions=0)
        rows = [base, dict(base, track_id=3, seed=3184000016), dict(base, seed=3184000017),
                dict(base, seed=38300), dict(base, seed=49300), dict(base, track_id=4),
                dict(base, seed=3184000006, status='unrun'),
                dict(base, seed=3184000007, retire_reason='act_timeout'),
                dict(base, seed=3184000008, invalid_actions=1)]
        self.assertEqual(set(evaluate.completed_cells({'rows': rows})), {(1, 3184000005)})
        rows.append(dict(base, mode='crossing_projection', seed=3184000002))
        self.assertEqual(set(evaluate.completed_cells({'rows': rows}, 'crossing_projection')), {(1, 3184000002)})

    def test_cells_reject_malformed_and_duplicate(self):
        self.assertEqual(evaluate.parse_cells('1:3184000001,3:3184000004'), [(1, 3184000001), (3, 3184000004)])
        for invalid in ('', '1', '1:a', '1:2:3', '1:2,1:2'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluate.parse_cells(invalid)

    def test_physical_contacts_streak_and_censored_reacquisition(self):
        states = [state(0., [1]*4), state(.02, [0]*4),
                  state(.04, [0]*4, rear=2., clearance=-.1, contacts=[0]),
                  state(.06, [1, 0, 0, 0], rear=3., contacts=[0]),
                  state(.08, [0]*4, rear=4.), state(.10, [0]*4, rear=5.)]
        result = evaluate.physical_metrics(states, [{'radius': 1.}])
        self.assertEqual(result['raw_ticks'], 5)
        self.assertEqual(result['no_wheel_road_contact_max_ticks'], 2)
        self.assertEqual(result['no_wheel_road_contact_max_s'], .04)
        self.assertEqual(result['any_wheel_offroad_max_ticks'], 5)
        self.assertEqual(result['physical_contact_events'], 1)
        self.assertEqual(result['physical_contact_ticks'], 2)
        self.assertEqual(result['min_fixture_clearance'], -.1)
        returned, censored = result['road_reacquisition']
        self.assertFalse(returned['censored'])
        self.assertAlmostEqual(returned['duration_or_censored_lower_bound_s'], .04)
        self.assertTrue(censored['censored'])
        self.assertAlmostEqual(censored['duration_or_censored_lower_bound_s'], .02)

    def test_reset_already_behind_is_not_observed_passage(self):
        result = evaluate.physical_metrics([state(0., [0]*4, rear=3.), state(.02, [1]*4, rear=4.)], [{'radius': 1.}])
        event = result['road_reacquisition'][0]
        self.assertTrue(event['left_censored'])
        self.assertTrue(event['censored'])
        self.assertEqual(event['censor_reason'], 'passage_unobserved')
        self.assertEqual(event['first_contact_return_t'], .02)

    def test_prefix_uses_pre_state_and_excludes_unmatched_laps(self):
        def row(step):
            return dict(step=step, steer=0., gas=1., brake=0.,
                        controller=dict(evaluation_only=dict(observation_sha256=str(step), pre=dict(x=step))))
        base: dict = dict(track_id=1, seed=3184000001, repeat=0, geometry_sha256='g', initial_state={},
                    initial_observation_sha256='o', completed=True, lapTimeMs=20000, decision_trace=[row(1), row(2)])
        candidate = copy.deepcopy(base)
        candidate['decision_trace'][1]['steer'] = .1
        candidate.update(completed=False, lapTimeMs=None)
        result = evaluate.pair_metrics(base, candidate)
        self.assertEqual(result['equal_action_prefix_decisions'], 1)
        self.assertTrue(result['first_changed_action_pre_state_equal'])
        self.assertIsNone(result['matched_lap_delta_ms'])

    def test_raw_reset_intent_precedes_even_implicit_resets_and_counter_is_post_decision(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            ledger = Path(temporary) / 'ledger.jsonl'
            calls = []

            def reset():
                calls.append(json.loads(ledger.read_text().splitlines()[-1])['reset_index'])

            raw = SimpleNamespace(reset=reset, step=lambda action: (None, -.4, False, False, {}))
            wrapper = SimpleNamespace(off_track_counter=99)

            def step(action):
                raw.step(action)
                wrapper.off_track_counter += 1
                return None, -.4, False, False, {}

            inner = SimpleNamespace(environment=wrapper, unwrapped=raw, step=step)
            observer = evaluate.make_observer(inner, io.StringIO(), ledger, {'seed': 1}, 'hash', None)
            raw.reset()
            raw.reset()
            self.assertEqual(calls, [1, 2])
            observer.state = lambda: {'t': 1.}
            observer.step([0., 1., 0.])
            self.assertEqual(observer.last['off_track_count_before'], 99)
            self.assertEqual(observer.last['off_track_count_after'], 100)
            self.assertEqual(observer.max_counter, 100)

    def test_timeout_preserves_protocol_and_partial_files(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            output = Path(temporary) / 'run'
            wrapper = Path(temporary) / 'collision_recovery.py'
            wrapper.write_text('class CollisionRecoveryAgent:\n    def __init__(self, driver):\n        self.driver = driver\n')

            def timeout(command, **kwargs):
                self.assertTrue((output / 'protocol.json').exists())
                protocol = evaluate.read_json(output / 'protocol.json')
                self.assertEqual(protocol['comparator_zip_sha256'], evaluate.CROSSING_SHA)
                self.assertEqual(protocol['arms'], ['crossing_projection', 'collision_recovery'])
                self.assertEqual(protocol['class_name'], 'CollisionRecoveryAgent')
                self.assertEqual(protocol['wrapper_file'], 'source/collision_recovery.py')
                self.assertFalse(protocol['fresh'])
                path = output / protocol['schedule'][0]['file']
                path.with_suffix('.raw.jsonl').write_text('{"partial": true}\n')
                raise evaluate.subprocess.TimeoutExpired(command, 180, output='retained')

            # No child, model, or environment is imported/constructed by this test.
            with patch.object(evaluate, 'validate_cells', return_value=[]), patch.object(evaluate.subprocess, 'run', side_effect=timeout):
                with self.assertRaisesRegex(RuntimeError, 'child_timeout'):
                    evaluate.run([(1, 3184000001)], output, baseline='crossing_projection',
                                 wrapper=wrapper,
                                 class_name='CollisionRecoveryAgent', candidate_name='collision_recovery')
            report = evaluate.read_json(output / 'episode-report.json')
            self.assertEqual([row['status'] for row in report['rows']], ['operator_error', 'unrun'])
            self.assertEqual(set(evaluate.read_json(output / 'summary.json')['arms']), {'crossing_projection', 'collision_recovery'})
            path = output / report['rows'][0]['file']
            self.assertIn('partial', path.with_suffix('.raw.jsonl').read_text())
            self.assertEqual(path.with_suffix('.stdout.txt').read_text(), 'retained')
            self.assertFalse((output / 'reset-ledger.jsonl').exists())
            # One real crossing ZIP + synthetic external-wrapper import smoke;
            # independent of the concurrently implemented policy. No env factory.
            child = evaluate.subprocess.run([str(evaluate.PYTHON), '-I', '-B',
                    str(output / 'source/scripts/evaluate_koi_collision_priority.py'),
                    '--output', str(output), '--worker', '0', '--import-only'],
                    cwd=output / 'model', env=evaluate.ENV, capture_output=True, text=True, timeout=30)
            self.assertEqual(child.returncode, 0, child.stderr)
            smoke = json.loads(child.stdout)
            self.assertEqual(smoke['environment_resets'], 0)
            self.assertEqual(smoke['model_type'], 'CollisionRecoveryAgent')
            self.assertEqual(smoke['driver_type'], 'ContactContinuityAgent')
            self.assertTrue(smoke['agent_file'].startswith(str(output / 'model')))
            self.assertTrue(smoke['footprint_file'].startswith(str(output / 'source')))
            self.assertTrue(smoke['evaluator_file'].startswith(str(evaluate.SNAPSHOT)))
            self.assertFalse((output / 'reset-ledger.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
