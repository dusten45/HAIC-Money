"""Synthetic/file-only paired operator tests; no simulator or historical replay."""

import ast
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import sys
import time
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import collect_joint_envelope_pairs as pairs


def calibration() -> dict[str, Any]:
    return dict(schema=pairs.CALIBRATION_SCHEMA, calibration_complete=True,
        performance=dict(lower=.05, upper=.05, target='per-reference envelope excess'),
        absolute_calibration=dict(CAL=[3184000002, 3184000013, 3184000015],
            HELD_OUT_FROM_FIT=[3184000001, 3184000006], paired_cost_residual=.44961874671412616,
            position_residual=[0.] + [.5] * 16, yaw_residual=[0.] + [.05] * 16,
            mapping=dict(by_provenance=dict(measured_image=dict(supported=True, n_cal=3,
                position_residual=.5, yaw_residual=.05)))))


def motion(provenance='measured_image', valid=True):
    return dict(valid=valid, provenance=provenance, position_uncertainty=.1, yaw_uncertainty=.01)


def comparison(*, common=True, upper=.2, veto=True) -> dict[str, Any]:
    return dict(common_support=common, cost_supported=np.array([common, common]),
        delta_interval=np.array([[0., 0.], [-.4, upper]]),
        poses=np.zeros((2, 19, 17, 3)), reference_costs=np.zeros((2, 19, 5)),
        reference_delta=np.zeros((2, 19, 5)), absolute_supported=np.ones((2, 19), bool),
        veto=np.full((2, 19), veto, bool), absolute_road_clearance=np.ones((2, 19, 17)),
        absolute_obstacle_clearance=np.full((2, 19, 17), np.inf))


class FakeSelector:
    def __init__(self, found: int | None = 18):
        self.found, self.calls = found, 0
        self.commits, self.observations = [], []

    def propose(self, completed, observation):
        if completed != self.calls:
            raise AssertionError('query not sequential')
        self.calls += 1
        action = np.array([.1, 0., .2], np.float32)
        anchor = (dict(completed_prefix=completed, actions=pairs.joint_actions(action).tolist())
                  if completed == self.found else None)
        return action, anchor, dict(completed_prefix=completed)

    def commit(self, action):
        self.commits.append(np.asarray(action).copy())

    def observe(self, observation):
        self.observations.append(observation)


def run_core(found: int | None = 18, arm='baseline', baseline=None) -> tuple[dict[str, Any], list, list, Any]:
    selector = FakeSelector(found) if arm == 'baseline' else None
    events, calls = [], []

    def step(action, index):
        events.append(('step', index))
        calls.append(np.asarray(action).copy())
        return index + 1

    result = pairs.collect_prefix_and_tail(arm, 0, step,
        lambda anchor: events.append(('saved', anchor['completed_prefix'])), lambda _: None,
        selector=selector, baseline=baseline)
    return result, events, calls, selector


class CoreTests(unittest.TestCase):
    def test_exact_cells_budgets_and_same_environment_cap(self):
        self.assertEqual(pairs.CELLS, ((1, 3184000003), (2, 3184000004), (3, 3184000005)))
        slots = pairs.schedule()
        self.assertEqual(len(slots), 9)
        self.assertEqual(sum(s['decisions'] for s in slots), 576)
        self.assertEqual(sum(s['raw_ticks'] for s in slots), 2763)
        self.assertEqual(9 * pairs.WARMUP, 459)
        self.assertEqual({s['decisions'] for s in slots}, {64})
        self.assertEqual(pairs.LIMITS['total_seconds'], 2400)
        self.assertEqual(pairs.LIMITS['child_seconds'], 600)
        self.assertEqual(pairs.LIMITS['address_space_bytes'], 4 * 1024**3)
        self.assertEqual(pairs.LIMITS['rss_bytes'], 1024**3)

    def test_first_anchor_is_after_eighteen_completed_holds(self):
        result, events, calls, selector = run_core(18)
        self.assertEqual(result['anchor']['completed_prefix'], 18)
        self.assertEqual(len(result['anchor']['prefix_actions']), 18)
        self.assertEqual(len(calls), 22)
        self.assertEqual(result['agent_calls'], 19)
        self.assertEqual(events[17:20], [('step', 17), ('saved', 18), ('step', 18)])
        self.assertEqual(len(selector.commits), 22)
        self.assertEqual(len(selector.observations), 22)
        self.assertTrue(all(a.dtype == np.float32 for a in calls))
        np.testing.assert_array_equal(calls[18:], [calls[18]] * 4)

    def test_last_anchor_inclusive_has_sixty_four_holds(self):
        result, _, calls, selector = run_core(60)
        self.assertEqual(len(calls), 64)
        self.assertEqual(result['agent_calls'], 61)
        self.assertEqual(selector.calls, 61)
        self.assertEqual(result['anchor']['completed_prefix'], 60)

    def test_not_found_is_not_failure_or_zero_success_label(self):
        result, events, calls, selector = run_core(None)
        self.assertEqual(result['status'], 'NOT_FOUND')
        self.assertIsNone(result['anchor'])
        self.assertEqual(len(calls), 60)
        self.assertEqual(selector.calls, 61)
        self.assertFalse(any(event[0] == 'saved' for event in events))
        gate = pairs.potential_pilot_gate([dict(status='NOT_FOUND', cell=[1, 3])])
        self.assertEqual(gate['outcome'], 'N/A')
        self.assertIsNone(gate['supported_selected_benefit']['numerator'])
        self.assertEqual(gate['supported_selected_benefit']['denominator'], 0)

    def test_offbyone_anchor_outside_window_rejected(self):
        with self.assertRaisesRegex(ValueError, 'off-by-one'):
            run_core(17)

    def test_replays_have_no_selector_and_no_agent_queries(self):
        baseline, _, base_calls, _ = run_core(18)
        with patch.object(pairs, 'PrefixSelector', side_effect=AssertionError('Agent path forbidden')), \
                patch.object(pairs, 'query_champion', side_effect=AssertionError('query forbidden')):
            repeat, _, repeat_calls, selector = run_core(arm='repeat', baseline=baseline)
            alternative, _, alt_calls, _ = run_core(arm='alternative', baseline=baseline)
        self.assertIsNone(selector)
        self.assertEqual(repeat['agent_calls'], 0)
        self.assertEqual(alternative['agent_calls'], 0)
        np.testing.assert_array_equal(repeat_calls, base_calls)
        np.testing.assert_array_equal(alt_calls[:18], base_calls[:18])
        np.testing.assert_array_equal(alt_calls[19:], base_calls[19:])
        np.testing.assert_array_equal(alt_calls[18], pairs.joint_actions(base_calls[18])[1])

    def test_replay_refuses_not_found_or_selector(self):
        baseline, _, _, _ = run_core(None)
        with self.assertRaisesRegex(ValueError, 'complete contemporary'):
            run_core(arm='alternative', baseline=baseline)
        with self.assertRaisesRegex(ValueError, 'exclusively'):
            pairs.collect_prefix_and_tail('repeat', None, Mock(), Mock(), Mock(), selector=FakeSelector())

    def test_towards_zero_does_not_cross_and_exact_zero_goes_positive(self):
        for steer, expected in ((.02, 0.), (-.02, -0.), (0., .04), (.35, .31), (-.35, -.31)):
            with self.subTest(steer=steer):
                actions = pairs.joint_actions(np.array([steer, 0., .02], np.float32))
                self.assertEqual(actions.dtype, np.float32)
                self.assertAlmostEqual(float(actions[1, 0]), expected, places=7)
                self.assertEqual(actions[1, 1], np.float32(.05))
                self.assertEqual(actions[1, 2], np.float32(0.))
                tail = pairs.tail_actions(actions, 'alternative')
                self.assertEqual(tail.shape, (4, 3))
                np.testing.assert_array_equal(tail[0], actions[1])
                np.testing.assert_array_equal(tail[1:], [actions[0]] * 3)

    def test_action32_refuses_silent_requantization_invalid_values(self):
        for action in ([.1, 0., .2], [float('nan'), 0., 0.], [0., -1., 0.], [2., 0., 0.]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                pairs.action32(action)
        action = np.array([.1, 0., .2], np.float32)
        np.testing.assert_array_equal(pairs.action32(action.tolist()), action)

    def test_step_failure_not_retried_or_replaced(self):
        selector = FakeSelector(18)
        step = Mock(side_effect=RuntimeError('partial'))
        with self.assertRaisesRegex(RuntimeError, 'partial'):
            pairs.collect_prefix_and_tail('baseline', 0, step, Mock(), Mock(), selector=selector)
        self.assertEqual(step.call_count, 1)
        self.assertEqual(selector.calls, 1)
        self.assertEqual(len(selector.commits), 1)
        self.assertEqual(len(selector.observations), 0)


class SelectionTests(unittest.TestCase):
    def inputs(self) -> tuple[Any, Any, Any, Any, Any]:
        from haic.algorithms.joint_control.physics import PhysicsState
        snapshot = dict(valid=True, state=PhysicsState(forward_speed=10.), mapping_motion=motion())
        proposal = np.array([.1, 0., .2], np.float32)
        info = dict(pixel_speed=60.)
        shield = dict(active=False, baseline_threat=False, rearm_blocked=False,
                      reason='baseline_clear', projection_speed=60.)
        witness = dict(pre_wheel=0., pre_recovery=False, captures=1, brake_history=True,
                       encounter_open=False, box_threatened=False)
        return snapshot, proposal, info, shield, witness

    def test_all_old_causal_preconditions_preserved(self):
        args = self.inputs()
        self.assertEqual(pairs.eligibility(*args, []), [])
        for index, key, value in ((0, 'valid', False), (2, 'pixel_speed', 81.),
                (3, 'active', True), (3, 'baseline_threat', True), (3, 'rearm_blocked', True),
                (3, 'projection_speed', None), (3, 'reason', 'unsupported'),
                (4, 'encounter_open', True), (4, 'box_threatened', True),
                (4, 'pre_recovery', True), (4, 'captures', 0), (4, 'brake_history', False)):
            changed = deepcopy(args)
            changed[index][key] = value
            with self.subTest(key=key):
                self.assertTrue(pairs.eligibility(*changed, []))
        for action in ([.36, 0., .2], [.1, .11, .2], [.1, 0., .01]):
            changed = list(deepcopy(args))
            changed[1] = np.asarray(action, np.float32)
            self.assertIn('action_outside_pilot_envelope',
                          pairs.eligibility(changed[0], changed[1], changed[2], changed[3], changed[4], []))

    def test_mapping_is_detached_latest_only_older_link_break(self):
        snapshots = [dict(mapping_motion=motion()), dict(mapping_motion=motion('unknown')),
                     dict(mapping_motion=motion())]
        original = deepcopy(snapshots)
        motions, reasons = pairs.calibrated_motions(snapshots, calibration()['absolute_calibration'])
        self.assertEqual(reasons, [])
        self.assertFalse(motions[0]['valid'])
        self.assertTrue(motions[1]['valid'])
        self.assertEqual(motions[1]['position_uncertainty'], .5)
        self.assertEqual(snapshots, original)
        _, reasons = pairs.calibrated_motions(snapshots[:-1], calibration()['absolute_calibration'])
        self.assertTrue(reasons)

    def test_selector_has_no_physical_label_argument(self):
        self.assertEqual(list(inspect.signature(pairs.PrefixSelector.__init__).parameters),
                         ['self', 'model', 'calibration', 'observation'])
        source = inspect.getsource(pairs.PrefixSelector)
        for forbidden in ('capture.', 'environment.', 'training.labels', 'actual_trajectory', 'physical_labels'):
            self.assertNotIn(forbidden, source)

    def selector(self):
        from haic.algorithms.joint_control.physics import PhysicsState
        selector: Any = object.__new__(pairs.PrefixSelector)
        snapshot = self.inputs()[0]
        hypotheses = PhysicsState(forward_speed=np.full(19, 10.), wheel_omega=np.zeros((19, 4)))
        selector.model, selector.calibration, selector.calls = object(), calibration(), 18
        selector.frames = deque_fixture = pairs.deque([np.zeros((84, 84))] * 4, maxlen=4)
        self.assertEqual(len(deque_fixture), 4)
        selector.snapshots = pairs.deque([deepcopy(snapshot) for _ in range(4)], maxlen=4)
        selector.observer = SimpleNamespace(snapshot=lambda: deepcopy(snapshot), shared_hypotheses=lambda: hypotheses)
        return selector

    def test_anchor_does_not_filter_winner_or_absolute_veto(self):
        selector = self.selector()
        _, action, info, shield, witness = self.inputs()
        result = comparison(upper=.3, veto=True)
        prediction = {key: np.zeros((2, 19, 16)) for key in ('relative_x', 'relative_y', 'yaw_delta', 'speed')}
        with patch.object(pairs, 'query_champion', return_value=(action, info, shield, witness)), \
                patch('haic.algorithms.joint_control.interval_comparison.extract_scene', return_value={'masks': 'observed'}), \
                patch('haic.algorithms.joint_control.interval_comparison.compare_candidates', return_value=deepcopy(result)) as old, \
                patch('haic.algorithms.joint_control.paired_residual.compare_candidates', return_value=deepcopy(result)) as new, \
                patch.object(pairs, 'physical_predictions', return_value=prediction):
            _, anchor, note = selector.propose(18, np.zeros((4, 84, 84), np.float32))
        self.assertIsNotNone(anchor)
        assert anchor is not None
        self.assertTrue(note['common_full_h4_cost_support'])
        self.assertFalse(pairs.predicted_selects(anchor['corrected']))
        self.assertEqual(old.call_count, 1)
        self.assertEqual(new.call_count, 1)
        self.assertIs(old.call_args.args[1], new.call_args.args[1])
        self.assertEqual(old.call_args.kwargs['paired_cost_residual'], .44961874671412616)

    def test_unsupported_full_h4_is_not_anchor_even_if_negative(self):
        selector = self.selector()
        _, action, info, shield, witness = self.inputs()
        result = comparison(common=False, upper=-1., veto=False)
        with patch.object(pairs, 'query_champion', return_value=(action, info, shield, witness)), \
                patch('haic.algorithms.joint_control.interval_comparison.extract_scene', return_value={}), \
                patch('haic.algorithms.joint_control.interval_comparison.compare_candidates', return_value=deepcopy(result)), \
                patch('haic.algorithms.joint_control.paired_residual.compare_candidates', return_value=deepcopy(result)), \
                patch.object(pairs, 'physical_predictions', side_effect=AssertionError('not an anchor')):
            _, anchor, _ = selector.propose(18, np.zeros((4, 84, 84), np.float32))
        self.assertIsNone(anchor)

    def test_post_anchor_query_is_refused(self):
        selector = self.selector()
        with self.assertRaisesRegex(ValueError, 'nonsequential'):
            selector.propose(19, np.zeros((4, 84, 84)))

    def test_reset_observation_is_consumed_exactly_once(self):
        observer, model = Mock(), Mock()
        observer.reset.return_value = self.inputs()[0]
        image = np.zeros((4, 84, 84), np.float32)
        with patch('haic.algorithms.joint_control.observer.TemporalObserver', return_value=observer):
            selector = pairs.PrefixSelector(model, calibration(), image)
        observer.reset.assert_called_once()
        observer.observe.assert_not_called()
        observer.commit_action.assert_not_called()
        model.reset.assert_called_once()
        self.assertEqual(len(selector.frames), 1)


@contextmanager
def frozen_agent():
    """Load exact eleven-member champion in an isolated Python namespace only."""
    before = pairs.champion_members()
    directory = pairs.BUNDLE / 'source'
    names = [n for n in sys.modules if n == 'haic_agent' or n.startswith('haic_agent.')]
    saved = {n: sys.modules.pop(n) for n in names}
    path, bytecode = sys.path[:], sys.dont_write_bytecode
    sys.path.insert(0, str(directory))
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location('_envelope_test_champion', directory / 'agent.py')
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module.Agent
    finally:
        sys.path[:] = path
        sys.dont_write_bytecode = bytecode
        for name in list(sys.modules):
            if name == 'haic_agent' or name.startswith('haic_agent.'):
                del sys.modules[name]
        sys.modules.update(saved)
        if pairs.champion_members() != before:
            raise AssertionError('exact frozen source changed')


def state_fingerprint(value):
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.tobytes()
    if is_dataclass(value) and not isinstance(value, type):
        return state_fingerprint(asdict(value))
    if isinstance(value, dict):
        return {k: state_fingerprint(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, pairs.deque)):
        return [state_fingerprint(v) for v in value]
    if hasattr(value, '__dict__'):
        return state_fingerprint(vars(value))
    if isinstance(value, float) and np.isnan(value):
        return 'NAN'
    return value


class FrozenQueryTests(unittest.TestCase):
    def image(self, index):
        frame = np.full((84, 84), .1, np.float32)
        center = 41 + index % 4
        frame[8:73, center - 12:center + 13] = .4
        frame[74:] = 0.
        frame[77:83, 10:13] = ((50 + index) * .085 + .27) / 18
        if index % 3 == 0:
            frame[42:46, 64:68] = .9
        return np.repeat(frame[None], 4, axis=0)

    def test_query_is_one_exact_champion_call_and_complete_state_noop(self):
        with frozen_agent() as Agent:
            reference, captured = Agent(), Agent()
            reference.reset(self.image(0))
            captured.reset(self.image(0))
            for index in range(6):
                image = self.image(index)
                original_act = captured.act
                calls = []

                def once(observation):
                    calls.append(True)
                    return original_act(observation)

                captured.act = once
                action, _, _, _ = pairs.query_champion(captured, image)
                del captured.act
                expected = reference.act(image)
                self.assertEqual(len(calls), 1)
                np.testing.assert_array_equal(action, expected)
                self.assertEqual(state_fingerprint(captured), state_fingerprint(reference))
                self.assertNotIn('act', vars(captured.driver))

    def test_private_capture_restored_on_champion_exception(self):
        with frozen_agent() as Agent:
            champion = Agent()
            champion.reset(self.image(0))
            before = champion.driver.act.__func__.__globals__['advance_boxes']
            champion.act = Mock(side_effect=RuntimeError('synthetic query error'))
            with self.assertRaisesRegex(RuntimeError, 'synthetic query error'):
                pairs.query_champion(champion, self.image(1))
            champion.act.assert_called_once()
            self.assertNotIn('act', vars(champion.driver))
            self.assertIs(champion.driver.act.__func__.__globals__['advance_boxes'], before)


class FakeRaw:
    def __init__(self, owner, fail_tick=None):
        self.owner, self.fail_tick, self.ticks, self.resets, self.t = owner, fail_tick, 0, 0, 0.

    def reset(self):
        self.resets += 1
        self.step(None)

    def step(self, action):
        self.ticks += 1
        if self.ticks == self.fail_tick:
            raise RuntimeError('synthetic raw partial')
        self.t += .02
        return self.owner.image, 0., False, False, {}


class FakeEnvironment:
    def __init__(self, cap, fail_tick=None):
        self.cap = cap
        self.image = np.zeros((4, 84, 84), np.float32)
        self.environment = SimpleNamespace(stack_state=self.image)
        self.unwrapped = FakeRaw(self, fail_tick)
        self.closed = False

    def reset(self):
        self.unwrapped.reset()
        for _ in range(50):
            self.unwrapped.step(np.zeros(3, np.float32))
        return self.image, {}

    def step(self, action):
        for _ in range(4):
            self.unwrapped.step(action)
        return self.image, 0., False, False, {}

    def close(self):
        self.closed = True


class FakeCapture:
    def __init__(self, environment, np_module):
        self.env = environment
        self.static = dict(track=[[0., 0., 0., 0.]], bodies={})

    def dynamic(self):
        return dict(t=self.env.unwrapped.t, time_limit=dict(maximum=50 + 4 * self.env.cap + 200))

    def full(self):
        return self.dynamic()


class CaptureBudgetTests(unittest.TestCase):
    def setup_output(self, root):
        _, Journal, BlockBudget, empty_counters, _ = pairs.capture_tools()
        pairs.save(root / 'counters.json', empty_counters(pairs.schedule()))
        historical = root / 'historical.json'
        pairs.save(historical, dict(initial_observation_sha256=hashlib.sha256(np.zeros((4, 84, 84), np.float32).tobytes()).hexdigest(),
                                    decision_trace='NEVER_READ_OR_MATCH_OLD_ACTIONS'))
        pairs.save(root / 'claim.json', dict(references=[dict(track_id=1, seed=3184000003, episode_path=str(historical))]))
        pairs.save(root / 'admission.json', {})
        pairs.save(root / 'calibration.json', calibration())
        return Journal, BlockBudget, empty_counters

    def run_arm(self, root, arm, found: int | None = 18, fail_tick=None):
        Journal, BlockBudget, empty_counters = self.tools
        environments, caps = [], []

        def factory(**kwargs):
            caps.append(kwargs['max_decisions'])
            value = FakeEnvironment(kwargs['max_decisions'], fail_tick)
            environments.append(value)
            return value

        with ExitStack() as stack:
            stack.enter_context(patch.object(pairs, 'capture_tools', return_value=(FakeCapture, Journal, BlockBudget,
                empty_counters, lambda *args: {'catalog': 'checked'})))
            for name in ('validate_frozen', 'validate_admission', 'verify_imports'):
                stack.enter_context(patch.object(pairs, name, return_value={}))
            stack.enter_context(patch.object(pairs, 'resource_admission', return_value={}))
            selector = stack.enter_context(patch.object(pairs, 'PrefixSelector', side_effect=lambda *args: FakeSelector(found)))
            slot = next(s for s in pairs.schedule() if s['track_id'] == 1 and s['arm'] == arm)
            result = pairs.collect_arm(root, dict(calibration_sha256='cal', source_sha256={}), slot, 'protocol',
                time.monotonic() + 60, np, factory, object() if arm == 'baseline' else None)
        self.assertEqual(caps, [64])
        self.assertEqual(selector.call_count, int(arm == 'baseline'))
        self.assertEqual(environments[0].unwrapped.resets, 1)
        self.assertTrue(environments[0].closed)
        return result

    def test_live_synthetic_capture_same_cap_full_repeat_and_anchor_prefix_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.tools = self.setup_output(root)
            for arm in pairs.ARMS:
                self.run_arm(root, arm)
            arms = [pairs.load_arm(root / f'1-3184000003-{arm}', 'protocol') for arm in pairs.ARMS]
            parity = pairs.pair_parity(*arms)
            self.assertTrue(parity['passed'])
            self.assertEqual(parity['prefix_raw_including_warmup'], 51 + 18 * 4)
            self.assertEqual(arms[0]['endpoints'][0]['state']['time_limit']['maximum'], 506)
            counters = pairs.read_json(root / 'counters.json')['totals']
            self.assertEqual(counters['resets_started'], 3)
            self.assertEqual(counters['decisions_completed'], 3 * 22)
            self.assertEqual(counters['raw_completed'], 3 * (51 + 4 * 22))
            self.assertEqual(arms[1]['state']['agent_calls'], 0)
            self.assertEqual(arms[2]['state']['agent_calls'], 0)

    def test_partial_reserves_whole_raw_hold_and_never_replays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.tools = self.setup_output(root)
            with self.assertRaisesRegex(RuntimeError, 'synthetic raw partial'):
                self.run_arm(root, 'baseline', fail_tick=54)
            counters = pairs.read_json(root / 'counters.json')
            self.assertEqual(counters['totals']['raw_started'], 55)
            self.assertEqual(counters['totals']['raw_actual_started'], 54)
            self.assertEqual(counters['totals']['raw_completed'], 53)
            arm = root / '1-3184000003-baseline'
            self.assertEqual(pairs.read_json(arm / 'receipt.json')['status'], 'PARTIAL')
            self.assertTrue((arm / 'raw.jsonl').read_text())
            with self.assertRaises(FileExistsError):
                self.run_arm(root, 'baseline')
            self.assertEqual(pairs.read_json(root / 'counters.json'), counters)

    def test_capture_not_found_uses_only_sixty_holds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.tools = self.setup_output(root)
            result = self.run_arm(root, 'baseline', found=None)
            self.assertEqual(result['status'], 'NOT_FOUND')
            self.assertEqual(result['counters']['decisions_completed'], 60)
            self.assertEqual(result['counters']['raw_completed'], 291)
            self.assertFalse((root / '1-3184000003-baseline/anchor.json').exists())

    def test_full_last_suffix_terminal_complete_but_earlier_terminal_partial(self):
        original_step = FakeEnvironment.step
        for final, tick in ((True, 51 + 4 * 22), (False, 51 + 4 * 21)):
            with self.subTest(final=final), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.tools = self.setup_output(root)

                def terminal_step(environment, action):
                    obs, reward, _, truncated, info = original_step(environment, action)
                    return obs, reward, environment.unwrapped.ticks == tick, truncated, info

                with patch.object(FakeEnvironment, 'step', terminal_step):
                    if final:
                        result = self.run_arm(root, 'baseline')
                        self.assertEqual(result['status'], 'COMPLETE')
                        self.assertEqual(result['counters']['decisions_completed'], 22)
                    else:
                        with self.assertRaisesRegex(ValueError, 'terminal prefix/suffix'):
                            self.run_arm(root, 'baseline')
                        receipt = pairs.read_json(root / '1-3184000003-baseline/receipt.json')
                        self.assertEqual(receipt['status'], 'PARTIAL')

    def test_reset_retry_and_overbudget_refused_before_delegate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, Budget, empty, _ = pairs.capture_tools()
            slot = dict(pairs.schedule()[0], decisions=0, raw_ticks=51)
            pairs.save(root / 'counters.json', empty([slot]))
            budget = Budget(root, slot, time.monotonic() + 60)
            budget.reserve(warmup=True)
            budget.begin('resets')
            for _ in range(51):
                budget.begin('raw')
                budget.complete('raw')
            budget.complete('resets')
            with self.assertRaisesRegex(RuntimeError, 'retry'):
                budget.begin('resets')
            budget.seal()
            with self.assertRaisesRegex(RuntimeError, 'budget'):
                budget.reserve()
            with self.assertRaisesRegex(RuntimeError, 'previously attempted'):
                Budget(root, slot, time.monotonic() + 60)


class AnalysisTests(unittest.TestCase):
    def row(self, **changes) -> dict[str, Any]:
        return dict(dict(cell=[1, 3], status='COMPLETE', parity=dict(passed=True), new_would_select=True,
            actual_cost_supported=True, actual_cost_interval=[-.4, -.1], actual_safe=True, contained=True), **changes)

    def test_gate_exact_denominators_unknown_not_zero_wins(self):
        rows = [self.row(), self.row(cell=[2, 4], new_would_select=False, actual_cost_supported=False,
            actual_cost_interval=None, actual_safe=None, contained=None), dict(cell=[3, 5], status='NOT_FOUND')]
        result = pairs.potential_pilot_gate(rows)
        self.assertEqual(result['outcome'], 'PASS')
        self.assertEqual(result['allocated_pairs'], 3)
        self.assertEqual(result['complete_pairs'], 2)
        self.assertEqual(result['selected_pairs'], 1)
        self.assertEqual(result['supported_selected_benefit'], dict(numerator=1, denominator=1, rate=1.))

    def test_selected_false_benefit_harm_or_tube_miss_blocks(self):
        for change in (dict(actual_cost_interval=[-.2, -.05]), dict(actual_safe=False), dict(contained=False)):
            result = pairs.potential_pilot_gate([self.row(), self.row(cell=[2, 4], **change)])
            self.assertEqual(result['outcome'], 'BLOCKED')
            self.assertEqual(result['blockers'], [[2, 4]])
            self.assertFalse(result['potential_full_pilot'])

    def test_selected_missing_actual_cost_or_partial_is_na(self):
        for row in (self.row(actual_cost_supported=False, actual_cost_interval=None), dict(status='PARTIAL')):
            result = pairs.potential_pilot_gate([row])
            self.assertEqual(result['outcome'], 'N/A')
            self.assertIsNone(result['supported_selected_benefit']['numerator'])
        self.assertEqual(pairs.potential_pilot_gate([self.row(), dict(status='PARTIAL')])['outcome'], 'N/A')

    def test_unselected_unsupported_other_region_does_not_block(self):
        other = self.row(cell=[2, 4], new_would_select=False, actual_cost_supported=False,
                         actual_cost_interval=None, actual_safe=False, contained=False)
        self.assertEqual(pairs.potential_pilot_gate([self.row(), other])['outcome'], 'PASS')

    def test_selectable_requires_both_cost_and_unchanged_absolute_guard(self):
        c = pairs.jsonable(comparison(upper=-.1, veto=False))
        self.assertTrue(pairs.predicted_selects(c))
        for key, value in (('common_support', False), ('delta_interval', [[0., 0.], [-1., -.05]]),
                           ('cost_supported', [True, False]), ('veto', [[False] * 19, [True] * 19]),
                           ('absolute_supported', [[True] * 19, [False] * 19])):
            self.assertFalse(pairs.predicted_selects(dict(c, **{key: value})))

    def test_containment_requires_same_scenario_whole_path(self):
        actual = np.zeros((17, 3))
        prediction = np.full((19, 17, 3), 10.)
        prediction[0, :9] = 0.
        prediction[1, 9:] = 0.
        bound = [0.] + [.5] * 16
        self.assertFalse(pairs.containment(actual, prediction, bound, bound)['contained'])
        prediction[2] = 0.
        result = pairs.containment(actual, prediction, bound, bound)
        self.assertTrue(result['contained'])
        self.assertEqual(result['whole_path_scenarios'], [2])

    def test_safety_labels_distinguish_partial_and_total_wheel_loss(self):
        state: dict[str, Any] = dict(wheels=[dict(tiles=[0])] * 4, damage_telemetry_valid=True,
                     damage=0., collision=False, contacts=[])
        end = deepcopy(state)
        end['wheels'][0] = dict(tiles=[])
        result = pairs.physical_labels(dict(states=[state, end], boundary_states=[state, end]))
        self.assertTrue(result['partial_wheel_loss'])
        self.assertTrue(result['safe'])
        end['wheels'] = [dict(tiles=[])] * 4
        result = pairs.physical_labels(dict(states=[state, end], boundary_states=[state, end]))
        self.assertTrue(result['all_wheels_lost'])
        self.assertFalse(result['safe'])

    def test_safety_harm_is_actual_not_predicted_label(self):
        state: dict[str, Any] = dict(wheels=[dict(tiles=[0])] * 4, damage_telemetry_valid=True,
                     damage=0., collision=False, contacts=[])
        changed = deepcopy(state)
        changed['contacts'] = [dict(touching=True, a=['obstacle', 0], b=['wheel', 0])]
        self.assertFalse(pairs.physical_labels(dict(states=[state, changed], boundary_states=[state, changed]))['safe'])
        changed['contacts'], changed['damage'] = [], .1
        self.assertFalse(pairs.physical_labels(dict(states=[state, changed], boundary_states=[state, changed]))['safe'])

    def test_final_boundary_damage_not_yet_visible_in_raw_posts(self):
        state: dict[str, Any] = dict(wheels=[dict(tiles=[0])] * 4, damage_telemetry_valid=True,
                     damage=0., collision=False, contacts=[])
        last_boundary = dict(state, damage=.4)
        trajectory = dict(states=[deepcopy(state) for _ in range(17)], boundary_states=[state, last_boundary])
        labels = pairs.physical_labels(trajectory)
        self.assertEqual(labels['damage_increase'], .4)
        self.assertFalse(labels['safe'])
        last_boundary['damage_telemetry_valid'] = False
        with self.assertRaisesRegex(ValueError, 'damage labels'):
            pairs.physical_labels(trajectory)

    def complete_pair(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        from haic.algorithms.joint_control.interval_comparison import extract_scene
        _, col = np.indices((84, 84))
        frame = np.where((col >= 31) & (col <= 53), .4, .65)
        scene = extract_scene(np.stack([frame] * 4), motions=[dict(valid=True, right=0.,
            forward=6., yaw_delta=0., residual_p90=0.)] * 3)
        actions = pairs.joint_actions(np.array([.1, 0., .2], np.float32))
        n = 18
        start_raw = 51 + 4 * n
        static = dict(track=[[0., 0., 0., 0.], [0., 0., 0., 100.],
                             [0., 0., 100., 100.], [0., 0., 100., 0.]])
        anchor: dict[str, Any] = dict(completed_prefix=n, prefix_actions=[actions[0].tolist()] * n,
            actions=actions.tolist(), actions_float32_hex=actions.tobytes().hex(), scene=pairs.jsonable(scene),
            original=pairs.jsonable(comparison(upper=.3, veto=False)),
            corrected=pairs.jsonable(comparison(upper=-.1, veto=False)),
            physical_predictions=dict(speed=np.full((2, 19, 16), 10.).tolist()))

        def physical(tick, speed):
            forward = max(0, tick - start_raw) * speed * .02
            return dict(t=tick * .02, hull=dict(position=[0., forward], angle=0., linearVelocity=[0., speed]),
                wheels=[dict(tiles=[0])] * 4, contacts=[], collision=False, damage=0., damage_telemetry_valid=True)

        def arm(name, speed) -> dict[str, Any]:
            issued = [actions[0].tolist()] * n + pairs.tail_actions(actions, name).tolist()
            raw = []
            for tick in range(1, start_raw + 17):
                action = None if tick == 1 else [0., 0., 0.] if tick <= 51 else issued[(tick - 52) // 4]
                # Prefix physical states match; only the intervention suffix differs.
                raw_speed = speed if tick > start_raw else 10.
                state = physical(tick, raw_speed)
                raw.extend([dict(event='PRE', action=action), dict(event='POST', action=action, reward=.1, state=state)])
            endpoints = [dict(state=deepcopy(raw[2 * (51 + 4 * i - 1) + 1]['state'])) for i in range(n + 5)]
            return dict(static=deepcopy(static), raw=raw, endpoints=endpoints,
                        state=dict(status='COMPLETE', anchor=deepcopy(anchor), actions=issued))

        baseline, alternative = arm('baseline', 12.5), arm('alternative', 15.)
        poses = np.stack([pairs.actual_trajectory(a, anchor)['poses'] for a in (baseline, alternative)])
        saved_poses = np.repeat(poses[:, None], 19, axis=1).tolist()
        for a in (baseline, alternative):
            a['state']['anchor']['original']['poses'] = deepcopy(saved_poses)
            a['state']['anchor']['corrected']['poses'] = deepcopy(saved_poses)
        return baseline, deepcopy(baseline), alternative

    def test_full_analysis_uses_saved_scene_five_refs_and_exact_raw_h4(self):
        from haic.algorithms.joint_control import interval_comparison
        base, repeat, alternative = self.complete_pair()
        # Extracted masks deliberately contain missing centers outside visibility.
        self.assertTrue(any(v is None for v in base['state']['anchor']['scene']['center_x']))
        with patch.object(interval_comparison, 'score_trajectories', wraps=interval_comparison.score_trajectories) as scorer:
            result = pairs.analyze_pair(base, repeat, alternative, calibration())
        self.assertEqual(scorer.call_count, 1)
        assert scorer.call_args is not None
        self.assertEqual(scorer.call_args.args[1].shape, (2, 17, 3))
        self.assertEqual(scorer.call_args.args[2].shape, (2, 16, 3))
        np.testing.assert_array_equal(scorer.call_args.args[0].road, base['state']['anchor']['scene']['road'])
        self.assertTrue(result['actual_cost_supported'])
        self.assertEqual(len(result['actual_reference_deltas']), 5)
        delta = np.asarray(result['actual_score']['reference_costs'])[1] - np.asarray(result['actual_score']['reference_costs'])[0]
        np.testing.assert_allclose(result['actual_reference_deltas'], delta, rtol=0, atol=0)
        self.assertEqual(result['actual_cost_interval'], [float(delta.min()), float(delta.max())])
        self.assertTrue(result['contained'])
        self.assertTrue(result['actual_safe'])
        self.assertAlmostEqual(result['effects']['1']['actual'][1], .2)
        self.assertAlmostEqual(result['effects']['4']['actual'][1], .8)
        self.assertEqual(result['original_predicted_order'], 'ambiguous')
        self.assertEqual(result['new_predicted_order'], 'alternative')
        self.assertFalse(result['new_data_fit'])
        self.assertIn('qualification', result['privileged_gt_proxies'][0])

    def test_full_analysis_rejects_future_control_mismatch_not_range_miss(self):
        base, repeat, alternative = self.complete_pair()
        alternative['state']['actions'][-1] = [0., 0., 0.]
        with self.assertRaisesRegex(ValueError, 'continuation mismatch'):
            pairs.analyze_pair(base, repeat, alternative, calibration())

    def test_full_analysis_retains_last_boundary_damage_and_invalidity(self):
        base, repeat, alternative = self.complete_pair()
        alternative['endpoints'][-1]['terminated'] = True
        alternative['endpoints'][-1]['state']['damage'] = .4
        result = pairs.analyze_pair(base, repeat, alternative, calibration())
        self.assertTrue(result['contained'])
        self.assertFalse(result['actual_safe'])
        self.assertEqual(result['physical_labels'][1]['damage_increase'], .4)
        alternative['endpoints'][-1]['state']['damage_telemetry_valid'] = False
        with self.assertRaisesRegex(ValueError, 'damage labels'):
            pairs.analyze_pair(base, repeat, alternative, calibration())


class ProvenanceTests(unittest.TestCase):
    def test_execute_skips_notfound_mates_and_refuses_second_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model').mkdir()
            pairs.save(root / 'preflight.json', {})
            admission = root / 'input-admission.json'
            pairs.save(admission, dict(expires_unix_s=time.time() + pairs.LIMITS['total_seconds']))
            p = dict(claim_sha256='claim', calibration_sha256='cal')
            with patch.object(pairs, 'validate_frozen', return_value=p), \
                    patch.object(pairs, 'validate_preflight'), patch.object(pairs, 'validate_admission'), \
                    patch.object(pairs, 'run_json_child', return_value={}), \
                    patch.object(pairs, 'load_arm', return_value=dict(state=dict(status='NOT_FOUND'), receipt_sha256='receipt')), \
                    patch.object(pairs.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as child:
                report: dict[str, Any] = pairs.execute(root, 'protocol', admission)
                self.assertIsNone(report['operator_error'])
                self.assertEqual([r['status'] for r in report['rows']], ['NOT_FOUND', 'SKIPPED_NOT_FOUND', 'SKIPPED_NOT_FOUND'] * 3)
                self.assertEqual(child.call_count, 3)
                self.assertEqual([c.args[0][-2] for c in child.call_args_list], ['0', '3', '6'])
                with self.assertRaisesRegex(ValueError, 'never restart'):
                    pairs.execute(root, 'protocol', admission)
                self.assertEqual(child.call_count, 3)

    def test_exclusive_save_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'receipt.json'
            pairs.save(path, {'before': 1})
            with self.assertRaises(FileExistsError):
                pairs.save(path, {'after': 2})
            self.assertEqual(pairs.read_json(path), {'before': 1})

    def test_targeted_closure_excludes_root_simulator_and_does_not_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, content in {'scripts/main.py': 'from haic import thing\nfrom training import env_factory\n',
                    'scripts/__init__.py': '', 'haic/__init__.py': '', 'haic/thing.py': 'from . import leaf\n',
                    'haic/leaf.py': 'raise RuntimeError("must not execute")\n',
                    'training/env_factory.py': 'raise RuntimeError("simulator")\n'}.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            pins = pairs.source_closure(root, ['scripts/main.py'], packages=('scripts', 'haic'))
            self.assertEqual(set(pins), {'scripts/main.py', 'scripts/__init__.py', 'haic/__init__.py',
                                         'haic/thing.py', 'haic/leaf.py'})

    def test_no_old_template_validate_collect_or_successor_act_calls(self):
        source = inspect.getsource(pairs)
        tree = ast.parse(source)
        forbidden = {'protocol_template', 'validate_protocol', 'pinned_inputs', 'TemporalSuccessor'}
        names = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
                 for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
        self.assertFalse(names & forbidden)
        self.assertNotIn('getrusage(', source)
        self.assertNotIn('historical[\'decision_trace\']', source)

    def test_strict_external_stack_no_replay_agent_instance_branch(self):
        source = inspect.getsource(pairs.load_runtime)
        self.assertIn("'fail_on_invalid_action'", source)
        self.assertIn("if arm == 'baseline':", source)
        self.assertEqual(source.count('entry.Agent()'), 1)
        self.assertLess(source.index("import_module('agent')"), source.index("sys.path.insert(0, str(SNAPSHOT))"))

    def test_worker_command_pins_cpu21_isolated_threads(self):
        command = pairs.worker_command('/tmp/kilo/test', 'digest', 2, 'preflight')
        self.assertEqual(command[:3], [str(pairs.PYTHON), '-I', '-B'])
        self.assertEqual(command[-2:], ['2', 'preflight'])
        self.assertEqual(pairs.THREAD_ENV['OMP_NUM_THREADS'], '1')
        self.assertEqual(pairs.THREAD_ENV['CUDA_VISIBLE_DEVICES'], '')

    def test_admission_exact_binding_time_and_total_cap(self):
        p = dict(claim_sha256='claim', calibration_sha256='cal')
        value = dict(schema='haic-joint-envelope-pairs-admission-v1', admitted=True,
            protocol_sha256='protocol', claim_sha256='claim', calibration_sha256='cal',
            created_unix_s=100., expires_unix_s=2500.)
        pairs.validate_admission(value, 'protocol', p, starting=True, now=101.)
        for change, now in ((dict(expires_unix_s=2501.), 101.), (dict(admitted=False), 101.),
                            ({}, 401.), ({}, 2500.), (dict(created_unix_s=float('nan')), 101.)):
            with self.assertRaises(ValueError):
                pairs.validate_admission(dict(value, **change), 'protocol', p, starting=True, now=now)

    def test_preflight_requires_all_three_zero_reset_arms_and_no_replay_agent(self):
        p = dict(cpu_affinity=[0])
        value: dict[str, Any] = dict(protocol_sha256='protocol', environment_resets=0, arms=[dict(arm=arm,
            import_only=True, environment_resets=0, agent_instances=int(arm == 'baseline'), cpu_affinity=[0],
            evaluator_file=str(pairs.SNAPSHOT / 'training/evaluate_closed_loop.py')) for arm in pairs.ARMS])
        pairs.validate_preflight(value, 'protocol', p)
        value['arms'][1]['agent_instances'] = 1
        with self.assertRaises(ValueError):
            pairs.validate_preflight(value, 'protocol', p)

    def test_calibration_requires_prior_exact_copy_and_targeted_pins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pins = {}
            for name in ('hud', 'motion', 'physics', 'observer', 'comparison', 'interval_comparison', 'paired_residual'):
                path = root / 'haic/algorithms/joint_control' / (name + '.py')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# synthetic source\n')
                pins[str(path)] = pairs.sha(path)
            c = calibration()
            c['absolute_calibration']['source_pins'] = {k: v for k, v in pins.items() if not k.endswith('paired_residual.py')}
            c['source_pins'] = pins
            prior = root / 'runs/joint-temporal-interval-v1/calibration.json'
            prior.parent.mkdir(parents=True)
            pairs.save(prior, c['absolute_calibration'])
            c['evidence_pins'] = {str(prior): pairs.sha(prior)}
            with patch.object(pairs, 'PRIOR_CALIBRATION_SHA', pairs.sha(prior)):
                pairs.validate_calibration(c, root)
                changed = deepcopy(c)
                changed['absolute_calibration']['position_residual'][-1] = .49
                with self.assertRaisesRegex(ValueError, 'ALL original'):
                    pairs.validate_calibration(changed, root)
                path.write_text('# changed predictor\n')
                with self.assertRaisesRegex(ValueError, 'differs'):
                    pairs.validate_calibration(c, root)

    def test_claim_binds_three_consumed_profiles_not_old_actions_or_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {}
            for name in (pairs.OPERATOR, pairs.COMPARATOR):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# synthetic pinned source\n')
                sources[str(path)] = pairs.sha(path)
            refs, evidence = [], {}
            for t, s in pairs.CELLS:
                path = root / f'{t}-{s}-old-crossing.json'
                pairs.save(path, dict(track_id=t, seed=s, mode='crossing_projection', completed=False,
                    map_id=None, obstacle_mode='official', obstacle_count=6, catalog=dict(obstacles=[{}] * 6),
                    initial_observation_sha256='pixels', decision_trace='OLD_ACTIONS_NOT_A_REFERENCE'))
                evidence[str(path)] = pairs.sha(path)
                refs.append(dict(track_id=t, seed=s, episode_path=str(path), episode_sha256=pairs.sha(path)))
            claim = dict(schema='haic-joint-envelope-pairs-consumed-train-v1', partition='TRAIN', fresh=False,
                new_paired_starts=True, cells=[list(c) for c in pairs.CELLS], max_resets=9, max_decisions=576,
                max_raw_ticks=2763, run_directory=str(root / 'pairs'), calibration_sha256='cal',
                source_sha256=sources, evidence_sha256=evidence, references=refs)
            pairs.validate_claim(claim, root / 'pairs', 'cal', root=root)
            for change in (dict(fresh=True), dict(max_resets=10), dict(calibration_sha256='other'),
                           dict(cells=[[1, 999]])):
                with self.assertRaises(ValueError):
                    pairs.validate_claim(dict(claim, **change), root / 'pairs', 'cal', root=root)


if __name__ == '__main__':
    unittest.main()
