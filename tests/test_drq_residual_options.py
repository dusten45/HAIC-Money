import unittest

import numpy as np
import torch

from common_adapter import ActionAdapter
from haic.algorithms.drq_v2 import (
    FeatureOptionReplay,
    OPTION_COUNT,
    OptionTransition,
    OptionTransitionAccumulator,
    ResidualOption,
    ResidualOptionQ,
    ResidualOptionState,
    apply_native_residual,
    double_q_targets,
    polyak_update_option_target,
    select_greedy_option,
    train_option_q_step,
)


class TestResidualActions(unittest.TestCase):
    def test_keep_is_exact_identity_and_other_options_use_native_coordinates(self):
        base = np.asarray([0.2, 0.4, -0.6], dtype=np.float32)
        keep = apply_native_residual(base, ResidualOption.KEEP)
        np.testing.assert_array_equal(keep, base)
        self.assertEqual(keep.tobytes(), base.tobytes())
        np.testing.assert_allclose(
            apply_native_residual(base, ResidualOption.STEER_MINUS),
            [0.05, 0.4, -0.6],
        )
        np.testing.assert_allclose(
            apply_native_residual(base, ResidualOption.STEER_PLUS),
            [0.35, 0.4, -0.6],
        )
        np.testing.assert_allclose(
            apply_native_residual(base, ResidualOption.COAST),
            [0.2, -1.0, -1.0],
        )
        np.testing.assert_allclose(
            apply_native_residual(base, ResidualOption.BRAKE),
            [0.2, -1.0, -0.5],
        )

    def test_actions_clip_at_native_limits_and_preserve_existing_stronger_brake(self):
        base = np.asarray([0.95, 0.8, 0.6], dtype=np.float32)
        np.testing.assert_array_equal(
            apply_native_residual(base, ResidualOption.STEER_PLUS),
            np.asarray([1.0, 0.8, 0.6], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            apply_native_residual(base, ResidualOption.BRAKE),
            np.asarray([0.95, -1.0, 0.6], dtype=np.float32),
        )

    def test_native_options_match_declared_official_action_semantics(self):
        adapter = ActionAdapter()
        base = np.asarray([0.2, 0.4, -0.6], dtype=np.float32)
        official_base = adapter.to_official(base)
        np.testing.assert_array_equal(
            adapter.to_official(apply_native_residual(base, ResidualOption.KEEP)), official_base,
        )
        np.testing.assert_allclose(
            adapter.to_official(apply_native_residual(base, ResidualOption.STEER_MINUS)),
            [official_base[0] - 0.15, official_base[1], official_base[2]],
        )
        np.testing.assert_allclose(
            adapter.to_official(apply_native_residual(base, ResidualOption.COAST)),
            [official_base[0], 0.0, 0.0],
        )
        np.testing.assert_allclose(
            adapter.to_official(apply_native_residual(base, ResidualOption.BRAKE)),
            [official_base[0], 0.0, max(official_base[2], 0.25)],
        )

    def test_invalid_action_option_and_parameters_are_rejected(self):
        for action in ([0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 1.1]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                apply_native_residual(action, ResidualOption.KEEP)
        with self.assertRaises(ValueError):
            apply_native_residual([0.0, 0.0, 0.0], 99)
        with self.assertRaises(ValueError):
            apply_native_residual([0.0, 0.0, 0.0], ResidualOption.KEEP, steering_delta=float("nan"))
        with self.assertRaises(ValueError):
            apply_native_residual([0.0, 0.0, 0.0], ResidualOption.KEEP, brake_floor_official=1.1)

    def test_greedy_selection_is_finite_and_deterministic(self):
        values = np.zeros(OPTION_COUNT, dtype=np.float32)
        self.assertIs(select_greedy_option(values), ResidualOption.KEEP)
        values[ResidualOption.COAST] = 1.0
        self.assertIs(select_greedy_option(torch.from_numpy(values)), ResidualOption.COAST)
        for invalid in ([0.0], [0.0, 0.0, 0.0, 0.0, np.inf]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_greedy_option(invalid)

    def test_intervention_margin_favors_keep_until_q_advantage_clears_threshold(self):
        values = np.asarray([10.0, 13.0, 12.0, 11.0, -5.0], dtype=np.float32)
        self.assertIs(select_greedy_option(values, intervention_margin=3.0), ResidualOption.KEEP)
        self.assertIs(select_greedy_option(values, intervention_margin=2.99), ResidualOption.STEER_MINUS)
        with self.assertRaises(ValueError):
            select_greedy_option(values, intervention_margin=-0.1)


class TestResidualOptionState(unittest.TestCase):
    def test_recomputes_base_each_decision_but_holds_selected_option(self):
        state = ResidualOptionState(duration=2)
        q_values = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        first, option = state.act([0.2, 0.0, 0.0], q_values)
        self.assertIs(option, ResidualOption.STEER_MINUS)
        np.testing.assert_allclose(first, [0.05, 0.0, 0.0])
        self.assertFalse(state.at_boundary)

        # The second base action changes, but a new Q preference cannot interrupt the hold.
        second, option = state.act([0.6, 0.2, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertIs(option, ResidualOption.STEER_MINUS)
        np.testing.assert_allclose(second, [0.45, 0.2, 0.0])
        self.assertTrue(state.at_boundary)
        with self.assertRaises(ValueError):
            state.act([0.0, 0.0, 0.0])

    def test_reset_clears_held_option_and_first_call_needs_no_prior_reset(self):
        state = ResidualOptionState()
        _, option = state.act([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertIs(option, ResidualOption.COAST)
        state.reset()
        self.assertTrue(state.at_boundary)
        _, option = state.act([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertIs(option, ResidualOption.BRAKE)

    def test_intervention_margin_and_residual_settings_are_frozen_in_state(self):
        state = ResidualOptionState(duration=1, intervention_margin=2.0, steering_delta=0.1)
        values = np.asarray([0.0, 1.9, 0.0, 0.0, 0.0], dtype=np.float32)
        action, option = state.act([0.4, 0.0, 0.0], values)
        self.assertIs(option, ResidualOption.KEEP)
        np.testing.assert_array_equal(action, np.asarray([0.4, 0.0, 0.0], dtype=np.float32))
        values[ResidualOption.STEER_MINUS] = 2.1
        action, option = state.act([0.4, 0.0, 0.0], values)
        self.assertIs(option, ResidualOption.STEER_MINUS)
        np.testing.assert_allclose(action, [0.3, 0.0, 0.0])


class TestOptionValueAndTransitions(unittest.TestCase):
    actor_hash = "a" * 64

    def test_q_head_accepts_batched_and_single_frozen_features(self):
        head = ResidualOptionQ(feature_dim=8, hidden_dim=16)
        self.assertEqual(head(torch.zeros(8), torch.zeros(3)).shape, (OPTION_COUNT,))
        self.assertEqual(head(torch.zeros(4, 8), torch.zeros(4, 3)).shape, (4, OPTION_COUNT))
        with self.assertRaises(ValueError):
            head(torch.zeros(7), torch.zeros(3))
        with self.assertRaises(ValueError):
            head(torch.zeros(2, 8), torch.zeros(3))

    def test_two_decision_transition_sums_discounted_raw_rewards(self):
        accumulator = OptionTransitionAccumulator(
            [1.0, 2.0], [0.1, -0.2, 0.3], ResidualOption.KEEP, gamma=0.9, max_duration=2,
        )
        self.assertIsNone(accumulator.append(-1.0))
        transition = accumulator.append(
            2.0, next_features=[3.0, 4.0], next_base_action=[0.0, 0.0, 0.0],
        )
        self.assertIsInstance(transition, OptionTransition)
        self.assertAlmostEqual(transition.reward, 0.8)
        self.assertEqual(transition.duration, 2)
        self.assertFalse(transition.terminal)
        self.assertAlmostEqual(0.9 ** transition.duration, 0.81)
        np.testing.assert_array_equal(transition.next_features, [3.0, 4.0])
        with self.assertRaises(RuntimeError):
            accumulator.append(0.0)

    def test_early_terminal_uses_actual_duration_and_never_bootstraps(self):
        accumulator = OptionTransitionAccumulator(
            [1.0], [0.0, 0.0, 0.0], ResidualOption.BRAKE, gamma=0.9, max_duration=2,
        )
        transition = accumulator.append(3.0, terminated=True, terminal=True)
        self.assertEqual(transition.duration, 1)
        self.assertEqual(transition.reward, 3.0)
        self.assertTrue(transition.terminal)
        self.assertIsNone(transition.next_features)
        self.assertIsNone(transition.next_base_action)

    def test_time_limit_truncation_bootstraps_only_from_final_state(self):
        accumulator = OptionTransitionAccumulator(
            [1.0], [0.0, 0.0, 0.0], ResidualOption.COAST, gamma=0.9, max_duration=2,
        )
        transition = accumulator.append(
            1.0, next_features=[2.0], next_base_action=[0.1, 0.2, 0.3], truncated=True,
        )
        self.assertTrue(transition.truncated)
        self.assertFalse(transition.terminal)
        np.testing.assert_array_equal(transition.next_features, [2.0])

        finished = OptionTransitionAccumulator(
            [1.0], [0.0, 0.0, 0.0], ResidualOption.KEEP, gamma=0.9, max_duration=2,
        ).append(1.0, truncated=True, terminal=True)
        self.assertTrue(finished.terminal)
        self.assertIsNone(finished.next_features)

    def test_incomplete_bootstrappable_transition_without_final_state_is_rejected(self):
        accumulator = OptionTransitionAccumulator(
            [1.0], [0.0, 0.0, 0.0], ResidualOption.KEEP, max_duration=1,
        )
        with self.assertRaises(ValueError):
            accumulator.append(1.0)
        with self.assertRaises(ValueError):
            OptionTransition(
                features=[1.0], base_action=[0.0, 0.0, 0.0], option=ResidualOption.KEEP,
                reward=1.0, duration=1, next_features=None, next_base_action=None,
                terminated=False, truncated=False, terminal=False,
            )

    def test_double_q_target_uses_actual_duration_and_online_argmax(self):
        targets = double_q_targets(
            reward=torch.tensor([2.8, 4.0]),
            duration=torch.tensor([2, 1]),
            terminal=torch.tensor([False, True]),
            online_next_q=torch.tensor([[0.0, 5.0, 1.0, 0.0, 0.0], [0, 1, 2, 3, 4.0]]),
            target_next_q=torch.tensor([[100.0, 5.0, 7.0, 9.0, 11.0], [1, 2, 3, 4, 5.0]]),
            gamma=0.9,
        )
        torch.testing.assert_close(targets, torch.tensor([2.8 + 0.9**2 * 5.0, 4.0]))

    def test_double_q_target_rejects_invalid_shapes_and_duration(self):
        with self.assertRaises(ValueError):
            double_q_targets(
                torch.ones(1), torch.zeros(1), torch.zeros(1), torch.zeros(1, OPTION_COUNT),
                torch.zeros(1, OPTION_COUNT),
            )
        with self.assertRaises(ValueError):
            double_q_targets(
                torch.ones(1), torch.zeros(2), torch.zeros(1), torch.zeros(1, OPTION_COUNT),
                torch.zeros(1, OPTION_COUNT),
            )
        with self.assertRaises(ValueError):
            double_q_targets(
                torch.ones(1), torch.tensor([1.5]), torch.zeros(1), torch.zeros(1, OPTION_COUNT),
                torch.zeros(1, OPTION_COUNT),
            )

    def test_feature_replay_wraps_and_restores_sample_rng(self):
        replay = FeatureOptionReplay(2, 2, self.actor_hash, seed=13)
        for value in (1.0, 2.0, 3.0):
            replay.add(OptionTransition(
                features=[value, value], base_action=[0.0, 0.0, 0.0], option=ResidualOption.KEEP,
                reward=value, duration=1, next_features=[value + 1, value + 1],
                next_base_action=[0.0, 0.0, 0.0], terminated=False, truncated=False, terminal=False,
            ))
        self.assertEqual(replay.size, 2)
        self.assertEqual(replay.cursor, 1)
        np.testing.assert_array_equal(replay.features[:2, 0], [3.0, 2.0])

        restored = FeatureOptionReplay(2, 2, self.actor_hash, seed=99)
        restored.load_state_dict(replay.state_dict())
        first = replay.sample(8)
        second = restored.sample(8)
        np.testing.assert_array_equal(first["features"], second["features"])
        np.testing.assert_array_equal(first["duration"], second["duration"])
        self.assertEqual(first["frozen_actor_sha256"], self.actor_hash)
        with self.assertRaises(ValueError):
            FeatureOptionReplay(2, 2, "b" * 64).load_state_dict(replay.state_dict())

    def test_feature_replay_rejects_invalid_contract_and_empty_sample(self):
        with self.assertRaises(ValueError):
            FeatureOptionReplay(0, 2, self.actor_hash)
        with self.assertRaises(ValueError):
            FeatureOptionReplay(2, 2, "not-a-hash")
        replay = FeatureOptionReplay(2, 2, self.actor_hash)
        with self.assertRaises(ValueError):
            replay.sample(1)
        transition = OptionTransition(
            features=[1.0, 2.0], base_action=[0.0, 0.0, 0.0], option=ResidualOption.KEEP,
            reward=1.0, duration=1, next_features=[2.0, 3.0], next_base_action=[0.0, 0.0, 0.0],
            terminated=False, truncated=False, terminal=False,
        )
        with self.assertRaises(ValueError):
            FeatureOptionReplay(2, 3, self.actor_hash).add(transition)
        replay.add(transition)
        snapshot = replay.state_dict()
        snapshot["durations"][0] = 0
        with self.assertRaises(ValueError):
            replay.load_state_dict(snapshot)

    def test_option_q_train_step_and_target_polyak_update(self):
        replay = FeatureOptionReplay(4, 2, self.actor_hash, seed=1)
        replay.add(OptionTransition(
            features=[1.0, 0.0], base_action=[0.0, 0.0, 0.0], option=ResidualOption.KEEP,
            reward=1.0, duration=2, next_features=[0.0, 1.0], next_base_action=[0.1, 0.0, 0.0],
            terminated=False, truncated=False, terminal=False,
        ))
        replay.add(OptionTransition(
            features=[0.0, 1.0], base_action=[0.1, 0.0, 0.0], option=ResidualOption.BRAKE,
            reward=-1.0, duration=1, next_features=None, next_base_action=None,
            terminated=True, truncated=False, terminal=True,
        ))
        online = ResidualOptionQ(feature_dim=2, hidden_dim=8)
        target = ResidualOptionQ(feature_dim=2, hidden_dim=8)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
        before = [parameter.detach().clone() for parameter in online.parameters()]
        metrics = train_option_q_step(
            online, target, optimizer, replay.sample(4), frozen_actor_sha256=self.actor_hash,
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, online.parameters())))

        with torch.no_grad():
            for parameter in online.parameters():
                parameter.add_(1.0)
        target_before = [parameter.detach().clone() for parameter in target.parameters()]
        online_now = [parameter.detach().clone() for parameter in online.parameters()]
        polyak_update_option_target(target, online, tau=0.5)
        for old, new, source in zip(target_before, target.parameters(), online_now):
            torch.testing.assert_close(new, old + 0.5 * (source - old))
        with self.assertRaises(ValueError):
            train_option_q_step(online, target, optimizer, replay.sample(2), frozen_actor_sha256="b" * 64)


if __name__ == "__main__":
    unittest.main()
