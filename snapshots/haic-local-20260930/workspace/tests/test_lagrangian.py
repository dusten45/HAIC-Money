import importlib
import importlib.util
import math
import unittest


def _implementation():
    if importlib.util.find_spec("training.lagrangian") is None:
        return None
    return importlib.import_module("training.lagrangian")


class LagrangianCostTests(unittest.TestCase):
    def setUp(self):
        self.api = _implementation()
        self.assertIsNotNone(
            self.api,
            "training.lagrangian must provide the TRAIN-only cost helpers",
        )

    def test_collision_and_damage_increment_are_counted_once(self):
        boundary = self.api.EpisodeBoundary()
        cost = self.api.compute_transition_cost(
            damage_before=0.0,
            damage_after=0.2,
            collision=True,
            off_track=False,
            finished=False,
            boundary=boundary,
        )

        self.assertAlmostEqual(cost.contact_cost, 0.2)
        self.assertAlmostEqual(cost.damage_stock_cost, 0.2 / 3000.0)
        self.assertEqual(cost.off_track_cost, 0.0)
        self.assertEqual(cost.failure_cost, 0.0)
        self.assertAlmostEqual(cost.total_cost, 0.2 + 0.2 / 3000.0)
        self.assertAlmostEqual(30.0 * cost.contact_cost, 6.0)

    def test_damage_increase_is_positive_cost_without_collision_label(self):
        cost = self.api.compute_transition_cost(
            damage_before=0.2,
            damage_after=0.4,
            collision=False,
            off_track=False,
            finished=False,
            boundary=self.api.EpisodeBoundary(),
        )

        self.assertAlmostEqual(cost.contact_cost, 0.2)

    def test_damage_decrease_does_not_create_negative_contact_cost(self):
        cost = self.api.compute_transition_cost(
            damage_before=0.4,
            damage_after=0.2,
            collision=False,
            off_track=False,
            finished=False,
            boundary=self.api.EpisodeBoundary(),
        )

        self.assertEqual(cost.contact_cost, 0.0)

    def test_collision_at_damage_cap_keeps_collision_floor(self):
        cost = self.api.compute_transition_cost(
            damage_before=1.0,
            damage_after=1.0,
            collision=True,
            off_track=False,
            finished=False,
            boundary=self.api.EpisodeBoundary(),
        )

        self.assertAlmostEqual(cost.contact_cost, 0.2)

    def test_off_track_and_environment_failure_costs_are_separate(self):
        cost = self.api.compute_transition_cost(
            damage_before=0.0,
            damage_after=0.0,
            collision=False,
            off_track=True,
            finished=False,
            boundary=self.api.EpisodeBoundary(terminated=True),
        )

        self.assertAlmostEqual(cost.off_track_cost, 1.0 / 30.0)
        self.assertAlmostEqual(cost.failure_cost, 0.2)

    def test_true_environment_time_cap_is_failure_but_finish_is_not(self):
        timed_out = self.api.compute_transition_cost(
            damage_before=0.0,
            damage_after=0.0,
            collision=False,
            off_track=False,
            finished=False,
            boundary=self.api.EpisodeBoundary(environment_truncated=True),
        )
        finished = self.api.compute_transition_cost(
            damage_before=0.0,
            damage_after=0.0,
            collision=False,
            off_track=False,
            finished=True,
            boundary=self.api.EpisodeBoundary(environment_truncated=True),
        )

        self.assertAlmostEqual(timed_out.failure_cost, 0.2)
        self.assertEqual(finished.failure_cost, 0.0)

    def test_registered_episode_decision_cap_counts_as_failure(self):
        capped = self.api.compute_transition_cost(
            damage_before=0.0,
            damage_after=0.0,
            collision=False,
            off_track=False,
            finished=False,
            boundary=self.api.EpisodeBoundary(episode_time_limit=True),
        )

        self.assertAlmostEqual(capped.failure_cost, 0.2)

    def test_collector_cut_is_not_an_environment_failure(self):
        cost = self.api.compute_transition_cost(
            damage_before=0.0,
            damage_after=0.0,
            collision=False,
            off_track=False,
            finished=False,
            boundary=self.api.EpisodeBoundary(collector_truncated=True),
        )

        self.assertEqual(cost.failure_cost, 0.0)

    def test_episode_accumulator_waits_for_real_environment_end(self):
        accumulator = self.api.EpisodeCostAccumulator()
        contact = self.api.CostComponents(contact_cost=0.2)
        collector_cut = self.api.EpisodeBoundary(collector_truncated=True)

        self.assertIsNone(
            accumulator.add_transition(contact, collector_cut, split="train")
        )
        self.assertEqual(accumulator.pending_steps, 1)
        self.assertAlmostEqual(accumulator.pending_cost, 0.2)

        finished = accumulator.add_transition(
            self.api.CostComponents(),
            self.api.EpisodeBoundary(environment_truncated=True),
            split="train",
        )
        self.assertIsInstance(finished, self.api.CompletedEpisodeCost)
        self.assertEqual(finished.steps, 2)
        self.assertAlmostEqual(finished.total_cost, 0.2)
        self.assertEqual(accumulator.pending_steps, 0)

    def test_multiplier_updates_only_after_eight_completed_episodes(self):
        fixed = self.api.LagrangeMultiplier(mode="fixed")
        adaptive = self.api.LagrangeMultiplier(mode="adaptive")
        accumulator = self.api.EpisodeCostAccumulator()

        for _ in range(7):
            episode = accumulator.add_transition(
                self.api.CostComponents(contact_cost=0.25),
                self.api.EpisodeBoundary(terminated=True),
                split="train",
            )
            self.assertIsNotNone(episode)
            self.assertIsNone(fixed.observe_completed_episode(episode))
            self.assertIsNone(adaptive.observe_completed_episode(episode))

        self.assertEqual(fixed.value, 30.0)
        self.assertEqual(adaptive.value, 30.0)

        eighth = accumulator.add_transition(
            self.api.CostComponents(contact_cost=0.25),
            self.api.EpisodeBoundary(terminated=True),
            split="train",
        )
        fixed_window = fixed.observe_completed_episode(eighth)
        adaptive_window = adaptive.observe_completed_episode(eighth)

        self.assertEqual(fixed_window.episode_count, 8)
        self.assertAlmostEqual(fixed_window.multiplier_after, 30.0)
        self.assertEqual(adaptive_window.episode_count, 8)
        self.assertAlmostEqual(adaptive_window.mean_cost, 0.25)
        self.assertAlmostEqual(adaptive_window.multiplier_after, 31.0)

    def test_adaptive_multiplier_is_clipped_to_configured_bounds(self):
        accumulator = self.api.EpisodeCostAccumulator()
        upper = self.api.LagrangeMultiplier(mode="adaptive", initial_value=59.0)
        lower = self.api.LagrangeMultiplier(mode="adaptive", initial_value=0.1)

        for _ in range(8):
            high_cost_episode = accumulator.add_transition(
                self.api.CostComponents(contact_cost=20.0),
                self.api.EpisodeBoundary(terminated=True),
                split="train",
            )
            upper.observe_completed_episode(high_cost_episode)
            zero_cost_episode = accumulator.add_transition(
                self.api.CostComponents(),
                self.api.EpisodeBoundary(terminated=True),
                split="train",
            )
            lower.observe_completed_episode(zero_cost_episode)

        self.assertEqual(upper.value, 60.0)
        self.assertEqual(lower.value, 0.0)

    def test_non_train_episode_cannot_enter_cost_window(self):
        accumulator = self.api.EpisodeCostAccumulator()
        with self.assertRaises(ValueError):
            accumulator.add_transition(
                self.api.CostComponents(),
                self.api.EpisodeBoundary(terminated=True),
                split="tune",
            )

    def test_collector_fragment_discarded_on_reset_is_not_a_complete_episode(self):
        accumulator = self.api.EpisodeCostAccumulator()
        multiplier = self.api.LagrangeMultiplier(mode="adaptive")
        accumulator.add_transition(
            self.api.CostComponents(contact_cost=0.2),
            self.api.EpisodeBoundary(collector_truncated=True),
            split="train",
        )

        incomplete = accumulator.discard_partial()

        self.assertIsInstance(incomplete, self.api.IncompleteEpisodeCost)
        self.assertEqual(multiplier.window_episode_count, 0)
        with self.assertRaises(TypeError):
            multiplier.observe_completed_episode(incomplete)

    def test_multiplier_rejects_nonfinite_configuration(self):
        with self.assertRaises(ValueError):
            self.api.LagrangeMultiplier(initial_value=math.nan)

    def test_cost_rejects_damage_outside_environment_range(self):
        with self.assertRaises(ValueError):
            self.api.compute_transition_cost(
                damage_before=0.0,
                damage_after=1.1,
                collision=False,
                off_track=False,
                finished=False,
                boundary=self.api.EpisodeBoundary(),
            )


if __name__ == "__main__":
    unittest.main()
