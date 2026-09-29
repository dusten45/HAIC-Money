"""Speed-path regressions: pedal headroom and the lap-time reward gradient.

The selected PPO actor drove 19.66s laps while the offline reference teacher
drove 17.32s on the same map. Rollout metrics showed sampled ``gas_max`` at
0.1153 against a 0.12 ceiling, so the actor was already asking for essentially
all the throttle the action transform allowed. These tests pin the two fixes:
the transform now has headroom above the cruise command, and the shaped reward
now prefers a faster lap over a slower one covering the same distance.
"""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class TestPedalExpansionTransform(unittest.TestCase):
    def test_expansion_one_reproduces_the_original_pedal_transform(self):
        # Break caught: changing the default transform would silently redefine
        # every action stored in an existing checkpoint.
        from haic_agent.networks import MAX_BRAKE, MAX_GAS, VisualActorCritic

        coordinates = torch.tensor([[0.4, 0.573], [-0.9, 1.959], [0.0, -1.2]])

        default = VisualActorCritic._bound_actions(coordinates)
        explicit = VisualActorCritic._bound_actions(coordinates, 1.0)
        expected_longitudinal = torch.tanh(coordinates[:, 1])

        torch.testing.assert_close(default, explicit)
        torch.testing.assert_close(
            default[:, 1], MAX_GAS * expected_longitudinal.clamp(min=0.0)
        )
        torch.testing.assert_close(
            default[:, 2], MAX_BRAKE * (-expected_longitudinal).clamp(min=0.0)
        )
        self.assertEqual(VisualActorCritic.pedal_limits(1.0), (MAX_GAS, MAX_BRAKE))

    def test_expansion_holds_the_cruise_command_while_raising_the_ceiling(self):
        # Break caught: scaling the pedal limit alone would multiply the cruise
        # throttle a resumed checkpoint already commands, not just its ceiling.
        from haic_agent.networks import MAX_GAS, VisualActorCritic

        # 0.573 is atanh(0.0622 / 0.12), the selected actor's mean rollout gas;
        # 1.959 is atanh(0.1153 / 0.12), its saturated maximum.
        cruise = torch.tensor([[0.0, 0.573]])
        saturated = torch.tensor([[0.0, 1.959]])

        baseline_cruise = float(VisualActorCritic._bound_actions(cruise)[0, 1])
        baseline_peak = float(VisualActorCritic._bound_actions(saturated)[0, 1])
        expanded_cruise = float(VisualActorCritic._bound_actions(cruise, 2.0)[0, 1])
        expanded_peak = float(VisualActorCritic._bound_actions(saturated, 2.0)[0, 1])

        self.assertLess(abs(expanded_cruise - baseline_cruise) / baseline_cruise, 0.1)
        self.assertGreater(expanded_peak, 1.4 * baseline_peak)
        self.assertLessEqual(expanded_peak, 2.0 * MAX_GAS)

    def test_expanded_actions_round_trip_and_keep_log_probabilities_consistent(self):
        # Break caught: forgetting the expansion in the inverse or the Jacobian
        # makes PPO reweight rollout actions it never actually took.
        from haic_agent.networks import VisualActorCritic

        torch.manual_seed(11)
        model = VisualActorCritic(use_hud=True, pedal_expansion=2.5)
        output = model(torch.rand(4, 4, 84, 84))
        actions, sampled_log_probability = model.sample_actions(output)
        recomputed = model.log_probability(
            actions, output.action_mean, output.action_log_std
        )
        recovered, _ = model.unbound_actions(actions)

        torch.testing.assert_close(
            sampled_log_probability, recomputed, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            model.bound_actions(recovered), actions, rtol=1e-4, atol=1e-4
        )
        self.assertTrue(torch.all(actions[:, 1] <= model.throttle_limit + 1e-6))
        self.assertTrue(torch.all(actions[:, 2] <= model.brake_limit + 1e-6))
        self.assertFalse(bool(torch.any((actions[:, 1] > 0.0) & (actions[:, 2] > 0.0))))

    def test_fresh_actor_starts_at_the_same_absolute_gas_at_any_expansion(self):
        # Break caught: keeping the old bias initializer would make an expanded
        # actor launch at several times the calibrated opening throttle.
        from haic_agent.networks import VisualActorCritic

        torch.manual_seed(3)
        observations = torch.rand(1, 4, 84, 84)
        gas_values = []
        for expansion in (1.0, 2.0, 3.5):
            torch.manual_seed(3)
            model = VisualActorCritic(pedal_expansion=expansion)
            with torch.no_grad():
                action = model.deterministic_actions(model(observations))
            gas_values.append(float(action[0, 1]))

        for gas in gas_values[1:]:
            self.assertAlmostEqual(gas, gas_values[0], places=4)

    def test_expansion_outside_the_supported_range_is_rejected(self):
        # Break caught: an expansion past a branch's ceiling would command a
        # pedal above the simulator's 1.0 limit, where the inverse is no longer
        # exact and the submission contract is violated.
        from haic_agent.networks import (
            MAX_BRAKE,
            MAX_BRAKE_EXPANSION,
            MAX_GAS,
            MAX_PEDAL_EXPANSION,
            MAX_THROTTLE_EXPANSION,
            VisualActorCritic,
        )

        self.assertAlmostEqual(MAX_GAS * MAX_THROTTLE_EXPANSION, 1.0)
        self.assertAlmostEqual(MAX_BRAKE * MAX_BRAKE_EXPANSION, 1.0)
        self.assertEqual(MAX_PEDAL_EXPANSION, MAX_BRAKE_EXPANSION)

        for invalid in (0.5, 0.0, float("nan"), MAX_PEDAL_EXPANSION + 0.1):
            with self.subTest(symmetric=invalid):
                with self.assertRaisesRegex(ValueError, "expansion"):
                    VisualActorCritic(pedal_expansion=invalid)

        with self.assertRaisesRegex(ValueError, "throttle_expansion"):
            VisualActorCritic(throttle_expansion=MAX_THROTTLE_EXPANSION + 0.1)
        with self.assertRaisesRegex(ValueError, "brake_expansion"):
            VisualActorCritic(brake_expansion=MAX_BRAKE_EXPANSION + 0.1)

    def test_each_branch_can_open_to_the_full_simulator_pedal_range(self):
        # Break caught: a single symmetric factor cannot reach gas 1.0 without
        # pushing brake past 1.0, which caps lap time below the 15s target.
        from haic_agent.networks import (
            MAX_BRAKE_EXPANSION,
            MAX_THROTTLE_EXPANSION,
            VisualActorCritic,
        )

        model = VisualActorCritic(
            throttle_expansion=MAX_THROTTLE_EXPANSION,
            brake_expansion=MAX_BRAKE_EXPANSION,
        )
        saturated = torch.tensor([[0.0, 120.0], [0.0, -120.0]])
        actions = model.bound_actions(saturated)

        self.assertAlmostEqual(model.throttle_limit, 1.0, places=6)
        self.assertAlmostEqual(model.brake_limit, 1.0, places=6)
        self.assertAlmostEqual(float(actions[0, 1]), 1.0, places=4)
        self.assertAlmostEqual(float(actions[1, 2]), 1.0, places=4)
        self.assertFalse(bool(torch.any((actions[:, 1] > 0.0) & (actions[:, 2] > 0.0))))

    def test_asymmetric_branches_round_trip_and_keep_log_probabilities(self):
        # Break caught: using one branch's expansion in the inverse or the
        # Jacobian silently reweights every braking sample PPO collected.
        from haic_agent.networks import VisualActorCritic

        torch.manual_seed(5)
        model = VisualActorCritic(throttle_expansion=6.0, brake_expansion=3.0)
        output = model(torch.rand(8, 4, 84, 84))
        actions, sampled = model.sample_actions(output)
        recomputed = model.log_probability(
            actions, output.action_mean, output.action_log_std
        )
        recovered, _ = model.unbound_actions(actions)

        torch.testing.assert_close(sampled, recomputed, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            model.bound_actions(recovered), actions, rtol=1e-4, atol=1e-4
        )


class TestExpandedActorRuntime(unittest.TestCase):
    def test_checkpoint_metadata_restores_the_expanded_pedal_range(self):
        # Break caught: dropping the metadata field makes the submission Agent
        # replay an expanded actor through the narrow original transform.
        from agent import Agent
        from haic_agent.networks import VisualActorCritic

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            policy = VisualActorCritic(pedal_expansion=2.0)
            torch.save(
                {
                    "model_state": policy.state_dict(),
                    "metadata": {"pedal_expansion": 2.0},
                },
                checkpoint,
            )
            loaded = Agent(
                policy_checkpoint=str(checkpoint),
                planner_enabled=False,
                strict_checkpoint_loading=True,
            )

        self.assertEqual(loaded.policy.pedal_expansion, 2.0)
        self.assertAlmostEqual(loaded.policy.throttle_limit, 0.24)
        observation = np.random.default_rng(5).random((4, 84, 84)).astype(np.float32)
        action = loaded.act(observation)
        self.assertEqual(action.shape, (3,))
        self.assertTrue(np.all(np.isfinite(action)))

    def test_checkpoint_without_the_field_keeps_the_original_pedal_range(self):
        # Break caught: defaulting to an expanded range would make every
        # previously trained checkpoint drive harder than it was selected for.
        from agent import Agent
        from haic_agent.networks import MAX_GAS, VisualActorCritic

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            torch.save(
                {"model_state": VisualActorCritic().state_dict(), "metadata": {}},
                checkpoint,
            )
            loaded = Agent(
                policy_checkpoint=str(checkpoint),
                planner_enabled=False,
                strict_checkpoint_loading=True,
            )

        self.assertEqual(loaded.policy.pedal_expansion, 1.0)
        self.assertAlmostEqual(loaded.policy.throttle_limit, MAX_GAS)


class TestTeacherPedalRange(unittest.TestCase):
    def test_demonstration_targets_use_the_teacher_profile_range(self):
        # Break caught: normalizing by the class constant clipped every faster
        # profile's throttle to 0.12, so imitation could never teach the speed
        # the benchmark teacher actually drives.
        from haic_agent.networks import VisualActorCritic
        from training.imitation import _teacher_action_tensor, teacher_pedal_reference

        fast_teacher = SimpleNamespace(max_gas=0.24)
        pedal_range = teacher_pedal_reference(fast_teacher)
        target = _teacher_action_tensor(
            np.asarray([0.0, 0.24, 0.0], dtype=np.float32),
            teacher_pedal_range=pedal_range,
            pedal_expansion=2.0,
            cap_fraction=1.0,
        )
        action = VisualActorCritic._bound_actions(target.unsqueeze(0), 2.0)[0]

        self.assertAlmostEqual(pedal_range[0], 0.24)
        self.assertAlmostEqual(float(action[1]), 0.24, places=4)

    def test_default_teacher_profile_keeps_the_original_target_mapping(self):
        # Break caught: the profile fix must not move the default teacher's
        # demonstrations, which every earlier warm-start was measured against.
        from haic_agent.networks import VisualActorCritic
        from training.imitation import TEACHER_POLICY_CAP_FRACTION, _teacher_action_tensor

        target = _teacher_action_tensor(np.asarray([0.0, 0.12, 0.0], dtype=np.float32))
        action = VisualActorCritic._bound_actions(target.unsqueeze(0))[0]

        self.assertAlmostEqual(
            float(action[1]), 0.12 * TEACHER_POLICY_CAP_FRACTION, places=5
        )


class TestLapTimeRewardGradient(unittest.TestCase):
    @staticmethod
    def _lap_reward(speed: float, *, tiles: int = 200) -> float:
        """Accumulate shaped reward for one full lap driven at a fixed speed."""
        from training.train_policy import shape_transition_reward

        # A decision advances the car by speed * 0.08s, so a fixed-length lap
        # takes proportionally fewer decisions at a higher speed.
        lap_distance = 1000.0
        decisions = max(1, int(round(lap_distance / (speed * 0.08))))
        total = 0.0
        for index in range(decisions):
            progress = (index + 1) / decisions
            labels = SimpleNamespace(
                speed=speed,
                tile_progress=progress,
                finished=index + 1 == decisions,
                collision=False,
                off_track=False,
                damage=0.0,
                lateral_error=0.0,
                road_half_width=8.0,
                heading_error=0.0,
            )
            # The simulator pays a fixed amount per tile crossed, so a lap is
            # worth the same total tile reward however fast it is driven.
            transition = SimpleNamespace(
                reward=1000.0 / decisions, labels=labels, terminated=False
            )
            total += shape_transition_reward(
                transition, index / decisions, previous_speed=speed
            )
        return total

    def test_a_faster_lap_earns_strictly_more_than_a_slower_one(self):
        # Break caught: a per-decision bonus proportional to speed sums to the
        # same value for every lap of the same length, so it expresses no
        # preference for finishing sooner.
        slow = self._lap_reward(35.0)
        fast = self._lap_reward(47.0)

        self.assertGreater(fast, slow)
        self.assertGreater(fast - slow, 1.0)

    def test_the_speed_charge_is_waived_after_a_collision_or_off_track_exit(self):
        # Break caught: charging for low speed while the car is already in a
        # failure state would double-count penalties it cannot act on.
        from training.train_policy import shape_transition_reward

        def shaped(speed, **failure):
            labels = SimpleNamespace(
                speed=speed,
                tile_progress=0.0,
                finished=False,
                collision=failure.get("collision", False),
                off_track=failure.get("off_track", False),
                damage=0.0,
                lateral_error=0.0,
                road_half_width=8.0,
                heading_error=0.0,
            )
            return shape_transition_reward(
                SimpleNamespace(reward=0.0, labels=labels, terminated=False), 0.0
            )

        for failure in ({"collision": True}, {"off_track": True}):
            with self.subTest(failure=failure):
                self.assertAlmostEqual(shaped(5.0, **failure), shaped(45.0, **failure))

    def test_stopping_in_front_of_a_hazard_still_costs_reward(self):
        # Break caught: waiving the speed charge outright near a hazard makes
        # idling in front of an obstacle the cheapest place on the track.
        from training.train_policy import shape_transition_reward

        def shaped(speed):
            labels = SimpleNamespace(
                speed=speed,
                tile_progress=0.0,
                finished=False,
                collision=False,
                off_track=False,
                damage=0.0,
                lateral_error=0.0,
                road_half_width=8.0,
                heading_error=0.0,
            )
            transition = SimpleNamespace(
                reward=0.0,
                labels=labels,
                terminated=False,
                action=np.zeros(3, dtype=np.float32),
            )
            hazard = np.asarray(
                [0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32
            )
            return shape_transition_reward(
                transition, 0.0, visual_features=hazard
            )

        from training.train_policy import HAZARD_SPEED_RELIEF, SPEED_TARGET

        # A full hazard relaxes the target rather than removing it, so a stopped
        # car still pays while anything at or above the relaxed target does not.
        hazard_target = SPEED_TARGET * (1.0 - HAZARD_SPEED_RELIEF)
        self.assertLess(shaped(0.0), shaped(hazard_target * 0.5))
        self.assertLess(shaped(hazard_target * 0.5), shaped(hazard_target))
        self.assertAlmostEqual(shaped(hazard_target), shaped(hazard_target + 10.0))


class TestThrottleSaturationDiagnostics(unittest.TestCase):
    def test_rollout_metrics_report_how_often_the_throttle_ceiling_binds(self):
        # Break caught: without this metric a lap-time investigation cannot tell
        # a cautious actor apart from one pinned against its action limit.
        from training.rollout import RolloutStorage
        from training.train_policy import rollout_action_metrics

        storage = RolloutStorage()
        for gas in (0.02, 0.06, 0.1199, 0.12):
            storage.add(
                observation=torch.zeros(4, 84, 84),
                action=torch.tensor([0.0, gas, 0.0]),
                pretransform_action=torch.zeros(2),
                log_probability=0.0,
                value=0.0,
                next_value=0.0,
                reward=0.0,
                terminated=False,
                truncated=True,
                auxiliary_targets=torch.zeros(10),
            )
        storage.rollout_speeds = (10.0, 20.0, 30.0, 40.0)

        metrics = rollout_action_metrics(storage, throttle_limit=0.12)

        self.assertAlmostEqual(metrics["throttle_limit"], 0.12)
        self.assertAlmostEqual(metrics["gas_saturation_fraction"], 0.5)
        self.assertAlmostEqual(metrics["mean_speed"], 25.0)
        self.assertAlmostEqual(metrics["max_speed"], 40.0)
        self.assertTrue(math.isfinite(metrics["gas_max"]))


if __name__ == "__main__":
    unittest.main()
