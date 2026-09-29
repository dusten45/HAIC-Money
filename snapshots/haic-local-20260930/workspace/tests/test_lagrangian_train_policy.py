import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


def _labels(**overrides):
    from training.labels import TrainingLabels

    values = {
        "speed": 70.0,
        "wheel_omega": (0.0, 0.0, 0.0, 0.0),
        "steering_angle": 0.0,
        "yaw_rate": 0.0,
        "tile_progress": 0.0,
        "collision": False,
        "damage": 0.0,
        "off_track": False,
        "finished": False,
        "lateral_error": 0.0,
        "road_half_width": 7.0,
        "heading_error": 0.0,
    }
    values.update(overrides)
    return TrainingLabels(**values)


def _transition(
    *,
    labels=None,
    observation_labels=None,
    reward=0.0,
    terminated=False,
    truncated=False,
    finish_time_s=None,
    simulator_out_of_bounds_failure=False,
):
    from training.env_factory import CollectedTransition

    pixels = np.zeros((4, 84, 84), dtype=np.float32)
    return CollectedTransition(
        observation=pixels,
        action=np.zeros(3, dtype=np.float32),
        next_observation=pixels.copy(),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        labels=labels or _labels(),
        hud_features=None,
        observation_labels=observation_labels,
        finish_time_s=finish_time_s,
        simulator_out_of_bounds_failure=simulator_out_of_bounds_failure,
    )


class _OneTransitionEnvironment:
    def __init__(self, transition):
        self.transition = transition
        self.closed = False

    def reset(self):
        return self.transition.observation.copy(), {}

    def step_transition(self, action):
        del action
        return self.transition

    def close(self):
        self.closed = True


class TestLagrangianRewardAndCollection(unittest.TestCase):
    def test_collector_records_episode_outcomes_for_both_ppo_modes(self):
        from haic_agent.networks import VisualActorCritic
        from training.train_policy import collect_rollout

        transition = _transition(
            labels=_labels(
                speed=82.0,
                tile_progress=0.91,
                collision=True,
                damage=0.4,
                finished=True,
            ),
            observation_labels=_labels(damage=0.2),
            truncated=True,
            finish_time_s=12.3,
        )
        model = VisualActorCritic(use_hud=False)
        with patch(
            "training.train_policy._create_environment_for_episode",
            return_value=_OneTransitionEnvironment(transition),
        ):
            rollout = collect_rollout(
                model,
                [(1, 101)],
                total_steps=1,
                max_decisions=4,
                lagrangian_mode="off",
            )

        outcome, = rollout.train_episode_outcomes
        self.assertEqual(outcome["episode"], {"track_id": 1, "seed": 101})
        self.assertTrue(outcome["episode_end_observed"])
        self.assertTrue(outcome["finished"])
        self.assertEqual(outcome["boundary"], "finished")
        self.assertAlmostEqual(outcome["finish_time_s"], 12.3)
        self.assertEqual(outcome["decisions"], 1)
        self.assertEqual(outcome["collision_decisions"], 1)
        self.assertEqual(outcome["collision_onsets"], 1)
        self.assertAlmostEqual(outcome["damage_max"], 0.4)
        self.assertAlmostEqual(outcome["damage_final"], 0.4)
        self.assertAlmostEqual(outcome["progress_end"], 0.91)
        self.assertAlmostEqual(outcome["mean_speed"], 82.0)
        self.assertAlmostEqual(outcome["max_speed"], 82.0)
        self.assertAlmostEqual(outcome["cost_components"]["contact_cost"], 0.2)

    def test_collector_cut_episode_outcome_is_incomplete(self):
        from haic_agent.networks import VisualActorCritic
        from training.train_policy import collect_rollout

        transition = _transition(labels=_labels(speed=55.0, tile_progress=0.3))
        model = VisualActorCritic(use_hud=False)
        with patch(
            "training.train_policy._create_environment_for_episode",
            return_value=_OneTransitionEnvironment(transition),
        ):
            rollout = collect_rollout(
                model,
                [(2, 202)],
                total_steps=1,
                max_decisions=4,
                lagrangian_mode="off",
            )

        self.assertEqual(rollout.train_episode_outcomes, [])
        partial, = rollout.incomplete_train_episode_outcomes
        self.assertFalse(partial["episode_end_observed"])
        self.assertFalse(partial["finished"])
        self.assertEqual(partial["boundary"], "collector_cutoff")
        self.assertAlmostEqual(partial["progress_end"], 0.3)

    def test_safety_cost_mode_removes_collision_damage_and_failure_penalties_from_reward(self):
        from training.train_policy import (
            COLLISION_PENALTY,
            REWARD_SCALE,
            TERMINAL_FAILURE_PENALTY,
            shape_transition_reward,
        )

        transition = _transition(
            labels=_labels(collision=True, damage=0.2),
            observation_labels=_labels(damage=0.0),
            reward=-0.4,
            terminated=True,
        )
        ordinary = shape_transition_reward(transition, 0.0)
        separated = shape_transition_reward(
            transition, 0.0, safety_costs_separate=True
        )

        expected_removed_penalties = (
            COLLISION_PENALTY + 0.1 * 0.2 + TERMINAL_FAILURE_PENALTY
        ) * REWARD_SCALE
        self.assertAlmostEqual(separated - ordinary, expected_removed_penalties)

    def test_safety_cost_mode_removes_simulator_out_of_bounds_reward(self):
        from training.train_policy import REWARD_SCALE, TIME_PENALTY, shape_transition_reward

        transition = _transition(
            reward=-100.0,
            terminated=True,
            simulator_out_of_bounds_failure=True,
        )

        separated = shape_transition_reward(
            transition, 0.0, safety_costs_separate=True
        )

        self.assertAlmostEqual(separated, -TIME_PENALTY * REWARD_SCALE)

    def test_safety_cost_mode_removes_out_of_bounds_reward_even_with_collision_label(self):
        from training.train_policy import REWARD_SCALE, TIME_PENALTY, shape_transition_reward

        transition = _transition(
            labels=_labels(collision=True, damage=0.2),
            observation_labels=_labels(damage=0.0),
            reward=-100.0,
            terminated=True,
            simulator_out_of_bounds_failure=True,
        )

        separated = shape_transition_reward(
            transition, 0.0, safety_costs_separate=True
        )

        self.assertAlmostEqual(separated, -TIME_PENALTY * REWARD_SCALE)

    def test_collector_preserves_underlying_out_of_bounds_failure_flag(self):
        from training.env_factory import CollectingEnvironment

        observation = np.zeros((4, 84, 84), dtype=np.float32)
        collector = object.__new__(CollectingEnvironment)
        collector.environment = object()
        collector._observation = observation.copy()
        collector.step = lambda action: (
            observation.copy(),
            -100.0,
            True,
            False,
            {"simulator_terminated": True, "finished": False},
        )
        with (
            patch("training.env_factory.collect_labels", return_value=_labels()),
            patch(
                "training.env_factory.labels_with_hud",
                return_value=(_labels(collision=True, damage=0.2), None),
            ),
        ):
            transition = collector.step_transition(np.zeros(3, dtype=np.float32))

        self.assertTrue(transition.simulator_out_of_bounds_failure)
        self.assertTrue(transition.labels.collision)

    def test_collector_routes_damage_labels_to_cost_stream_and_marks_real_end(self):
        from haic_agent.networks import VisualActorCritic
        from training.lagrangian import EpisodeBoundary
        from training.lagrangian_ppo import LagrangianActorCritic
        from training.train_policy import collect_rollout

        transition = _transition(
            labels=_labels(collision=True, damage=0.2),
            observation_labels=_labels(damage=0.0),
            reward=-0.4,
            terminated=True,
        )
        model = LagrangianActorCritic(VisualActorCritic(use_hud=False))
        with patch(
            "training.train_policy._create_environment_for_episode",
            return_value=_OneTransitionEnvironment(transition),
        ):
            rollout = collect_rollout(
                model,
                [(1, 101)],
                total_steps=1,
                max_decisions=4,
                lagrangian_mode="fixed",
            )

        self.assertEqual(len(rollout), 1)
        self.assertEqual(rollout.steps[0].boundary, EpisodeBoundary(terminated=True))
        self.assertAlmostEqual(rollout.steps[0].cost, 0.2 + 0.2 / 3000.0 + 0.2)
        self.assertEqual(len(rollout.completed_episode_costs), 1)
        self.assertEqual(rollout.completed_episode_costs[0].steps, 1)
        self.assertEqual(rollout.steps[0].observation.shape, (4, 84, 84))
        self.assertEqual(rollout.steps[0].action.shape, (3,))

    def test_episode_decision_cap_completes_failure_but_rollout_cut_stays_incomplete(self):
        from haic_agent.networks import VisualActorCritic
        from training.lagrangian import EpisodeBoundary
        from training.lagrangian_ppo import LagrangianActorCritic
        from training.train_policy import collect_rollout

        model = LagrangianActorCritic(VisualActorCritic(use_hud=False))
        transition = _transition(labels=_labels())

        with patch(
            "training.train_policy._create_environment_for_episode",
            return_value=_OneTransitionEnvironment(transition),
        ):
            episode_limited = collect_rollout(
                model,
                [(1, 101)],
                total_steps=1,
                max_decisions=1,
                lagrangian_mode="adaptive",
            )
        self.assertEqual(
            episode_limited.steps[0].boundary,
            EpisodeBoundary(episode_time_limit=True),
        )
        self.assertAlmostEqual(episode_limited.steps[0].cost, 0.2)
        self.assertEqual(len(episode_limited.completed_episode_costs), 1)

        with patch(
            "training.train_policy._create_environment_for_episode",
            return_value=_OneTransitionEnvironment(transition),
        ):
            collector_limited = collect_rollout(
                model,
                [(1, 101)],
                total_steps=1,
                max_decisions=4,
                lagrangian_mode="adaptive",
            )
        self.assertEqual(
            collector_limited.steps[0].boundary,
            EpisodeBoundary(collector_truncated=True),
        )
        self.assertEqual(len(collector_limited.completed_episode_costs), 0)
        self.assertEqual(len(collector_limited.incomplete_episode_costs), 1)


class TestLagrangianTrainerIntegration(unittest.TestCase):
    def test_adaptive_multiplier_resume_preserves_partial_completed_episode_window(self):
        from training.lagrangian import (
            CostComponents,
            EpisodeBoundary,
            EpisodeCostAccumulator,
            LagrangeMultiplier,
        )

        source = LagrangeMultiplier(mode="adaptive", window_size=4)
        accumulator = EpisodeCostAccumulator()
        for cost in (0.1, 0.2):
            episode = accumulator.add_transition(
                CostComponents(contact_cost=cost),
                EpisodeBoundary(terminated=True),
                split="train",
            )
            source.observe_completed_episode(episode)

        restored = LagrangeMultiplier(mode="adaptive", window_size=4)
        restored.load_state_dict(source.state_dict())

        self.assertEqual(restored.completed_episode_count, 2)
        self.assertEqual(restored.window_episode_count, 2)
        for multiplier in (source, restored):
            for cost in (0.0, 0.0):
                episode = accumulator.add_transition(
                    CostComponents(contact_cost=cost),
                    EpisodeBoundary(terminated=True),
                    split="train",
                )
                multiplier.observe_completed_episode(episode)
        self.assertEqual(restored.value, source.value)
        self.assertEqual(restored.completed_episode_count, source.completed_episode_count)

    def test_adaptive_train_update_uses_completed_train_episode_and_exports_actor_only(self):
        from haic_agent.networks import VisualActorCritic
        from training.env_factory import split_track_seeds
        from training.lagrangian_ppo import LagrangianPPOUpdater
        from training.train_policy import train

        finished = _transition(
            labels=_labels(tile_progress=1.0, finished=True),
            terminated=True,
        )
        update_calls = []

        def mock_update(updater, rollout, *, lagrange_multiplier):
            update_calls.append(
                {
                    "model": updater.model,
                    "rollout": rollout,
                    "lambda": lagrange_multiplier,
                }
            )
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "cost_value_loss": 0.0,
                "auxiliary_loss": 0.0,
                "entropy": 0.0,
                "grad_norm": 0.0,
                "lagrange_multiplier": lagrange_multiplier,
            }

        split = split_track_seeds(train=((1, 101),), tune=(), held_out=())
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "training.train_policy._create_environment_for_episode",
                return_value=_OneTransitionEnvironment(finished),
            ),
            patch.object(LagrangianPPOUpdater, "update", mock_update),
        ):
            result = train(
                output_directory=Path(directory),
                total_steps=1,
                max_decisions=1,
                seed=17,
                updates=1,
                split=split,
                teacher_warmup_epochs=0,
                use_hud=False,
                tune_selection=False,
                lagrangian_mode="adaptive",
                lagrangian_window_size=1,
            )

            actor_checkpoint = torch.load(
                Path(directory) / "policy.pt", map_location="cpu", weights_only=False
            )
            inference_actor = VisualActorCritic(use_hud=False)
            inference_actor.load_state_dict(actor_checkpoint["model_state"], strict=True)
            training_checkpoint = torch.load(
                Path(directory) / "lagrangian-training.pt",
                map_location="cpu",
                weights_only=False,
            )

        self.assertEqual(len(update_calls), 1)
        self.assertEqual(update_calls[0]["lambda"], 30.0)
        self.assertEqual(len(update_calls[0]["rollout"].completed_episode_costs), 1)
        self.assertAlmostEqual(result["lagrangian"]["lambda_after"], 29.75)
        self.assertEqual(result["training_checkpoint"], str(Path(directory) / "lagrangian-training.pt"))
        self.assertNotIn("cost_value_head.weight", actor_checkpoint["model_state"])
        self.assertTrue(all(name.startswith("policy.") or name.startswith("cost_value_head.") for name in training_checkpoint["model_state"]))
        self.assertEqual(training_checkpoint["metadata"]["lagrangian_mode"], "adaptive")


if __name__ == "__main__":
    unittest.main()
