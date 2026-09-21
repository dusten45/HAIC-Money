import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class TestRolloutCollection(unittest.TestCase):
    def test_manual_episode_and_rollout_limits_mark_gae_boundaries(self):
        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import collect_rollout

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=0.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.0,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
        )

        class _Environment:
            def reset(self):
                return pixels, {}

            def step_transition(self, action):
                return CollectedTransition(
                    observation=pixels,
                    action=action,
                    next_observation=pixels,
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    labels=labels,
                    hud_features=None,
                )

            def close(self):
                pass

        class _Output:
            value = torch.zeros(1)

        class _Model:
            def __call__(self, observations):
                return _Output()

            def sample_actions_with_pretransform(self, output):
                return torch.zeros(1, 3), torch.zeros(1), torch.zeros(1, 2)

        with patch(
            "training.train_policy._create_environment_for_episode",
            side_effect=lambda episode, max_decisions: _Environment(),
        ):
            rollout = collect_rollout(
                _Model(), [(1, 1)], total_steps=3, max_decisions=2
            )

        self.assertEqual(len(rollout), 3)
        self.assertFalse(rollout.steps[0].truncated)
        self.assertTrue(rollout.steps[1].truncated)
        self.assertTrue(rollout.steps[2].truncated)


class TestRepeatedPPOTraining(unittest.TestCase):
    def test_training_distributes_total_steps_across_multiple_on_policy_updates(self):
        from training.train_policy import train

        metrics = {
            "finish_rate": 0.0,
            "median_finished_lap_time_s": None,
            "p90_finished_lap_time_s": None,
            "mean_progress": 0.1,
        }
        action_stats = {"steer_mean": -0.1, "brake_fraction": 0.08}

        class _Updater:
            def __init__(self, model, config):
                self.update_sizes = []
                self.config = config
                updater_instances.append(self)

            def update(self, rollout):
                self.update_sizes.append(len(rollout))
                return {"policy_loss": 0.0}

        updater_instances = []
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("training.train_policy.collect_rollout", side_effect=[range(5), range(5)]) as collect,
                patch("training.train_policy.rollout_action_metrics", return_value=action_stats) as action_metrics,
                patch("training.train_policy.PPOUpdater", _Updater),
                patch("training.train_policy.evaluate_policy", return_value=metrics) as evaluate,
                patch("training.train_policy.hud_ablation", return_value={"hud_enabled": metrics}),
                patch("training.train_policy.save_best_checkpoint", return_value=True) as save_best,
            ):
                result = train(
                    output_directory=Path(directory),
                    total_steps=10,
                    max_decisions=20,
                    evaluation_max_decisions=7,
                    seed=42,
                    updates=2,
                    learning_rate=1e-4,
                    teacher_warmup_epochs=0,
                )

        self.assertEqual(collect.call_count, 2)
        self.assertEqual(action_metrics.call_count, 2)
        self.assertEqual([call.kwargs["total_steps"] for call in collect.call_args_list], [5, 5])
        self.assertEqual(evaluate.call_count, 3)
        self.assertEqual(
            [call.kwargs["max_decisions"] for call in evaluate.call_args_list],
            [7, 7, 20],
        )
        self.assertEqual(save_best.call_count, 2)
        self.assertEqual(result["training_steps"], 10)
        self.assertEqual(result["updates_completed"], 2)
        self.assertEqual([item["steps"] for item in result["updates"]], [5, 5])
        self.assertEqual([item["action_metrics"] for item in result["updates"]], [action_stats, action_stats])
        self.assertIn("full_tune_eval_s", result["timings_s"])
        self.assertAlmostEqual(updater_instances[0].config.learning_rate, 1e-4)

    def test_teacher_warmup_runs_before_ppo_and_receives_only_train_episodes(self):
        from training.env_factory import split_track_seeds
        from training.train_policy import train

        metrics = {
            "finish_rate": 0.0,
            "median_finished_lap_time_s": None,
            "p90_finished_lap_time_s": None,
            "mean_progress": 0.1,
        }
        events = []
        split = split_track_seeds(
            train=((1, 101),), tune=((2, 201),), held_out=((3, 301),)
        )

        class _Updater:
            def __init__(self, model, config):
                del model, config

            def update(self, rollout):
                del rollout
                events.append("ppo_update")
                return {"policy_loss": 0.0}

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "training.train_policy.collect_teacher_demonstrations",
                    side_effect=lambda episodes, **kwargs: (
                        events.append(("collect_teacher", tuple(episodes.train), kwargs["max_decisions"]))
                        or "demo-data"
                    ),
                ),
                patch(
                    "training.train_policy.behavioral_cloning_warmup",
                    side_effect=lambda model, demonstrations, **kwargs: (
                        events.append(("teacher_warmup", demonstrations))
                        or {"enabled": True, "final_action_mse": 0.01}
                    ),
                ),
                patch(
                    "training.train_policy.collect_rollout",
                    side_effect=lambda *args, **kwargs: (events.append("ppo_rollout") or range(2)),
                ),
                patch(
                    "training.train_policy.rollout_action_metrics",
                    return_value={"steer_mean": 0.0},
                ),
                patch("training.train_policy.PPOUpdater", _Updater),
                patch("training.train_policy.evaluate_policy", return_value=metrics),
                patch("training.train_policy.hud_ablation", return_value={"hud_enabled": metrics}),
                patch("training.train_policy.save_best_checkpoint", return_value=True),
            ):
                result = train(
                    output_directory=Path(directory),
                    total_steps=2,
                    max_decisions=4,
                    seed=42,
                    updates=1,
                    split=split,
                    teacher_warmup_epochs=1,
                )

        self.assertEqual(
            events,
            [
                ("collect_teacher", split.train, 4),
                ("teacher_warmup", "demo-data"),
                "ppo_rollout",
                "ppo_update",
            ],
        )
        self.assertTrue(result["teacher_warmup"]["enabled"])
        self.assertEqual(result["teacher_warmup"]["final_action_mse"], 0.01)
        self.assertIn("teacher_warmup_s", result["timings_s"])


class TestTrainingSignalScaling(unittest.TestCase):
    def test_auxiliary_sensor_targets_are_normalized_by_physical_scale(self):
        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import auxiliary_targets

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=20.0,
            wheel_omega=(25.0, -50.0, 100.0, 0.0),
            steering_angle=0.2,
            yaw_rate=-1.5,
            tile_progress=0.0,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
            lateral_error=4.0,
            road_half_width=8.0,
            heading_error=torch.pi / 2,
        )
        transition = CollectedTransition(
            observation=pixels,
            action=torch.zeros(3).numpy(),
            next_observation=pixels,
            reward=0.0,
            terminated=False,
            truncated=False,
            labels=labels,
            hud_features=None,
        )

        targets = auxiliary_targets(transition)

        torch.testing.assert_close(
            targets,
            torch.tensor([0.5, 0.25, -0.5, 1.0, 0.0, 0.5, -0.5, 0.5, 1.0, 0.0]),
        )

    def test_auxiliary_visual_targets_match_the_observation_used_for_the_action(self):
        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import auxiliary_targets

        pixels = torch.zeros(4, 84, 84).numpy()
        observation_labels = TrainingLabels(
            speed=12.0,
            wheel_omega=(10.0, 20.0, 30.0, 40.0),
            steering_angle=0.1,
            yaw_rate=0.2,
            tile_progress=0.1,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
            lateral_error=-2.0,
            road_half_width=8.0,
            heading_error=-torch.pi / 2,
        )
        next_labels = TrainingLabels(
            speed=40.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.2,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
        )
        transition = CollectedTransition(
            observation=pixels,
            action=torch.zeros(3).numpy(),
            next_observation=pixels,
            reward=0.0,
            terminated=False,
            truncated=False,
            labels=next_labels,
            hud_features=None,
            observation_labels=observation_labels,
        )

        targets = auxiliary_targets(transition)

        torch.testing.assert_close(
            targets,
            torch.tensor([0.3, 0.1, 0.2, 0.3, 0.4, 0.25, 0.2 / 3.0, -0.25, -1.0, 0.0]),
        )

    def test_shaped_rewards_are_scaled_before_critic_returns(self):
        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import shape_transition_reward

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=0.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.3,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
        )
        transition = CollectedTransition(
            observation=pixels,
            action=torch.zeros(3).numpy(),
            next_observation=pixels,
            reward=1.0,
            terminated=False,
            truncated=False,
            labels=labels,
            hud_features=None,
        )

        reward = shape_transition_reward(transition, previous_progress=0.1)

        self.assertAlmostEqual(reward, 0.299)

    def test_route_deviation_and_heading_error_add_dense_training_penalties(self):
        import math
        from types import SimpleNamespace

        from training.train_policy import shape_transition_reward

        def make_transition(lateral_error, heading_error):
            labels = SimpleNamespace(
                tile_progress=0.0,
                finished=False,
                collision=False,
                off_track=False,
                damage=0.0,
                lateral_error=lateral_error,
                road_half_width=8.0,
                heading_error=heading_error,
            )
            return SimpleNamespace(reward=0.0, labels=labels)

        centered = shape_transition_reward(make_transition(0.0, 0.0), 0.0)
        edge_and_misaligned = shape_transition_reward(
            make_transition(8.0, math.pi / 2.0), 0.0
        )

        self.assertAlmostEqual(centered, -0.001)
        self.assertAlmostEqual(edge_and_misaligned, -0.101)


class TestRolloutActionMetrics(unittest.TestCase):
    def test_rollout_metrics_expose_steering_and_mutually_exclusive_pedals(self):
        from training.rollout import RolloutStorage
        from training.train_policy import rollout_action_metrics

        storage = RolloutStorage()
        for action in (
            torch.tensor([-0.2, 0.01, 0.0]),
            torch.tensor([0.4, 0.0, 0.3]),
            torch.tensor([0.1, 0.02, 0.0]),
        ):
            storage.add(
                observation=torch.zeros(4, 84, 84),
                action=action,
                pretransform_action=torch.zeros(2),
                log_probability=0.0,
                value=0.0,
                next_value=0.0,
                reward=0.0,
                terminated=False,
                truncated=False,
                auxiliary_targets=torch.zeros(10),
            )

        metrics = rollout_action_metrics(storage)

        self.assertAlmostEqual(metrics["brake_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["gas_mean"], 0.01)
        self.assertAlmostEqual(metrics["gas_max"], 0.02)
        self.assertEqual(metrics["pedal_overlap_fraction"], 0.0)
