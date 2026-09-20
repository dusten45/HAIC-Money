import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class TestRolloutAdvantages(unittest.TestCase):
    def test_terminal_transition_does_not_bootstrap_from_the_next_episode(self):
        # Break caught: carrying GAE across a terminal state lets the next
        # episode's reward contaminate a completed episode's return.
        from training.rollout import RolloutStorage

        storage = RolloutStorage()
        observation = torch.zeros(4, 84, 84)
        action = torch.tensor([0.0, 0.5, 0.0])
        storage.add(
            observation=observation,
            action=action,
            log_probability=0.0,
            value=0.5,
            next_value=999.0,
            reward=1.0,
            terminated=True,
            truncated=False,
            auxiliary_targets=torch.zeros(7),
        )
        storage.add(
            observation=observation,
            action=action,
            log_probability=0.0,
            value=0.0,
            next_value=0.0,
            reward=100.0,
            terminated=False,
            truncated=True,
            auxiliary_targets=torch.zeros(7),
        )

        storage.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)

        self.assertAlmostEqual(storage.steps[0].advantage, 0.5)
        self.assertAlmostEqual(storage.steps[0].return_, 1.0)


class TestPPOUpdate(unittest.TestCase):
    def test_best_checkpoint_rejects_lower_ranked_candidates_and_persists_the_winner(self):
        # Break caught: unconditional policy.pt writes can replace a completed
        # fast-lap checkpoint with a lower-finish-rate or slower-lap candidate.
        from haic_agent.networks import VisualActorCritic
        from training.ppo import PPOConfig, PPOUpdater
        from training.train_policy import save_best_checkpoint

        winning_metrics = {
            "finish_rate": 0.5,
            "median_finished_lap_time_s": 20.0,
            "mean_progress": 0.2,
        }
        lower_finish_metrics = {
            "finish_rate": 0.4,
            "median_finished_lap_time_s": 1.0,
            "mean_progress": 1.0,
        }
        slower_lap_metrics = {
            "finish_rate": 0.5,
            "median_finished_lap_time_s": 30.0,
            "mean_progress": 1.0,
        }
        equal_rank_lower_progress = {
            "finish_rate": 0.5,
            "median_finished_lap_time_s": 20.0,
            "mean_progress": 0.1,
        }
        model = VisualActorCritic()
        updater = PPOUpdater(model, PPOConfig())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            self.assertTrue(
                save_best_checkpoint(
                    checkpoint, model, updater, step=10, tune_metrics=winning_metrics, metadata={}
                )
            )
            with torch.no_grad():
                next(model.parameters()).add_(1.0)
            self.assertFalse(
                save_best_checkpoint(
                    checkpoint, model, updater, step=20, tune_metrics=lower_finish_metrics, metadata={}
                )
            )
            self.assertFalse(
                save_best_checkpoint(
                    checkpoint, model, updater, step=30, tune_metrics=slower_lap_metrics, metadata={}
                )
            )
            self.assertFalse(
                save_best_checkpoint(
                    checkpoint, model, updater, step=40,
                    tune_metrics=equal_rank_lower_progress, metadata={}
                )
            )
            saved = torch.load(checkpoint, map_location="cpu")

        self.assertEqual(saved["step"], 10)
        self.assertEqual(saved["metadata"]["tune_metrics"], winning_metrics)

    def test_tune_evaluation_reports_auxiliary_estimation_error_from_pixel_predictions(self):
        # Break caught: a HUD ablation that reports only driving outcome cannot
        # show whether its visual auxiliary predictions contributed anything.
        from haic_agent.networks import VisualActorCritic
        from haic_agent.observation import extract_hud_features
        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import evaluate_policy

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=1.0,
            wheel_omega=(2.0, 3.0, 4.0, 5.0),
            steering_angle=0.1,
            yaw_rate=0.2,
            tile_progress=0.1,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
        )
        transition = CollectedTransition(
            observation=pixels,
            action=torch.tensor([0.0, 0.0, 0.0]).numpy(),
            next_observation=pixels,
            reward=0.0,
            terminated=True,
            truncated=False,
            labels=labels,
            hud_features=extract_hud_features(pixels),
        )

        class _WrappedEnvironment:
            def close(self):
                pass

        class _Environment:
            environment = _WrappedEnvironment()

            class unwrapped:
                finish_time_s = None

            def reset(self):
                return pixels, {}

            def step_transition(self, action):
                return transition

        with patch("training.train_policy.create_training_environment", return_value=_Environment()):
            metrics = evaluate_policy(VisualActorCritic(), [(1, 1)], max_decisions=1)

        self.assertIn("auxiliary_mse", metrics)
        self.assertGreaterEqual(metrics["auxiliary_mse"], 0.0)

    def test_checkpoint_selection_prioritizes_completion_then_lap_time_then_progress(self):
        # Break caught: using reward as a checkpoint tie-breaker can choose a
        # faster-rewarding policy over a faster completed lap.
        from training.train_policy import selection_score

        completed_slow = selection_score(
            {"finish_rate": 0.5, "median_finished_lap_time_s": 20.0, "mean_progress": 0.9}
        )
        completed_fast = selection_score(
            {"finish_rate": 0.5, "median_finished_lap_time_s": 15.0, "mean_progress": 0.1}
        )
        unfinished = selection_score(
            {"finish_rate": 0.0, "median_finished_lap_time_s": None, "mean_progress": 0.99}
        )

        self.assertGreater(completed_fast, completed_slow)
        self.assertGreater(completed_slow, unfinished)

    def test_default_training_split_uses_simulator_valid_positive_track_ids(self):
        # Break caught: a reproducible default split containing track zero
        # fails before the first simulator-backed training decision.
        from training.train_policy import DEFAULT_SPLIT

        for episode in (*DEFAULT_SPLIT.train, *DEFAULT_SPLIT.tune, *DEFAULT_SPLIT.held_out):
            self.assertGreaterEqual(episode[0], 1)

    def test_training_cleanup_closes_the_collector_wrapped_environment(self):
        # Break caught: rollout collection leaves the local simulator open when
        # CollectingEnvironment itself does not define a close method.
        from training.train_policy import close_training_environment

        class _WrappedEnvironment:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        class _Collector:
            def __init__(self):
                self.environment = _WrappedEnvironment()

        collector = _Collector()
        close_training_environment(collector)

        self.assertTrue(collector.environment.closed)

    def test_one_real_rollout_minibatch_updates_parameters_with_finite_losses(self):
        # Break caught: a detached PPO loss can look finite while leaving the
        # policy parameters unchanged after an optimization step.
        from haic_agent.networks import VisualActorCritic
        from training.ppo import PPOConfig, PPOUpdater
        from training.rollout import RolloutStorage

        torch.manual_seed(13)
        model = VisualActorCritic()
        storage = RolloutStorage()
        for index in range(8):
            observation = torch.rand(4, 84, 84)
            next_observation = torch.rand(4, 84, 84)
            with torch.no_grad():
                output = model(observation.unsqueeze(0))
                action, log_probability = model.sample_actions(output)
                next_value = model(next_observation.unsqueeze(0)).value.item()
            storage.add(
                observation=observation,
                action=action.squeeze(0),
                log_probability=log_probability.item(),
                value=output.value.item(),
                next_value=next_value,
                reward=float(index + 1) / 10.0,
                terminated=False,
                truncated=index == 7,
                auxiliary_targets=torch.zeros(7),
            )
        storage.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
        before = next(model.parameters()).detach().clone()

        metrics = PPOUpdater(model, PPOConfig(epochs=1, minibatch_size=8)).update(storage)

        self.assertFalse(torch.equal(before, next(model.parameters()).detach()))
        for value in metrics.values():
            self.assertTrue(torch.isfinite(torch.tensor(value)))

    def test_checkpoint_round_trip_restores_training_step_and_model_weights(self):
        # Break caught: a saved training checkpoint that omits model state or
        # its step cannot resume a reproducible PPO run.
        from haic_agent.networks import VisualActorCritic
        from training.ppo import PPOConfig, PPOUpdater
        from training.train_policy import load_checkpoint, save_checkpoint

        torch.manual_seed(17)
        model = VisualActorCritic()
        updater = PPOUpdater(model, PPOConfig())
        expected = next(model.parameters()).detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            save_checkpoint(checkpoint, model, updater, step=23, metadata={"smoke": True})
            restored_model = VisualActorCritic()
            restored_updater = PPOUpdater(restored_model, PPOConfig())
            saved = load_checkpoint(checkpoint, restored_model, restored_updater)

        self.assertEqual(saved["step"], 23)
        self.assertEqual(saved["metadata"], {"smoke": True})
        torch.testing.assert_close(expected, next(restored_model.parameters()).detach())


if __name__ == "__main__":
    unittest.main()
