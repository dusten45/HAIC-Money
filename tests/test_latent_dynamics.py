import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


class TestLatentDynamicsEnsemble(unittest.TestCase):
    def test_forward_and_predict_keep_batch_and_horizon_dimensions_finite(self):
        # Break caught: flattening candidate horizons mixes CEM action sequences
        # or returns a scalar uncertainty for the entire candidate population.
        from haic_agent.dynamics import LatentDynamicsEnsemble

        torch.manual_seed(31)
        model = LatentDynamicsEnsemble(num_members=3, hidden_size=32)
        latent = torch.randn(2, 4, 128)
        action = torch.tensor(
            [
                [[-0.2, 0.3, 0.0]] * 4,
                [[0.4, 0.7, 0.1]] * 4,
            ],
            dtype=torch.float32,
        )

        output = model(latent, action)
        prediction = model.predict(latent, action)

        self.assertEqual(output.next_latent_residual.shape, (3, 2, 4, 128))
        self.assertEqual(output.progress_delta.shape, (3, 2, 4))
        self.assertEqual(output.reward.shape, (3, 2, 4))
        self.assertEqual(output.collision_logits.shape, (3, 2, 4))
        self.assertEqual(output.off_track_logits.shape, (3, 2, 4))
        self.assertEqual(prediction.next_latent.shape, (2, 4, 128))
        self.assertEqual(prediction.progress_delta.shape, (2, 4))
        self.assertEqual(prediction.reward.shape, (2, 4))
        self.assertEqual(prediction.collision_probability.shape, (2, 4))
        self.assertEqual(prediction.off_track_probability.shape, (2, 4))
        self.assertEqual(prediction.uncertainty.shape, (2, 4))
        for value in (
            prediction.next_latent,
            prediction.progress_delta,
            prediction.reward,
            prediction.collision_probability,
            prediction.off_track_probability,
            prediction.uncertainty,
        ):
            self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.all((prediction.collision_probability >= 0.0) & (prediction.collision_probability <= 1.0)))
        self.assertTrue(torch.all((prediction.off_track_probability >= 0.0) & (prediction.off_track_probability <= 1.0)))

    def test_transition_builder_uses_next_decision_observation_and_label_boundary(self):
        # Break caught: using the pre-action frame or an intermediate raw tick
        # trains a four-tick dynamics model against a mismatched target.
        from haic_agent.observation import extract_hud_features
        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_dynamics import build_transition_batch

        class _Encoder:
            def encode_observation(self, observations):
                return observations[:, :1, :1, :1].reshape(-1, 1).repeat(1, 128)

        before = np.zeros((4, 84, 84), dtype=np.float32)
        after = np.full((4, 84, 84), 0.75, dtype=np.float32)
        transition = CollectedTransition(
            observation=before,
            action=np.array([0.25, 0.5, 0.0], dtype=np.float32),
            next_observation=after,
            reward=3.5,
            terminated=False,
            truncated=False,
            labels=TrainingLabels(
                speed=0.0,
                wheel_omega=(0.0, 0.0, 0.0, 0.0),
                steering_angle=0.0,
                yaw_rate=0.0,
                tile_progress=0.70,
                collision=True,
                damage=0.0,
                off_track=False,
                finished=False,
            ),
            hud_features=extract_hud_features(after),
        )

        batch = build_transition_batch(_Encoder(), [transition], previous_progress=[0.25])

        torch.testing.assert_close(batch.latent, torch.zeros(1, 128))
        torch.testing.assert_close(batch.next_latent, torch.full((1, 128), 0.75))
        torch.testing.assert_close(batch.action, torch.tensor([[0.25, 0.5, 0.0]]))
        torch.testing.assert_close(batch.progress_delta, torch.tensor([0.45]))
        torch.testing.assert_close(batch.reward, torch.tensor([3.5]))
        torch.testing.assert_close(batch.collision, torch.tensor([1.0]))
        torch.testing.assert_close(batch.off_track, torch.tensor([0.0]))

    def test_ensemble_disagreement_increases_for_out_of_distribution_inputs(self):
        # Break caught: averaging members before measuring spread masks the
        # epistemic uncertainty that should penalize unsupported CEM candidates.
        from haic_agent.dynamics import LatentDynamicsEnsemble
        from training.train_dynamics import DynamicsBatch, train_dynamics_step

        torch.manual_seed(5)
        model = LatentDynamicsEnsemble(num_members=4, hidden_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        in_distribution = torch.zeros(6, 128)
        in_actions = torch.zeros(6, 3)
        train_batch = DynamicsBatch(
            latent=torch.zeros(16, 128),
            action=torch.zeros(16, 3),
            next_latent=torch.zeros(16, 128),
            progress_delta=torch.zeros(16),
            reward=torch.zeros(16),
            collision=torch.zeros(16),
            off_track=torch.zeros(16),
        )
        for _ in range(20):
            train_dynamics_step(model, optimizer, train_batch)
        out_of_distribution = torch.full((6, 128), 12.0)
        out_actions = torch.tensor([[1.0, 1.0, 1.0]]).repeat(6, 1)

        with torch.no_grad():
            in_uncertainty = model.predict(in_distribution, in_actions).uncertainty.mean()
            out_uncertainty = model.predict(out_of_distribution, out_actions).uncertainty.mean()

        self.assertGreater(out_uncertainty, in_uncertainty)

    def test_synthetic_dynamics_loss_decreases_and_checkpoint_round_trips(self):
        # Break caught: a detached ensemble loss can report numbers while never
        # improving predictions or preserving learned weights for planning.
        from haic_agent.dynamics import LatentDynamicsEnsemble
        from training.train_dynamics import (
            DynamicsBatch,
            dynamics_loss,
            load_dynamics_checkpoint,
            save_dynamics_checkpoint,
            train_dynamics_step,
        )

        torch.manual_seed(19)
        latent = torch.randn(48, 128) * 0.1
        action = torch.zeros(48, 3)
        action[:, 0] = torch.linspace(-0.5, 0.5, 48)
        next_latent = latent.clone()
        next_latent[:, 0] += action[:, 0]
        progress_delta = action[:, 0] * 0.2
        reward = progress_delta * 2.0
        batch = DynamicsBatch(
            latent=latent,
            action=action,
            next_latent=next_latent,
            progress_delta=progress_delta,
            reward=reward,
            collision=(action[:, 0] > 0.35).float(),
            off_track=(action[:, 0] < -0.35).float(),
        )
        model = LatentDynamicsEnsemble(num_members=3, hidden_size=48)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        initial = dynamics_loss(model(batch.latent, batch.action), batch).total.item()
        for _ in range(80):
            metrics = train_dynamics_step(model, optimizer, batch)
        final = dynamics_loss(model(batch.latent, batch.action), batch).total.item()

        self.assertTrue(np.isfinite(metrics["total_loss"]))
        self.assertLess(final, initial * 0.55)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "dynamics.pt"
            save_dynamics_checkpoint(checkpoint, model, optimizer, step=80, metadata={"synthetic": True})
            restored = LatentDynamicsEnsemble(num_members=3, hidden_size=48)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=3e-3)
            saved = load_dynamics_checkpoint(checkpoint, restored, restored_optimizer)
            restored_prediction = restored.predict(batch.latent, batch.action)

        self.assertEqual(saved["step"], 80)
        self.assertEqual(saved["metadata"], {"synthetic": True})
        self.assertTrue(torch.isfinite(restored_prediction.uncertainty).all())

    def test_collection_policy_loads_the_trained_ppo_checkpoint_and_records_source(self):
        # Break caught: initializing a fresh visual policy during dynamics
        # collection changes the latent/action distribution that CEM will see.
        from haic_agent.networks import VisualActorCritic
        from training.train_dynamics import collection_policy

        torch.manual_seed(43)
        trained_policy = VisualActorCritic()
        expected = next(trained_policy.parameters()).detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            torch.save({"model_state": trained_policy.state_dict(), "step": 12}, checkpoint)
            loaded_policy, source = collection_policy(checkpoint)

        self.assertEqual(source, str(checkpoint))
        self.assertFalse(loaded_policy.training)
        torch.testing.assert_close(expected, next(loaded_policy.parameters()).detach())
        with self.assertRaisesRegex(ValueError, "policy checkpoint"):
            collection_policy(None)


if __name__ == "__main__":
    unittest.main()
