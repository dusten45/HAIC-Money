import unittest

import torch


class TestVisualActorCritic(unittest.TestCase):
    def test_pixel_encoder_fuses_full_frame_and_hud_into_finite_contract(self):
        # Break caught: replacing the scene branch with HUD-only features makes
        # scenery outside the HUD unable to influence the policy latent.
        from haic_agent.networks import VisualActorCritic

        torch.manual_seed(7)
        model = VisualActorCritic(use_hud=True)
        observations = torch.rand(2, 4, 84, 84)
        changed_scenery = observations.clone()
        changed_scenery[:, :, :60, :] = 1.0 - changed_scenery[:, :, :60, :]

        output = model(observations)
        changed_output = model(changed_scenery)

        self.assertEqual(output.latent.shape, (2, 128))
        self.assertEqual(output.action_mean.shape, (2, 2))
        self.assertEqual(output.action_log_std.shape, (2, 2))
        self.assertEqual(output.value.shape, (2,))
        self.assertEqual(output.auxiliary_predictions.shape, (2, 10))
        self.assertTrue(torch.isfinite(output.latent).all())
        self.assertTrue(torch.isfinite(output.action_mean).all())
        self.assertTrue(torch.isfinite(output.action_log_std).all())
        self.assertTrue(torch.isfinite(output.value).all())
        self.assertTrue(torch.isfinite(output.auxiliary_predictions).all())
        self.assertFalse(torch.allclose(output.latent, changed_output.latent))

    def test_sampled_environment_actions_stay_bounded_with_finite_log_probability(self):
        # Break caught: emitting unconstrained Normal samples can send gas,
        # brake, or steering outside the simulator action space.
        from haic_agent.networks import VisualActorCritic

        torch.manual_seed(11)
        model = VisualActorCritic()
        output = model(torch.rand(16, 4, 84, 84))
        actions, log_probability = model.sample_actions(output)
        recomputed_log_probability = model.log_probability(
            actions, output.action_mean, output.action_log_std
        )

        self.assertTrue(torch.isfinite(actions).all())
        self.assertEqual(actions.shape, (16, 3))
        self.assertTrue(torch.isfinite(log_probability).all())
        self.assertTrue(torch.isfinite(recomputed_log_probability).all())
        self.assertTrue(torch.all(actions[:, 0] >= -1.0))
        self.assertTrue(torch.all(actions[:, 0] <= 1.0))
        self.assertTrue(torch.all(actions[:, 1:] >= 0.0))
        self.assertTrue(torch.all(actions[:, 1:] <= 1.0))
        self.assertTrue(torch.all(actions[:, 1] <= 0.0200001))
        self.assertTrue(torch.all(actions[:, 2] <= 0.0300001))
        self.assertTrue(torch.all((actions[:, 1] == 0.0) | (actions[:, 2] == 0.0)))
        torch.testing.assert_close(log_probability, recomputed_log_probability)

    def test_signed_longitudinal_action_maps_to_mutually_exclusive_pedals(self):
        from haic_agent.networks import VisualActorCritic

        coordinates = torch.tensor([[0.0, 0.25], [0.0, -0.5], [0.0, 0.0]])

        actions = VisualActorCritic._bound_actions(coordinates)

        torch.testing.assert_close(actions[:, 0], torch.zeros(3))
        torch.testing.assert_close(
            actions[:, 1], torch.tensor([0.02 * torch.tanh(torch.tensor(0.25)), 0.0, 0.0])
        )
        torch.testing.assert_close(
            actions[:, 2], torch.tensor([0.0, 0.03 * torch.tanh(torch.tensor(0.5)), 0.0])
        )
        unconstrained, _ = VisualActorCritic._unbound_actions(actions)
        torch.testing.assert_close(unconstrained, coordinates, atol=1e-6, rtol=0.0)

    def test_initial_policy_uses_calibrated_left_steer_and_gentle_forward_bias(self):
        from haic_agent.networks import VisualActorCritic

        torch.manual_seed(23)
        model = VisualActorCritic()
        output = model(torch.zeros(1, 4, 84, 84))

        self.assertEqual(output.action_mean.shape, (1, 2))
        torch.testing.assert_close(
            output.action_log_std.exp(), torch.tensor([[0.35, 0.4]]), atol=1e-5, rtol=0.0
        )
        action = model.deterministic_actions(output)[0]
        self.assertAlmostEqual(float(action[0]), -0.12, delta=0.02)
        self.assertGreaterEqual(float(action[1]), 0.009)
        self.assertLessEqual(float(action[1]), 0.011)
        self.assertEqual(float(action[2]), 0.0)

    def test_initial_signed_pedal_distribution_explores_braking(self):
        from haic_agent.networks import VisualActorCritic

        torch.manual_seed(31)
        model = VisualActorCritic()
        output = model(torch.zeros(256, 4, 84, 84))
        actions, _ = model.sample_actions(output)
        brake_fraction = float((actions[:, 2] > 0.001).float().mean())

        self.assertGreater(brake_fraction, 0.03)
        self.assertLess(brake_fraction, 0.2)
        self.assertLessEqual(float(actions[:, 2].max()), 0.0300001)
        self.assertTrue(torch.all((actions[:, 1] == 0.0) | (actions[:, 2] == 0.0)))

    def test_extreme_policy_scale_and_saturated_actions_keep_transformed_density_finite(self):
        # Break caught: an unconstrained log standard deviation can overflow a
        # Normal scale, while saturated tanh/sigmoid actions make log-Jacobians
        # negative infinity during PPO collection.
        from haic_agent.networks import VisualActorCritic

        model = VisualActorCritic()
        with torch.no_grad():
            model.policy_log_std.fill_(100.0)
        output = model(torch.zeros(4, 4, 84, 84))
        actions, sampled_log_probability = model.sample_actions(output)
        saturated_actions = torch.tensor(
            [[1.0, 0.02, 0.0], [-1.0, 0.0, 0.03]], dtype=torch.float32
        )
        boundary_log_probability = model.log_probability(
            saturated_actions,
            output.action_mean[:2],
            output.action_log_std[:2],
        )

        self.assertTrue(torch.isfinite(output.action_log_std).all())
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(torch.isfinite(sampled_log_probability).all())
        self.assertTrue(torch.isfinite(boundary_log_probability).all())

    def test_encoder_contract_returns_128_latents_for_pixel_batches(self):
        # Break caught: changing the encoder width breaks downstream dynamics
        # models that consume the frozen 128-dimensional latent contract.
        from haic_agent.networks import VisualActorCritic

        latent = VisualActorCritic().encode_observation(torch.zeros(3, 4, 84, 84))

        self.assertEqual(latent.shape, (3, 128))
        self.assertTrue(torch.isfinite(latent).all())


if __name__ == "__main__":
    unittest.main()
