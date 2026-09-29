import importlib
import importlib.util
import unittest

import torch


def _implementation():
    if importlib.util.find_spec("training.lagrangian_ppo") is None:
        return None
    return importlib.import_module("training.lagrangian_ppo")


class LagrangianPPOTests(unittest.TestCase):
    def setUp(self):
        self.api = _implementation()
        self.assertIsNotNone(
            self.api,
            "training.lagrangian_ppo must provide the training-only PPO interface",
        )

    def _add_step(
        self,
        storage,
        *,
        reward,
        reward_value,
        next_reward_value,
        cost,
        cost_value,
        next_cost_value,
        boundary,
        pixel_value=0.0,
        observation=None,
        pretransform_action=None,
        old_log_probability=0.0,
    ):
        storage.add(
            observation=(
                torch.full((4, 84, 84), pixel_value)
                if observation is None
                else observation
            ),
            action=torch.zeros(3),
            pretransform_action=(
                torch.zeros(2)
                if pretransform_action is None
                else pretransform_action
            ),
            old_log_probability=old_log_probability,
            reward=reward,
            reward_value=reward_value,
            next_reward_value=next_reward_value,
            cost=cost,
            cost_value=cost_value,
            next_cost_value=next_cost_value,
            boundary=boundary,
            auxiliary_targets=torch.zeros(10),
        )

    def test_collector_cut_bootstraps_both_critics_but_stops_gae_recursion(self):
        storage = self.api.LagrangianRolloutStorage()
        self._add_step(
            storage,
            reward=0.5,
            reward_value=2.0,
            next_reward_value=4.0,
            cost=0.1,
            cost_value=0.4,
            next_cost_value=3.0,
            boundary=self.api.EpisodeBoundary(collector_truncated=True),
        )

        storage.compute_returns_and_advantages(
            reward_gamma=0.99, cost_gamma=1.0, gae_lambda=0.95
        )
        step = storage.steps[0]

        self.assertAlmostEqual(step.reward_advantage, 2.46, places=5)
        self.assertAlmostEqual(step.cost_advantage, 2.7, places=5)
        self.assertAlmostEqual(step.reward_return, 4.46, places=5)
        self.assertAlmostEqual(step.cost_return, 3.1, places=5)

    def test_true_episode_time_limit_does_not_bootstrap_either_episode_return(self):
        storage = self.api.LagrangianRolloutStorage()
        self._add_step(
            storage,
            reward=0.5,
            reward_value=2.0,
            next_reward_value=4.0,
            cost=0.1,
            cost_value=0.4,
            next_cost_value=3.0,
            boundary=self.api.EpisodeBoundary(episode_time_limit=True),
        )

        storage.compute_returns_and_advantages(
            reward_gamma=0.99, cost_gamma=1.0, gae_lambda=0.95
        )
        step = storage.steps[0]

        self.assertAlmostEqual(step.reward_advantage, -1.5, places=5)
        self.assertAlmostEqual(step.cost_advantage, -0.3, places=5)

    def test_gae_does_not_cross_a_completed_episode_boundary(self):
        storage = self.api.LagrangianRolloutStorage()
        self._add_step(
            storage,
            reward=1.0,
            reward_value=0.0,
            next_reward_value=100.0,
            cost=0.2,
            cost_value=0.0,
            next_cost_value=100.0,
            boundary=self.api.EpisodeBoundary(terminated=True),
        )
        self._add_step(
            storage,
            reward=2.0,
            reward_value=0.0,
            next_reward_value=0.0,
            cost=0.4,
            cost_value=0.0,
            next_cost_value=0.0,
            boundary=self.api.EpisodeBoundary(collector_truncated=True),
            pixel_value=1.0,
        )

        storage.compute_returns_and_advantages(
            reward_gamma=0.99, cost_gamma=1.0, gae_lambda=0.95
        )

        self.assertAlmostEqual(storage.steps[0].reward_advantage, 1.0)
        self.assertAlmostEqual(storage.steps[0].cost_advantage, 0.2)
        self.assertAlmostEqual(storage.steps[1].reward_advantage, 2.0)
        self.assertAlmostEqual(storage.steps[1].cost_advantage, 0.4)

    def test_policy_advantage_combines_raw_reward_and_cost_gae_before_normalizing(self):
        storage = self.api.LagrangianRolloutStorage()
        for reward, cost in ((1.0, 1.0), (2.0, 0.0)):
            self._add_step(
                storage,
                reward=reward,
                reward_value=0.0,
                next_reward_value=0.0,
                cost=cost,
                cost_value=0.0,
                next_cost_value=0.0,
                boundary=self.api.EpisodeBoundary(terminated=True),
            )
        storage.compute_returns_and_advantages(
            reward_gamma=0.99, cost_gamma=1.0, gae_lambda=0.95
        )

        raw_batch = storage.batch(lagrange_multiplier=0.5, normalize_advantages=False)
        normalized_batch = storage.batch(lagrange_multiplier=0.5)

        torch.testing.assert_close(raw_batch.advantages, torch.tensor([0.5, 2.0]))
        torch.testing.assert_close(
            normalized_batch.advantages, torch.tensor([-1.0, 1.0])
        )

    def test_actor_only_export_strictly_loads_without_cost_head(self):
        from haic_agent.networks import VisualActorCritic

        torch.manual_seed(7)
        actor = VisualActorCritic(use_hud=False)
        model = self.api.LagrangianActorCritic(actor)
        exported = model.actor_state_dict()
        inference_actor = VisualActorCritic(use_hud=False)
        inference_actor.load_state_dict(exported, strict=True)

        self.assertEqual(set(exported), set(actor.state_dict()))
        self.assertFalse(any("cost" in key for key in exported))
        for key, value in actor.state_dict().items():
            torch.testing.assert_close(exported[key], value)
        self.assertEqual(
            set(model.state_dict()),
            {f"policy.{key}" for key in actor.state_dict()}
            | {f"cost_value_head.{key}" for key in model.cost_value_head.state_dict()},
        )

    def test_cost_value_head_uses_actor_latent_features(self):
        from haic_agent.networks import VisualActorCritic

        model = self.api.LagrangianActorCritic(VisualActorCritic(use_hud=False))
        with torch.no_grad():
            model.cost_value_head.weight.fill_(0.01)
        output = model(torch.rand(2, 4, 84, 84))

        self.assertEqual(output.cost_value.shape, (2,))
        self.assertEqual(output.latent.shape, (2, 128))
        output.cost_value.sum().backward()
        self.assertIsNotNone(model.policy.fusion[0].weight.grad)
        self.assertGreater(float(model.policy.fusion[0].weight.grad.abs().sum()), 0.0)

    def test_wrapper_forwards_visual_features_for_existing_train_only_recorder(self):
        from haic_agent.networks import VisualActorCritic

        policy = VisualActorCritic(use_visual_features=True)
        model = self.api.LagrangianActorCritic(policy)
        model(torch.zeros(1, 4, 84, 84))

        self.assertTrue(
            hasattr(model, "last_visual_features"),
            "the training wrapper must expose the existing visual-feature recorder input",
        )
        self.assertIs(model.last_visual_features, policy.last_visual_features)

    def test_ppo_update_trains_actor_and_cost_value_head(self):
        from haic_agent.networks import VisualActorCritic
        from training.ppo import PPOConfig

        torch.manual_seed(13)
        model = self.api.LagrangianActorCritic(VisualActorCritic(use_hud=False))
        storage = self.api.LagrangianRolloutStorage()
        for index in range(8):
            observation = torch.rand(4, 84, 84)
            with torch.no_grad():
                output = model(observation.unsqueeze(0))
                _, log_probability, pretransform_action = model.sample_actions_with_pretransform(output)
            self._add_step(
                storage,
                reward=float(index + 1) / 10.0,
                reward_value=float(output.value.item()),
                next_reward_value=0.0,
                cost=0.1 if index in (2, 6) else 0.0,
                cost_value=float(output.cost_value.item()),
                next_cost_value=0.0,
                boundary=self.api.EpisodeBoundary(terminated=index == 7),
                pixel_value=float(observation.mean()),
                observation=observation,
                pretransform_action=pretransform_action[0],
                old_log_probability=float(log_probability.item()),
            )
        storage.compute_returns_and_advantages(
            reward_gamma=0.99, cost_gamma=1.0, gae_lambda=0.95
        )
        actor_before = model.policy.policy_mean.weight.detach().clone()
        cost_head_before = model.cost_value_head.weight.detach().clone()
        updater = self.api.LagrangianPPOUpdater(
            model, PPOConfig(learning_rate=1e-4, epochs=1, minibatch_size=8)
        )

        metrics = updater.update(storage, lagrange_multiplier=30.0)

        self.assertFalse(torch.equal(actor_before, model.policy.policy_mean.weight.detach()))
        self.assertFalse(torch.equal(cost_head_before, model.cost_value_head.weight.detach()))
        self.assertGreater(metrics["cost_value_loss"], 0.0)
        self.assertAlmostEqual(metrics["lagrange_multiplier"], 30.0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
