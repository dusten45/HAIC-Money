import copy
import math
import unittest

import torch
from torch import nn
import numpy as np

from haic.algorithms.rlpd.agent import (
    PixelRLPDAgent,
    RLPDConfig,
    sample_balanced_batch,
    sac_bellman_target,
)
from haic.algorithms.rlpd.model import (
    NUM_QS,
    PixelActor,
    PixelCritic,
    PixelEncoder,
    sample_squashed_normal,
    select_target_q_values,
    squashed_normal_log_prob,
)
from haic.algorithms.rlpd.replay import FrameStackReplay
from tests.test_rlpd_replay import transition


class TestRLPDModel(unittest.TestCase):
    def test_pixel_encoder_accepts_uint8_and_normalized_float(self):
        encoder = PixelEncoder().eval()
        pixels = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
        with torch.inference_mode():
            uint8_features = encoder(pixels)
            float_features = encoder(pixels.float() / 255.0)
        self.assertEqual(uint8_features.shape, (2, 50))
        self.assertTrue(torch.allclose(uint8_features, float_features, atol=1e-6, rtol=0))
        with self.assertRaises(ValueError):
            encoder(torch.zeros((2, 3, 84, 84)))

    def test_squashed_log_density_matches_manual_normal_and_jacobian(self):
        mean = torch.tensor([[0.2, -0.4, 0.6]], dtype=torch.float64, requires_grad=True)
        log_std = torch.tensor([[-0.1, 0.3, -0.6]], dtype=torch.float64, requires_grad=True)
        pre_tanh = torch.tensor([[0.7, -1.1, 2.4]], dtype=torch.float64, requires_grad=True)
        actual = squashed_normal_log_prob(mean, log_std, pre_tanh)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std).log_prob(pre_tanh)
        jacobian = 2.0 * (math.log(2.0) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh))
        expected = (normal - jacobian).sum(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-10, rtol=0))
        actual.sum().backward()
        for value in (mean.grad, log_std.grad, pre_tanh.grad):
            self.assertIsNotNone(value)
            self.assertTrue(torch.isfinite(value).all())

    def test_extreme_pre_tanh_samples_have_finite_density_and_policy_gradients(self):
        mean = torch.tensor([[19.0, -19.0, 0.1]], requires_grad=True)
        log_std = torch.tensor([[-20.0, 2.0, -1.0]], requires_grad=True)
        action, log_prob, pre_tanh = sample_squashed_normal(
            mean,
            log_std,
            generator=torch.Generator().manual_seed(9),
        )
        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertTrue(torch.isfinite(pre_tanh).all())
        log_prob.sum().backward()
        self.assertGreater(float(mean.grad.abs().sum()), 0.0)
        self.assertGreater(float(log_std.grad.abs().sum()), 0.0)

    def test_tanh_mode_is_bounded_and_not_sampled_noise(self):
        actor = PixelActor().eval()
        with torch.inference_mode():
            mean, _ = actor.distribution(torch.zeros((2, 4, 84, 84), dtype=torch.uint8))
            first = actor.sample(torch.zeros((2, 4, 84, 84), dtype=torch.uint8), deterministic=True)[0]
            second = actor.sample(torch.zeros((2, 4, 84, 84), dtype=torch.uint8), deterministic=True)[0]
        self.assertTrue(torch.allclose(first, torch.tanh(mean)))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.all(first >= -1.0))
        self.assertTrue(torch.all(first <= 1.0))

    def test_critic_has_ten_independent_heads_and_selected_min_uses_only_indices(self):
        critic = PixelCritic()
        values = critic(
            torch.zeros((3, 4, 84, 84), dtype=torch.uint8),
            torch.zeros((3, 3)),
        )
        self.assertEqual(values.shape, (3, NUM_QS))
        self.assertEqual(len(critic.q_heads), 10)
        values = torch.tensor([[0.0, 1.0, -2.0, 3.0], [3.0, -1.0, 2.0, 4.0]])
        self.assertTrue(torch.equal(select_target_q_values(values, torch.tensor([1])), torch.tensor([[1.0], [-1.0]])))
        self.assertTrue(torch.equal(
            select_target_q_values(values, torch.tensor([[0, 2], [1, 3]])),
            torch.tensor([[-2.0], [-1.0]]),
        ))
        with self.assertRaises(ValueError):
            select_target_q_values(values, torch.tensor([], dtype=torch.long))
        generator = torch.Generator().manual_seed(44)
        selections = {
            int(torch.randperm(10, generator=generator)[0])
            for _ in range(32)
        }
        self.assertGreater(len(selections), 1)

    def test_temperature_contract_matches_pinned_source_convention(self):
        beta = torch.tensor(math.log(0.1), requires_grad=True)
        alpha = beta.exp()
        entropy = torch.tensor([[0.5], [1.5]])
        target_entropy = -1.5
        loss = (alpha * (entropy - target_entropy)).mean()
        loss.backward()
        expected = float(alpha.detach()) * float((entropy - target_entropy).mean())
        self.assertAlmostEqual(float(beta.grad), expected, places=7)
        self.assertGreater(float(beta.grad), 0.0)
        self.assertAlmostEqual(target_entropy, -3 / 2)

    def test_one_step_target_bootstraps_time_limit_and_masks_true_terminal(self):
        reward = torch.tensor([[2.0], [-3.0]])
        # Row 0 is a pure timeout (truncated=True, terminal=False); row 1 is terminal.
        terminal = torch.tensor([[False], [True]])
        next_q = torch.tensor([[5.0], [99.0]])
        target = sac_bellman_target(reward, terminal, next_q, gamma=0.99)
        self.assertTrue(torch.allclose(target, torch.tensor([[6.95], [-3.0]]), atol=1e-6))

    def test_entropy_backup_changes_only_the_optional_bellman_term(self):
        reward = torch.tensor([[1.0]])
        terminal = torch.tensor([[False]])
        next_q = torch.tensor([[4.0]])
        log_prob = torch.tensor([[-2.0]])
        alpha = torch.tensor(0.25)
        default = sac_bellman_target(reward, terminal, next_q, gamma=0.9)
        with_entropy = sac_bellman_target(
            reward,
            terminal,
            next_q,
            gamma=0.9,
            backup_entropy=True,
            alpha=alpha,
            next_log_prob=log_prob,
        )
        self.assertAlmostEqual(float(default), 4.6, places=6)
        self.assertAlmostEqual(float(with_entropy), 5.05, places=6)
        with self.assertRaises(ValueError):
            sac_bellman_target(
                reward,
                terminal,
                next_q,
                gamma=0.9,
                backup_entropy=True,
            )

    def test_agent_update_finite_and_actor_image_encoder_gradient_stopped(self):
        torch.manual_seed(17)
        agent = PixelRLPDAgent(RLPDConfig(batch_size=2), seed=3)
        cnn_before = {
            key: value.detach().clone()
            for key, value in agent.critic.encoder.convolution.state_dict().items()
        }
        q_before = [
            {key: value.detach().clone() for key, value in head.state_dict().items()}
            for head in agent.critic.q_heads
        ]
        observation = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
        batch = {
            "observation": observation.numpy(),
            "next_observation": observation.flip(-1).numpy(),
            "action": torch.zeros((2, 3)).numpy(),
            "proposed_action": torch.zeros((2, 3)).numpy(),
            "reward": torch.zeros(2).numpy(),
            "terminal": torch.zeros(2, dtype=torch.bool).numpy(),
            "source": ["online", "offline"],
        }
        metrics = agent.update(batch)
        self.assertEqual(metrics["gradient_steps"], 1)
        self.assertEqual(metrics["offline_samples"], 1)
        self.assertEqual(metrics["online_samples"], 1)
        self.assertTrue(all(math.isfinite(float(value)) for value in metrics.values()))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in agent.actor.encoder.convolution.parameters()
        ))
        self.assertTrue(any(
            parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
            for parameter in agent.actor.encoder.projection.parameters()
        ))
        for key, value in agent.actor.encoder.convolution.state_dict().items():
            self.assertTrue(torch.equal(value, cnn_before[key]))
        for before, head in zip(q_before, agent.critic.q_heads):
            self.assertTrue(any(
                not torch.equal(value, head.state_dict()[key])
                for key, value in before.items()
            ))
        self.assertGreater(metrics["actor_grad_norm"], 0.0)
        self.assertGreater(metrics["critic_grad_norm"], 0.0)

    def test_target_ema_tau_endpoints(self):
        agent = PixelRLPDAgent(RLPDConfig(batch_size=2, tau=1.0), seed=0)
        with torch.no_grad():
            for parameter in agent.critic.parameters():
                parameter.fill_(0.25)
            for parameter in agent.target_critic.parameters():
                parameter.zero_()
        agent._update_target()
        for source, target in zip(agent.critic.parameters(), agent.target_critic.parameters()):
            self.assertTrue(torch.equal(source, target))

        agent.config = copy.copy(agent.config)
        self.assertEqual(len(agent.critic.q_heads), 10)

    def test_checkpoint_restore_matches_next_replay_batch_update_and_action(self):
        def replay_pair():
            offline = FrameStackReplay(6, seed=50, source="offline")
            online = FrameStackReplay(6, seed=51, source="online")
            for replay, source_episode, base, geometry_seed in (
                (offline, 100, 10, 9101),
                (online, 200, 20, 9201),
            ):
                replay.add(transition(
                    episode=source_episode,
                    step_index=0,
                    base=base,
                    terminated=True,
                    next_observation=np.full((4, 84, 84), 230, dtype=np.uint8),
                    geometry_seed=geometry_seed,
                ))
                replay.finalize() if replay.source == "offline" else None
            return offline, online

        torch.manual_seed(101)
        uninterrupted = PixelRLPDAgent(RLPDConfig(batch_size=2), seed=4)
        offline_a, online_a = replay_pair()
        checkpoint = copy.deepcopy(uninterrupted.checkpoint_state(
            offline_replay=offline_a,
            online_replay=online_a,
            trainer_state={"offline_dataset_sha256": "dataset"},
        ))
        expected_batch = sample_balanced_batch(
            offline_a, online_a, batch_size=2, offline_count=1
        )
        expected_metrics = uninterrupted.update(expected_batch)
        observation = np.full((4, 84, 84), 0.5, dtype=np.float32)
        expected_action = uninterrupted.act(observation, deterministic=False)
        expected_actor = {
            key: value.detach().clone()
            for key, value in uninterrupted.actor.state_dict().items()
        }

        torch.manual_seed(999)
        restored = PixelRLPDAgent(RLPDConfig(batch_size=2), seed=4)
        offline_b, online_b = replay_pair()
        trainer = restored.load_checkpoint_state(
            checkpoint,
            offline_replay=offline_b,
            online_replay=online_b,
        )
        self.assertEqual(trainer, {"offline_dataset_sha256": "dataset"})
        actual_batch = sample_balanced_batch(
            offline_b, online_b, batch_size=2, offline_count=1
        )
        self.assertTrue(np.array_equal(actual_batch["indices"], expected_batch["indices"]))
        for key in ("observation", "action", "reward", "next_observation", "terminal", "source"):
            self.assertTrue(np.array_equal(actual_batch[key], expected_batch[key]))
        actual_metrics = restored.update(actual_batch)
        self.assertEqual(actual_metrics, expected_metrics)
        actual_action = restored.act(observation, deterministic=False)
        self.assertTrue(np.array_equal(actual_action, expected_action))
        for key, value in restored.actor.state_dict().items():
            self.assertTrue(torch.equal(value, expected_actor[key]))


if __name__ == "__main__":
    unittest.main()
