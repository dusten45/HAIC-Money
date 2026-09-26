import copy
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from common_adapter import Transition
from scripts.diagnose.dreamerv3 import diagnose_world_model
from dreamer_v3 import (
    Actor,
    balanced_kl_loss,
    CategoricalRSSM,
    continuation_weights,
    Critic,
    continue_prediction_loss,
    ConvDecoder,
    ConvEncoder,
    DreamerV3Agent,
    DreamerV3Config,
    ExportedDreamerV3Actor,
    LayerNormGRUCell,
    NoValidSequenceError,
    RewardHead,
    lambda_returns,
    Uint8SequenceReplay,
    load_exported_actor,
    symexp,
    symlog,
)
from agent import Agent, DREAMERV3_ACTOR_FORMAT


class TestDreamerV3(unittest.TestCase):
    def test_symlog_symexp_reversibility(self):
        x = torch.tensor([-100.0, -10.0, -1.0, -0.01, 0.0, 0.01, 1.0, 10.0, 100.0])
        s = symlog(x)
        rec = symexp(s)
        torch.testing.assert_close(rec, x, atol=1e-5, rtol=1e-5)

    def test_layernorm_gru_cell(self):
        cell = LayerNormGRUCell(input_dim=64, hidden_dim=128)
        x = torch.randn(4, 64)
        h = torch.randn(4, 128)
        next_h = cell(x, h)
        self.assertEqual(next_h.shape, (4, 128))
        self.assertTrue(torch.isfinite(next_h).all())

    def test_conv_encoder_decoder_reconstruction(self):
        enc = ConvEncoder(in_channels=4, embed_dim=256)
        dec = ConvDecoder(in_features=256, out_channels=4)

        obs = torch.rand(2, 4, 84, 84)
        embed = enc(obs)
        self.assertEqual(embed.shape, (2, 256))

        rec = dec(embed)
        self.assertEqual(rec.shape, (2, 4, 84, 84))
        self.assertTrue(torch.isfinite(rec).all())

    def test_rssm_unimix_and_straight_through_gradients(self):
        rssm = CategoricalRSSM(
            action_dim=3,
            embed_dim=256,
            hidden_dim=128,
            num_categoricals=8,
            num_classes=8,
            unimix=0.01,
        )
        h = torch.zeros(2, 128)
        z = torch.zeros(2, 64)
        a = torch.zeros(2, 3)
        embed = torch.randn(2, 256)

        # Prior step
        next_h, next_z, pr_logits, pr_probs = rssm.step_prior(h, z, a)
        self.assertEqual(next_h.shape, (2, 128))
        self.assertEqual(next_z.shape, (2, 64))
        self.assertTrue((pr_probs >= 0.01 / 8).all())

        # Posterior step
        post_h, post_z, po_logits, po_probs = rssm.step_post(h, z, a, embed)
        self.assertEqual(post_h.shape, (2, 128))
        self.assertEqual(post_z.shape, (2, 64))

        # Straight-through gradient flow check
        loss = post_z.sum()
        loss.backward()
        self.assertIsNotNone(rssm.post_net[0].weight.grad)
        self.assertTrue(torch.isfinite(rssm.post_net[0].weight.grad).all())
        self.assertGreater(rssm.post_net[0].weight.grad.abs().sum().item(), 0)

    def test_rssm_is_first_reset(self):
        rssm = CategoricalRSSM(
            action_dim=3,
            embed_dim=256,
            hidden_dim=128,
            num_categoricals=8,
            num_classes=8,
        )
        h = torch.randn(2, 128)
        z = torch.randn(2, 64)
        first_mask = torch.tensor([[1.0], [0.0]])

        h = h * (1.0 - first_mask)
        z = z * (1.0 - first_mask)

        self.assertTrue((h[0] == 0.0).all())
        self.assertTrue((z[0] == 0.0).all())
        self.assertFalse((h[1] == 0.0).all())
        self.assertFalse((z[1] == 0.0).all())

    def test_rssm_sequence_applies_each_action_to_the_successor_observation(self):
        rssm = CategoricalRSSM(
            action_dim=1,
            embed_dim=1,
            hidden_dim=1,
            num_categoricals=1,
            num_classes=2,
        )
        prior_actions = []

        def fake_prior(h, z, action):
            prior_actions.append(action.detach().clone())
            logits = torch.zeros(h.shape[0], 1, 2)
            return h, z, logits, torch.full_like(logits, 0.5)

        def fake_post(h, z, action, embed):
            logits = torch.zeros(h.shape[0], 1, 2)
            return h + embed[:, :1], z, logits, torch.full_like(logits, 0.5)

        rssm.step_prior = fake_prior
        rssm.step_post = fake_post
        states, _, _ = rssm.observe_sequence(
            embeds=torch.tensor([[[10.0], [20.0], [30.0]]]),
            actions=torch.tensor([[[0.1], [0.2]]]),
            is_first=torch.tensor([[True, False]]),
        )

        torch.testing.assert_close(states[0, :, 0], torch.tensor([10.0, 30.0, 60.0]))
        torch.testing.assert_close(
            torch.stack(prior_actions, dim=1),
            torch.tensor([[[0.0], [0.1], [0.2]]]),
        )

    def test_rssm_burnin_context_does_not_receive_learning_gradients(self):
        rssm = CategoricalRSSM(
            action_dim=3,
            embed_dim=8,
            hidden_dim=8,
            num_categoricals=2,
            num_classes=3,
        )
        embeds = torch.randn(1, 5, 8, requires_grad=True)
        actions = torch.randn(1, 4, 3)
        is_first = torch.tensor([[True, False, False, False]])
        states, _, _ = rssm.observe_sequence(embeds, actions, is_first, burnin=2)
        states[:, 2:, :rssm.hidden_dim].square().sum().backward()

        self.assertTrue(torch.equal(embeds.grad[:, :2], torch.zeros_like(embeds.grad[:, :2])))
        self.assertGreater(embeds.grad[:, 2:].abs().sum().item(), 0.0)

    def test_actor_bounded_normal_score_gradient_and_entropy(self):
        actor = Actor(in_features=2, action_dim=1)
        with torch.no_grad():
            actor.mean_head.weight.zero_()
            actor.mean_head.bias.zero_()
            actor.std_head.weight.zero_()
            actor.std_head.bias.zero_()
        state = torch.zeros(1, 2)
        dist = actor.distribution(state)
        self.assertTrue(torch.allclose(dist.mean, torch.zeros_like(dist.mean)))
        self.assertTrue(((dist.stddev >= 0.1) & (dist.stddev <= 1.0)).all())

        sampled = (dist.mean.detach() + 0.5 * dist.stddev.detach()).requires_grad_(True)
        log_prob = actor.log_prob(state, sampled)
        positive_grad = torch.autograd.grad(
            -log_prob.mean(), actor.mean_head.bias, retain_graph=True
        )[0]
        negative_grad = torch.autograd.grad(log_prob.mean(), actor.mean_head.bias)[0]
        self.assertGreater(positive_grad.abs().sum().item(), 0.0)
        torch.testing.assert_close(positive_grad, -negative_grad)
        self.assertIsNone(sampled.grad)

        entropy = actor.entropy(state).sum()
        entropy_grad = torch.autograd.grad(entropy, actor.std_head.bias)[0]
        self.assertGreater(entropy_grad.abs().sum().item(), 0.0)

        torch.manual_seed(11)
        actions, _ = actor(torch.zeros(512, 2))
        self.assertTrue((actions.abs() > 1.0).any())
        deterministic, _ = actor(torch.zeros(2, 2), deterministic=True)
        self.assertTrue((deterministic.abs() <= 1.0).all())

    def test_imagined_score_does_not_backpropagate_through_prior_actions(self):
        torch.manual_seed(260926)
        rssm = CategoricalRSSM(
            action_dim=3, embed_dim=8, hidden_dim=8,
            num_categoricals=2, num_classes=3,
        )
        actor = Actor(in_features=rssm.state_dim, action_dim=3)
        h, z = torch.zeros(4, 8), torch.zeros(4, 6)
        first_action, _ = actor(torch.cat([h, z], -1))
        next_h, next_z, _, _ = rssm.step_prior(h, z, first_action.clamp(-1, 1))
        next_state = torch.cat([next_h, next_z], -1)
        next_action, _ = actor(next_state)

        def score(state):
            return -(
                actor.log_prob(state, next_action) + 3e-4 * actor.entropy(state)
            ).mean()

        attached = score(next_state)
        stopped = score(next_state.detach())
        torch.testing.assert_close(stopped, attached)
        unwanted = torch.autograd.grad(attached, first_action, retain_graph=True)[0]
        self.assertGreater(unwanted.norm().item(), 0.0)
        self.assertIsNone(torch.autograd.grad(
            stopped, first_action, allow_unused=True, retain_graph=True
        )[0])
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in
                            torch.autograd.grad(stopped, tuple(actor.parameters()))))

    def test_twohot_reward_and_value_heads_encode_decode_raw_returns(self):
        reward = RewardHead(in_features=4, bins=255)
        critic = Critic(in_features=4, bins=255)
        state = torch.randn(6, 4)
        targets = torch.tensor([-1000.0, -2.5, -0.01, 0.0, 3.0, 1000.0])

        reward_logits = reward(state)
        value_logits = critic(state)
        self.assertEqual(reward_logits.shape, (6, 255))
        self.assertEqual(value_logits.shape, (6, 255))
        torch.testing.assert_close(
            reward.bin_centers,
            -reward.bin_centers.flip(0),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(reward.pred(state), torch.zeros(6), atol=1e-5, rtol=0.0)
        reward_loss = reward.loss(reward_logits, targets).mean()
        value_loss = critic.loss(value_logits, targets).mean()
        self.assertTrue(torch.isfinite(reward_loss))
        self.assertTrue(torch.isfinite(value_loss))
        (reward_loss + value_loss).backward()
        self.assertTrue(torch.isfinite(reward.net[-1].weight.grad).all())
        self.assertTrue(torch.isfinite(critic.net[-1].weight.grad).all())
        self.assertEqual(reward.pred(state).shape, targets.shape)
        self.assertEqual(critic.pred(state).shape, targets.shape)

    def test_kl_free_nats_are_applied_to_both_gradient_paths(self):
        prior_logits = torch.zeros(1, 2, 3, requires_grad=True)
        posterior_logits = torch.zeros(1, 2, 3, requires_grad=True)
        prior_probs = torch.softmax(prior_logits, dim=-1)
        posterior_probs = torch.softmax(posterior_logits, dim=-1)
        loss, dyn, rep = balanced_kl_loss(
            prior_probs,
            posterior_probs,
            dyn_weight=0.8,
            rep_weight=0.2,
            free_nats=1.0,
        )
        torch.testing.assert_close(loss, torch.tensor(1.0))
        torch.testing.assert_close(dyn, torch.ones_like(dyn))
        torch.testing.assert_close(rep, torch.ones_like(rep))
        prior_grad, posterior_grad = torch.autograd.grad(
            loss, (prior_logits, posterior_logits)
        )
        self.assertEqual(prior_grad.abs().sum().item(), 0.0)
        self.assertEqual(posterior_grad.abs().sum().item(), 0.0)

        divergent_logits = torch.tensor([[[5.0, 0.0, -5.0], [-5.0, 0.0, 5.0]]], requires_grad=True)
        divergent_probs = torch.softmax(divergent_logits, dim=-1)
        divergent_loss, _, _ = balanced_kl_loss(
            prior_probs.detach(),
            divergent_probs,
            dyn_weight=0.8,
            rep_weight=0.2,
            free_nats=1.0,
        )
        self.assertGreater(divergent_loss.item(), 1.0)
        divergent_grad = torch.autograd.grad(divergent_loss, divergent_logits)[0]
        self.assertGreater(divergent_grad.abs().sum().item(), 0.0)

    def test_overshooting_kl_trains_multistep_prior_against_future_posterior(self):
        cfg = DreamerV3Config(
            device="cpu", embed_dim=8, hidden_dim=8,
            num_categoricals=2, num_classes=3,
            overshoot_horizon=4, kl_free_nats=0.1,
        )
        agent = DreamerV3Agent(cfg, seed=91)
        states = torch.randn(2, 6, agent.rssm.state_dim, requires_grad=True)
        actions = torch.randn(2, 5, 3)
        posterior_logits = torch.full((2, 6, 2, 3), -4.0)
        posterior_logits[..., 0] = 4.0

        loss = agent._overshooting_kl_loss(states, actions, posterior_logits)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.1)
        grad = torch.autograd.grad(loss, agent.rssm.prior_net[-1].weight)[0]
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_terminal_imagination_weight_stays_zero_after_terminal(self):
        continues = torch.tensor([[0.5], [0.0], [0.7], [0.9]])
        weights = continuation_weights(continues, gamma=0.99)
        torch.testing.assert_close(
            weights[:, 0], torch.tensor([1.0, 0.495, 0.0, 0.0])
        )

    def test_terminal_positive_weight_increases_positive_continue_gradient(self):
        logits = torch.zeros(2, requires_grad=True)
        targets = torch.tensor([1.0, 0.0])
        loss = continue_prediction_loss(logits, targets, positive_weight=10.0)
        grad = torch.autograd.grad(loss, logits)[0]
        self.assertLess(grad[0].item(), 0.0)
        self.assertGreater(grad[1].item(), 0.0)
        self.assertAlmostEqual(abs(grad[0].item() / grad[1].item()), 10.0, places=5)

    def test_lambda_return_stops_bootstrapping_after_terminal(self):
        returns = lambda_returns(
            rewards=torch.tensor([[1.0], [10.0]]),
            continues=torch.tensor([[1.0], [0.0]]),
            values=torch.tensor([[100.0], [200.0], [300.0]]),
            gamma=0.99,
            lambda_=0.95,
        )
        torch.testing.assert_close(returns[:, 0], torch.tensor([20.305, 10.0]))

    def test_sequence_replay_buffer(self):
        replay = Uint8SequenceReplay(capacity=100)
        for i in range(50):
            obs = np.full((4, 84, 84), i, dtype=np.uint8)
            act = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            trans = Transition(
                observation=obs,
                action=act,
                reward=1.0,
                next_observation=obs,
                terminated=(i == 49),
                truncated=False,
                step=i,
            )
            replay.add(trans)

        self.assertEqual(replay.size, 50)
        batch = replay.sample_sequence(batch_size=4, seq_len=10)
        self.assertEqual(batch["observations"].shape, (4, 11, 4, 84, 84))
        self.assertEqual(batch["actions"].shape, (4, 10, 3))
        self.assertEqual(batch["rewards"].shape, (4, 10))
        self.assertEqual(batch["is_first"].shape, (4, 10))
        self.assertGreaterEqual(batch["is_first"][:, 0].sum().item(), 0)

    def test_sequence_replay_keeps_terminal_successor_separate_from_reset(self):
        replay = Uint8SequenceReplay(capacity=8)
        for episode_id, step, pixel, next_pixel, reward, terminal in (
            (1, 0, 10, 11, 101.0, False),
            (1, 1, 11, 12, 102.0, True),
            (2, 0, 20, 21, 201.0, False),
        ):
            replay.add(Transition(
                observation=np.full((4, 84, 84), pixel, dtype=np.uint8),
                action=np.full(3, pixel / 100.0, dtype=np.float32),
                reward=reward,
                next_observation=np.full((4, 84, 84), next_pixel, dtype=np.uint8),
                terminated=terminal,
                truncated=False,
                terminal=terminal,
                episode_id=episode_id,
                step=step,
            ))

        batch = replay.sample_sequence(batch_size=2, seq_len=2)
        expected = np.asarray([10, 11, 12], dtype=np.float32) / 255.0
        np.testing.assert_allclose(
            batch["observations"][:, :, 0, 0, 0].numpy(),
            np.broadcast_to(expected, (2, 3)),
        )
        np.testing.assert_allclose(batch["rewards"].numpy(), [[101.0, 102.0]] * 2)
        np.testing.assert_allclose(batch["actions"][:, :, 0].numpy(), [[0.1, 0.11]] * 2)
        np.testing.assert_array_equal(batch["is_first"].numpy(), [[True, False]] * 2)
        np.testing.assert_array_equal(batch["is_last"].numpy(), [[False, True]] * 2)
        np.testing.assert_array_equal(batch["is_terminal"].numpy(), [[False, True]] * 2)

    def test_sequence_replay_uses_truncation_successor_not_next_episode(self):
        replay = Uint8SequenceReplay(capacity=4)
        replay.add(Transition(
            observation=np.full((4, 84, 84), 7, dtype=np.uint8),
            action=np.zeros(3, dtype=np.float32),
            reward=1.0,
            next_observation=np.full((4, 84, 84), 8, dtype=np.uint8),
            terminated=False,
            truncated=True,
            terminal=False,
            episode_id=1,
            step=0,
        ))
        replay.add(Transition(
            observation=np.full((4, 84, 84), 99, dtype=np.uint8),
            action=np.zeros(3, dtype=np.float32),
            reward=2.0,
            next_observation=np.full((4, 84, 84), 100, dtype=np.uint8),
            terminated=False,
            truncated=False,
            episode_id=2,
            step=0,
        ))

        batch = replay.sample_sequence(batch_size=1, seq_len=1)
        np.testing.assert_allclose(
            batch["observations"][0, :, 0, 0, 0].numpy(),
            np.asarray([7, 8], dtype=np.float32) / 255.0,
        )
        self.assertFalse(batch["is_terminal"][0, 0].item())
        self.assertTrue(batch["is_last"][0, 0].item())

    def test_sequence_replay_serializes_boundary_successors(self):
        replay = Uint8SequenceReplay(capacity=4)
        replay.add(Transition(
            observation=np.full((4, 84, 84), 1, dtype=np.uint8),
            action=np.zeros(3, dtype=np.float32),
            reward=1.0,
            next_observation=np.full((4, 84, 84), 2, dtype=np.uint8),
            terminated=True,
            truncated=False,
            terminal=True,
            episode_id=1,
            step=0,
        ))
        restored = Uint8SequenceReplay(capacity=4)
        restored.load_state_dict(replay.state_dict())
        batch = restored.sample_sequence(batch_size=1, seq_len=1, terminal_fraction=1.0)
        np.testing.assert_allclose(
            batch["observations"][0, :, 0, 0, 0].numpy(),
            np.asarray([1, 2], dtype=np.float32) / 255.0,
        )
        self.assertTrue(batch["terminal_anchored"][0].item())

    def test_sequence_replay_drops_overwritten_boundary_successors(self):
        replay = Uint8SequenceReplay(capacity=3)
        replay.add(Transition(
            observation=np.full((4, 84, 84), 1, dtype=np.uint8),
            action=np.zeros(3, dtype=np.float32),
            reward=1.0,
            next_observation=np.full((4, 84, 84), 90, dtype=np.uint8),
            terminated=True,
            truncated=False,
            terminal=True,
            episode_id=1,
            step=0,
        ))
        for step, pixel in enumerate((10, 11, 12)):
            replay.add(Transition(
                observation=np.full((4, 84, 84), pixel, dtype=np.uint8),
                action=np.zeros(3, dtype=np.float32),
                reward=float(step),
                next_observation=np.full((4, 84, 84), pixel + 1, dtype=np.uint8),
                terminated=False,
                truncated=False,
                episode_id=2,
                step=step,
            ))

        self.assertNotIn(0, replay._boundary_observations)
        self.assertNotIn(0, replay._terminal_sequence_ids)
        batch = replay.sample_sequence(batch_size=1, seq_len=2)
        np.testing.assert_allclose(
            batch["observations"][0, :, 0, 0, 0].numpy(),
            np.asarray([10, 11, 12], dtype=np.float32) / 255.0,
        )

    def test_sequence_replay_returns_same_episode_burnin_prefix(self):
        replay = Uint8SequenceReplay(capacity=12)
        for step in range(7):
            replay.add(Transition(
                observation=np.full((4, 84, 84), step + 1, dtype=np.uint8),
                action=np.full(3, step / 10.0, dtype=np.float32),
                reward=float(step + 100),
                next_observation=np.full((4, 84, 84), step + 2, dtype=np.uint8),
                terminated=step == 6,
                truncated=False,
                terminal=step == 6,
                episode_id=1,
                step=step,
            ))

        batch = replay.sample_sequence(batch_size=4, seq_len=2, burnin=2)
        self.assertEqual(batch["observations"].shape, (4, 5, 4, 84, 84))
        self.assertEqual(batch["actions"].shape, (4, 4, 3))
        self.assertEqual(batch["rewards"].shape, (4, 4))
        pixels = batch["observations"][:, :, 0, 0, 0].numpy() * 255.0
        np.testing.assert_allclose(np.diff(pixels, axis=1), np.ones((4, 4)), atol=1e-4)
        np.testing.assert_array_equal(batch["is_first"][:, 1:].numpy(), False)

    def test_reset_origin_windows_supervise_short_episode_start_and_end(self):
        for terminal in (True, False):
            with self.subTest(terminal=terminal):
                replay = Uint8SequenceReplay(capacity=39)
                for step in range(39):
                    replay.add(Transition(
                        observation=np.full((4, 84, 84), step, dtype=np.uint8),
                        action=np.full(3, step / 100.0, dtype=np.float32),
                        reward=float(step),
                        next_observation=np.full((4, 84, 84), step + 1, dtype=np.uint8),
                        terminated=terminal and step == 38,
                        truncated=not terminal and step == 38,
                        terminal=terminal and step == 38,
                        episode_id=1, step=step,
                    ))
                with self.assertRaisesRegex(ValueError, "need at least 40"):
                    replay.sample_sequence(1, 32, burnin=8)

                restored = Uint8SequenceReplay(capacity=39)
                restored.load_state_dict(replay.state_dict())
                seen_burnins = set()
                for _ in range(96):
                    batch = restored.sample_sequence(
                        1, 32, burnin=8, reset_start_fraction=1.0,
                        terminal_fraction=1.0,
                    )
                    burnin = batch["effective_burnin"]
                    seen_burnins.add(burnin)
                    self.assertTrue(batch["reset_start_anchored"])
                    self.assertTrue(batch["is_first"][0, 0].item())
                    self.assertFalse(batch["is_first"][0, 1:].any().item())
                    np.testing.assert_allclose(
                        batch["observations"][0, :, 0, 0, 0].numpy() * 255,
                        np.arange(33 + burnin), atol=1e-4,
                    )
                    np.testing.assert_allclose(
                        batch["actions"][0, :, 0].numpy(),
                        np.arange(32 + burnin) / 100.0, atol=1e-6,
                    )
                    if burnin == 7:
                        self.assertEqual(batch["is_terminal"].sum().item(), int(terminal))
                        self.assertTrue(batch["is_last"][0, -1].item())
                        self.assertEqual(batch["terminal_anchored"][0].item(), terminal)
                self.assertEqual(seen_burnins, set(range(8)))

    def test_reset_origin_window_rejects_ring_when_reset_is_overwritten(self):
        replay = Uint8SequenceReplay(capacity=35)
        for step in range(39):
            replay.add(Transition(
                observation=np.full((4, 84, 84), step, dtype=np.uint8),
                action=np.zeros(3, dtype=np.float32), reward=float(step),
                next_observation=np.full((4, 84, 84), step + 1, dtype=np.uint8),
                terminated=step == 38, truncated=False, terminal=step == 38,
                episode_id=1, step=step,
            ))
        restored = Uint8SequenceReplay(capacity=35)
        restored.load_state_dict(replay.state_dict())
        with self.assertRaisesRegex(NoValidSequenceError, "reset-origin"):
            restored.sample_sequence(1, 32, burnin=8, reset_start_fraction=1.0)

    def test_exact_32_decision_episode_uses_zero_burnin_reset_window(self):
        replay = Uint8SequenceReplay(capacity=32)
        for step in range(32):
            replay.add(Transition(
                observation=np.full((4, 84, 84), step, dtype=np.uint8),
                action=np.zeros(3, dtype=np.float32), reward=0.0,
                next_observation=np.full((4, 84, 84), step + 1, dtype=np.uint8),
                terminated=step == 31, truncated=False, terminal=step == 31,
                episode_id=1, step=step,
            ))
        batch = replay.sample_sequence(1, 32, burnin=8, reset_start_fraction=1.0)
        self.assertEqual(batch["effective_burnin"], 0)
        self.assertEqual(batch["effective_seq_len"], 32)
        self.assertFalse(batch["short_episode_anchored"])
        self.assertTrue(batch["is_terminal"][0, -1].item())
        np.testing.assert_allclose(
            batch["observations"][0, -1, 0, 0, 0].item() * 255, 32, atol=1e-4
        )

    def test_reset_origin_mode_falls_back_when_no_full_burnin_window_exists(self):
        replay = Uint8SequenceReplay(capacity=80)
        for episode_id in (1, 2):
            for step in range(39):
                obs = np.full((4, 84, 84), step, dtype=np.uint8)
                replay.add(Transition(
                    observation=obs, action=np.zeros(3, dtype=np.float32),
                    reward=0.0, next_observation=np.full_like(obs, step + 1),
                    terminated=step == 38, truncated=False, terminal=step == 38,
                    episode_id=episode_id, step=step,
                ))
        with self.assertRaisesRegex(NoValidSequenceError, "no valid transition sequence"):
            replay.sample_sequence(1, 32, burnin=8)
        with patch("dreamer_v3.np.random.random", return_value=1.0):
            batch = replay.sample_sequence(
                1, 32, burnin=8, reset_start_fraction=0.25,
            )
        self.assertTrue(batch["reset_start_anchored"])
        self.assertTrue(batch["is_first"][0, 0].item())
        self.assertLess(batch["effective_burnin"], 8)

    def test_reset_origin_update_uses_effective_burnin_for_short_episode(self):
        cfg = DreamerV3Config(
            device="cpu", embed_dim=64, hidden_dim=32,
            num_categoricals=4, num_classes=4, batch_size=1,
            seq_len=32, burnin_steps=8, reset_start_fraction=1.0,
            replay_capacity=39,
        )
        agent = DreamerV3Agent(cfg, seed=79)
        for step in range(39):
            obs = np.full((4, 84, 84), step, dtype=np.uint8)
            agent.observe(Transition(
                observation=obs, action=np.zeros(3, dtype=np.float32),
                reward=float(step), next_observation=np.full_like(obs, step + 1),
                terminated=step == 38, truncated=False, terminal=step == 38,
                episode_id=1, step=step,
            ))
        metrics = agent.update(model_only=True)
        self.assertTrue(np.isfinite(metrics["loss_wm"]))
        self.assertGreaterEqual(metrics["effective_burnin"], 0)
        self.assertLessEqual(metrics["effective_burnin"], 7)
        self.assertEqual(metrics["reset_start_anchored"], 1.0)
        self.assertEqual(metrics["terminal_label_count"],
                         float(metrics["effective_burnin"] == 7))
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "short-reset.pt"
            agent.save_checkpoint(checkpoint)
            diagnostic = diagnose_world_model(checkpoint)
        self.assertEqual(diagnostic["world_model"]["reset_start_anchored"], True)
        self.assertLess(diagnostic["world_model"]["effective_burnin"], 8)

    def test_exact_length_short_episode_windows_have_no_padding_or_reset_leak(self):
        for length in (1, 7, 31):
            for terminal in (False, True):
                with self.subTest(length=length, terminal=terminal):
                    replay = Uint8SequenceReplay(capacity=length + 2)
                    for step in range(length):
                        replay.add(Transition(
                            observation=np.full((4, 84, 84), step + 1, dtype=np.uint8),
                            action=np.full(3, step / 100.0, dtype=np.float32),
                            reward=float(step),
                            next_observation=np.full(
                                (4, 84, 84), 100 + step, dtype=np.uint8
                            ) if step == length - 1 else np.full(
                                (4, 84, 84), step + 2, dtype=np.uint8
                            ),
                            terminated=terminal and step == length - 1,
                            truncated=not terminal and step == length - 1,
                            terminal=terminal and step == length - 1,
                            episode_id=1, step=step,
                        ))
                    replay.add(Transition(
                        observation=np.full((4, 84, 84), 240, dtype=np.uint8),
                        action=np.zeros(3, dtype=np.float32), reward=0.0,
                        next_observation=np.full((4, 84, 84), 241, dtype=np.uint8),
                        terminated=False, truncated=False, episode_id=2, step=0,
                    ))
                    restored = Uint8SequenceReplay(capacity=length + 2)
                    restored.load_state_dict(replay.state_dict())
                    batch = restored.sample_sequence(
                        2, 32, burnin=8, short_episode_fraction=1.0,
                        terminal_fraction=0.5,
                    )
                    self.assertEqual(batch["effective_seq_len"], length)
                    self.assertEqual(batch["effective_burnin"], 0)
                    self.assertTrue(batch["short_episode_anchored"])
                    self.assertEqual(batch["actions"].shape, (2, length, 3))
                    self.assertEqual(batch["observations"].shape, (2, length + 1, 4, 84, 84))
                    self.assertTrue(batch["is_first"][:, 0].all().item())
                    self.assertTrue(batch["is_last"][:, -1].all().item())
                    self.assertEqual(batch["is_terminal"].sum().item(), 2 * int(terminal))
                    self.assertEqual(torch.unique(batch["sequence_ids"][:, 0]).numel(), 1)
                    np.testing.assert_allclose(
                        batch["observations"][:, -1, 0, 0, 0].numpy() * 255,
                        np.full(2, 99 + length), atol=1e-4,
                    )

    def test_short_episode_rejects_overwritten_reset_and_missing_result(self):
        partial = Uint8SequenceReplay(capacity=3)
        partial.add(Transition(
            observation=np.zeros((4, 84, 84), dtype=np.uint8),
            action=np.zeros(3, dtype=np.float32), reward=0.0,
            next_observation=np.ones((4, 84, 84), dtype=np.uint8),
            terminated=False, truncated=False, episode_id=0, step=0,
        ))
        with self.assertRaisesRegex(NoValidSequenceError, "short episode"):
            partial.sample_sequence(1, 32, short_episode_fraction=1.0)

        replay = Uint8SequenceReplay(capacity=3)
        for step in range(4):
            replay.add(Transition(
                observation=np.full((4, 84, 84), step, dtype=np.uint8),
                action=np.zeros(3, dtype=np.float32), reward=0.0,
                next_observation=np.full((4, 84, 84), step + 1, dtype=np.uint8),
                terminated=step == 3, truncated=False, terminal=step == 3,
                episode_id=1, step=step,
            ))
        with self.assertRaisesRegex(NoValidSequenceError, "short episode"):
            replay.sample_sequence(1, 32, short_episode_fraction=1.0)

        single = Uint8SequenceReplay(capacity=3)
        single.add(Transition(
            observation=np.zeros((4, 84, 84), dtype=np.uint8),
            action=np.zeros(3, dtype=np.float32), reward=0.0,
            next_observation=np.ones((4, 84, 84), dtype=np.uint8),
            terminated=True, truncated=False, episode_id=2, step=0,
        ))
        state = single.state_dict()
        del state["boundary_observations"]
        restored = Uint8SequenceReplay(capacity=3)
        restored.load_state_dict(state)
        with self.assertRaisesRegex(NoValidSequenceError, "short episode"):
            restored.sample_sequence(1, 32, short_episode_fraction=1.0)

    def test_short_episode_mode_falls_back_without_full_windows(self):
        replay = Uint8SequenceReplay(capacity=64)
        for episode_id in (1, 2):
            for step in range(31):
                replay.add(Transition(
                    observation=np.full((4, 84, 84), step, dtype=np.uint8),
                    action=np.zeros(3, dtype=np.float32), reward=0.0,
                    next_observation=np.full((4, 84, 84), step + 1, dtype=np.uint8),
                    terminated=step == 30, truncated=False, terminal=step == 30,
                    episode_id=episode_id, step=step,
                ))
        with self.assertRaisesRegex(NoValidSequenceError, "no valid transition"):
            replay.sample_sequence(1, 32, burnin=8)
        with patch("dreamer_v3.np.random.random", return_value=1.0):
            batch = replay.sample_sequence(
                2, 32, burnin=8, short_episode_fraction=0.25,
            )
        self.assertTrue(batch["short_episode_anchored"])
        self.assertEqual(batch["effective_seq_len"], 31)
        self.assertTrue(batch["is_last"][:, -1].all().item())

    def test_fractional_short_mode_waits_for_completed_episode_without_crashing(self):
        cfg = DreamerV3Config(
            device="cpu", embed_dim=64, hidden_dim=32,
            num_categoricals=4, num_classes=4, batch_size=1,
            seq_len=32, burnin_steps=8, short_episode_fraction=0.25,
            warmup_steps=0, replay_capacity=40,
        )
        agent = DreamerV3Agent(cfg, seed=81)
        for step in range(5):
            agent.replay.add(Transition(
                observation=np.full((4, 84, 84), step, dtype=np.uint8),
                action=np.zeros(3, dtype=np.float32), reward=0.0,
                next_observation=np.full((4, 84, 84), step + 1, dtype=np.uint8),
                terminated=False, truncated=False, episode_id=1, step=step,
            ))
        with self.assertRaises(NoValidSequenceError):
            agent.replay.sample_sequence(1, 32, burnin=8, short_episode_fraction=0.25)
        self.assertEqual(agent.update(model_only=True), {})

    def test_fractional_short_mode_uses_real_complete_episode_before_full_window(self):
        replay = Uint8SequenceReplay(capacity=40)
        for step in range(34):
            replay.add(Transition(
                observation=np.full((4, 84, 84), step, dtype=np.uint8),
                action=np.zeros(3, dtype=np.float32), reward=0.0,
                next_observation=np.full((4, 84, 84), step + 1, dtype=np.uint8),
                terminated=step == 30, truncated=False, terminal=step == 30,
                episode_id=1 if step <= 30 else 2, step=step if step <= 30 else step - 31,
            ))
        with patch("dreamer_v3.np.random.random", return_value=1.0):
            batch = replay.sample_sequence(1, 32, burnin=8, short_episode_fraction=0.25)
        self.assertTrue(batch["short_episode_anchored"])
        self.assertEqual((batch["effective_burnin"], batch["effective_seq_len"]), (0, 31))
        self.assertEqual(batch["sequence_ids"][0, -1].item(), 30)

    def test_one_step_short_update_bootstraps_only_on_truncation(self):
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                cfg = DreamerV3Config(
                    device="cpu", embed_dim=64, hidden_dim=32,
                    num_categoricals=4, num_classes=4, batch_size=2,
                    seq_len=32, burnin_steps=8, short_episode_fraction=1.0,
                    imagination_horizon=2, warmup_steps=0, replay_capacity=4,
                )
                agent = DreamerV3Agent(cfg, seed=79)
                agent.observe(Transition(
                    observation=np.full((4, 84, 84), 10, dtype=np.uint8),
                    action=np.zeros(3, dtype=np.float32), reward=3.0,
                    next_observation=np.full((4, 84, 84), 20, dtype=np.uint8),
                    terminated=terminal, truncated=not terminal, terminal=terminal,
                    episode_id=1, step=0,
                ))
                with patch.object(agent.critic, "pred", side_effect=lambda states: torch.full(
                    states.shape[:-1], 2.0, device=states.device
                )), patch("dreamer_v3.lambda_returns", wraps=lambda_returns) as returns:
                    metrics = agent.update()
                self.assertTrue(np.isfinite(metrics["loss_wm"]))
                self.assertTrue(np.isfinite(metrics["loss_actor"]))
                self.assertEqual(metrics["effective_seq_len"], 1.0)
                self.assertEqual(metrics["short_episode_anchored"], 1.0)
                self.assertEqual(metrics["unique_short_episodes"], 1.0)
                self.assertEqual(metrics["terminal_label_count"], 2.0 * terminal)
                self.assertEqual(metrics["unique_terminal_events"], float(terminal))
                replay_call = returns.call_args_list[-1]
                torch.testing.assert_close(replay_call.args[1], torch.full(
                    (1, 2), 0.0 if terminal else 1.0
                ))
                torch.testing.assert_close(
                    lambda_returns(*replay_call.args),
                    torch.full((1, 2), 3.0 if terminal else 3.0 + cfg.gamma * 2.0),
                )
                with tempfile.TemporaryDirectory() as tmpdir:
                    checkpoint = Path(tmpdir) / "short.pt"
                    agent.save_checkpoint(checkpoint)
                    report = diagnose_world_model(checkpoint)
                self.assertEqual(report["world_model"]["effective_seq_len"], 1)
                self.assertTrue(report["world_model"]["short_episode_anchored"])

    def test_replay_rejects_interior_reset_with_one_learning_step(self):
        replay = Uint8SequenceReplay(capacity=3)
        for episode_id in (1, 2):
            replay.add(Transition(
                observation=np.full((4, 84, 84), episode_id, dtype=np.uint8),
                action=np.zeros(3, dtype=np.float32), reward=0.0,
                next_observation=np.full((4, 84, 84), episode_id + 1, dtype=np.uint8),
                terminated=episode_id == 2, truncated=False,
                episode_id=episode_id, step=0,
            ))
        with self.assertRaisesRegex(NoValidSequenceError, "no valid transition"):
            replay.sample_sequence(1, 1, burnin=1)

    def test_terminal_window_fraction_anchors_sequences_on_terminal_actions(self):
        replay = Uint8SequenceReplay(capacity=16)
        for step in range(10):
            obs = np.full((4, 84, 84), step, dtype=np.uint8)
            replay.add(Transition(
                observation=obs,
                action=np.zeros(3, dtype=np.float32),
                reward=float(step),
                next_observation=np.full_like(obs, step + 1),
                terminated=step == 9,
                truncated=False,
                terminal=step == 9,
                episode_id=1,
                step=step,
            ))

        batch = replay.sample_sequence(
            batch_size=8, seq_len=3, burnin=1, terminal_fraction=0.5
        )
        anchored = batch["terminal_anchored"]
        self.assertEqual(anchored.sum().item(), 4)
        self.assertTrue(batch["is_terminal"][anchored, -1].all())
        self.assertFalse(batch["is_terminal"][anchored, :-1].any())
        self.assertEqual(batch["is_terminal"].sum().item(), 4)
        self.assertEqual(batch["is_terminal"].numel(), 32)
        self.assertEqual(torch.unique(
            batch["sequence_ids"][batch["is_terminal"]]
        ).numel(), 1)

        uniform = replay.sample_sequence(batch_size=8, seq_len=3, burnin=1)
        self.assertFalse(uniform["terminal_anchored"].any())

    def test_legacy_replay_without_boundary_frame_fails_closed(self):
        replay = Uint8SequenceReplay(capacity=4)
        replay.add(Transition(
            observation=np.full((4, 84, 84), 1, dtype=np.uint8),
            action=np.zeros(3, dtype=np.float32),
            reward=1.0,
            next_observation=np.full((4, 84, 84), 2, dtype=np.uint8),
            terminated=True,
            truncated=False,
            terminal=True,
            episode_id=1,
            step=0,
        ))
        legacy_state = replay.state_dict()
        del legacy_state["boundary_observations"]
        restored = Uint8SequenceReplay(capacity=4)
        restored.load_state_dict(legacy_state)

        with self.assertRaisesRegex(ValueError, "no valid transition sequence"):
            restored.sample_sequence(batch_size=1, seq_len=1)

    def test_dreamerv3_agent_step_and_update(self):
        cfg = DreamerV3Config(
            device="cpu",
            embed_dim=128,
            hidden_dim=64,
            num_categoricals=8,
            num_classes=8,
            batch_size=2,
            seq_len=8,
            warmup_steps=10,
            replay_capacity=100,
        )
        agent = DreamerV3Agent(cfg, seed=42)

        for step in range(20):
            obs = np.random.rand(4, 84, 84).astype(np.float32)
            proposed = agent.act(obs, deterministic=False)
            applied = agent.action_adapter.to_official(proposed)
            act = agent.action_adapter.to_native(applied)
            trans = Transition(
                observation=obs,
                action=act,
                applied_action=applied,
                reward=0.5,
                next_observation=obs,
                terminated=(step == 19),
                truncated=False,
                step=step,
            )
            agent.observe(trans)

        self.assertEqual(agent.environment_steps, 20)
        with patch.object(agent.actor, "log_prob", wraps=agent.actor.log_prob) as score, \
                patch.object(agent.actor, "entropy", wraps=agent.actor.entropy) as entropy:
            metrics = agent.update()
        self.assertEqual(score.call_count, 1)
        self.assertEqual(entropy.call_count, 1)
        self.assertFalse(score.call_args.args[0].requires_grad)
        self.assertFalse(entropy.call_args.args[0].requires_grad)
        for key in (
            "loss_wm",
            "loss_obs",
            "loss_reward",
            "loss_continue",
            "loss_kl",
            "loss_critic",
            "loss_replay_value",
            "loss_actor",
            "kl_dyn_mean",
            "kl_rep_mean",
            "actor_advantage_mean",
            "actor_entropy_mean",
            "imagination_weight_mean",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]), f"metric {key} was not finite")

    def test_imagined_and_replay_returns_use_current_critic_only(self):
        cfg = DreamerV3Config(
            device="cpu", embed_dim=64, hidden_dim=32,
            num_categoricals=4, num_classes=4, batch_size=1,
            seq_len=2, burnin_steps=0, imagination_horizon=2,
            warmup_steps=0, replay_capacity=8,
        )
        agent = DreamerV3Agent(cfg, seed=75)
        for step in range(4):
            observation = np.full((4, 84, 84), step / 10.0, dtype=np.float32)
            agent.observe(Transition(
                observation=observation,
                action=np.zeros(3, dtype=np.float32),
                reward=float(step),
                next_observation=np.full_like(observation, (step + 1) / 10.0),
                terminated=step == 3, truncated=False, terminal=step == 3,
                step=step,
            ))

        def constant_value(value):
            return lambda states: torch.full(states.shape[:-1], value, device=states.device)

        with patch.object(agent.critic, "pred", side_effect=constant_value(2.0)), \
                patch.object(agent.critic_target, "pred", side_effect=constant_value(-9.0)) as slow, \
                patch.object(agent.critic, "loss", wraps=agent.critic.loss) as value_loss, \
                patch("dreamer_v3.lambda_returns", wraps=lambda_returns) as returns:
            agent.update()

        self.assertEqual(returns.call_count, 2)
        for call in returns.call_args_list:
            torch.testing.assert_close(call.args[2], torch.full_like(call.args[2], 2.0))
        self.assertEqual(slow.call_count, 1)
        torch.testing.assert_close(
            value_loss.call_args_list[1].args[1],
            torch.full_like(value_loss.call_args_list[1].args[1], -9.0),
        )

    def test_replay_only_update_trains_world_model_before_policy_warmup(self):
        cfg = DreamerV3Config(
            device="cpu", embed_dim=64, hidden_dim=32,
            num_categoricals=4, num_classes=4, batch_size=2,
            seq_len=4, burnin_steps=2, warmup_steps=100,
            overshoot_horizon=3, overshoot_kl_weight=0.1,
            replay_capacity=32,
        )
        agent = DreamerV3Agent(cfg, seed=61)
        for step in range(12):
            observation = np.full((4, 84, 84), step / 12.0, dtype=np.float32)
            agent.observe(Transition(
                observation=observation,
                action=np.zeros(3, dtype=np.float32),
                reward=float(step),
                next_observation=np.full_like(observation, (step + 1) / 12.0),
                terminated=step == 11,
                truncated=False,
                terminal=step == 11,
                step=step,
            ))

        self.assertLess(agent.environment_steps, cfg.warmup_steps)
        self.assertEqual(agent.update(), {})
        actor_before = {key: value.clone() for key, value in agent.actor.state_dict().items()}
        critic_before = {key: value.clone() for key, value in agent.critic.state_dict().items()}
        decoder_before = agent.decoder.net[-1].weight.clone()
        metrics = agent.update(model_only=True)

        self.assertIn("loss_wm", metrics)
        self.assertIn("loss_overshoot_kl", metrics)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertEqual(agent.gradient_steps, 1)
        self.assertTrue(all(
            torch.equal(value, actor_before[key])
            for key, value in agent.actor.state_dict().items()
        ))
        self.assertTrue(all(
            torch.equal(value, critic_before[key])
            for key, value in agent.critic.state_dict().items()
        ))
        self.assertFalse(torch.equal(agent.decoder.net[-1].weight, decoder_before))

    def test_external_warmup_transition_advances_online_recurrent_state(self):
        cfg = DreamerV3Config(
            device="cpu", embed_dim=64, hidden_dim=32,
            num_categoricals=4, num_classes=4, replay_capacity=8,
        )
        agent = DreamerV3Agent(cfg, seed=4)
        obs = np.full((4, 84, 84), 0.25, dtype=np.float32)
        action = np.asarray([0.2, -0.4, 0.6], dtype=np.float32)
        agent.observe(Transition(
            observation=obs,
            action=action,
            reward=1.0,
            next_observation=np.full_like(obs, 0.5),
            terminated=False,
            truncated=False,
            step=0,
        ))

        self.assertFalse(agent._online_first)
        self.assertGreater(agent._online_h.abs().sum().item(), 0.0)
        torch.testing.assert_close(agent._online_prev_a, torch.from_numpy(action).unsqueeze(0))

    def test_warmup_to_policy_handoff_matches_continuous_recurrent_prefix(self):
        cfg = DreamerV3Config(
            device="cpu", embed_dim=64, hidden_dim=32,
            num_categoricals=4, num_classes=4, replay_capacity=8,
        )
        trained = DreamerV3Agent(cfg, seed=44)
        reference = DreamerV3Agent(cfg, seed=44)
        observation = np.full((4, 84, 84), 0.2, dtype=np.float32)
        next_observation = np.full((4, 84, 84), 0.4, dtype=np.float32)
        executed_action = np.asarray([0.3, -0.5, 0.7], dtype=np.float32)
        transition = Transition(
            observation=observation,
            action=executed_action,
            reward=1.0,
            next_observation=next_observation,
            terminated=False,
            truncated=False,
            step=0,
        )

        prefix_rng = torch.get_rng_state().clone()
        torch.set_rng_state(prefix_rng)
        trained.observe(transition)
        torch.set_rng_state(prefix_rng)
        reference._advance_online_observation(observation, deterministic=False)
        reference._online_prev_a.copy_(torch.from_numpy(executed_action).unsqueeze(0))

        torch.testing.assert_close(trained._online_h, reference._online_h)
        torch.testing.assert_close(trained._online_z, reference._online_z)
        torch.testing.assert_close(trained._online_prev_a, reference._online_prev_a)
        torch.testing.assert_close(
            trained.act(next_observation, deterministic=True),
            reference.act(next_observation, deterministic=True),
        )

    def test_observe_replaces_proposed_online_action_with_executed_native_action(self):
        cfg = DreamerV3Config(
            device="cpu", embed_dim=64, hidden_dim=32,
            num_categoricals=4, num_classes=4, replay_capacity=8,
        )
        agent = DreamerV3Agent(cfg, seed=5)
        obs = np.full((4, 84, 84), 0.25, dtype=np.float32)
        proposed = agent.act(obs, deterministic=True)
        self.assertEqual(set(agent._last_policy_stats), {"mean", "std", "entropy"})
        self.assertEqual(agent._last_policy_stats["std"].shape, (3,))
        online_h = agent._online_h.clone()
        online_z = agent._online_z.clone()
        executed_official = agent.action_adapter.to_official(
            np.asarray([0.1, 2.0, -2.0], dtype=np.float32)
        )
        executed_native = agent.action_adapter.to_native(executed_official)
        agent.observe(Transition(
            observation=obs,
            action=executed_native,
            reward=1.0,
            next_observation=np.full_like(obs, 0.5),
            terminated=False,
            truncated=False,
            applied_action=executed_official,
            step=0,
        ))

        self.assertFalse(np.array_equal(proposed, executed_native))
        torch.testing.assert_close(agent._online_h, online_h)
        torch.testing.assert_close(agent._online_z, online_z)
        torch.testing.assert_close(
            agent._online_prev_a, torch.from_numpy(executed_native).unsqueeze(0)
        )

    def test_checkpoint_save_and_restore(self):
        cfg = DreamerV3Config(
            device="cpu",
            embed_dim=128,
            hidden_dim=64,
            num_categoricals=8,
            num_classes=8,
            batch_size=2,
            seq_len=8,
            warmup_steps=5,
            replay_capacity=50,
        )
        agent1 = DreamerV3Agent(cfg, seed=10)
        obs = np.ones((4, 84, 84), dtype=np.float32)
        for i in range(10):
            trans = Transition(
                observation=obs,
                action=np.zeros(3, dtype=np.float32),
                reward=1.0,
                next_observation=obs,
                terminated=False,
                truncated=False,
                step=i,
            )
            agent1.observe(trans)
        agent1.update()
        agent1._ret_p5 = -3.0
        agent1._ret_p95 = 9.0

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pt"
            agent1.save_checkpoint(ckpt_path)

            agent2 = DreamerV3Agent(cfg, seed=99)
            agent2.load_checkpoint(ckpt_path)
            self.assertEqual(agent2._ret_p5, -3.0)
            self.assertEqual(agent2._ret_p95, 9.0)

            for (k1, v1), (k2, v2) in zip(
                agent1.encoder.state_dict().items(), agent2.encoder.state_dict().items()
            ):
                self.assertTrue(torch.equal(v1, v2))

            legacy_path = Path(tmpdir) / "legacy-v1.pt"
            torch.save({"format": "haic-dreamerv3-checkpoint-v1"}, legacy_path)
            with self.assertRaisesRegex(ValueError, "full-stack decoder"):
                agent2.load_checkpoint(legacy_path)
            for (k1, v1), (k2, v2) in zip(
                agent1.actor.state_dict().items(), agent2.actor.state_dict().items()
            ):
                self.assertTrue(torch.equal(v1, v2))

    def test_diagnostic_uses_v2_heads_and_burnin_sequences(self):
        cfg = DreamerV3Config(
            device="cpu", embed_dim=64, hidden_dim=32,
            num_categoricals=4, num_classes=4, batch_size=2,
            seq_len=4, burnin_steps=2, replay_capacity=32,
        )
        agent = DreamerV3Agent(cfg, seed=73)
        for step in range(12):
            observation = np.full((4, 84, 84), step / 12.0, dtype=np.float32)
            agent.observe(Transition(
                observation=observation,
                action=np.zeros(3, dtype=np.float32),
                reward=float(step),
                next_observation=np.full_like(observation, (step + 1) / 12.0),
                terminated=step == 11,
                truncated=False,
                terminal=step == 11,
                step=step,
            ))
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "dreamer-v2.pt"
            agent.save_checkpoint(checkpoint)
            report = diagnose_world_model(checkpoint)

        self.assertEqual(report["environment_steps"], 12)
        self.assertTrue(np.isfinite(report["world_model"]["reward_mae"]))
        self.assertTrue(report["latent_imagination"]["values_finite"])

    def test_exported_actor_cpu_agent_inference_and_reset(self):
        cfg = DreamerV3Config(
            device="cpu",
            embed_dim=128,
            hidden_dim=64,
            num_categoricals=8,
            num_classes=8,
        )
        agent = DreamerV3Agent(cfg, seed=7)

        with tempfile.TemporaryDirectory() as tmpdir:
            actor_path = Path(tmpdir) / "actor.pt"
            agent.export_actor(actor_path)

            # Load through Agent interface in agent.py
            deployed = Agent(model_path=str(actor_path))
            self.assertEqual(deployed.format, DREAMERV3_ACTOR_FORMAT)

            obs1 = np.ones((4, 84, 84), dtype=np.float32)
            obs2 = np.zeros((4, 84, 84), dtype=np.float32)

            # Run 2 episodes and verify recurrent reset
            deployed.reset(None)
            act1 = deployed.act(obs1)
            act2 = deployed.act(obs2)
            self.assertEqual(act1.shape, (3,))
            self.assertEqual(act2.shape, (3,))
            self.assertTrue(-1.0 <= act1[0] <= 1.0)
            self.assertTrue(0.0 <= act1[1] <= 1.0)
            self.assertTrue(0.0 <= act1[2] <= 1.0)

            # Reset and verify exact deterministic reproducibility
            deployed.reset(None)
            act1_repeat = deployed.act(obs1)
            act2_repeat = deployed.act(obs2)
            np.testing.assert_allclose(act1, act1_repeat, atol=1e-6)
            np.testing.assert_allclose(act2, act2_repeat, atol=1e-6)

            # Verify speed (< 10 ms per step on CPU)
            t0 = time.perf_counter()
            for _ in range(50):
                deployed.act(obs1)
            avg_ms = (time.perf_counter() - t0) / 50 * 1000
            self.assertLess(avg_ms, 10.0)


if __name__ == "__main__":
    unittest.main()
