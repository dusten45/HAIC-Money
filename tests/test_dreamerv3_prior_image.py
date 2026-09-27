"""Synthetic contracts for the separate Dreamer free-prior image update."""

from __future__ import annotations

import copy
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from common_adapter import Transition
from dreamer_v3 import DreamerV3Agent, DreamerV3Config, NoValidSequenceError, Uint8SequenceReplay
from haic.algorithms.dreamer_v3.prior_image import prior_image_aux_update


def _batch(markers=(10, 20, 30, 40, 50, 60), *, burnin=1, batch_size=1):
    steps = len(markers) - 1
    observations = torch.zeros(batch_size, steps + 1, 4, 84, 84)
    for t, pixel in enumerate(markers):
        observations[:, t, -1] = pixel / 255.0
    actions = torch.tensor([[(t + 1) / 10, 0, 0] for t in range(steps)]).reshape(1, steps, 3)
    first = torch.zeros(batch_size, steps, dtype=torch.bool)
    last = torch.zeros_like(first)
    if steps:
        first[:, 0] = True
        last[:, -1] = True
    return {
        "observations": observations,
        "actions": actions.expand(batch_size, -1, -1).clone(),
        "is_first": first,
        "is_last": last,
        "is_terminal": last.clone(),
        "sequence_ids": torch.arange(steps).expand(batch_size, -1).clone(),
        "effective_burnin": burnin,
        "effective_seq_len": steps - burnin,
    }


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.reads = []
        self.grad_enabled = []

    def forward(self, stacks):
        self.reads.extend(stacks[:, -1, 0, 0].tolist())
        self.grad_enabled.append(torch.is_grad_enabled())
        return stacks[:, -1].mean(dim=(1, 2)).unsqueeze(-1)


class _RSSM(nn.Module):
    hidden_dim = stoch_dim = 1

    def __init__(self):
        super().__init__()
        self.prior_gain = nn.Parameter(torch.tensor(0.25))
        self.posterior_calls = []
        self.prior_calls = []

    def step_post(self, h, z, prev_a, embed):
        self.posterior_calls.append((prev_a.detach().clone(), embed.detach().clone(), torch.is_grad_enabled()))
        return h + self.prior_gain * (prev_a[:, :1] + embed), embed, None, None

    def step_prior(self, h, z, action):
        self.prior_calls.append((h.detach().clone(), z.detach().clone(), action.detach().clone(),
                                 torch.is_grad_enabled(), h.requires_grad))
        return h + self.prior_gain * action[:, :1] + 0.1 * z, z, None, None


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, state):
        return (self.gain * state[:, :1] + 0.03 * state[:, 1:2]).view(-1, 1, 1, 1).expand(-1, 1, 84, 84)


def _toy_agent(batch, *, observation_scale=2.5):
    cfg = DreamerV3Config(
        device="cpu", batch_size=batch["actions"].shape[0],
        burnin_steps=batch["effective_burnin"], seq_len=batch["effective_seq_len"],
        terminal_window_fraction=0.5, reset_start_fraction=0.25,
        short_episode_fraction=0.25, observation_loss_scale=observation_scale,
    )
    encoder, rssm, decoder = _Encoder(), _RSSM(), _Decoder()
    actor, critic, target = nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1)
    return SimpleNamespace(
        config=cfg, device=torch.device("cpu"), replay=SimpleNamespace(sample_sequence=Mock(return_value=batch)),
        encoder=encoder, rssm=rssm, decoder=decoder,
        wm_optimizer=torch.optim.Adam((*rssm.parameters(), *decoder.parameters()), lr=1e-3),
        actor=actor, critic=critic, critic_target=target,
        actor_optimizer=torch.optim.Adam(actor.parameters()),
        critic_optimizer=torch.optim.Adam(critic.parameters()),
        gradient_steps=7, environment_steps=13,
    )


class PriorImageTests(unittest.TestCase):
    def test_logged_action_t_predicts_only_successor_frame_t_plus_one(self):
        batch = _batch()
        agent = _toy_agent(batch)
        with patch.object(agent.wm_optimizer, "step", wraps=agent.wm_optimizer.step) as step:
            metrics = prior_image_aux_update(agent, horizon=2, weight=0.25)

        step.assert_called_once_with()
        self.assertEqual((metrics["anchor_count"], metrics["target_count"],
                          metrics["effective_horizon"]), (2.0, 4.0, 2.0))
        np.testing.assert_allclose([float(call[2][0, 0]) for call in agent.rssm.prior_calls],
                                   [0.2, 0.3, 0.4, 0.5], atol=1e-7)
        self.assertTrue(all(call[3] for call in agent.rssm.prior_calls))
        self.assertFalse(agent.rssm.prior_calls[0][4])  # Detached posterior, not a prior input graph.
        self.assertTrue(agent.rssm.prior_calls[1][4])   # Later prior inputs retain the prior graph.

        # Independent scalar reference: both the latent and decoded image feed back.
        next_pixels = []
        previous = {0: 20 / 255, 2: 40 / 255}
        for index, (h, z, action, _, _) in enumerate(agent.rssm.prior_calls):
            group = 0 if index < 2 else 2
            next_h = float(h.item()) + 0.25 * float(action[0, 0]) + 0.1 * float(z.item())
            pixel = min(1.0, max(0.0, previous[group] + 0.1 * next_h + 0.03 * float(z.item())))
            next_pixels.append(pixel)
            previous[group] = pixel
        targets = [30 / 255, 40 / 255, 50 / 255, 60 / 255]
        mse = sum((pred - true) ** 2 for pred, true in zip(next_pixels, targets, strict=True)) / 4
        self.assertAlmostEqual(metrics["frame_mse"], mse, places=6)
        self.assertAlmostEqual(metrics["aux_loss"], 0.25 * 0.5 * 2.5 * mse, places=6)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertGreater(agent.rssm.prior_gain.grad.abs().item(), 0.0)
        self.assertGreater(agent.decoder.gain.grad.abs().item(), 0.0)
        self.assertEqual(agent.gradient_steps, 7)
        self.assertEqual(agent.environment_steps, 13)

    def test_future_target_perturbation_changes_loss_not_same_anchor_prior_inputs(self):
        batch = _batch()
        changed = copy.deepcopy(batch)
        changed["observations"][:, 2, -1] = 200 / 255.0
        before, after = _toy_agent(batch), _toy_agent(changed)
        loss_before = prior_image_aux_update(before, horizon=2, weight=0.25)
        loss_after = prior_image_aux_update(after, horizon=2, weight=0.25)
        for original, altered in zip(before.rssm.prior_calls[:2], after.rssm.prior_calls[:2], strict=True):
            for left, right in zip(original[:3], altered[:3], strict=True):
                torch.testing.assert_close(left, right)
        self.assertNotEqual(loss_before["frame_mse"], loss_after["frame_mse"])

        # The last successor is never observed by the posterior; all rollout
        # inputs stay the same even when that target alone is changed.
        changed = copy.deepcopy(batch)
        changed["observations"][:, -1, -1] = 240 / 255.0
        final = _toy_agent(changed)
        final_loss = prior_image_aux_update(final, horizon=2, weight=0.25)
        self.assertNotEqual(loss_before["frame_mse"], final_loss["frame_mse"])
        self.assertEqual(before.encoder.reads, final.encoder.reads)
        self.assertEqual(len(before.encoder.reads), batch["actions"].shape[1])
        self.assertTrue(all(not enabled for enabled in before.encoder.grad_enabled))
        self.assertTrue(all(not call[2] for call in before.rssm.posterior_calls))
        for original, altered in zip(before.rssm.prior_calls, final.rssm.prior_calls, strict=True):
            for left, right in zip(original[:3], altered[:3], strict=True):
                torch.testing.assert_close(left, right)

    def test_full_horizon_first_and_last_anchors_use_native_actions_without_future_leak(self):
        batch = _batch(markers=tuple(range(41)), burnin=8, batch_size=2)
        batch["actions"][:, :, 0] = (torch.arange(40) % 7).float() / 10
        before = _toy_agent(batch)
        result = prior_image_aux_update(before, horizon=8, weight=0.25)
        self.assertEqual((result["anchor_count"], result["target_count"],
                          result["effective_horizon"]), (4.0, 32.0, 8.0))
        self.assertEqual(len(before.rssm.prior_calls), 16)
        for call, action_index in zip(before.rssm.prior_calls,
                                      (*range(8, 16), *range(32, 40)), strict=True):
            torch.testing.assert_close(call[2], batch["actions"][:, action_index])
        self.assertTrue(all(not enabled for enabled in before.encoder.grad_enabled))
        self.assertTrue(all(not call[2] for call in before.rssm.posterior_calls))

        # The terminal successor is a loss target, not a posterior input.
        changed_last = copy.deepcopy(batch)
        changed_last["observations"][:, 40, -1] = 240 / 255.0
        after = _toy_agent(changed_last)
        changed_result = prior_image_aux_update(after, horizon=8, weight=0.25)
        self.assertNotEqual(result["frame_mse"], changed_result["frame_mse"])
        self.assertEqual(before.encoder.reads, after.encoder.reads)
        for original, altered in zip(before.rssm.prior_calls, after.rssm.prior_calls, strict=True):
            for left, right in zip(original[:3], altered[:3], strict=True):
                torch.testing.assert_close(left, right)

        # A later current observation can reanchor the late window, but cannot
        # become an input to the earlier free-running eight-step prediction.
        changed_early = copy.deepcopy(batch)
        changed_early["observations"][:, 9, -1] = 200 / 255.0
        later = _toy_agent(changed_early)
        later_result = prior_image_aux_update(later, horizon=8, weight=0.25)
        self.assertNotEqual(result["frame_mse"], later_result["frame_mse"])
        for original, altered in zip(before.rssm.prior_calls[:8], later.rssm.prior_calls[:8], strict=True):
            for left, right in zip(original[:3], altered[:3], strict=True):
                torch.testing.assert_close(left, right)

    def test_overlapping_full_horizon_and_identical_anchors_count_targets(self):
        for markers, expected_anchors, expected_actions in (
            ((10, 20, 30, 40, 50), 2.0, [0.2, 0.3, 0.3, 0.4]),
            ((10, 20, 30, 40), 1.0, [0.2, 0.3]),
        ):
            with self.subTest(markers=markers):
                agent = _toy_agent(_batch(markers=markers, burnin=1))
                result = prior_image_aux_update(agent, horizon=2, weight=0.25)
                self.assertEqual(result["anchor_count"], expected_anchors)
                self.assertEqual(result["target_count"], float(len(expected_actions)))
                np.testing.assert_allclose([call[2][0, 0].item() for call in agent.rssm.prior_calls],
                                           expected_actions, atol=1e-7)

    def test_burnin_context_actions_and_observations_are_preserved(self):
        batch = _batch(burnin=2)
        before = _toy_agent(batch)
        prior_image_aux_update(before, horizon=2, weight=1.0)
        np.testing.assert_allclose([float(call[0][0, 0]) for call in before.rssm.posterior_calls],
                                   [0.0, 0.1, 0.2, 0.3, 0.4], atol=1e-7)
        self.assertAlmostEqual(float(before.rssm.prior_calls[0][0].item()),
                               0.25 * (10 / 255 + 0.1 + 20 / 255 + 0.2 + 30 / 255))
        changed = copy.deepcopy(batch)
        changed["observations"][:, 0, -1] = 120 / 255.0
        after = _toy_agent(changed)
        prior_image_aux_update(after, horizon=2, weight=1.0)
        self.assertNotEqual(float(before.rssm.prior_calls[0][0].item()),
                            float(after.rssm.prior_calls[0][0].item()))

    def test_terminal_successor_and_completed_short_episode_are_counted_once(self):
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                replay = Uint8SequenceReplay(capacity=4)
                replay.add(Transition(
                    observation=np.full((4, 84, 84), 10, dtype=np.uint8),
                    action=np.asarray([0.2, 0, 0], dtype=np.float32), reward=0,
                    next_observation=np.full((4, 84, 84), 50, dtype=np.uint8),
                    terminated=terminal, truncated=not terminal, terminal=terminal, episode_id=1, step=0,
                ))
                replay.add(Transition(
                    observation=np.full((4, 84, 84), 220, dtype=np.uint8),
                    action=np.zeros(3, dtype=np.float32), reward=0,
                    next_observation=np.full((4, 84, 84), 221, dtype=np.uint8),
                    terminated=False, truncated=False, episode_id=2, step=0,
                ))
                batch = replay.sample_sequence(1, 32, burnin=8, short_episode_fraction=1.0)
                self.assertAlmostEqual(batch["observations"][0, -1, -1, 0, 0].item(), 50 / 255)
                agent = _toy_agent(batch)
                agent.replay = replay
                agent.config.seq_len = 32
                agent.config.burnin_steps = 8
                agent.config.short_episode_fraction = 1.0
                metrics = prior_image_aux_update(agent, horizon=8, weight=0.25)
                self.assertEqual((metrics["anchor_count"], metrics["target_count"],
                                  metrics["effective_horizon"]), (1.0, 1.0, 1.0))
                self.assertEqual(len(agent.rssm.prior_calls), 1)
                np.testing.assert_allclose(agent.encoder.reads, [10 / 255], atol=1e-7)

    def test_terminal_at_end_caps_rollout_and_counts_its_real_successor(self):
        batch = _batch(markers=(10, 20, 30, 220), burnin=0)
        agent = _toy_agent(batch)
        result = prior_image_aux_update(agent, horizon=8, weight=0.25)
        self.assertEqual((result["anchor_count"], result["target_count"],
                          result["effective_horizon"]), (2.0, 4.0, 3.0))
        np.testing.assert_allclose([call[2][0, 0].item() for call in agent.rssm.prior_calls],
                                   [0.1, 0.2, 0.3, 0.3], atol=1e-7)
        np.testing.assert_allclose(agent.encoder.reads, [10 / 255, 20 / 255, 30 / 255], atol=1e-7)

    def test_malformed_cross_episode_batch_is_rejected_before_any_optimizer_step(self):
        for field in ("is_last", "is_terminal", "is_first", "sequence_ids"):
            with self.subTest(field=field):
                batch = _batch()
                if field == "sequence_ids":
                    batch[field][:, 3:] += 2
                else:
                    batch[field][:, 3 if field == "is_first" else 2] = True
                agent = _toy_agent(batch)
                with patch.object(agent.wm_optimizer, "step", wraps=agent.wm_optimizer.step) as step:
                    with self.assertRaisesRegex(ValueError, "crosses an episode boundary"):
                        prior_image_aux_update(agent, horizon=8, weight=0.25)
                step.assert_not_called()
                self.assertEqual(agent.encoder.reads, [])

    def test_sampling_settings_and_unchanged_actor_critic_target_optimizers_and_steps(self):
        batch = _batch(batch_size=2)
        agent = _toy_agent(batch)
        for optimizer, module in ((agent.actor_optimizer, agent.actor), (agent.critic_optimizer, agent.critic)):
            for param in module.parameters():
                optimizer.state[param]["step"] = torch.tensor(3.)
                optimizer.state[param]["exp_avg"] = torch.ones_like(param)
                optimizer.state[param]["exp_avg_sq"] = torch.ones_like(param)
        modules = [agent.actor, agent.critic, agent.critic_target]
        params_before = [copy.deepcopy(module.state_dict()) for module in modules]
        opts_before = [copy.deepcopy(opt.state_dict()) for opt in (agent.actor_optimizer, agent.critic_optimizer)]
        prior_image_aux_update(agent, horizon=8, weight=0.25)
        agent.replay.sample_sequence.assert_called_once_with(
            2, 4, burnin=1, terminal_fraction=0.5,
            reset_start_fraction=0.25, short_episode_fraction=0.25,
        )
        self.assertEqual((agent.gradient_steps, agent.environment_steps), (7, 13))
        for module, state in zip(modules, params_before, strict=True):
            for name, expected in state.items():
                torch.testing.assert_close(module.state_dict()[name], expected, rtol=0, atol=0)
        for opt, before in zip((agent.actor_optimizer, agent.critic_optimizer), opts_before, strict=True):
            now = opt.state_dict()
            self.assertEqual(now["param_groups"], before["param_groups"])
            for key, state in before["state"].items():
                for name, expected in state.items():
                    torch.testing.assert_close(now["state"][key][name], expected, rtol=0, atol=0)

    def test_rejects_invalid_budget_and_missing_valid_sequence_without_step(self):
        agent = _toy_agent(_batch())
        for horizon in (0, -1, 9, 1.5, True, "8"):
            with self.subTest(horizon=horizon), self.assertRaisesRegex(ValueError, "horizon"):
                prior_image_aux_update(agent, horizon=horizon, weight=0.25)
        for weight in (0, -1, float("nan"), float("inf"), -float("inf"), True, "0.25"):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "weight"):
                prior_image_aux_update(agent, horizon=8, weight=weight)
        agent.replay.sample_sequence.assert_not_called()
        agent.replay.sample_sequence.side_effect = NoValidSequenceError("no valid transition sequence")
        with patch.object(agent.wm_optimizer, "step", wraps=agent.wm_optimizer.step) as step:
            with self.assertRaisesRegex(NoValidSequenceError, "no valid transition"):
                prior_image_aux_update(agent, horizon=8, weight=0.25)
        step.assert_not_called()
        agent.replay.sample_sequence.side_effect = None
        empty = _batch(markers=(10,))
        agent.replay.sample_sequence.return_value = empty
        with self.assertRaisesRegex(ValueError, "learning window"):
            prior_image_aux_update(agent, horizon=8, weight=0.25)

    def test_native_rssm_prior_seed_reproducibility_on_synthetic_batch(self):
        batch = _batch(markers=(10, 20, 30, 40), burnin=1)
        config = DreamerV3Config(device="cpu", embed_dim=8, hidden_dim=8, num_categoricals=2,
                                 num_classes=3, batch_size=1, seq_len=2, burnin_steps=1,
                                 replay_capacity=8)
        agents = [DreamerV3Agent(config, seed=32), DreamerV3Agent(config, seed=32)]
        results = []
        for agent in agents:
            agent.replay.sample_sequence = Mock(return_value=batch)
            torch.manual_seed(404)
            results.append(prior_image_aux_update(agent, horizon=8, weight=0.25))
            self.assertGreater(agent.decoder.net[-1].weight.grad.abs().sum().item(), 0)
            self.assertEqual((agent.gradient_steps, agent.environment_steps), (0, 0))
        self.assertEqual(results[0], results[1])
        for name, tensor in agents[0].rssm.state_dict().items():
            torch.testing.assert_close(tensor, agents[1].rssm.state_dict()[name], rtol=0, atol=0)
        for name, tensor in agents[0].decoder.state_dict().items():
            torch.testing.assert_close(tensor, agents[1].decoder.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
