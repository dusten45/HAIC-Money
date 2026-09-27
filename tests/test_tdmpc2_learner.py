"""Synthetic TD-MPC2 learner tests; no environment interaction."""

import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from haic.algorithms.tdmpc2.learner import TDMPC2Learner, TDMPC2LearnerConfig
from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel
from haic.algorithms.tdmpc2.replay import EpisodeReplay


def small_model(*, episodic=True):
    cfg = TDMPC2ModelConfig(
        obs_shape={"rgb": (4, 64, 64)}, num_channels=2, latent_dim=32,
        mlp_dim=32, num_q=3, num_bins=11, dropout=0, tau=0.2,
        episodic=episodic,
    )
    return WorldModel(cfg)


def sample(*, h=2, b=3):
    return {
        "obs": torch.randint(0, 256, (h + 1, b, 4, 64, 64), dtype=torch.uint8),
        "action": torch.rand(h, b, 3) * 2 - 1,
        "reward": torch.randn(h, b, 1),
        "terminated": torch.zeros(h, b, 1),
        "truncated": torch.zeros(h, b, 1),
        "terminal": torch.zeros(h, b, 1),
        "bootstrap_mask": torch.ones(h, b, 1),
    }


class TestTDMPC2Learner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._old_threads)

    def setUp(self):
        torch.manual_seed(127)

    def test_td_target_distinguishes_finish_and_timeout(self):
        learner = TDMPC2Learner(small_model(), TDMPC2LearnerConfig(episode_length=100, horizon=1, batch_size=3))
        self.assertAlmostEqual(learner.discount, 0.95)
        next_z = torch.zeros(1, 3, learner.model.cfg.latent_dim, requires_grad=True)
        reward = torch.tensor([[[1.0], [2.0], [3.0]]])
        # The middle transition is a HAIC finish, the last is a time limit.
        terminal = torch.tensor([[[0.0], [1.0], [0.0]]])
        truncated = torch.tensor([[[0.0], [1.0], [1.0]]])
        with patch.object(learner.model, "Q", return_value=torch.full_like(reward, 7.0)) as critic:
            target = learner._td_target(next_z, reward, terminal, truncated)
        torch.testing.assert_close(target, torch.tensor([[[7.65], [2.0], [9.65]]]))
        self.assertFalse(target.requires_grad)
        self.assertEqual(critic.call_args.kwargs, {"return_type": "min", "target": True})

    def test_joint_update_has_finite_connected_losses_and_exact_target_ema(self):
        model = small_model()
        learner = TDMPC2Learner(model, TDMPC2LearnerConfig(episode_length=500, horizon=2, batch_size=3))
        self.assertAlmostEqual(learner.discount, 0.99)
        data = sample()
        data["terminated"][-1, 0] = 1
        data["terminal"][-1, 0] = 1
        data["bootstrap_mask"][-1, 0] = 0
        before = {
            "encoder": model._encoder["rgb"][2].weight.detach().clone(),
            "dynamics": model._dynamics[-1].weight.detach().clone(),
            "reward": model._reward[-1].weight.detach().clone(),
            "q": model._Qs[0][-1].weight.detach().clone(),
            "termination": model._termination[-1].weight.detach().clone(),
            "policy": model._pi[-1].weight.detach().clone(),
        }
        target_before = [p.detach().clone() for p in model._target_Qs.parameters()]

        class Replay:
            requested_batch_size = None

            def sample(self, batch_size):
                self.requested_batch_size = batch_size
                return data

        replay = Replay()
        metrics = learner.update(replay)
        self.assertEqual(replay.requested_batch_size, 3)
        for value in metrics.values():
            self.assertTrue(math.isfinite(value))
        self.assertGreater(metrics["consistency_loss"], 0)
        self.assertGreater(metrics["reward_loss"], 0)
        self.assertGreater(metrics["value_loss"], 0)
        self.assertGreater(metrics["termination_loss"], 0)
        self.assertLessEqual(metrics["grad_norm"], 1e5)
        self.assertGreaterEqual(metrics["pi_scale"], 1.0)
        self.assertFalse(model.training)

        after = {
            "encoder": model._encoder["rgb"][2].weight,
            "dynamics": model._dynamics[-1].weight,
            "reward": model._reward[-1].weight,
            "q": model._Qs[0][-1].weight,
            "termination": model._termination[-1].weight,
            "policy": model._pi[-1].weight,
        }
        for name, parameter in after.items():
            self.assertTrue(torch.isfinite(parameter).all(), name)
            self.assertGreater((before[name] - parameter).abs().sum().item(), 0, name)
        for old, online, target in zip(target_before, model._Qs.parameters(), model._target_Qs.parameters()):
            torch.testing.assert_close(target, old.lerp(online.detach(), model.cfg.tau), rtol=1e-5, atol=1e-6)
            self.assertFalse(target.requires_grad)

    def test_td_target_uses_observed_next_encoded_state_not_predicted_latent(self):
        model = small_model()
        learner = TDMPC2Learner(model, TDMPC2LearnerConfig(episode_length=500, horizon=1, batch_size=2))
        data = sample(h=1, b=2)
        actual_encode = model.encode
        encoded = []

        def record_encode(obs, task):
            latent = actual_encode(obs, task)
            encoded.append((obs, latent))
            return latent

        with patch.object(model, "encode", side_effect=record_encode), patch.object(
            learner, "_td_target", wraps=learner._td_target
        ) as target:
            learner.update(data)
        self.assertEqual(len(encoded), 2)
        torch.testing.assert_close(encoded[0][0], data["obs"][1:])
        torch.testing.assert_close(target.call_args.args[0], encoded[0][1])
        self.assertFalse(target.call_args.args[0].requires_grad)

    def test_finish_label_trains_termination_head_but_timeout_does_not(self):
        model = small_model()
        learner = TDMPC2Learner(model, TDMPC2LearnerConfig(episode_length=500, horizon=1, batch_size=2))
        with torch.no_grad():
            model._termination[-1].weight.zero_()
            model._termination[-1].bias.fill_(2)
        data = sample(h=1, b=2)
        data["truncated"][:] = 1
        data["terminal"][0, 0] = 1  # HAIC finish: raw terminated is zero.
        data["bootstrap_mask"][0, 0] = 0
        expected = F.binary_cross_entropy_with_logits(torch.full((1, 2, 1), 2.0), data["terminal"]).item()
        with patch.object(learner, "_td_target", wraps=learner._td_target) as target:
            metrics = learner.update(data)
        torch.testing.assert_close(target.call_args.args[2], data["terminal"])
        self.assertAlmostEqual(metrics["termination_loss"], expected, places=5)

    def test_rejects_raw_termination_bootstrap_mask_on_finish(self):
        learner = TDMPC2Learner(small_model(), TDMPC2LearnerConfig(episode_length=500, horizon=1, batch_size=2))
        data = sample(h=1, b=2)
        data["truncated"][0, 0] = 1
        data["terminal"][0, 0] = 1
        with self.assertRaisesRegex(ValueError, "bootstrap mask"):
            learner.update(data)

    def test_completed_episode_replay_feeds_semantic_finish_to_learner(self):
        replay = EpisodeReplay(capacity=8, horizon=2, seed=3)
        pixels = torch.randint(0, 256, (4, 64, 64), dtype=torch.uint8)
        replay.start_episode(pixels)
        replay.add_step(pixels, torch.zeros(3), 0.5)
        replay.add_step(pixels, torch.zeros(3), 1.0, truncated=True, terminal=True)
        sampled = replay.sample(2)
        self.assertTrue(torch.all(sampled["terminated"] == 0))
        self.assertTrue(torch.all(sampled["terminal"][-1] == 1))
        self.assertTrue(torch.all(sampled["bootstrap_mask"][-1] == 0))
        learner = TDMPC2Learner(small_model(), TDMPC2LearnerConfig(episode_length=500, horizon=2, batch_size=2))
        self.assertTrue(math.isfinite(learner.update(replay)["total_loss"]))

    def test_non_episodic_model_rejects_semantic_terminal(self):
        learner = TDMPC2Learner(small_model(episodic=False), TDMPC2LearnerConfig(episode_length=500, horizon=1, batch_size=2))
        data = sample(h=1, b=2)
        data["terminal"][0, 0] = 1
        data["bootstrap_mask"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "episodic=True"):
            learner.update(data)


if __name__ == "__main__":
    unittest.main()
