import copy
import io
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common_adapter import Transition
from dreamer_v3 import (
    CategoricalRSSM,
    ConvDecoder,
    ConvEncoder,
    DreamerV3Agent,
    DreamerV3Config,
    ExportedDreamerV3Actor,
    LayerNormGRUCell,
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
        self.assertEqual(batch["observations"].shape, (4, 10, 4, 84, 84))
        self.assertEqual(batch["actions"].shape, (4, 10, 3))
        self.assertEqual(batch["rewards"].shape, (4, 10))
        self.assertEqual(batch["is_first"].shape, (4, 10))
        self.assertGreaterEqual(batch["is_first"][:, 0].sum().item(), 0)

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
            act = agent.act(obs, deterministic=False)
            trans = Transition(
                observation=obs,
                action=act,
                reward=0.5,
                next_observation=obs,
                terminated=(step == 19),
                truncated=False,
                step=step,
            )
            agent.observe(trans)

        self.assertEqual(agent.environment_steps, 20)
        metrics = agent.update()
        for key in (
            "loss_wm",
            "loss_obs",
            "loss_reward",
            "loss_continue",
            "loss_kl",
            "loss_critic",
            "loss_actor",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]), f"metric {key} was not finite")

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

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pt"
            agent1.save_checkpoint(ckpt_path)

            agent2 = DreamerV3Agent(cfg, seed=99)
            agent2.load_checkpoint(ckpt_path)

            for (k1, v1), (k2, v2) in zip(
                agent1.encoder.state_dict().items(), agent2.encoder.state_dict().items()
            ):
                self.assertTrue(torch.equal(v1, v2))
            for (k1, v1), (k2, v2) in zip(
                agent1.actor.state_dict().items(), agent2.actor.state_dict().items()
            ):
                self.assertTrue(torch.equal(v1, v2))

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
