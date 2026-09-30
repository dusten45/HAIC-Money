"""CPU-only actor repair contracts; no real protocol freeze or learner run."""

from dataclasses import asdict
import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from haic.algorithms.rlpd import agent as agent_module
from scripts import train_rlpd_recovery_actor_only as repair


class FakeEncoder(nn.Module):
    latent_dim = 4

    def __init__(self):
        super().__init__()
        self.convolution = nn.Linear(4, 4)
        self.projection = nn.Linear(4, 4)

    def forward(self, images):
        return self.projection(self.convolution(images.float().mean((2, 3)) / 255.0))


class FakeActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FakeEncoder()
        self.trunk = nn.Sequential(nn.Linear(4, 8), nn.Tanh())
        self.mean = nn.Linear(8, 3)
        self.log_std = nn.Linear(8, 3)


class FakeCritic(nn.Module):
    def __init__(self, num_qs=10):
        super().__init__()
        self.encoder = FakeEncoder()
        self.head = nn.Linear(4, num_qs)


def source_state():
    original = agent_module.PixelRLPDAgent()
    # Populate original Adam moments/steps, including the frozen std/projection.
    for optimizer, parameters in ((original.actor_optimizer, original.actor.parameters()),
                                  (original.critic_optimizer, original.critic.parameters()),
                                  (original.temperature_optimizer, [original.log_alpha])):
        optimizer.zero_grad(set_to_none=True)
        torch.stack([p.square().sum() for p in parameters if p.requires_grad]).sum().backward()
        optimizer.step()
    return {"format": "haic-rlpd-training-checkpoint-v1", "config": asdict(original.config),
            **{name: getattr(original, name).state_dict() for name in
               ("actor", "critic", "target_critic", "actor_optimizer", "critic_optimizer", "temperature_optimizer")},
            "log_alpha": original.log_alpha.detach().clone(),
            "environment_steps": 131072, "gradient_steps": 130073}


def image_pool():
    rng = np.random.default_rng(11)
    images = rng.integers(0, 256, (7, 4, 84, 84), dtype=np.uint8)
    return {"guide": repair.deduplicate(images, np.tile([-.7, -.8, .9], (7, 1)).astype(np.float32)),
            "prior": repair.deduplicate(images), "protected": repair.deduplicate(images[:3])}


def snapshot_fixture(seed=4272000001):
    n = 26
    images = np.stack([np.zeros((4, 84, 84), np.uint8), np.full((4, 84, 84), 20, np.uint8)])
    hashes = np.full(n, "unused", dtype="U64")
    for step, image in zip((0, 25), images):
        hashes[step] = hashlib.sha256((image.astype(np.float32) / 255).tobytes()).hexdigest()
    truncated = np.zeros(n, bool)
    truncated[-1] = True
    finished = np.zeros(n, bool)
    finished[-1] = seed in repair.FINISHED
    reasons = np.full(n, "", dtype="U64")
    reasons[-1] = "" if finished[-1] else "crash"
    data = {"step": np.arange(n), "terminated": np.zeros(n, bool), "truncated": truncated,
            "finished": finished, "retire_reason": reasons, "snapshot_steps": np.array([0, 25]),
            "observation_snapshots": images, "observation_sha256": hashes}
    receipt = {"steps": n, "actor_id": "v5", "actor_sha256": repair.ACTOR_SHA,
               "geometry_seed": seed, "track_id": 1, "censored": False, "mode": "policy",
               "source_road_hash_matches": True, "terminated": False, "truncated": True,
               "finished": bool(finished[-1]), "reason": "finished" if finished[-1] else "crash",
               "initial_observation_sha256": str(hashes[0])}
    return data, receipt


class ActorOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(60)
        self.actor_patch = patch.object(agent_module, "PixelActor", FakeActor)
        self.critic_patch = patch.object(agent_module, "PixelCritic", FakeCritic)
        self.actor_patch.start()
        self.critic_patch.start()
        self.addCleanup(self.actor_patch.stop)
        self.addCleanup(self.critic_patch.stop)

    def make_agent(self):
        state = source_state()
        return repair.ActorOnlyAgent(agent_module.RLPDConfig(), state), state

    def test_full_2048_budget_frozen_weights_and_adam(self):
        agent, state = self.make_agent()
        self.assertEqual(repair.state_sha(agent.actor.state_dict()), repair.state_sha(state["actor"]))
        self.assertEqual(repair.state_sha(agent.actor_optimizer.state_dict()), repair.state_sha(state["actor_optimizer"]))
        initial_mean = agent.actor.mean.weight.detach().clone()
        original_steps = [agent.actor_optimizer.state[p]["step"].item() for p in agent.actor.mean.parameters()]
        pools = agent.prepare_features(image_pool(), batch_size=2)
        logs = []
        result = repair.fit(agent, pools, journal=logs.append)
        self.assertEqual(result["actor_only_updates"], 2048)
        self.assertEqual(len(logs), 2048)
        self.assertEqual(result["frozen_hashes_before"], result["frozen_hashes_after"])
        self.assertEqual(agent.gradient_steps, 0)
        self.assertEqual(agent.environment_steps, 0)
        self.assertFalse(torch.equal(initial_mean, agent.actor.mean.weight))
        for i, p in enumerate(agent.actor.mean.parameters()):
            self.assertEqual(agent.actor_optimizer.state[p]["step"].item(), original_steps[i] + 2048)
        self.assertTrue(all(p.grad is None for p in agent.actor.log_std.parameters()))
        self.assertTrue(all(p.grad is None for p in agent.actor.encoder.parameters()))
        self.assertLess(result["final_prediction_errors"]["guide"]["mse"],
                        result["initial_prediction_errors"]["guide"]["mse"])
        self.assertEqual(logs[-1]["batch"], {"guide": 16, "prior": 32, "protected": 16})

    def test_three_native_axes_receive_gradients(self):
        agent, _ = self.make_agent()
        repair.fit(agent, agent.prepare_features(image_pool()), updates=1)
        assert agent.actor.mean.weight.grad is not None
        self.assertTrue((agent.actor.mean.weight.grad.abs().sum(1) > 0).all())
        self.assertEqual(agent.actor.mean.weight.grad.shape[0], 3)

    def test_retention_targets_are_original_same_image_mean(self):
        agent, _ = self.make_agent()
        pools = image_pool()
        prepared = agent.prepare_features(pools)
        for name in ("prior", "protected"):
            with torch.no_grad():
                obs = torch.from_numpy(pools[name]["images"])
                expected = agent.reference_actor.mean(agent.reference_actor.trunk(agent.reference_actor.encoder(obs))).tanh()
            torch.testing.assert_close(prepared[name]["targets"], expected, rtol=0, atol=0)
            self.assertFalse(torch.equal(prepared[name]["targets"], prepared["guide"]["targets"][:len(expected)]))
        original = prepared["prior"]["targets"].clone()
        repair.fit(agent, prepared, updates=5)
        torch.testing.assert_close(prepared["prior"]["targets"], original, rtol=0, atol=0)

    def test_sac_forbidden_and_no_environment_callers(self):
        agent, _ = self.make_agent()
        with self.assertRaisesRegex(RuntimeError, "SAC forbidden"):
            agent.update({})
        source = inspect.getsource(repair)
        for forbidden in ("build_env(", "env.reset(", "train_loop(", "EpisodeCollector("):
            self.assertNotIn(forbidden, source)
        self.assertEqual(len(repair.SOURCE_FILES), len(repair.parent.SOURCE_FILES) + 2)

    def test_literal_protocol_pins_budget_sources_and_objective(self):
        protocol = repair.contract(agent_module.RLPDConfig(), {"dataset_manifest_sha256": repair.DATASET_SHA})
        self.assertEqual(protocol["budget"]["actor_only_updates"], 2048)
        self.assertEqual(protocol["budget"]["seed"], 60)
        self.assertEqual(protocol["budget"]["guide_weight"], 1)
        self.assertEqual(protocol["budget"]["retention_weight"], 1)
        self.assertEqual(protocol["budget"]["sac_updates"], 0)
        self.assertEqual(protocol["source_checkpoint_sha256"], repair.parent.CHECKPOINT_SHA)
        self.assertEqual(set(protocol["source_hashes"]), repair.SOURCE_FILES)
        self.assertEqual(protocol["objective"]["trainable"], ["actor.trunk", "actor.mean"])
        self.assertFalse(protocol["objective"]["reward_used"])
        self.assertFalse(protocol["resume_supported"])

    def test_dedup_keeps_fixed_row_weights(self):
        images = np.zeros((3, 4, 84, 84), np.uint8)
        images[2] = 1
        pool = repair.deduplicate(images)
        np.testing.assert_array_equal(pool["weights"], [2, 1])
        self.assertEqual(repair.pool_receipt(pool)["rows"], 3)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            repair.deduplicate(images, np.eye(3, dtype=np.float32))

    def test_protected_full_snapshot_selection_and_initial_only(self):
        for seed, count in ((4272000001, 3), (4272000002, 1)):
            data, receipt = snapshot_fixture(seed)
            self.assertEqual(len(repair.validate_snapshots(data, receipt, seed)), count)

    def test_initial_positive_snapshot_retains_double_weight_after_dedup(self):
        data, receipt = snapshot_fixture()
        pool = repair.deduplicate(repair.validate_snapshots(data, receipt, 4272000001))
        np.testing.assert_array_equal(pool["weights"], [2, 1])

    def test_protected_file_sha_checked_before_snapshot_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "runs" / "protected"
            folder.mkdir(parents=True)
            manifest = folder / "manifest.json"
            manifest.write_text('{"episodes_complete": 36, "files_sha256": {}}')
            expected = repair.sha256_file(manifest)
            manifest.write_text('{"episodes_complete": 12, "files_sha256": {}}')
            with patch.object(repair.parent, "ROOT", root), patch.object(repair, "PROTECTED", "runs/protected"), \
                    patch.object(repair, "PROTECTED_SHA", expected), self.assertRaisesRegex(ValueError, "changed frozen evidence"):
                repair.load_protected()

    def test_no_resume_or_nonpositive_budget(self):
        agent, _ = self.make_agent()
        pools = agent.prepare_features(image_pool())
        with self.assertRaises(ValueError):
            repair.fit(agent, pools, updates=0)
        repair.fit(agent, pools, updates=1)
        with self.assertRaises(ValueError):
            repair.fit(agent, pools, updates=1)

    def test_nonfinite_targets_fail_before_optimizer_step(self):
        agent, _ = self.make_agent()
        pools = agent.prepare_features(image_pool())
        before = repair.state_sha(agent.actor.state_dict())
        pools["guide"]["targets"].fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            repair.fit(agent, pools, updates=1)
        self.assertEqual(before, repair.state_sha(agent.actor.state_dict()))
        self.assertEqual(agent.actor_only_updates, 0)

    def test_protected_rejects_image_hash_and_missing_snapshots(self):
        data, receipt = snapshot_fixture()
        data["observation_snapshots"][1, 0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            repair.validate_snapshots(data, receipt, 4272000001)
        data, receipt = snapshot_fixture()
        data["snapshot_steps"] = np.array([0])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            repair.validate_snapshots(data, receipt, 4272000001)

    def test_protected_rejects_partial_or_wrong_finish(self):
        for key, value in (("censored", True), ("finished", False), ("actor_sha256", "x" * 64),
                           ("source_road_hash_matches", False)):
            data, receipt = snapshot_fixture()
            receipt[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                repair.validate_snapshots(data, receipt, 4272000001)
        data, receipt = snapshot_fixture()
        data["truncated"][-1] = False
        with self.assertRaises(ValueError):
            repair.validate_snapshots(data, receipt, 4272000001)

    def test_standard_actor_export_unchanged_format(self):
        agent, _ = self.make_agent()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            agent.export_actor(path, source_sha256="a" * 64, protocol_sha256="b" * 64,
                               training_seed=60, environment_contract={"actor_only_updates": 0})
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["format"], "haic-rlpd-pixel-actor-v1")
            self.assertEqual(payload["gradient_steps"], 0)
            self.assertEqual(repair.state_sha(payload["actor_state_dict"]), repair.state_sha(agent.actor.state_dict()))


if __name__ == "__main__":
    unittest.main()
