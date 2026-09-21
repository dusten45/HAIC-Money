import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from common_adapter import (
    ActionAdapter,
    ActionSpec,
    CPUActorAdapter,
    EpisodeCollector,
    ObservationSpec,
    Transition,
    build_checkpoint_manifest,
)
from drq_v2 import (
    DrQv2Agent,
    DrQv2Config,
    Uint8Replay,
    random_shift,
)


class _SequenceEnv:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.actions = []
        self.index = 0
        self.observation_space = type("Space", (), {"shape": (4, 84, 84)})()

    def reset(self, *, seed=None, options=None):
        self.index = 0
        return np.full((4, 84, 84), 0.25, dtype=np.float32), {
            "track_id": 3,
            "seed": 99,
        }

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32))
        outcome = self.outcomes[min(self.index, len(self.outcomes) - 1)]
        self.index += 1
        value = 0.25 + self.index / 10.0
        observation = np.full((4, 84, 84), value, dtype=np.float32)
        terminated, truncated, info = outcome
        return observation, 1.0, terminated, truncated, dict(info)


class TestCommonAdapter(unittest.TestCase):
    def test_observation_conversion_preserves_chw_frame_order(self):
        spec = ObservationSpec()
        chw = np.zeros(spec.shape, dtype=np.float32)
        chw[0, 0, 0] = 1.0
        chw[3, -1, -1] = 0.5
        encoded = spec.to_uint8(chw)
        self.assertEqual(encoded.dtype, np.uint8)
        np.testing.assert_allclose(spec.from_uint8(encoded), encoded.astype(np.float32) / 255.0)

        hwc = np.transpose(chw, (1, 2, 0))
        np.testing.assert_array_equal(spec.to_uint8(hwc, source_channel_order="HWC"), encoded)

    def test_action_adapter_maps_symmetric_native_head_to_official_box(self):
        adapter = ActionAdapter(ActionSpec())
        np.testing.assert_allclose(adapter.to_official([-1.0, -1.0, 1.0]), [-1.0, 0.0, 1.0])
        np.testing.assert_allclose(adapter.to_official([0.0, 0.0, 0.0]), [0.0, 0.5, 0.5])
        np.testing.assert_allclose(adapter.to_native([-1.0, 0.0, 1.0]), [-1.0, -1.0, 1.0])
        with self.assertRaises(ValueError):
            adapter.to_official([0.0, np.nan, 0.0])

    def test_collector_calls_one_high_level_step_and_distinguishes_truncation(self):
        env = _SequenceEnv([(False, True, {}), (False, False, {})])
        collector = EpisodeCollector(env, gamma=0.99)
        observation, _ = collector.reset(seed=4)
        self.assertEqual(observation.shape, (4, 84, 84))
        transition = collector.step([0.0, 0.0, 0.0])
        self.assertFalse(transition.terminated)
        self.assertTrue(transition.truncated)
        self.assertTrue(transition.bootstrap_allowed)
        self.assertEqual(len(env.actions), 1)

        env = _SequenceEnv([(False, True, {"finished": True})])
        collector = EpisodeCollector(env, gamma=0.99)
        collector.reset()
        transition = collector.step([0.0, 0.0, 0.0])
        self.assertFalse(transition.bootstrap_allowed)
        with self.assertRaises(RuntimeError):
            collector.step([0.0, 0.0, 0.0])

    def test_checkpoint_manifest_contains_contract_fingerprints(self):
        manifest = build_checkpoint_manifest(
            algorithm="drq-v2",
            reward_contract={"finish_bonus": 0.0},
            frame_skip=4,
            max_steps=2000,
            seeds=[7],
            model_path="actor.pt",
            replay_path="replay.npz",
        )
        self.assertEqual(manifest["algorithm"], "drq-v2")
        self.assertEqual(manifest["observation"]["shape"], [4, 84, 84])
        self.assertEqual(manifest["action"]["official_low"], [-1.0, 0.0, 0.0])
        self.assertEqual(manifest["paths"]["replay"], "replay.npz")


class TestUint8Replay(unittest.TestCase):
    def transition(self, episode, step, reward=1.0, terminated=False, truncated=False):
        value = np.float32(((episode * 10 + step) % 80) / 100.0)
        observation = np.full((4, 84, 84), value, dtype=np.float32)
        next_observation = np.full((4, 84, 84), value + 0.01, dtype=np.float32)
        return Transition(
            observation=observation,
            action=np.zeros(3, dtype=np.float32),
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
            terminal=terminated,
            episode_id=episode,
            step=step,
        )

    def test_three_step_return_stops_at_terminal_without_crossing_episode(self):
        replay = Uint8Replay(capacity=32, n_step=3, gamma=0.9, seed=1)
        for step in range(2):
            replay.add(self.transition(1, step, reward=float(step + 1), terminated=step == 1))
        replay.add(self.transition(2, 0, reward=100.0))
        batch = replay.sample(1, indices=[0])
        self.assertAlmostEqual(float(batch["reward"][0]), 1.0 + 0.9 * 2.0)
        self.assertAlmostEqual(float(batch["discount"][0]), 0.0)
        self.assertEqual(int(batch["horizon"][0]), 2)
        self.assertEqual(int(batch["episode_id"][0]), 1)

    def test_truncation_bootstraps_from_boundary_observation(self):
        replay = Uint8Replay(capacity=16, n_step=3, gamma=0.9, seed=1)
        replay.add(self.transition(4, 0, reward=2.0, truncated=True))
        batch = replay.sample(1, indices=[0])
        self.assertAlmostEqual(float(batch["discount"][0]), 0.9)
        self.assertEqual(int(batch["terminated"][0]), 0)
        self.assertEqual(int(batch["truncated"][0]), 1)
        self.assertTrue(np.isfinite(batch["next_observation"]).all())

    def test_sampling_never_uses_cross_episode_n_step_path(self):
        replay = Uint8Replay(capacity=16, n_step=3, gamma=0.99, seed=2)
        replay.add(self.transition(1, 0))
        replay.add(self.transition(2, 0))
        with self.assertRaises(ValueError):
            replay.sample(1, indices=[0])


class TestDrQv2(unittest.TestCase):
    def test_random_shift_is_seeded_and_keeps_uint8_contract(self):
        observations = torch.arange(2 * 4 * 84 * 84, dtype=torch.uint8).reshape(2, 4, 84, 84)
        first = random_shift(observations, pad=2, seed=7)
        second = random_shift(observations, pad=2, seed=7)
        self.assertEqual(first.dtype, torch.uint8)
        self.assertEqual(first.shape, observations.shape)
        self.assertTrue(torch.equal(first, second))

    def test_agent_update_checkpoint_restore_and_cpu_action_parity(self):
        config = DrQv2Config(
            replay_capacity=32,
            batch_size=2,
            warmup_steps=0,
            n_step=2,
            feature_dim=32,
            hidden_dim=32,
            device="cpu",
        )
        agent = DrQv2Agent(config, seed=3)
        for step in range(8):
            transition = Transition(
                observation=np.full((4, 84, 84), step / 10.0, dtype=np.float32),
                action=np.zeros(3, dtype=np.float32),
                reward=1.0,
                next_observation=np.full((4, 84, 84), (step + 1) / 10.0, dtype=np.float32),
                terminated=False,
                truncated=False,
                episode_id=0,
                step=step,
            )
            agent.observe(transition)
        metrics = agent.update()
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        observation = np.zeros((4, 84, 84), dtype=np.float32)
        adapter = CPUActorAdapter(agent.actor, agent.action_adapter)
        action_one = adapter.act(observation)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            agent.save_checkpoint(checkpoint)
            restored = DrQv2Agent(config, seed=99)
            restored.load_checkpoint(checkpoint)
            restored_adapter = CPUActorAdapter(restored.actor, restored.action_adapter)
            action_two = restored_adapter.act(observation)
        np.testing.assert_allclose(action_one, action_two, atol=1e-6)
        self.assertTrue(np.isfinite(action_one).all())
        self.assertTrue(np.all(action_one >= [-1.0, 0.0, 0.0]))
        self.assertTrue(np.all(action_one <= [1.0, 1.0, 1.0]))

    def test_manifest_source_hash_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            path.write_text("value = 1\n")
            manifest = build_checkpoint_manifest(
                algorithm="drq-v2",
                source_paths=[path],
                reward_contract={},
                frame_skip=4,
                max_steps=1,
                seeds=[1],
            )
            self.assertEqual(
                manifest["source_hashes"][str(path)],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
