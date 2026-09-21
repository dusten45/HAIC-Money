import copy
import hashlib
import random
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from unittest import mock

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
    load_exported_actor,
    random_shift,
)


def _pixel_stack(episode, step):
    pixels = np.arange(84 * 84, dtype=np.uint16).reshape(84, 84)
    return np.stack([
        ((pixels + episode * 31 + max(0, step + channel - 3) * 7) % 256).astype(np.uint8)
        for channel in range(4)
    ])


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
        self.assertEqual(encoded[0, 0, 0], 255)
        self.assertEqual(encoded[3, -1, -1], 128)
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
        np.testing.assert_array_equal(adapter.to_official([2.0, -2.0, 2.0]), [1.0, 0.0, 1.0])

    def test_frozen_specs_reject_changed_bounds_order_and_frame_skip(self):
        for changes in ({"low": -1.0}, {"high": 255.0}, {"uint8_scale": 255.5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ObservationSpec(**changes)
        for changes in (
            {"native_low": (-2.0, -1.0, -1.0)},
            {"official_low": (-1.0, -1.0, 0.0)},
            {"order": ("gas", "steer", "brake")},
            {"method": "other"},
            {"frame_skip": 8},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ActionSpec(**changes)

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

    def test_collector_outcomes_actions_and_metadata_reach_replay(self):
        for terminated, truncated, info, expected_discount in (
            (False, True, {"finished": True}, 0.0),
            (True, False, {"retire_reason": "crash"}, 0.0),
            (True, False, {"retire_reason": "off_track"}, 0.0),
            (True, True, {}, 0.0),
            (False, True, {}, 0.9),
        ):
            with self.subTest(terminated=terminated, truncated=truncated, info=info):
                env = _SequenceEnv([(terminated, truncated, info)])
                collector = EpisodeCollector(env, gamma=0.9)
                collector.reset()
                transition = collector.step([2.0, -2.0, 2.0])
                replay = Uint8Replay(capacity=8, n_step=3, gamma=0.9)
                replay.add(transition)
                batch = replay.sample(1, indices=[0])
                self.assertAlmostEqual(float(batch["discount"][0]), expected_discount)
                self.assertEqual(len(env.actions), 1)
                np.testing.assert_array_equal(transition.applied_action, env.actions[0])
                np.testing.assert_array_equal(batch["action"][0], [1.0, -1.0, 1.0])
                self.assertEqual(transition.info["track_id"], 3)
                self.assertEqual(transition.info["seed"], 99)
                collector.reset()
                self.assertEqual(collector.episode_id, 1)
                self.assertEqual(collector.step_index, 0)

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

    def test_manifest_rejects_inconsistent_frame_skip_and_missing_source(self):
        kwargs = dict(algorithm="drq-v2", reward_contract={}, max_steps=1, seeds=[0])
        with self.assertRaises(ValueError):
            build_checkpoint_manifest(**kwargs, frame_skip=8, action_spec=ActionSpec())
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            build_checkpoint_manifest(
                **kwargs, frame_skip=4, source_paths=[Path(directory) / "missing.py"],
            )


class TestUint8Replay(unittest.TestCase):
    def transition(self, episode, step, reward=1.0, terminated=False, truncated=False):
        observation = ObservationSpec().from_uint8(_pixel_stack(episode, step))
        next_observation = ObservationSpec().from_uint8(_pixel_stack(episode, step + 1))
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
        for horizon in (1, 2, 3):
            with self.subTest(horizon=horizon):
                replay = Uint8Replay(capacity=16, n_step=3, gamma=0.9, seed=1)
                for step in range(horizon):
                    replay.add(self.transition(4, step, reward=step + 1, truncated=step == horizon - 1))
                replay.add(self.transition(5, 0, reward=100.0))
                batch = replay.sample(1, indices=[0])
                self.assertAlmostEqual(float(batch["discount"][0]), 0.9**horizon)
                self.assertAlmostEqual(
                    float(batch["reward"][0]), sum((step + 1) * 0.9**step for step in range(horizon)), places=6,
                )
                self.assertEqual(int(batch["terminated"][0]), 0)
                self.assertEqual(int(batch["truncated"][0]), 1)
                np.testing.assert_array_equal(batch["next_observation"][0], _pixel_stack(4, horizon))

    def test_sampling_never_uses_cross_episode_n_step_path(self):
        replay = Uint8Replay(capacity=16, n_step=3, gamma=0.99, seed=2)
        replay.add(self.transition(1, 0))
        replay.add(self.transition(2, 0))
        with self.assertRaises(ValueError):
            replay.sample(1, indices=[0])
        replay = Uint8Replay(capacity=16, n_step=3)
        for step in range(3):
            replay.add(self.transition(1, step))
        replay.add(self.transition(2, 0))
        with self.assertRaises(ValueError):
            replay.sample(1, indices=[0])

    def test_ring_wrap_preserves_available_stacks_without_episode_start(self):
        replay = Uint8Replay(capacity=8, n_step=3, gamma=0.9, seed=1)
        for step in range(12):
            replay.add(self.transition(0, step, reward=step + 1))
        self.assertEqual(replay.oldest_sequence, 4)
        self.assertEqual(replay.valid_indices(), [7, 8])
        batch = replay.sample(2, indices=[7, 8])
        for row, step in enumerate((7, 8)):
            np.testing.assert_array_equal(batch["observation"][row], _pixel_stack(0, step))
            np.testing.assert_array_equal(batch["next_observation"][row], _pixel_stack(0, step + 3))
            self.assertAlmostEqual(
                float(batch["reward"][row]), sum((step + offset + 1) * 0.9**offset for offset in range(3)), places=5,
            )
        restored = Uint8Replay(capacity=8, n_step=3, gamma=0.9, seed=99)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(restored.valid_indices(), [7, 8])
        expected, actual = replay.sample(2), restored.sample(2)
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])

    def test_ring_wrap_preserves_terminal_boundary_and_new_episode_padding(self):
        replay = Uint8Replay(capacity=8, n_step=3, gamma=0.9)
        for step in range(12):
            replay.add(self.transition(0, step, truncated=step == 11))
        replay.add(self.transition(1, 0, terminated=True))
        self.assertEqual(replay.valid_indices(), [8, 9, 10, 11, 12])
        batch = replay.sample(2, indices=[11, 12])
        np.testing.assert_array_equal(batch["next_observation"][0], _pixel_stack(0, 12))
        np.testing.assert_array_equal(batch["observation"][1], _pixel_stack(1, 0))
        np.testing.assert_allclose(batch["discount"], [0.9, 0.0])

    def test_reset_padding_and_reused_episode_id_do_not_cross_a_reset(self):
        replay = Uint8Replay(capacity=16, n_step=1)
        for step in range(5):
            replay.add(self.transition(0, step))
        for step in range(4):
            np.testing.assert_array_equal(replay.sample(1, indices=[step])["observation"][0], _pixel_stack(0, step))
        for n_step in (1, 3):
            with self.subTest(n_step=n_step):
                replay = Uint8Replay(capacity=8, n_step=n_step)
                replay.add(self.transition(0, 0))
                replay.add(self.transition(0, 0))
                with self.assertRaises(ValueError):
                    replay.sample(1, indices=[0])


class TestDrQv2(unittest.TestCase):
    def make_agent(self, device="cpu", seed=3):
        agent = DrQv2Agent(DrQv2Config(
            replay_capacity=32, batch_size=2, warmup_steps=0, n_step=3,
            feature_dim=16, hidden_dim=16, device=device,
        ), seed=seed)
        for step in range(12):
            agent.observe(Transition(
                observation=_pixel_stack(0, step), action=np.zeros(3, dtype=np.float32),
                reward=step + 1, next_observation=_pixel_stack(0, step + 1),
                terminated=False, truncated=False, episode_id=0, step=step,
            ))
        return agent

    def assert_state_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(expected.cpu(), actual.cpu()))
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(expected, actual)
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_state_equal(expected[key], actual[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(expected), len(actual))
            for first, second in zip(expected, actual):
                self.assert_state_equal(first, second)
        else:
            self.assertEqual(expected, actual)

    def test_random_shift_is_seeded_and_keeps_uint8_contract(self):
        observations = torch.arange(2 * 4 * 84 * 84, dtype=torch.uint8).reshape(2, 4, 84, 84)
        first = random_shift(observations, pad=2, seed=7)
        second = random_shift(observations, pad=2, seed=7)
        self.assertEqual(first.dtype, torch.uint8)
        self.assertEqual(first.shape, observations.shape)
        self.assertTrue(torch.equal(first, second))

    def check_checkpoint_continuation(self, device):
        agent = self.make_agent(device)
        actor_before = copy.deepcopy(agent.actor.state_dict())
        critics_before = [copy.deepcopy(critic.state_dict()) for critic in (agent.critic_one, agent.critic_two)]
        targets_before = [copy.deepcopy(target.state_dict()) for target in (agent.target_one, agent.target_two)]
        agent.update()
        self.assertEqual(agent.gradient_steps, 1)
        self.assert_state_equal(actor_before, agent.actor.state_dict())
        for before, target in zip(targets_before, (agent.target_one, agent.target_two)):
            self.assert_state_equal(before, target.state_dict())
        for before, critic in zip(critics_before, (agent.critic_one, agent.critic_two)):
            self.assertTrue(any(not torch.equal(before[key], critic.state_dict()[key]) for key in before))
        metrics = agent.update()
        self.assertEqual(agent.gradient_steps, 2)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertTrue(any(not torch.equal(actor_before[key], agent.actor.state_dict()[key]) for key in actor_before))
        self.assertTrue(agent.actor_optimizer.state)
        for before, target, critic in zip(
            targets_before, (agent.target_one, agent.target_two), (agent.critic_one, agent.critic_two),
        ):
            for name, value in target.state_dict().items():
                expected = before[name].mul(1.0 - agent.config.tau).add(critic.state_dict()[name], alpha=agent.config.tau)
                self.assertTrue(torch.equal(value, expected))
            self.assertTrue(all(parameter.grad is None for parameter in target.parameters()))
            self.assertTrue(all(parameter.requires_grad for parameter in critic.parameters()))
        observation = ObservationSpec().from_uint8(_pixel_stack(0, 12))
        trainer_state = {
            "episode_id": 0, "step_index": 12, "native_action_prefix": [np.zeros(3, dtype=np.float32)],
            "sampler_rng": np.random.default_rng(917).bit_generator.state,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            agent.save_checkpoint(checkpoint, trainer_state=trainer_state)
            checkpoint.with_suffix(".manifest.json").unlink()
            expected_action = agent.act(observation, deterministic=False)
            expected_python_random = random.random()
            expected_batch = agent.replay.sample(2)
            expected_metrics = [agent.update(), agent.update()]
            expected_state = copy.deepcopy(agent._payload())
            restored = DrQv2Agent(agent.config, seed=99)
            with mock.patch("drq_v2.torch.load", wraps=torch.load) as loader:
                returned_state = restored.load_checkpoint(checkpoint)
                self.assertEqual(loader.call_args.kwargs["map_location"], "cpu")
            self.assert_state_equal(trainer_state, returned_state)
            np.testing.assert_array_equal(expected_action, restored.act(observation, deterministic=False))
            self.assertEqual(expected_python_random, random.random())
            self.assert_state_equal(expected_batch, restored.replay.sample(2))
            self.assertEqual(expected_metrics, [restored.update(), restored.update()])
            self.assert_state_equal(expected_state, restored._payload())

    def test_cpu_checkpoint_restores_actor_targets_and_stochastic_updates(self):
        self.check_checkpoint_continuation("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_checkpoint_restores_actor_targets_and_stochastic_updates(self):
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True):
            self.check_checkpoint_continuation("cuda")

    def test_checkpoint_rejects_contract_mismatch_before_mutating_agent(self):
        agent = self.make_agent()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            agent.save_checkpoint(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            invalid_path = Path(directory) / "invalid.pt"
            for keys, value in (
                (("config", "tau"), 0.7),
                (("config", "actor_update_frequency"), 7),
                (("config", "augmentation_pad"), 2),
                (("config", "replay_capacity"), 64),
                (("config", "gamma"), 0.9),
                (("observation_spec", "high"), 255.0),
                (("observation_spec", "channel_order"), "HWC"),
                (("observation_spec", "control_plane_fingerprint"), "extra-plane"),
                (("action_spec", "frame_skip"), 8),
                (("action_spec", "order"), ("gas", "steer", "brake")),
                (("manifest", "observation", "normalization"), [-1.0, 1.0]),
                (("manifest", "action", "frame_skip"), 8),
                (("manifest", "frame_skip"), 8),
            ):
                with self.subTest(keys=keys):
                    corrupted = copy.deepcopy(payload)
                    field = corrupted
                    for key in keys[:-1]:
                        field = field[key]
                    field[keys[-1]] = value
                    torch.save(corrupted, invalid_path)
                    restored = DrQv2Agent(agent.config, seed=99)
                    before = copy.deepcopy(restored._payload())
                    with self.assertRaises(ValueError):
                        restored.load_checkpoint(invalid_path)
                    self.assert_state_equal(before, restored._payload())

    def test_legacy_checkpoints_remain_loadable_without_claiming_cuda_rng_restore(self):
        agent = self.make_agent()
        payload = agent._payload()
        payload.pop("torch_cuda_rng_state")
        # Configuration reconstructed from the historical JSON manifest has a list shape.
        config = replace(agent.config, observation_shape=list(agent.config.observation_shape))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pt"
            for saved_device in ("cpu", "cuda"):
                with self.subTest(saved_device=saved_device):
                    payload["config"]["device"] = saved_device
                    torch.save(payload, checkpoint)
                    restored = DrQv2Agent(config, seed=99)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        self.assertIsNone(restored.load_checkpoint(checkpoint))
                    if saved_device == "cuda":
                        self.assertEqual(len(caught), 1)
                        self.assertIn("no CUDA RNG state", str(caught[0].message))
                    else:
                        self.assertEqual(caught, [])
                    self.assert_state_equal(agent.actor.state_dict(), restored.actor.state_dict())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_checkpoint_can_relocate_weights_and_optimizer_to_cpu(self):
        agent = self.make_agent("cuda")
        agent.update()
        agent.update()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            agent.save_checkpoint(checkpoint)
            restored = DrQv2Agent(replace(agent.config, device="cpu"), seed=99)
            with self.assertWarnsRegex(RuntimeWarning, "device type changed"):
                restored.load_checkpoint(checkpoint)
            self.assert_state_equal(agent.actor.state_dict(), restored.actor.state_dict())
            self.assert_state_equal(agent.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict())
            for optimizer in (restored.actor_optimizer, restored.critic_optimizer):
                for state in optimizer.state.values():
                    self.assertTrue(all(value.device.type == "cpu" for value in state.values() if isinstance(value, torch.Tensor)))
            self.assertTrue(all(np.isfinite(value) for value in restored.update().values()))

    def test_export_load_preserves_rng_and_cpu_action_trace(self):
        agent = self.make_agent()
        agent.update()
        agent.update()
        observations = [ObservationSpec().from_uint8(_pixel_stack(0, step)) for step in (0, 1, 5)]
        with tempfile.TemporaryDirectory() as directory:
            export = agent.export_actor(Path(directory) / "actor.pt")
            before = torch.get_rng_state().clone()
            cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
            actor, adapter, spec = load_exported_actor(export)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            if cuda_before:
                self.assert_state_equal(cuda_before, torch.cuda.get_rng_state_all())
            expected = CPUActorAdapter(agent.actor, agent.action_adapter).deterministic_trace(observations)
            policy = CPUActorAdapter(actor, adapter, spec)
            self.assertEqual(expected, policy.deterministic_trace(observations))
            self.assertEqual(expected, policy.deterministic_trace(observations))
            actions = np.asarray(expected["actions"])
            self.assertTrue(np.all(actions >= [-1.0, 0.0, 0.0]))
            self.assertTrue(np.all(actions <= [1.0, 1.0, 1.0]))

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
