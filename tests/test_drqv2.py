import copy
import hashlib
import json
import random
import subprocess
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
    def make_agent(self, device="cpu", seed=3, steering_logit_l2=0.0):
        agent = DrQv2Agent(DrQv2Config(
            replay_capacity=32, batch_size=2, warmup_steps=0, n_step=3,
            feature_dim=16, hidden_dim=16, device=device,
            steering_logit_l2=steering_logit_l2,
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

    def test_steering_logit_coefficient_is_finite_nonnegative_and_zero_by_default(self):
        self.assertEqual(DrQv2Config().steering_logit_l2, 0.0)
        self.assertEqual(DrQv2Config(steering_logit_l2=0.001).steering_logit_l2, 0.001)
        for coefficient in (-0.001, float("nan"), float("inf"), float("-inf")):
            with self.subTest(coefficient=coefficient), self.assertRaisesRegex(ValueError, "steering_logit_l2"):
                DrQv2Config(steering_logit_l2=coefficient)

    def test_actor_objective_uses_only_steering_logits_from_same_forward_and_view(self):
        agent = self.make_agent(steering_logit_l2=0.001)
        agent.config.actor_update_frequency = 1
        with torch.no_grad():
            agent.actor.policy.weight.zero_()
            agent.actor.policy.bias.copy_(torch.tensor([2.0, 20.0, -30.0]))
        actor_outputs, q_outputs = [], []
        forward_actor, forward_q = agent.actor.forward, agent.critic_one.forward

        def record_actor(observation, **kwargs):
            result = forward_actor(observation, **kwargs)
            actor_outputs.append((observation, result))
            return result

        def record_q(observation, action):
            result = forward_q(observation, action)
            q_outputs.append((observation, action, result.detach().clone()))
            return result

        with (
            mock.patch.object(agent.actor, "forward", side_effect=record_actor) as actor_forward,
            mock.patch.object(agent.critic_one, "forward", side_effect=record_q),
            mock.patch("drq_v2.random_shift", wraps=random_shift) as augmentation,
        ):
            metrics = agent.update()
        self.assertEqual(actor_forward.call_count, 2)  # Target action, then actor objective.
        self.assertEqual(actor_forward.call_args.kwargs, {"return_logits": True})
        self.assertEqual(augmentation.call_count, 3)  # Keep the independent actor augmentation.
        actor_observation, (action, logits) = actor_outputs[-1]
        self.assertIs(q_outputs[-1][0], actor_observation)
        self.assertIs(q_outputs[-1][1], action)
        self.assertTrue(torch.equal(action, logits.tanh()))
        expected_q_loss = -q_outputs[-1][2].mean()
        expected_penalty = 0.001 * logits[:, 0].detach().square().mean()
        self.assertEqual(metrics["actor_q_loss"], float(expected_q_loss))
        self.assertEqual(metrics["steering_logit_mean_square"], 4.0)
        self.assertEqual(metrics["steering_logit_l2_penalty"], float(expected_penalty))
        self.assertEqual(metrics["actor_loss"], float(expected_q_loss + expected_penalty))
        self.assertEqual(metrics["steering_logit_abs_mean"], 2.0)
        self.assertEqual(metrics["steering_logit_abs_max"], 2.0)
        self.assertEqual(metrics["steering_saturation_fraction"], 0.0)
        self.assertAlmostEqual(metrics["steering_tanh_derivative_mean"], 1.0 - float(torch.tanh(torch.tensor(2.0)).square()))
        self.assertAlmostEqual(metrics["steering_logit_l2_grad_norm"], 0.004 / np.sqrt(2))
        self.assertGreater(metrics["actor_grad_norm"], 0.0)
        self.assertGreater(metrics["critic_grad_norm"], 0.0)

    def test_steering_penalty_gradient_survives_exact_tanh_saturation(self):
        for coefficient in (0.0, 0.001):
            with self.subTest(coefficient=coefficient):
                agent = self.make_agent(steering_logit_l2=coefficient)
                agent.config.actor_update_frequency = 1
                with torch.no_grad():
                    agent.actor.policy.weight.zero_()
                    agent.actor.policy.bias.copy_(torch.tensor([20.0, 30.0, -40.0]))
                before = copy.deepcopy(agent.actor.state_dict())
                metrics = agent.update()
                self.assertEqual(metrics["steering_logit_mean_square"], 400.0)
                self.assertEqual(metrics["steering_saturation_fraction"], 1.0)
                self.assertEqual(metrics["steering_tanh_derivative_mean"], 0.0)
                self.assertEqual(metrics["steering_tanh_zero_fraction"], 1.0)
                torch.testing.assert_close(
                    agent.actor.policy.bias.grad, torch.tensor([40.0 * coefficient, 0.0, 0.0]),
                )
                self.assertTrue(torch.equal(agent.actor.policy.weight.grad[1:], torch.zeros_like(agent.actor.policy.weight.grad[1:])))
                self.assertTrue(torch.equal(before["policy.bias"][1:], agent.actor.policy.bias[1:]))
                self.assertAlmostEqual(metrics["steering_logit_l2_grad_norm"], 40.0 * coefficient / np.sqrt(2))
                if coefficient:
                    self.assertGreater(metrics["actor_grad_norm"], 0.0)
                    self.assertGreater(metrics["actor_steering_head_grad_norm"], 0.0)
                    self.assertLess(float(agent.actor.policy.bias[0].detach()), 20.0)
                else:
                    self.assertEqual(metrics["actor_grad_norm"], 0.0)
                    self.assert_state_equal(before, agent.actor.state_dict())

    def check_zero_coefficient_baseline(self, device):
        baseline, agent = self.make_agent(device), self.make_agent(device)
        observation = ObservationSpec().from_uint8(_pixel_stack(0, 12))
        tensor = torch.as_tensor(observation, device=device).unsqueeze(0)
        rng_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
        with torch.no_grad():
            logits = baseline.actor.policy(baseline.actor.trunk(baseline.actor.encoder(tensor)))
            expected_action = torch.tanh(logits)
            self.assertTrue(torch.equal(agent.actor(tensor), expected_action))
            action, actual_logits = agent.actor(tensor, return_logits=True)
            self.assertTrue(torch.equal(actual_logits, logits))
            self.assertTrue(torch.equal(action, expected_action))
        native = expected_action.squeeze(0).cpu().numpy()
        np.testing.assert_array_equal(agent.act(observation), native)
        noisy = np.clip(native + baseline.rng.normal(0.0, baseline.exploration_std(), size=native.shape), -1.0, 1.0).astype(np.float32)
        np.testing.assert_array_equal(agent.act(observation, deterministic=False), noisy)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        if cuda_before:
            self.assert_state_equal(cuda_before, torch.cuda.get_rng_state_all())

        # Independent reference of the pre-regularization update, including its RNG order.
        expected_losses = []
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
            for _ in range(2):
                values = baseline._batch_tensors(baseline.replay.sample(baseline.config.batch_size))
                current = random_shift(values["observation"], pad=baseline.config.augmentation_pad)
                following = random_shift(values["next_observation"], pad=baseline.config.augmentation_pad)
                with torch.no_grad():
                    next_action = baseline.actor(following)
                    noise = (torch.randn_like(next_action) * baseline.config.target_policy_noise).clamp(
                        -baseline.config.target_policy_noise_clip, baseline.config.target_policy_noise_clip,
                    )
                    next_action = (next_action + noise).clamp(-1.0, 1.0)
                    target = values["reward"] + values["discount"] * torch.minimum(
                        baseline.target_one(following, next_action), baseline.target_two(following, next_action),
                    )
                q1 = baseline.critic_one(current, values["action"])
                q2 = baseline.critic_two(current, values["action"])
                critic_loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
                baseline.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critics = list(baseline.critic_one.parameters()) + list(baseline.critic_two.parameters())
                torch.nn.utils.clip_grad_norm_(critics, 10.0)
                baseline.critic_optimizer.step()
                baseline.gradient_steps += 1
                actor_loss = torch.zeros((), device=device)
                if baseline.gradient_steps % baseline.config.actor_update_frequency == 0:
                    for parameter in critics:
                        parameter.requires_grad_(False)
                    actor_observation = random_shift(values["observation"], pad=baseline.config.augmentation_pad)
                    actor_loss = -baseline.critic_one(actor_observation, baseline.actor(actor_observation)).mean()
                    baseline.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(baseline.actor.parameters(), 10.0)
                    baseline.actor_optimizer.step()
                    for parameter in critics:
                        parameter.requires_grad_(True)
                if baseline.gradient_steps % baseline.config.target_update_frequency == 0:
                    baseline._soft_update_targets()
                expected_losses.append((float(critic_loss.detach()), float(actor_loss.detach())))
            expected_state = copy.deepcopy(baseline._payload())
        metrics = [agent.update(), agent.update()]
        self.assertEqual(expected_losses, [(row["critic_loss"], row["actor_loss"]) for row in metrics])
        self.assertEqual(metrics[-1]["actor_loss"], metrics[-1]["actor_q_loss"])
        self.assertEqual(metrics[-1]["steering_logit_l2_penalty"], 0.0)
        self.assertEqual(metrics[-1]["steering_logit_l2_grad_norm"], 0.0)
        actual_state = agent._payload()
        expected_state.pop("last_actor_metrics")
        actual_state.pop("last_actor_metrics")
        self.assert_state_equal(expected_state, actual_state)
        for expected, actual in zip(baseline.actor.parameters(), agent.actor.parameters()):
            self.assertTrue(torch.equal(expected.grad, actual.grad))

    def test_zero_coefficient_preserves_baseline_cpu_actions_updates_and_rng(self):
        self.check_zero_coefficient_baseline("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_zero_coefficient_preserves_baseline_cuda_actions_updates_and_rng(self):
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True):
            self.check_zero_coefficient_baseline("cuda")

    def check_checkpoint_continuation(self, device, steering_logit_l2):
        agent = self.make_agent(device, steering_logit_l2=steering_logit_l2)
        actor_before = copy.deepcopy(agent.actor.state_dict())
        critics_before = [copy.deepcopy(critic.state_dict()) for critic in (agent.critic_one, agent.critic_two)]
        targets_before = [copy.deepcopy(target.state_dict()) for target in (agent.target_one, agent.target_two)]
        first_metrics = agent.update()
        self.assertEqual(agent.gradient_steps, 1)
        self.assertEqual(first_metrics["actor_updated"], 0.0)
        self.assertEqual(agent.last_actor_metrics, {})
        self.assert_state_equal(actor_before, agent.actor.state_dict())
        for before, target in zip(targets_before, (agent.target_one, agent.target_two)):
            self.assert_state_equal(before, target.state_dict())
        for before, critic in zip(critics_before, (agent.critic_one, agent.critic_two)):
            self.assertTrue(any(not torch.equal(before[key], critic.state_dict()[key]) for key in before))
        metrics = agent.update()
        self.assertEqual(agent.gradient_steps, 2)
        self.assertEqual(metrics["actor_updated"], 1.0)
        self.assertEqual(metrics["actor_metrics_gradient_step"], 2.0)
        self.assertAlmostEqual(
            metrics["steering_logit_l2_penalty"], steering_logit_l2 * metrics["steering_logit_mean_square"],
        )
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
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(saved["config"]["steering_logit_l2"], steering_logit_l2)
            self.assertEqual(saved["last_actor_metrics"], agent.last_actor_metrics)
            checkpoint.with_suffix(".manifest.json").unlink()
            expected_action = agent.act(observation, deterministic=False)
            expected_python_random = random.random()
            expected_batch = agent.replay.sample(2)
            expected_metrics = [agent.update(), agent.update()]
            self.assertEqual(expected_metrics[0]["actor_updated"], 0.0)
            self.assertEqual(expected_metrics[0]["actor_loss"], 0.0)
            for key, value in saved["last_actor_metrics"].items():
                self.assertEqual(expected_metrics[0][key], value)
            self.assertEqual(expected_metrics[1]["actor_metrics_gradient_step"], 4.0)
            expected_state = copy.deepcopy(agent._payload())
            restored = DrQv2Agent(agent.config, seed=99)
            with mock.patch("drq_v2.torch.load", wraps=torch.load) as loader:
                returned_state = restored.load_checkpoint(checkpoint)
                self.assertEqual(loader.call_args.kwargs["map_location"], "cpu")
            self.assert_state_equal(trainer_state, returned_state)
            self.assertEqual(restored.last_actor_metrics, saved["last_actor_metrics"])
            np.testing.assert_array_equal(expected_action, restored.act(observation, deterministic=False))
            self.assertEqual(expected_python_random, random.random())
            self.assert_state_equal(expected_batch, restored.replay.sample(2))
            self.assertEqual(expected_metrics, [restored.update(), restored.update()])
            self.assert_state_equal(expected_state, restored._payload())

    def test_cpu_checkpoint_restores_actor_targets_and_stochastic_updates(self):
        for coefficient in (0.0, 0.001):
            with self.subTest(coefficient=coefficient):
                self.check_checkpoint_continuation("cpu", coefficient)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_checkpoint_restores_actor_targets_and_stochastic_updates(self):
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True):
            for coefficient in (0.0, 0.001):
                with self.subTest(coefficient=coefficient):
                    self.check_checkpoint_continuation("cuda", coefficient)

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
                (("config", "steering_logit_l2"), 0.001),
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
        payload.pop("last_actor_metrics")
        payload["config"].pop("steering_logit_l2")
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
                    self.assertEqual(restored.last_actor_metrics, {})

    def test_checkpoint_only_defaults_the_known_legacy_zero_coefficient(self):
        agent = self.make_agent()
        payload = agent._payload()
        payload["config"].pop("steering_logit_l2")
        legacy = copy.deepcopy(payload)
        restored = self.make_agent(seed=99)
        before = copy.deepcopy(restored._payload())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pt"
            for missing in payload["config"]:
                with self.subTest(missing=missing):
                    corrupt = copy.deepcopy(legacy)
                    corrupt["config"].pop(missing)
                    torch.save(corrupt, checkpoint)
                    with self.assertRaisesRegex(ValueError, "configuration fields"):
                        restored.load_checkpoint(checkpoint)
                    self.assert_state_equal(before, restored._payload())
            corrupt = copy.deepcopy(legacy)
            corrupt["config"]["unknown_objective"] = 0.0
            torch.save(corrupt, checkpoint)
            with self.assertRaisesRegex(ValueError, "configuration fields"):
                restored.load_checkpoint(checkpoint)
            torch.save(legacy, checkpoint)
            restored.load_checkpoint(checkpoint)
            self.assert_state_equal(agent.actor.state_dict(), restored.actor.state_dict())
            treatment = self.make_agent(steering_logit_l2=0.001)
            before = copy.deepcopy(treatment._payload())
            with self.assertRaisesRegex(ValueError, "steering_logit_l2"):
                treatment.load_checkpoint(checkpoint)
            self.assert_state_equal(before, treatment._payload())
            treatment.save_checkpoint(checkpoint)
            before = copy.deepcopy(restored._payload())
            with self.assertRaisesRegex(ValueError, "steering_logit_l2"):
                restored.load_checkpoint(checkpoint)
            self.assert_state_equal(before, restored._payload())

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
        from agent import Agent

        observations = [ObservationSpec().from_uint8(_pixel_stack(0, step)) for step in (0, 1, 5)]
        for coefficient in (0.0, 0.001):
            agent = self.make_agent(steering_logit_l2=coefficient)
            agent.update()
            agent.update()
            expected = CPUActorAdapter(agent.actor, agent.action_adapter).deterministic_trace(observations)
            for legacy in ((False, True) if coefficient == 0.0 else (False,)):
                with self.subTest(coefficient=coefficient, legacy=legacy), tempfile.TemporaryDirectory() as directory:
                    export = agent.export_actor(Path(directory) / "actor.pt")
                    payload = torch.load(export, map_location="cpu", weights_only=True)
                    self.assertEqual(payload["config"]["steering_logit_l2"], coefficient)
                    self.assertEqual(set(payload["state_dict"]), set(agent.actor.state_dict()))
                    if legacy:
                        payload["config"].pop("steering_logit_l2")
                        torch.save(payload, export)
                    before = torch.get_rng_state().clone()
                    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
                    actor, adapter, spec = load_exported_actor(export)
                    submission = Agent(model_path=export)
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))
                    if cuda_before:
                        self.assert_state_equal(cuda_before, torch.cuda.get_rng_state_all())
                    policy = CPUActorAdapter(actor, adapter, spec)
                    self.assertEqual(expected, policy.deterministic_trace(observations))
                    self.assertEqual(expected, policy.deterministic_trace(observations))
                    for _ in range(2):
                        submission.reset(observations[0])
                        actual = [submission.act(observation).tolist() for observation in observations]
                        self.assertEqual(expected["actions"], actual)
                    actions = np.asarray(expected["actions"])
                    self.assertTrue(np.all(actions >= [-1.0, 0.0, 0.0]))
                    self.assertTrue(np.all(actions <= [1.0, 1.0, 1.0]))

    def test_current_runtime_export_loads_in_isolated_cpu21_minimal_agent(self):
        cpu_python = Path("/tmp/kilo/haic-cpu21/bin/python")
        if not cpu_python.is_file():
            self.skipTest("isolated Torch 2.1 CPU interpreter unavailable")
        agent = self.make_agent("cuda" if torch.cuda.is_available() else "cpu", steering_logit_l2=0.001)
        agent.update()
        agent.update()
        observations = [np.full((4, 84, 84), value, dtype=np.float32) for value in (0.0, 0.25, 1.0)]
        expected = CPUActorAdapter(copy.deepcopy(agent.actor).cpu(), agent.action_adapter).deterministic_trace(observations)
        script = """
import json
import sys
import numpy as np
import torch
from agent import Agent

assert torch.__version__ == '2.1.0+cpu'
assert not {'drq_v2', 'common_adapter', 'train_drqv2', 'stable_baselines3'} & set(sys.modules)
observations = [np.full((4, 84, 84), value, dtype=np.float32) for value in (0.0, 0.25, 1.0)]
traces = []
for _ in range(2):
    policy = Agent(model_path=sys.argv[1])
    policy.reset(observations[0])
    traces.append([policy.act(observation).tolist() for observation in observations])
print(json.dumps({'traces': traces, 'coefficient': policy.export_metadata['config']['steering_logit_l2']}))
"""
        with tempfile.TemporaryDirectory() as directory:
            export = agent.export_actor(Path(directory) / "actor.pt")
            result = subprocess.run(
                [str(cpu_python), "-c", script, str(export)],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads(result.stdout)
        self.assertEqual(actual["coefficient"], 0.001)
        self.assertEqual(actual["traces"][0], actual["traces"][1])
        # CPU reloads must be exact; different Torch versions can round differently.
        np.testing.assert_allclose(actual["traces"][0], expected["actions"], rtol=0.0, atol=1e-6)

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
