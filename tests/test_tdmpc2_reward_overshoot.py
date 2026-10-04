"""Synthetic-only H3 reward-overshoot contracts; never construct a HAIC env."""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from haic.algorithms.tdmpc2.learner import TDMPC2Learner, TDMPC2LearnerConfig
from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel, soft_ce
from haic.algorithms.tdmpc2.replay import EpisodeReplay
from haic.algorithms.tdmpc2.reward_overshoot import OvershootLearner, OvershootReplay


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pixels(value):
    return np.full((4, 64, 64), value, dtype=np.uint8)


def episode(replay, initial, count, *, ending="timeout"):
    replay.start_episode(pixels(initial))
    for index in range(count):
        final = index == count - 1
        action = np.array([initial / 100, index / 10, -index / 10], dtype=np.float32)
        replay.add_step(
            pixels(initial + index + 1), action, float(initial + index + 1),
            terminated=final and ending == "raw",
            truncated=final and ending != "raw",
            terminal=final and ending in ("raw", "finish"),
        )


def tiny_model():
    return WorldModel(TDMPC2ModelConfig(
        obs_shape={"rgb": (4, 64, 64)}, num_channels=2, latent_dim=32,
        mlp_dim=32, num_q=3, num_bins=11, dropout=0, tau=0.2, episodic=True,
    ))


def assert_exact_tree(a, b):
    assert type(a) is type(b)
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_exact_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_exact_tree(x, y)
    elif isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.cpu().contiguous().numpy().tobytes() == b.cpu().contiguous().numpy().tobytes()
    else:
        assert a == b


@pytest.mark.parametrize("augment", [False, True])
def test_h3_population_bytes_and_both_sampler_rngs_are_unchanged(augment):
    overshoot = OvershootReplay(32, seed=41)
    for start, length, ending in ((10, 6, "timeout"), (30, 4, "finish"), (50, 5, "raw")):
        episode(overshoot, start, length, ending=ending)
    original = EpisodeReplay(32, horizon=3, seed=999)
    original.load_state_dict(overshoot.state_dict())
    original_sample = EpisodeReplay.sample
    calls = []

    def count_base(self, *args, **kwargs):
        calls.append(self)
        return original_sample(self, *args, **kwargs)

    with patch.object(EpisodeReplay, "sample", count_base):
        for _ in range(2):
            gen_a = torch.Generator().manual_seed(501)
            gen_b = torch.Generator().manual_seed(501)
            a = overshoot.sample(96, augment=augment, generator=gen_a)
            b = original.sample(96, augment=augment, generator=gen_b)
            assert calls[-2:] == [overshoot, original]
            assert len(a) == len(b) + 3
            for name in b:
                assert_exact_tree(a[name], b[name])
            assert_exact_tree(gen_a.get_state(), gen_b.get_state())
            assert_exact_tree(overshoot.rng.bit_generator.state, original.rng.bit_generator.state)
    assert len(calls) == 4  # Exactly one base sample per call, no resampling.


def test_suffix_actions_rewards_and_masks_never_cross_finish_timeout_raw_end():
    replay = OvershootReplay(18, seed=3)
    for start, length, ending in ((10, 3, "timeout"), (30, 4, "finish"),
                                  (50, 5, "raw"), (70, 6, "timeout")):
        episode(replay, start, length, ending=ending)
    batch = replay.sample(500)
    assert batch["overshoot_action"].shape == (2, 500, 3)
    assert batch["overshoot_reward"].shape == (2, 500, 1)
    assert batch["overshoot_mask"].shape == (2, 500, 1)
    assert set(batch["episode_id"].tolist()) == {0, 1, 2, 3}
    lengths = (3, 4, 5, 6)
    initial = (10, 30, 50, 70)
    endings = ("timeout", "finish", "raw", "timeout")
    for column, (ep_id, start) in enumerate(zip(batch["episode_id"].tolist(), batch["start_step"].tolist())):
        length = lengths[ep_id]
        for offset in range(2):
            index = start + 3 + offset
            valid = index < length
            assert batch["overshoot_mask"][offset, column, 0].item() == float(valid)
            expected_action = (
                np.array([initial[ep_id] / 100, index / 10, -index / 10], dtype=np.float32)
                if valid else np.zeros(3, dtype=np.float32)
            )
            assert batch["overshoot_action"][offset, column].numpy().tobytes() == expected_action.tobytes()
            assert batch["overshoot_reward"][offset, column, 0].item() == (
                float(initial[ep_id] + index + 1) if valid else 0.0
            )
        if start + 3 == length:
            assert batch["truncated"][-1, column, 0].item() == float(endings[ep_id] != "raw")
            assert batch["terminal"][-1, column, 0].item() == float(endings[ep_id] != "timeout")
            assert batch["bootstrap_mask"][-1, column, 0].item() == float(endings[ep_id] == "timeout")
            assert not batch["overshoot_mask"][:, column].any()


def test_partial_episode_prefix_trim_and_eviction_never_read_other_episode():
    replay = OvershootReplay(6, seed=12, include_partial=True)
    replay.start_episode(pixels(10))
    for index in range(9):
        replay.add_step(pixels(11 + index), np.array([0, index / 10, 0], np.float32), index + 1.)
    assert replay.state_dict()["active"]["start_step"] == 3
    batch = replay.sample(100)
    assert set(batch["start_step"].tolist()) == {3, 4, 5, 6}
    for column, start in enumerate(batch["start_step"].tolist()):
        assert batch["overshoot_mask"][:, column, 0].tolist() == [
            float(start + 3 < 9), float(start + 4 < 9),
        ]
    replay.add_step(pixels(20), np.zeros(3, np.float32), 10., truncated=True)
    episode(replay, 40, 3)
    assert replay.state_dict()["episodes"][0]["start_step"] == 7
    after = replay.sample(120)
    assert set(after["episode_id"].tolist()) == {0, 1}
    assert all(start == (7 if ep_id == 0 else 0) for ep_id, start in zip(
        after["episode_id"].tolist(), after["start_step"].tolist()
    ))
    assert not after["overshoot_mask"].any()
    assert not after["overshoot_action"].any()
    assert not after["overshoot_reward"].any()


def test_disabled_aux_one_update_bitwise_parity_including_optimizer_and_rng():
    torch.manual_seed(27)
    replay = OvershootReplay(12, seed=31)
    episode(replay, 10, 5, ending="finish")
    data = replay.sample(3)
    source = tiny_model()
    config = TDMPC2LearnerConfig(episode_length=500, horizon=3, batch_size=3)
    original = TDMPC2Learner(deepcopy(source), config)
    disabled = OvershootLearner(deepcopy(source), config, aux_enabled=False)
    assert_exact_tree(original.state_dict(), disabled.state_dict())
    before = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    metrics_a = original.update(data)
    after_a = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    metrics_b = disabled.update(data)
    after_b = torch.get_rng_state().clone()
    assert metrics_a == metrics_b and "overshoot_reward_loss" not in metrics_b
    assert_exact_tree(after_a, after_b)
    assert_exact_tree(original.state_dict(), disabled.state_dict())
    assert_exact_tree(original.optim.state_dict(), disabled.optim.state_dict())
    assert_exact_tree(original.pi_optim.state_dict(), disabled.pi_optim.state_dict())


def test_enabled_with_only_masked_suffix_preserves_original_h3_update():
    torch.manual_seed(91)
    replay = OvershootReplay(3, seed=11)
    episode(replay, 10, 3, ending="timeout")
    data = replay.sample(2)
    assert not data["overshoot_mask"].any()
    source = tiny_model()
    config = TDMPC2LearnerConfig(episode_length=500, batch_size=2)
    original = TDMPC2Learner(deepcopy(source), config)
    extended = OvershootLearner(deepcopy(source), config)
    before = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    expected = original.update(data)
    after_original = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    actual = extended.update(data)
    assert actual.pop("overshoot_reward_loss") == 0.0
    assert actual == expected
    assert_exact_tree(after_original, torch.get_rng_state())
    assert_exact_tree(original.state_dict(), extended.state_dict())
    assert_exact_tree(original.optim.state_dict(), extended.optim.state_dict())
    assert_exact_tree(original.pi_optim.state_dict(), extended.pi_optim.state_dict())


def test_depth_five_reward_terms_match_weighted_ce_and_masked_gradients():
    torch.manual_seed(77)
    learner = OvershootLearner(tiny_model(), TDMPC2LearnerConfig(episode_length=500, batch_size=2))
    with torch.no_grad():
        learner.model._reward[-1].weight.normal_(std=0.05)
    z2 = torch.randn(2, 32, requires_grad=True)
    a3 = torch.randn(2, 3)
    z3 = learner.model.next(z2, a3)
    extra_action = torch.randn(2, 2, 3, requires_grad=True)
    extra_reward = torch.tensor([[[1.2], [0.0]], [[-2.3], [0.0]]])
    mask = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])
    logits4 = learner.model.reward(z3, extra_action[0])
    z4 = learner.model.next(z3, extra_action[0])
    logits5 = learner.model.reward(z4, extra_action[1])
    expected = (
        learner.cfg.rho ** 3 * (soft_ce(logits4, extra_reward[0], learner.model.cfg) * mask[0]).mean()
        + learner.cfg.rho ** 4 * (soft_ce(logits5, extra_reward[1], learner.model.cfg) * mask[1]).mean()
    ) / 3
    actual = learner._overshoot_reward_loss(z3, extra_action, extra_reward, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    for grad in (learner.model._reward[-1].weight.grad, learner.model._dynamics[-1].weight.grad,
                 extra_action.grad, z2.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    action_grad, z2_grad = extra_action.grad, z2.grad
    assert action_grad is not None and z2_grad is not None
    assert torch.count_nonzero(action_grad[:, 1]) == 0
    assert torch.count_nonzero(z2_grad[1]) == 0
    assert action_grad[1, 0].abs().sum() > 0  # Step 5 backpropagates through z4.
    assert action_grad[1, 1].abs().sum() == 0


def test_active_learner_adds_only_reward_terms_before_one_joint_step():
    torch.manual_seed(49)
    replay = OvershootReplay(10, seed=10)
    episode(replay, 10, 5, ending="finish")
    data = replay.sample(5)
    learner = OvershootLearner(tiny_model(), TDMPC2LearnerConfig(episode_length=500, batch_size=5))
    with patch.object(learner.optim, "step", wraps=learner.optim.step) as joint, patch.object(
        learner.pi_optim, "step", wraps=learner.pi_optim.step
    ) as actor, patch.object(learner.model, "soft_update_target_Q", wraps=learner.model.soft_update_target_Q) as ema:
        metrics = learner.update(data)
    assert joint.call_count == actor.call_count == ema.call_count == 1
    assert metrics["overshoot_reward_loss"] > 0
    assert metrics["reward_loss"] > metrics["overshoot_reward_loss"]
    assert all(np.isfinite(v) for v in metrics.values())
    assert not learner.model.training


@pytest.mark.parametrize("bad", ["shape", "suffix_at_timeout", "suffix_at_finish", "gap", "nonzero_masked", "unmarked_finish"])
def test_invalid_suffix_or_semantic_labels_fail_closed(bad):
    torch.manual_seed(60)
    learner = OvershootLearner(tiny_model(), TDMPC2LearnerConfig(episode_length=500, batch_size=2))
    data = {
        "obs": torch.randint(0, 255, (4, 2, 4, 64, 64), dtype=torch.uint8),
        "action": torch.zeros(3, 2, 3), "reward": torch.zeros(3, 2, 1),
        "terminated": torch.zeros(3, 2, 1), "truncated": torch.zeros(3, 2, 1),
        "terminal": torch.zeros(3, 2, 1), "bootstrap_mask": torch.ones(3, 2, 1),
        "overshoot_action": torch.zeros(2, 2, 3), "overshoot_reward": torch.zeros(2, 2, 1),
        "overshoot_mask": torch.zeros(2, 2, 1),
    }
    if bad == "shape":
        data["overshoot_action"] = data["overshoot_action"][:1]
    elif bad in ("suffix_at_timeout", "suffix_at_finish"):
        data["truncated"][-1] = 1
        if bad == "suffix_at_finish":
            data["terminal"][-1] = 1
            data["bootstrap_mask"][-1] = 0
        data["overshoot_mask"][0] = 1
    elif bad == "gap":
        data["overshoot_mask"][1] = 1
    elif bad == "nonzero_masked":
        data["overshoot_reward"][0] = 1
    else:
        data["terminal"][-1] = 1
        data["bootstrap_mask"][-1] = 0
    with pytest.raises(ValueError):
        learner.update(data)


def test_semantic_finish_does_not_bootstrap_but_timeout_does():
    replay = OvershootReplay(6, seed=7)
    episode(replay, 10, 3, ending="finish")
    episode(replay, 30, 3, ending="timeout")
    batch = replay.sample(60)
    for column, ep_id in enumerate(batch["episode_id"].tolist()):
        assert batch["truncated"][-1, column, 0] == 1
        assert batch["terminal"][-1, column, 0] == float(ep_id == 0)
        assert batch["bootstrap_mask"][-1, column, 0] == float(ep_id == 1)
        assert not batch["overshoot_mask"][:, column].any()
    learner = OvershootLearner(tiny_model(), TDMPC2LearnerConfig(episode_length=500, batch_size=60))
    next_z = torch.zeros(3, 2, 32)
    reward = torch.ones(3, 2, 1)
    terminal = torch.tensor([[[0.], [0.]], [[0.], [0.]], [[1.], [0.]]])
    truncated = torch.tensor([[[0.], [0.]], [[0.], [0.]], [[1.], [1.]]])
    with patch.object(learner.model, "Q", return_value=torch.full_like(reward, 7.)):
        target = learner._td_target(next_z, reward, terminal, truncated)
    assert target[-1, 0, 0].item() == 1.
    assert target[-1, 1, 0].item() == pytest.approx(1 + learner.discount * 7)
