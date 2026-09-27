from __future__ import annotations

import numpy as np
import pytest
import torch

from common_adapter import Transition
from drq_v2 import DrQv2Agent, DrQv2Config, Uint8Replay
from haic.algorithms.drq_v2.retention import RetentionReplaySampler, update_retained
from haic.algorithms.drq_v2.teacher_study import RNGStreams
from scripts import train_drq_retention_r7 as r7


def filled_replay(seed: int, length: int = 96) -> Uint8Replay:
    replay = Uint8Replay(capacity=128, n_step=3, gamma=.99, seed=seed)
    for sequence in range(length):
        episode, step = divmod(sequence, 12)
        obs = np.full((4, 84, 84), (sequence + seed) % 255, dtype=np.uint8)
        following = np.full((4, 84, 84), (sequence + seed + 1) % 255, dtype=np.uint8)
        replay.add(Transition(
            observation=obs, next_observation=following,
            action=np.asarray([.1, -.5, .25], dtype=np.float32), reward=float(step),
            terminated=step == 11, truncated=False, terminal=step == 11,
            info={}, episode_id=episode, step=step,
            applied_action=np.asarray([0, 0, 0], dtype=np.float32),
        ))
    return replay


def test_exact_32_to_32_batches_with_immutable_source_and_deterministic_rng():
    source = filled_replay(3)
    online = filled_replay(9)
    for key in r7.SOURCE_FIELDS:
        getattr(source, key).flags.writeable = False
    before = source.actions.copy()
    a = RetentionReplaySampler(online, source, seed=21)
    b = RetentionReplaySampler(online, source, seed=21)
    for _ in range(3):
        batch_a, batch_b = a.sample(), b.sample()
        for key in ("source", "source_indices", "episode_id", "observation", "action"):
            np.testing.assert_array_equal(batch_a[key], batch_b[key])
        assert sum(batch_a["source"] == 1) == sum(batch_a["source"] == 0) == 32
        assert batch_a["observation"].dtype == np.float32
        assert np.all(batch_a["observation"] >= 0) and np.all(batch_a["observation"] <= 1)
    with pytest.raises(ValueError, match="batch size"):
        a.sample(63)
    np.testing.assert_array_equal(source.actions, before)
    with pytest.raises(ValueError):
        source.actions[0, 0] = 1


def test_actor_preservation_updates_student_but_not_teacher():
    torch.set_num_threads(1)
    cfg = DrQv2Config(feature_dim=8, hidden_dim=8, batch_size=64,
                      replay_capacity=128, warmup_steps=0, device="cpu")
    agent = DrQv2Agent(cfg, seed=4)
    agent.replay = filled_replay(9)
    source = filled_replay(3)
    sampler = RetentionReplaySampler(agent.replay, source, seed=20)
    teacher = type(agent.actor)(feature_dim=8, hidden_dim=8)
    teacher.load_state_dict(agent.actor.state_dict())
    teacher.eval().requires_grad_(False)
    original = {key: value.clone() for key, value in teacher.state_dict().items()}
    rng = RNGStreams(13, "cpu")
    batch = sampler.sample()
    result_a = update_retained(agent, batch, rng, teacher=None, lambda_preserve=0.0)
    with torch.no_grad():
        agent.actor.policy.bias[0].add_(.1)
    result_b = update_retained(agent, sampler.sample(), rng, teacher=teacher, lambda_preserve=1.0)
    assert result_a["sampled_source_count"] == result_b["sampled_source_count"] == 32
    assert result_a["sampled_online_count"] == result_b["sampled_online_count"] == 32
    assert result_a["actor_updated"] == 0 and result_b["actor_updated"] == 1
    assert result_a["preservation_loss"] == 0
    assert result_b["preservation_loss"] > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    bad_batch = dict(batch, source=np.zeros(64, dtype=np.uint8))
    with pytest.raises(ValueError, match="precisely 32"):
        update_retained(agent, bad_batch, rng, teacher=None, lambda_preserve=0.0)
    with pytest.raises(ValueError, match="only for positive"):
        update_retained(agent, batch, rng, teacher=None, lambda_preserve=1.0)


def test_drift_snapshot_identity_is_zero_and_reports_both_critics():
    torch.set_num_threads(1)
    agent = DrQv2Agent(DrQv2Config(feature_dim=8, hidden_dim=8, batch_size=64), seed=3)
    teacher = agent.actor
    pixels = np.zeros((4, 4, 84, 84), dtype=np.uint8)
    result = r7.drift_snapshot(agent, teacher, agent.critic_one, pixels,
                               {"source_seed": 0, "family": np.array(["easy"] * 4)}, update=0)
    assert result["count"] == 4
    assert result["steering_abs_mean"] == result["brake_abs_mean"] == 0
    assert result["any_axis_ge_0_10_fraction"] == 0
    assert result["actor_feature_cosine_mean"] == pytest.approx(1.0, abs=1e-6)
    assert result["current_q_source_action_mean"] == result["frozen_q_source_action_mean"]


def test_float32_parity_cache_keeps_source_success_roads_separate(tmp_path):
    path = tmp_path / "cache.npz"
    samples = np.zeros((2, 20, 4, 84, 84), dtype=np.float32)
    np.savez(path, observations=samples, source_seed=np.array([0, 1], dtype=np.int8),
             geometry_seed=np.array([123, 456], dtype=np.int64))
    pixels, metadata = r7._read_diagnostic_cache(path, 0, {123: "anchor", 456: "reversal"})
    assert pixels.shape == (20, 4, 84, 84)
    assert set(metadata["geometry_seed"]) == {123}
    assert set(metadata["family"]) == {"anchor"}
    with pytest.raises(ValueError, match="outside frozen"):
        r7._read_diagnostic_cache(path, 1, {123: "anchor"})
