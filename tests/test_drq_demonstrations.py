import copy
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from common_adapter import ObservationSpec, Transition
from drq_v2 import DrQv2Agent, DrQv2Config, Uint8Replay
from training.drq_demonstrations import (
    DRQ_DEMONSTRATION_FORMAT,
    collect_drq_teacher_demonstrations,
    load_drq_demonstrations,
    save_drq_demonstrations,
)


def _observation(value: int) -> np.ndarray:
    return np.full((4, 84, 84), value, dtype=np.uint8)


def _transition(episode: int, step: int, action_value: float) -> Transition:
    return Transition(
        observation=_observation(step),
        action=np.full(3, action_value, dtype=np.float32),
        reward=float(step + 1),
        next_observation=_observation(step + 1),
        terminated=False,
        truncated=False,
        episode_id=episode,
        step=step,
    )


class _Teacher:
    def __init__(self):
        self.reset_calls = 0

    def reset(self, observation):
        self.reset_calls += 1

    def act(self, observation):
        return np.array([0.2, 0.12, 0.28], dtype=np.float32)


class _Environment:
    def __init__(self):
        self.actions = []
        self.steps = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        return np.zeros((4, 84, 84), dtype=np.float32), {"track_id": 1, "seed": 7}

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.steps += 1
        return (
            np.full((4, 84, 84), self.steps / 10, dtype=np.float32),
            float(self.steps),
            self.steps == 2,
            False,
            {"retire_reason": "crash" if self.steps == 2 else None},
        )

    def close(self):
        self.closed = True


def test_teacher_collection_preserves_official_action_and_stores_native_transition(tmp_path):
    environments = []

    def environment_factory(episode, *, max_decisions):
        environment = _Environment()
        environments.append(environment)
        return environment

    replay, metadata = collect_drq_teacher_demonstrations(
        [(1, 7)],
        max_decisions=3,
        n_step=1,
        teacher_factory=_Teacher,
        environment_factory=environment_factory,
    )
    expected_official = np.array([0.2, 0.12, 0.28], dtype=np.float32)
    np.testing.assert_allclose(environments[0].actions, [expected_official, expected_official])
    sampled = replay.sample(1, indices=[0])
    np.testing.assert_allclose(sampled["action"][0], [0.2, -0.76, -0.44])
    assert int(sampled["terminal"][0]) == 0
    terminal = replay.sample(1, indices=[1])
    assert int(terminal["terminal"][0]) == 1
    assert float(terminal["discount"][0]) == 0.0
    assert environments[0].closed
    assert metadata["source_action_space"] == "official-[steer,gas,brake]"
    assert metadata["stored_action_space"] == "symmetric-native-3d"

    artifact = save_drq_demonstrations(tmp_path / "demonstrations.pt", replay, metadata=metadata)
    restored, restored_metadata = load_drq_demonstrations(artifact)
    assert restored_metadata == metadata
    expected, actual = replay.sample(1), restored.sample(1)
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    assert payload["format"] == DRQ_DEMONSTRATION_FORMAT


def test_agent_mixes_a_fixed_demo_quota_without_mutating_permanent_replay(tmp_path):
    config = DrQv2Config(
        replay_capacity=8,
        batch_size=4,
        warmup_steps=0,
        n_step=1,
        feature_dim=8,
        hidden_dim=8,
    )
    agent = DrQv2Agent(config, seed=5)
    demonstrations = Uint8Replay(capacity=8, n_step=1, gamma=config.gamma, seed=17)
    for step in range(6):
        agent.observe(_transition(0, step, -0.5))
        demonstrations.add(_transition(1, step, 0.5))
    agent.configure_demonstrations(
        demonstrations,
        batch_size=2,
        lineage={"artifact_sha256": "demo-hash"},
    )
    frozen_demo = copy.deepcopy(demonstrations.state_dict())
    environment_steps = agent.environment_steps
    with (
        mock.patch.object(agent.replay, "sample", wraps=agent.replay.sample) as online_sample,
        mock.patch.object(demonstrations, "sample", wraps=demonstrations.sample) as demo_sample,
    ):
        batch = agent._sample_training_batch()
    online_sample.assert_called_once_with(2)
    demo_sample.assert_called_once_with(2)
    np.testing.assert_array_equal(batch["is_demonstration"], [False, False, True, True])
    np.testing.assert_allclose(batch["action"][:2], -0.5)
    np.testing.assert_allclose(batch["action"][2:], 0.5)

    for step in range(6, 20):
        agent.observe(_transition(0, step, -0.25))
    assert agent.environment_steps == environment_steps + 14
    assert agent.replay.size == 8
    current_demo = demonstrations.state_dict()
    for key in (
        "next_sequence", "size", "frames", "actions", "rewards", "terminated",
        "truncated", "terminal", "episode_ids", "episode_steps", "sequence_ids",
    ):
        if isinstance(frozen_demo[key], np.ndarray):
            np.testing.assert_array_equal(frozen_demo[key], current_demo[key])
        else:
            assert frozen_demo[key] == current_demo[key]

    checkpoint = agent.save_checkpoint(tmp_path / "checkpoint.pt")
    restored = DrQv2Agent(config, seed=99)
    restored.load_checkpoint(checkpoint)
    assert restored.demonstration_batch_size == 2
    assert restored.demonstration_lineage == {"artifact_sha256": "demo-hash"}
    assert restored.demonstration_replay_size == demonstrations.size
    restored_state = restored.demonstration_replay.state_dict()
    for key in (
        "frames", "actions", "rewards", "terminated", "truncated", "terminal",
        "episode_ids", "episode_steps", "sequence_ids",
    ):
        np.testing.assert_array_equal(restored_state[key], current_demo[key])
    expected_sample = agent.demonstration_replay.sample(2)
    actual_sample = restored.demonstration_replay.sample(2)
    for key in expected_sample:
        np.testing.assert_array_equal(expected_sample[key], actual_sample[key])
    actor_path = agent.export_actor(tmp_path / "actor.pt")
    actor_payload = torch.load(actor_path, map_location="cpu", weights_only=False)
    assert "demonstrations" not in actor_payload
    assert "vision_corridor" not in repr(actor_payload)


def test_online_only_agent_keeps_original_batch_path():
    config = DrQv2Config(
        replay_capacity=8,
        batch_size=2,
        warmup_steps=0,
        n_step=1,
        feature_dim=8,
        hidden_dim=8,
    )
    agent = DrQv2Agent(config, seed=2)
    for step in range(4):
        agent.observe(_transition(0, step, -0.1))
    batch = agent._sample_training_batch()
    assert batch["observation"].shape[0] == config.batch_size
    assert not batch["is_demonstration"].any()
    metrics = agent.update()
    assert metrics["demonstration_replay_size"] == 0.0
    assert metrics["demonstration_batch_size"] == 0.0
