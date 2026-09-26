from __future__ import annotations

from collections import Counter
from io import BytesIO
import hashlib

import numpy as np
import pytest

from common_adapter import ActionAdapter, ObservationSpec, Transition
from drq_v2 import Uint8Replay
from haic.algorithms.drq_v2.teacher_replay import (
    TeacherDataset,
    TeacherEpisode,
    TwoSourceReplay,
)


SPEC = ObservationSpec()
ACTION_ADAPTER = ActionAdapter()
ACTOR_HASH = hashlib.sha256(b"teacher-test-actor").hexdigest()


def _latest_stack(latest_frames: list[np.ndarray], index: int) -> np.ndarray:
    return np.stack([latest_frames[max(0, index - 3 + channel)] for channel in range(4)])


def _transitions(
    episode_id: int,
    rewards: list[float],
    *,
    ending: str = "terminated",
    frame_base: int = 10,
) -> list[Transition]:
    latest = [
        np.full((84, 84), frame_base + index, dtype=np.uint8)
        for index in range(len(rewards) + 1)
    ]
    rows = []
    for step, reward in enumerate(rewards):
        final = step == len(rewards) - 1
        terminated = final and ending == "terminated"
        truncated = final and ending in {"timeout", "finished-timeout"}
        finished = final and ending in {"finished", "finished-timeout"}
        action = np.asarray([0.1 * (step % 5), -0.5, 0.25], dtype=np.float32)
        rows.append(Transition(
            observation=SPEC.from_uint8(_latest_stack(latest, step)),
            action=action,
            reward=reward,
            next_observation=SPEC.from_uint8(_latest_stack(latest, step + 1)),
            terminated=terminated,
            truncated=truncated,
            terminal=bool(terminated or finished),
            info={"finished": finished, "progress": float(step), "damage": 0.0},
            episode_id=episode_id,
            step=step,
            applied_action=ACTION_ADAPTER.to_official(action, clip=False),
        ))
    return rows


def _episode(
    episode_id: int,
    rewards: list[float],
    *,
    ending: str = "terminated",
    frame_base: int = 10,
    source_id: str = "teacher-source-a",
    geometry_id: str = "road-17",
    complete: bool = True,
) -> TeacherEpisode:
    return TeacherEpisode.from_transitions(
        _transitions(episode_id, rewards, ending=ending, frame_base=frame_base),
        source_id=source_id,
        source_actor_sha256=ACTOR_HASH,
        geometry_id=geometry_id,
        track_id=2,
        complete=complete,
    )


def _dataset(*episodes: TeacherEpisode) -> TeacherDataset:
    dataset = TeacherDataset(episodes)
    dataset.seal()
    return dataset


def _online_replay(count: int, *, seed: int = 3) -> Uint8Replay:
    replay = Uint8Replay(capacity=count + 4, n_step=3, gamma=0.9, seed=seed)
    latest = [np.full((84, 84), 100 + index, dtype=np.uint8) for index in range(count + 1)]
    for step in range(count):
        action = np.asarray([0.0, -0.25, 0.5], dtype=np.float32)
        final = step == count - 1
        replay.add(Transition(
            observation=SPEC.from_uint8(_latest_stack(latest, step)),
            action=action,
            reward=float(step + 1),
            next_observation=SPEC.from_uint8(_latest_stack(latest, step + 1)),
            terminated=final,
            truncated=False,
            terminal=final,
            episode_id=123,
            step=step,
            applied_action=ACTION_ADAPTER.to_official(action),
        ))
    return replay


def test_reset_stack_padding_and_frame_efficient_float_reconstruction():
    episode = _episode(1, [1.0, 2.0, 3.0, 4.0], frame_base=30)
    dataset = _dataset(episode)

    np.testing.assert_array_equal(episode.observation(0), SPEC.from_uint8(_latest_stack(
        [np.full((84, 84), 30 + i, dtype=np.uint8) for i in range(5)], 0
    )))
    assert episode.observation(0).dtype == np.float32
    assert episode.observation(0).shape == (4, 84, 84)
    assert episode.observation(0).min() >= 0.0
    assert episode.observation(0).max() <= 1.0
    assert episode.frames.shape == (5, 84, 84)
    assert dataset.memory_bytes < episode.steps * 4 * 84 * 84


def test_n_step_reward_action_and_next_observation_alignment():
    dataset = _dataset(_episode(1, [1.0, 2.0, 3.0], frame_base=40))
    row = dataset.build_n_step(0, gamma=0.9, n_step=3)
    np.testing.assert_array_equal(row.action, [0.0, -0.5, 0.25])
    np.testing.assert_array_equal(row.applied_action, ACTION_ADAPTER.to_official(row.action))
    assert row.reward == pytest.approx(1.0 + 0.9 * 2.0 + 0.9**2 * 3.0)
    assert row.horizon == 3
    assert row.terminal
    assert row.discount == 0.0
    assert row.progress == 0.0
    assert row.damage == 0.0
    expected_final = np.full((84, 84), 43, dtype=np.uint8)
    np.testing.assert_array_equal(row.next_observation, SPEC.from_uint8(np.stack([
        np.full((84, 84), value, dtype=np.uint8) for value in (40, 41, 42, 43)
    ])))
    np.testing.assert_allclose(row.next_observation[-1], expected_final / 255.0)

    early = dataset.build_n_step(1, gamma=0.9, n_step=3)
    assert early.horizon == 2
    assert early.reward == pytest.approx(2.0 + 0.9 * 3.0)
    np.testing.assert_allclose(early.action, [0.1, -0.5, 0.25])

    ordinary = _dataset(_episode(2, [1.0, 2.0, 3.0, 4.0], frame_base=44))
    ordinary_target = ordinary.build_n_step(0, gamma=0.9, n_step=3)
    assert ordinary_target.horizon == 3
    assert not ordinary_target.terminal
    assert ordinary_target.discount == pytest.approx(0.9**3)


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_terminal_1_2_3_step_endings_align_final_observation(horizon: int):
    rewards = [float(index + 1) for index in range(horizon)]
    episode = _episode(10 + horizon, rewards, frame_base=50)
    dataset = _dataset(episode)
    target = dataset.build_n_step(0, gamma=0.8, n_step=3)

    assert target.horizon == horizon
    assert target.reward == pytest.approx(sum(reward * 0.8**i for i, reward in enumerate(rewards)))
    assert target.terminated
    assert target.discount == 0.0
    assert target.next_observation[-1, 0, 0] == pytest.approx((50 + horizon) / 255.0)


def test_finished_and_truncated_is_terminal_but_pure_timeout_bootstraps():
    finished = _dataset(_episode(20, [5.0], ending="finished-timeout", frame_base=80))
    finished_target = finished.build_n_step(0, gamma=0.9, n_step=3)
    assert finished_target.truncated
    assert finished_target.finished
    assert finished_target.terminal
    assert finished_target.discount == 0.0
    assert finished_target.next_observation[-1, 0, 0] == pytest.approx(81 / 255.0)

    finish_only = _dataset(_episode(21, [3.0], ending="finished", frame_base=82))
    finish_only_target = finish_only.build_n_step(0)
    assert finish_only_target.finished
    assert not finish_only_target.truncated
    assert finish_only_target.terminal
    assert finish_only_target.discount == 0.0

    for horizon in (1, 2, 3):
        dataset = _dataset(_episode(
            30 + horizon,
            [float(index + 1) for index in range(horizon)],
            ending="timeout",
            frame_base=90,
        ))
        target = dataset.build_n_step(0, gamma=0.9, n_step=3)
        assert target.truncated
        assert not target.finished
        assert not target.terminal
        assert target.discount == pytest.approx(0.9**horizon)
        assert target.horizon == horizon
        assert target.next_observation[-1, 0, 0] == pytest.approx((90 + horizon) / 255.0)


def test_transition_factory_audits_executed_native_and_applied_official_actions():
    rows = _transitions(40, [1.0, 2.0])
    rows[-1].info["retire_reason"] = "crash"
    episode = TeacherEpisode.from_transitions(
        rows,
        source_id="source-40",
        source_actor_sha256=ACTOR_HASH,
        geometry_id="geometry-40",
        track_id=1,
    )
    np.testing.assert_array_equal(episode.actions, np.stack([row.action for row in rows]))
    np.testing.assert_array_equal(episode.applied_actions,
                                  np.stack([row.applied_action for row in rows]))
    np.testing.assert_array_equal(episode.progress, [0.0, 1.0])
    np.testing.assert_array_equal(episode.damage, [0.0, 0.0])
    assert episode.retire_reasons == (None, "crash")

    bad_rows = _transitions(41, [1.0])
    bad_rows[0].applied_action = np.asarray([0.0, 0.8, 0.25], dtype=np.float32)
    with pytest.raises(ValueError, match="audit"):
        TeacherEpisode.from_transitions(
            bad_rows,
            source_id="source-41",
            source_actor_sha256=ACTOR_HASH,
            geometry_id="geometry-41",
            track_id=1,
        )

    with pytest.raises(ValueError, match="native actions"):
        TeacherEpisode(
            frames=np.zeros((2, 84, 84), dtype=np.uint8),
            actions=[[1.01, 0.0, 0.0]],
            applied_actions=[[1.01, 0.5, 0.5]],
            rewards=[0.0],
            terminated=[True],
            truncated=[False],
            finished=[False],
            episode_id="bad-action",
            source_id="source",
            source_actor_sha256=ACTOR_HASH,
            geometry_id="geometry",
            track_id=1,
        )


def test_rejects_missing_final_frame_cross_episode_stream_and_partial_seal():
    rows = _transitions(50, [1.0, 2.0])
    rows[-1].next_observation = None
    with pytest.raises(ValueError, match="next/boundary frame"):
        TeacherEpisode.from_transitions(
            rows,
            source_id="source",
            source_actor_sha256=ACTOR_HASH,
            geometry_id="geometry",
            track_id=1,
        )

    mixed = _transitions(51, [1.0]) + _transitions(52, [2.0])
    with pytest.raises(ValueError, match="different episodes"):
        TeacherEpisode.from_transitions(
            mixed,
            source_id="source",
            source_actor_sha256=ACTOR_HASH,
            geometry_id="geometry",
            track_id=1,
        )

    partial = _episode(53, [1.0, 2.0], ending="none", complete=False)
    dataset = TeacherDataset([partial])
    with pytest.raises(ValueError, match="incomplete"):
        dataset.seal()
    incomplete_terminal = TeacherDataset([
        _episode(54, [1.0], ending="terminated", complete=False)
    ])
    with pytest.raises(ValueError, match="incomplete"):
        incomplete_terminal.seal()


def test_n_step_rows_never_cross_episode_or_mix_geometry_tags():
    first = _episode(60, [1.0], ending="timeout", frame_base=100, geometry_id="road-a")
    second = _episode(61, [1000.0, 2000.0], frame_base=120, geometry_id="road-b")
    dataset = _dataset(first, second)
    target = dataset.build_n_step(0, gamma=0.9, n_step=3)

    assert target.horizon == 1
    assert target.reward == 1.0
    assert target.episode_id == "60"
    assert target.geometry_id == "road-a"
    assert target.discount == pytest.approx(0.9)
    assert len(dataset.valid_indices(gamma=0.9, n_step=3)) == 3


def test_dataset_digest_round_trip_tamper_detection_and_post_seal_immutability():
    episode = _episode(70, [1.0, 2.0, 3.0])
    dataset = _dataset(episode)
    digest = dataset.digest
    serialized = dataset.to_bytes()
    restored = TeacherDataset.from_bytes(serialized, expected_digest=digest)
    assert restored.digest == digest
    assert restored.sealed
    np.testing.assert_array_equal(restored.episodes[0].frames, episode.frames)
    np.testing.assert_array_equal(restored.episodes[0].progress, episode.progress)
    np.testing.assert_array_equal(restored.episodes[0].damage, episode.damage)
    assert restored.episodes[0].retire_reasons == episode.retire_reasons
    np.testing.assert_array_equal(restored.episodes[0].observation(0), episode.observation(0))
    np.testing.assert_array_equal(restored.sample_at_indices([0])['action'],
                                  dataset.sample_at_indices([0])['action'])

    with np.load(BytesIO(serialized), allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["rewards"][0] += np.float32(0.25)
    damaged = BytesIO()
    np.savez_compressed(damaged, **arrays)
    with pytest.raises(ValueError, match="digest mismatch"):
        TeacherDataset.from_bytes(damaged.getvalue())
    with pytest.raises(ValueError, match="expected digest"):
        TeacherDataset.from_bytes(serialized, expected_digest="0" * 64)

    with pytest.raises(RuntimeError, match="immutable"):
        dataset.add_episode(_episode(71, [1.0]))
    with pytest.raises(ValueError):
        episode.frames[0, 0, 0] = 1
    with pytest.raises(ValueError):
        episode.frames.setflags(write=True)
    with pytest.raises(AttributeError, match="immutable"):
        episode.geometry_id = "mutated"
    with pytest.raises(TypeError):
        episode.metadata["retire_reason"] = "mutated"


def test_dataset_rejects_nonfinite_rewards_and_out_of_range_observations():
    rows = _transitions(80, [1.0])
    rows[0].reward = float("nan")
    with pytest.raises(ValueError, match="rewards"):
        TeacherEpisode.from_transitions(
            rows,
            source_id="source",
            source_actor_sha256=ACTOR_HASH,
            geometry_id="geometry",
            track_id=1,
        )

    rows = _transitions(81, [1.0])
    rows[0].observation[0, 0, 0] = 1.5
    with pytest.raises(ValueError, match="invalid current observation"):
        TeacherEpisode.from_transitions(
            rows,
            source_id="source",
            source_actor_sha256=ACTOR_HASH,
            geometry_id="geometry",
            track_id=1,
        )


def test_two_source_sampler_uses_exact_quotas_and_preserves_source_provenance():
    online = _online_replay(70)
    teacher = _dataset(_episode(90, [float(i + 1) for i in range(24)], frame_base=140))
    first = TwoSourceReplay(online, teacher, mode="teacher-replay", seed=19)
    second = TwoSourceReplay(online, teacher, mode="teacher-replay", seed=19)

    batch = first.sample()
    repeat = second.sample()
    assert Counter(batch["sources"]) == {"online": 48, "teacher": 16}
    assert Counter(batch["source"]) == {0: 48, 1: 16}
    assert len(batch["source_indices"]) == 64
    assert len(batch["source_tags"]) == 64
    np.testing.assert_array_equal(batch["observation"], repeat["observation"])
    np.testing.assert_array_equal(batch["source_indices"], repeat["source_indices"])
    assert batch["source_tags"] == repeat["source_tags"]
    for source, tag, index in zip(batch["sources"], batch["source_tags"],
                                  batch["source_indices"], strict=True):
        assert tag["source"] == source
        assert tag["source_index"] == int(index)
        if source == "teacher":
            assert tag["geometry_id"] == "road-17"
            assert tag["source_actor_sha256"] == ACTOR_HASH

    online_only = TwoSourceReplay(online, None, mode="online-only", seed=19).sample()
    assert Counter(online_only["sources"]) == {"online": 64}
    assert Counter(online_only["source"]) == {0: 64}
    assert len(online_only["source_tags"]) == 64


def test_two_source_sampler_fails_closed_on_undersized_valid_pools():
    teacher = _dataset(_episode(100, [1.0] * 15, frame_base=160))
    with pytest.raises(ValueError, match="online pool"):
        TwoSourceReplay(_online_replay(47), teacher, mode="teacher-replay", seed=0).sample()

    with pytest.raises(ValueError, match="teacher pool"):
        TwoSourceReplay(_online_replay(64), teacher, mode="teacher-replay", seed=0).sample()

    with pytest.raises(ValueError, match="online pool"):
        TwoSourceReplay(_online_replay(47), None, mode="online-only", seed=0).sample()
