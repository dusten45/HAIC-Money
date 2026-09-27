"""Synthetic contracts for TD-MPC2 pixel replay; no environment interaction."""

from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from haic.algorithms.tdmpc2.replay import EpisodeReplay, random_shift


def image(value: int) -> np.ndarray:
    return np.full((4, 8, 8), value, dtype=np.uint8)


def step(replay: EpisodeReplay, value: int, *, terminated: bool = False, truncated: bool = False, terminal=None) -> None:
    replay.add_step(
        image(value), np.array([0.25, -0.5, 1.0], dtype=np.float32), float(value),
        terminated=terminated, truncated=truncated, terminal=terminal,
    )


def test_full_horizon_uses_initial_and_actual_final_observation_without_dummy_targets():
    replay = EpisodeReplay(capacity=12, horizon=3, seed=7)
    assert replay.start_episode(image(10)) == 0
    step(replay, 11)
    step(replay, 12)
    assert replay.eligible_windows() == 0
    step(replay, 13, truncated=True)
    batch = replay.sample(2)
    assert batch["obs"].shape == (4, 2, 4, 8, 8)
    assert batch["obs"].dtype == torch.uint8
    assert batch["action"].shape == (3, 2, 3)
    assert batch["reward"].shape == (3, 2, 1)
    assert batch["bootstrap_mask"].shape == (3, 2, 1)
    assert batch["obs"][:, 0, 0, 0, 0].tolist() == [10, 11, 12, 13]
    assert batch["reward"][:, 0, 0].tolist() == [11, 12, 13]
    assert batch["terminated"][:, 0, 0].tolist() == [0, 0, 0]
    assert batch["truncated"][:, 0, 0].tolist() == [0, 0, 1]
    assert batch["bootstrap_mask"][:, 0, 0].tolist() == [1, 1, 1]


def test_uniform_windows_never_cross_resets_or_include_short_episode():
    replay = EpisodeReplay(capacity=30, horizon=2, seed=14)
    replay.start_episode(image(1))
    step(replay, 2, terminated=True)  # Too short for H=2.
    replay.start_episode(image(10))
    step(replay, 11)
    step(replay, 12)
    step(replay, 13, truncated=True)
    replay.start_episode(image(20))
    step(replay, 21)
    step(replay, 22, terminated=True)
    assert replay.eligible_windows() == 3
    batch = replay.sample(128)
    assert set(batch["episode_id"].tolist()) == {1, 2}
    for column, episode_id in enumerate(batch["episode_id"].tolist()):
        start = int(batch["start_step"][column])
        base = 10 if episode_id == 1 else 20
        assert batch["obs"][:, column, 0, 0, 0].tolist() == list(range(base + start, base + start + 3))
        assert batch["reward"][:, column, 0].tolist() == [base + start + 1, base + start + 2]


def test_partial_episode_is_opt_in_and_explicitly_discarded_at_budget_cut():
    replay = EpisodeReplay(capacity=6, horizon=2)
    replay.start_episode(image(10))
    step(replay, 11)
    step(replay, 12)
    assert len(replay) == 0
    assert replay.active_length == 2
    assert replay.eligible_windows() == 0
    assert replay.eligible_windows(include_partial=True) == 1
    with pytest.raises(ValueError, match="eligible"):
        replay.sample(1)
    partial = replay.sample(1, include_partial=True)
    assert partial["terminal"].sum() == 0
    assert partial["bootstrap_mask"].tolist() == [[[1.0]], [[1.0]]]
    with pytest.raises(RuntimeError, match="previous episode"):
        replay.start_episode(image(30))
    assert replay.discard_partial() == 2
    assert replay.eligible_windows(include_partial=True) == 0
    assert replay.start_episode(image(30)) == 1
    step(replay, 31, truncated=True)
    assert len(replay) == 1


def test_explicit_partial_sampling_mode_reuses_only_full_live_windows():
    replay = EpisodeReplay(capacity=4, horizon=2, include_partial=True)
    replay.start_episode(image(0))
    step(replay, 1)
    with pytest.raises(ValueError, match="eligible"):
        replay.sample(1)
    step(replay, 2)
    assert replay.eligible_windows() == 1
    assert replay.sample(1)["obs"][:, 0, 0, 0, 0].tolist() == [0, 1, 2]
    step(replay, 3, terminated=True)
    assert replay.active_length == 0
    assert replay.eligible_windows() == 2
    with pytest.raises(RuntimeError, match="start_episode"):
        step(replay, 4)


def test_bootstrap_distinguishes_raw_termination_timeout_and_haic_finish():
    replay = EpisodeReplay(capacity=3, horizon=1, seed=5)
    for initial, finish, termination in [(10, False, True), (20, False, False), (30, True, False)]:
        replay.start_episode(image(initial))
        step(
            replay, initial + 1, terminated=termination, truncated=not termination,
            terminal=(termination or finish),
        )
    batch = replay.sample(128)
    assert set(batch["episode_id"].tolist()) == {0, 1, 2}
    for column, episode_id in enumerate(batch["episode_id"].tolist()):
        flag = lambda name: float(batch[name][0, column, 0])
        if episode_id == 0:
            assert (flag("terminated"), flag("truncated"), flag("terminal"), flag("bootstrap_mask")) == (1, 0, 1, 0)
        elif episode_id == 1:
            assert (flag("terminated"), flag("truncated"), flag("terminal"), flag("bootstrap_mask")) == (0, 1, 0, 1)
        else:
            assert (flag("terminated"), flag("truncated"), flag("terminal"), flag("bootstrap_mask")) == (0, 1, 1, 0)


def test_caller_can_disable_truncation_bootstrap_without_relabeling_flags():
    replay = EpisodeReplay(capacity=2, horizon=1, bootstrap_on_truncation=False)
    replay.start_episode(image(10))
    step(replay, 11, truncated=True)
    batch = replay.sample(1)
    assert batch["terminated"].item() == 0
    assert batch["truncated"].item() == 1
    assert batch["terminal"].item() == 0
    assert batch["bootstrap_mask"].item() == 0


def test_eviction_trims_oldest_episode_without_inventing_new_reset():
    replay = EpisodeReplay(capacity=3, horizon=2)
    replay.start_episode(image(10))
    step(replay, 11)
    step(replay, 12, truncated=True)
    replay.start_episode(image(20))
    step(replay, 21)
    step(replay, 22, truncated=True)
    assert len(replay) == 3
    assert replay.num_episodes == 2
    assert replay.eligible_windows() == 1  # Old episode has only one surviving step.
    batch = replay.sample(12)
    assert batch["episode_id"].tolist() == [1] * 12
    assert batch["obs"][:, 0, 0, 0, 0].tolist() == [20, 21, 22]
    replay.start_episode(image(30))
    step(replay, 31)
    step(replay, 32, terminated=True)
    assert len(replay) == 3
    assert replay.eligible_windows() == 1
    assert replay.sample(1)["obs"][:, 0, 0, 0, 0].tolist() == [30, 31, 32]


def test_live_long_episode_retains_suffix_and_true_step_index():
    replay = EpisodeReplay(capacity=3, horizon=2, include_partial=True)
    replay.start_episode(image(1))
    for value in (2, 3, 4, 5, 6):
        step(replay, value)
    assert replay.active_length == 3
    assert replay.eligible_windows() == 2
    batch = replay.sample(32)
    assert set(batch["start_step"].tolist()) == {2, 3}
    for column, start in enumerate(batch["start_step"].tolist()):
        assert batch["obs"][:, column, 0, 0, 0].tolist() == [start + 1, start + 2, start + 3]


def test_validation_is_atomic_and_float_observations_are_quantized_on_ingest():
    replay = EpisodeReplay(capacity=3, horizon=1)
    input_obs = np.full((4, 8, 8), 0.5, dtype=np.float32)
    replay.start_episode(input_obs)
    input_obs[:] = 0
    with pytest.raises(ValueError, match="action"):
        replay.add_step(image(1), np.array([2, 0, 0], dtype=np.float32), 1.0)
    with pytest.raises(ValueError, match="reward"):
        replay.add_step(image(1), np.zeros(3, np.float32), float("nan"))
    with pytest.raises(ValueError, match="boundary"):
        replay.add_step(image(1), np.zeros(3, np.float32), 1.0, terminal=True)
    with pytest.raises(ValueError, match="include terminations"):
        replay.add_step(image(1), np.zeros(3, np.float32), 1.0, terminated=True, terminal=False)
    assert replay.active_length == 0
    action = np.array([0.25, -0.5, 1.0], dtype=np.float32)
    replay.add_step(image(1), action, 1.0, truncated=True)
    action[:] = 0
    assert replay.sample(1)["obs"][0, 0, 0, 0, 0] == 128
    assert replay.sample(1)["action"][0, 0].tolist() == [0.25, -0.5, 1.0]


def test_invalid_observations_and_constructor_are_rejected():
    with pytest.raises(ValueError, match="horizon"):
        EpisodeReplay(capacity=2, horizon=3)
    with pytest.raises(ValueError, match="capacity"):
        EpisodeReplay(capacity=0, horizon=1)
    replay = EpisodeReplay(capacity=3, horizon=1)
    with pytest.raises(ValueError, match="observation"):
        replay.start_episode(np.zeros((4, 8, 9), np.uint8))
    with pytest.raises(ValueError, match="observation"):
        replay.start_episode(np.ones((4, 8, 8), np.float64))
    replay.start_episode(image(0))
    with pytest.raises(ValueError, match="shape"):
        replay.add_step(image(1)[:3], np.zeros(3, np.float32), 1.0)
    assert replay.active_length == 0


def test_sampling_seed_is_reproducible_and_no_implicit_training_gate():
    first = EpisodeReplay(capacity=7, horizon=2, seed=89)
    second = EpisodeReplay(capacity=7, horizon=2, seed=89)
    for replay in (first, second):
        replay.start_episode(image(0))
        for value in (1, 2, 3, 4):
            step(replay, value, truncated=value == 4)
    a = first.sample(40)
    b = second.sample(40)
    for key in ("obs", "action", "episode_id", "start_step"):
        torch.testing.assert_close(a[key], b[key])


def test_shift_matches_upstream_grid_geometry_and_shares_channel_offset():
    rows = torch.arange(8, dtype=torch.float32)[:, None] * 13
    columns = torch.arange(8, dtype=torch.float32)[None, :]
    pattern = rows + columns
    obs = torch.stack((pattern, pattern + 100), dim=0)[None, None].repeat(3, 2, 1, 1, 1)
    generator = torch.Generator().manual_seed(11)
    shifted = random_shift(obs, pad=3, generator=generator)
    expected_offsets = torch.randint(0, 7, (6, 1, 1, 2), generator=torch.Generator().manual_seed(11))
    assert shifted.shape == obs.shape
    assert shifted.dtype == torch.float32
    assert len({tuple(pair) for pair in expected_offsets[:, 0, 0].tolist()}) > 1
    for time in range(3):
        for column in range(2):
            dx, dy = expected_offsets[2 * time + column, 0, 0].tolist()
            reference = F.pad(pattern[None, None], (3, 3, 3, 3), mode="replicate")
            reference = reference[0, 0, dy : dy + 8, dx : dx + 8]
            torch.testing.assert_close(shifted[time, column, 0], reference, atol=1e-4, rtol=0)
            torch.testing.assert_close(shifted[time, column, 1] - shifted[time, column, 0],
                                       torch.full_like(reference, 100), atol=1e-4, rtol=0)


def test_augmentation_is_opt_in_and_keeps_pixel_scale_for_encoder():
    replay = EpisodeReplay(capacity=2, horizon=1)
    replay.start_episode(image(70))
    step(replay, 80, terminated=True)
    raw = replay.sample(1)
    augmented = replay.sample(1, augment=True, generator=torch.Generator().manual_seed(4))
    assert raw["obs"].dtype == torch.uint8
    assert augmented["obs"].dtype == torch.float32
    torch.testing.assert_close(augmented["obs"][:, 0, 0, 0, 0], torch.tensor([70.0, 80.0]))
    assert random_shift(torch.from_numpy(image(100))[None], pad=0).dtype == torch.float32
    with pytest.raises(ValueError, match="shape"):
        random_shift(torch.zeros(2, 4, 8, 9))


def test_completed_episode_checkpoint_restores_sample_order_and_is_independent():
    replay = EpisodeReplay(capacity=5, horizon=2, seed=51)
    replay.start_episode(image(10))
    step(replay, 11)
    step(replay, 12, truncated=True, terminal=True)
    replay.start_episode(image(20))
    step(replay, 21)
    step(replay, 22, terminated=True)
    replay.sample(13)  # Advance sampler before checkpointing.
    state = replay.state_dict()
    restored = EpisodeReplay(capacity=5, horizon=2, seed=999)
    restored.load_state_dict(state)
    original_batch, restored_batch = replay.sample(40), restored.sample(40)
    for name in original_batch:
        torch.testing.assert_close(original_batch[name], restored_batch[name])
    assert restored_batch["episode_id"].tolist() == original_batch["episode_id"].tolist()
    state["episodes"][0]["observations"][:] = 255
    state["episodes"][0]["terminal"][:] = False
    assert replay.state_dict()["episodes"][0]["observations"][0, 0, 0, 0] == 10
    assert restored.state_dict()["episodes"][0]["terminal"][-1]


def test_live_episode_checkpoint_restores_trim_offsets_and_boundary_transition():
    replay = EpisodeReplay(capacity=3, horizon=2, include_partial=True)
    replay.start_episode(image(1))
    for value in (2, 3, 4, 5):
        step(replay, value)
    snapshot = replay.state_dict()
    restored = EpisodeReplay(capacity=3, horizon=2, include_partial=True, seed=20)
    restored.load_state_dict(snapshot)
    assert restored.active_length == 3
    assert restored.eligible_windows() == 2
    assert restored.state_dict()["active"]["start_step"] == 1
    torch.testing.assert_close(replay.sample(8)["obs"], restored.sample(8)["obs"])
    step(replay, 6, truncated=True)
    step(restored, 6, truncated=True)
    assert restored.active_length == 0
    assert restored.state_dict()["episodes"][0]["start_step"] == 2
    original_batch, restored_batch = replay.sample(8), restored.sample(8)
    for name in original_batch:
        torch.testing.assert_close(original_batch[name], restored_batch[name])


def test_corrupt_checkpoint_is_rejected_without_mutating_live_replay():
    replay = EpisodeReplay(capacity=2, horizon=1)
    replay.start_episode(image(10))
    step(replay, 11, truncated=True)
    valid = replay.state_dict()
    invalid = deepcopy(valid)
    invalid["episodes"][0]["truncated"][-1] = False
    with pytest.raises(ValueError, match="final boundary"):
        replay.load_state_dict(invalid)
    invalid = deepcopy(valid)
    invalid["rng_state"] = {"bit_generator": "Invalid"}
    with pytest.raises(ValueError, match="RNG"):
        replay.load_state_dict(invalid)
    invalid = deepcopy(valid)
    invalid["episodes"][0]["actions"][0, 0] = 2.0
    with pytest.raises(ValueError, match="actions"):
        replay.load_state_dict(invalid)
    assert replay.sample(1)["obs"][:, 0, 0, 0, 0].tolist() == [10, 11]
    wrong_horizon = EpisodeReplay(capacity=2, horizon=2)
    with pytest.raises(ValueError, match="horizon"):
        wrong_horizon.load_state_dict(valid)
