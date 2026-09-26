import copy
import unittest

import numpy as np

from haic.algorithms.rlpd.agent import sample_balanced_batch
from haic.algorithms.rlpd.replay import FrameStackReplay, PixelTransition


def stack(values):
    return np.stack([np.full((84, 84), value, dtype=np.uint8) for value in values])


def state(step, base):
    if step == 0:
        return stack([base] * 4)
    count = min(step, 3)
    first = step - count + 1
    return stack([base] * (4 - count) + [base + value for value in range(first, step + 1)])


def transition(
    *,
    episode=1,
    step_index=0,
    base=10,
    terminated=False,
    truncated=False,
    terminal=None,
    next_observation=None,
    track_id=1,
    geometry_seed=7001,
):
    terminal = terminated if terminal is None else terminal
    if next_observation is None:
        next_observation = state(step_index + 1, base)
    return PixelTransition(
        observation=state(step_index, base),
        proposed_action=np.array([0.25, -0.5, 0.0], dtype=np.float32),
        executed_action=np.array([0.25, -0.5, 0.0], dtype=np.float32),
        applied_action=np.array([0.25, 0.25, 0.5], dtype=np.float32),
        reward=float(step_index),
        next_observation=next_observation,
        terminated=terminated,
        truncated=truncated,
        terminal=terminal,
        episode_id=episode,
        step=step_index,
        track_id=track_id,
        geometry_seed=geometry_seed,
    )


class TestFrameStackReplay(unittest.TestCase):
    def test_stores_one_frame_and_reconstructs_episode_start_padding(self):
        replay = FrameStackReplay(8, seed=0, source="online")
        replay.add(transition(step_index=0, base=10))
        self.assertEqual(replay.valid_count, 0)
        replay.add(transition(step_index=1, base=10))
        self.assertEqual(replay.valid_count, 1)
        observation = replay._observation_stack(0)
        self.assertTrue(np.all(observation == 10))
        next_observation = replay._next_stack(0)
        expected = state(1, 10)
        self.assertTrue(np.array_equal(next_observation, expected))
        self.assertEqual(replay.frames.nbytes, 8 * 84 * 84)

    def test_terminal_and_time_limit_keep_actual_final_stack_without_reset_leak(self):
        replay = FrameStackReplay(8, seed=1, source="offline")
        final_stack = stack([201, 202, 203, 204])
        replay.add(transition(step_index=0, base=10))
        replay.add(transition(
            step_index=1,
            base=10,
            terminated=True,
            next_observation=final_stack,
        ))
        self.assertTrue(np.array_equal(replay._next_stack(1), final_stack))
        self.assertTrue(replay.terminal[1])

        timeout_stack = stack([211, 212, 213, 214])
        replay.add(transition(
            episode=2,
            step_index=0,
            base=50,
            truncated=True,
            terminal=False,
            next_observation=timeout_stack,
            geometry_seed=7002,
        ))
        self.assertFalse(replay.terminal[2])
        self.assertTrue(np.array_equal(replay._next_stack(2), timeout_stack))
        self.assertTrue(np.all(replay._observation_stack(2) == 50))
        self.assertNotIn(50, replay._next_stack(1))

    def test_unfinished_nonterminal_row_is_not_sampleable_until_successor_exists(self):
        replay = FrameStackReplay(4, seed=2, source="online")
        replay.add(transition(step_index=0))
        with self.assertRaises(ValueError):
            replay.sample(1)
        replay.add(transition(step_index=1, terminated=True))
        batch = replay.sample(8)
        self.assertEqual(batch["observation"].shape, (8, 4, 84, 84))
        self.assertTrue(np.isin(batch["indices"], [0, 1]).all())

    def test_ring_overwrite_invalidates_lost_stack_prefix_but_keeps_terminal_boundary(self):
        replay = FrameStackReplay(5, seed=3, source="online")
        for step_index in range(5):
            replay.add(transition(
                episode=3,
                step_index=step_index,
                base=30,
                terminated=step_index == 4,
                next_observation=stack([90, 91, 92, 93]) if step_index == 4 else None,
            ))
        self.assertEqual(replay.valid_count, 5)
        replay.add(transition(episode=4, step_index=0, base=60, geometry_seed=8001))
        self.assertNotIn(1, replay._valid_rows[: replay.valid_count])
        self.assertNotIn(2, replay._valid_rows[: replay.valid_count])
        self.assertNotIn(3, replay._valid_rows[: replay.valid_count])
        self.assertIn(4, replay._valid_rows[: replay.valid_count])
        self.assertTrue(np.array_equal(replay._next_stack(4), stack([90, 91, 92, 93])))

    def test_offline_finalization_is_immutable_and_sampling_reports_source(self):
        offline = FrameStackReplay(4, seed=4, source="offline")
        online = FrameStackReplay(4, seed=5, source="online")
        for replay, episode, base, geometry_seed in (
            (offline, 10, 10, 7100),
            (online, 20, 20, 7200),
        ):
            replay.add(transition(
                episode=episode,
                step_index=0,
                base=base,
                terminated=True,
                next_observation=stack([40, 41, 42, 43]),
                geometry_seed=geometry_seed,
            ))
        offline.finalize()
        with self.assertRaises(RuntimeError):
            offline.add(transition(episode=11))
        batch = sample_balanced_batch(offline, online, batch_size=8, offline_count=4)
        self.assertEqual(batch["source_counts"], {"offline": 4, "online": 4})
        self.assertEqual(int(np.count_nonzero(batch["source"] == "offline")), 4)
        self.assertEqual(batch["action"].shape, (8, 3))
        self.assertTrue(np.all(batch["applied_action"][:, 1:] >= 0.0))

    def test_balanced_sampler_rejects_empty_or_unbalanced_sources(self):
        empty_offline = FrameStackReplay(4, seed=6, source="offline", immutable=True)
        online = FrameStackReplay(4, seed=7, source="online")
        online.add(transition(terminated=True))
        with self.assertRaises(ValueError):
            sample_balanced_batch(empty_offline, online, batch_size=64, offline_count=32)
        with self.assertRaises(ValueError):
            sample_balanced_batch(online, online, batch_size=64, offline_count=0)

    def test_state_restore_preserves_sampler_rng_and_boundary_frames(self):
        original = FrameStackReplay(6, seed=8, source="offline")
        original.add(transition(step_index=0, terminated=True, next_observation=stack([70, 71, 72, 73])))
        state_dict = copy.deepcopy(original.state_dict())
        expected = original.sample(5)
        restored = FrameStackReplay(6, seed=999, source="offline")
        restored.load_state_dict(state_dict)
        actual = restored.sample(5)
        self.assertTrue(np.array_equal(actual["indices"], expected["indices"]))
        self.assertTrue(np.array_equal(actual["observation"], expected["observation"]))
        self.assertTrue(np.array_equal(actual["next_observation"], expected["next_observation"]))
        with self.assertRaises(ValueError):
            FrameStackReplay(7, seed=8, source="offline").load_state_dict(state_dict)

    def test_state_restore_preserves_nontrivial_valid_row_order(self):
        original = FrameStackReplay(6, seed=8, source="online")
        original.add(transition(step_index=0))
        original.add(transition(step_index=1, terminated=True))
        self.assertEqual(original._valid_rows[:original.valid_count].tolist(), [1, 0])
        saved = copy.deepcopy(original.state_dict())

        restored = FrameStackReplay(6, seed=999, source="online")
        restored.load_state_dict(saved)
        self.assertEqual(restored._valid_rows[:restored.valid_count].tolist(), [1, 0])
        expected = original.sample(8)
        actual = restored.sample(8)
        for key in ("indices", "observation", "next_observation", "action"):
            np.testing.assert_array_equal(actual[key], expected[key])
        for replay in (original, restored):
            replay.add(transition(episode=2, step_index=0, terminated=True, geometry_seed=7002))
            rows = replay._valid_rows[:replay.valid_count]
            np.testing.assert_array_equal(replay._valid_positions[rows], np.arange(replay.valid_count))
        np.testing.assert_array_equal(
            restored._valid_rows[:restored.valid_count], original._valid_rows[:original.valid_count]
        )
        np.testing.assert_array_equal(restored.sample(8)["indices"], original.sample(8)["indices"])

    def test_state_restore_rejects_overwritten_four_frame_prefix(self):
        original = FrameStackReplay(5, seed=3, source="online")
        for step_index in range(5):
            original.add(transition(
                episode=3,
                step_index=step_index,
                base=30,
                terminated=step_index == 4,
            ))
        original.add(transition(episode=4, step_index=0, base=60, geometry_seed=8001))
        self.assertEqual(original._valid_rows[:original.valid_count].tolist(), [4])

        restored = FrameStackReplay(5, seed=999, source="online")
        restored.load_state_dict(copy.deepcopy(original.state_dict()))
        self.assertEqual(restored._valid_rows[:restored.valid_count].tolist(), [4])
        self.assertFalse(restored._has_valid_stack_prefix(3))
        np.testing.assert_array_equal(restored.sample(5)["indices"], [4] * 5)

    def test_tiny_ring_never_samples_missing_four_frame_prefix(self):
        replay = FrameStackReplay(3, seed=3, source="online")
        for step_index in range(4):
            replay.add(transition(step_index=step_index, terminated=step_index == 3))
        self.assertEqual(replay.valid_count, 0)
        with self.assertRaises(ValueError):
            replay.sample(1)
        restored = FrameStackReplay(3, source="online")
        restored.load_state_dict(copy.deepcopy(replay.state_dict()))
        self.assertEqual(restored.valid_count, 0)

    def test_wrapped_resume_preserves_batches_through_later_overwrites(self):
        original = FrameStackReplay(5, seed=81, source="online")
        original.add(transition(episode=1, step_index=0, base=10))
        original.add(transition(episode=1, step_index=1, base=10, terminated=True))
        original.add(transition(episode=2, step_index=0, base=40, geometry_seed=7002))
        original.add(transition(episode=2, step_index=1, base=40, geometry_seed=7002))
        original.add(transition(episode=2, step_index=2, base=40, geometry_seed=7002, truncated=True))
        restored = FrameStackReplay(5, seed=999, source="online")
        restored.load_state_dict(copy.deepcopy(original.state_dict()))
        self.assertNotEqual(original._valid_rows[:original.valid_count].tolist(), list(range(5)))

        additions = (
            transition(episode=3, step_index=0, base=80, geometry_seed=7003),
            transition(episode=3, step_index=1, base=80, geometry_seed=7003, terminated=True),
        )
        for next_transition in (None, *additions):
            if next_transition is not None:
                original.add(next_transition)
                restored.add(next_transition)
            expected_rows = original._valid_rows[:original.valid_count]
            np.testing.assert_array_equal(restored._valid_rows[:restored.valid_count], expected_rows)
            for replay in (original, restored):
                np.testing.assert_array_equal(
                    replay._valid_positions[expected_rows], np.arange(replay.valid_count)
                )
                for index in expected_rows:
                    replay._observation_stack(int(index))
                    replay._next_stack(int(index))
            expected, actual = original.sample(16), restored.sample(16)
            for key in ("indices", "observation", "next_observation", "reward", "terminated", "truncated"):
                np.testing.assert_array_equal(actual[key], expected[key])

    def test_state_restore_rejects_missing_or_invalid_sampler_order(self):
        original = FrameStackReplay(6, seed=8, source="online")
        original.add(transition(step_index=0))
        original.add(transition(step_index=1, terminated=True))
        saved = copy.deepcopy(original.state_dict())
        for rows in (None, np.array([0, 0]), np.array([1, 2]), np.array([1])):
            with self.subTest(rows=rows):
                broken = copy.deepcopy(saved)
                if rows is None:
                    broken.pop("valid_rows")
                else:
                    broken["valid_rows"] = rows
                with self.assertRaisesRegex(ValueError, "valid rows"):
                    FrameStackReplay(6, source="online").load_state_dict(broken)

        legacy = copy.deepcopy(saved)
        legacy["format"] = "haic-rlpd-frame-replay-v1"
        with self.assertRaisesRegex(ValueError, "exact resume"):
            FrameStackReplay(6, source="online").load_state_dict(legacy)

    def test_transition_validates_action_coordinates_and_terminal_contract(self):
        with self.assertRaises(ValueError):
            transition().__class__(
                **{
                    **transition().__dict__,
                    "applied_action": np.array([0.0, -0.1, 0.0], dtype=np.float32),
                }
            )
        with self.assertRaises(ValueError):
            transition().__class__(
                **{
                    **transition().__dict__,
                    "terminated": True,
                    "terminal": False,
                }
            )


if __name__ == "__main__":
    unittest.main()
