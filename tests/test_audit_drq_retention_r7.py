"""Synthetic replay and trace failure controls; no simulator or frozen run is opened."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import audit_drq_retention_r7 as audit


class ReplayValidityTests(unittest.TestCase):
    def replay(self):
        cap = 100000
        sequence = np.full(cap, -1, dtype=np.int64)
        episode = np.full(cap, -1, dtype=np.int64)
        steps = np.full(cap, -1, dtype=np.int64)
        sequence[:5] = np.arange(5)
        episode[:5] = [0, 0, 0, 1, 1]
        steps[:5] = [0, 1, 2, 0, 1]
        terminal = np.zeros(cap, dtype=bool)
        terminal[2] = True
        return {"capacity": cap, "action_dim": 3, "n_step": 3, "gamma": .99,
                "next_sequence": 5, "size": 5, "sequence_ids": sequence,
                "episode_ids": episode, "episode_steps": steps, "terminal": terminal,
                "terminated": np.zeros(cap, dtype=bool), "truncated": np.zeros(cap, dtype=bool),
                "boundary_observations": {2: np.empty((4, 84, 84), dtype=np.uint8)}}

    def test_terminal_short_horizon_and_next_stack(self):
        valid, latest = audit.valid_starts(self.replay(), first=0, last=5)
        np.testing.assert_array_equal(valid, [True, True, True, False, False])
        np.testing.assert_array_equal(latest[:3], [2, 2, 2])

    def test_missing_boundary_invalidates_terminal_start(self):
        state = self.replay()
        state["boundary_observations"] = {}
        valid, _ = audit.valid_starts(state, first=0, last=5)
        self.assertFalse(valid[:3].any())

    def test_rolling_replay_missing_observation_history(self):
        state = self.replay()
        state["size"] = 3
        valid, _ = audit.valid_starts(state, first=2, last=5)
        self.assertFalse(valid[0])


class SampleChecksTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.object(audit, "UPDATES", 2)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.tags = np.tile(np.array([1] * 32 + [0] * 32, dtype=np.uint8), (2, 1))
        self.indices = np.tile(np.arange(32, dtype=np.int64), (2, 2))
        self.episodes = np.zeros((2, 64), dtype=np.int64)
        self.kwargs = {"source_first": 0, "source_valid": np.ones(64, dtype=bool),
                       "source_ids": np.zeros(64, dtype=np.int64),
                       "online_valid": np.ones(10010, dtype=bool),
                       "online_ids": np.zeros(10010, dtype=np.int64),
                       "online_latest": np.arange(10010),
                       "source_roads": {0: 17}, "online_roads": {0: 18}, "diagnostic": {99}}

    def check(self):
        return audit.check_trace(self.tags, self.indices, self.episodes, **self.kwargs)

    def test_valid_32_32_slots(self):
        result = self.check()
        self.assertEqual((result["source_rows"], result["online_rows"]), (64, 64))

    def test_duplicate_source_and_online_each_rejected(self):
        for column in (1, 33):
            with self.subTest(column=column):
                original = self.indices[0, column]
                self.indices[0, column] = 0
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    self.check()
                self.indices[0, column] = original

    def test_invalid_episode_and_nstep_rejected(self):
        self.episodes[0, 0] = 9
        with self.assertRaisesRegex(ValueError, "episode_id"):
            self.check()
        self.episodes[0, 0] = 0
        self.kwargs["source_valid"][0] = False
        with self.assertRaisesRegex(ValueError, "valid original"):
            self.check()

    def test_future_online_start_and_successor_rejected(self):
        self.indices[0, 32] = 10001
        with self.assertRaisesRegex(ValueError, "inserted after"):
            self.check()
        self.indices[0, 32] = 0
        self.kwargs["online_latest"][0] = 10001
        with self.assertRaisesRegex(ValueError, "inserted after"):
            self.check()

    def test_invalid_tags_quota_and_diagnostic_road_rejected(self):
        self.tags[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "tags"):
            self.check()
        self.tags[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "32 source"):
            self.check()
        self.tags[0, 0] = 1
        self.kwargs["source_roads"][0] = 99
        with self.assertRaisesRegex(ValueError, "TRAIN-DIAGNOSTIC"):
            self.check()

    def test_failure_never_writes_pass_receipt(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(audit, "audit", side_effect=ValueError("bad trace")):
            with patch("sys.argv", ["audit", "--repo-root", tmp]):
                with self.assertRaisesRegex(ValueError, "bad trace"):
                    audit.main()
            self.assertFalse((Path(tmp) / audit.RECEIPT).exists())


if __name__ == "__main__":
    unittest.main()
