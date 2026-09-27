"""Synthetic sample-audit contracts without real checkpoint loads or resets."""

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import audit_drq_final_replay_samples as audit


class TestFinalSourceSampleAudit(unittest.TestCase):
    def setUp(self):
        self.frames = np.broadcast_to(np.zeros((84, 84), dtype=np.uint8), (100000, 84, 84))
        self.actions = np.broadcast_to(np.zeros(3, dtype=np.float32), (100000, 3))
        self.state = {
            "capacity": 100000, "size": 100000, "next_sequence": 100000,
            "frames": self.frames, "actions": self.actions,
            "episode_ids": np.zeros(100000, dtype=np.int64),
            "episode_steps": np.arange(100000, dtype=np.int64),
            "terminal": np.zeros(100000, dtype=bool),
            "terminated": np.zeros(100000, dtype=bool),
            "truncated": np.zeros(100000, dtype=bool),
        }
        self.receipt = {"valid_n_step_starts": 99998, "geometry_seeds": [43],
                        "retained_window_schedule_episodes": 2,
                        "scheduled_episodes_consumed": 1,
                        "historical_prefix_episodes_consumed": 0}
        self.resets = [{"event": "reset", "episode_id": 0, "geometry_seed": 43,
                        "seed": 43, "track_id": 2, "schedule_index": 0,
                        "schedule_phase": "retained_window"}]

    def check(self, excluded=()):
        valid = np.zeros(100000, dtype=bool)
        valid[:99998] = True
        with patch.object(audit.old, "valid_starts", return_value=(valid, None)):
            return audit.check_pool(self.state, self.receipt, self.resets, set(excluded))

    def test_valid_pool_joins_sealed_reset(self):
        valid, ids, roads = self.check(excluded=(99,))
        self.assertEqual(int(valid.sum()), 99998)
        self.assertEqual(roads, {0: 43})
        self.assertEqual(len(ids), 100000)

    def test_diagnostic_road_or_unseen_road_rejected(self):
        with self.assertRaisesRegex(ValueError, "excluded road"):
            self.check(excluded=(43,))
        self.state["episode_ids"][99999] = 9
        self.state["episode_steps"][99999] = 0
        self.state["terminal"][99998] = True
        with self.assertRaisesRegex(ValueError, "reset ledger"):
            self.check()

    def test_episode_steps_terminal_and_count_rejected(self):
        self.state["episode_steps"][40] = 0
        with self.assertRaisesRegex(ValueError, "sequence"):
            self.check()
        self.state["episode_steps"][40] = 40
        self.receipt["valid_n_step_starts"] = 2
        with self.assertRaisesRegex(ValueError, "validity"):
            self.check()

    def test_no_pass_receipt_on_failed_input(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(audit, "audit", side_effect=ValueError("tampered sample")):
            with patch("sys.argv", ["audit", "--repo-root", tmp]):
                with self.assertRaisesRegex(ValueError, "tampered sample"):
                    audit.main()
            self.assertFalse((Path(tmp) / audit.RECEIPT).exists())

    def test_update_free_online_warmup_must_match_control(self):
        shape = audit.WARMUP + 1
        control = {key: np.zeros(shape, dtype=np.int64) for key in
                   ("frames", "actions", "rewards", "terminated", "truncated", "terminal",
                    "episode_ids", "episode_steps", "sequence_ids")}
        control["boundary_observations"] = {5: np.zeros((4, 84, 84), dtype=np.uint8)}
        treatment = {key: value.copy() if isinstance(value, np.ndarray) else dict(value)
                     for key, value in control.items()}
        treatment["rewards"][audit.WARMUP] = 1
        audit.check_online_warmup(treatment, control)
        treatment["actions"][17] = 1
        with self.assertRaisesRegex(ValueError, "online warmup"):
            audit.check_online_warmup(treatment, control)

    def test_source_step_must_join_actual_replay_action_and_road(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "steps.jsonl"
            row = {"decision": 1, "sequence_id": 0, "source_seed": 0,
                   "partition": "TRAIN", "episode_id": 0, "episode_step": 0,
                   "schedule_index": 0, "schedule_phase": "retained_window",
                   "geometry_seed": 43, "track_id": 2,
                   "unnoised_native_action": [0., 0., 0.], "noise": [0., 0., 0.],
                   "native_action": [0.9, 0., 0.], "reward": 0.,
                   "terminal": False, "terminated": False, "truncated": False}
            ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.state["rewards"] = np.zeros(100000, dtype=np.float32)
            with self.assertRaisesRegex(ValueError, "actor-noise"):
                audit.check_source_steps(ledger, self.state,
                                         {"decisions": 100000, "source_seed": 0,
                                          "retained_window_schedule_episodes": 1,
                                          "collection_rng_seeds": {"action_noise_seed": 7}}, {0: 43})


if __name__ == "__main__":
    unittest.main()
