"""Synthetic NPZ/provenance checks only; no real G0 or reserved-road analysis."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from haic.algorithms.rlpd.visual_motion import visual_motion_score
from scripts.analyze_rlpd_pixel_motion import (
    ACTION_KEYS, SOURCE, analyze, contact_stall_labels, rule_template, sha256_file,
)


class TestVisualMotionScore(unittest.TestCase):
    def test_static_and_moving_four_frame_stacks(self):
        self.assertEqual(visual_motion_score(np.zeros((4, 84, 84), dtype=np.float32)), 0.0)
        moving = np.zeros((4, 84, 84), dtype=np.uint8)
        moving[1::2] = 255
        self.assertEqual(visual_motion_score(moving), 1.0)
        self.assertEqual(visual_motion_score(moving.astype(np.float32) / 255.0), 1.0)
        half = np.zeros((4, 84, 84), dtype=np.float32)
        half[1::2] = 0.5
        self.assertEqual(visual_motion_score(half), 0.5)

    def test_input_contract_is_exact_and_bounded(self):
        bad = (
            np.zeros((3, 84, 84), np.uint8), np.zeros((4, 84, 84, 1), np.uint8),
            np.zeros((4, 84, 84), np.float64),
            np.full((4, 84, 84), np.nan, np.float32),
            np.full((4, 84, 84), 1.01, np.float32),
        )
        for item in bad:
            with self.subTest(shape=item.shape, dtype=item.dtype), self.assertRaises(ValueError):
                visual_motion_score(item)

    def test_label_excludes_current_contact_and_resets_at_episode_boundary(self):
        speed = np.ones(23)
        tiles = np.zeros(23, dtype=np.int32)
        contact = np.zeros(23, dtype=np.bool_)
        contact[0] = True
        labels = contact_stall_labels(speed, tiles, contact)
        self.assertEqual(labels[:4], [False] * 4)
        self.assertTrue(all(labels[4:21]))
        self.assertEqual(labels[21:], [False, False])
        contact[:] = False
        contact[-1] = True
        self.assertFalse(contact_stall_labels(speed, tiles, contact)[-1])
        self.assertEqual(contact_stall_labels(speed[:5], tiles[:5], contact[:5]), [False] * 5)
        tiles[10] = 1
        self.assertFalse(contact_stall_labels(speed, tiles, np.array([True] + [False] * 22))[14 - 1])


class TestPixelMotionOperator(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run = self.root / "runs/20260926-rlpd-g0-completion-v1"
        (self.run / "traces").mkdir(parents=True)
        (self.root / "experiments").mkdir()
        (self.root / "runs/freezes").mkdir()
        self.protocol_path = self.root / SOURCE["protocol_path"]
        self.manifest_path = self.root / SOURCE["manifest_path"]
        self.ledger_path = self.root / SOURCE["cells_path"]
        self.rule_path = self.root / "runs/freezes/motion.json"
        self.output = self.root / "runs/motion-result.json"
        self.cells = [
            {"partition": "TRAIN", "track_id": 1, "geometry_seed": 4272000001 + i, "obstacles": True}
            for i in range(12)
        ]
        self.actors = [
            {"id": "long-horizon-seed11", "sha256": "a" * 64, "action_mode": "exported_tanh_mean"},
            {"id": "entropy-v5-author-seed50", "sha256": "b" * 64, "action_mode": "exported_tanh_mean"},
        ]
        self.rows = []
        for cell in self.cells:
            for actor in self.actors:
                index = len(self.rows)
                trace_name = f"traces/seed-{cell['geometry_seed']}-track-1-{actor['id']}.npz"
                arrays = self.make_trace(index)
                np.savez_compressed(self.run / trace_name, **arrays)
                self.rows.append({
                    "partition": "TRAIN", "track_id": 1, "geometry_seed": cell["geometry_seed"],
                    "actor_id": actor["id"], "actor_sha256": actor["sha256"],
                    "road_centerline_sha256": hashlib.sha256(str(cell["geometry_seed"]).encode()).hexdigest(),
                    "trace_path": trace_name, "steps": 7, "driven_raw_frames": 28,
                    "reset_initial_raw_frames": 1, "reset_noop_raw_frames": 50,
                    "raw_reward_sum": 7.0, "max_progress": 0.5,
                    "summary": {"outcome": "finished" if index < 6 else "off_track"},
                })
        self.seal()

    @staticmethod
    def make_trace(index):
        stack = np.full((4, 84, 84), index, np.uint8)
        if index == 0:
            stack[2:] = 255
        next_frame = np.full((7, 84, 84), index, np.uint8)
        if index == 0:
            next_frame[0] = 255
        observation = np.concatenate((stack[-1:], next_frame[:-1]))
        action = np.tile(np.array([0.2, 0.3, 0.1], np.float32), (7, 1))
        contact = np.zeros(7, dtype=np.bool_)
        if index == 0:
            contact[0] = True
        final = np.array([False] * 6 + [True], dtype=np.bool_)
        return {
            "initial_stack": stack, "observation_frame": observation, "next_frame": next_frame,
            "speed_m_s": np.ones(7, dtype=np.float64), "new_tiles": np.zeros(7, dtype=np.int32),
            "contact": contact, "finished": final if index < 6 else np.zeros(7, dtype=np.bool_),
            "terminated": final, "truncated": np.zeros(7, dtype=np.bool_),
            "raw_frames": np.full(7, 4, dtype=np.int32),
            "summed_reward": np.ones(7, dtype=np.float64),
            "progress": np.linspace(0, 0.5, 7),
            "proposed_native_action": action, "executed_native_action": action.copy(),
            "commanded_official_action": action.copy(), "raw_official_action": action.astype(np.float64),
        }

    def seal(self):
        protocol = {
            "format": "haic-rlpd-g0-diagnostic-v1", "status": "frozen", "partition": "TRAIN",
            "interventions": False, "learner_updates": 0, "frame_skip": 4, "max_steps": 2000,
            "geometry_audit_sha256": "c" * 64, "cells": self.cells, "actors": self.actors,
        }
        self.protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        for row in self.rows:
            row["trace_sha256"] = sha256_file(self.run / row["trace_path"])
        self.ledger_path.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")
        manifest = {
            "format": "haic-rlpd-g0-diagnostic-result-v1", "role": "TRAIN-only-failure-diagnostic",
            "ranked": False, "protocol_path": SOURCE["protocol_path"],
            "protocol_sha256": sha256_file(self.protocol_path),
            "cells_sha256": sha256_file(self.ledger_path), "cell_count": 24, "geometry_count": 12,
            "geometry_audit_sha256": "c" * 64, "decisions_spent": 168,
            "driven_raw_frames": 672, "reset_initial_raw_frames": 24, "reset_noop_raw_frames": 1200,
        }
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.source = dict(SOURCE)
        self.source.update({
            "protocol_sha256": sha256_file(self.protocol_path),
            "manifest_sha256": sha256_file(self.manifest_path),
            "cells_sha256": sha256_file(self.ledger_path),
        })
        self.rule_path.write_text(json.dumps(rule_template(self.source)), encoding="utf-8")
        self.rule_sha = sha256_file(self.rule_path)

    def analyze(self):
        return analyze(self.root, "runs/freezes/motion.json", self.rule_sha,
                       "runs/motion-result.json", source=self.source)

    def replace_trace(self, index, change):
        path = self.run / self.rows[index]["trace_path"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        change(arrays)
        np.savez_compressed(path, **arrays)

    def test_all24_decisions_and_all_six_successful_controls_and_geometry_groups(self):
        result = self.analyze()
        self.assertEqual(result, json.loads(self.output.read_text(encoding="utf-8")))
        self.assertEqual(result["episode_count"], 24)
        self.assertEqual(result["decision_count"], 168)
        self.assertEqual(result["finished_controls"], 6)
        self.assertEqual(len(result["geometries"]), 12)
        self.assertEqual(result["geometries"][0]["geometry_seed"], self.cells[0]["geometry_seed"])
        self.assertTrue(all(len(group["episodes"]) == 2 for group in result["geometries"]))
        self.assertEqual(sum(len(episode["scores_by_decision"]) for group in result["geometries"]
                             for episode in group["episodes"]), 168)
        self.assertEqual(sum(episode["outcome"] == "finished" for group in result["geometries"]
                             for episode in group["episodes"]), 6)
        self.assertAlmostEqual(result["geometries"][0]["episodes"][0]["scores_by_decision"][0], 1 / 3)
        self.assertEqual(result["geometries"][0]["episodes"][0]["contact_stall_labels_by_decision"],
                         [False] * 4 + [True] * 3)
        counts = result["distributions"]
        self.assertEqual(counts["contact_stall"]["count"], 3)
        self.assertEqual(counts["other_decisions"]["count"], 165)
        self.assertEqual(counts["all_successful_controls"]["count"], 42)
        self.assertEqual(counts["all_successful_controls"]["scores_sorted"],
                         sorted(counts["all_successful_controls"]["scores_sorted"]))
        self.assertIsNone(result["selected_threshold"])
        self.assertIsNone(result["causal_claim"])
        self.assertFalse(result["updated_policy"])

    def test_label_telemetry_never_changes_a_pixel_score(self):
        original = self.analyze()
        self.output.unlink()
        def alter(arrays):
            arrays["speed_m_s"][:] = 10.0
            arrays["contact"][:] = False
            arrays["new_tiles"][:] = 1
        self.replace_trace(0, alter)
        self.seal()
        changed = self.analyze()
        self.assertEqual(
            original["geometries"][0]["episodes"][0]["scores_by_decision"],
            changed["geometries"][0]["episodes"][0]["scores_by_decision"],
        )
        self.assertEqual(changed["distributions"]["contact_stall"]["count"], 0)

    def test_missing_action_even_last_row_fails_before_output(self):
        self.replace_trace(23, lambda arrays: arrays.pop(ACTION_KEYS[-1]))
        self.seal()
        with self.assertRaisesRegex(ValueError, "missing or duplicate columns"):
            self.analyze()
        self.assertFalse(self.output.exists())

    def test_corrupt_last_trace_or_protocol_rejected_before_output(self):
        last = self.run / self.rows[-1]["trace_path"]
        last.write_bytes(last.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "frozen hash mismatch"):
            self.analyze()
        self.assertFalse(self.output.exists())
        self.seal()
        self.protocol_path.write_text(self.protocol_path.read_text() + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen hash mismatch"):
            self.analyze()
        self.assertFalse(self.output.exists())

    def test_rehashed_malformed_shape_or_nonroundtripping_history_fails_before_output(self):
        self.replace_trace(23, lambda arrays: arrays.update(next_frame=arrays["next_frame"][:-1]))
        self.seal()
        with self.assertRaisesRegex(ValueError, "malformed pixel column"):
            self.analyze()
        self.assertFalse(self.output.exists())
        self.replace_trace(23, lambda arrays: arrays.update(next_frame=self.make_trace(23)["next_frame"]))
        self.replace_trace(23, lambda arrays: arrays["observation_frame"].__setitem__(1, 255))
        self.seal()
        with self.assertRaisesRegex(ValueError, "history disagrees"):
            self.analyze()
        self.assertFalse(self.output.exists())

    def test_rule_hash_semantics_and_output_are_exclusive(self):
        self.assertEqual(rule_template()["source"], SOURCE)
        self.assertEqual(rule_template()["score"]["learned_threshold"], None)
        with self.assertRaisesRegex(ValueError, "frozen hash mismatch"):
            analyze(self.root, "runs/freezes/motion.json", "0" * 64,
                    "runs/motion-result.json", source=self.source)
        altered = rule_template(self.source)
        altered["score"]["learned_threshold"] = 0.1
        self.rule_path.write_text(json.dumps(altered), encoding="utf-8")
        self.rule_sha = sha256_file(self.rule_path)
        with self.assertRaisesRegex(ValueError, "exact diagnostic-only freeze"):
            self.analyze()
        self.assertFalse(self.output.exists())
        self.seal()
        self.analyze()
        with self.assertRaisesRegex(ValueError, "new exclusive"):
            self.analyze()
        with self.assertRaisesRegex(ValueError, "new exclusive"):
            analyze(self.root, "runs/freezes/motion.json", self.rule_sha,
                    "runs/20260926-rlpd-g0-completion-v1/manifest.json", source=self.source)


if __name__ == "__main__":
    unittest.main()
