"""Synthetic frozen G0 fixtures only; no simulator, actor, or real trace extraction."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.analyze_rlpd_g0_windows import ANCHORS, SOURCE, TRACE_KEYS, analyze, freeze_template, sha256_file
from scripts.render_rlpd_g0_window import render_window


class TestG0FixedWindows(unittest.TestCase):
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
        self.freeze_path = self.root / "runs/freezes/fixed.json"
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
                arrays = self.make_trace(index)
                if index == 0:
                    arrays["raw_frames"][-1] = 2
                    arrays["raw_finish_phase_bits"][-1, 2:] = 255  # padding, not missing telemetry
                    arrays["raw_finish_phase_bits"][1, 0] = 255  # real missing raw-tick sample
                trace_path = f"traces/seed-{cell['geometry_seed']}-track-1-{actor['id']}.npz"
                np.savez_compressed(self.run / trace_path, **arrays)
                finished = index < 6
                first = index == 0
                self.rows.append({
                    "partition": "TRAIN", "track_id": 1, "geometry_seed": cell["geometry_seed"],
                    "actor_id": actor["id"], "actor_sha256": actor["sha256"],
                    "road_centerline_sha256": hashlib.sha256(str(cell["geometry_seed"]).encode()).hexdigest(),
                    "trace_path": trace_path,
                    "steps": 41, "driven_raw_frames": int(arrays["raw_frames"].sum()), "reset_initial_raw_frames": 1,
                    "reset_noop_raw_frames": 50, "raw_reward_sum": float(arrays["summed_reward"].sum()),
                    "max_progress": float(arrays["progress"].max()),
                    "summary": {
                        "outcome": "finished" if finished else "unknown",
                        "first_observed_step": 2 if first else None,
                        "first_observed_events": ["observed_contact"] if first else [],
                        "first_event_status": "unknown" if first else "none",
                        "missing_steps": [0] if first else [], "final_negative_streak": 0,
                        "qualified_nonfinish": index == 6,
                        "unassessed_precursors": ["finish_phase"] if index == 6 else [],
                    },
                })
        self.seal()

    def make_trace(self, index):
        n = 41
        stack = np.zeros((4, 84, 84), dtype=np.uint8)
        next_frame = np.broadcast_to(np.arange(1, n + 1, dtype=np.uint8)[:, None, None], (n, 84, 84)).copy()
        observation = np.concatenate((stack[-1][None], next_frame[:-1]))
        action = np.tile(np.array([0.2, 0.3, 0.1], dtype=np.float32), (n, 1))
        new_tiles = np.zeros(n, dtype=np.int32)
        new_tiles[[0, 20]] = 1
        if index == 23:
            new_tiles[-1] = 1
        contact = np.zeros(n, dtype=np.bool_)
        distance = np.ones(n, dtype=np.float64)
        directed = np.full(n, 0.01, dtype=np.float64)
        if index == 0:
            contact[2] = True
            distance[3] = 10.0
            directed[0] = np.nan
        qualified = np.zeros(n, dtype=np.bool_)
        if index < 6:
            qualified[35:] = True
        if index == 6:
            qualified[30:] = True
        finished = np.zeros(n, dtype=np.bool_)
        terminated = np.zeros(n, dtype=np.bool_)
        finished[-1] = index < 6
        terminated[-1] = True
        return {
            "initial_stack": stack, "observation_frame": observation, "next_frame": next_frame,
            "proposed_native_action": action, "executed_native_action": action.copy(),
            "commanded_official_action": action.copy(), "raw_official_action": action.astype(np.float64),
            "summed_reward": np.ones(n, dtype=np.float64), "new_tiles": new_tiles,
            "directed_delta": directed, "centerline_distance_m": distance,
            "centerline_fraction": np.linspace(0, 0.8, n), "speed_m_s": np.ones(n),
            "heading_error_rad": np.zeros(n), "x_m": np.zeros(n), "y_m": np.zeros(n),
            "progress": np.linspace(0, 0.8, n), "damage": np.zeros(n),
            "contact": contact, "off_track_counter": np.zeros(n, dtype=np.int32),
            "finish_qualified": qualified, "finished": finished, "terminated": terminated,
            "truncated": np.zeros(n, dtype=np.bool_), "finish_phase": np.array(["unqualified"] * n),
            "raw_frames": np.full(n, 4, dtype=np.int32),
            "raw_finish_phase_bits": np.zeros((n, 4), dtype=np.uint8),
        }

    def seal(self):
        protocol = {
            "format": "haic-rlpd-g0-diagnostic-v1", "status": "frozen", "partition": "TRAIN",
            "interventions": False, "learner_updates": 0, "cells": self.cells, "actors": self.actors,
            "geometry_audit_sha256": "c" * 64, "centerline_far_threshold_m": 9.0,
            "event_rules": {"stall_window": 20, "tile_window": 30, "directed_delta_epsilon": 0.0001},
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
            "geometry_audit_sha256": "c" * 64, "decisions_spent": 984,
            "driven_raw_frames": sum(row["driven_raw_frames"] for row in self.rows),
            "reset_initial_raw_frames": 24, "reset_noop_raw_frames": 1200,
        }
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.source = dict(SOURCE)
        self.source.update({
            "manifest_sha256": sha256_file(self.manifest_path),
            "protocol_sha256": sha256_file(self.protocol_path),
            "cells_sha256": sha256_file(self.ledger_path),
        })
        self.freeze_path.write_text(json.dumps(freeze_template(self.source)), encoding="utf-8")
        self.freeze_sha = sha256_file(self.freeze_path)

    def run_analysis(self, name="runs/synthetic-windows"):
        return analyze(
            self.root, "runs/freezes/fixed.json", self.freeze_sha, name, source=self.source,
        )

    def windows(self):
        return [json.loads(line) for line in (self.root / "runs/synthetic-windows/windows.jsonl").read_text().splitlines()]

    def test_exact_freeze_schema_and_hash_are_required(self):
        self.assertEqual(freeze_template()["source"], SOURCE)
        self.assertEqual(
            freeze_template()["analysis_source_sha256"],
            sha256_file(Path(__file__).resolve().parents[1] / "scripts/analyze_rlpd_g0_windows.py"),
        )
        self.assertEqual(freeze_template()["anchors"], list(ANCHORS))
        self.assertEqual(freeze_template()["manual_rubric"]["status"], "unreviewed")
        with self.assertRaisesRegex(ValueError, "frozen hash mismatch"):
            analyze(self.root, "runs/freezes/fixed.json", "0" * 64, "runs/synthetic-windows", source=self.source)
        altered = freeze_template(self.source)
        altered["window_radius_decisions"] = 19
        self.freeze_path.write_text(json.dumps(altered), encoding="utf-8")
        self.freeze_sha = sha256_file(self.freeze_path)
        with self.assertRaisesRegex(ValueError, "exact fixed-window schema"):
            self.run_analysis()
        self.assertFalse((self.root / "runs/synthetic-windows").exists())

    def test_all_rows_controls_anchor_missingness_clipping_and_full_pixel_indexing(self):
        receipt = self.run_analysis()
        windows = self.windows()
        self.assertEqual(receipt["window_count"], 144)
        self.assertEqual(len(windows), 24 * len(ANCHORS))
        self.assertEqual(len({(w["geometry_seed"], w["actor_id"]) for w in windows}), 24)
        self.assertEqual(sum(w["outcome"] == "finished" for w in windows), 36)
        self.assertEqual(receipt["finished_controls"], 6)
        self.assertFalse(receipt["labels_assigned"])
        self.assertIsNone(receipt["causal_claim"])
        self.assertEqual(sha256_file(self.root / "runs/synthetic-windows/windows.jsonl"), receipt["windows_sha256"])
        first = windows[0]
        self.assertEqual((first["anchor"], first["anchor_step"], first["window_start"], first["window_end_inclusive"]),
                         ("first_observed", 2, 0, 22))
        self.assertEqual((first["left_clipped_decisions"], first["right_clipped_decisions"]), (18, 0))
        self.assertEqual(first["missing_diagnostic_steps"], [0])
        self.assertEqual(first["missing_directed_delta_steps"], [0])
        self.assertEqual(first["missing_raw_phase_steps"], [1])
        self.assertEqual(first["observations"]["contact_decisions"], 1)
        self.assertEqual(first["contact_sheet_steps"], [0, 2, 5, 10, 15, 20, 22])
        self.assertEqual(first["tiles_sha256"], sha256_file(self.root / "runs/synthetic-windows" / first["tiles_path"]))
        with np.load(self.root / "runs/synthetic-windows" / first["tiles_path"], allow_pickle=False) as tiles:
            self.assertEqual(set(tiles.files), (TRACE_KEYS - {"initial_stack"}) | {
                "decision_steps", "observation_stack_at_start", "contact_sheet_steps", "contact_sheet_tiles",
            })
            np.testing.assert_array_equal(tiles["decision_steps"], np.arange(23))
            np.testing.assert_array_equal(tiles["observation_stack_at_start"], np.zeros((4, 84, 84), np.uint8))
            np.testing.assert_array_equal(tiles["contact_sheet_steps"], first["contact_sheet_steps"])
            self.assertEqual(tiles["contact_sheet_tiles"].shape, (7, 2, 84, 84))
            self.assertEqual(tiles["contact_sheet_tiles"].dtype, np.uint8)
            self.assertEqual(int(tiles["contact_sheet_tiles"][1, 1, 0, 0]), 3)
            np.testing.assert_array_equal(tiles["proposed_native_action"], tiles["executed_native_action"])
            self.assertEqual(len(tiles["summed_reward"]), 23)
        terminal = windows[5]
        self.assertEqual((terminal["anchor_step"], terminal["window_start"], terminal["window_end_inclusive"]),
                         (40, 20, 40))
        self.assertEqual((terminal["left_clipped_decisions"], terminal["right_clipped_decisions"]), (0, 20))
        self.assertEqual(terminal["missing_raw_phase_steps"], [])
        with np.load(self.root / "runs/synthetic-windows" / terminal["tiles_path"], allow_pickle=False) as tiles:
            np.testing.assert_array_equal(tiles["observation_stack_at_start"][:, 0, 0], [17, 18, 19, 20])
            np.testing.assert_array_equal(tiles["decision_steps"], np.arange(20, 41))

    def test_missing_anchors_are_not_substituted_or_filtered(self):
        self.run_analysis()
        windows = self.windows()
        for anchor, reason in (
            ("first_observed", "no_observed_first_event"),
            ("first_contact", "no_observed_contact"),
            ("first_centerline_far", "no_centerline_far_proxy"),
        ):
            entry = next(w for w in windows if w["actor_id"] == self.actors[1]["id"]
                         and w["geometry_seed"] == self.cells[0]["geometry_seed"] and w["anchor"] == anchor)
            self.assertIsNone(entry["anchor_step"])
            self.assertEqual(entry["missing_anchor_reason"], reason)
            self.assertIsNone(entry["tiles_path"])
        last = next(w for w in windows if w["geometry_seed"] == self.cells[-1]["geometry_seed"]
                    and w["actor_id"] == self.actors[-1]["id"] and w["anchor"] == "after_last_new_tile")
        self.assertEqual(last["missing_anchor_reason"], "no_decision_after_last_new_tile")
        self.assertIsNone(last["window_start"])
        self.assertEqual(len(windows), 144)

    def test_no_tile_visits_cannot_invent_after_last_tile_anchor(self):
        trace_path = self.run / self.rows[-1]["trace_path"]
        with np.load(trace_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["new_tiles"][:] = 0
        np.savez_compressed(trace_path, **arrays)
        self.rows[-1]["summary"].update({
            "first_observed_step": 29, "first_observed_events": ["no_new_tiles"],
            "first_event_status": "observed",
        })
        self.seal()
        self.run_analysis()
        entry = self.windows()[-2]
        self.assertEqual(entry["anchor"], "first_qualified")
        last_tile = self.windows()[-3]
        self.assertEqual(last_tile["anchor"], "after_last_new_tile")
        self.assertEqual(last_tile["missing_anchor_reason"], "no_new_tile_visit")
        self.assertIsNone(last_tile["anchor_step"])

    def test_corrupted_manifest_and_last_trace_fail_before_any_output(self):
        self.manifest_path.write_text(self.manifest_path.read_text() + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen hash mismatch"):
            self.run_analysis()
        self.assertFalse((self.root / "runs/synthetic-windows").exists())
        self.seal()
        last_trace = self.run / self.rows[-1]["trace_path"]
        last_trace.write_bytes(last_trace.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "frozen hash mismatch"):
            self.run_analysis()
        self.assertFalse((self.root / "runs/synthetic-windows").exists())

    def test_rehashed_but_malformed_trace_or_ledger_summary_fails_closed(self):
        last_trace = self.run / self.rows[-1]["trace_path"]
        with np.load(last_trace, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["next_frame"] = arrays["next_frame"][:-1]
        np.savez_compressed(last_trace, **arrays)
        self.seal()
        with self.assertRaisesRegex(ValueError, "pixel frames are malformed"):
            self.run_analysis()
        self.assertFalse((self.root / "runs/synthetic-windows").exists())
        arrays = self.make_trace(23)
        np.savez_compressed(last_trace, **arrays)
        self.rows[0]["summary"]["first_observed_step"] = 3
        self.seal()
        with self.assertRaisesRegex(ValueError, "ledger summary differs"):
            self.run_analysis()
        self.assertFalse((self.root / "runs/synthetic-windows").exists())

    def test_exclusive_output_rejects_original_g0_and_existing_directory(self):
        with self.assertRaisesRegex(ValueError, "new exclusive"):
            self.run_analysis("runs/20260926-rlpd-g0-completion-v1")
        with self.assertRaisesRegex(ValueError, "new exclusive"):
            self.run_analysis("runs/freezes")
        self.run_analysis()
        with self.assertRaisesRegex(ValueError, "new exclusive"):
            self.run_analysis()

    def test_renderer_requires_pinned_manifest_tiles_and_temporary_destination(self):
        self.run_analysis()
        source_script = Path(__file__).resolve().parents[1] / "scripts/analyze_rlpd_g0_windows.py"
        (self.root / "scripts").mkdir()
        (self.root / "scripts/analyze_rlpd_g0_windows.py").write_bytes(source_script.read_bytes())
        preview = self.root / "previews"
        preview.mkdir()
        manifest_sha = sha256_file(self.root / "runs/synthetic-windows/manifest.json")
        kwargs = dict(
            root=self.root, run_dir="runs/synthetic-windows",
            geometry_seed=self.cells[0]["geometry_seed"], actor_id=self.actors[0]["id"],
            anchor="first_observed", manifest_sha256=manifest_sha,
            allowed_output_dir=preview,
        )
        image = preview / "first-event.png"
        result = render_window(**kwargs, output_path=image)
        self.assertTrue(image.is_file())
        self.assertEqual(result["image_sha256"], sha256_file(image))
        self.assertFalse(result["labels_assigned"])
        with self.assertRaisesRegex(ValueError, "frozen hash mismatch"):
            render_window(**{**kwargs, "manifest_sha256": "0" * 64}, output_path=preview / "bad.png")
        with self.assertRaisesRegex(ValueError, "new PNG"):
            render_window(**kwargs, output_path=self.root / "outside.png")
        missing = {**kwargs, "actor_id": self.actors[1]["id"], "output_path": preview / "missing.png"}
        with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
            render_window(**missing)
        (self.root / "runs/synthetic-windows/tiles/row-00-first_observed.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "frozen hash mismatch"):
            render_window(**kwargs, output_path=preview / "changed.png")


if __name__ == "__main__":
    unittest.main()
