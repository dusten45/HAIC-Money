"""Contract tests for the source-bound consumed-cell three-arm comparator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools import evaluate_bare_generalization as fresh


def subject():
    from tools import compare_reused_bare_candidate

    return compare_reused_bare_candidate


def screen_protocol():
    return {"schema_version": 1, "name": "observed-margin-generalization-v1",
            "control_agent_sha256": "b" * 64, "candidate_agent_sha256": "c" * 64,
            "partitions": {"screen": {"track_ids": [1, 2, 3, 4],
                                      "seeds": [101, 102, 103, 104]},
                           "confirmation": {"track_ids": [1], "seeds": [201]},
                           "blind": {"track_ids": [1], "seeds": [301]}}}


def protocol(screen):
    harness = subject()
    return {"schema_version": 1, "status": "PREREGISTERED", "name": "reused-three-arm-v1",
            "screen_protocol_sha256": "a" * 64,
            "agent_sha256": {arm: letter * 64 for arm, letter in
                             (("baseline", "b"), ("margin", "c"), ("ego_switch", "d"))},
            "controller_classes": {"baseline": "_CompoundClearingBrakeCarryController",
                                   "margin": "_ObservedMarginArbitrationController",
                                   "ego_switch": "_ObservedEgoSideSwitchController"},
            "model_sha256": "e" * 64,
            "runtime_helper_sha256": {"action_smoothing.py": "f" * 64,
                                      "action_representation.py": "0" * 64},
            "max_steps": 2000, "frame_skip": 4,
            "development_cells": [list(cell) for cell in harness.expected_development_cells(screen)],
            "gates": {"fresh_confirmation": "SEALED", "blind": "SEALED",
                      "sota_promotion": False}}


class BoundaryTests(unittest.TestCase):
    def test_exact_cells_are_seven_legacy_plus_full_consumed_screen_grid(self):
        expected = subject().expected_development_cells(screen_protocol())
        self.assertEqual(len(expected), 23)
        self.assertEqual(expected[:7], subject().LEGACY_CELLS)
        self.assertEqual(expected[7:11], ((1, 101), (1, 102), (1, 103), (1, 104)))
        self.assertEqual(expected[-1], (4, 104))

    def test_rejects_holdout_seed_across_all_track_ids_and_extra_cells(self):
        harness = subject()
        screen = screen_protocol()
        registered = protocol(screen)
        harness.validate_protocol(registered, screen, "a" * 64, {201, 301})
        registered["development_cells"].append([9, 201])
        with self.assertRaisesRegex(ValueError, "exact"):
            harness.validate_protocol(registered, screen, "a" * 64, {201, 301})
        registered["development_cells"] = [list(cell) for cell in harness.expected_development_cells(screen)]
        registered["agent_sha256"]["margin"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "screen source"):
            harness.validate_protocol(registered, screen, "a" * 64, {201, 301})
        registered["agent_sha256"]["margin"] = "c" * 64
        screen["partitions"]["confirmation"]["seeds"] = [101]
        with self.assertRaisesRegex(ValueError, "confirmation|blind"):
            harness.validate_protocol(registered, screen, "a" * 64, {101, 301})

    def test_recursive_holdout_scan_excludes_screen_seed_only_by_partition_name(self):
        harness = subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "old.json").write_text(json.dumps({"partitions": {
                "screen": {"seeds": [11]}, "confirmation": {"seeds": [201]},
                "blind": {"seeds": [301]}},
                "other_holdouts": {"blind": {"seeds": [401]},
                                   "fresh_confirmation": {"seeds": [501]}}}), encoding="utf-8")
            self.assertEqual(harness.sealed_holdout_seeds(root), {201, 301, 401, 501})


class ReceiptTests(unittest.TestCase):
    def test_screen_consumption_requires_every_canonical_receipt(self):
        harness = subject()
        screen = screen_protocol()
        identity = {"protocol_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh.prepare_run(root, identity)
            with self.assertRaisesRegex(ValueError, "unconsumed"):
                harness.verify_screen_consumed(screen, root, identity)
            for track in screen["partitions"]["screen"]["track_ids"]:
                for seed in screen["partitions"]["screen"]["seeds"]:
                    for arm in ("control", "candidate"):
                        fresh.record_cell(root, identity, {"partition": "screen", "arm": arm,
                            "track_id": track, "seed": seed, "repeat": 0, "finished": False,
                            "progress": .5, "collision_count": 0, "damage": 0.0,
                            "lap_time_ms": None, "error": None})
            self.assertEqual(harness.verify_screen_consumed(screen, root, identity), 16)

    def test_receipt_resume_rejects_changed_identity_and_tampered_row(self):
        harness = subject()
        identity = {"protocol_sha256": "a" * 64, "source_sha256": {"baseline": "b" * 64}}
        row = {"arm": "baseline", "track_id": 1, "seed": 11, "finished": False,
               "progress": .8, "lap_time_ms": None, "collision_count": 0, "damage": 0.0,
               "error": None}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness.prepare_run(root, identity)
            self.assertTrue(harness.record_cell(root, identity, row))
            self.assertEqual(harness.load_cell(root, identity, "baseline", 1, 11), row)
            self.assertFalse(harness.record_cell(root, identity, row))
            with self.assertRaisesRegex(ValueError, "identity"):
                harness.prepare_run(root, {**identity, "protocol_sha256": "c" * 64})
            path = harness.cell_path(root, "baseline", 1, 11)
            record = json.loads(path.read_text(encoding="utf-8"))
            record["row"]["collision_count"] = 1
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest"):
                harness.load_cell(root, identity, "baseline", 1, 11)

    def test_strict_paired_gate_is_diagnostic_only(self):
        harness = subject()
        cells = ((1, 11),)
        def row(arm, finished, progress, contacts):
            return {"arm": arm, "track_id": 1, "seed": 11, "finished": finished,
                    "progress": progress, "lap_time_ms": 1000 if finished else None,
                    "collision_count": contacts, "damage": .2 * contacts}
        rows = [row("baseline", False, .7, 0), row("margin", True, 1.0, 0),
                row("ego_switch", True, 1.0, 0)]
        compared = harness.compare_arms(rows, cells, "baseline", "ego_switch")
        self.assertEqual(compared["decision"], "RETAIN")
        self.assertFalse(compared["sota_promotion"])
        rows[-1]["collision_count"] = 1
        self.assertEqual(harness.compare_arms(rows, cells, "baseline", "ego_switch")["decision"], "REJECT")

    def test_margin_safety_repair_retained_when_primary_metrics_tie(self):
        harness = subject()
        cells = ((1, 11),)
        rows = [
            {"arm": arm, "track_id": 1, "seed": 11, "finished": True,
             "progress": 1.0, "lap_time_ms": 1000, "collision_count": contacts,
             "damage": .2 * contacts}
            for arm, contacts in (("baseline", 1), ("margin", 2), ("ego_switch", 1))
        ]
        self.assertEqual(harness.compare_arms(rows, cells, "margin", "ego_switch")["decision"],
                         "RETAIN_SAFETY_REPAIR")
        self.assertEqual(harness.compare_arms(rows, cells, "baseline", "ego_switch")["decision"],
                         "INCONCLUSIVE")
        rows[-1]["lap_time_ms"] = 1020
        self.assertEqual(harness.compare_arms(rows, cells, "margin", "ego_switch")["decision"],
                         "INCONCLUSIVE")

    def test_overall_retention_requires_baseline_gain_and_margin_safety_repair(self):
        harness = subject()
        identity = {"protocol_sha256": "a" * 64, "screen_protocol_sha256": "b" * 64}
        rows = [
            {"arm": arm, "track_id": 1, "seed": 11, "finished": finished,
             "progress": progress, "lap_time_ms": 1000 if finished else None,
             "collision_count": contacts, "damage": .2 * contacts, "error": None}
            for arm, finished, progress, contacts in (
                ("baseline", False, .7, 0), ("margin", True, 1.0, 1),
                ("ego_switch", True, 1.0, 0))
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness.prepare_run(root, identity)
            for row in rows:
                harness.record_cell(root, identity, row)
            result = harness.report(root, identity, ((1, 11),))
            self.assertEqual(result["decision"], "RETAIN_DIAGNOSTIC_CANDIDATE")
            self.assertFalse(result["fresh_generalization"])
            self.assertFalse(result["confirmation_unlocked"])
            self.assertFalse(result["blind_opened"])

    def test_operational_failure_with_missing_arms_is_rejected(self):
        harness = subject()
        identity = {"protocol_sha256": "a" * 64, "screen_protocol_sha256": "b" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness.prepare_run(root, identity)
            harness.record_cell(root, identity, {"arm": "baseline", "track_id": 1,
                "seed": 11, "error": "worker crashed"})
            self.assertEqual(harness.report(root, identity, ((1, 11),))["decision"], "REJECT")


if __name__ == "__main__":
    unittest.main()
