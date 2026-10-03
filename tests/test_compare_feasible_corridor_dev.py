"""Boundary and receipt tests for the consumed-only corridor comparison."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools import compare_reused_bare_candidate as prior
from tools import evaluate_bare_generalization as fresh


def subject():
    from tools import compare_feasible_corridor_dev

    return compare_feasible_corridor_dev


def screen_protocol(name: str, seeds: list[int], control: str, candidate: str) -> dict:
    return {
        "schema_version": 1,
        "name": name,
        "control_agent_sha256": control,
        "candidate_agent_sha256": candidate,
        "model_sha256": "5" * 64,
        "runtime_helper_sha256": {"action_smoothing.py": "6" * 64,
                                   "action_representation.py": "7" * 64},
        "controller_classes": {"control": "_CompoundClearingBrakeCarryController",
                               "candidate": "_ObservedEgoSideSwitchController"},
        "partitions": {
            "screen": {"track_ids": [1, 2, 3, 4], "seeds": seeds,
                       "spot_check_cells": []},
            "confirmation": {"track_ids": [1, 2, 3, 4], "seeds": [9001]},
            "blind": {"track_ids": [1, 2, 3, 4], "seeds": [9002]},
        },
    }


def protocols() -> tuple[dict, dict]:
    margin = screen_protocol("observed-margin-generalization-v1",
                             [3857792434, 2951861974, 2786341258, 3892761381],
                             "1" * 64, "2" * 64)
    ego = screen_protocol("observed-ego-side-switch-generalization-v1",
                          [2938666309, 533603584, 4089604952, 1162965374,
                           1190129265, 583088201, 3876897788, 1180885023],
                          "3" * 64, "4" * 64)
    return margin, ego


def protocol(margin: dict, ego: dict) -> dict:
    harness = subject()
    return {
        "schema_version": 1,
        "name": "feasible-corridor-dev-v1",
        "status": "PREREGISTERED",
        "evidence_scope": "CONSUMED_DEVELOPMENT_ONLY",
        "margin_screen_protocol_sha256": "a" * 64,
        "margin_screen_freeze_sha256": "b" * 64,
        "margin_screen_summary_sha256": "c" * 64,
        "margin_screen_receipts_sha256": "1" * 64,
        "ego_screen_protocol_sha256": "d" * 64,
        "ego_screen_freeze_sha256": "e" * 64,
        "ego_screen_summary_sha256": "f" * 64,
        "ego_screen_receipts_sha256": "2" * 64,
        "agent_sha256": {"baseline": "3" * 64, "candidate": "8" * 64},
        "controller_classes": {"baseline": "_CompoundClearingBrakeCarryController",
                               "candidate": "_FeasibleCorridorObstacleController"},
        "model_sha256": "5" * 64,
        "runtime_helper_sha256": {"action_smoothing.py": "6" * 64,
                                  "action_representation.py": "7" * 64},
        "max_steps": 2000,
        "frame_skip": 4,
        "development_cells": [list(cell) for cell in harness.expected_development_cells(margin, ego)],
        "mechanism_cells": [list(cell) for cell in harness.MECHANISM_CELLS],
        "gates": {"fresh_confirmation": "SEALED", "blind": "SEALED",
                  "sota_promotion": False},
    }


def screen_identity(screen: dict, protocol_hash: str) -> dict:
    return {
        "protocol_sha256": protocol_hash,
        "source_sha256": {"control": screen["control_agent_sha256"],
                          "candidate": screen["candidate_agent_sha256"]},
        "model_sha256": screen["model_sha256"],
        "helper_sha256": screen["runtime_helper_sha256"],
        "harness_sha256": "9" * 64,
        "environment_sha256": {"core/vendor/car_racing.py": "0" * 64},
        "platform": "test-platform",
        "python": "3.11.15",
    }


def row(arm: str, track: int, seed: int, *, contacts: int = 0) -> dict:
    return {"partition": "screen", "arm": arm, "track_id": track,
            "seed": seed, "repeat": 0, "finished": False, "progress": .5,
            "lap_time_ms": None, "collision_count": contacts,
            "damage": .2 * contacts, "retire_reason": "off_track",
            "offtrack_samples": 0, "partial_offtrack_samples": 0,
            "action_trace_sha256": "a" * 64, "steps": 150, "error": None}


def write_rejected_screen(root: Path, screen: dict, identity: dict) -> None:
    fresh.prepare_run(root, identity)
    for track in screen["partitions"]["screen"]["track_ids"]:
        for seed in screen["partitions"]["screen"]["seeds"]:
            fresh.record_cell(root, identity, row("control", track, seed))
            fresh.record_cell(root, identity, row("candidate", track, seed,
                                                   contacts=int(track == 1 and seed == screen["partitions"]["screen"]["seeds"][0])))
    fresh._atomic_json(root / "screen-summary.json", fresh.report_phase(root, identity, screen, "screen"))


class CellBoundaryTests(unittest.TestCase):
    def test_exact_55_cells_and_fixed_ten_mechanism_cells(self):
        harness = subject()
        margin, ego = protocols()
        cells = harness.expected_development_cells(margin, ego)
        self.assertEqual(len(cells), 55)
        self.assertEqual(len(set(cells)), 55)
        self.assertEqual(cells[:7], prior.LEGACY_CELLS)
        self.assertEqual(cells[7:11], tuple((1, seed) for seed in
                         margin["partitions"]["screen"]["seeds"]))
        self.assertEqual(cells[23:31], tuple((1, seed) for seed in
                         ego["partitions"]["screen"]["seeds"]))
        self.assertEqual(cells[-1], (4, 1180885023))
        self.assertEqual(len(harness.MECHANISM_CELLS), 10)
        self.assertEqual(set(harness.MECHANISM_CELLS[-2:]),
                         {(1, 1190129265), (2, 4089604952)})

    def test_protocol_rejects_extra_cells_holdouts_and_unsealed_claims(self):
        harness = subject()
        margin, ego = protocols()
        registered = protocol(margin, ego)
        harness.validate_protocol(registered, margin, ego, "a" * 64, "d" * 64,
                                  {9001, 9002})
        registered["development_cells"].append([4, 9001])
        with self.assertRaisesRegex(ValueError, "exact"):
            harness.validate_protocol(registered, margin, ego, "a" * 64,
                                      "d" * 64, {9001, 9002})
        registered["development_cells"].pop()
        registered["mechanism_cells"] = [[1, 42]]
        with self.assertRaisesRegex(ValueError, "mechanism"):
            harness.validate_protocol(registered, margin, ego, "a" * 64,
                                      "d" * 64, {9001, 9002})
        registered["mechanism_cells"] = [list(cell) for cell in harness.MECHANISM_CELLS]
        registered["evidence_scope"] = "FRESH"
        with self.assertRaisesRegex(ValueError, "CONSUMED_DEVELOPMENT_ONLY"):
            harness.validate_protocol(registered, margin, ego, "a" * 64,
                                      "d" * 64, {9001, 9002})
        registered["evidence_scope"] = "CONSUMED_DEVELOPMENT_ONLY"
        registered.pop("ego_screen_receipts_sha256")
        with self.assertRaisesRegex(ValueError, "ego_screen_receipts_sha256"):
            harness.validate_protocol(registered, margin, ego, "a" * 64,
                                      "d" * 64, {9001, 9002})
        registered["ego_screen_receipts_sha256"] = "2" * 64
        ego["partitions"]["confirmation"]["seeds"] = [2938666309]
        with self.assertRaisesRegex(ValueError, "confirmation|blind"):
            harness.validate_protocol(registered, margin, ego, "a" * 64,
                                      "d" * 64, {2938666309, 9001, 9002})

    def test_work_plan_reuses_only_ego_screen_control(self):
        harness = subject()
        margin, ego = protocols()
        plan = harness.work_plan(harness.expected_development_cells(margin, ego),
                                 harness.ego_screen_cells(ego))
        self.assertEqual(len(plan), 110)
        self.assertEqual(sum(task[2] == "reuse" for task in plan), 32)
        self.assertEqual(sum(task[2] == "worker" for task in plan), 78)
        self.assertEqual([task for task in plan if task[:2] == ((1, 2938666309), "baseline")],
                         [((1, 2938666309), "baseline", "reuse")])
        self.assertEqual([task for task in plan if task[:2] == ((1, 11), "baseline")],
                         [((1, 11), "baseline", "worker")])
        triage = harness.work_plan(harness.MECHANISM_CELLS, harness.ego_screen_cells(ego))
        self.assertEqual(len(triage), 20)
        self.assertEqual(sum(task[2] == "reuse" for task in triage), 2)
        self.assertEqual(sum(task[2] == "worker" for task in triage), 18)

    def test_output_root_must_stay_under_ignored_artifacts(self):
        harness = subject()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, r"\.haic-artifacts"):
                harness.validate_output_root(Path(directory))
        accepted = harness.validate_output_root(harness.ROOT / ".haic-artifacts" / "corridor-dev" / "run")
        self.assertTrue(accepted.is_relative_to(harness.ROOT / ".haic-artifacts"))


class ReceiptTests(unittest.TestCase):
    def test_rejected_screen_is_complete_and_tamper_evident(self):
        harness = subject()
        margin, _ = protocols()
        identity = screen_identity(margin, "a" * 64)
        reference = {"model_sha256": "5" * 64,
                     "helper_sha256": margin["runtime_helper_sha256"],
                     "harness_sha256": "9" * 64,
                     "environment_sha256": {"core/vendor/car_racing.py": "0" * 64},
                     "platform": "test-platform", "python": "3.11.15"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rejected_screen(root, margin, identity)
            self.assertEqual(harness.verify_rejected_screen(margin, root, "a" * 64,
                                                             reference), 16)
            cell = fresh.cell_path(root, "screen", "control", 1, 3857792434, 0)
            envelope = json.loads(cell.read_text(encoding="utf-8"))
            envelope["row"]["collision_count"] = 3
            cell.write_text(json.dumps(envelope), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest"):
                harness.verify_rejected_screen(margin, root, "a" * 64, reference)

    def test_screen_lineage_mismatch_blocks_reuse(self):
        harness = subject()
        ego = protocols()[1]
        identity = screen_identity(ego, "d" * 64)
        reference = {"model_sha256": "5" * 64,
                     "helper_sha256": ego["runtime_helper_sha256"],
                     "harness_sha256": "9" * 64,
                     "environment_sha256": {"core/vendor/car_racing.py": "0" * 64},
                     "platform": "test-platform", "python": "3.11.15"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rejected_screen(root, ego, identity)
            reference["environment_sha256"] = {"core/vendor/car_racing.py": "f" * 64}
            with self.assertRaisesRegex(ValueError, "environment"):
                harness.verify_rejected_screen(ego, root, "d" * 64, reference)

    def test_rejected_screen_with_failed_spot_repeat_cannot_be_reused(self):
        harness = subject()
        ego = protocols()[1]
        first_seed = ego["partitions"]["screen"]["seeds"][0]
        ego["partitions"]["screen"]["spot_check_cells"] = [[1, first_seed]]
        identity = screen_identity(ego, "d" * 64)
        reference = {key: identity[key] for key in
                     ("model_sha256", "helper_sha256", "harness_sha256",
                      "environment_sha256", "platform", "python")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rejected_screen(root, ego, identity)
            for arm in ("control", "candidate"):
                repeated = row(arm, 1, first_seed, contacts=int(arm == "candidate"))
                repeated["repeat"] = 1
                if arm == "control":
                    repeated["action_trace_sha256"] = "b" * 64
                fresh.record_cell(root, identity, repeated)
            fresh._atomic_json(root / "screen-summary.json",
                               fresh.report_phase(root, identity, ego, "screen"))
            with self.assertRaisesRegex(ValueError, "spot|nondeterministic"):
                harness.verify_rejected_screen(ego, root, "d" * 64, reference)

    def test_imported_control_retains_action_hash_and_source_receipt(self):
        harness = subject()
        ego = protocols()[1]
        identity = screen_identity(ego, "d" * 64)
        with tempfile.TemporaryDirectory() as directory:
            screen_root = Path(directory) / "screen"
            output_root = Path(directory) / "dev"
            write_rejected_screen(screen_root, ego, identity)
            dev_identity = {"protocol_sha256": "f" * 64}
            fresh.prepare_run(output_root, dev_identity)
            imported = harness.import_ego_control(screen_root, identity, output_root,
                                                  dev_identity, 1, 2938666309)
            self.assertEqual(imported["action_trace_sha256"], "a" * 64)
            self.assertEqual(imported["receipt_origin"], "EGO_SCREEN_CONTROL")
            self.assertEqual(imported["source_receipt_sha256"],
                             fresh.digest(fresh.cell_path(screen_root, "screen", "control", 1, 2938666309, 0)))
            self.assertEqual(harness.load_cell(output_root, dev_identity, "baseline", 1, 2938666309), imported)
            self.assertFalse(harness.import_ego_control(screen_root, identity, output_root,
                                                        dev_identity, 1, 2938666309)["error"])

    def test_summary_cannot_promote_candidate_even_when_it_gains_a_finish(self):
        harness = subject()
        identity = {"protocol_sha256": "a" * 64,
                    "margin_screen_protocol_sha256": "b" * 64,
                    "ego_screen_protocol_sha256": "c" * 64}
        cells = ((1, 11),)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh.prepare_run(root, identity)
            for arm, finished, progress in (("baseline", False, .8),
                                            ("candidate", True, 1.0)):
                harness.record_cell(root, identity, {"arm": arm, "track_id": 1,
                    "seed": 11, "finished": finished, "progress": progress,
                    "lap_time_ms": 1000 if finished else None,
                    "collision_count": 0, "damage": 0.0, "error": None,
                    "action_trace_sha256": "a" * 64})
            summary = harness.report(root, identity, cells, mechanism_only=True)
            self.assertEqual(summary["comparison"]["decision"], "RETAIN")
            self.assertEqual(summary["evidence_scope"], "CONSUMED_DEVELOPMENT_ONLY")
            self.assertEqual(summary["decision"], "MECHANISM_DIAGNOSTIC")
            self.assertFalse(summary["fresh_generalization"])
            self.assertFalse(summary["sota_promotion"])
            self.assertFalse(summary["confirmation_unlocked"])
            self.assertFalse(summary["blind_opened"])

    def test_mechanism_summary_waits_for_every_fixed_cell(self):
        harness = subject()
        identity = {"protocol_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh.prepare_run(root, identity)
            summary = harness.report(root, identity, ((1, 11),), mechanism_only=True)
            self.assertEqual(summary["decision"], "INCOMPLETE")
            self.assertEqual(summary["comparison"]["missing_cells"], [[1, 11]])


if __name__ == "__main__":
    unittest.main()
