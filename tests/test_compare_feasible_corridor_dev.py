"""Boundary and receipt tests for the consumed-only corridor comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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


def development_row(arm: str, track: int, seed: int, **changes) -> dict:
    value = row(arm, track, seed)
    value.pop("partition")
    value.pop("repeat")
    value.update(receipt_origin="COLD_WORKER")
    value.update(changes)
    return value


def write_mechanism_rows(root: Path, identity: dict, *, changes: dict | None = None,
                         ego_cells: tuple[tuple[int, int], ...] = ()) -> None:
    harness = subject()
    changes = changes or {}
    for track, seed in harness.MECHANISM_CELLS:
        for arm in harness.ARMS:
            row_changes = changes.get((track, seed, arm), {})
            if arm == "baseline" and (track, seed) in ego_cells:
                provenance = {"receipt_origin": "EGO_SCREEN_CONTROL",
                              "source_receipt_sha256": "f" * 64}
            else:
                provenance = {"receipt_origin": "COLD_WORKER"}
            harness.record_cell(root, identity,
                                development_row(arm, track, seed, **{**provenance, **row_changes}))


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
                    "receipt_origin": "COLD_WORKER",
                    "action_trace_sha256": ("a" if arm == "baseline" else "b") * 64})
            summary = harness.report(root, identity, cells, mechanism_only=True, ego_cells=())
            self.assertEqual(summary["comparison"]["decision"], "RETAIN")
            self.assertEqual(summary["evidence_scope"], "CONSUMED_DEVELOPMENT_ONLY")
            self.assertEqual(summary["decision"], "MECHANISM_DIAGNOSTIC_ALLOW_FULL")
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
            summary = harness.report(root, identity, ((1, 11),),
                                     mechanism_only=True, ego_cells=())
            self.assertEqual(summary["decision"], "INCOMPLETE")
            self.assertEqual(summary["comparison"]["missing_cells"], [[1, 11]])


class TriageGateTests(unittest.TestCase):
    def test_full_run_refuses_to_start_without_complete_mechanism_summary(self):
        harness = subject()
        margin, ego = protocols()
        identity = {"protocol_sha256": "a" * 64}
        artifacts = harness.ROOT / ".haic-artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts) as directory:
            root = Path(directory)
            with patch.object(harness, "_run_worker", side_effect=AssertionError("episode started")):
                with self.assertRaisesRegex(ValueError, "mechanism|triage"):
                    harness.run(root, identity, {}, {}, {}, {},
                                harness.expected_development_cells(margin, ego),
                                harness.ego_screen_cells(ego), mechanism_only=False)

    def test_contact_regression_is_recorded_but_does_not_block_development(self):
        harness = subject()
        ego_cells = harness.ego_screen_cells(protocols()[1])
        identity = {"protocol_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh.prepare_run(root, identity)
            cell = harness.MECHANISM_CELLS[0]
            write_mechanism_rows(root, identity, ego_cells=ego_cells, changes={
                (*cell, "candidate"): {"action_trace_sha256": "b" * 64,
                                         "collision_count": 1, "damage": .2},
            })
            summary = harness.report(root, identity, harness.MECHANISM_CELLS,
                                     mechanism_only=True, ego_cells=ego_cells)
            self.assertEqual(summary["comparison"]["decision"], "REJECT")
            self.assertEqual(summary["triage_gate"]["status"], "ALLOW_FULL_CONSUMED_DEVELOPMENT")
            self.assertEqual(summary["triage_gate"]["adverse_contact_cells"], [[*cell, 1]])
            self.assertEqual(summary["triage_gate"]["adverse_damage_cells"], [[*cell, .2]])
            self.assertIn("COMPARISON_REJECT", summary["decision"])
            fresh._atomic_json(root / "mechanism-summary.json", summary)
            self.assertEqual(harness.require_triage_gate(root, identity, ego_cells), summary)

    def test_mechanism_gate_blocks_lost_finish_new_crash_and_both_dnf_decline(self):
        harness = subject()
        identity = {"protocol_sha256": "a" * 64}
        cells = harness.MECHANISM_CELLS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh.prepare_run(root, identity)
            write_mechanism_rows(root, identity, changes={
                (*cells[0], "baseline"): {"finished": True, "progress": 1.0,
                                           "lap_time_ms": 1000, "retire_reason": None},
                (*cells[0], "candidate"): {"action_trace_sha256": "b" * 64},
                (*cells[1], "candidate"): {"retire_reason": "crash"},
                (*cells[2], "candidate"): {"progress": .4},
            })
            summary = harness.report(root, identity, cells, mechanism_only=True,
                                     ego_cells=())
            reasons = summary["triage_gate"]["reasons"]
            self.assertEqual(summary["triage_gate"]["status"], "BLOCK_FULL")
            self.assertTrue(any("control finish lost" in reason for reason in reasons))
            self.assertTrue(any("new crash DNF" in reason for reason in reasons))
            self.assertTrue(any("both-DNF progress declined" in reason for reason in reasons))
            fresh._atomic_json(root / "mechanism-summary.json", summary)
            with self.assertRaisesRegex(ValueError, "new protocol"):
                harness.require_triage_gate(root, identity, ())

    def test_mechanism_gate_requires_action_change_and_reproducible_rows(self):
        harness = subject()
        identity = {"protocol_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh.prepare_run(root, identity)
            write_mechanism_rows(root, identity)
            summary = harness.report(root, identity, harness.MECHANISM_CELLS,
                                     mechanism_only=True, ego_cells=())
            self.assertEqual(summary["triage_gate"]["status"], "BLOCK_FULL")
            self.assertTrue(any("action trace" in reason for reason in summary["triage_gate"]["reasons"]))
            bad = harness.MECHANISM_CELLS[0]
            candidate_path = harness.cell_path(root, "candidate", *bad)
            candidate_path.unlink()
            harness.record_cell(root, identity, development_row("candidate", *bad,
                               action_trace_sha256="invalid", error="worker failed"))
            failed = harness.report(root, identity, harness.MECHANISM_CELLS,
                                    mechanism_only=True, ego_cells=())
            self.assertEqual(failed["triage_gate"]["status"], "BLOCK_FULL")
            self.assertTrue(any("operational" in reason or "repro" in reason
                                for reason in failed["triage_gate"]["reasons"]))

    def test_nonfinite_or_out_of_range_progress_and_damage_block_triage(self):
        harness = subject()
        cell = harness.MECHANISM_CELLS[0]
        for field, bad_value in (("progress", math.inf), ("damage", math.nan),
                                 ("progress", 1.5), ("damage", -.1)):
            with self.subTest(field=field):
                rows = []
                for track, seed in harness.MECHANISM_CELLS:
                    rows.append(development_row("baseline", track, seed))
                    changes = {"action_trace_sha256": "b" * 64} if (track, seed) == cell else {}
                    if (track, seed) == cell:
                        changes[field] = bad_value
                    rows.append(development_row("candidate", track, seed, **changes))
                gate = harness.derive_triage_gate(rows, harness.MECHANISM_CELLS, ())
                self.assertEqual(gate["status"], "BLOCK_FULL")
                self.assertTrue(any("numeric" in reason for reason in gate["reasons"]))

    def test_full_gate_recomputes_summary_and_requires_all_twenty_rows(self):
        harness = subject()
        identity = {"protocol_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh.prepare_run(root, identity)
            first = harness.MECHANISM_CELLS[0]
            write_mechanism_rows(root, identity, changes={
                (*first, "candidate"): {"action_trace_sha256": "b" * 64}})
            summary = harness.report(root, identity, harness.MECHANISM_CELLS,
                                     mechanism_only=True, ego_cells=())
            fresh._atomic_json(root / "mechanism-summary.json", summary)
            harness.cell_path(root, "candidate", *first).unlink()
            with self.assertRaisesRegex(ValueError, "20|missing|incomplete"):
                harness.require_triage_gate(root, identity, ())

    def test_full_gate_rejects_a_spoofed_allow_summary(self):
        harness = subject()
        identity = {"protocol_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh.prepare_run(root, identity)
            write_mechanism_rows(root, identity)
            summary = harness.report(root, identity, harness.MECHANISM_CELLS,
                                     mechanism_only=True, ego_cells=())
            self.assertEqual(summary["triage_gate"]["status"], "BLOCK_FULL")
            summary["triage_gate"]["status"] = "ALLOW_FULL_CONSUMED_DEVELOPMENT"
            fresh._atomic_json(root / "mechanism-summary.json", summary)
            with self.assertRaisesRegex(ValueError, "differs"):
                harness.require_triage_gate(root, identity, ())

    def test_report_separates_rows_from_new_cold_worker_episodes(self):
        harness = subject()
        margin, ego = protocols()
        identity = {"protocol_sha256": "a" * 64}
        ego_cells = harness.ego_screen_cells(ego)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh.prepare_run(root, identity)
            mechanism = harness.report(root, identity, harness.MECHANISM_CELLS,
                                       mechanism_only=True, ego_cells=ego_cells)
            full = harness.report(root, identity,
                                  harness.expected_development_cells(margin, ego),
                                  mechanism_only=False, ego_cells=ego_cells)
            self.assertEqual((mechanism["rows_expected"], mechanism["cold_worker_episodes_expected"]),
                             (20, 18))
            self.assertEqual((full["rows_expected"], full["cold_worker_episodes_expected"]),
                             (110, 78))
            self.assertNotIn("episodes_expected", mechanism)
            self.assertNotIn("episodes_expected", full)


if __name__ == "__main__":
    unittest.main()
