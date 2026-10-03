"""Version and freeze boundaries for the fallback-speed consumed comparator."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import compare_feasible_corridor_dev as v1
from tools import evaluate_bare_generalization as fresh
from tests.test_compare_feasible_corridor_dev import protocol, protocols, write_mechanism_rows


def subject():
    return importlib.import_module("tools.compare_feasible_corridor_dev_v2")


def v2_protocol() -> tuple[dict, dict, dict]:
    margin, ego = protocols()
    registered = protocol(margin, ego)
    registered["name"] = "feasible-corridor-dev-v2"
    registered["controller_classes"] = {
        "baseline": "_CompoundClearingBrakeCarryController",
        "candidate": "_FeasibleCorridorFallbackSpeedController",
    }
    registered["prior_development_v1"] = {
        "protocol_sha256": "1" * 64,
        "result_sha256": "2" * 64,
        "triage_summary_sha256": "3" * 64,
        "status": "TRIAGE_BLOCK_FULL",
    }
    return registered, margin, ego


def identity() -> dict:
    wrapper = subject()
    return {"protocol_sha256": "a" * 64,
            "wrapper_sha256": fresh.digest(Path(wrapper.__file__)),
            "protocol_name": wrapper.NAME,
            "candidate_route_class": wrapper.CLASSES["candidate"]}


def write_prior_fixture(root: Path) -> tuple[dict, dict, Path]:
    paths = {name: root / f"{name}.json" for name in
             ("protocol", "result", "triage_summary", "freeze", "full_summary")}
    fresh._atomic_json(paths["protocol"], {"name": "feasible-corridor-dev-v1"})
    protocol_hash = fresh.digest(paths["protocol"])
    ego = protocols()[1]
    ego_path = root / "ego-screen-protocol.json"
    fresh._atomic_json(ego_path, ego)
    frozen = {"protocol_sha256": protocol_hash,
              "ego_screen_protocol_sha256": fresh.digest(ego_path),
              "comparator_sha256": fresh.digest(Path(v1.__file__))}
    fresh.prepare_run(root, frozen)
    write_mechanism_rows(root, frozen, ego_cells=v1.ego_screen_cells(ego))
    summary = v1.report(root, frozen, v1.MECHANISM_CELLS,
                        mechanism_only=True, ego_cells=v1.ego_screen_cells(ego))
    fresh._atomic_json(paths["triage_summary"], summary)
    triage_hash = fresh.digest(paths["triage_summary"])
    fresh._atomic_json(paths["result"], {
        "experiment": "feasible-corridor-dev-v1", "status": "TRIAGE_BLOCK_FULL",
        "protocol_sha256": protocol_hash,
        "mechanism_summary_sha256": triage_hash,
        "freeze_sha256": fresh.digest(paths["freeze"]),
    })
    registered = {"prior_development_v1": {
        "protocol_sha256": protocol_hash,
        "result_sha256": fresh.digest(paths["result"]),
        "triage_summary_sha256": triage_hash,
        "status": "TRIAGE_BLOCK_FULL",
    }}
    return registered, paths, ego_path


class V2ProtocolTests(unittest.TestCase):
    def test_exact_v2_protocol_delegates_frozen_55_cell_boundary(self):
        wrapper = subject()
        registered, margin, ego = v2_protocol()
        with patch.object(wrapper, "validate_prior_development", return_value={}):
            cells = wrapper.validate_protocol(registered, margin, ego,
                                              "a" * 64, "d" * 64, {9001, 9002})
        self.assertEqual(len(cells), 55)
        self.assertEqual(cells, v1.expected_development_cells(margin, ego))
        self.assertEqual(wrapper.MECHANISM_CELLS, v1.MECHANISM_CELLS)
        registered["development_cells"].append([4, 9001])
        with patch.object(wrapper, "validate_prior_development", return_value={}):
            with self.assertRaisesRegex(ValueError, "exact"):
                wrapper.validate_protocol(registered, margin, ego,
                                          "a" * 64, "d" * 64, {9001, 9002})

    def test_v1_name_or_route_class_cannot_masquerade_as_v2(self):
        wrapper = subject()
        registered, margin, ego = v2_protocol()
        registered["name"] = "feasible-corridor-dev-v1"
        with self.assertRaisesRegex(ValueError, "v2"):
            wrapper.validate_protocol(registered, margin, ego,
                                      "a" * 64, "d" * 64, {9001, 9002})
        registered["name"] = wrapper.NAME
        registered["controller_classes"]["candidate"] = "_FeasibleCorridorObstacleController"
        with self.assertRaisesRegex(ValueError, "v2|route"):
            wrapper.validate_protocol(registered, margin, ego,
                                      "a" * 64, "d" * 64, {9001, 9002})

    def test_protocol_and_output_paths_are_v2_only(self):
        wrapper = subject()
        wrapper.validate_protocol_path(wrapper.PROTOCOL_PATH)
        with self.assertRaisesRegex(ValueError, "v2"):
            wrapper.validate_protocol_path(v1.ROOT / "experiments" / "feasible-corridor-dev-v1.json")
        accepted = wrapper.validate_output_root(wrapper.OUTPUT_ROOT)
        self.assertEqual(accepted, wrapper.OUTPUT_ROOT.resolve())
        with self.assertRaisesRegex(ValueError, "v2"):
            wrapper.validate_output_root(v1.ROOT / ".haic-artifacts" / "feasible-corridor-dev-v1" / "run")
        with self.assertRaisesRegex(ValueError, "v2"):
            wrapper.validate_output_root(wrapper.OUTPUT_ROOT / "nested")

    def test_identity_freezes_both_comparators_and_exact_route(self):
        wrapper = subject()
        registered, margin, ego = v2_protocol()
        v1_identity = {"comparator_sha256": "b" * 64,
                       "protocol_sha256": "c" * 64,
                       "source_sha256": {"baseline": "d" * 64, "candidate": "e" * 64}}
        with patch.object(v1, "build_identity", return_value=v1_identity), \
             patch.object(wrapper, "validate_prior_development", return_value={"status": "TRIAGE_BLOCK_FULL"}):
            frozen = wrapper.build_identity(wrapper.PROTOCOL_PATH, registered,
                                            {}, margin, ego)
        self.assertEqual(frozen["comparator_sha256"], "b" * 64)
        self.assertEqual(frozen["wrapper_sha256"], fresh.digest(Path(wrapper.__file__)))
        self.assertEqual(frozen["protocol_name"], wrapper.NAME)
        self.assertEqual(frozen["candidate_route_class"], wrapper.CLASSES["candidate"])
        self.assertEqual(frozen["prior_development_v1"], {"status": "TRIAGE_BLOCK_FULL"})
        self.assertNotIn("wrapper_sha256", v1_identity)

    def test_prior_v1_hashes_status_and_comparator_source_are_checked(self):
        wrapper = subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registered, paths, ego_path = write_prior_fixture(root)
            with patch.object(wrapper, "PRIOR_V1_PATHS", paths), \
                 patch.object(v1, "EGO_SCREEN_PROTOCOL", ego_path):
                proof = wrapper.validate_prior_development(registered)
                self.assertEqual(proof["result_sha256"], registered["prior_development_v1"]["result_sha256"])
                paths["triage_summary"].write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "prior|triage"):
                    wrapper.validate_prior_development(registered)

    def test_prior_v1_receipt_deletion_or_mutation_blocks_preflight(self):
        wrapper = subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registered, paths, ego_path = write_prior_fixture(root)
            receipt = v1.cell_path(root, "candidate", *v1.MECHANISM_CELLS[0])
            original = receipt.read_bytes()
            with patch.object(wrapper, "PRIOR_V1_PATHS", paths), \
                 patch.object(v1, "EGO_SCREEN_PROTOCOL", ego_path):
                wrapper.validate_prior_development(registered)
                envelope = json.loads(original)
                envelope["row"]["progress"] = .9
                receipt.write_text(json.dumps(envelope), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "receipt|digest|triage"):
                    wrapper.validate_prior_development(registered)
                receipt.write_bytes(original)
                receipt.unlink()
                with self.assertRaisesRegex(ValueError, "receipt|missing|triage"):
                    wrapper.validate_prior_development(registered)

    def test_changed_wrapper_hash_blocks_resume(self):
        wrapper = subject()
        frozen = identity()
        frozen["wrapper_sha256"] = "0" * 64
        with patch.object(v1, "check_frozen_inputs"):
            with self.assertRaisesRegex(ValueError, "wrapper"):
                wrapper.check_frozen_inputs(frozen, {}, {})


class V2TriageTests(unittest.TestCase):
    def test_direct_run_rejects_noncanonical_cells_and_ego_reuse_before_workers(self):
        wrapper = subject()
        registered, margin, ego = v2_protocol()
        full = v1.expected_development_cells(margin, ego)
        exact_ego = v1.ego_screen_cells(ego)
        bad_protocol = {**registered,
                        "development_cells": registered["development_cells"] + [[4, 9001]]}
        cases = (
            (registered, wrapper.MECHANISM_CELLS + ((4, 11),), exact_ego, True),
            (registered, wrapper.MECHANISM_CELLS, (), True),
            (registered, full, (), False),
            (bad_protocol, full, exact_ego, True),
        )
        wrapper.OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
        for candidate_protocol, cells, ego_cells, mechanism_only in cases:
            with self.subTest(cells=len(cells), ego_cells=len(ego_cells), mechanism_only=mechanism_only):
                with tempfile.TemporaryDirectory(dir=wrapper.OUTPUT_PARENT) as directory:
                    root = Path(directory)
                    with patch.object(wrapper, "OUTPUT_ROOT", root), \
                         patch.object(v1, "_run_worker", side_effect=AssertionError("episode started")):
                        with self.assertRaisesRegex(ValueError, "v2 run scope|exact"):
                            wrapper.run(root, identity(), candidate_protocol, {}, {}, {},
                                        cells, ego_cells, mechanism_only=mechanism_only)

    def test_v1_mechanism_summary_cannot_unlock_v2_full_run(self):
        wrapper = subject()
        frozen = identity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh.prepare_run(root, frozen)
            first = wrapper.MECHANISM_CELLS[0]
            write_mechanism_rows(root, frozen, changes={
                (*first, "candidate"): {"action_trace_sha256": "b" * 64}})
            v1_summary = v1.report(root, frozen, wrapper.MECHANISM_CELLS,
                                   mechanism_only=True, ego_cells=())
            self.assertEqual(v1_summary["triage_gate"]["status"],
                             "ALLOW_FULL_CONSUMED_DEVELOPMENT")
            fresh._atomic_json(root / "mechanism-summary.json", v1_summary)
            with self.assertRaisesRegex(ValueError, "v2|differs"):
                wrapper.require_triage_gate(root, frozen, ())
            v2_summary = wrapper.report(root, frozen, wrapper.MECHANISM_CELLS,
                                        mechanism_only=True, ego_cells=())
            fresh._atomic_json(root / "mechanism-summary.json", v2_summary)
            self.assertEqual(wrapper.require_triage_gate(root, frozen, ()), v2_summary)

    def test_full_run_requires_v2_triage_before_any_worker(self):
        wrapper = subject()
        registered, margin, ego = v2_protocol()
        wrapper.OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=wrapper.OUTPUT_PARENT) as directory:
            root = Path(directory)
            with patch.object(wrapper, "OUTPUT_ROOT", root), \
                 patch.object(v1, "_run_worker", side_effect=AssertionError("episode started")):
                with self.assertRaisesRegex(ValueError, "mechanism|triage"):
                    wrapper.run(root, identity(), registered, {}, {}, {},
                                v1.expected_development_cells(margin, ego),
                                v1.ego_screen_cells(ego), mechanism_only=False)

    def test_mechanism_run_checks_wrapper_before_any_worker(self):
        wrapper = subject()
        registered, _, ego = v2_protocol()
        wrapper.OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
        frozen = identity()
        frozen["wrapper_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(dir=wrapper.OUTPUT_PARENT) as directory:
            root = Path(directory)
            with patch.object(wrapper, "OUTPUT_ROOT", root), \
                 patch.object(v1, "_run_worker", side_effect=AssertionError("episode started")):
                with self.assertRaisesRegex(ValueError, "wrapper"):
                    wrapper.run(root, frozen, registered, {}, {}, {},
                                wrapper.MECHANISM_CELLS,
                                v1.ego_screen_cells(ego), mechanism_only=True)

    def test_v2_reports_keep_cold_episode_counts_and_scope(self):
        wrapper = subject()
        registered, margin, ego = v2_protocol()
        frozen = identity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh.prepare_run(root, frozen)
            screen_cells = v1.ego_screen_cells(ego)
            triage = wrapper.report(root, frozen, wrapper.MECHANISM_CELLS,
                                    mechanism_only=True, ego_cells=screen_cells)
            full = wrapper.report(root, frozen,
                                  v1.expected_development_cells(margin, ego),
                                  mechanism_only=False, ego_cells=screen_cells)
            self.assertEqual((triage["rows_expected"], triage["cold_worker_episodes_expected"]),
                             (20, 18))
            self.assertEqual((full["rows_expected"], full["cold_worker_episodes_expected"]),
                             (110, 78))
            for summary in (triage, full):
                self.assertEqual(summary["protocol_name"], wrapper.NAME)
                self.assertEqual(summary["wrapper_sha256"], frozen["wrapper_sha256"])
                self.assertEqual(summary["evidence_scope"], "CONSUMED_DEVELOPMENT_ONLY")
                self.assertFalse(summary["fresh_generalization"])


if __name__ == "__main__":
    unittest.main()
