"""Frozen predecessor and run-scope tests for consumed corridor v3."""

from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import compare_feasible_corridor_dev as v1
from tools import compare_feasible_corridor_dev_v2 as v2
from tools import evaluate_bare_generalization as fresh
from tests.test_compare_feasible_corridor_dev import protocol, protocols, write_mechanism_rows
from tests.test_compare_feasible_corridor_dev_v2 import write_prior_fixture


def subject():
    return importlib.import_module("tools.compare_feasible_corridor_dev_v3")


def v3_protocol() -> tuple[dict, dict, dict]:
    margin, ego = protocols()
    registered = protocol(margin, ego)
    registered["name"] = "feasible-corridor-dev-v3"
    registered["controller_classes"] = {
        "baseline": "_CompoundClearingBrakeCarryController",
        "candidate": "_FeasibleCorridorSideVetoPaceController",
    }
    registered["prior_development_v1"] = {
        "protocol_sha256": "1" * 64, "result_sha256": "2" * 64,
        "triage_summary_sha256": "3" * 64, "status": "TRIAGE_BLOCK_FULL",
    }
    registered["prior_development_v2"] = {
        "protocol_sha256": "4" * 64, "result_sha256": "5" * 64,
        "triage_summary_sha256": "6" * 64, "status": "TRIAGE_BLOCK_FULL",
    }
    return registered, margin, ego


def identity() -> dict:
    wrapper = subject()
    return {"protocol_sha256": "a" * 64,
            "wrapper_v3_sha256": fresh.digest(Path(wrapper.__file__)),
            "protocol_name": wrapper.NAME,
            "candidate_route_class": wrapper.CLASSES["candidate"]}


def write_prior_v2_fixture(root: Path) -> tuple[dict, dict, Path, dict]:
    v1_root = root / "v1"
    v1_root.mkdir()
    registered_v1, paths_v1, ego_path = write_prior_fixture(v1_root)
    with patch.object(v2, "PRIOR_V1_PATHS", paths_v1), \
         patch.object(v1, "EGO_SCREEN_PROTOCOL", ego_path):
        v1_proof = v2.validate_prior_development(registered_v1)
    v2_root = root / "v2"
    v2_root.mkdir()
    paths_v2 = {name: v2_root / f"{name}.json" for name in
                ("protocol", "result", "triage_summary", "freeze", "full_summary")}
    fresh._atomic_json(paths_v2["protocol"], {
        "name": v2.NAME, "prior_development_v1": registered_v1["prior_development_v1"],
        "source_snapshots": {"baseline": "synthetic/control.py",
                             "candidate": "synthetic/candidate.py"}})
    protocol_sha256 = fresh.digest(paths_v2["protocol"])
    ego = fresh._read_json(ego_path)
    ego_cells = v1.ego_screen_cells(ego)
    frozen = {"protocol_sha256": protocol_sha256,
              "ego_screen_protocol_sha256": fresh.digest(ego_path),
              "comparator_sha256": fresh.digest(Path(v1.__file__)),
              "wrapper_sha256": fresh.digest(Path(v2.__file__)),
              "source_sha256": {"baseline": "b" * 64, "candidate": "c" * 64},
              "model_sha256": "d" * 64,
              "protocol_name": v2.NAME,
              "candidate_route_class": v2.CLASSES["candidate"],
              "prior_development_v1": v1_proof}
    fresh.prepare_run(v2_root, frozen)
    write_mechanism_rows(v2_root, frozen, ego_cells=ego_cells)
    summary = v2.report(v2_root, frozen, v1.MECHANISM_CELLS,
                        mechanism_only=True, ego_cells=ego_cells)
    fresh._atomic_json(paths_v2["triage_summary"], summary)
    triage_sha256 = fresh.digest(paths_v2["triage_summary"])
    receipt_entries = {}
    for track, seed in v1.MECHANISM_CELLS:
        for arm in v1.ARMS:
            receipt = v1.cell_path(v2_root, arm, track, seed)
            receipt_entries[str(receipt.relative_to(v2_root)).replace("\\", "/")] = fresh.digest(receipt)
    receipts_sha256 = hashlib.sha256(fresh._canonical(receipt_entries)).hexdigest()
    fresh._atomic_json(paths_v2["result"], {
        "experiment": v2.NAME, "status": "TRIAGE_BLOCK_FULL",
        "protocol_sha256": protocol_sha256,
        "mechanism_summary_sha256": triage_sha256,
        "freeze_sha256": fresh.digest(paths_v2["freeze"]),
        "baseline_agent_sha256": frozen["source_sha256"]["baseline"],
        "candidate_agent_sha256": frozen["source_sha256"]["candidate"],
        "model_sha256": frozen["model_sha256"],
        "gates": {"full_development": "NOT_OPENED", "fresh_screen": "NOT_OPENED"},
    })
    registered = {"prior_development_v1": registered_v1["prior_development_v1"],
                  "prior_development_v2": {
                      "protocol_sha256": protocol_sha256,
                      "result_sha256": fresh.digest(paths_v2["result"]),
                      "triage_summary_sha256": triage_sha256,
                      "triage_receipts_sha256": receipts_sha256,
                      "status": "TRIAGE_BLOCK_FULL"}}
    return registered, paths_v1, ego_path, paths_v2


class V3ProtocolTests(unittest.TestCase):
    def test_exact_v3_route_delegates_frozen_55_and_ten_cells(self):
        wrapper = subject()
        registered, margin, ego = v3_protocol()
        with patch.object(wrapper, "validate_prior_development", return_value={}), \
             patch.object(v2, "validate_prior_development", return_value={}):
            cells = wrapper.validate_protocol(registered, margin, ego,
                                              "a" * 64, "d" * 64, {9001, 9002})
        self.assertEqual(cells, v1.expected_development_cells(margin, ego))
        self.assertEqual(wrapper.MECHANISM_CELLS, v1.MECHANISM_CELLS)
        registered["name"] = v2.NAME
        with self.assertRaisesRegex(ValueError, "v3"):
            wrapper.validate_protocol(registered, margin, ego,
                                      "a" * 64, "d" * 64, {9001, 9002})
        registered["name"] = wrapper.NAME
        registered["controller_classes"]["candidate"] = v2.CLASSES["candidate"]
        with self.assertRaisesRegex(ValueError, "v3|route"):
            wrapper.validate_protocol(registered, margin, ego,
                                      "a" * 64, "d" * 64, {9001, 9002})

    def test_protocol_and_output_paths_are_v3_only(self):
        wrapper = subject()
        self.assertEqual(wrapper.validate_protocol_path(wrapper.PROTOCOL_PATH),
                         wrapper.PROTOCOL_PATH.resolve())
        self.assertEqual(wrapper.validate_output_root(wrapper.OUTPUT_ROOT),
                         wrapper.OUTPUT_ROOT.resolve())
        with self.assertRaisesRegex(ValueError, "v3"):
            wrapper.validate_protocol_path(v2.PROTOCOL_PATH)
        with self.assertRaisesRegex(ValueError, "v3"):
            wrapper.validate_output_root(v2.OUTPUT_ROOT)
        with self.assertRaisesRegex(ValueError, "v3"):
            wrapper.validate_output_root(wrapper.OUTPUT_ROOT / "nested")

    def test_prior_v2_proof_recomputes_all_receipts_and_v1_lineage(self):
        wrapper = subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registered, paths_v1, ego_path, paths_v2 = write_prior_v2_fixture(root)
            with patch.object(wrapper, "PRIOR_V2_PATHS", paths_v2), \
                 patch.object(v2, "PRIOR_V1_PATHS", paths_v1), \
                 patch.object(v1, "EGO_SCREEN_PROTOCOL", ego_path), \
                 patch.object(v2, "validate_protocol"), \
                 patch.object(v2, "check_frozen_inputs"):
                proof = wrapper.validate_prior_development(registered)
                self.assertEqual(proof["status"], "TRIAGE_BLOCK_FULL")
                self.assertEqual(proof["v2_wrapper_sha256"], fresh.digest(Path(v2.__file__)))
                self.assertEqual(proof["triage_receipts_sha256"],
                                 registered["prior_development_v2"]["triage_receipts_sha256"])
                receipt = v1.cell_path(paths_v2["freeze"].parent, "candidate", *v1.MECHANISM_CELLS[0])
                original = receipt.read_bytes()
                envelope = json.loads(original)
                envelope["row"]["action_latency_max_ms"] = 99.0
                payload = {"identity": envelope["identity"], "row": envelope["row"]}
                envelope["digest"] = hashlib.sha256(fresh._canonical(payload)).hexdigest()
                fresh._atomic_json(receipt, envelope)
                with self.assertRaisesRegex(ValueError, "receipt|manifest"):
                    wrapper.validate_prior_development(registered)
                receipt.write_bytes(original)
                receipt.unlink()
                with self.assertRaisesRegex(ValueError, "receipt|missing"):
                    wrapper.validate_prior_development(registered)

    def test_v1_receipt_mutation_through_v2_freeze_blocks_v3(self):
        wrapper = subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registered, paths_v1, ego_path, paths_v2 = write_prior_v2_fixture(root)
            with patch.object(wrapper, "PRIOR_V2_PATHS", paths_v2), \
                 patch.object(v2, "PRIOR_V1_PATHS", paths_v1), \
                 patch.object(v1, "EGO_SCREEN_PROTOCOL", ego_path), \
                 patch.object(v2, "validate_protocol"), \
                 patch.object(v2, "check_frozen_inputs"):
                wrapper.validate_prior_development(registered)
                receipt = v1.cell_path(paths_v1["freeze"].parent, "candidate", *v1.MECHANISM_CELLS[0])
                receipt.unlink()
                with self.assertRaisesRegex(ValueError, "prior|receipt|missing"):
                    wrapper.validate_prior_development(registered)

    def test_prior_v2_extra_cell_receipt_blocks_v3(self):
        wrapper = subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registered, paths_v1, ego_path, paths_v2 = write_prior_v2_fixture(root)
            with patch.object(wrapper, "PRIOR_V2_PATHS", paths_v2), \
                 patch.object(v2, "PRIOR_V1_PATHS", paths_v1), \
                 patch.object(v1, "EGO_SCREEN_PROTOCOL", ego_path), \
                 patch.object(v2, "validate_protocol"), \
                 patch.object(v2, "check_frozen_inputs"):
                wrapper.validate_prior_development(registered)
                extra = v1.cell_path(paths_v2["freeze"].parent, "candidate", 4, 123)
                extra.parent.mkdir(parents=True, exist_ok=True)
                extra.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "extra|unexpected|receipt"):
                    wrapper.validate_prior_development(registered)

    def test_identity_preserves_v1_v2_hashes_and_freezes_v3_wrapper(self):
        wrapper = subject()
        registered, margin, ego = v3_protocol()
        inherited = {"comparator_sha256": "1" * 64,
                     "wrapper_sha256": "2" * 64,
                     "protocol_name": v2.NAME,
                     "candidate_route_class": v2.CLASSES["candidate"]}
        with patch.object(wrapper, "validate_prior_development", return_value={"status": "TRIAGE_BLOCK_FULL"}), \
             patch.object(v2, "build_identity", return_value=inherited):
            frozen = wrapper.build_identity(wrapper.PROTOCOL_PATH, registered,
                                            {}, margin, ego)
        self.assertEqual(frozen["comparator_sha256"], "1" * 64)
        self.assertEqual(frozen["wrapper_v2_sha256"], "2" * 64)
        self.assertEqual(frozen["wrapper_sha256"], fresh.digest(Path(wrapper.__file__)))
        self.assertEqual(frozen["protocol_name"], wrapper.NAME)
        self.assertEqual(frozen["candidate_route_class"], wrapper.CLASSES["candidate"])
        self.assertNotIn("wrapper_v2_sha256", inherited)


class V3RunTests(unittest.TestCase):
    def test_direct_run_rejects_noncanonical_cells_before_workers(self):
        wrapper = subject()
        registered, margin, ego = v3_protocol()
        exact_ego = v1.ego_screen_cells(ego)
        wrapper.OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=wrapper.OUTPUT_PARENT) as directory:
            root = Path(directory)
            with patch.object(wrapper, "OUTPUT_ROOT", root), \
                 patch.object(v1, "_run_worker", side_effect=AssertionError("episode started")):
                with self.assertRaisesRegex(ValueError, "v3 run scope|exact"):
                    wrapper.run(root, {}, registered, {}, {}, {},
                                wrapper.MECHANISM_CELLS + ((4, 11),),
                                exact_ego, mechanism_only=True)

    def test_full_run_needs_its_own_complete_v3_triage(self):
        wrapper = subject()
        registered, margin, ego = v3_protocol()
        wrapper.OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=wrapper.OUTPUT_PARENT) as directory:
            root = Path(directory)
            frozen = {"protocol_sha256": "a" * 64,
                      "wrapper_sha256": "b" * 64}
            with patch.object(wrapper, "OUTPUT_ROOT", root), \
                 patch.object(v1, "_run_worker", side_effect=AssertionError("episode started")):
                with self.assertRaisesRegex(ValueError, "v3 mechanism|triage"):
                    wrapper.run(root, frozen, registered, {}, {}, {},
                                v1.expected_development_cells(margin, ego),
                                v1.ego_screen_cells(ego), mechanism_only=False)

    def test_reports_keep_consumed_scope_and_cold_counts(self):
        wrapper = subject()
        registered, margin, ego = v3_protocol()
        frozen = {"protocol_sha256": "a" * 64,
                  "wrapper_sha256": "b" * 64,
                  "wrapper_v2_sha256": "c" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh.prepare_run(root, frozen)
            ego_cells = v1.ego_screen_cells(ego)
            triage = wrapper.report(root, frozen, wrapper.MECHANISM_CELLS,
                                    mechanism_only=True, ego_cells=ego_cells)
            full = wrapper.report(root, frozen,
                                  v1.expected_development_cells(margin, ego),
                                  mechanism_only=False, ego_cells=ego_cells)
            self.assertEqual((triage["rows_expected"], triage["cold_worker_episodes_expected"]), (20, 18))
            self.assertEqual((full["rows_expected"], full["cold_worker_episodes_expected"]), (110, 78))
            self.assertEqual(triage["protocol_name"], wrapper.NAME)
            self.assertEqual(triage["evidence_scope"], "CONSUMED_DEVELOPMENT_ONLY")
            self.assertFalse(full["fresh_generalization"])


if __name__ == "__main__":
    unittest.main()
