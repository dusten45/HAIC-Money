"""The v4 consumed comparator must preserve every blocked predecessor receipt."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import compare_feasible_corridor_dev as v1
from tools import compare_feasible_corridor_dev_v2 as v2
from tools import compare_feasible_corridor_dev_v3 as v3
from tools import evaluate_bare_generalization as fresh
from tests.test_compare_feasible_corridor_dev import protocol, protocols, write_mechanism_rows
from tests.test_compare_feasible_corridor_dev_v3 import write_prior_v2_fixture


def subject():
    return importlib.import_module("tools.compare_feasible_corridor_dev_v4")


def v4_protocol() -> tuple[dict, dict, dict]:
    margin, ego = protocols()
    registered = protocol(margin, ego)
    registered["name"] = "feasible-corridor-dev-v4"
    registered["controller_classes"] = {
        "baseline": "_CompoundClearingBrakeCarryController",
        "candidate": "_FeasibleCorridorTemporalSideController",
    }
    for version in (1, 2, 3):
        registered[f"prior_development_v{version}"] = {
            "protocol_sha256": str(version) * 64,
            "result_sha256": str(version + 3) * 64,
            "triage_summary_sha256": str(version + 6) * 64,
            "triage_receipts_sha256": str(version + 9) * 64,
            "status": "TRIAGE_BLOCK_FULL",
        }
    return registered, margin, ego


def write_prior_v3_fixture(root: Path) -> tuple[dict, dict, Path, dict, dict]:
    registered_v2, paths_v1, ego_path, paths_v2 = write_prior_v2_fixture(root)
    with patch.object(v3, "PRIOR_V2_PATHS", paths_v2), \
         patch.object(v2, "PRIOR_V1_PATHS", paths_v1), \
         patch.object(v1, "EGO_SCREEN_PROTOCOL", ego_path), \
         patch.object(v2, "validate_protocol"), \
         patch.object(v2, "check_frozen_inputs"):
        v2_proof = v3.validate_prior_development(registered_v2)
    v3_root = root / "v3"
    v3_root.mkdir()
    paths_v3 = {name: v3_root / f"{name}.json" for name in
                ("protocol", "result", "triage_summary", "freeze", "full_summary")}
    fresh._atomic_json(paths_v3["protocol"], {
        "name": v3.NAME,
        "prior_development_v2": registered_v2["prior_development_v2"],
        "source_snapshots": {"baseline": "synthetic/control.py",
                             "candidate": "synthetic/candidate.py"},
    })
    protocol_sha256 = fresh.digest(paths_v3["protocol"])
    ego = fresh._read_json(ego_path)
    ego_cells = v1.ego_screen_cells(ego)
    frozen = {
        "protocol_sha256": protocol_sha256,
        "ego_screen_protocol_sha256": fresh.digest(ego_path),
        "comparator_sha256": fresh.digest(Path(v1.__file__)),
        "wrapper_v2_sha256": fresh.digest(Path(v2.__file__)),
        "wrapper_sha256": fresh.digest(Path(v3.__file__)),
        "source_sha256": {"baseline": "b" * 64, "candidate": "c" * 64},
        "model_sha256": "d" * 64,
        "protocol_name": v3.NAME,
        "candidate_route_class": v3.CLASSES["candidate"],
        "prior_development_v2": v2_proof,
    }
    fresh.prepare_run(v3_root, frozen)
    write_mechanism_rows(v3_root, frozen, ego_cells=ego_cells)
    summary = v3.report(v3_root, frozen, v1.MECHANISM_CELLS,
                        mechanism_only=True, ego_cells=ego_cells)
    assert summary["decision"] == "TRIAGE_BLOCK_FULL"
    fresh._atomic_json(paths_v3["triage_summary"], summary)
    receipts = {}
    for track, seed in v1.MECHANISM_CELLS:
        for arm in v1.ARMS:
            receipt = v1.cell_path(v3_root, arm, track, seed)
            receipts[str(receipt.relative_to(v3_root)).replace("\\", "/")] = fresh.digest(receipt)
    receipts_sha256 = hashlib.sha256(fresh._canonical(receipts)).hexdigest()
    fresh._atomic_json(paths_v3["result"], {
        "experiment": v3.NAME,
        "status": "TRIAGE_BLOCK_FULL",
        "protocol_sha256": protocol_sha256,
        "mechanism_summary_sha256": fresh.digest(paths_v3["triage_summary"]),
        "freeze_sha256": fresh.digest(paths_v3["freeze"]),
        "baseline_agent_sha256": frozen["source_sha256"]["baseline"],
        "candidate_agent_sha256": frozen["source_sha256"]["candidate"],
        "model_sha256": frozen["model_sha256"],
        "gates": {"mechanism": "BLOCK_FULL", "full_development": "NOT_OPENED",
                  "fresh_screen": "NOT_OPENED", "confirmation": "SEALED_UNRUN",
                  "blind": "SEALED_UNRUN", "sota_promotion": False},
    })
    registered = {
        "prior_development_v1": registered_v2["prior_development_v1"],
        "prior_development_v2": registered_v2["prior_development_v2"],
        "prior_development_v3": {
            "protocol_sha256": protocol_sha256,
            "result_sha256": fresh.digest(paths_v3["result"]),
            "triage_summary_sha256": fresh.digest(paths_v3["triage_summary"]),
            "triage_receipts_sha256": receipts_sha256,
            "status": "TRIAGE_BLOCK_FULL",
        },
    }
    return registered, paths_v1, ego_path, paths_v2, paths_v3


def predecessor_paths(wrapper, root: Path):
    registered, paths_v1, ego_path, paths_v2, paths_v3 = write_prior_v3_fixture(root)
    return registered, paths_v3, (
        patch.object(wrapper, "PRIOR_V3_PATHS", paths_v3),
        patch.object(v3, "PRIOR_V2_PATHS", paths_v2),
        patch.object(v2, "PRIOR_V1_PATHS", paths_v1),
        patch.object(v1, "EGO_SCREEN_PROTOCOL", ego_path),
        patch.object(v2, "validate_protocol"),
        patch.object(v2, "check_frozen_inputs"),
        patch.object(v3, "validate_protocol"),
        patch.object(v3, "check_frozen_inputs"),
    )


class V4ProtocolTests(unittest.TestCase):
    def test_route_and_cell_grid_are_exactly_v4_and_consumed(self):
        wrapper = subject()
        registered, margin, ego = v4_protocol()
        with patch.object(wrapper, "validate_prior_development", return_value={}), \
             patch.object(v3, "validate_prior_development", return_value={}), \
             patch.object(v2, "validate_prior_development", return_value={}):
            cells = wrapper.validate_protocol(registered, margin, ego,
                                              "a" * 64, "d" * 64, {9001, 9002})
        self.assertEqual(cells, v1.expected_development_cells(margin, ego))
        self.assertEqual(wrapper.MECHANISM_CELLS, v1.MECHANISM_CELLS)
        registered["name"] = v3.NAME
        with self.assertRaisesRegex(ValueError, "v4"):
            wrapper.validate_protocol(registered, margin, ego,
                                      "a" * 64, "d" * 64, {9001, 9002})
        registered["name"] = wrapper.NAME
        registered["controller_classes"]["candidate"] = v3.CLASSES["candidate"]
        with self.assertRaisesRegex(ValueError, "v4|route"):
            wrapper.validate_protocol(registered, margin, ego,
                                      "a" * 64, "d" * 64, {9001, 9002})

    def test_protocol_and_output_paths_cannot_alias_v3(self):
        wrapper = subject()
        self.assertEqual(wrapper.validate_protocol_path(wrapper.PROTOCOL_PATH),
                         wrapper.PROTOCOL_PATH.resolve())
        self.assertEqual(wrapper.validate_output_root(wrapper.OUTPUT_ROOT),
                         wrapper.OUTPUT_ROOT.resolve())
        with self.assertRaisesRegex(ValueError, "v4"):
            wrapper.validate_protocol_path(v3.PROTOCOL_PATH)
        with self.assertRaisesRegex(ValueError, "v4"):
            wrapper.validate_output_root(v3.OUTPUT_ROOT)
        with self.assertRaisesRegex(ValueError, "v4"):
            wrapper.validate_output_root(wrapper.OUTPUT_ROOT / "nested")

    def test_v3_proof_recomputes_manifest_all_rows_and_v1_v2_lineage(self):
        wrapper = subject()
        with tempfile.TemporaryDirectory() as directory:
            registered, paths_v3, patches = predecessor_paths(wrapper, Path(directory))
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                proof = wrapper.validate_prior_development(registered)
                self.assertEqual(proof["status"], "TRIAGE_BLOCK_FULL")
                self.assertEqual(proof["v3_wrapper_sha256"], fresh.digest(Path(v3.__file__)))
                self.assertEqual(proof["triage_receipts_sha256"],
                                 registered["prior_development_v3"]["triage_receipts_sha256"])
                self.assertEqual(proof["prior_development_v2"]["status"], "TRIAGE_BLOCK_FULL")
                receipt = v1.cell_path(paths_v3["freeze"].parent, "candidate", *v1.MECHANISM_CELLS[0])
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

    def test_v3_extra_receipt_and_v2_lineage_mutation_block_v4(self):
        wrapper = subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registered, paths_v3, patches = predecessor_paths(wrapper, root)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                wrapper.validate_prior_development(registered)
                extra = v1.cell_path(paths_v3["freeze"].parent, "candidate", 4, 123)
                extra.parent.mkdir(parents=True, exist_ok=True)
                extra.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "extra|unexpected|receipt"):
                    wrapper.validate_prior_development(registered)
                extra.unlink()
                v2_receipt = v1.cell_path(root / "v2", "candidate", *v1.MECHANISM_CELLS[0])
                v2_receipt.unlink()
                with self.assertRaisesRegex(ValueError, "prior|receipt|missing"):
                    wrapper.validate_prior_development(registered)

    def test_v3_result_or_summary_mutation_blocks_v4(self):
        wrapper = subject()
        with tempfile.TemporaryDirectory() as directory:
            registered, paths_v3, patches = predecessor_paths(wrapper, Path(directory))
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                wrapper.validate_prior_development(registered)
                paths_v3["result"].write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "result"):
                    wrapper.validate_prior_development(registered)

    def test_identity_keeps_v1_v2_v3_hashes_and_pins_v4(self):
        wrapper = subject()
        registered, margin, ego = v4_protocol()
        inherited = {"comparator_sha256": "1" * 64,
                     "wrapper_v2_sha256": "2" * 64,
                     "wrapper_sha256": "3" * 64,
                     "protocol_name": v3.NAME,
                     "candidate_route_class": v3.CLASSES["candidate"]}
        with patch.object(wrapper, "validate_prior_development", return_value={"status": "TRIAGE_BLOCK_FULL"}), \
             patch.object(v3, "build_identity", return_value=inherited):
            frozen = wrapper.build_identity(wrapper.PROTOCOL_PATH, registered,
                                            {}, margin, ego)
        self.assertEqual(frozen["comparator_sha256"], "1" * 64)
        self.assertEqual(frozen["wrapper_v2_sha256"], "2" * 64)
        self.assertEqual(frozen["wrapper_v3_sha256"], "3" * 64)
        self.assertEqual(frozen["wrapper_sha256"], fresh.digest(Path(wrapper.__file__)))
        self.assertEqual(frozen["candidate_route_class"], "_FeasibleCorridorTemporalSideController")
        self.assertNotIn("wrapper_v3_sha256", inherited)


class V4RunTests(unittest.TestCase):
    def test_direct_run_rejects_extra_cell_before_worker(self):
        wrapper = subject()
        registered, margin, ego = v4_protocol()
        with patch.object(wrapper, "validate_run_scope", wraps=wrapper.validate_run_scope), \
             patch.object(v1, "_run_worker", side_effect=AssertionError("episode started")):
            with self.assertRaisesRegex(ValueError, "v4 run scope|exact"):
                wrapper.run(wrapper.OUTPUT_ROOT, {}, registered, {}, {}, {},
                            wrapper.MECHANISM_CELLS + ((4, 11),),
                            v1.ego_screen_cells(ego), mechanism_only=True)

    def test_full_run_requires_complete_matching_v4_triage(self):
        wrapper = subject()
        registered, margin, ego = v4_protocol()
        wrapper.OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=wrapper.OUTPUT_PARENT) as directory:
            root = Path(directory)
            with patch.object(wrapper, "OUTPUT_ROOT", root), \
                 patch.object(v1, "_run_worker", side_effect=AssertionError("episode started")):
                with self.assertRaisesRegex(ValueError, "v4 mechanism|triage"):
                    wrapper.run(root, {"protocol_sha256": "a" * 64}, registered,
                                {}, {}, {}, v1.expected_development_cells(margin, ego),
                                v1.ego_screen_cells(ego), mechanism_only=False)

    def test_report_is_consumed_only_and_counts_cold_workers(self):
        wrapper = subject()
        registered, margin, ego = v4_protocol()
        frozen = {"protocol_sha256": "a" * 64,
                  "wrapper_sha256": "b" * 64,
                  "wrapper_v3_sha256": "c" * 64,
                  "wrapper_v2_sha256": "d" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh.prepare_run(root, frozen)
            ego_cells = v1.ego_screen_cells(ego)
            triage = wrapper.report(root, frozen, wrapper.MECHANISM_CELLS,
                                    mechanism_only=True, ego_cells=ego_cells)
            full = wrapper.report(root, frozen, v1.expected_development_cells(margin, ego),
                                  mechanism_only=False, ego_cells=ego_cells)
            self.assertEqual((triage["rows_expected"], triage["cold_worker_episodes_expected"]), (20, 18))
            self.assertEqual((full["rows_expected"], full["cold_worker_episodes_expected"]), (110, 78))
            self.assertEqual(triage["protocol_name"], wrapper.NAME)
            self.assertEqual(triage["evidence_scope"], "CONSUMED_DEVELOPMENT_ONLY")
            self.assertFalse(full["fresh_generalization"])


if __name__ == "__main__":
    unittest.main()
