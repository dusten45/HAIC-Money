"""The v5 consumed comparison must preserve v4's rejected paired triage."""

from __future__ import annotations

from contextlib import ExitStack
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
from tools import compare_feasible_corridor_dev_v4 as v4
from tools import evaluate_bare_generalization as fresh
from tests.test_compare_feasible_corridor_dev import protocol, protocols, write_mechanism_rows
from tests.test_compare_feasible_corridor_dev_v4 import write_prior_v3_fixture


def subject():
    return importlib.import_module("tools.compare_feasible_corridor_dev_v5")


def v5_protocol() -> tuple[dict, dict, dict]:
    margin, ego = protocols()
    registered = protocol(margin, ego)
    registered["name"] = "feasible-corridor-dev-v5"
    registered["controller_classes"] = {
        "baseline": "_CompoundClearingBrakeCarryController",
        "candidate": "_FeasibleCorridorSideWidthController",
    }
    for version in (1, 2, 3):
        registered[f"prior_development_v{version}"] = {
            "protocol_sha256": str(version) * 64,
            "result_sha256": str(version + 3) * 64,
            "triage_summary_sha256": str(version + 6) * 64,
            "triage_receipts_sha256": str(version + 9) * 64,
            "status": "TRIAGE_BLOCK_FULL",
        }
    registered["prior_development_v4"] = {
        "protocol_sha256": "a" * 64,
        "result_sha256": "b" * 64,
        "triage_summary_sha256": "c" * 64,
        "triage_receipts_sha256": "d" * 64,
        "status": "TRIAGE_ALLOW_FULL_COMPARISON_REJECT",
    }
    return registered, margin, ego


def _nested_predecessor_patches(paths_v1, ego_path, paths_v2, paths_v3):
    return (
        patch.object(v4, "PRIOR_V3_PATHS", paths_v3),
        patch.object(v3, "PRIOR_V2_PATHS", paths_v2),
        patch.object(v2, "PRIOR_V1_PATHS", paths_v1),
        patch.object(v1, "EGO_SCREEN_PROTOCOL", ego_path),
        patch.object(v2, "validate_protocol"),
        patch.object(v2, "check_frozen_inputs"),
        patch.object(v3, "validate_protocol"),
        patch.object(v3, "check_frozen_inputs"),
    )


def _predecessor_patches(wrapper, paths_v1, ego_path, paths_v2, paths_v3, paths_v4):
    return (
        patch.object(wrapper, "PRIOR_V4_PATHS", paths_v4),
        *_nested_predecessor_patches(paths_v1, ego_path, paths_v2, paths_v3),
        patch.object(v4, "validate_protocol"),
        patch.object(v4, "check_frozen_inputs"),
    )


def write_prior_v4_fixture(root: Path) -> tuple[dict, tuple]:
    registered_v3, paths_v1, ego_path, paths_v2, paths_v3 = write_prior_v3_fixture(root)
    with ExitStack() as stack:
        for current in _nested_predecessor_patches(paths_v1, ego_path, paths_v2,
                                                   paths_v3):
            stack.enter_context(current)
        v3_proof = v4.validate_prior_development(registered_v3)
    v4_root = root / "v4"
    v4_root.mkdir()
    paths_v4 = {name: v4_root / f"{name}.json" for name in
                ("protocol", "result", "triage_summary", "freeze", "full_summary")}
    fresh._atomic_json(paths_v4["protocol"], {
        "name": v4.NAME,
        "prior_development_v3": registered_v3["prior_development_v3"],
        "source_snapshots": {"baseline": "synthetic/control.py",
                             "candidate": "synthetic/candidate.py"},
    })
    protocol_sha256 = fresh.digest(paths_v4["protocol"])
    ego = fresh._read_json(ego_path)
    ego_cells = v1.ego_screen_cells(ego)
    frozen = {
        "protocol_sha256": protocol_sha256,
        "ego_screen_protocol_sha256": fresh.digest(ego_path),
        "comparator_sha256": fresh.digest(Path(v1.__file__)),
        "wrapper_v2_sha256": fresh.digest(Path(v2.__file__)),
        "wrapper_v3_sha256": fresh.digest(Path(v3.__file__)),
        "wrapper_sha256": fresh.digest(Path(v4.__file__)),
        "source_sha256": {"baseline": "b" * 64, "candidate": "c" * 64},
        "model_sha256": "d" * 64,
        "protocol_name": v4.NAME,
        "candidate_route_class": v4.CLASSES["candidate"],
        "prior_development_v3": v3_proof,
    }
    fresh.prepare_run(v4_root, frozen)
    changed_cell = v1.MECHANISM_CELLS[0]
    write_mechanism_rows(v4_root, frozen,
                         changes={(*changed_cell, "candidate"):
                                  {"action_trace_sha256": "e" * 64,
                                   "collision_count": 1, "damage": .2}},
                         ego_cells=ego_cells)
    summary = v4.report(v4_root, frozen, v1.MECHANISM_CELLS,
                        mechanism_only=True, ego_cells=ego_cells)
    assert summary["decision"] == "TRIAGE_ALLOW_FULL_COMPARISON_REJECT"
    assert summary["triage_gate"]["status"] == "ALLOW_FULL_CONSUMED_DEVELOPMENT"
    assert summary["comparison"]["decision"] == "REJECT"
    fresh._atomic_json(paths_v4["triage_summary"], summary)
    receipts = {}
    for track, seed in v1.MECHANISM_CELLS:
        for arm in v1.ARMS:
            receipt = v1.cell_path(v4_root, arm, track, seed)
            receipts[str(receipt.relative_to(v4_root)).replace("\\", "/")] = fresh.digest(receipt)
    receipts_sha256 = hashlib.sha256(fresh._canonical(receipts)).hexdigest()
    fresh._atomic_json(paths_v4["result"], {
        "experiment": v4.NAME,
        "status": "TRIAGE_ALLOW_FULL_COMPARISON_REJECT",
        "protocol_sha256": protocol_sha256,
        "mechanism_summary_sha256": fresh.digest(paths_v4["triage_summary"]),
        "freeze_sha256": fresh.digest(paths_v4["freeze"]),
        "baseline_agent_sha256": frozen["source_sha256"]["baseline"],
        "candidate_agent_sha256": frozen["source_sha256"]["candidate"],
        "model_sha256": frozen["model_sha256"],
        "gates": {"mechanism": "ALLOW_FULL_CONSUMED_DEVELOPMENT",
                  "full_development": "NOT_OPENED", "fresh_screen": "NOT_OPENED",
                  "confirmation": "SEALED_UNRUN", "blind": "SEALED_UNRUN",
                  "sota_promotion": False},
    })
    registered = {
        "prior_development_v1": registered_v3["prior_development_v1"],
        "prior_development_v2": registered_v3["prior_development_v2"],
        "prior_development_v3": registered_v3["prior_development_v3"],
        "prior_development_v4": {
            "protocol_sha256": protocol_sha256,
            "result_sha256": fresh.digest(paths_v4["result"]),
            "triage_summary_sha256": fresh.digest(paths_v4["triage_summary"]),
            "triage_receipts_sha256": receipts_sha256,
            "status": "TRIAGE_ALLOW_FULL_COMPARISON_REJECT",
        },
    }
    return registered, (paths_v1, ego_path, paths_v2, paths_v3, paths_v4)


class V5PreflightTests(unittest.TestCase):
    def test_exact_route_and_consumed_scope(self):
        wrapper = subject()
        registered, margin, ego = v5_protocol()
        with patch.object(wrapper, "validate_prior_development", return_value={}), \
             patch.object(v4, "validate_prior_development", return_value={}), \
             patch.object(v3, "validate_prior_development", return_value={}), \
             patch.object(v2, "validate_prior_development", return_value={}):
            cells = wrapper.validate_protocol(registered, margin, ego,
                                              "a" * 64, "d" * 64, {9001, 9002})
        self.assertEqual(cells, v1.expected_development_cells(margin, ego))
        self.assertEqual(len(cells), 55)
        self.assertEqual(wrapper.MECHANISM_CELLS, v1.MECHANISM_CELLS)
        registered["controller_classes"]["candidate"] = v4.CLASSES["candidate"]
        with self.assertRaisesRegex(ValueError, "v5|route"):
            wrapper.validate_protocol(registered, margin, ego,
                                      "a" * 64, "d" * 64, {9001, 9002})

    def test_prior_v4_rejected_comparison_and_complete_receipts_are_recomputed(self):
        wrapper = subject()
        with tempfile.TemporaryDirectory() as directory:
            registered, paths = write_prior_v4_fixture(Path(directory))
            with ExitStack() as stack:
                for current in _predecessor_patches(wrapper, *paths):
                    stack.enter_context(current)
                proof = wrapper.validate_prior_development(registered)
                self.assertEqual(proof["status"], "TRIAGE_ALLOW_FULL_COMPARISON_REJECT")
                self.assertEqual(proof["v4_wrapper_sha256"], fresh.digest(Path(v4.__file__)))
                self.assertEqual(proof["prior_development_v3"]["status"], "TRIAGE_BLOCK_FULL")
                receipt = v1.cell_path(paths[-1]["freeze"].parent, "candidate",
                                       *v1.MECHANISM_CELLS[0])
                original = receipt.read_bytes()
                envelope = json.loads(original)
                envelope["row"]["action_latency_max_ms"] = 99.0
                payload = {"identity": envelope["identity"], "row": envelope["row"]}
                envelope["digest"] = hashlib.sha256(fresh._canonical(payload)).hexdigest()
                fresh._atomic_json(receipt, envelope)
                with self.assertRaisesRegex(ValueError, "manifest|receipt"):
                    wrapper.validate_prior_development(registered)
                receipt.write_bytes(original)
                extra = v1.cell_path(paths[-1]["freeze"].parent, "candidate", 4, 123)
                extra.parent.mkdir(parents=True, exist_ok=True)
                extra.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unexpected|receipt"):
                    wrapper.validate_prior_development(registered)

    def test_prior_v4_wrong_status_or_full_summary_blocks_v5(self):
        wrapper = subject()
        with tempfile.TemporaryDirectory() as directory:
            registered, paths = write_prior_v4_fixture(Path(directory))
            with ExitStack() as stack:
                for current in _predecessor_patches(wrapper, *paths):
                    stack.enter_context(current)
                registered["prior_development_v4"]["status"] = "TRIAGE_BLOCK_FULL"
                with self.assertRaisesRegex(ValueError, "prior v4|status"):
                    wrapper.validate_prior_development(registered)
                registered["prior_development_v4"]["status"] = "TRIAGE_ALLOW_FULL_COMPARISON_REJECT"
                paths[-1]["full_summary"].write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "full"):
                    wrapper.validate_prior_development(registered)

    def test_v5_protocol_output_identity_are_separate_from_v4(self):
        wrapper = subject()
        registered, margin, ego = v5_protocol()
        self.assertEqual(wrapper.validate_protocol_path(wrapper.PROTOCOL_PATH),
                         wrapper.PROTOCOL_PATH.resolve())
        self.assertEqual(wrapper.validate_output_root(wrapper.OUTPUT_ROOT),
                         wrapper.OUTPUT_ROOT.resolve())
        with self.assertRaisesRegex(ValueError, "v5"):
            wrapper.validate_protocol_path(v4.PROTOCOL_PATH)
        with self.assertRaisesRegex(ValueError, "v5"):
            wrapper.validate_output_root(v4.OUTPUT_ROOT)
        inherited = {"comparator_sha256": "1" * 64,
                     "wrapper_v2_sha256": "2" * 64,
                     "wrapper_v3_sha256": "3" * 64,
                     "wrapper_sha256": "4" * 64}
        with patch.object(wrapper, "validate_prior_development", return_value={"status": "TRIAGE_ALLOW_FULL_COMPARISON_REJECT"}), \
             patch.object(v4, "build_identity", return_value=inherited):
            identity = wrapper.build_identity(wrapper.PROTOCOL_PATH, registered,
                                              {}, margin, ego)
        self.assertEqual(identity["wrapper_v4_sha256"], "4" * 64)
        self.assertEqual(identity["candidate_route_class"],
                         "_FeasibleCorridorSideWidthController")

    def test_direct_run_rejects_extra_cell_before_worker(self):
        wrapper = subject()
        registered, _, ego = v5_protocol()
        with patch.object(v1, "_run_worker", side_effect=AssertionError("episode started")):
            with self.assertRaisesRegex(ValueError, "v5 run scope|exact"):
                wrapper.run(wrapper.OUTPUT_ROOT, {}, registered, {}, {}, {},
                            wrapper.MECHANISM_CELLS + ((4, 11),),
                            v1.ego_screen_cells(ego), mechanism_only=True)

    def test_full_run_requires_own_matching_v5_triage(self):
        wrapper = subject()
        registered, margin, ego = v5_protocol()
        wrapper.OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=wrapper.OUTPUT_PARENT) as directory:
            root = Path(directory)
            with patch.object(wrapper, "OUTPUT_ROOT", root), \
                 patch.object(v1, "_run_worker", side_effect=AssertionError("episode started")):
                with self.assertRaisesRegex(ValueError, "v5 mechanism|triage"):
                    wrapper.run(root, {"protocol_sha256": "a" * 64}, registered,
                                {}, {}, {}, v1.expected_development_cells(margin, ego),
                                v1.ego_screen_cells(ego), mechanism_only=False)


if __name__ == "__main__":
    unittest.main()
