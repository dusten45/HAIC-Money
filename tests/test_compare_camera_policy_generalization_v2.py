"""Safety contracts for the next camera-policy evaluation, without episodes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import compare_camera_policy_generalization_v2 as subject
from tools import compare_camera_policy_generalization as previous
from tools import evaluate_bare_generalization as fresh


class UnboundProtocolTests(unittest.TestCase):
    def test_template_cannot_derive_or_open_a_fresh_cell(self):
        template = json.loads(subject.TEMPLATE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(template["status"], "UNBOUND")
        self.assertNotIn("seed_salt_hex", template)
        self.assertNotIn("partitions", template)
        with patch.object(subject, "derived_seeds") as derive, \
             patch.object(subject, "_run_cold_episode") as cold:
            with self.assertRaisesRegex(ValueError, "unbound|source"):
                subject.validate_protocol(template, set())
            derive.assert_not_called()
            cold.assert_not_called()

    def test_uncommitted_source_rejected_before_salt_generation(self):
        template = json.loads(subject.TEMPLATE_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(subject.secrets, "token_hex") as salt:
                with self.assertRaisesRegex(ValueError, "committed"):
                    subject.bind_source(template, "not-a-commit", "_CandidateController",
                                        Path(temporary) / "model.pt", Path(temporary) / "protocol.json",
                                        Path(temporary) / "control.py", Path(temporary) / "candidate.py")
                salt.assert_not_called()

    def test_preflight_without_bound_protocol_starts_no_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(subject, "PROTOCOL_PATH", Path(temporary) / "missing.json"), \
                 patch.object(subject, "OUTPUT_ROOT", Path(temporary) / "run"), \
                 patch.object(sys, "argv", ["compare_camera_policy_generalization_v2.py",
                                            "--preflight-only"]), \
                 patch.object(subject, "_run_cold_episode") as cold:
                with self.assertRaisesRegex(ValueError, "unbound"):
                    subject.main()
                cold.assert_not_called()

    def test_binding_writes_single_selector_pair_and_a_fixed_protocol(self):
        template = json.loads(subject.TEMPLATE_PATH.read_text(encoding="utf-8"))
        source = (b"class _CandidateController:\n    pass\n"
                  b"class Agent:\n    def __init__(self):\n"
                  b"        self.route = _CompoundClearingBrakeCarryController()\n")
        synthetic = {"screen": list(range(991001, 991009)),
                     "confirmation": list(range(992001, 992017)),
                     "blind": list(range(993001, 993009))}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.pt"
            model.write_bytes(b"synthetic model")
            protocol_path = root / "protocol.json"
            control = root / "control.py"
            candidate = root / "candidate.py"
            with patch.object(subject, "_committed_agent_source",
                              return_value=("c" * 40, source)), \
                 patch.object(subject.secrets, "token_hex", return_value="0" * 32), \
                 patch.object(subject, "derived_seeds", side_effect=lambda _, phase: synthetic[phase]), \
                 patch.object(fresh, "historical_geometry_seeds", return_value=set()):
                bound = subject.bind_source(template, "c" * 40, "_CandidateController",
                                            model, protocol_path, control, candidate)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    subject.bind_source(template, "c" * 40, "_CandidateController",
                                        model, protocol_path, control, candidate)
            self.assertEqual(control.read_bytes(), source)
            self.assertEqual(candidate.read_bytes(), source.replace(
                b"_CompoundClearingBrakeCarryController()", b"_CandidateController()"))
            self.assertEqual(json.loads(protocol_path.read_text(encoding="utf-8")), bound)
            self.assertEqual(bound["partitions"]["screen"]["spot_check_cells"],
                             [[1, 991001], [4, 991008]])
            self.assertEqual(bound["decision_thresholds"], previous.THRESHOLDS)
            self.assertEqual(bound["source_construction"]["implementation_commit"], "c" * 40)
            tampered = copy.deepcopy(bound)
            tampered["environment_sha256"]["env_wrapper.py"] = "0" * 64
            with patch.object(subject, "derived_seeds", side_effect=lambda _, phase: synthetic[phase]):
                with self.assertRaisesRegex(ValueError, "environment"):
                    subject.validate_protocol(tampered, set())


class FrozenRunnerTests(unittest.TestCase):
    def test_v1_decision_engine_and_strict_thresholds_are_reused(self):
        self.assertIs(subject.compare_pairs, previous.compare_pairs)
        self.assertEqual(subject.THRESHOLDS, previous.THRESHOLDS)

    def test_changed_v1_decision_engine_cannot_bind(self):
        template = json.loads(subject.TEMPLATE_PATH.read_text(encoding="utf-8"))
        source = (b"class _CandidateController:\n    pass\n"
                  b"self.route = _CompoundClearingBrakeCarryController()\n")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(subject, "_committed_agent_source",
                              return_value=("c" * 40, source)), \
                 patch.object(subject, "_decision_engine_sha256", return_value="0" * 64), \
                 patch.object(subject.secrets, "token_hex") as salt:
                with self.assertRaisesRegex(ValueError, "decision engine"):
                    subject.bind_source(template, "c" * 40, "_CandidateController",
                                        root / "model.pt", root / "protocol.json",
                                        root / "control.py", root / "candidate.py")
                salt.assert_not_called()

    def test_confirmation_requires_a_sealed_retained_screen(self):
        protocol = {"partitions": {"screen": {"track_ids": [1], "seeds": [901],
                                               "spot_check_cells": []},
                                   "confirmation": {"track_ids": [1], "seeds": [902],
                                                    "spot_check_cells": []}}}
        identity = {"protocol_sha256": "a" * 64,
                    "source_sha256": {"candidate": "b" * 64},
                    "runner_sha256": "c" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh.prepare_run(root, identity)
            with patch.object(subject, "_run_cold_episode") as cold:
                with self.assertRaisesRegex(ValueError, "finalist"):
                    subject.run_partition(root, identity, protocol, {}, "confirmation")
                cold.assert_not_called()


if __name__ == "__main__":
    unittest.main()
