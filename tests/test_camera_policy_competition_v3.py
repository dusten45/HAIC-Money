"""V3 source, protocol, and phase seals; these tests run no episodes."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import compare_camera_policy_competition_v3 as subject
from tools import compare_camera_policy_generalization as v1
from tools import evaluate_bare_generalization as fresh


def _result(arm: str, finished: bool) -> dict:
    return {"finished": finished, "progress": 1.0 if finished else 0.5,
            "lap_time_ms": 20_000 if finished else None,
            "retire_reason": None if finished else "off_track",
            "collision_count": 0, "damage": 0.0,
            "steps": 100,
            "offtrack_samples": 0, "partial_offtrack_samples": 0,
            "initialization_ms": 10.0, "reset_ms": 1.0,
            "action_latency_max_ms": 1.0, "peak_worker_rss_mib": 220.0,
            "action_trace_sha256": ("1" if arm == "control" else "2") * 64,
            "error": None}


def _protocol() -> dict:
    screen_seeds = list(range(801, 809))
    confirmation_seeds = list(range(811, 827))
    blind_seeds = list(range(831, 839))
    return {"partitions": {
        "screen": {"track_ids": [1, 2, 3, 4], "seeds": screen_seeds,
                   "spot_check_cells": [[1, 801], [4, 808]]},
        "confirmation": {"track_ids": [1, 2, 3, 4], "seeds": confirmation_seeds,
                         "spot_check_cells": [[1, 811], [2, 812], [3, 813], [4, 814]]},
        "blind": {"track_ids": [1, 2, 3, 4], "seeds": blind_seeds,
                  "spot_check_cells": [[1, 831], [4, 838]]}},
        "controller_classes": {"control": subject.CONTROL_CLASS,
                               "candidate": subject.CANDIDATE_CLASS}}


def _identity() -> dict:
    return {"protocol_sha256": "a" * 64,
            "source_sha256": {"control": "b" * 64, "candidate": "c" * 64},
            "model_sha256": "f" * 64, "harness_sha256": "0" * 64,
            "helper_sha256": {"action_smoothing.py": "1" * 64,
                              "action_representation.py": "2" * 64},
            "environment_sha256": {"env_wrapper.py": "3" * 64},
            "runner_sha256": "d" * 64, "decision_engine_sha256": "e" * 64}


def _record_pairs(root: Path, identity: dict, phase: str, protocol: dict,
                  include_repeats: bool = True) -> None:
    for track, seed, repeat in fresh.expected_cells(protocol, phase):
        if repeat and not include_repeats:
            continue
        for arm in fresh.ARMS:
            fresh.record_cell(root, identity, {"partition": phase, "arm": arm,
                                               "track_id": track, "seed": seed, "repeat": repeat,
                                               **_result(arm, arm == "candidate")})


def _rewrite_row(path: Path, **changes) -> None:
    envelope = fresh._read_json(path)
    envelope["row"].update(changes)
    envelope["digest"] = hashlib.sha256(fresh._canonical({
        "identity": envelope["identity"], "row": envelope["row"]})).hexdigest()
    fresh._atomic_json(path, envelope)


class TemplateAndSourceTests(unittest.TestCase):
    def test_template_is_unbound_and_rejects_seed_material(self):
        template = json.loads(subject.TEMPLATE_PATH.read_text(encoding="utf-8"))
        subject.validate_template(template)
        self.assertEqual(template["name"], "camera-policy-competition-v3")
        self.assertEqual(template["status"], "UNBOUND")
        self.assertNotIn("seed_salt_hex", template)
        self.assertNotIn("partitions", template)
        for key, value in (("seed_salt_hex", "0" * 32), ("partitions", {})):
            altered = {**template, key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unbound|seed"):
                subject.validate_template(altered)

    def test_salted_seeds_are_distinct_and_historical_collision_aborts(self):
        seeds = subject.derived_seed_partitions("0" * 32)
        self.assertEqual([len(seeds[phase]) for phase in fresh.PHASES], [8, 16, 8])
        flat = [seed for phase in fresh.PHASES for seed in seeds[phase]]
        self.assertEqual(len(set(flat)), 32)
        self.assertNotEqual(seeds["screen"], subject.derived_seed_partitions("1" * 32)["screen"])
        subject.require_fresh_seeds(seeds, set())
        with self.assertRaisesRegex(ValueError, "historical|fresh"):
            subject.require_fresh_seeds(seeds, {flat[0]})
        duplicate = copy.deepcopy(seeds)
        duplicate["blind"][0] = flat[0]
        with self.assertRaisesRegex(ValueError, "collision|distinct"):
            subject.require_fresh_seeds(duplicate, set())

    def test_final_result_cannot_make_its_own_protocol_seeds_historical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "camera-policy-competition-v3.json"
            result = root / "camera-policy-competition-v3-result.json"
            old = root / "older-study.json"
            protocol.write_text('{"partitions":{"screen":{"seeds":[22]}}}', encoding="utf-8")
            result.write_text('{"protocol_sha256":"' + fresh.digest(protocol) +
                              '","cells":[{"seed":22}]}', encoding="utf-8")
            old.write_text('{"cells":[{"seed":11}],"geometry_seed":12}', encoding="utf-8")
            self.assertEqual(subject.historical_geometry_seeds_v3(root, protocol), {11, 12})
            result.write_text('{"protocol_sha256":"' + "0" * 64 +
                              '","cells":[{"seed":22}]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "result.*protocol"):
                subject.historical_geometry_seeds_v3(root, protocol)

    def test_only_pinned_committed_agent_and_candidate_class_are_accepted(self):
        full, source = subject._committed_agent_source()
        expected = subprocess.run(["git", "show", f"{subject.IMPLEMENTATION_COMMIT}:agent.py"],
                                  cwd=subject.ROOT, capture_output=True, check=True).stdout
        self.assertEqual(full, subject.IMPLEMENTATION_COMMIT)
        self.assertEqual(source, expected)
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         "291d93081a64d49f18507ec7baaba411e06510bb533221ce07ae585bc4e77b61")
        self.assertIn(b"class _BoundedSideHoldController", source)
        self.assertEqual(source.count(b"_CompoundClearingBrakeCarryController()"), 1)
        selected = subject.selected_source(source)
        self.assertEqual(selected, expected.replace(
            b"_CompoundClearingBrakeCarryController()",
            b"_BoundedSideHoldController()", 1))
        self.assertNotEqual(selected, source)

    def test_wrong_commit_class_or_blob_fails_before_salt_generation(self):
        template = json.loads(subject.TEMPLATE_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            arguments = (template, base / "model.pt", base / "protocol.json",
                         base / "control.py", base / "candidate.py")
            for field, value in (("IMPLEMENTATION_COMMIT", "0" * 40),
                                 ("CANDIDATE_CLASS", "_OtherController")):
                with self.subTest(field=field), patch.object(subject, field, value), \
                     patch.object(subject.secrets, "token_hex") as salt:
                    with self.assertRaisesRegex(ValueError, "template|unbound"):
                        subject.bind_source(*arguments)
                    salt.assert_not_called()
            with patch.object(subject, "_require_committed_runtime"), \
                 patch.object(subject, "_committed_bytes", return_value=b"changed blob"), \
                 patch.object(subject.secrets, "token_hex") as salt:
                with self.assertRaisesRegex(ValueError, "blob"):
                    subject.bind_source(*arguments)
                salt.assert_not_called()

    def test_uncommitted_template_fails_before_salt_generation(self):
        template = json.loads(subject.TEMPLATE_PATH.read_text(encoding="utf-8"))

        def corrupted_template(path: str, commit: str = "HEAD") -> bytes:
            return (b"changed template" if path.endswith(".template.json")
                    else (subject.ROOT / path).read_bytes())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(subject, "_committed_bytes", side_effect=corrupted_template), \
                 patch.object(subject.secrets, "token_hex") as salt:
                with self.assertRaisesRegex(ValueError, "template must be committed"):
                    subject.bind_source(template, root / "model.pt", root / "protocol.json",
                                        root / "control.py", root / "candidate.py")
                salt.assert_not_called()

    def test_restore_snapshots_recovers_missing_without_new_salt_or_overwrite(self):
        _, source = subject._committed_agent_source()
        selected = subject.selected_source(source)
        protocol = {"control_agent_sha256": hashlib.sha256(source).hexdigest(),
                    "candidate_agent_sha256": hashlib.sha256(selected).hexdigest(),
                    "source_construction": {"selector_from": f"{subject.CONTROL_CLASS}()",
                                            "selector_to": f"{subject.CANDIDATE_CLASS}()"}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control, candidate = root / "control.py", root / "candidate.py"
            with patch.object(subject.secrets, "token_hex") as salt:
                subject.restore_snapshots(protocol, control, candidate)
                salt.assert_not_called()
            self.assertEqual(control.read_bytes(), source)
            self.assertEqual(candidate.read_bytes(), selected)
            control.unlink()
            subject.restore_snapshots(protocol, control, candidate)
            self.assertEqual(control.read_bytes(), source)
            candidate.write_bytes(selected + b"# tampered")
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                subject.restore_snapshots(protocol, control, candidate)
            self.assertEqual(candidate.read_bytes(), selected + b"# tampered")

    def test_bound_protocol_writer_never_overwrites_existing_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.json"
            subject.write_new_protocol(path, {"status": "PREREGISTERED"})
            first = path.read_bytes()
            with self.assertRaises(FileExistsError):
                subject.write_new_protocol(path, {"status": "TAMPERED"})
            self.assertEqual(path.read_bytes(), first)

    def test_preflight_rejects_alternate_model_before_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(sys, "argv", ["v3", "--model", str(Path(temporary) / "other.pt"),
                                           "preflight"]), \
                 patch.object(subject, "_run_cold_episode") as cold:
                with self.assertRaisesRegex(ValueError, "root model"):
                    subject.main()
                cold.assert_not_called()

    def test_running_python_torch_numpy_versions_are_checked_before_commands(self):
        subject.require_runtime_versions()
        for versions in (("3.10", "2.1.0", "1.26.4"),
                         ("3.11", "2.2.0", "1.26.4"),
                         ("3.11", "2.1.0", "2.0.0")):
            with self.subTest(versions=versions), \
                 patch.object(subject, "_runtime_versions", return_value=versions):
                with self.assertRaisesRegex(ValueError, "Python 3.11|Torch 2.1|NumPy 1.26"):
                    subject.require_runtime_versions()
                with patch.object(sys, "argv", ["v3", "preflight"]), \
                     patch.object(subject, "_run_cold_episode") as cold:
                    with self.assertRaisesRegex(ValueError, "Python 3.11|Torch 2.1|NumPy 1.26"):
                        subject.main()
                    cold.assert_not_called()

    def test_restore_snapshots_is_a_recognized_cli_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing_protocol = Path(temporary) / "missing-v3-protocol.json"
            with patch.object(subject, "PROTOCOL_PATH", missing_protocol), \
                 patch.object(sys, "argv", ["v3", "restore-snapshots"]), \
                 patch.object(subject, "_runtime_versions", return_value=("3.11", "0.0", "0.0")), \
                 patch.object(subject, "restore_snapshots") as restore, \
                 patch.object(subject, "_run_cold_episode") as cold:
                with self.assertRaisesRegex(ValueError, "unbound v3 protocol"):
                    subject.main()
                restore.assert_not_called()
                cold.assert_not_called()

    def test_source_pair_rejects_even_one_extra_candidate_byte(self):
        source = b"class _BoundedSideHoldController: pass\nroute = _CompoundClearingBrakeCarryController()\n"
        selected = source.replace(b"_CompoundClearingBrakeCarryController()",
                                  b"_BoundedSideHoldController()")
        protocol = {"source_construction": {"selector_from": "_CompoundClearingBrakeCarryController()",
                                            "selector_to": "_BoundedSideHoldController()"},
                    "control_agent_sha256": hashlib.sha256(source).hexdigest(),
                    "candidate_agent_sha256": hashlib.sha256(selected).hexdigest()}
        with tempfile.TemporaryDirectory() as temporary:
            control, candidate = (Path(temporary) / name for name in ("control.py", "candidate.py"))
            control.write_bytes(source)
            candidate.write_bytes(selected)
            subject.validate_source_pair(protocol, control, candidate)
            candidate.write_bytes(selected + b"\n")
            with self.assertRaisesRegex(ValueError, "solely|SHA256"):
                subject.validate_source_pair(protocol, control, candidate)


class SealedPhaseTests(unittest.TestCase):
    def test_v1_rejected_local_contact_spike_can_unlock_v3_confirmation(self):
        protocol = _protocol()
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh.prepare_run(root, identity)
            _record_pairs(root, identity, "screen", protocol)
            _rewrite_row(fresh.cell_path(root, "screen", "candidate", 1, 802, 0),
                         collision_count=3)
            _rewrite_row(fresh.cell_path(root, "screen", "control", 1, 803, 0),
                         collision_count=3)
            cells = fresh.expected_cells(protocol, "screen")
            rows = [fresh.load_cell(root, identity, "screen", arm, track, seed, repeat)
                    for track, seed, repeat in cells for arm in fresh.ARMS]
            self.assertEqual(v1.compare_pairs(rows, cells, "screen")["decision"], "REJECT")
            self.assertEqual(subject.report_phase(root, identity, protocol, "screen")["decision"], "RETAIN")
            subject.freeze_finalist(root, identity, protocol)
            subject.require_phase(root, identity, protocol, "confirmation")

    def test_v3_rejected_aggregate_contact_increase_cannot_unlock(self):
        protocol = _protocol()
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh.prepare_run(root, identity)
            _record_pairs(root, identity, "screen", protocol)
            _rewrite_row(fresh.cell_path(root, "screen", "candidate", 1, 802, 0),
                         collision_count=1)
            self.assertEqual(subject.report_phase(root, identity, protocol, "screen")["decision"], "REJECT")
            with self.assertRaisesRegex(ValueError, "screen must RETAIN"):
                subject.freeze_finalist(root, identity, protocol)
            with self.assertRaisesRegex(ValueError, "finalist"):
                subject.require_phase(root, identity, protocol, "confirmation")

    def test_extra_receipt_file_rejects_instead_of_being_ignored(self):
        protocol = _protocol()
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh.prepare_run(root, identity)
            _record_pairs(root, identity, "screen", protocol)
            extra = root / "cells" / "screen" / "candidate" / "duplicate.json"
            extra.write_text("{}", encoding="utf-8")
            summary = subject.report_phase(root, identity, protocol, "screen")
            self.assertEqual(summary["decision"], "REJECT")
            self.assertIn("unexpected", " ".join(summary["reasons"]))

    def test_confirmation_waits_for_v3_finalist_and_receipt_bound_seal(self):
        protocol = _protocol()
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh.prepare_run(root, identity)
            with self.assertRaisesRegex(ValueError, "finalist"):
                subject.require_phase(root, identity, protocol, "confirmation")
            _record_pairs(root, identity, "screen", protocol)
            self.assertEqual(subject.report_phase(root, identity, protocol, "screen")["decision"], "RETAIN")
            subject.freeze_finalist(root, identity, protocol)
            subject.require_phase(root, identity, protocol, "confirmation")
            receipt = fresh.cell_path(root, "screen", "candidate", 1, 801, 0)
            envelope = fresh._read_json(receipt)
            envelope["row"]["progress"] = 0.99
            envelope["digest"] = hashlib.sha256(fresh._canonical({
                "identity": identity, "row": envelope["row"]})).hexdigest()
            fresh._atomic_json(receipt, envelope)
            with self.assertRaisesRegex(ValueError, "seal|receipt"):
                subject.require_phase(root, identity, protocol, "confirmation")

    def test_blind_requires_retained_confirmation_and_accepted_seal(self):
        protocol = _protocol()
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh.prepare_run(root, identity)
            _record_pairs(root, identity, "screen", protocol)
            subject.freeze_finalist(root, identity, protocol)
            with self.assertRaisesRegex(ValueError, "confirmation"):
                subject.require_phase(root, identity, protocol, "blind")
            _record_pairs(root, identity, "confirmation", protocol)
            subject.seal_confirmation(root, identity, protocol)
            subject.require_phase(root, identity, protocol, "blind")

    def test_repeat_action_mismatch_rejects_without_extra_statistical_pair(self):
        protocol = _protocol()
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh.prepare_run(root, identity)
            _record_pairs(root, identity, "screen", protocol)
            summary = subject.report_phase(root, identity, protocol, "screen")
            self.assertEqual(summary["decision"], "RETAIN")
            self.assertEqual(summary["canonical_cells"], 32)
            receipt = fresh.cell_path(root, "screen", "candidate", 1, 801, 1)
            envelope = fresh._read_json(receipt)
            envelope["row"]["action_trace_sha256"] = "f" * 64
            envelope["digest"] = hashlib.sha256(fresh._canonical({
                "identity": identity, "row": envelope["row"]})).hexdigest()
            fresh._atomic_json(receipt, envelope)
            summary = subject.report_phase(root, identity, protocol, "screen")
            self.assertEqual(summary["decision"], "REJECT")
            self.assertEqual(summary["canonical_cells"], 32)
            self.assertIn("repeat", " ".join(summary["reasons"]))

    def test_missing_or_failed_repeat_makes_phase_incomplete_or_rejected(self):
        protocol = _protocol()
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh.prepare_run(root, identity)
            _record_pairs(root, identity, "screen", protocol, include_repeats=False)
            self.assertEqual(subject.report_phase(root, identity, protocol, "screen")["decision"], "INCOMPLETE")
            for track, seed in protocol["partitions"]["screen"]["spot_check_cells"]:
                for arm in fresh.ARMS:
                    fresh.record_cell(root, identity, {"partition": "screen", "arm": arm,
                                                       "track_id": track, "seed": seed, "repeat": 1,
                                                       **_result(arm, arm == "candidate")})
            receipt = fresh.cell_path(root, "screen", "candidate", 1, 801, 1)
            envelope = fresh._read_json(receipt)
            envelope["row"]["error"] = "worker failed"
            envelope["digest"] = hashlib.sha256(fresh._canonical({
                "identity": identity, "row": envelope["row"]})).hexdigest()
            fresh._atomic_json(receipt, envelope)
            self.assertEqual(subject.report_phase(root, identity, protocol, "screen")["decision"], "REJECT")

    def test_final_decision_recomputes_receipts_and_rechecks_both_seals(self):
        protocol = _protocol()
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh.prepare_run(root, identity)
            _record_pairs(root, identity, "screen", protocol)
            subject.freeze_finalist(root, identity, protocol)
            _record_pairs(root, identity, "confirmation", protocol)
            subject.seal_confirmation(root, identity, protocol)
            _record_pairs(root, identity, "blind", protocol)
            fresh._atomic_json(root / "screen-summary.json", {"decision": "REJECT",
                                                       "net_finish_gain": -99})
            final = subject.report_combined(root, identity, protocol)
            self.assertEqual(final["decision"], "RETAIN")
            self.assertEqual(final["canonical_cells"], 128)
            self.assertEqual(final["net_finish_gain"], 128)
            _rewrite_row(fresh.cell_path(root, "screen", "candidate", 1, 802, 0),
                         progress=0.99)
            with self.assertRaisesRegex(ValueError, "seal|receipt"):
                subject.report_combined(root, identity, protocol)

    def test_resumed_phase_rechecks_mutation_before_summary_publication(self):
        protocol = _protocol()
        original_digest = fresh.digest
        original_report = subject.report_phase
        for kind in ("source", "model", "helper", "environment", "runner", "gate"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                control = root / "control.py"
                candidate = root / "candidate.py"
                model = root / "model.pt"
                protocol_path = root / "protocol.json"
                source = b"class _BoundedSideHoldController: pass\nroute = _CompoundClearingBrakeCarryController()\n"
                selected = source.replace(b"_CompoundClearingBrakeCarryController()",
                                          b"_BoundedSideHoldController()")
                control.write_bytes(source)
                candidate.write_bytes(selected)
                model.write_bytes(b"model")
                protocol_path.write_text("{}", encoding="utf-8")
                bound = {**protocol,
                         "control_agent_sha256": original_digest(control),
                         "candidate_agent_sha256": original_digest(candidate),
                         "model_sha256": original_digest(model),
                         "runtime_helper_sha256": {
                             name: original_digest(subject.ROOT / name)
                             for name in ("action_smoothing.py", "action_representation.py")},
                         "source_construction": {
                             "selector_from": "_CompoundClearingBrakeCarryController()",
                             "selector_to": "_BoundedSideHoldController()"},
                         "runner_sha256": original_digest(Path(subject.__file__)),
                         "decision_engine_sha256": original_digest(Path(subject.gate.__file__)),
                         "template_sha256": original_digest(subject.TEMPLATE_PATH),
                         "evaluation_harness_sha256": original_digest(Path(fresh.__file__))}
                paths = {"control": control, "candidate": candidate, "model": model,
                         "protocol": protocol_path}
                identity = subject.build_identity(protocol_path, bound, paths)
                bound["environment_sha256"] = identity["environment_sha256"]
                fresh.prepare_run(root, identity)
                _record_pairs(root, identity, "screen", protocol)
                targets = {"source": control, "model": model,
                           "helper": subject.ROOT / "action_smoothing.py",
                           "environment": subject.ROOT / "env_wrapper.py",
                           "runner": Path(subject.__file__),
                           "gate": Path(subject.gate.__file__)}
                target = targets[kind]
                active = False

                def fault_digest(path: Path) -> str:
                    if active and Path(path).resolve() == target.resolve():
                        return "0" * 64
                    return original_digest(path)

                def mutate_after_report(*arguments):
                    nonlocal active
                    summary = original_report(*arguments)
                    active = True
                    return summary

                with patch.object(fresh, "digest", side_effect=fault_digest), \
                     patch.object(subject, "report_phase", side_effect=mutate_after_report):
                    with self.assertRaisesRegex(ValueError, "changed|hash"):
                        subject.run_partition(root, identity, bound, paths, "screen")
                self.assertFalse((root / "screen-summary.json").exists())


if __name__ == "__main__":
    unittest.main()
