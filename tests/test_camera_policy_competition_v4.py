"""V4 source, protocol, and phase seals; these tests run no episodes."""

from __future__ import annotations

import copy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import compare_camera_policy_competition_v4 as subject
from tools import compare_camera_policy_generalization as v1
from tools import evaluate_bare_generalization as fresh
from tools import competition_camera_gate_v3 as historical_gate


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


def _candidate_pin_is_ready() -> bool:
    return (re.fullmatch(r"[0-9a-f]{40}", subject.CANDIDATE_BASE_COMMIT) is not None
            and re.fullmatch(r"[0-9a-f]{64}", subject.CANDIDATE_BASE_BLOB_SHA256)
            is not None)


def _gate_rows(phase: str, losses: int, net_gain: int) -> tuple[list[dict], list[tuple]]:
    """Spread losses across seeds so the fixture does not trip the cluster veto."""
    protocol = _protocol()
    cells = fresh.expected_cells(protocol, phase)
    canonical = [(track, seed) for track, seed, repeat in cells if repeat == 0]
    control_count = 40 if phase == "confirmation" else 20
    seed_count = 16 if phase == "confirmation" else 8
    lost_indices = {index * (seed_count + 1) % control_count for index in range(losses)}
    gains = set(range(control_count, control_count + losses + net_gain))
    rows = []
    for track, seed, repeat in cells:
        index = canonical.index((track, seed))
        for arm in fresh.ARMS:
            finished = (index < control_count if arm == "control" else
                        (index < control_count and index not in lost_indices) or index in gains)
            rows.append({"partition": phase, "arm": arm, "track_id": track,
                         "seed": seed, "repeat": repeat, **_result(arm, finished)})
    return rows, cells


class PracticalGateTests(unittest.TestCase):
    def test_practical_phase_boundaries_accept_small_gains_without_rescoring_v3(self):
        for phase, losses, net_gain in (("screen", 3, 2),
                                        ("confirmation", 6, 4), ("blind", 3, 2)):
            rows, cells = _gate_rows(phase, losses, net_gain)
            with self.subTest(phase=phase):
                practical = subject.compare_pairs(rows, cells, phase)
                self.assertEqual(practical["decision"], "RETAIN", practical["reasons"])
                self.assertEqual(practical["lost_control_finishes"], losses)
                self.assertEqual(practical["net_finish_gain"], net_gain)
                self.assertEqual(historical_gate.compare_pairs(rows, cells, phase)["decision"],
                                 "REJECT")

    def test_practical_gate_still_rejects_excess_loss_and_insufficient_finish_gain(self):
        for phase, losses, net_gain, reason in (
                ("screen", 4, 2, "lost control finishes"),
                ("confirmation", 7, 4, "lost control finishes"),
                ("blind", 4, 2, "lost control finishes"),
                ("screen", 0, 1, "net finish gain"),
                ("confirmation", 0, 3, "net finish gain"),
                ("blind", 0, 1, "net finish gain")):
            rows, cells = _gate_rows(phase, losses, net_gain)
            with self.subTest(phase=phase, losses=losses, net_gain=net_gain):
                summary = subject.compare_pairs(rows, cells, phase)
                self.assertEqual(summary["decision"], "REJECT")
                self.assertIn(reason, " ".join(summary["reasons"]))

    def test_combined_practical_gate_accepts_twelve_gains_but_rejects_eleven(self):
        for gains, decision in (((3, 6, 3), "RETAIN"), ((3, 5, 3), "REJECT")):
            summaries = {}
            for phase, gain in zip(fresh.PHASES, gains):
                rows, cells = _gate_rows(phase, 0, gain)
                summary = subject.compare_pairs(rows, cells, phase)
                summary.update({"protocol_sha256": "a" * 64,
                                "candidate_agent_sha256": "b" * 64,
                                "runner_sha256": "c" * 64,
                                "decision_engine_sha256": "d" * 64})
                summaries[phase] = summary
            with self.subTest(gains=gains):
                combined = subject.combined_decision(summaries)
                self.assertEqual(combined["decision"], decision, combined["reasons"])
                self.assertEqual(combined["net_finish_gain"], sum(gains))


class TemplateAndSourceTests(unittest.TestCase):
    def test_template_is_unbound_and_rejects_seed_material(self):
        template = json.loads(subject.TEMPLATE_PATH.read_text(encoding="utf-8"))
        subject.validate_template(template)
        self.assertEqual(template["name"], "camera-policy-competition-v4")
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
            protocol = root / "camera-policy-competition-v4.json"
            result = root / "camera-policy-competition-v4-result.json"
            old = root / "older-study.json"
            protocol.write_text('{"partitions":{"screen":{"seeds":[22]}}}', encoding="utf-8")
            result.write_text('{"protocol_sha256":"' + fresh.digest(protocol) +
                              '","cells":[{"seed":22}]}', encoding="utf-8")
            old.write_text('{"cells":[{"seed":11}],"geometry_seed":12}', encoding="utf-8")
            self.assertEqual(subject.historical_geometry_seeds_v4(root, protocol), {11, 12})
            result.write_text('{"protocol_sha256":"' + "0" * 64 +
                              '","cells":[{"seed":22}]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "result.*protocol"):
                subject.historical_geometry_seeds_v4(root, protocol)

    def test_all_v3_reserved_seeds_are_historical_even_if_blind_is_sealed(self):
        v3 = json.loads((subject.ROOT / "experiments" /
                         "camera-policy-competition-v3.json").read_text(encoding="utf-8"))
        reserved = {seed for phase in fresh.PHASES
                    for seed in v3["partitions"][phase]["seeds"]}
        self.assertEqual(len(reserved), 32)
        protected = subject.historical_geometry_seeds_v4(
            subject.ROOT / "experiments", subject.PROTOCOL_PATH)
        self.assertTrue(reserved <= protected)

    def test_committed_requirements_blob_is_pinned_as_runtime_provenance(self):
        blob = subject._committed_bytes("requirements.txt")
        self.assertEqual(hashlib.sha256(blob).hexdigest(), subject.REQUIREMENTS_BLOB_SHA256)
        self.assertEqual(subject._requirements_blob_sha256(), subject.REQUIREMENTS_BLOB_SHA256)

    def test_practical_gate_is_independent_and_historical_v3_stays_frozen(self):
        v3_gate = subject.ROOT / "tools" / "competition_camera_gate_v3.py"
        self.assertEqual(fresh.digest(v3_gate),
                         "0e2ea38793d980605c607e1b490eede26d32a60e223e3c70b6386a8cdb2a7a02")
        self.assertNotEqual(v3_gate.read_bytes(), Path(subject.gate.__file__).read_bytes())
        self.assertEqual(fresh.digest(Path(subject.gate.__file__)), subject.GATE_SHA256)

    def test_unbound_template_cannot_change_the_preregistered_relaxation(self):
        template = json.loads(subject.TEMPLATE_PATH.read_text(encoding="utf-8"))
        for key, value in (("profile", "strict"), ("consecutive_prior_rejections", 0),
                           ("maximum_fresh_studies", 100)):
            changed = copy.deepcopy(template)
            changed["evaluation_policy"] = {key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "template|unbound"):
                subject.validate_template(changed)

    def test_preregistered_prior_rejections_reject_changed_result_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("camera-policy-generalization-v1-result.json",
                         "camera-policy-generalization-v2-result.json",
                         "camera-policy-competition-v3-result.json"):
                (directory / name).write_bytes((subject.ROOT / "experiments" / name).read_bytes())
            proof = subject.prior_rejection_evidence(directory)
            self.assertEqual(len(proof), 3)
            self.assertTrue(all(row["decision"] == "REJECT" for row in proof))
            (directory / "camera-policy-generalization-v1-result.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "prior rejection|historical"):
                subject.prior_rejection_evidence(directory)

    def test_committed_source_blobs_are_distinct_and_selector_only(self):
        if not _candidate_pin_is_ready():
            self.skipTest("V4 candidate implementation commit/blob are pending development audit")
        control, candidate_base = subject._committed_agent_sources()
        expected_control = subprocess.run(
            ["git", "show", f"{subject.CONTROL_IMPLEMENTATION_COMMIT}:agent.py"],
            cwd=subject.ROOT, capture_output=True, check=True).stdout
        expected_base = subprocess.run(
            ["git", "show", f"{subject.CANDIDATE_BASE_COMMIT}:agent.py"],
            cwd=subject.ROOT, capture_output=True, check=True).stdout
        self.assertEqual(control, expected_control)
        self.assertEqual(candidate_base, expected_base)
        self.assertEqual(hashlib.sha256(control).hexdigest(), subject.CONTROL_BLOB_SHA256)
        self.assertEqual(hashlib.sha256(candidate_base).hexdigest(),
                         subject.CANDIDATE_BASE_BLOB_SHA256)
        self.assertNotEqual(control, candidate_base)
        candidate_declaration = f"class {subject.CANDIDATE_CLASS}".encode("ascii")
        self.assertNotIn(candidate_declaration, control)
        self.assertIn(candidate_declaration, candidate_base)
        selected = subject.selected_source(candidate_base)
        self.assertEqual(selected, candidate_base.replace(
            b"_CompoundClearingBrakeCarryController()",
            f"{subject.CANDIDATE_CLASS}()".encode("ascii"), 1))
        self.assertNotEqual(selected, candidate_base)

    def test_wrong_commit_class_or_blob_fails_before_salt_generation(self):
        template = json.loads(subject.TEMPLATE_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            arguments = (template, base / "model.pt", base / "protocol.json",
                         base / "control.py", base / "candidate.py")
            for field, value in (("CONTROL_IMPLEMENTATION_COMMIT", "0" * 40),
                                 ("CANDIDATE_BASE_COMMIT", "0" * 40),
                                 ("CANDIDATE_CLASS", "_OtherController")):
                with self.subTest(field=field), patch.object(subject, field, value), \
                     patch.object(subject.secrets, "token_hex") as salt:
                    with self.assertRaisesRegex(ValueError, "template|unbound"):
                        subject.bind_source(*arguments)
                    salt.assert_not_called()
            with patch.object(subject, "_require_committed_runtime"), \
                 patch.object(subject, "_committed_agent_sources",
                              side_effect=ValueError("candidate blob mismatch")), \
                 patch.object(subject.secrets, "token_hex") as salt:
                with self.assertRaisesRegex(ValueError, "blob"):
                    subject.bind_source(*arguments)
                salt.assert_not_called()

    def test_two_committed_sources_require_expected_hash_selector_and_class(self):
        control = subject._committed_bytes("agent.py", subject.CONTROL_IMPLEMENTATION_COMMIT)
        candidate_base = (f"class {subject.CANDIDATE_CLASS}(_BoundedSideHoldController):\n"
                          "    pass\nroute = _CompoundClearingBrakeCarryController()\n").encode("ascii")
        candidate_hash = hashlib.sha256(candidate_base).hexdigest()
        def source(path: str, commit: str = "HEAD") -> bytes:
            return control if commit == subject.CONTROL_IMPLEMENTATION_COMMIT else candidate_base
        with patch.object(subject, "CANDIDATE_BASE_COMMIT", "a" * 40), \
             patch.object(subject, "CANDIDATE_BASE_BLOB_SHA256", candidate_hash), \
             patch.object(subject, "_committed_bytes", side_effect=source):
            self.assertEqual(subject._committed_agent_sources(), (control, candidate_base))
            with patch.object(subject, "CANDIDATE_BASE_BLOB_SHA256", "0" * 64):
                with self.assertRaisesRegex(ValueError, "candidate base blob"):
                    subject._committed_agent_sources()
            with patch.object(subject, "CANDIDATE_CLASS", "_OtherController"):
                with self.assertRaisesRegex(ValueError, "candidate base blob|class"):
                    subject._committed_agent_sources()
            wrong_parent = candidate_base.replace(b"(_BoundedSideHoldController)", b"(object)")
            with patch.object(subject, "CANDIDATE_BASE_BLOB_SHA256",
                              hashlib.sha256(wrong_parent).hexdigest()), \
                 patch.object(subject, "_committed_bytes",
                              side_effect=lambda path, commit="HEAD":
                              control if commit == subject.CONTROL_IMPLEMENTATION_COMMIT
                              else wrong_parent):
                with self.assertRaisesRegex(ValueError, "class"):
                    subject._committed_agent_sources()
            with patch.object(subject, "_committed_bytes",
                              side_effect=lambda path, commit="HEAD":
                              control if commit == subject.CONTROL_IMPLEMENTATION_COMMIT
                              else candidate_base.replace(
                                  b"_CompoundClearingBrakeCarryController()", b"not_the_route()")):
                with self.assertRaisesRegex(ValueError, "candidate base blob|selector"):
                    subject._committed_agent_sources()

    def test_bound_protocol_pins_both_commits_and_selected_candidate_bytes(self):
        control = subject._committed_bytes("agent.py", subject.CONTROL_IMPLEMENTATION_COMMIT)
        candidate_base = (f"class {subject.CANDIDATE_CLASS}: pass\n"
                          "route = _CompoundClearingBrakeCarryController()\n").encode("ascii")
        selected = candidate_base.replace(
            b"_CompoundClearingBrakeCarryController()",
            f"{subject.CANDIDATE_CLASS}()".encode("ascii"))
        salt = "0" * 32
        seeds = subject.derived_seed_partitions(salt)
        protocol = {
            "schema_version": 1, "status": "PREREGISTERED", "name": subject.NAME,
            "seed_salt_hex": salt,
            "control_agent_sha256": hashlib.sha256(control).hexdigest(),
            "candidate_agent_sha256": hashlib.sha256(selected).hexdigest(),
            "model_sha256": "a" * 64,
            "runtime_helper_sha256": {"action_smoothing.py": "b" * 64,
                                      "action_representation.py": "c" * 64},
            "controller_classes": {"control": subject.CONTROL_CLASS,
                                   "candidate": subject.CANDIDATE_CLASS},
            "source_snapshots": subject.SOURCE_PATHS,
            "source_construction": {
                "control_implementation_commit": subject.CONTROL_IMPLEMENTATION_COMMIT,
                "control_blob_sha256": subject.CONTROL_BLOB_SHA256,
                "candidate_base_commit": subject.CANDIDATE_BASE_COMMIT,
                "candidate_base_blob_sha256": subject.CANDIDATE_BASE_BLOB_SHA256,
                "selector_from": f"{subject.CONTROL_CLASS}()",
                "selector_to": f"{subject.CANDIDATE_CLASS}()"},
            "training": False, "max_steps": 2000, "frame_skip": 4,
            "official_participants_commit": subject.PARTICIPANTS_COMMIT,
            "runtime_provenance": subject.RUNTIME_PROVENANCE,
            "requirements_blob_sha256": subject.REQUIREMENTS_BLOB_SHA256,
            "decision_thresholds": subject.THRESHOLDS,
            "evaluation_policy": subject.EVALUATION_POLICY,
            "prior_rejection_evidence": subject.prior_rejection_evidence(
                subject.ROOT / "experiments"),
            "template_sha256": fresh.digest(subject.TEMPLATE_PATH),
            "decision_engine_sha256": fresh.digest(Path(subject.gate.__file__)),
            "runner_sha256": fresh.digest(Path(subject.__file__)),
            "evaluation_harness_sha256": fresh.digest(Path(fresh.__file__)),
            "environment_sha256": {},
            "partitions": {phase: {"track_ids": subject.TRACKS,
                                   "seeds": seeds[phase],
                                   "spot_check_cells": [
                                       [track, seeds[phase][index]]
                                       for track, index in subject.SPOT_INDEX[phase]]}
                           for phase in fresh.PHASES},
        }
        with patch.object(subject, "_committed_agent_sources",
                          return_value=(control, candidate_base)), \
             patch.object(subject, "_environment_hashes", return_value={}):
            subject.validate_protocol(protocol, set())
            for field, value in (("candidate_base_commit", "f" * 40),
                                 ("candidate_base_blob_sha256", "f" * 64),
                                 ("control_blob_sha256", "f" * 64)):
                altered = copy.deepcopy(protocol)
                altered["source_construction"][field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "source"):
                    subject.validate_protocol(altered, set())
            altered = copy.deepcopy(protocol)
            altered["candidate_agent_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "source hashes"):
                subject.validate_protocol(altered, set())
            altered = copy.deepcopy(protocol)
            altered["requirements_blob_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "requirements"):
                subject.validate_protocol(altered, set())
            altered = copy.deepcopy(protocol)
            altered["evaluation_policy"]["fallback"]["minimum_combined_net_finish_gain"] = 0
            with self.assertRaisesRegex(ValueError, "practical evaluation policy"):
                subject.validate_protocol(altered, set())

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
        if not _candidate_pin_is_ready():
            self.skipTest("V4 candidate implementation commit/blob are pending development audit")
        control_source, candidate_base = subject._committed_agent_sources()
        selected = subject.selected_source(candidate_base)
        protocol = {"control_agent_sha256": hashlib.sha256(control_source).hexdigest(),
                    "candidate_agent_sha256": hashlib.sha256(selected).hexdigest(),
                    "source_construction": {"selector_from": f"{subject.CONTROL_CLASS}()",
                                            "selector_to": f"{subject.CANDIDATE_CLASS}()",
                                            "candidate_base_blob_sha256":
                                            hashlib.sha256(candidate_base).hexdigest()}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control, candidate = root / "control.py", root / "candidate.py"
            with patch.object(subject.secrets, "token_hex") as salt:
                subject.restore_snapshots(protocol, control, candidate)
                salt.assert_not_called()
            self.assertEqual(control.read_bytes(), control_source)
            self.assertEqual(candidate.read_bytes(), selected)
            control.unlink()
            subject.restore_snapshots(protocol, control, candidate)
            self.assertEqual(control.read_bytes(), control_source)
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
            with patch.object(sys, "argv", ["v4", "--model", str(Path(temporary) / "other.pt"),
                                           "preflight"]), \
                 patch.object(subject, "_run_cold_episode") as cold:
                with self.assertRaisesRegex(ValueError, "root model"):
                    subject.main()
                cold.assert_not_called()

    def test_running_runtime_versions_are_checked_before_commands(self):
        subject.require_runtime_versions()
        installed_version = metadata.version
        with patch.object(metadata, "version",
                          side_effect=lambda name: "2.1.1" if name == "torch"
                          else installed_version(name)):
            with self.assertRaisesRegex(ValueError, "imported Torch"):
                subject._runtime_versions()
        good = ("3.11", "2.1.0", "1.26.0", "0.29.1", "4.8.1.78", "2.3.10")
        bad = ((0, "3.10"), (1, "2.1.1"), (2, "1.26.4"),
               (3, "0.30.0"), (4, "4.9.0.0"), (5, "2.3.11"))
        for index, replacement in bad:
            versions = (*good[:index], replacement, *good[index + 1:])
            with self.subTest(versions=versions), \
                 patch.object(subject, "_runtime_versions", return_value=versions):
                with self.assertRaisesRegex(
                        ValueError, "Python 3.11|Torch 2.1.0|NumPy 1.26.0|Gymnasium 0.29.1|OpenCV 4.8.1.78|Box2D 2.3.10"):
                    subject.require_runtime_versions()
                with patch.object(sys, "argv", ["v4", "preflight"]), \
                     patch.object(subject, "_run_cold_episode") as cold:
                    with self.assertRaisesRegex(
                            ValueError, "Python 3.11|Torch 2.1.0|NumPy 1.26.0|Gymnasium 0.29.1|OpenCV 4.8.1.78|Box2D 2.3.10"):
                        subject.main()
                    cold.assert_not_called()

    def test_restore_snapshots_is_a_recognized_cli_mode(self):
        with patch.object(sys, "argv", ["v4", "restore-snapshots"]), \
             patch.object(subject, "_runtime_versions", return_value=("3.11", "0.0", "0.0", "0.0", "0.0", "0.0")), \
             patch.object(subject, "_run_cold_episode") as cold:
            with self.assertRaisesRegex(ValueError, "unbound v4 protocol"):
                subject.main()
            cold.assert_not_called()

    def test_source_pair_rejects_even_one_extra_candidate_byte(self):
        source = b"control source\nroute = _CompoundClearingBrakeCarryController()\n"
        candidate_base = (f"class {subject.CANDIDATE_CLASS}: pass\n"
                          "route = _CompoundClearingBrakeCarryController()\n").encode("ascii")
        selected = candidate_base.replace(b"_CompoundClearingBrakeCarryController()",
                                          f"{subject.CANDIDATE_CLASS}()".encode("ascii"))
        protocol = {"source_construction": {"selector_from": "_CompoundClearingBrakeCarryController()",
                                            "selector_to": f"{subject.CANDIDATE_CLASS}()",
                                            "candidate_base_blob_sha256": hashlib.sha256(candidate_base).hexdigest()},
                    "control_agent_sha256": hashlib.sha256(source).hexdigest(),
                    "candidate_agent_sha256": hashlib.sha256(selected).hexdigest()}
        with tempfile.TemporaryDirectory() as temporary:
            control, candidate = (Path(temporary) / name for name in ("control.py", "candidate.py"))
            control.write_bytes(source)
            candidate.write_bytes(selected)
            with patch.object(subject, "_committed_agent_sources",
                              return_value=(source, candidate_base)):
                subject.validate_source_pair(protocol, control, candidate)
            candidate.write_bytes(selected + b"\n")
            with patch.object(subject, "_committed_agent_sources",
                              return_value=(source, candidate_base)):
                with self.assertRaisesRegex(ValueError, "selector|SHA256"):
                    subject.validate_source_pair(protocol, control, candidate)
            candidate.write_bytes(source.replace(
                b"_CompoundClearingBrakeCarryController()",
                f"{subject.CANDIDATE_CLASS}()".encode("ascii")))
            with patch.object(subject, "_committed_agent_sources",
                              return_value=(source, candidate_base)):
                with self.assertRaisesRegex(ValueError, "pinned blobs|selector swap"):
                    subject.validate_source_pair(protocol, control, candidate)
            candidate.write_bytes(selected)
            control.write_bytes(source + b"# tampered\n")
            with patch.object(subject, "_committed_agent_sources",
                              return_value=(source, candidate_base)):
                with self.assertRaisesRegex(ValueError, "pinned blobs|selector swap"):
                    subject.validate_source_pair(protocol, control, candidate)


class SealedPhaseTests(unittest.TestCase):
    def test_v1_rejected_local_contact_spike_can_unlock_v4_confirmation(self):
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

    def test_v4_rejected_aggregate_contact_increase_cannot_unlock(self):
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

    def test_receipt_inventory_rejects_alias_to_expected_file(self):
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = fresh.cell_path(root, "screen", "candidate", 1, 801, 0)
            expected.parent.mkdir(parents=True)
            expected.write_text("{}", encoding="utf-8")
            alias = expected.parent / "alias.json"
            alias.write_text("{}", encoding="utf-8")
            original_resolve = Path.resolve

            def alias_resolve(path: Path, *args, **kwargs) -> Path:
                return original_resolve(expected) if path == alias else original_resolve(path, *args, **kwargs)

            with patch.object(Path, "resolve", alias_resolve):
                extras = subject._unexpected_receipts(root, protocol, "screen")
            self.assertTrue(any("alias.json" in reason for reason in extras))

    def test_receipt_inventory_rejects_symlink_even_at_expected_path(self):
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = fresh.cell_path(root, "screen", "candidate", 1, 801, 0)
            expected.parent.mkdir(parents=True)
            expected.write_text("{}", encoding="utf-8")
            original_is_symlink = Path.is_symlink

            def marked_symlink(path: Path) -> bool:
                return path == expected or original_is_symlink(path)

            with patch.object(Path, "is_symlink", marked_symlink):
                extras = subject._unexpected_receipts(root, protocol, "screen")
            self.assertTrue(any("symlink" in reason for reason in extras))

    def test_confirmation_waits_for_v4_finalist_and_receipt_bound_seal(self):
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

    def test_unspotted_canonical_action_change_invalidates_v4_finalist_seal(self):
        protocol = _protocol()
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh.prepare_run(root, identity)
            _record_pairs(root, identity, "screen", protocol)
            first = subject.report_phase(root, identity, protocol, "screen")
            self.assertEqual(first["decision"], "RETAIN")
            subject.freeze_finalist(root, identity, protocol)
            subject.require_phase(root, identity, protocol, "confirmation")
            receipt = fresh.cell_path(root, "screen", "candidate", 1, 802, 0)
            _rewrite_row(receipt, action_trace_sha256="f" * 64)
            changed = subject.report_phase(root, identity, protocol, "screen")
            self.assertEqual(changed["decision"], "RETAIN")
            self.assertNotEqual(first["receipt_tree_sha256"], changed["receipt_tree_sha256"])
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
                source = b"control source\nroute = _CompoundClearingBrakeCarryController()\n"
                candidate_base = (f"class {subject.CANDIDATE_CLASS}: pass\n"
                                  "route = _CompoundClearingBrakeCarryController()\n").encode("ascii")
                selected = candidate_base.replace(
                    b"_CompoundClearingBrakeCarryController()",
                    f"{subject.CANDIDATE_CLASS}()".encode("ascii"))
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
                             "selector_to": f"{subject.CANDIDATE_CLASS}()",
                             "candidate_base_blob_sha256": hashlib.sha256(candidate_base).hexdigest()},
                         "runner_sha256": original_digest(Path(subject.__file__)),
                         "decision_engine_sha256": original_digest(Path(subject.gate.__file__)),
                         "template_sha256": original_digest(subject.TEMPLATE_PATH),
                         "evaluation_harness_sha256": original_digest(Path(fresh.__file__))}
                bound["requirements_blob_sha256"] = subject.REQUIREMENTS_BLOB_SHA256
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
                     patch.object(subject, "_committed_agent_sources",
                                  return_value=(source, candidate_base)), \
                     patch.object(subject, "report_phase", side_effect=mutate_after_report):
                    with self.assertRaisesRegex(ValueError, "changed|hash"):
                        subject.run_partition(root, identity, bound, paths, "screen")
                self.assertFalse((root / "screen-summary.json").exists())


if __name__ == "__main__":
    unittest.main()
