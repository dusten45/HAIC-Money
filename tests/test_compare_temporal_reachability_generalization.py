"""Contract tests for the fresh temporal reachability comparison."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import evaluate_bare_generalization as fresh


def subject():
    return importlib.import_module("tools.compare_temporal_reachability_generalization")


def registered_protocol() -> dict:
    protocol = copy.deepcopy(json.loads(subject().PROTOCOL_PATH.read_text(encoding="utf-8")))
    protocol["status"] = "PREREGISTERED"
    protocol["control_agent_sha256"] = "a" * 64
    protocol["candidate_agent_sha256"] = "b" * 64
    protocol["controller_classes"]["candidate"] = "_TemporalReachabilityController"
    protocol["source_construction"]["implementation_commit"] = "1" * 40
    protocol["source_construction"]["agent_blob_sha256"] = "c" * 64
    protocol["source_construction"]["selector_to"] = "_TemporalReachabilityController()"
    protocol["source_construction"]["agent_blob_sha256"] = protocol["control_agent_sha256"]
    return protocol


def row(track: int, seed: int, arm: str, *, finished: bool = False,
        progress: float = .6, lap_time_ms: int | None = None,
        contacts: int = 0, damage: float = 0.0,
        retire_reason: str | None = "off_track") -> dict:
    return {"track_id": track, "seed": seed, "repeat": 0, "arm": arm,
            "finished": finished, "progress": progress, "lap_time_ms": lap_time_ms,
            "collision_count": contacts, "damage": damage, "retire_reason": retire_reason,
            "offtrack_samples": 0, "partial_offtrack_samples": 0,
            "initialization_ms": 100, "reset_ms": 1, "action_latency_max_ms": 2,
            "peak_worker_rss_mib": 230, "error": None}


class ProtocolTests(unittest.TestCase):
    def test_draft_and_pending_source_cannot_start(self):
        wrapper = subject()
        draft = registered_protocol()
        draft["status"] = "DRAFT_UNFROZEN"
        with self.assertRaisesRegex(ValueError, "PREREGISTERED"):
            wrapper.validate_protocol(draft, set())
        registered = registered_protocol()
        registered["candidate_agent_sha256"] = "PENDING_SOURCE_SHA256"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            wrapper.validate_protocol(registered, set())

    def test_fixed_fresh_seeds_budget_and_cross_track_holdout(self):
        wrapper = subject()
        protocol = registered_protocol()
        wrapper.validate_protocol(protocol, set())
        self.assertEqual([len(protocol["partitions"][p]["seeds"]) for p in fresh.PHASES],
                         [8, 16, 8])
        self.assertEqual([len(fresh.expected_cells(protocol, p)) for p in fresh.PHASES],
                         [34, 68, 34])
        reused = protocol["partitions"]["screen"]["seeds"][0]
        with self.assertRaisesRegex(ValueError, "historical"):
            wrapper.validate_protocol(protocol, {reused})
        protocol["partitions"]["screen"]["track_ids"] = [1]
        with self.assertRaisesRegex(ValueError, "track|spot"):
            wrapper.validate_protocol(protocol, set())

    def test_threshold_and_seed_derivation_are_not_editable_after_registration(self):
        wrapper = subject()
        protocol = registered_protocol()
        protocol["decision_thresholds"]["maximum_both_dnf_progress_loss"] = .1
        with self.assertRaisesRegex(ValueError, "threshold"):
            wrapper.validate_protocol(protocol, set())
        protocol = registered_protocol()
        protocol["partitions"]["blind"]["seeds"][0] += 1
        with self.assertRaisesRegex(ValueError, "derived|spot"):
            wrapper.validate_protocol(protocol, set())

    def test_exact_output_and_protocol_paths(self):
        wrapper = subject()
        self.assertEqual(wrapper.validate_protocol_path(wrapper.PROTOCOL_PATH),
                         wrapper.PROTOCOL_PATH.resolve())
        self.assertEqual(wrapper.validate_output_root(wrapper.OUTPUT_ROOT),
                         wrapper.OUTPUT_ROOT.resolve())
        with self.assertRaisesRegex(ValueError, "protocol path"):
            wrapper.validate_protocol_path(wrapper.PROTOCOL_PATH.with_name("other.json"))
        with self.assertRaisesRegex(ValueError, "output root"):
            wrapper.validate_output_root(wrapper.OUTPUT_ROOT.parent / "other")

    def test_source_pair_must_differ_only_in_agent_selector(self):
        wrapper = subject()
        protocol = registered_protocol()
        original = b"x = _CompoundClearingBrakeCarryController()\n"
        selected = b"x = _TemporalReachabilityController()\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control, candidate = root / "control.py", root / "candidate.py"
            control.write_bytes(original)
            candidate.write_bytes(selected)
            protocol["control_agent_sha256"] = hashlib.sha256(original).hexdigest()
            protocol["candidate_agent_sha256"] = hashlib.sha256(selected).hexdigest()
            protocol["source_construction"]["agent_blob_sha256"] = protocol["control_agent_sha256"]
            wrapper.validate_source_pair(protocol, control, candidate)
            candidate.write_bytes(selected + b"# extra change\n")
            with self.assertRaisesRegex(ValueError, "sole|selector"):
                wrapper.validate_source_pair(protocol, control, candidate)


class DecisionTests(unittest.TestCase):
    def test_finish_gain_with_offsetting_contacts_and_damage_is_allowed(self):
        wrapper = subject()
        cells = [(1, 901, 0), (2, 901, 0)]
        rows = [row(1, 901, "control", progress=.6, contacts=2, damage=.4),
                row(1, 901, "candidate", finished=True, progress=1,
                    lap_time_ms=20000, contacts=3, damage=.6, retire_reason=None),
                row(2, 901, "control", progress=.7, contacts=3, damage=.6),
                row(2, 901, "candidate", progress=.7, contacts=1, damage=.2)]
        result = wrapper.compare_pairs(rows, cells)
        self.assertEqual(result["decision"], "RETAIN")
        self.assertEqual((result["control_contacts"], result["candidate_contacts"]), (5, 4))
        self.assertEqual(len(result["paired_cells"]), 2)

    def test_finish_loss_new_crash_and_severe_dnf_loss_veto(self):
        wrapper = subject()
        cells = [(1, 901, 0)]
        control = row(1, 901, "control", finished=True, progress=1,
                      lap_time_ms=20000, retire_reason=None)
        candidate = row(1, 901, "candidate", progress=.8)
        self.assertEqual(wrapper.compare_pairs([control, candidate], cells)["decision"], "REJECT")
        control = row(1, 901, "control", progress=.7, retire_reason="off_track")
        candidate = row(1, 901, "candidate", progress=.9, retire_reason="crash")
        self.assertIn("new crash", " ".join(wrapper.compare_pairs([control, candidate], cells)["reasons"]))
        candidate["retire_reason"] = "off_track"
        candidate["progress"] = .649
        self.assertIn("DNF progress", " ".join(wrapper.compare_pairs([control, candidate], cells)["reasons"]))

    def test_aggregate_contacts_damage_and_runtime_veto(self):
        wrapper = subject()
        cells = [(1, 901, 0)]
        control = row(1, 901, "control", progress=.4)
        candidate = row(1, 901, "candidate", progress=.8, contacts=1)
        self.assertIn("aggregate contacts", " ".join(wrapper.compare_pairs([control, candidate], cells)["reasons"]))
        candidate["collision_count"] = 0
        candidate["damage"] = .2
        self.assertIn("aggregate damage", " ".join(wrapper.compare_pairs([control, candidate], cells)["reasons"]))
        candidate["damage"] = 0
        candidate["action_latency_max_ms"] = 5001
        self.assertIn("runtime", " ".join(wrapper.compare_pairs([control, candidate], cells)["reasons"]))

    def test_completed_lap_aggregate_and_single_regression_caps(self):
        wrapper = subject()
        cells = [(1, 901, 0), (2, 901, 0)]
        base = [row(track, 901, "control", finished=True, progress=1,
                    lap_time_ms=1000, retire_reason=None) for track in (1, 2)]
        candidates = [row(track, 901, "candidate", finished=True, progress=1,
                          lap_time_ms=lap, retire_reason=None)
                      for track, lap in ((1, 1080), (2, 990))]
        self.assertEqual(wrapper.compare_pairs(base + candidates, cells)["decision"], "INCONCLUSIVE")
        candidates[0]["lap_time_ms"] = 1120
        self.assertIn("single shared finish", " ".join(
            wrapper.compare_pairs(base + candidates, cells)["reasons"]))
        candidates[0]["lap_time_ms"] = 1080
        candidates[1]["lap_time_ms"] = 1040
        self.assertIn("aggregate shared finish", " ".join(
            wrapper.compare_pairs(base + candidates, cells)["reasons"]))

    def test_missing_repeat_does_not_count_as_evidence_or_unlock(self):
        wrapper = subject()
        protocol = registered_protocol()
        identity = {"protocol_sha256": "a" * 64,
                    "source_sha256": {"candidate": "b" * 64},
                    "wrapper_sha256": "c" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh.prepare_run(root, identity)
            with self.assertRaisesRegex(ValueError, "screen"):
                wrapper.freeze_finalist(root, identity, protocol)

    def test_custom_phase_seals_block_confirmation_blind_and_tampered_receipts(self):
        wrapper = subject()
        protocol = {"partitions": {
            phase: {"track_ids": [1], "seeds": [seed],
                    "spot_check_cells": [[1, seed]]}
            for phase, seed in (("screen", 901), ("confirmation", 902), ("blind", 903))}}
        identity = {"protocol_sha256": "a" * 64,
                    "source_sha256": {"candidate": "b" * 64},
                    "wrapper_sha256": "c" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh.prepare_run(root, identity)
            with self.assertRaisesRegex(ValueError, "finalist"):
                wrapper.require_phase(root, identity, protocol, "confirmation")
            for phase in ("screen", "confirmation"):
                seed = protocol["partitions"][phase]["seeds"][0]
                for repeat in (0, 1):
                    for arm in fresh.ARMS:
                        measured = row(1, seed, arm, finished=arm == "candidate",
                                       progress=1.0 if arm == "candidate" else .5,
                                       lap_time_ms=20000 if arm == "candidate" else None,
                                       retire_reason=None if arm == "candidate" else "off_track")
                        measured.update({"partition": phase, "repeat": repeat,
                                         "action_trace_sha256": ("1" if arm == "control" else "2") * 64})
                        fresh.record_cell(root, identity, measured)
                if phase == "screen":
                    self.assertEqual(wrapper.report_phase(root, identity, protocol, phase)["decision"], "RETAIN")
                    wrapper.freeze_finalist(root, identity, protocol)
                    wrapper.require_phase(root, identity, protocol, "confirmation")
                    with self.assertRaisesRegex(ValueError, "confirmation"):
                        wrapper.require_phase(root, identity, protocol, "blind")
            wrapper.seal_confirmation(root, identity, protocol)
            wrapper.require_phase(root, identity, protocol, "blind")
            receipt = fresh.cell_path(root, "screen", "candidate", 1, 901, 0)
            envelope = fresh._read_json(receipt)
            envelope["row"]["progress"] = .99
            payload = {"identity": envelope["identity"], "row": envelope["row"]}
            envelope["digest"] = hashlib.sha256(fresh._canonical(payload)).hexdigest()
            fresh._atomic_json(receipt, envelope)
            with self.assertRaisesRegex(ValueError, "seal"):
                wrapper.require_phase(root, identity, protocol, "confirmation")


class ParallelColdWorkerTests(unittest.TestCase):
    def test_worker_limit_rejects_zero_and_more_than_three(self):
        wrapper = subject()
        for count in (0, 4, -1):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "workers"):
                wrapper.validate_worker_count(count)
        for count in (1, 2, 3):
            self.assertEqual(wrapper.validate_worker_count(count), count)

    def test_failed_worker_still_rechecks_frozen_inputs(self):
        wrapper = subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {"control": root / "control.py", "model": root / "model.pt"}
            protocol = {"controller_classes": {"control": "Control"}}
            checks = []
            with patch.object(wrapper, "check_frozen_inputs",
                              side_effect=lambda *_: checks.append(1)), \
                 patch.object(wrapper.subprocess, "run",
                              side_effect=subprocess.CalledProcessError(1, "synthetic",
                                                                       stderr="failed")):
                measured = wrapper._run_cold_episode({}, protocol, paths, "control", 1, 901)
            self.assertEqual(len(checks), 2)
            self.assertIn("failed", measured["stderr"])

    def test_parallel_worker_bound_canonical_receipts_and_resume(self):
        wrapper = subject()
        protocol = {"partitions": {"screen": {"track_ids": [1],
                                                "seeds": [901, 902, 903],
                                                "spot_check_cells": []}},
                    "controller_classes": {"control": "Control", "candidate": "Candidate"}}
        identity = {"protocol_sha256": "a" * 64,
                    "source_sha256": {"candidate": "b" * 64},
                    "wrapper_sha256": "c" * 64}
        guard = threading.Lock()
        rendezvous = threading.Barrier(2)
        active = 0
        peak = 0
        launched = 0
        freezes = []
        recorded = []

        def fake_worker(command, **_kwargs):
            nonlocal active, peak, launched
            payload = json.loads(command[-1])
            arm = Path(payload["source"]).stem
            with guard:
                ordinal = launched
                launched += 1
                active += 1
                peak = max(peak, active)
            if ordinal < 2:
                rendezvous.wait(timeout=1)
            time.sleep(.005 if arm == "candidate" else .015)
            with guard:
                active -= 1
            measured = row(1, payload["seed"], arm, progress=.7 if arm == "candidate" else .5)
            measured["action_trace_sha256"] = ("1" if arm == "control" else "2") * 64
            return SimpleNamespace(stdout=json.dumps({key: value for key, value in measured.items()
                                                      if key not in ("track_id", "seed", "repeat", "arm")}))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {arm: root / f"{arm}.py" for arm in fresh.ARMS}
            paths["model"] = root / "model.pt"
            paths["protocol"] = root / "protocol.json"
            fresh.prepare_run(root, identity)
            original_record = fresh.record_cell

            def capture_record(root_path, frozen, measured):
                recorded.append((measured["seed"], measured["arm"]))
                return original_record(root_path, frozen, measured)

            with patch.object(wrapper, "check_frozen_inputs", side_effect=lambda *_: freezes.append(1)), \
                 patch.object(wrapper.subprocess, "run", side_effect=fake_worker), \
                 patch.object(fresh, "record_cell", side_effect=capture_record):
                summary = wrapper.run_partition(root, identity, protocol, paths, "screen", workers=2)
                again = wrapper.run_partition(root, identity, protocol, paths, "screen", workers=2)
            self.assertEqual(summary, again)
            self.assertEqual(summary["decision"], "RETAIN")
            self.assertEqual((launched, peak, len(freezes)), (6, 2, 12))
            self.assertEqual(recorded, [(seed, arm) for seed in (901, 902, 903)
                                        for arm in fresh.ARMS])

    def test_failed_parallel_worker_has_canonical_failure_receipt(self):
        wrapper = subject()
        protocol = {"partitions": {"screen": {"track_ids": [1],
                                                "seeds": [901, 902],
                                                "spot_check_cells": []}},
                    "controller_classes": {"control": "Control", "candidate": "Candidate"}}
        identity = {"protocol_sha256": "a" * 64,
                    "source_sha256": {"candidate": "b" * 64},
                    "wrapper_sha256": "c" * 64}
        def fake_worker(command, **_kwargs):
            payload = json.loads(command[-1])
            arm = Path(payload["source"]).stem
            if (arm, payload["seed"]) == ("candidate", 901):
                raise subprocess.CalledProcessError(1, command, stderr="synthetic failure")
            measured = row(1, payload["seed"], arm, progress=.7 if arm == "candidate" else .5)
            return SimpleNamespace(stdout=json.dumps({key: value for key, value in measured.items()
                                                      if key not in ("track_id", "seed", "repeat", "arm")}))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {arm: root / f"{arm}.py" for arm in fresh.ARMS}
            paths.update({"model": root / "model.pt", "protocol": root / "protocol.json"})
            fresh.prepare_run(root, identity)
            with patch.object(wrapper, "check_frozen_inputs"), \
                 patch.object(wrapper.subprocess, "run", side_effect=fake_worker):
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    wrapper.run_partition(root, identity, protocol, paths, "screen", workers=2)
            control = fresh.load_cell(root, identity, "screen", "control", 1, 901, 0)
            failed = fresh.load_cell(root, identity, "screen", "candidate", 1, 901, 0)
            self.assertIsNotNone(control)
            self.assertIn("synthetic failure", failed["stderr"])
            self.assertEqual(fresh._read_json(root / "screen-summary.json")["decision"], "REJECT")


if __name__ == "__main__":
    unittest.main()
