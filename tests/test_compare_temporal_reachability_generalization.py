"""Contract tests for the fresh temporal reachability comparison."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

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
        draft = fresh._read_json(wrapper.PROTOCOL_PATH)
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


if __name__ == "__main__":
    unittest.main()
