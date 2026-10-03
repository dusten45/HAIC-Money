"""Contracts for the next fresh bare-controller screen; no fresh episode runs here."""

from __future__ import annotations

import copy
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from tools import compare_camera_policy_generalization as subject
from tools import evaluate_bare_generalization as fresh


def bound_protocol() -> dict:
    protocol = copy.deepcopy(json.loads(subject.PROTOCOL_PATH.read_text(encoding="utf-8")))
    protocol["status"] = "PREREGISTERED"
    protocol["control_agent_sha256"] = "a" * 64
    protocol["candidate_agent_sha256"] = "b" * 64
    protocol["controller_classes"]["candidate"] = "_GuardedCameraController"
    protocol["source_construction"].update({
        "implementation_commit": "c" * 40,
        "agent_blob_sha256": protocol["control_agent_sha256"],
        "selector_to": "_GuardedCameraController()",
    })
    return protocol


def result(*, finished: bool, progress: float, trace: str) -> dict:
    return {"finished": finished, "progress": progress,
            "lap_time_ms": 20000 if finished else None,
            "retire_reason": None if finished else "off_track",
            "collision_count": 0, "damage": 0.0,
            "offtrack_samples": 0, "partial_offtrack_samples": 0,
            "initialization_ms": 10.0, "reset_ms": 1.0,
            "action_latency_max_ms": 1.0, "peak_worker_rss_mib": 220.0,
            "action_trace_sha256": trace * 64, "error": None}


def screen_rows() -> tuple[list[dict], list[tuple[int, int, int]]]:
    cells = [(track, seed, 0) for track in (1, 2, 3, 4) for seed in range(901, 909)]
    rows = []
    for track, seed, repeat in cells:
        for arm in fresh.ARMS:
            values = result(finished=False, progress=.5,
                            trace="1" if arm == "control" else "2")
            rows.append({"track_id": track, "seed": seed, "repeat": repeat,
                         "arm": arm, **values})
    for index in (0, 1, 2):
        candidate = rows[2 * index + 1]
        candidate.update(result(finished=True, progress=1.0, trace="2"))
    return rows, cells


class ProtocolTests(unittest.TestCase):
    def test_unbound_source_cannot_enter_evaluation(self):
        draft = bound_protocol()
        draft["status"] = "PREREGISTERED_PENDING_SOURCE"
        with self.assertRaisesRegex(ValueError, "frozen source"):
            subject.validate_protocol(draft, set())
        with patch("sys.argv", ["compare_camera_policy_generalization.py", "--partition", "screen"]), \
             patch.object(fresh, "_read_json", return_value=draft), \
             patch.object(fresh, "historical_geometry_seeds", return_value=set()), \
             patch.object(subject, "_run_cold_episode") as cold:
            with self.assertRaisesRegex(ValueError, "frozen source"):
                subject.main()
        cold.assert_not_called()

    def test_new_seeds_are_fixed_unique_and_not_historical(self):
        protocol = bound_protocol()
        historical = fresh.historical_geometry_seeds(subject.ROOT / "experiments",
                                                      exclude=subject.PROTOCOL_PATH)
        subject.validate_protocol(protocol, historical)
        seeds = [seed for phase in fresh.PHASES
                 for seed in protocol["partitions"][phase]["seeds"]]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(len(seeds), 32)
        self.assertFalse(set(seeds) & historical)
        old_v6 = fresh._read_json(subject.ROOT / "experiments" /
                                  "temporal-reachability-generalization-v1.json")
        old_seeds = {seed for phase in fresh.PHASES
                     for seed in old_v6["partitions"][phase]["seeds"]}
        self.assertFalse(set(seeds) & old_seeds)
        self.assertEqual([len(fresh.expected_cells(protocol, phase)) for phase in fresh.PHASES],
                         [34, 68, 34])
        self.assertEqual(protocol["partitions"]["screen"]["track_ids"], [1, 2, 3, 4])

    def test_seed_threshold_route_and_source_binding_cannot_drift(self):
        protocol = bound_protocol()
        protocol["partitions"]["screen"]["seeds"][0] += 1
        with self.assertRaisesRegex(ValueError, "spot|grid"):
            subject.validate_protocol(protocol, set())
        protocol = bound_protocol()
        protocol["decision_thresholds"]["maximum_single_shared_finish_time_ratio"] = 1.20
        with self.assertRaisesRegex(ValueError, "threshold"):
            subject.validate_protocol(protocol, set())
        self.assertEqual(subject.THRESHOLDS["maximum_aggregate_shared_finish_time_ratio"], 1.10)
        self.assertEqual(subject.THRESHOLDS["maximum_single_shared_finish_time_ratio"], 1.15)
        protocol = bound_protocol()
        protocol["source_construction"]["selector_to"] = "_OtherController()"
        with self.assertRaisesRegex(ValueError, "selector"):
            subject.validate_protocol(protocol, set())
        protocol = bound_protocol()
        protocol["controller_classes"]["control"] = "_OtherController"
        with self.assertRaisesRegex(ValueError, "route"):
            subject.validate_protocol(protocol, set())

    def test_source_pair_allows_only_the_exact_agent_selector_edit(self):
        protocol = bound_protocol()
        control_bytes = b"self.route = _CompoundClearingBrakeCarryController()\n"
        candidate_bytes = b"self.route = _GuardedCameraController()\n"
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "control.py"
            candidate = Path(temporary) / "candidate.py"
            control.write_bytes(control_bytes)
            candidate.write_bytes(candidate_bytes)
            protocol["control_agent_sha256"] = hashlib.sha256(control_bytes).hexdigest()
            protocol["candidate_agent_sha256"] = hashlib.sha256(candidate_bytes).hexdigest()
            subject.validate_source_pair(protocol, control, candidate)
            candidate.write_bytes(candidate_bytes + b"# hidden change\n")
            with self.assertRaisesRegex(ValueError, "solely"):
                subject.validate_source_pair(protocol, control, candidate)

    def test_control_snapshot_must_equal_committed_agent_blob(self):
        protocol = bound_protocol()
        source = b"class Agent: pass\n"
        protocol["source_construction"]["agent_blob_sha256"] = hashlib.sha256(source).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "control.py"
            control.write_bytes(source)
            with patch.object(subject.subprocess, "run", return_value=SimpleNamespace(stdout=source)):
                subject.verify_committed_source(protocol, control)
            with patch.object(subject.subprocess, "run", return_value=SimpleNamespace(stdout=source + b"# changed")):
                with self.assertRaisesRegex(ValueError, "committed"):
                    subject.verify_committed_source(protocol, control)


class RunnerTests(unittest.TestCase):
    def test_finish_first_tail_budgets_and_severe_progress_veto(self):
        rows, cells = screen_rows()
        self.assertEqual(subject.compare_pairs(rows, cells, "screen")["decision"], "RETAIN")
        # One loss, new crash and moderate DNF loss are tolerated when four
        # new finishes still give the required net gain of three.
        rows[6].update(result(finished=True, progress=1.0, trace="1"))
        rows[9].update(result(finished=True, progress=1.0, trace="2"))
        rows[11]["retire_reason"] = "crash"
        rows[13]["progress"] = .39
        allowed = subject.compare_pairs(rows, cells, "screen")
        self.assertEqual(allowed["decision"], "RETAIN")
        self.assertEqual((allowed["lost_control_finishes"], allowed["new_candidate_crashes"],
                          allowed["moderate_both_dnf_losses"]), (1, 1, 1))
        rows[13]["progress"] = .34
        severe = subject.compare_pairs(rows, cells, "screen")
        self.assertEqual(severe["decision"], "REJECT")
        self.assertIn("severe both-DNF", " ".join(severe["reasons"]))
        rows[13]["progress"] = .39
        rows[14].update(result(finished=True, progress=1.0, trace="1"))
        rows[17].update(result(finished=True, progress=1.0, trace="2"))
        two_losses = subject.compare_pairs(rows, cells, "screen")
        self.assertEqual(two_losses["net_finish_gain"], 3)
        self.assertIn("lost control finishes exceed", " ".join(two_losses["reasons"]))
        rows[14].update(result(finished=False, progress=.5, trace="1"))
        rows[17].update(result(finished=False, progress=.5, trace="2"))
        rows[15]["retire_reason"] = "crash"
        excess = subject.compare_pairs(rows, cells, "screen")
        self.assertIn("new candidate crashes exceed", " ".join(excess["reasons"]))

    def test_pace_and_local_contact_caps_still_bound_finish_gain(self):
        rows, cells = screen_rows()
        # Two existing shared finishes supply a meaningful paired pace test.
        for index in (3, 4):
            rows[2 * index].update(result(finished=True, progress=1.0, trace="1"))
            rows[2 * index + 1].update(result(finished=True, progress=1.0, trace="2"))
        rows[7]["lap_time_ms"] = 22800  # 14% slower
        rows[9]["lap_time_ms"] = 19000  # 5% faster, aggregate 4.5% slower
        self.assertEqual(subject.compare_pairs(rows, cells, "screen")["decision"], "RETAIN")
        rows[7]["lap_time_ms"] = 23200
        self.assertIn("single shared-finish", " ".join(
            subject.compare_pairs(rows, cells, "screen")["reasons"]))
        rows[7]["lap_time_ms"] = 22400
        rows[9]["lap_time_ms"] = 22000
        self.assertIn("aggregate shared-finish", " ".join(
            subject.compare_pairs(rows, cells, "screen")["reasons"]))
        rows[7]["lap_time_ms"] = 20000
        rows[9]["lap_time_ms"] = 20000
        rows[11]["collision_count"] = 3
        rows[12]["collision_count"] = 3  # aggregate unchanged, local spike forbidden
        self.assertIn("single-cell contact", " ".join(
            subject.compare_pairs(rows, cells, "screen")["reasons"]))

    def test_one_screen_cell_writes_cold_paired_and_repeat_receipts(self):
        protocol = {"partitions": {"screen": {"track_ids": [1], "seeds": [901, 902, 903],
                                               "spot_check_cells": [[1, 901]]}},
                    "controller_classes": {"control": "Control", "candidate": "Candidate"}}
        identity = {"protocol_sha256": "a" * 64,
                    "source_sha256": {"candidate": "b" * 64},
                    "runner_sha256": "c" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh.prepare_run(root, identity)
            calls = []

            def fake_cold(_identity, _protocol, _paths, arm, track, seed):
                calls.append((arm, track, seed))
                return result(finished=arm == "candidate",
                              progress=1.0 if arm == "candidate" else .5,
                              trace="1" if arm == "control" else "2")

            with patch.object(subject, "_run_cold_episode", side_effect=fake_cold), \
                 redirect_stdout(StringIO()):
                summary = subject.run_partition(root, identity, protocol, {}, "screen", workers=1)
            self.assertEqual(summary["decision"], "RETAIN")
            self.assertEqual(len(calls), 8)
            self.assertEqual(len(summary["paired_cells"]), 3)
            self.assertEqual(summary["candidate_finishes"], 3)
            for arm in fresh.ARMS:
                for repeat in (0, 1):
                    self.assertIsNotNone(fresh.load_cell(root, identity, "screen", arm, 1, 901, repeat))
            subject.freeze_finalist(root, identity, protocol)
            subject.require_phase(root, identity, protocol, "confirmation")
            receipt = fresh.cell_path(root, "screen", "candidate", 1, 901, 0)
            envelope = fresh._read_json(receipt)
            envelope["row"]["progress"] = .99
            envelope["digest"] = hashlib.sha256(fresh._canonical({
                "identity": identity, "row": envelope["row"]})).hexdigest()
            fresh._atomic_json(receipt, envelope)
            with self.assertRaisesRegex(ValueError, "seal no longer matches"):
                subject.require_phase(root, identity, protocol, "confirmation")

    def test_confirmation_without_screen_seal_starts_no_worker(self):
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
