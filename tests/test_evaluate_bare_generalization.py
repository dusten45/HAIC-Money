"""Contract tests for sealed, paired bare-agent evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest


def subject():
    from tools import evaluate_bare_generalization

    return evaluate_bare_generalization


def protocol(seeds=(901, 902, 903), spot=()):
    return {
        "schema_version": 1,
        "status": "PREREGISTERED",
        "name": "fresh-bare-test-v1",
        "control_agent_sha256": "a" * 64,
        "candidate_agent_sha256": "b" * 64,
        "model_sha256": "c" * 64,
        "runtime_helper_sha256": {"action_smoothing.py": "d" * 64,
                                  "action_representation.py": "e" * 64},
        "controller_classes": {"control": "Control", "candidate": "Candidate"},
        "max_steps": 2000,
        "frame_skip": 4,
        "partitions": {
            phase: {
                "track_ids": [1, 2, 3, 4],
                "seeds": [seed],
                "spot_check_cells": list(spot) if phase == "screen" else [],
            }
            for phase, seed in zip(("screen", "confirmation", "blind"), seeds)
        },
    }


class ProtocolTests(unittest.TestCase):
    def test_excludes_all_prior_geometry_even_from_other_track_id(self):
        with tempfile.TemporaryDirectory() as directory:
            experiments = Path(directory)
            (experiments / "old.json").write_text(
                json.dumps({
                    "reserved_training_seeds": [81],
                    "partitions": {"blind": {"track_ids": [9], "seeds": [82]}},
                    "development_cells": [[9, 83]],
                    "consumed_runs": [{"map": {"track_id": 4, "seed": 84}}],
                }), encoding="utf-8",
            )
            old = subject().historical_geometry_seeds(experiments)
            self.assertTrue({81, 82, 83, 84} <= old)
            for seed in (81, 82, 83, 84):
                with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "historical"):
                    subject().validate_protocol(protocol((seed, 902, 903)), old)

    def test_partition_overlap_budget_and_spot_membership_rejected(self):
        harness = subject()
        with self.assertRaisesRegex(ValueError, "overlap"):
            harness.validate_protocol(protocol((901, 901, 903)), set())
        invalid = protocol()
        invalid["max_steps"] = 1000
        with self.assertRaisesRegex(ValueError, "2000"):
            harness.validate_protocol(invalid, set())
        with self.assertRaisesRegex(ValueError, "spot"):
            harness.validate_protocol(protocol(spot=((1, 999),)), set())

    def test_action_validation_rejects_silent_noop_cases(self):
        import numpy as np

        harness = subject()
        np.testing.assert_array_equal(
            harness.validate_action([-.25, .5, 0]), np.array([-.25, .5, 0], dtype=np.float32)
        )
        for action in ((0, 1), (0, 1, 2), (float("nan"), 0, 0), (0, 1.01, 0)):
            with self.subTest(action=action), self.assertRaises(ValueError):
                harness.validate_action(action)


class ResultTests(unittest.TestCase):
    def test_resume_refuses_different_identity_or_tampered_cell(self):
        harness = subject()
        identity = {"protocol_sha256": "a" * 64, "source_sha256": {"control": "b" * 64}, "model_sha256": "c" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness.prepare_run(root, identity)
            row = {"partition": "screen", "arm": "control", "track_id": 1, "seed": 901,
                   "repeat": 0, "finished": True, "progress": 1.0, "lap_time_ms": 1000,
                   "collision_count": 0, "damage": 0.0, "action_trace_sha256": "d" * 64}
            harness.record_cell(root, identity, row)
            self.assertEqual(harness.load_cell(root, identity, "screen", "control", 1, 901, 0), row)
            self.assertFalse(harness.record_cell(root, identity, row))
            with self.assertRaisesRegex(ValueError, "identity"):
                harness.prepare_run(root, {**identity, "model_sha256": "e" * 64})
            cell_path = harness.cell_path(root, "screen", "control", 1, 901, 0)
            record = json.loads(cell_path.read_text(encoding="utf-8"))
            record["row"]["collision_count"] = 5
            cell_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest"):
                harness.load_cell(root, identity, "screen", "control", 1, 901, 0)

    def test_paired_comparator_prioritizes_finish_and_rejects_contacts(self):
        harness = subject()
        cells = [(1, 901, 0), (2, 901, 0)]
        def row(track, arm, finished, progress, lap, contacts=0):
            return {"track_id": track, "seed": 901, "repeat": 0, "arm": arm,
                    "finished": finished, "progress": progress, "lap_time_ms": lap,
                    "collision_count": contacts, "damage": 0.0,
                    "offtrack_samples": 2 if arm == "control" else 1,
                    "partial_offtrack_samples": 4 if arm == "control" else 3}
        rows = [row(1, "control", True, 1, 1000), row(1, "candidate", True, 1, 990),
                row(2, "control", False, .7, None), row(2, "candidate", True, 1, 1200)]
        result = harness.compare_pairs(rows, cells)
        self.assertEqual(result["decision"], "RETAIN")
        self.assertEqual((result["control_finishes"], result["candidate_finishes"]), (1, 2))
        self.assertEqual((result["control_offtrack_samples"], result["candidate_offtrack_samples"]), (4, 2))
        self.assertEqual((result["control_partial_offtrack_samples"], result["candidate_partial_offtrack_samples"]), (8, 6))
        rows[-1]["collision_count"] = 1
        self.assertEqual(harness.compare_pairs(rows, cells)["decision"], "REJECT")

    def test_equal_finishes_prioritize_progress_before_lap_time(self):
        harness = subject()
        cells = [(1, 901, 0), (2, 901, 0)]
        rows = [
            {"track_id": track, "seed": 901, "repeat": 0, "arm": arm,
             "finished": track == 1, "progress": 1.0 if track == 1 else progress,
             "lap_time_ms": lap if track == 1 else None, "collision_count": 0,
             "damage": 0.0, "offtrack_samples": 0, "partial_offtrack_samples": 0}
            for track in (1, 2)
            for arm, progress, lap in (("control", .6, 1000), ("candidate", .7, 1020))
        ]
        result = harness.compare_pairs(rows, cells)
        self.assertEqual(result["decision"], "RETAIN")
        self.assertEqual(result["paired_completed_time_delta_ms"], 20)
        for row in rows:
            if row["track_id"] == 2 and row["arm"] == "candidate":
                row["progress"] = .6000000015
        self.assertEqual(harness.compare_pairs(rows, cells)["decision"], "INCONCLUSIVE")

    def test_recheck_detects_helper_change(self):
        harness = subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {kind: root / f"{kind}.bin" for kind in ("control", "candidate", "model", "protocol")}
            for path in paths.values():
                path.write_bytes(b"stable")
            sha = hashlib.sha256(b"stable").hexdigest()
            identity = {"source_sha256": {"control": sha, "candidate": sha},
                        "model_sha256": sha, "protocol_sha256": sha,
                        "harness_sha256": harness.digest(Path(harness.__file__)),
                        "helper_sha256": {"action_smoothing.py": "0" * 64},
                        "environment_sha256": {}}
            with self.assertRaisesRegex(ValueError, "helper"):
                harness.check_frozen_inputs(identity, paths)

    def test_phase_gates_require_frozen_finalist_and_accepted_confirmation(self):
        harness = subject()
        identity = {"protocol_sha256": "a" * 64, "source_sha256": {"candidate": "b" * 64}}
        plan = protocol()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness.prepare_run(root, identity)
            with self.assertRaisesRegex(ValueError, "finalist"):
                harness.require_phase(root, identity, "confirmation", plan)
            with self.assertRaisesRegex(ValueError, "screen"):
                harness.freeze_finalist(root, identity, plan)
            for phase in ("screen", "confirmation"):
                for track in plan["partitions"][phase]["track_ids"]:
                    seed = plan["partitions"][phase]["seeds"][0]
                    for arm in ("control", "candidate"):
                        harness.record_cell(root, identity, {
                            "partition": phase, "arm": arm, "track_id": track, "seed": seed, "repeat": 0,
                            "finished": arm == "candidate", "progress": 1.0 if arm == "candidate" else .5,
                            "lap_time_ms": 1000 if arm == "candidate" else None,
                            "collision_count": 0, "damage": 0.0, "action_trace_sha256": "d" * 64,
                        })
            harness.freeze_finalist(root, identity, plan)
            harness.require_phase(root, identity, "confirmation", plan)
            with self.assertRaisesRegex(ValueError, "confirmation"):
                harness.require_phase(root, identity, "blind", plan)
            harness.seal_confirmation(root, identity, plan)
            harness.require_phase(root, identity, "blind", plan)
            harness.cell_path(root, "screen", "control", 1, 901, 0).unlink()
            with self.assertRaisesRegex(ValueError, "seal"):
                harness.require_phase(root, identity, "confirmation", plan)

    def test_missing_spot_check_blocks_finalist_freeze(self):
        harness = subject()
        plan = protocol(spot=[[1, 901]])
        identity = {"protocol_sha256": "a" * 64, "source_sha256": {"candidate": "b" * 64}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness.prepare_run(root, identity)
            for track in plan["partitions"]["screen"]["track_ids"]:
                for arm in ("control", "candidate"):
                    harness.record_cell(root, identity, {
                        "partition": "screen", "arm": arm, "track_id": track, "seed": 901, "repeat": 0,
                        "finished": arm == "candidate", "progress": 1.0 if arm == "candidate" else .5,
                        "lap_time_ms": 1000 if arm == "candidate" else None,
                        "collision_count": 0, "damage": 0.0, "action_trace_sha256": "d" * 64,
                    })
            self.assertEqual(harness.report_phase(root, identity, plan, "screen")["decision"], "INCOMPLETE")
            with self.assertRaisesRegex(ValueError, "screen"):
                harness.freeze_finalist(root, identity, plan)


if __name__ == "__main__":
    unittest.main()
