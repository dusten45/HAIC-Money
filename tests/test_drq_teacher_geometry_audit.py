from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_drq_teacher_geometry import audit_geometry


class DrQTeacherGeometryAuditTests(unittest.TestCase):
    def test_collects_partition_reserved_and_episode_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            (root / "runs" / "seed0").mkdir(parents=True)
            (root / "experiments" / "study.json").write_text(json.dumps({
                "partitions": {"screen": {"track_ids": [10], "seeds": [101, 102]}},
                "reserved_training_seeds": [201],
                "sampler_seed": 900,
            }))
            (root / "runs" / "seed0" / "episodes.jsonl").write_text(
                '{"event":"reset","track_id":1,"seed":301}\n'
                '{"event":"end","track_id":1,"seed":301}\n'
            )
            report = audit_geometry(root, [4_250_100_001])
            self.assertTrue(report["passed"])
            self.assertEqual(report["known_excluded_geometry_seeds"], [101, 102, 201, 301])

    def test_finds_exact_prior_candidate_token_across_track_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "prior.md").write_text(
                "previous geometry used: 4250100001\n"
                "not the same token: x4250100001 and 42501000010\n"
            )
            report = audit_geometry(root, [4_250_100_001])
            self.assertFalse(report["passed"])
            self.assertEqual(report["candidate_hits"]["4250100001"], ["docs/prior.md"])

    def test_rejects_malformed_jsonl_in_scanned_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            (root / "runs" / "partial.jsonl").write_text('{invalid\n')
            report = audit_geometry(root, [4_250_100_001])
            self.assertFalse(report["passed"])
            self.assertTrue(any("invalid JSONL" in error for error in report["parse_errors"]))

    def test_requires_unique_uint32_candidate_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unique"):
                audit_geometry(Path(directory), [5, 5])
            with self.assertRaisesRegex(ValueError, "uint32"):
                audit_geometry(Path(directory), [2**32])


if __name__ == "__main__":
    unittest.main()
