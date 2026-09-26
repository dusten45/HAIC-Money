from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.audit_drq_training_seeds import (
    FRESHNESS_LIMITATION,
    SeedAuditError,
    SeedCollisionError,
    audit_training_seeds,
    main,
)


class TrainingSeedAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.protocol_path = "experiments/previous-study.json"
        self.audit_path = "experiments/previous-study-geometry-audit.json"
        self.ledger_path = "runs/previous-source/episodes.jsonl"
        self.forbidden_path = self.root / "sealed" / "blind-road-and-outcomes.json"
        self.forbidden_path.parent.mkdir(parents=True)
        self.forbidden_path.write_bytes(b"forbidden blind file; never open")

        self.ledger_rows = [
            {"event": "reset", "track_id": 1, "seed": 309},
            {"event": "end", "track_id": 2, "seed": 309},
        ]
        self.audit_record = {
            "passed": True,
            "parse_errors": [],
            "global_freshness_claim": FRESHNESS_LIMITATION,
            "known_excluded_geometry_seed_count": 2,
            "known_excluded_geometry_seeds": [313, 314],
            "candidate_seeds": [312],
            "source_snapshot_count": 1,
            "source_snapshots": [
                {"path": "sealed/blind-road-and-outcomes.json", "sha256": "0" * 64},
            ],
        }
        self.protocol_record = {
            "partitions": {
                "blind": {"seeds": [305], "track_ids": [999], "repeats": 2},
                "screen": {"seeds": [306], "track_ids": [7], "repeats": 2},
            },
            "reserved_training_seeds": [307],
            "training_pools": {"teacher_training": {"seeds": [308], "track_ids": [1]}},
            "source_actors": [{
                "training_geometry_seeds": [310],
                "episodes_path": self.ledger_path,
                "episodes_sha256": "",
            }],
            "geometry_audit": {
                "candidate_seeds": [311],
                "report_path": self.audit_path,
                "report_sha256": "",
            },
        }
        self.freeze_sources()

    def _freeze(self, path: str, raw: bytes) -> str:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def freeze_sources(self) -> None:
        ledger_raw = b"".join(json.dumps(row).encode() + b"\n" for row in self.ledger_rows)
        ledger_sha = self._freeze(self.ledger_path, ledger_raw)
        prior_sha = self._freeze(self.audit_path, json.dumps(self.audit_record).encode())
        self.protocol_record["source_actors"][0]["episodes_sha256"] = ledger_sha
        self.protocol_record["geometry_audit"]["report_sha256"] = prior_sha
        protocol_sha = self._freeze(self.protocol_path, json.dumps(self.protocol_record).encode())
        self.sources = {
            "repo_root": self.root,
            "protocol_sources": {self.protocol_path: protocol_sha},
            "prior_audit_sources": {self.audit_path: prior_sha},
            "training_ledgers": {self.ledger_path: ledger_sha},
        }

    def audit(self, seeds: list[int] | None = None, **changes: object) -> dict:
        sources = dict(self.sources)
        sources.update(changes)
        return audit_training_seeds(list(range(1000, 1100)) if seeds is None else seeds, **sources)

    def test_only_declared_seed_sources_read_blind_id_is_exclusion_only(self) -> None:
        reads = []
        original = Path.read_bytes

        def guarded_read(path: Path) -> bytes:
            self.assertNotEqual(path, self.forbidden_path)
            reads.append(path.relative_to(self.root).as_posix())
            return original(path)

        with patch.object(Path, "read_bytes", guarded_read):
            report = self.audit()
        expected = [self.protocol_path, self.audit_path, self.ledger_path]
        self.assertEqual(reads, expected)
        self.assertEqual(report["read_paths"], expected)
        self.assertEqual({row["path"] for row in report["source_evidence"]}, set(expected))
        for row in report["source_evidence"]:
            self.assertEqual(row["sha256"], self.sources[
                {"protocol": "protocol_sources", "prior_seed_audit": "prior_audit_sources",
                 "training_ledger": "training_ledgers"}[row["kind"]]
            ][row["path"]])
        self.assertTrue(report["passed"])
        self.assertEqual(report["blind_data_access"], "none; protocol blind seed IDs are exclusion-only")
        self.assertEqual(report["freshness_claim"], FRESHNESS_LIMITATION)
        self.assertEqual(report["structural_blind_geometry_comparison"], "not performed")

    def test_blind_protocol_id_and_training_seed_collide_across_track_ids(self) -> None:
        with self.assertRaises(SeedCollisionError) as caught:
            self.audit(list(range(300, 400)))
        report = caught.exception.report
        self.assertFalse(report["passed"])
        self.assertEqual([row["seed"] for row in report["matched_collisions"]], list(range(305, 315)))
        self.assertEqual(report["matched_collisions"][0]["sources"], [
            {"path": self.protocol_path, "field": "partitions.blind.seeds"},
        ])
        self.assertEqual(report["matched_collisions"][4]["sources"], [
            {"path": self.ledger_path, "field": "episodes.jsonl.seed"},
        ])

    def test_proposed_duplicates_and_invalid_uint32_fail_closed(self) -> None:
        for invalid in (-1, 1 << 32, True, 2.5):
            with self.subTest(invalid=invalid), self.assertRaises(SeedAuditError):
                self.audit(list(range(1000, 1099)) + [invalid])
        with self.assertRaisesRegex(SeedAuditError, "100 unique"):
            self.audit(list(range(1000, 1099)) + [1000])
        with self.assertRaisesRegex(SeedAuditError, "100 unique"):
            self.audit([1000, 1001])

    def test_unfrozen_changed_and_missing_source_rejected(self) -> None:
        with self.assertRaises(SeedAuditError):
            self.audit(training_ledgers={self.ledger_path: ""})
        with self.assertRaises(SeedAuditError):
            self.audit(training_ledgers={"runs/absent/episodes.jsonl": "0" * 64})
        with self.assertRaises(SeedAuditError):
            self.audit(prior_audit_sources={})
        self._freeze(self.ledger_path, b'{"seed": 9999, "track_id": 1}\n')
        with self.assertRaisesRegex(SeedAuditError, "SHA-256 mismatch"):
            self.audit()

    def test_unknown_or_unapproved_paths_do_not_get_opened(self) -> None:
        original = Path.read_bytes

        def guarded_read(path: Path) -> bytes:
            if path == self.forbidden_path:
                raise AssertionError("attempted to open blind data")
            return original(path)

        with patch.object(Path, "read_bytes", guarded_read):
            for candidate in (
                "evaluations/blind-road-and-outcomes.json",
                "submissions/blind-road-and-outcomes.json",
                "runs/train/evaluations/episodes.jsonl",
                "runs/train/blind/episodes.jsonl",
                "runs/train/../sealed/episodes.jsonl",
                "/tmp/episodes.jsonl",
            ):
                with self.subTest(path=candidate), self.assertRaises(SeedAuditError):
                    self.audit(training_ledgers={candidate: "0" * 64})
            with self.assertRaises(SeedAuditError):
                self.audit(protocol_sources={"sealed/blind-road-and-outcomes.json": "0" * 64})

    def test_symlink_into_sealed_directory_rejected(self) -> None:
        (self.root / "runs" / "linked").symlink_to(self.forbidden_path.parent, target_is_directory=True)
        with self.assertRaisesRegex(SeedAuditError, "symlink"):
            self.audit(training_ledgers={"runs/linked/episodes.jsonl": "0" * 64})

    def test_opaque_protocol_and_past_audit_fail_closed(self) -> None:
        self.protocol_record["partitions"]["blind"]["road_path"] = str(self.forbidden_path)
        self.freeze_sources()
        with self.assertRaisesRegex(SeedAuditError, "opaque"):
            self.audit()
        del self.protocol_record["partitions"]["blind"]["road_path"]
        self.protocol_record["unlisted_seeds"] = [999]
        self.freeze_sources()
        with self.assertRaisesRegex(SeedAuditError, "opaque"):
            self.audit()
        del self.protocol_record["unlisted_seeds"]
        self.audit_record["passed"] = False
        self.freeze_sources()
        with self.assertRaisesRegex(SeedAuditError, "not frozen"):
            self.audit()

    def test_dreamer_learner_rng_seeds_are_not_geometry_ids(self) -> None:
        self.protocol_record["learner_seeds"] = [0, 1]
        self.freeze_sources()
        self.assertTrue(self.audit()["passed"])
        self.protocol_record["learner_seeds"] = [False]
        self.freeze_sources()
        with self.assertRaisesRegex(SeedAuditError, "learner RNG"):
            self.audit()

    def test_opaque_training_ledger_and_missing_ledger_reference_fail_closed(self) -> None:
        self.ledger_rows = [{"event": "reset", "track_id": 1, "not_a_seed": 999}]
        self.freeze_sources()
        with self.assertRaisesRegex(SeedAuditError, "opaque training"):
            self.audit()
        self.ledger_rows = [{"event": "reset", "track_id": 1, "geometry_seed": 309}]
        self.freeze_sources()
        with self.assertRaises(SeedCollisionError):
            self.audit(list(range(300, 400)))
        self.protocol_record["source_actors"][0]["episodes_sha256"] = "0" * 64
        self.sources["protocol_sources"][self.protocol_path] = self._freeze(
            self.protocol_path, json.dumps(self.protocol_record).encode()
        )
        with self.assertRaisesRegex(SeedAuditError, "actor ledger"):
            self.audit()

    def test_cli_reports_collision_and_limits_historical_claim(self) -> None:
        args = [
            "--repo-root", str(self.root),
            "--protocol", f"{self.protocol_path}={self.sources['protocol_sources'][self.protocol_path]}",
            "--prior-audit", f"{self.audit_path}={self.sources['prior_audit_sources'][self.audit_path]}",
            "--training-ledger", f"{self.ledger_path}={self.sources['training_ledgers'][self.ledger_path]}",
            "--seed-start", "1000", "--seed-count", "100",
        ]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(args), 0)
        self.assertEqual(json.loads(stdout.getvalue())["freshness_claim"], FRESHNESS_LIMITATION)
        stdout = io.StringIO()
        args[args.index("1000")] = "300"
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(args), 1)
        self.assertFalse(json.loads(stdout.getvalue())["passed"])


if __name__ == "__main__":
    unittest.main()
