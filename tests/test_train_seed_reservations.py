"""Synthetic-only TRAIN registry checks; no project ledgers or simulator imports."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from haic import train_seed_reservations as reservations


def cell(seed: int, track: int = 1, obstacles: bool = True) -> dict:
    return {"partition": "TRAIN", "track_id": track, "geometry_seed": seed,
            "obstacles": obstacles}


def clear_audit(candidates: tuple) -> dict:
    return {"format": reservations.AUDIT_FORMAT, "partition": "TRAIN", "status": "clear",
            "cells": [dict(candidate) for candidate in candidates], "collisions": [],
            "blockers": [], "consumed_seeds": [], "reserved_seeds": [],
            "source": "synthetic/candidate-inventory.json"}


class TrainSeedReservationTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / "shared-claims"
        self.root.mkdir()

    def reserve(self, cells: list[dict], audit=clear_audit, **kwargs) -> list[dict]:
        return reservations.reserve_train_seeds(self.root, cells, audit, study_id="study-A", **kwargs)

    def test_batch_records_complete_audit_evidence_and_optional_protocol(self) -> None:
        reports: list[dict] = []

        def re_audit(candidates: tuple) -> dict:
            descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(descriptor)
            with self.assertRaises(TypeError):
                candidates[0]["geometry_seed"] = 999
            report = clear_audit(candidates)
            reports.append(report)
            return report

        requested = [cell(13), cell(14, track=2, obstacles=False)]
        records = self.reserve(requested, re_audit, protocol_id="p1",
                               protocol_path="experiments/frozen-p1.json",
                               protocol_sha256="a" * 64)
        digest = hashlib.sha256(json.dumps(reports[0], sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=True).encode("ascii")).hexdigest()
        self.assertEqual(len(records), 2)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ["seed-13.json", "seed-14.json"])
        for record, requested_cell in zip(records, requested):
            with self.subTest(seed=requested_cell["geometry_seed"]):
                self.assertEqual(json.loads((self.root / f"seed-{record['geometry_seed']}.json").read_text()),
                                 record)
                self.assertEqual({key: record[key] for key in requested_cell}, requested_cell)
                self.assertEqual(record["format"], reservations.CLAIM_FORMAT)
                self.assertEqual(record["study_id"], "study-A")
                self.assertEqual(record["protocol_id"], "p1")
                self.assertEqual(record["protocol_path"], "experiments/frozen-p1.json")
                self.assertEqual(record["protocol_sha256"], "a" * 64)
                self.assertEqual(record["status"], "reserved")
                self.assertTrue(record["claimed_at_utc"].endswith("Z"))
                self.assertEqual(record["audit_source"], reports[0]["source"])
                self.assertEqual(record["audit_digest_sha256"], digest)

    def test_competing_claims_are_serialized_across_tracks(self) -> None:
        barrier = threading.Barrier(3)
        callbacks: list[int] = []

        def audit(candidates: tuple) -> dict:
            callbacks.append(candidates[0]["track_id"])
            return clear_audit(candidates)

        def attempt(track: int) -> str:
            barrier.wait(timeout=5)
            try:
                self.reserve([cell(15, track=track)], audit)
                return "claimed"
            except reservations.SeedUnavailableError:
                return "unavailable"

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(attempt, 1)
            second = executor.submit(attempt, 2)
            barrier.wait(timeout=5)
            self.assertEqual({first.result(timeout=5), second.result(timeout=5)},
                             {"claimed", "unavailable"})
        self.assertEqual(len(callbacks), 1)
        self.assertEqual([p.name for p in self.root.iterdir()], ["seed-15.json"])
        self.assertIn(json.loads((self.root / "seed-15.json").read_text())["track_id"], (1, 2))

    def test_disjoint_parallel_lanes_claim_without_global_snapshot_conflict(self) -> None:
        barrier = threading.Barrier(3)

        def attempt(seed: int) -> list[dict]:
            barrier.wait(timeout=5)
            return self.reserve([cell(seed)], clear_audit)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(attempt, 16)
            second = executor.submit(attempt, 17000)
            barrier.wait(timeout=5)
            self.assertEqual({first.result(timeout=5)[0]["geometry_seed"],
                              second.result(timeout=5)[0]["geometry_seed"]}, {16, 17000})
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ["seed-16.json", "seed-17000.json"])

    def test_callback_failure_and_invalid_report_leave_no_artifacts(self) -> None:
        def fails(_: tuple) -> dict:
            raise RuntimeError("audit source ambiguous")

        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            self.reserve([cell(1)], fails)
        self.assertEqual(list(self.root.iterdir()), [])
        for edit in (
            lambda report: report.update(status="no_known_recorded_overlap"),
            lambda report: report.update(cells=[cell(2)]),
            lambda report: report.update(collisions=[{"geometry_seed": 1}]),
            lambda report: report.update(blockers=["incomplete ledger"]),
            lambda report: report.update(consumed_seeds=[1]),
            lambda report: report.update(reserved_seeds=[1]),
            lambda report: report.pop("consumed_seeds"),
            lambda report: report.update(partition="TRAIN-DIAGNOSTIC"),
            lambda report: report.update(source=""),
        ):
            with self.subTest(edit=edit):
                def bad_audit(candidates: tuple) -> dict:
                    report = clear_audit(candidates)
                    edit(report)
                    return report

                with self.assertRaises(reservations.ReservationError):
                    self.reserve([cell(1)], bad_audit)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_duplicate_or_conflicting_batch_never_partially_writes(self) -> None:
        with self.assertRaises(reservations.ReservationError):
            self.reserve([cell(2), cell(2, track=3)])
        self.assertEqual(list(self.root.iterdir()), [])
        self.reserve([cell(2)])

        def must_not_run(_: tuple) -> dict:
            self.fail("already claimed seed must block before re-audit")

        with self.assertRaises(reservations.SeedUnavailableError):
            self.reserve([cell(1), cell(2, track=9)], must_not_run)
        self.assertFalse((self.root / "seed-1.json").exists())

    def test_candidate_became_claimed_while_callback_ran(self) -> None:
        record = self.reserve_record(3)

        def writes_during_audit(candidates: tuple) -> dict:
            self.assertEqual(candidates[0]["geometry_seed"], 3)
            (self.root / "seed-3.json").write_text(json.dumps(record))
            return clear_audit(candidates)

        with self.assertRaises(reservations.SeedUnavailableError):
            self.reserve([cell(3)], writes_during_audit)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["seed-3.json"])

    def reserve_record(self, seed: int) -> dict:
        self.reserve([cell(seed)])
        record_path = self.root / f"seed-{seed}.json"
        record = json.loads(record_path.read_text())
        record_path.unlink()  # Fixture construction only; production never removes claims.
        return record

    def test_existing_consumed_and_retired_statuses_do_not_authorize_reuse(self) -> None:
        record = self.reserve_record(4)
        for status in ("reserved", "consumed", "retired-unobserved"):
            with self.subTest(status=status):
                record["status"] = status
                path = self.root / "seed-4.json"
                path.write_text(json.dumps(record))
                with self.assertRaises(reservations.SeedUnavailableError):
                    self.reserve([cell(4, track=3, obstacles=False)])
                path.unlink()  # Fixture reset only.

    def test_malformed_records_and_symlinks_fail_closed(self) -> None:
        claim = self.reserve_record(5)
        path = self.root / "seed-5.json"
        bad_records = [
            b"{",
            b'{"format":"v1","format":"v2"}',
            json.dumps({**claim, "geometry_seed": 6}).encode(),
            json.dumps({**claim, "status": "released"}).encode(),
        ]
        for raw in bad_records:
            with self.subTest(raw=raw):
                path.write_bytes(raw)
                with self.assertRaises(reservations.ReservationError):
                    self.reserve([cell(10)])
                self.assertFalse((self.root / "seed-10.json").exists())
        path.unlink()
        outside = self.base / "external.json"
        outside.write_text(json.dumps(claim))
        path.symlink_to(outside)
        with self.assertRaises(reservations.ReservationError):
            self.reserve([cell(10)])
        path.unlink()
        alias = self.base / "registry-alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(reservations.ReservationError):
            reservations.reserve_train_seeds(alias, [cell(10)], clear_audit, study_id="study-A")
        nested = alias / "nested"
        with self.assertRaises(reservations.ReservationError):
            reservations.reserve_train_seeds(nested, [cell(10)], clear_audit, study_id="study-A")
        (self.root / "unknown.json").write_text("{}")
        with self.assertRaises(reservations.ReservationError):
            self.reserve([cell(10)])

    def test_io_failure_after_first_claim_keeps_partial_evidence(self) -> None:
        original = os.open

        def fail_second(path: str | Path, flags: int, *args, **kwargs) -> int:
            if path == "seed-12.json" and flags & os.O_EXCL:
                raise OSError("disk unavailable")
            return original(path, flags, *args, **kwargs)

        with patch.object(reservations.os, "open", side_effect=fail_second):
            with self.assertRaises(reservations.PartialBatchError) as caught:
                self.reserve([cell(11), cell(12)])
        self.assertEqual(caught.exception.created_seeds, (11,))
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["seed-11.json"])
        with self.assertRaises(reservations.SeedUnavailableError):
            self.reserve([cell(11, track=2)])

    def test_input_types_and_only_train_restriction(self) -> None:
        for invalid in (cell(True), cell(-1), {**cell(1), "partition": "CONFIRMATION"},
                        {**cell(1), "obstacles": 1}, {**cell(1), "track_id": True}):
            with self.subTest(invalid=invalid):
                with self.assertRaises(reservations.ReservationError):
                    self.reserve([invalid])
        with self.assertRaises(reservations.ReservationError):
            self.reserve([cell(1)], protocol_sha256="a" * 64)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
