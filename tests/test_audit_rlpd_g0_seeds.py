from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.audit_rlpd_g0_seeds as audit_module
from scripts.audit_rlpd_g0_seeds import (SeedAuditBlocked, SeedAuditError,
                                         audit_g0_seeds, write_clean_receipt,
                                         R5_EPISTEMIC_LIMIT, R5_ERRATUM_PATH,
                                         REQUIRED_COLLECTIONS)


class G0SeedAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog_protocol = "experiments/drqv2-geometry-augmentation-v1.json"
        self.catalog_path = "runs/geometry-catalog/catalog.json"
        self.rlpd_path = "experiments/pixel-rlpd-v1.json"
        self.teacher_path = "experiments/drqv2-teacher-replay-v1.json"
        self.mix_path = "experiments/drqv2-geometry-mix-v1-r4.json"
        self.dreamer_path = "experiments/dreamerv3-b1-v1.json"
        self.protocols = (self.rlpd_path, self.teacher_path, self.mix_path,
                          self.dreamer_path, self.catalog_protocol)
        self.source_ledgers = tuple(
            f"runs/teacher/control-seed{i}/episodes.jsonl" for i in (0, 1)
        )
        self.partial_ledger = "runs/drq-r4/learner-0/episodes.jsonl"
        self.r6_ledger = "runs/drq-r6/learner-0/episodes.jsonl"
        self.partial_receipt = "runs/drq-r4/protocol-superseded-after-partial.json"
        self.attempt_receipt = "runs/drq-r4/learner-0/precheckpoint-abort.json"
        self.required_ledgers = (*self.source_ledgers, self.partial_ledger, self.r6_ledger)
        self.rows = {
            self.source_ledgers[0]: [{"event": "reset", "seed": 701, "track_id": 1}],
            self.source_ledgers[1]: [{"event": "reset", "seed": 702, "track_id": 2}],
            self.partial_ledger: [{"event": "reset", "seed": 801, "track_id": 1}],
            self.r6_ledger: [{"event": "reset", "seed": 901,
                              "geometry_seed": 901, "track_id": 3}],
        }
        self.rlpd = {
            "format": "haic-pixel-rlpd-study-v1",
            "reserved_training_seeds": [111], "training_geometry_seeds": [112],
            "teacher_data_cells": [{"geometry_seed": 112}],
            "partitions": self.partitions(201),
            "future_full_reservation": {
                "training_geometry_seeds": [4000005017],
                "teacher_data_cells": [{"geometry_seed": 4000005017}],
                **self.partitions(301),
            },
            "geometry_audit": {},
        }
        self.teacher = {
            "study_id": "drqv2-teacher-replay-v1",
            "reserved_training_seeds": [401],
            "known_excluded_geometry_seeds": [402],
            "geometry_audit": {"candidate_seeds": [403]},
            "partitions": self.partitions(410),
            "training_pools": {"online_training": {"seeds": [421]},
                               "teacher_training": {"seeds": [422]}},
            "source_actors": [],
        }
        self.dreamer = {
            "format": "haic-dreamerv3-study-protocol-v1",
            "reserved_training_seeds": [500],
            "training_development": {"seeds": [501], "cells": [{"seed": 501}]},
            "partitions": self.partitions(510),
        }
        self.augmentation = {
            "format": "haic-drq-training-geometry-protocol-v1",
            "candidate_seeds": list(range(3910800001, 3910800513)),
            "exclusion_seed_ids": {"reserved": [601], "heldout": [602], "blind": [603]},
        }
        self.mix = {
            "format": "haic-drq-geometry-mix-study-v1",
            "environment": {"partition": "TRAIN", "geometry_seeds": [610]},
            "diagnostic_pool": {"partition": "TRAIN-DIAGNOSTIC", "geometry_seeds": [611]},
            "training_pool": {"partition": "TRAIN", "geometry_seeds": [612]},
            "catalog": {"path": self.catalog_path},
        }
        self._freeze_all()
        self.blind = self.root / "evaluations/blind/episodes.jsonl"
        self.blind.parent.mkdir(parents=True)
        self.blind.write_text('this blind episode must not be opened\n')
        self._pin_experiments()

    def _pin_experiments(self) -> None:
        self.experiment_catalog_sha256, evidence, _ = audit_module._experiment_catalog(self.root)
        self.experiment_content_sha256 = audit_module._digest(evidence)

    @staticmethod
    def partitions(start: int) -> dict:
        return {name: {"seeds": [start + n]} for n, name in enumerate(
            ("screen", "confirmation", "blind"))}

    def _write(self, name: str, raw: bytes) -> str:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def _freeze_all(self) -> None:
        shas = {}
        for path, rows in self.rows.items():
            shas[path] = self._write(path, b"".join(json.dumps(row).encode() + b"\n" for row in rows))
        self.rlpd["geometry_audit"]["teacher_source_ledgers"] = [
            {"path": path, "sha256": shas[path], "geometry_seeds": [self.rows[path][0]["seed"]]}
            for path in self.source_ledgers
        ]
        self.teacher["source_actors"] = [
            {"episodes_path": path, "episodes_sha256": shas[path],
             "training_geometry_seeds": [self.rows[path][0]["seed"]]}
            for path in self.source_ledgers
        ]
        aug_sha = self._write(self.catalog_protocol, json.dumps(self.augmentation).encode())
        catalog = {"format": "haic-drq-training-geometry-catalog-v1",
                   "protocol_sha256": aug_sha,
                   "seed_audit": {"proposed_seeds": self.augmentation["candidate_seeds"]}}
        catalog_sha = self._write(self.catalog_path, json.dumps(catalog).encode())
        self.mix["catalog"]["sha256"] = catalog_sha
        for path, record in ((self.rlpd_path, self.rlpd), (self.teacher_path, self.teacher),
                             (self.dreamer_path, self.dreamer)):
            self._write(path, json.dumps(record).encode())
        mix_sha = self._write(self.mix_path, json.dumps(self.mix).encode())
        attempt_sha = self._write(self.attempt_receipt, json.dumps({
            "format": "haic-drq-geometry-mix-partial-arm-abort-v1",
            "protocol_path": self.mix_path, "protocol_sha256": mix_sha,
            "episodes_sha256": shas[self.partial_ledger],
            "last_geometry_seed": self.rows[self.partial_ledger][0]["seed"],
            "full_checkpoint_written": False, "actor_candidate_written": False,
        }).encode())
        self._write(self.partial_receipt, json.dumps({
            "format": "haic-drq-geometry-mix-precheckpoint-supersession-v1",
            "status": "partial-train-attempt-incomplete",
            "superseded_protocol_path": self.mix_path,
            "superseded_protocol_sha256": mix_sha,
            "attempt_receipt_path": self.attempt_receipt,
            "attempt_receipt_sha256": attempt_sha,
        }).encode())

    def audit(self, seed_start: int = 3000, **kwargs):
        params = {"repo_root": self.root,
                  "experiment_catalog_sha256": self.experiment_catalog_sha256,
                  "experiment_content_sha256": self.experiment_content_sha256,
                  "required_protocols": self.protocols,
                  "required_ledgers": self.required_ledgers, "training_roots": (),
                  "required_collections": {},
                  "catalog_protocol": self.catalog_protocol, "catalog_path": self.catalog_path,
                  "partial_receipts": (self.partial_receipt,),
                  "attempt_receipts": (self.attempt_receipt,)}
        params.update(kwargs)
        return audit_g0_seeds(seed_start, 2, **params)

    def _add_synthetic_r5(self) -> None:
        """Temp-only r5 chain; patch pinned bytes only inside synthetic tests."""
        r5_root = "runs/drq-r5"
        r5_run = f"{r5_root}/learner-0-uniform"
        protocol_path = "experiments/drqv2-geometry-mix-v1-r5.json"
        ledger_path = f"{r5_run}/episodes.jsonl"
        receipt_path = f"{r5_run}/precheckpoint-abort.json"
        supersession_path = f"{r5_root}/protocol-superseded-after-partial.json"
        metrics_path = f"{r5_run}/step-metrics.jsonl"
        trace_path = f"{r5_run}/replay-sample-trace-step-000016384.npz"
        seeds = list(range(3910800001, 3910800037))
        protocol = copy.deepcopy(self.mix)
        protocol["environment"]["geometry_seeds"] = seeds
        protocol["training_pool"]["geometry_seeds"] = seeds
        protocol_sha = self._write(protocol_path, json.dumps(protocol).encode())
        self.r5_rows = []
        for i, seed in enumerate(seeds):
            self.r5_rows.append({"event": "reset", "episode_id": i, "track_id": 1,
                                 "geometry_seed": seed, "seed": seed,
                                 "additional_online_step": 15971 if i == 35 else i * 450})
            if i < 35:
                self.r5_rows.append({"event": "end", "episode_id": i,
                                     "track_id": 1, "seed": seed})
        ledger_sha = self._write(ledger_path, b"".join(
            json.dumps(row).encode() + b"\n" for row in self.r5_rows))
        metrics_sha = self._write(metrics_path, b'{"decision":16384}\n')
        trace_sha = self._write(trace_path, b"synthetic replay trace bytes")
        invalid = "f" * 65
        receipt_sha = self._write(receipt_path, json.dumps({
            "format": "haic-drq-geometry-mix-partial-arm-abort-v1",
            "study_id": "drqv2-geometry-mix-v1-r5", "run_dir": r5_run,
            "protocol_path": protocol_path, "protocol_sha256": protocol_sha,
            "episodes_sha256": invalid, "step_metrics_sha256": metrics_sha,
            "online_replay_sample_trace_sha256": trace_sha,
            "unique_training_geometry_seeds_observed": 36,
            "last_geometry_seed": seeds[-1], "last_episode_id": 35, "last_episode_step": 412,
            "online_decisions": 16384, "learner_updates": 6384,
            "source_seed": 0, "variant": "uniform", "full_checkpoint_written": False,
            "actor_candidate_written": False, "resume_allowed": False,
            "evaluation_receipts": [],
        }).encode())
        supersession_sha = self._write(supersession_path, json.dumps({
            "format": "haic-drq-geometry-mix-precheckpoint-supersession-v1",
            "study_id": "drqv2-geometry-mix-v1-r5", "status": "partial-train-attempt-incomplete",
            "superseded_protocol_path": protocol_path, "superseded_protocol_sha256": protocol_sha,
            "attempt_receipt_path": receipt_path, "attempt_receipt_sha256": receipt_sha,
            "online_decisions": 16384, "learner_updates": 6384,
            "full_checkpoint_written": False, "resume_allowed": False,
            "evaluation_receipts": [], "diagnostic_or_heldout_access": False,
        }).encode())
        self.r5_profile = {
            "attempt_receipt_path": receipt_path, "attempt_receipt_sha256": receipt_sha,
            "supersession_path": supersession_path, "supersession_sha256": supersession_sha,
            "ledger_path": ledger_path, "ledger_sha256": ledger_sha,
            "protocol_path": protocol_path, "protocol_sha256": protocol_sha,
            "step_metrics_path": metrics_path, "step_metrics_sha256": metrics_sha,
            "trace_path": trace_path, "trace_sha256": trace_sha,
            "invalid_receipt_episodes_sha256": invalid,
            "ordered_reset_count": 36, "end_count": 35, "online_decisions": 16384,
            "learner_updates": 6384, "last_episode_id": 35, "last_episode_step": 412,
            "last_geometry_seed": seeds[-1],
        }
        self.r5_erratum = {
            "format": "haic-rlpd-g0-r5-receipt-erratum-v1",
            "candidate_seeds": list(range(3000, 3012)),
            "candidate_seeds_sha256": audit_module._digest(list(range(3000, 3012))),
            "track_id": 2, "r5": self.r5_profile,
            "original_ledger_bytes_attested": False,
            "epistemic_limit": R5_EPISTEMIC_LIMIT,
        }
        self._write(R5_ERRATUM_PATH, json.dumps(self.r5_erratum).encode())
        self.protocols += (protocol_path,)
        self.required_ledgers += (ledger_path,)
        self.partial_receipt_paths = (self.partial_receipt, supersession_path)
        self.attempt_receipt_paths = (self.attempt_receipt, receipt_path)
        self._pin_experiments()

    def _audit_with_r5(self, seed_start: int = 3000):
        with patch.object(audit_module, "R5_PROFILE", self.r5_profile):
            return self.audit(seed_start, partial_receipts=self.partial_receipt_paths,
                              attempt_receipts=self.attempt_receipt_paths,
                              r5_erratum=R5_ERRATUM_PATH)

    def _add_synthetic_collections(self) -> None:
        self.collection_paths = sorted(REQUIRED_COLLECTIONS)
        self.collection_rows = {}
        self.collection_hashes = {}
        self.collection_seeds = {}
        self.rlpd["teacher_data_cells"] = []
        self.rlpd["training_geometry_seeds"] = []
        for index, path in enumerate(self.collection_paths):
            seeds = (1100 + index * 10, 1101 + index * 10)
            self.collection_seeds[path] = seeds
            for seed in seeds:
                self.rlpd["teacher_data_cells"].append(
                    {"geometry_seed": seed, "track_id": 1, "obstacles": True})
                self.rlpd["training_geometry_seeds"].append(seed)
            self.collection_rows[path] = [
                {"event": "reset", "cell_index": 0, "episode_id": 0,
                 "geometry_seed": seeds[0], "track_id": 1, "obstacles": True},
                {"event": "stored_episode", "episode_id": 0, "geometry_seed": seeds[0],
                 "track_id": 1, "path": "episodes/episode-0000.npz", "sha256": "a" * 64,
                 "teacher_actor_sha256": "b" * 64, "finished": False,
                 "progress": 0.5, "retire_reason": "off_track", "reward": 12.0,
                 "steps": 50, "terminal": True, "terminated": True, "truncated": False},
                {"event": "reset", "cell_index": 1, "episode_id": 1,
                 "geometry_seed": seeds[1], "track_id": 1, "obstacles": True},
                {"event": "discarded_incomplete_episode", "episode_id": 1,
                 "geometry_seed": seeds[1], "track_id": 1, "decisions_spent": 17,
                 "reason": "teacher decision cap reached; no synthetic terminal inserted"},
            ]
        self._freeze_all()
        self._pin_experiments()
        for path, rows in self.collection_rows.items():
            self.collection_hashes[path] = self._write(
                path, b"".join(json.dumps(row).encode() + b"\n" for row in rows))

    def _audit_with_collections(self, seed_start: int = 3000):
        return self.audit(seed_start, required_collections={
            path: self.rlpd_path for path in self.collection_paths},
            collection_sha256=self.collection_hashes)

    def test_deterministic_twelve_cells_and_hashed_inventory_never_reads_blind(self) -> None:
        original = Path.read_bytes
        reads = []

        def read(path: Path) -> bytes:
            self.assertNotEqual(path, self.blind)
            reads.append(path.relative_to(self.root).as_posix())
            return original(path)

        with patch.object(Path, "read_bytes", read):
            report = self.audit()
        self.assertEqual(set(reads), {self.catalog_path, self.partial_receipt, self.attempt_receipt,
                                      *self.protocols, *self.required_ledgers})
        self.assertEqual(report, self.audit())
        self.assertEqual(report["candidate_seeds"], list(range(3000, 3012)))
        self.assertEqual(report["cells"], [
            {"track_id": 2, "geometry_seed": n, "partition": "TRAIN", "obstacles": True}
            for n in range(3000, 3012)])
        self.assertEqual(report["status"], "no_known_recorded_overlap")
        self.assertIn("historical pilot schedules are incomplete", report["freshness_claim"])
        self.assertFalse(report["protocol_frozen"])
        self.assertEqual(len(report["source_inventory_sha256"]), 64)
        self.assertEqual({row["path"] for row in report["experiment_content_inventory"]},
                         set(self.protocols))
        self.assertEqual(report["experiment_content_inventory_sha256"],
                         audit_module._digest(report["experiment_content_inventory"]))
        self.assertEqual(report["excluded_seed_count"] >= 512, True)

    def test_full_drq_catalog_reserves_ids_beyond_selected_roads(self) -> None:
        with self.assertRaises(SeedAuditBlocked) as ctx:
            self.audit(3910800501)
        report = ctx.exception.report
        self.assertEqual([hit["seed"] for hit in report["collisions"]],
                         list(range(3910800501, 3910800513)))
        self.assertIn({"path": self.catalog_protocol, "field": "candidate_seeds"},
                      report["collisions"][0]["sources"])
        self.assertEqual(report["status"], "blocked")

    def test_retired_future_pool_and_blind_protocol_ids_exclude_across_track_ids(self) -> None:
        for seed, expected_field in ((4000005017, "future_full_reservation.training_geometry_seeds"),
                                     (303, "future_full_reservation.blind.seeds"),
                                     (203, "partitions.blind.seeds")):
            with self.subTest(seed=seed), self.assertRaises(SeedAuditBlocked) as ctx:
                self.audit(seed)
            self.assertIn({"path": self.rlpd_path, "field": expected_field},
                          ctx.exception.report["collisions"][0]["sources"])

    def test_r6_reset_seed_and_geometry_seed_are_both_parsed(self) -> None:
        with self.assertRaises(SeedAuditBlocked) as ctx:
            self.audit(901)
        self.assertIn({"path": self.r6_ledger, "field": "line:1.seed"},
                      ctx.exception.report["collisions"][0]["sources"])
        self.assertIn({"path": self.r6_ledger, "field": "line:1.geometry_seed"},
                      ctx.exception.report["collisions"][0]["sources"])
        self.rows[self.r6_ledger][0]["geometry_seed"] = 902
        self._freeze_all()
        with self.assertRaises(SeedAuditBlocked) as ctx:
            self.audit(902)
        self.assertEqual(ctx.exception.report["ambiguities"][0]["reason"],
                         "seed and geometry_seed disagree")
        self.assertIn({"path": self.r6_ledger, "field": "line:1.geometry_seed"},
                      ctx.exception.report["collisions"][0]["sources"])

    def test_budget_stop_row_keeps_seed_excluded(self) -> None:
        self.rows[self.r6_ledger].append({"event": "budget-stop", "geometry_seed": 915,
                                           "track_id": 2, "budget_interrupted": True})
        self._freeze_all()
        with self.assertRaises(SeedAuditBlocked) as ctx:
            self.audit(915)
        self.assertIn({"path": self.r6_ledger, "field": "line:2.geometry_seed"},
                      ctx.exception.report["collisions"][0]["sources"])

    def test_teacher_source_and_dreamer_development_exclusions(self) -> None:
        for seed, path, field in ((701, self.source_ledgers[0], "line:1.seed"),
                                  (500, self.dreamer_path, "reserved_training_seeds"),
                                  (801, self.partial_ledger, "line:1.seed")):
            with self.subTest(seed=seed), self.assertRaises(SeedAuditBlocked) as ctx:
                self.audit(seed)
            self.assertIn({"path": path, "field": field},
                          ctx.exception.report["collisions"][0]["sources"])

    def test_exact_token_ambiguity_is_blocking_not_inferred_geometry(self) -> None:
        self.rlpd["geometry_audit"]["freshness_statement"] = "arbitrary seed=3000; not a geometry field"
        self._freeze_all()
        self._pin_experiments()
        with self.assertRaises(SeedAuditBlocked) as ctx:
            self.audit()
        self.assertEqual(ctx.exception.report["collisions"], [])
        self.assertTrue(any(item["path"] == self.rlpd_path and item["seed"] == 3000
                            for item in ctx.exception.report["ambiguities"]))
        self.rlpd["geometry_audit"]["freshness_statement"] = "not tokens: 13000, x3000, 0.3000"
        self._freeze_all()
        self._pin_experiments()
        self.assertTrue(self.audit()["passed"])

    def test_discovered_new_training_ledger_and_symlink_are_safe(self) -> None:
        extra = "runs/drq-r6/learner-1/episodes.jsonl"
        self._write(extra, b'{"event":"reset","track_id":1,"geometry_seed":3010}\n')
        with self.assertRaises(SeedAuditBlocked) as ctx:
            self.audit(training_roots=("runs/drq-r6",))
        self.assertIn({"path": extra, "field": "line:1.geometry_seed"},
                      ctx.exception.report["collisions"][0]["sources"])
        (self.root / extra).unlink()
        (self.root / extra).symlink_to(self.blind)
        with self.assertRaisesRegex(SeedAuditError, "symlink"):
            self.audit(training_roots=("runs/drq-r6",))

    def test_malformed_missing_or_changed_required_input_fails_closed(self) -> None:
        for corrupt in (b'not JSON\n', b'{"seed":1,"seed":2,"track_id":1}\n',
                        b'{"event":"reset","track_id":1}\n',
                        b'{"event":"reset","track_id":1,"seed":false}\n'):
            with self.subTest(corrupt=corrupt):
                self._write(self.r6_ledger, corrupt)
                with self.assertRaises(SeedAuditError):
                    self.audit()
        self._freeze_all()
        (self.root / self.r6_ledger).unlink()
        with self.assertRaisesRegex(SeedAuditError, "missing"):
            self.audit()
        self._freeze_all()
        (self.root / self.partial_receipt).unlink()
        with self.assertRaisesRegex(SeedAuditError, "missing"):
            self.audit()
        self._freeze_all()
        self.mix["catalog"]["sha256"] = "0" * 64
        self._write(self.mix_path, json.dumps(self.mix).encode())
        self._pin_experiments()
        with self.assertRaisesRegex(SeedAuditError, "mismatch"):
            self.audit(partial_receipts=(), attempt_receipts=())

    def test_teacher_seed_set_and_protocol_path_type_must_match(self) -> None:
        self.rlpd["geometry_audit"]["teacher_source_ledgers"][0]["geometry_seeds"] = [777]
        self._write(self.rlpd_path, json.dumps(self.rlpd).encode())
        self._pin_experiments()
        with self.assertRaisesRegex(SeedAuditError, "inventory disagrees"):
            self.audit()
        self._freeze_all()
        self._write(self.rlpd_path, json.dumps(self.dreamer).encode())
        self._pin_experiments()
        with self.assertRaisesRegex(SeedAuditError, "path/schema mismatch"):
            self.audit()

    def test_partial_abort_receipt_hash_mismatch_blocks_even_if_protocol_valid(self) -> None:
        self._write(self.partial_ledger, b'{"event":"reset","seed":801,"track_id":1}\n'
                    b'{"event":"end","seed":801,"track_id":1}\n')
        with self.assertRaisesRegex(SeedAuditError, "hash mismatch"):
            self.audit()

    def test_r5_erratum_accepts_only_pinned_chain_and_marks_limit(self) -> None:
        self._add_synthetic_r5()
        with patch.object(audit_module, "R5_PROFILE", self.r5_profile):
            with self.assertRaisesRegex(SeedAuditError, "hash mismatch"):
                self.audit(partial_receipts=self.partial_receipt_paths,
                           attempt_receipts=self.attempt_receipt_paths)
        report = self._audit_with_r5()
        self.assertEqual(report["status"], "no_known_recorded_overlap")
        self.assertIs(report["r5_erratum"]["original_ledger_bytes_attested"], False)
        self.assertEqual(report["r5_erratum"]["epistemic_limit"], R5_EPISTEMIC_LIMIT)
        self.assertEqual(report["r5_erratum"]["sha256"],
                         hashlib.sha256((self.root / R5_ERRATUM_PATH).read_bytes()).hexdigest())
        self.assertEqual({row["kind"] for row in report["source_inventory"] if
                          row["path"] in (R5_ERRATUM_PATH, self.r5_profile["step_metrics_path"],
                                          self.r5_profile["trace_path"])}, {"erratum", "auxiliary"})
        with patch.object(audit_module, "R5_PROFILE", self.r5_profile):
            write_clean_receipt(report, repo_root=self.root,
                                output="experiments/synthetic-r5-seed-audit.json", training_roots=())
        self.assertEqual(json.loads((self.root / "experiments/synthetic-r5-seed-audit.json").read_text()),
                         report)

    def test_r5_erratum_wrong_candidate_missing_or_altered_is_rejected(self) -> None:
        self._add_synthetic_r5()
        with self.assertRaisesRegex(SeedAuditError, "candidate or limitation mismatch"):
            self._audit_with_r5(3001)
        self.r5_erratum["original_ledger_bytes_attested"] = True
        self._write(R5_ERRATUM_PATH, json.dumps(self.r5_erratum).encode())
        self._pin_experiments()
        with self.assertRaisesRegex(SeedAuditError, "candidate or limitation mismatch"):
            self._audit_with_r5()
        self.r5_erratum["original_ledger_bytes_attested"] = False
        self._write(R5_ERRATUM_PATH, json.dumps(self.r5_erratum).encode())
        self._pin_experiments()
        (self.root / R5_ERRATUM_PATH).unlink()
        with self.assertRaisesRegex(SeedAuditError, "experiment JSON catalog changed"):
            self._audit_with_r5()
        with self.assertRaisesRegex(SeedAuditError, "only the exact r5 erratum path"):
            self.audit(r5_erratum="experiments/other.json")

    def test_r5_erratum_detects_any_changed_pinned_source_bytes(self) -> None:
        self._add_synthetic_r5()
        changes = (
            (self.r5_profile["ledger_path"], b'{"event":"reset","seed":3910800001,"track_id":1}\n'),
            (self.r5_profile["attempt_receipt_path"], b'{"altered":true}\n'),
            (self.r5_profile["supersession_path"], b'{"altered":true}\n'),
            (self.r5_profile["step_metrics_path"], b'{"decision":16383}\n'),
            (self.r5_profile["trace_path"], b"changed synthetic trace"),
            (self.r5_profile["protocol_path"], b'{"altered":true}\n'),
            (self.source_ledgers[0], b'{"event":"reset","seed":705,"track_id":1}\n'),
        )
        for name, raw in changes:
            with self.subTest(path=name):
                original = (self.root / name).read_bytes()
                self._write(name, raw)
                if name.startswith("experiments/"):
                    self._pin_experiments()
                with self.assertRaises(SeedAuditError):
                    self._audit_with_r5()
                self._write(name, original)
                if name.startswith("experiments/"):
                    self._pin_experiments()
        self.assertTrue(self._audit_with_r5()["passed"])

    def test_r5_erratum_rejects_changed_typed_counts_or_unreserved_seed(self) -> None:
        self._add_synthetic_r5()
        for rows, pattern in ((self.r5_rows[:-2], "ordered reset/end"),
                              (copy.deepcopy(self.r5_rows), "outside reserved TRAIN catalog pool")):
            with self.subTest(pattern=pattern):
                if len(rows) == len(self.r5_rows):
                    rows[0]["seed"] = 3910800800
                    rows[0]["geometry_seed"] = 3910800800
                    rows[1]["seed"] = 3910800800
                self.r5_profile["ledger_sha256"] = self._write(
                    self.r5_profile["ledger_path"], b"".join(
                        json.dumps(row).encode() + b"\n" for row in rows))
                self._write(R5_ERRATUM_PATH, json.dumps(self.r5_erratum).encode())
                self._pin_experiments()
                with self.assertRaisesRegex(SeedAuditError, pattern):
                    self._audit_with_r5()

    def test_r5_erratum_cannot_attest_rewritten_valid_hash_field(self) -> None:
        self._add_synthetic_r5()
        receipt_path = self.r5_profile["attempt_receipt_path"]
        receipt = json.loads((self.root / receipt_path).read_text())
        receipt["episodes_sha256"] = "0" * 64
        self.r5_profile["invalid_receipt_episodes_sha256"] = receipt["episodes_sha256"]
        self.r5_profile["attempt_receipt_sha256"] = self._write(
            receipt_path, json.dumps(receipt).encode())
        supersession_path = self.r5_profile["supersession_path"]
        supersession = json.loads((self.root / supersession_path).read_text())
        supersession["attempt_receipt_sha256"] = self.r5_profile["attempt_receipt_sha256"]
        self.r5_profile["supersession_sha256"] = self._write(
            supersession_path, json.dumps(supersession).encode())
        self._write(R5_ERRATUM_PATH, json.dumps(self.r5_erratum).encode())
        self._pin_experiments()
        with self.assertRaisesRegex(SeedAuditError, "sole documented r5 receipt defect"):
            self._audit_with_r5()

    def test_four_prior_collection_ledgers_inventory_both_rows_and_discard(self) -> None:
        self._add_synthetic_collections()
        original = Path.read_bytes
        reads = []

        def read(path: Path) -> bytes:
            self.assertNotEqual(path, self.blind)
            reads.append(path.relative_to(self.root).as_posix())
            return original(path)

        with patch.object(Path, "read_bytes", read):
            report = self._audit_with_collections()
        self.assertEqual({row["path"] for row in report["source_inventory"]
                          if row["kind"] == "collection"}, set(self.collection_paths))
        self.assertEqual(set(self.collection_paths) & set(reads), set(self.collection_paths))
        self.assertEqual(report["status"], "no_known_recorded_overlap")
        first = self.collection_paths[0]
        with self.assertRaises(SeedAuditBlocked) as ctx:
            self._audit_with_collections(self.collection_seeds[first][0])
        self.assertIn({"path": first, "field": "line:1.geometry_seed"},
                      ctx.exception.report["collisions"][0]["sources"])
        self.assertIn({"path": first, "field": "line:2.geometry_seed"},
                      ctx.exception.report["collisions"][0]["sources"])
        self.assertIn({"path": first, "field": "line:3.geometry_seed"},
                      ctx.exception.report["collisions"][1]["sources"])
        self.assertIn({"path": first, "field": "line:4.geometry_seed"},
                      ctx.exception.report["collisions"][1]["sources"])

    def test_prior_collection_missing_or_drifted_bytes_fail_closed(self) -> None:
        self._add_synthetic_collections()
        report = self._audit_with_collections()
        missing = self.collection_paths[0]
        (self.root / missing).unlink()
        with self.assertRaisesRegex(SeedAuditError, "missing"):
            self._audit_with_collections()
        self._write(missing, b"".join(json.dumps(row).encode() + b"\n"
                                       for row in self.collection_rows[missing]))
        changed = self.collection_paths[1]
        rows = copy.deepcopy(self.collection_rows[changed])
        rows[1]["reward"] = 12.5  # Still valid schema, but historical bytes drifted.
        self._write(changed, b"".join(json.dumps(row).encode() + b"\n" for row in rows))
        with self.assertRaisesRegex(SeedAuditError, "hash drift"):
            self._audit_with_collections()
        with self.assertRaisesRegex(SeedAuditError, "source changed"):
            write_clean_receipt(report, repo_root=self.root,
                                output="experiments/prior-test-seed-audit.json", training_roots=())
        self.assertFalse((self.root / "experiments/prior-test-seed-audit.json").exists())

    def test_prior_collection_pair_and_schema_corruption_blocks(self) -> None:
        self._add_synthetic_collections()
        path = self.collection_paths[0]
        baseline = copy.deepcopy(self.collection_rows[path])
        changes = (
            lambda rows: rows[1].__setitem__("geometry_seed", rows[3]["geometry_seed"]),
            lambda rows: rows[1].__setitem__("track_id", 2),
            lambda rows: rows[1].__setitem__("path", "evaluations/blind/episode-0000.npz"),
            lambda rows: rows[0].__setitem__("obstacles", False),
            lambda rows: rows[0].__setitem__("cell_index", 1),
            lambda rows: rows[2].__setitem__("geometry_seed", rows[0]["geometry_seed"]),
            lambda rows: rows[3].__setitem__("event", "stored_episode"),
            lambda rows: rows[1].__setitem__("sha256", "a" * 63),
            lambda rows: rows[1].__setitem__("truncated", True),
            lambda rows: rows[3].__setitem__("reason", "unknown"),
        )
        for change in changes:
            with self.subTest(change=change):
                rows = copy.deepcopy(baseline)
                change(rows)
                self.collection_hashes[path] = self._write(
                    path, b"".join(json.dumps(row).encode() + b"\n" for row in rows))
                with self.assertRaises(SeedAuditError):
                    self._audit_with_collections()
        for invalid in (baseline[:3], baseline[1:3], baseline + baseline[:2]):
            with self.subTest(rows=len(invalid)):
                self.collection_hashes[path] = self._write(
                    path, b"".join(json.dumps(row).encode() + b"\n" for row in invalid))
                with self.assertRaises(SeedAuditError):
                    self._audit_with_collections()
        self.collection_hashes[path] = self._write(
            path, b'{"event":"reset","event":"reset"}\n'
            + json.dumps(baseline[1]).encode() + b"\n")
        with self.assertRaisesRegex(SeedAuditError, "duplicate JSON key"):
            self._audit_with_collections()

    def test_opt_in_receipt_write_rechecks_inventory_and_never_replaces(self) -> None:
        report = self.audit()
        output = "experiments/rlpd-g0-seed-audit.json"
        target = self.root / output
        self.assertFalse(target.exists())  # Read-only audit never writes a receipt.
        self._write(self.r6_ledger, b'{"event":"reset","seed":990,"track_id":1}\n')
        with self.assertRaisesRegex(SeedAuditError, "source changed"):
            write_clean_receipt(report, repo_root=self.root, output=output, training_roots=())
        self.assertFalse(target.exists())
        self._freeze_all()
        report = self.audit()
        extra = "runs/drq-r6/learner-1/episodes.jsonl"
        self._write(extra, b'{"event":"reset","seed":997,"track_id":1}\n')
        with self.assertRaisesRegex(SeedAuditError, "new TRAIN ledger"):
            write_clean_receipt(report, repo_root=self.root, output=output,
                                training_roots=("runs/drq-r6",))
        (self.root / extra).unlink()
        write_clean_receipt(report, repo_root=self.root, output=output,
                            training_roots=("runs/drq-r6",))
        self.assertEqual(json.loads(target.read_text()), report)
        self.assertEqual(self.audit(), report)  # Own clean receipt is not a new protocol.
        with self.assertRaisesRegex(SeedAuditError, "new experiments"):
            write_clean_receipt(report, repo_root=self.root, output=output,
                                training_roots=("runs/drq-r6",))
        with self.assertRaises(SeedAuditError):
            write_clean_receipt(report, repo_root=self.root,
                                output="evaluations/blind/episodes.jsonl", training_roots=())

    def test_new_cross_lane_protocol_or_unknown_seed_audit_fails_closed(self) -> None:
        baseline = self.audit()
        protocol = "experiments/dreamerv3-p1-new-study-v1.json"
        self._write(protocol, json.dumps({"format": "haic-dreamerv3-p1-protocol-v1",
                                          "reserved_training_seeds": baseline["candidate_seeds"]}).encode())
        with self.assertRaisesRegex(SeedAuditError, "experiment JSON catalog changed"):
            self.audit()
        with self.assertRaisesRegex(SeedAuditError, "experiment JSON catalog changed"):
            write_clean_receipt(baseline, repo_root=self.root,
                                output="experiments/new-protocol-seed-audit.json", training_roots=())
        self.assertFalse((self.root / "experiments/new-protocol-seed-audit.json").exists())
        (self.root / protocol).unlink()
        bogus = "experiments/dreamer-p1-seed-audit.json"
        self._write(bogus, b'{"format":"haic-dreamerv3-p1-cross-lane-seed-audit-v1"}')
        with self.assertRaisesRegex(SeedAuditError, "unrecognized experiment seed-audit"):
            self.audit()

    def test_existing_untyped_json_mutation_and_exact_candidate_token_block(self) -> None:
        name = "experiments/drqv2-legacy-notes.json"
        self._write(name, b'{"note":"seed=13000, x3000, 0.3000"}')
        self._pin_experiments()  # Synthetic baseline only; production pin is immutable.
        report = self.audit()
        evidence = {row["path"]: row for row in report["experiment_content_inventory"]}
        self.assertIn(name, evidence)
        self.assertEqual(evidence[name]["sha256"],
                         hashlib.sha256((self.root / name).read_bytes()).hexdigest())
        self._write(name, b'{"training_geometry_seeds":[3000]}')
        with self.assertRaisesRegex(SeedAuditError, "experiment JSON bytes changed"):
            self.audit()
        with self.assertRaisesRegex(SeedAuditError, "experiment JSON content changed"):
            write_clean_receipt(report, repo_root=self.root,
                                output="experiments/untyped-test-seed-audit.json", training_roots=())
        self.assertFalse((self.root / "experiments/untyped-test-seed-audit.json").exists())
        self._pin_experiments()  # Reach the independent exact-token gate with fixed bytes.
        with self.assertRaises(SeedAuditBlocked) as ctx:
            self.audit()
        self.assertEqual(ctx.exception.report["collisions"], [])
        self.assertIn({"path": name, "line": 1, "seed": 3000,
                       "reason": "exact token in otherwise-untyped experiment JSON"},
                      ctx.exception.report["ambiguities"])

    def test_frozen_self_protocol_must_pin_exact_clean_receipt_and_cells(self) -> None:
        self._add_synthetic_r5()
        report = self._audit_with_r5()
        audit_path = "experiments/synthetic-g0-seed-audit.json"
        with patch.object(audit_module, "R5_PROFILE", self.r5_profile):
            write_clean_receipt(report, repo_root=self.root,
                                output=audit_path, training_roots=())
        protocol_path = "experiments/pixel-rlpd-g0-diagnostic-v1.json"
        frozen = {"format": "haic-rlpd-g0-diagnostic-v1", "status": "frozen",
                  "partition": "TRAIN", "cells": copy.deepcopy(report["cells"]),
                  "geometry_audit_path": audit_path,
                  "geometry_audit_sha256": hashlib.sha256((self.root / audit_path).read_bytes()).hexdigest(),
                  "r5_erratum_path": R5_ERRATUM_PATH,
                  "r5_erratum_sha256": report["r5_erratum"]["sha256"]}
        self._write(protocol_path, json.dumps(frozen).encode())
        self.assertEqual(self._audit_with_r5(), report)
        altered = copy.deepcopy(frozen)
        altered["cells"][0]["geometry_seed"] += 1
        self._write(protocol_path, json.dumps(altered).encode())
        with self.assertRaisesRegex(SeedAuditError, "experiment JSON catalog changed"):
            self._audit_with_r5()
        altered = copy.deepcopy(frozen)
        altered["geometry_audit_sha256"] = "0" * 64
        self._write(protocol_path, json.dumps(altered).encode())
        with self.assertRaisesRegex(SeedAuditError, "experiment JSON catalog changed"):
            self._audit_with_r5()
        self._write(protocol_path, json.dumps(frozen).encode())
        self._write("experiments/dreamerv3-p1-new-study-v1.json",
                    b'{"format":"haic-dreamerv3-p1-protocol-v1"}')
        with self.assertRaisesRegex(SeedAuditError, "experiment JSON catalog changed"):
            self._audit_with_r5()

    def test_unrecognized_or_missing_schema_and_bad_range_fail_closed(self) -> None:
        baseline = copy.deepcopy(self.rlpd)
        for edit in (lambda r: r.__setitem__("format", "unknown"),
                     lambda r: r.pop("reserved_training_seeds"),
                     lambda r: r["partitions"]["blind"].pop("seeds"),
                     lambda r: r["partitions"].pop("blind"),
                     lambda r: r["future_full_reservation"].pop("blind"),
                     lambda r: r.__setitem__("seed_range", [3000, 9000])):
            with self.subTest(edit=edit):
                self.rlpd = copy.deepcopy(baseline)
                edit(self.rlpd)
                self._freeze_all()
                self._pin_experiments()
                with self.assertRaises(SeedAuditError):
                    self.audit()
        self.rlpd = baseline
        self._freeze_all()
        for bad in (-1, True, (1 << 32) - 10, 2.2):
            with self.subTest(seed_start=bad), self.assertRaises(SeedAuditError):
                self.audit(bad)
        with self.assertRaises(SeedAuditError):
            self.audit(required_protocols=(self.rlpd_path,))


if __name__ == "__main__":
    unittest.main()
