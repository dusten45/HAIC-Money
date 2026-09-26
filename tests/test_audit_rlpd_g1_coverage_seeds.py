"""Synthetic-only G1 inventory tests; never open repository episodes or a simulator."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import scripts.audit_rlpd_g1_coverage_seeds as audit
from haic.train_seed_reservations import AUDIT_FORMAT, reserve_train_seeds


class G1CoverageSeedInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / audit.TRAIN_CLAIMS).mkdir(parents=True)
        (self.root / audit.TRAIN_CLAIMS / ".gitkeep").touch()
        self.write(audit.G0_PROTOCOL, {
            "format": "haic-rlpd-g0-diagnostic-v1", "partition": "TRAIN",
            "actors": [{"id": "long-horizon-seed11"}, {"id": "entropy-v5-author-seed50"}],
            "cells": self.g0_cells(),
        })
        self.write(audit.G0_AUDIT, {
            "format": "haic-rlpd-g0-seed-audit-v1", "candidate_seeds": list(range(4272000001, 4272000013)),
            "cells": self.g0_cells(),
        })
        g0 = self.load(audit.G0_PROTOCOL)
        g0["geometry_audit_sha256"] = self.sha(audit.G0_AUDIT)
        self.write(audit.G0_PROTOCOL, g0)
        self.write(audit.CATALOG_PROTOCOL, {
            "format": "haic-drq-training-geometry-protocol-v1",
            "candidate_seeds": list(range(3910800001, 3910800513)),
        })
        self.write(audit.CATALOG, {
            "format": "haic-drq-training-geometry-catalog-v1",
            "protocol_sha256": self.sha(audit.CATALOG_PROTOCOL),
            "seed_audit": {"proposed_seeds": list(range(3910800001, 3910800513))},
        })
        self.write(audit.R6_PROTOCOL, {
            "format": "haic-drq-geometry-mix-study-v1",
            "environment": {"partition": "TRAIN", "geometry_seeds": [3910800001]},
            "diagnostic_pool": {"partition": "TRAIN-DIAGNOSTIC", "geometry_seeds": [3910800002]},
        })
        self.write(audit.R7_PROTOCOL, {
            "format": "haic-drq-retention-study-v1",
            "r6_protocol_path": audit.R6_PROTOCOL, "r6_protocol_sha256": self.sha(audit.R6_PROTOCOL),
            "runs": [{"run_dir": f"runs/20260926-drqv2-retention-r7/learner-{n}",
                      "source_seed": n // 6,
                      "variant": ("uniform", "failure_weighted", "easy_retention")[n // 2 % 3],
                      "condition": ("r7a", "r7b")[n % 2],
                      "rng_seeds": self.rng_schedule(103000 + n // 6)} for n in range(12)],
        })
        self.write("experiments/dreamerv3-reused-train-diversity-v1.json", {
            "format": "haic-dreamerv3-reused-train-diagnostic-v1",
            "cells": [{"geometry_seed": 80001, "track_id": 1}],
        })
        self.write("experiments/pixel-rlpd-entropy-target-ablation-v3.json", {
            "format": "haic-pixel-rlpd-study-v1", "reserved_training_seeds": [85000],
            "training_geometry_seeds": [86000],
            "future_full_reservation": {"training_geometry_seeds": [87000]},
            "partitions": {"screen": {"seeds": [88000]},
                           "confirmation": {"seeds": [88001]}, "blind": {"seeds": [88002]}},
        })
        # The old G0 candidate-bound erratum exists but cannot attest G1.
        self.write("experiments/drqv2-geometry-mix-v1-r5-seed-audit-erratum-v1.json", {
            "format": "haic-rlpd-g0-r5-receipt-erratum-v1", "candidate_seeds": list(range(4272000001, 4272000013))})
        self.write(audit.G0_CELLS, b"".join(json.dumps({
            "actor_id": actor, "partition": "TRAIN", "track_id": 1, "geometry_seed": seed,
        }).encode() + b"\n" for seed in range(4272000001, 4272000013)
            for actor in ("long-horizon-seed11", "entropy-v5-author-seed50")))
        self.write(audit.G0_MANIFEST, {
            "format": "haic-rlpd-g0-diagnostic-result-v1", "cell_count": 24,
            "geometry_count": 12, "protocol_sha256": self.sha(audit.G0_PROTOCOL),
            "geometry_audit_sha256": self.sha(audit.G0_AUDIT),
            "cells_sha256": self.sha(audit.G0_CELLS),
        })
        self.write(f"{audit.G0_CLAIMS}/claim.json", {
            "format": "haic-rlpd-g0-geometry-claim-v1", "status": "reserved-once",
            "geometry_seeds": list(range(4272000001, 4272000013)),
            "protocol_sha256": self.sha(audit.G0_PROTOCOL),
            "geometry_audit_sha256": self.sha(audit.G0_AUDIT),
        })
        self.write(audit.R5_LEDGER, self.rows([
            {"event": "reset", "seed": 390000, "track_id": 1},
            {"event": "end", "seed": 390000, "track_id": 1},
        ]))
        self.r5_abort_sha = self.write(audit.R5_ABORT, {
            "format": "haic-drq-geometry-mix-partial-arm-abort-v1", "episodes_sha256": audit.R5_BAD_HASH,
        })
        self.r5_ledger_sha = self.sha(audit.R5_LEDGER)
        self.write("runs/20260926-drqv2-retention-r7/learner-0/episodes.jsonl", self.rows([
            {"event": "reset", "seed": 90001, "geometry_seed": 90001, "track_id": 2},
            {"event": "end", "seed": 90001, "track_id": 2},
        ]))
        self.write("runs/20260925-pixel-rlpd-entropy-target-ablation-v5/prior-data/collection.jsonl", self.rows([
            {"event": "reset", "geometry_seed": 91001, "track_id": 1, "obstacles": True,
             "cell_index": 0, "episode_id": 0},
            {"event": "discarded_incomplete_episode", "geometry_seed": 91001, "track_id": 1,
             "episode_id": 0},
        ]))
        self.blind = self.root / "runs/20260926-drqv2-retention-r7/learner-0/evaluations/blind/episodes.jsonl"
        self.blind.parent.mkdir(parents=True)
        self.blind.write_text("must not open\n")
        self.blind_camel = self.root / "runs/study/blindEpisodes/episodes.jsonl"
        self.blind_camel.parent.mkdir(parents=True)
        self.blind_camel.write_text("must not open camel-case blind\n")
        self.eval_camel = self.root / "runs/study/evalEpisodes/episodes.jsonl"
        self.eval_camel.parent.mkdir(parents=True)
        self.eval_camel.write_text("must not open camel-case evaluator\n")
        self.evaluator_hyphen = self.root / "runs/20260921-drqv2-evaluator-gate/episodes.jsonl"
        self.evaluator_hyphen.parent.mkdir(parents=True)
        self.evaluator_hyphen.write_text("must not open hyphenated evaluator\n")
        self.private_camel = self.root / "runs/privateEpisodes/episodes.jsonl"
        self.private_camel.parent.mkdir(parents=True)
        self.private_camel.write_text("must not open private episode\n")

    @staticmethod
    def rng_schedule(seed: int) -> dict[str, int]:
        return {key: seed for key in ("track_seed", "geometry_seed", "actor_rng_seed",
                                      "replay_rng_seed", "target_noise_seed", "update_rng_seed")}

    @staticmethod
    def rows(values: list[dict]) -> bytes:
        return b"".join(json.dumps(row).encode() + b"\n" for row in values)

    @staticmethod
    def g0_cells() -> list[dict]:
        return [{"track_id": 1, "geometry_seed": seed, "partition": "TRAIN", "obstacles": True}
                for seed in range(4272000001, 4272000013)]

    def write(self, name: str, value: dict | bytes) -> str:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def sha(self, name: str) -> str:
        return hashlib.sha256((self.root / name).read_bytes()).hexdigest()

    def load(self, name: str) -> dict:
        return json.loads((self.root / name).read_text())

    def run_audit(self, seed: int = 3000, **kwargs) -> dict:
        kwargs.setdefault("required_ledgers", ())
        kwargs.setdefault("required_collections", ())
        with patch.object(audit, "R5_ABORT_SHA", self.r5_abort_sha), patch.object(
                audit, "R5_LEDGER_SHA", self.r5_ledger_sha):
            return audit.audit_g1_coverage_seeds(seed, repo_root=self.root, **kwargs)

    def attest(self) -> str:
        return self.write(audit.R5_G1_ATTESTATION, {
            "format": "haic-rlpd-g1-r5-source-attestation-v1",
            "scope": "r5 malformed receipt hash; independently hashed TRAIN ledger road IDs only",
            "receipt_path": audit.R5_ABORT, "receipt_sha256": self.r5_abort_sha,
            "ledger_path": audit.R5_LEDGER, "ledger_sha256": self.r5_ledger_sha,
            "invalid_receipt_episodes_sha256": audit.R5_BAD_HASH,
            "original_ledger_bytes_attested": False,
        })

    def clean(self, seed: int = 3000) -> dict:
        attestation = self.attest()
        initial = self.run_audit(seed, r5_attestation_sha256=attestation)
        report = self.run_audit(seed, r5_attestation_sha256=attestation,
                                expected_experiments_sha256=initial["experiment_inventory_sha256"],
                                expected_sources_sha256=initial["source_inventory_sha256"])
        return report

    def test_fixed_24_no_replacement_and_blind_sentinel_never_opened(self) -> None:
        original = Path.read_bytes

        def guarded(path: Path) -> bytes:
            self.assertFalse(any("blind" in part.casefold()
                                 or part.casefold().startswith("private")
                                 or "evaluator" in part.casefold()
                                 or part.casefold().startswith("eval") for part in path.parts))
            return original(path)

        with patch.object(Path, "read_bytes", guarded):
            report = self.clean()
        self.assertEqual(report["status"], "no_known_recorded_overlap")
        self.assertEqual(report["format"], "haic-rlpd-g1-coverage-seed-inventory-v2")
        self.assertEqual([c["geometry_seed"] for c in report["cells"]], list(range(3000, 3024)))
        self.assertTrue(all(c["track_id"] == 1 and c["obstacles"] is True
                            and c["partition"] == "TRAIN" for c in report["cells"]))
        self.assertFalse(report["protocol_frozen"])
        self.assertEqual(report["blind_episode_reads"], 0)

    def test_ambiguous_unblinded_training_path_blocks_instead_of_skipping(self) -> None:
        path = "runs/unblinded-training/episodes.jsonl"
        self.write(path, self.rows([{"event": "reset", "seed": 3000, "track_id": 1},
                                    {"event": "end", "seed": 3000, "track_id": 1}]))
        report = self.run_audit(3000)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("ambiguous protected/TRAIN" in b["reason"]
                            for b in report["blockers"]))

    def test_drq_unselected_catalog_and_g0_used_across_tracks(self) -> None:
        for seed, path in ((3910800501, audit.CATALOG_PROTOCOL), (4272000001, audit.G0_PROTOCOL)):
            with self.subTest(seed=seed):
                report = self.run_audit(seed)
                self.assertEqual(report["status"], "BLOCKED")
                self.assertTrue(any(source["path"] == path for source in report["collisions"][0]["sources"]))
        other = "runs/other-track-training/episodes.jsonl"
        self.write(other, self.rows([{"event": "reset", "seed": 4272000001, "track_id": 2},
                                     {"event": "end", "seed": 4272000001, "track_id": 2}]))
        report = self.run_audit(4272000001)
        self.assertTrue(any(source["path"] == other for source in report["collisions"][0]["sources"]))

    def test_screen_diagnostic_confirmation_and_blind_seeds_stay_excluded(self) -> None:
        for seed, partition in ((3910800002, "TRAIN-DIAGNOSTIC"), (88000, "screen"),
                                (88001, "confirmation"), (88002, "blind")):
            with self.subTest(partition=partition):
                report = self.run_audit(seed)
                self.assertEqual(report["status"], "BLOCKED")
                self.assertEqual(report["collisions"][0]["geometry_seed"], seed)

    def test_shared_cross_lane_claim_blocks_even_without_interaction(self) -> None:
        cell = {"partition": "TRAIN", "track_id": 3, "geometry_seed": 3000, "obstacles": False}

        def cleared(cells: tuple) -> dict:
            return {"format": AUDIT_FORMAT, "partition": "TRAIN", "status": "clear",
                    "cells": [dict(candidate) for candidate in cells], "collisions": [],
                    "blockers": [], "consumed_seeds": [], "reserved_seeds": [],
                    "source": "synthetic/independent-lane-audit.json"}

        reserve_train_seeds(self.root / audit.TRAIN_CLAIMS, [cell], cleared, study_id="drq-unrelated-lane")
        report = self.run_audit(3000)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(source["path"].endswith("seed-3000.json")
                            for source in report["collisions"][0]["sources"]))

    def test_cli_reaudits_and_claims_before_any_reset(self) -> None:
        args = ["audit_rlpd_g1_coverage_seeds", "--repo-root", str(self.root),
                "--seed-start", "3000", "--reserve", "--study-id", "g1-synthetic"]
        with (patch.object(sys, "argv", args), patch.object(audit, "R5_ABORT_SHA", self.r5_abort_sha),
              patch.object(audit, "R5_LEDGER_SHA", self.r5_ledger_sha), redirect_stdout(io.StringIO()) as output):
            self.assertEqual(audit.main(), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(len(result["claims"]), 24)
        self.assertEqual(result["locked_audit"]["status"], "no_known_recorded_overlap")
        self.assertEqual(result["locked_claim_audit"]["source"],
                         f"haic-rlpd-g1-coverage-seed-inventory-v2:{result['locked_audit']['source_inventory_sha256']}")
        claim_digest = hashlib.sha256(json.dumps(result["locked_claim_audit"], sort_keys=True,
                                                 separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
        self.assertTrue(all(claim["audit_digest_sha256"] == claim_digest
                            for claim in result["claims"]))
        self.assertTrue((self.root / audit.TRAIN_CLAIMS / "seed-3000.json").is_file())
        self.assertEqual(self.run_audit()["status"], "BLOCKED")
        protocol_path = "experiments/g1-synthetic-protocol.json"
        frozen_sha = self.write(protocol_path, {
            "format": "haic-rlpd-g1-coverage-protocol-v1", "status": "frozen",
            "study_id": "g1-synthetic", "partition": "TRAIN", "cells": result["cells"],
            "train_claims_sha256": result["train_claims_sha256"]})
        self_mode = dict(self_study_id="g1-synthetic", self_protocol_path=protocol_path,
                         self_protocol_sha256=frozen_sha)
        fresh = self.run_audit(**self_mode)
        self.assertEqual(fresh["status"], "no_known_recorded_overlap", fresh["blockers"])
        self.assertTrue(fresh["self_claims_verified"])
        self.assertEqual(fresh["train_claims_sha256"], result["train_claims_sha256"])
        self.assertEqual(self.run_audit()["status"], "BLOCKED")
        claim_file = self.root / audit.TRAIN_CLAIMS / "seed-3001.json"
        original_claim = claim_file.read_bytes()
        changed_claim = json.loads(original_claim)
        changed_claim["study_id"] = "foreign-lane"
        claim_file.write_text(json.dumps(changed_claim))  # Synthetic mutation only.
        self.assertEqual(self.run_audit(**self_mode)["status"], "BLOCKED")
        claim_file.write_bytes(original_claim)
        self.write("runs/after-frozen/episodes.jsonl", self.rows([
            {"event": "reset", "seed": 3000, "track_id": 2}]))
        self.assertEqual(self.run_audit(**self_mode)["status"], "BLOCKED")

    def test_claim_prints_locked_evidence_after_unrelated_intervening_edit(self) -> None:
        original = audit.audit_g1_coverage_seeds
        calls = 0

        def midflight(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            if calls == 1:
                self.write("experiments/new-unrelated-drq.json", {
                    "format": "haic-drq-geometry-mix-study-v1",
                    "environment": {"partition": "TRAIN", "geometry_seeds": [99001]}})
            return result

        args = ["audit_rlpd_g1_coverage_seeds", "--repo-root", str(self.root),
                "--seed-start", "3000", "--reserve", "--study-id", "g1-synthetic"]
        with (patch.object(sys, "argv", args), patch.object(audit, "R5_ABORT_SHA", self.r5_abort_sha),
              patch.object(audit, "R5_LEDGER_SHA", self.r5_ledger_sha),
              patch.object(audit, "audit_g1_coverage_seeds", side_effect=midflight),
              redirect_stdout(io.StringIO()) as output):
            self.assertEqual(audit.main(), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(calls, 2)
        self.assertNotEqual(result["source_inventory_sha256"],
                            result["locked_audit"]["source_inventory_sha256"])
        self.assertEqual(result["locked_claim_audit"]["source"],
                         f"haic-rlpd-g1-coverage-seed-inventory-v2:{result['locked_audit']['source_inventory_sha256']}")

    def test_candidate_consumed_in_training_reset_and_partial_record(self) -> None:
        ledger = "runs/other-lane/learner/episodes.jsonl"
        self.write(ledger, self.rows([{"event": "reset", "seed": 3000, "track_id": 4},
                                      {"event": "end", "seed": 3000, "track_id": 4}]))
        report = self.run_audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(source["path"] == ledger for source in report["collisions"][0]["sources"]))
        self.write(ledger, self.rows([{"event": "reset", "seed": 3000, "track_id": 4}]))
        report = self.run_audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(issue["path"] == ledger for issue in report["blockers"]))

    def test_dreamer_collection_receipt_counts_actual_train_interaction(self) -> None:
        receipt = "runs/20260926-dreamerv3-reused-train-source1-v1/collection/teacher/collection-result.json"
        self.write(receipt, {"allowed_cells": [{"geometry_seed": 3000, "track_id": 2}],
                             "episode_rows": [{"geometry_seed": 3000, "track_id": 2,
                                               "decisions": 37, "status": "partial"}]})
        report = self.run_audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(source["path"] == receipt for source in report["collisions"][0]["sources"]))
        self.write(receipt, {"allowed_cells": [{"geometry_seed": 99001, "track_id": 2}],
                             "episode_rows": [{"geometry_seed": 99001, "track_id": 2,
                                               "decisions": 37, "status": "partial"}]})
        self.assertEqual(self.run_audit()["status"], "no_known_recorded_overlap")

    def test_candidate_ambiguous_partial_identity_blocks_but_unrelated_does_not(self) -> None:
        path = "runs/drq-other-lane/episodes.jsonl"
        self.write(path, self.rows([{"event": "reset", "seed": 3000,
                                     "geometry_seed": 3001, "track_id": 1}]))
        self.assertTrue(any(issue["path"] == path for issue in self.run_audit()["blockers"]))
        self.write(path, self.rows([{"event": "reset", "seed": 99001,
                                     "geometry_seed": 99002, "track_id": 1}]))
        report = self.run_audit()
        self.assertEqual(report["status"], "no_known_recorded_overlap")
        self.assertTrue(any(issue["path"] == path for issue in report["provenance_warnings"]))

    def test_budget_ended_v5_reset_has_bound_receipt_not_a_false_blocker(self) -> None:
        study = "pixel-rlpd-entropy-target-ablation-v5"
        protocol = f"experiments/{study}.json"
        self.write(protocol, {"format": "haic-pixel-rlpd-study-v1",
                              "reserved_training_seeds": [92000, 92001],
                              "partitions": {"screen": {"seeds": [93000]},
                                             "confirmation": {"seeds": [93001]},
                                             "blind": {"seeds": [93002]}}})
        run = "runs/20260925-pixel-rlpd-entropy-target-ablation-v5/rlpd-author-target-seed50"
        ledger = f"{run}/episodes.jsonl"
        self.write(ledger, self.rows([{"event": "reset", "seed": 92000, "track_id": 1},
                                      {"event": "end", "seed": 92000, "track_id": 1, "global_step": 70},
                                      {"event": "reset", "seed": 92001, "track_id": 1}]))
        receipt = {"format": "haic-rlpd-entropy-run-result-v1", "study_id": study,
                   "environment_steps": 100,
                   "candidates": [{"protocol_sha256": self.sha(protocol),
                                   "environment_steps": 100, "resume_episode_steps": 30}]}
        self.write(f"{run}/result.json", receipt)
        report = self.run_audit()
        self.assertEqual(report["status"], "no_known_recorded_overlap")
        self.assertFalse(any(issue["path"] == ledger for issue in report["provenance_warnings"]))
        receipt["candidates"][0]["resume_episode_steps"] = 29
        self.write(f"{run}/result.json", receipt)
        report = self.run_audit(92001)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(issue["path"] == ledger for issue in report["blockers"]))

    def test_r5_ledger_provenance_loss_only_blocks_relevant_candidate(self) -> None:
        self.write(audit.R5_LEDGER, self.rows([{"event": "reset", "seed": 390000, "track_id": 1},
                                               {"event": "end", "seed": 390000, "track_id": 1},
                                               {"event": "reset", "seed": 390001, "track_id": 1}]))
        report = self.run_audit()
        self.assertEqual(report["status"], "no_known_recorded_overlap")
        self.assertEqual(report["r5_evidence"]["status"], "unverified")
        report = self.run_audit(390001)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(source["path"] == audit.R5_LEDGER
                            for source in report["collisions"][0]["sources"]))

    def test_new_dreamer_cell_retired_pool_r7_reset_and_prior_collection(self) -> None:
        for seed, path in ((80001, "dreamerv3-reused-train-diversity-v1.json"),
                           (87000, "pixel-rlpd-entropy-target-ablation-v3.json"),
                           (90001, "learner-0/episodes.jsonl"), (91001, "prior-data/collection.jsonl")):
            with self.subTest(seed=seed):
                hit = self.run_audit(seed)["collisions"][0]
                self.assertEqual(hit["geometry_seed"], seed)
                self.assertTrue(any(path in source["path"] for source in hit["sources"]))

    def test_r7_rng_103000_is_sampler_not_road(self) -> None:
        report = self.clean(103000)
        self.assertEqual(report["status"], "no_known_recorded_overlap")

    def test_unreviewed_rng_field_cannot_disguise_a_candidate_road(self) -> None:
        path = "experiments/pixel-rlpd-unknown-rng-v1.json"
        self.write(path, {"format": "haic-pixel-rlpd-study-v1",
                          "rng_seeds": self.rng_schedule(3000)})
        report = self.run_audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(issue["path"] == path for issue in report["blockers"]))
        self.assertFalse(report["collisions"])

    def test_new_json_candidate_collision_but_unrelated_json_and_byte_changes_warn(self) -> None:
        report = self.clean()
        self.write("experiments/dreamerv3-new-protocol.json", {
            "format": "unreviewed-new", "cells": [{"geometry_seed": 3000, "track_id": 1}]})
        changed = self.run_audit(3000, r5_attestation_sha256=self.sha(audit.R5_G1_ATTESTATION),
                                 expected_experiments_sha256=report["experiment_inventory_sha256"],
                                 expected_sources_sha256=report["source_inventory_sha256"])
        self.assertEqual(changed["status"], "BLOCKED")
        self.assertTrue(any(b["path"] == "experiments/dreamerv3-new-protocol.json" for b in changed["blockers"]))
        (self.root / "experiments/dreamerv3-new-protocol.json").unlink()
        self.write("experiments/dreamerv3-new-protocol.json", {
            "format": "haic-dreamerv3-reused-train-source1-v1",
            "cells": [{"geometry_seed": 99001, "track_id": 1}]})
        changed = self.run_audit(3000, expected_experiments_sha256=report["experiment_inventory_sha256"],
                                 expected_sources_sha256=report["source_inventory_sha256"])
        self.assertEqual(changed["status"], "no_known_recorded_overlap")
        self.assertTrue(changed["repository_changed_since_audit_snapshot"])
        self.assertTrue(changed["provenance_warnings"])
        protocol = self.load(audit.R7_PROTOCOL)
        self.write(audit.R7_PROTOCOL, json.dumps(protocol, indent=2).encode())
        changed = self.run_audit(3000, r5_attestation_sha256=self.sha(audit.R5_G1_ATTESTATION),
                                 expected_experiments_sha256=report["experiment_inventory_sha256"],
                                 expected_sources_sha256=report["source_inventory_sha256"])
        self.assertEqual(changed["status"], "no_known_recorded_overlap")
        self.assertTrue(any(b["field"] == "names+bytes" for b in changed["provenance_warnings"]))

    def test_incomplete_r7_live_ledger_missing_collection_and_bad_schema(self) -> None:
        r7 = "runs/20260926-drqv2-retention-r7/learner-0/episodes.jsonl"
        self.write(r7, self.rows([{"event": "reset", "geometry_seed": 90200,
                                    "seed": 90200, "track_id": 2}]))
        report = self.run_audit(90200)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["collisions"][0]["geometry_seed"], 90200)
        self.assertTrue(any(b["path"] == r7 and "missing source" in b["reason"]
                            for b in report["blockers"]))
        collection = "runs/20260925-pixel-rlpd-entropy-target-ablation-v5/prior-data/collection.jsonl"
        (self.root / collection).unlink()
        report = self.run_audit(3000, required_collections=(collection,))
        self.assertTrue(any(b["path"] == collection and "missing" in b["reason"]
                            for b in report["provenance_warnings"]))
        self.write(audit.R7_PROTOCOL, {"format": "unknown", "runs": []})
        report = self.run_audit(3000)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(b["path"] == audit.R7_PROTOCOL and b["field"] == "schema"
                            for b in report["blockers"]))

    def test_duplicate_json_unknown_geometry_field_and_unsafe_symlink_block(self) -> None:
        r7 = "runs/20260926-drqv2-retention-r7/learner-0/episodes.jsonl"
        self.write(r7, b'{"event":"reset","seed":3000,"seed":90002,"track_id":2}\n')
        self.assertTrue(any("duplicate JSON key" in b["reason"] for b in
                            self.run_audit()["blockers"]))
        self.write(r7, self.rows([{"event": "reset", "seed": 90001, "track_id": 2},
                                  {"event": "end", "seed": 90001, "track_id": 2}]))
        r7_protocol = self.load(audit.R7_PROTOCOL)
        r7_protocol["geometry_seed_range"] = [3000, 3023]
        self.write(audit.R7_PROTOCOL, r7_protocol)
        self.assertTrue(any("unknown seed-bearing field" in b["reason"] for b in
                            self.run_audit()["blockers"]))
        extra = self.root / "runs/synthetic/episodes.jsonl"
        extra.parent.mkdir(parents=True)
        extra.symlink_to(self.blind)
        self.assertTrue(any("unsafe symlink" in b["reason"] for b in
                            self.run_audit()["blockers"]))

    def test_unknown_nested_seed_list_does_not_disappear_from_dreamer_protocol(self) -> None:
        path = "experiments/dreamerv3-reused-train-diversity-v1.json"
        original = self.load(path)
        for key in ("seeds", "training_seeds", "training_seed_ids", "road_ids"):
            with self.subTest(key=key):
                protocol = dict(original)
                protocol["training_pool"] = {key: [3000]}
                self.write(path, protocol)
                report = self.run_audit(3000)
                self.assertEqual(report["status"], "BLOCKED")
                self.assertTrue(any(b["path"] == path and f"training_pool.{key}" in b["reason"]
                                    and "unknown seed-bearing field" in b["reason"]
                                    for b in report["blockers"]))

    def test_any_started_train_directory_missing_episode_ledger_blocks(self) -> None:
        marker = "runs/extra-train/learner-0/run-config.json"
        alternate = "runs/extra-train/learner-1/run_config.json"
        abort = "runs/extra-train/learner-0/precheckpoint-abort.json"
        self.write(marker, {"format": "synthetic-started-train", "study_id": "synthetic"})
        self.write(alternate, {"format": "synthetic-rlpd-run-config", "study_id": "synthetic"})
        self.write(abort, {"format": "synthetic-partial-abort", "geometry_seed": 3000})
        report = self.run_audit(3000)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(b["path"] == marker and "no TRAIN episode" in b["reason"]
                            for b in report["provenance_warnings"]))
        self.assertTrue(any(b["path"] == abort and "no TRAIN episode" in b["reason"]
                            for b in report["blockers"]))
        self.assertTrue(any(b["path"] == alternate and "no TRAIN episode" in b["reason"]
                            for b in report["provenance_warnings"]))
        self.assertTrue(any(entry["path"] == marker for entry in report["source_inventory"]))
        self.assertTrue(any(entry["path"] == alternate for entry in report["source_inventory"]))

    def test_historical_unclosed_r4_reset_without_typed_abort_is_not_waived(self) -> None:
        path = "runs/20260925-drqv2-geometry-mix-v1-r4/learner-0-uniform/episodes.jsonl"
        self.write(path, self.rows([{
            "event": "reset", "seed": 3000, "geometry_seed": 3000,
            "track_id": 1, "episode_id": 36,
        }]))
        report = self.run_audit(3000)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(b["path"] == path and "precheckpoint-abort.json" in b["reason"]
                            for b in report["blockers"]))

    def test_g0_corrupted_chain_cannot_hide_new_candidate_in_cells_or_claim(self) -> None:
        rows = (self.root / audit.G0_CELLS).read_bytes()
        self.write(audit.G0_CELLS, rows + self.rows([{"actor_id": "wrong",
                                                      "geometry_seed": 3000, "track_id": 1}]))
        report = self.run_audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(source["path"] == audit.G0_CELLS for source in report["collisions"][0]["sources"]))
        self.write(audit.G0_CELLS, rows)
        claim_path = f"{audit.G0_CLAIMS}/claim.json"
        claim = self.load(claim_path)
        claim["geometry_seeds"].append(3000)
        self.write(claim_path, claim)
        report = self.run_audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(source["path"] == claim_path for source in report["collisions"][0]["sources"]))

    def test_start_marker_candidate_with_different_ledger_cannot_clear(self) -> None:
        ledger = "runs/another-lane/learner/episodes.jsonl"
        self.write(ledger, self.rows([{"event": "reset", "seed": 99001, "track_id": 1},
                                      {"event": "end", "seed": 99001, "track_id": 1}]))
        marker = "runs/another-lane/learner/run_config.json"
        self.write(marker, {"partition": "TRAIN", "geometry_seed": 3000})
        report = self.run_audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(source["path"] == marker for source in report["collisions"][0]["sources"]))

    def test_failed_train_discovery_blocks_all_candidate_claims(self) -> None:
        def broken_walk(*args, **kwargs):
            kwargs["onerror"](OSError("cannot enumerate TRAIN runs"))
            yield from ()

        with patch.object(audit.os, "walk", side_effect=broken_walk):
            report = self.run_audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(issue["path"] == "runs" and issue["field"] == "discovery"
                            for issue in report["blockers"]))

    def test_range_text_and_ledger_road_alias_are_candidate_relevant(self) -> None:
        path = "experiments/unreviewed-lane.json"
        for value in ({"training_seed_range": {"start": 2990, "count": 24}},
                      {"seed_start": 2990, "seed_count": 24},
                      {"note": "seed=3000 was inspected before this protocol"}):
            with self.subTest(value=value):
                self.write(path, value)
                report = self.run_audit()
                self.assertEqual(report["status"], "BLOCKED")
                self.assertTrue(any(issue["path"] == path for issue in report["blockers"]))
        (self.root / path).unlink()
        ledger = "runs/another-lane/episodes.jsonl"
        self.write(ledger, self.rows([{"event": "reset", "seed": 99001,
                                      "road_id": 3000, "track_id": 1},
                                     {"event": "end", "seed": 99001, "track_id": 1}]))
        report = self.run_audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(issue["path"] == ledger for issue in report["blockers"]))

    def test_r5_current_ledger_verified_without_reusing_g0_erratum(self) -> None:
        report = self.run_audit()
        self.assertEqual(report["status"], "no_known_recorded_overlap")
        self.assertFalse(report["r5_evidence"]["original_ledger_bytes_attested"])
        att_sha = self.attest()
        report = self.run_audit(r5_attestation_sha256="0" * 64)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(b["path"] == audit.R5_ABORT for b in report["blockers"]))
        self.assertEqual(self.clean()["status"], "no_known_recorded_overlap")
        self.write("experiments/rlpd-g1-coverage-v1.json", {
            "format": "haic-rlpd-g1-protocol-v1", "cells": [{"geometry_seed": 3000}]})
        forged = self.run_audit(r5_attestation_sha256=att_sha)
        self.assertTrue(any(b["path"] == "experiments/rlpd-g1-coverage-v1.json"
                            for b in forged["blockers"]))
        self.assertEqual(forged["status"], "BLOCKED")

    def test_bad_candidate_range_or_type_rejected(self) -> None:
        for seed in (-1, True, 2**32 - 23, 1.0):
            with self.subTest(seed=seed), self.assertRaises(audit.InventoryError):
                self.run_audit(seed)

    def test_same_name_ledger_drift_and_unknown_seed_field_block(self) -> None:
        baseline = self.clean()
        ledger = "runs/20260926-drqv2-retention-r7/learner-0/episodes.jsonl"
        self.write(ledger, self.rows([{"event": "reset", "seed": 90002, "track_id": 2},
                                      {"event": "end", "seed": 90002, "track_id": 2}]))
        report = self.run_audit(
            r5_attestation_sha256=self.sha(audit.R5_G1_ATTESTATION),
            expected_experiments_sha256=baseline["experiment_inventory_sha256"],
            expected_sources_sha256=baseline["source_inventory_sha256"])
        self.assertEqual(report["status"], "no_known_recorded_overlap")
        self.assertTrue(report["repository_changed_since_audit_snapshot"])
        self.write(ledger, self.rows([{"event": "reset", "seed": 90002, "track_id": 2,
                                      "geometry_seed_candidates": [3000]},
                                     {"event": "end", "seed": 90002, "track_id": 2}]))
        report = self.run_audit()
        self.assertTrue(any("unknown seed-bearing ledger field" in b["reason"] for b in report["blockers"]))


if __name__ == "__main__":
    unittest.main()
