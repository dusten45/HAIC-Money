"""ID-only Dreamer seed-audit contracts; all files are synthetic TRAIN metadata."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

from scripts import audit_dreamerv3_p1_seeds as audit


class DreamerP1SeedAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sources: dict[str, dict[str, str]] = {kind: {} for kind in audit.KINDS}

        ledger = "runs/drq-source/episodes.jsonl"
        collection = next(iter(audit.KNOWN_INPUTS["prior_collection_ledger"]))
        prior_audit = "experiments/synthetic-geometry-audit.json"
        self.add("training_ledger", ledger, '{"event":"reset","track_id":1,"seed":10}\n')
        self.add("prior_collection_ledger", collection,
                 '{"event":"reset","track_id":2,"geometry_seed":11}\n')
        self.add("prior_seed_audit", prior_audit, {
            "passed": True, "parse_errors": [],
            "global_freshness_claim": audit.FRESHNESS_LIMITATION,
            "known_excluded_geometry_seeds": [12],
            "known_excluded_geometry_seed_count": 1,
            "candidate_seeds": [13], "source_snapshot_count": 1,
            "source_snapshots": [{"path": "experiments/example.json", "sha256": "a" * 64}],
        })
        self.teacher_path = "experiments/drqv2-teacher-replay-v1-r3.json"
        self.add("protocol", self.teacher_path, {
            "study_id": "drqv2-teacher-replay-v1-r3",
            "reserved_training_seeds": [20], "known_excluded_geometry_seeds": [21],
            "geometry_audit": {"candidate_seeds": [22], "report_path": prior_audit,
                               "report_sha256": self.sources["prior_seed_audit"][prior_audit]},
            "partitions": self.partitions(23),
            "training_pools": {"online_training": {"seeds": [26]},
                               "teacher_training": {"seeds": [27]}},
            "source_actors": [{"training_geometry_seeds": [10], "episodes_path": ledger,
                               "episodes_sha256": self.sources["training_ledger"][ledger]}],
        })
        self.pixel_path = "experiments/pixel-rlpd-entropy-target-ablation-v5.json"
        self.add("protocol", self.pixel_path, {
            "format": "haic-pixel-rlpd-study-v1", "reserved_training_seeds": [30],
            "training_geometry_seeds": [31], "teacher_data_cells": [{"geometry_seed": 31}],
            "partitions": self.partitions(32),
            "geometry_audit": {"teacher_source_ledgers": [{
                "geometry_seeds": [10], "path": ledger,
                "sha256": self.sources["training_ledger"][ledger],
            }]},
        })
        self.dreamer_path = "experiments/dreamerv3-b1-terminal-positive-weight-local-v9.json"
        self.add("protocol", self.dreamer_path, {
            "format": "haic-dreamerv3-study-protocol-v1",
            "reserved_training_seeds": [40],
            "training_development": {"seeds": [41], "cells": [{"seed": 41}]},
            "partitions": self.partitions(42),
        })
        self.mix_path = "experiments/drqv2-geometry-mix-v1-r6.json"
        self.mix = {
            "format": "haic-drq-geometry-mix-study-v1",
            "environment": {"partition": "TRAIN", "geometry_seeds": [51]},
            "diagnostic_pool": {"partition": "TRAIN-DIAGNOSTIC", "geometry_seeds": [52]},
            "training_pool": {"partition": "TRAIN", "geometry_seeds": [53]},
            "seeds_by_source": [{"learner_seed": 0, "geometry_seed": 103000}],
            "runs": [{"learner_seed": 0, "geometry_seed": 103000}],
        }
        self.add("protocol", self.mix_path, self.mix)

    @staticmethod
    def partitions(first: int) -> dict:
        return {name: {"seeds": [first + index]} for index, name in enumerate(
            ("screen", "confirmation", "blind"))}

    def add(self, kind: str, path: str, value: dict | str) -> None:
        data = (json.dumps(value, sort_keys=True) if isinstance(value, dict) else value).encode()
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.sources[kind][path] = hashlib.sha256(data).hexdigest()

    def run_audit(self, seed: int = 77) -> dict:
        return audit.audit_dreamerv3_p1_seeds(
            "dreamerv3-p1-synthetic", [{"track_id": 1, "geometry_seed": seed}],
            repo_root=self.root, protocol_sources=self.sources["protocol"],
            prior_audit_sources=self.sources["prior_seed_audit"],
            training_ledgers=self.sources["training_ledger"],
            prior_collection_ledgers=self.sources["prior_collection_ledger"],
        )

    def add_g0(self) -> int:
        seed = 4272000001
        cells = [{"track_id": 1, "geometry_seed": seed, "partition": "TRAIN", "obstacles": True}]
        receipt = {
            "format": "haic-rlpd-g0-seed-audit-v1", "status": "no_known_recorded_overlap",
            "passed": True, "collisions": [], "ambiguities": [], "protocol_frozen": False,
            "freshness_claim": audit.FRESHNESS_LIMITATION,
            "candidate_seeds": [seed], "candidate_seeds_sha256": audit._digest_json([seed]),
            "cells": cells, "source_inventory": [{"path": "old", "sha256": "0" * 64}],
            "experiment_content_inventory": [{"path": "historical", "sha256": "1" * 64}],
        }
        for name in ("source_inventory", "experiment_content_inventory"):
            receipt[f"{name}_sha256"] = audit._digest_json(receipt[name])
        self.add("prior_seed_audit", audit.G0_AUDIT, receipt)
        protocol = {"format": "haic-rlpd-g0-diagnostic-v1", "status": "frozen",
                    "partition": "TRAIN", "geometry_audit_path": audit.G0_AUDIT,
                    "geometry_audit_sha256": self.sources["prior_seed_audit"][audit.G0_AUDIT],
                    "actors": [{"id": "actor-a"}], "cells": cells}
        self.add("protocol", audit.G0_PROTOCOL, protocol)
        claim_path = f"{audit.G0_CLAIMS}/{receipt['candidate_seeds_sha256']}.json"
        claim = {"format": "haic-rlpd-g0-geometry-claim-v1", "status": "reserved-once",
                 "geometry_audit_sha256": self.sources["prior_seed_audit"][audit.G0_AUDIT],
                 "protocol_sha256": self.sources["protocol"][audit.G0_PROTOCOL],
                 "geometry_seeds": [seed], "output_root": audit.G0_RUN}
        self.add("prior_seed_audit", claim_path, claim)
        del self.sources["prior_seed_audit"][claim_path]
        cells_path = f"{audit.G0_RUN}/cells.jsonl"
        self.add("training_ledger", cells_path, json.dumps({
            "track_id": 1, "geometry_seed": seed, "partition": "TRAIN", "actor_id": "actor-a",
        }) + "\n")
        del self.sources["training_ledger"][cells_path]
        self.add("prior_seed_audit", f"{audit.G0_RUN}/manifest.json", {
            "format": "haic-rlpd-g0-diagnostic-result-v1", "protocol_path": audit.G0_PROTOCOL,
            "protocol_sha256": self.sources["protocol"][audit.G0_PROTOCOL],
            "geometry_audit_sha256": self.sources["prior_seed_audit"][audit.G0_AUDIT],
            "geometry_count": 1, "cell_count": 1,
            "cells_sha256": hashlib.sha256((self.root / cells_path).read_bytes()).hexdigest(),
        })
        del self.sources["prior_seed_audit"][f"{audit.G0_RUN}/manifest.json"]
        return seed

    def test_r6_rng_schedule_is_not_a_geometry_allocation(self) -> None:
        ids: dict[int, list[dict[str, str]]] = {}
        audit._protocol(self.mix, self.mix_path, ids, self.sources)
        self.assertEqual(set(ids), {51, 52, 53})
        self.assertNotIn(103000, ids)

    def test_real_frozen_r6_schema_parses(self) -> None:
        root = Path(__file__).resolve().parents[1]
        record = json.loads((root / self.mix_path).read_text(encoding="utf-8"))
        ids: dict[int, list[dict[str, str]]] = {}
        audit._protocol(record, self.mix_path, ids, self.sources)
        self.assertIn(3910800001, ids)
        self.assertNotIn(103000, ids)

    def test_changed_rng_schedule_or_unknown_geometry_field_fails(self) -> None:
        record = copy.deepcopy(self.mix)
        record["runs"][0]["geometry_seed"] += 1
        with self.assertRaisesRegex(audit.SeedAuditError, "RNG differs"):
            audit._protocol(record, self.mix_path, {}, self.sources)
        record = copy.deepcopy(self.mix)
        record["new_pool"] = {"geometry_seeds": [4242]}
        with self.assertRaisesRegex(audit.SeedAuditError, "unknown geometry seed field"):
            audit._protocol(record, self.mix_path, {}, self.sources)

    def test_all_declared_sources_parse_but_never_authorize_collection(self) -> None:
        receipt = self.run_audit()
        self.assertFalse(receipt["passed"])
        self.assertFalse(receipt["inventory_complete"])
        self.assertEqual(receipt["matched_collisions"], [])
        self.assertEqual(receipt["parse_errors"], [])
        self.assertEqual(len(receipt["source_evidence"]), 7)
        self.assertEqual(receipt["proposed_seeds"], [77])
        self.assertEqual(receipt["inventory_blockers"], [audit.INVENTORY_BLOCKER, audit.R5_BLOCKER])
        self.assertFalse(self.run_audit(10)["passed"])
        self.assertEqual(self.run_audit(10)["matched_collisions"][0]["seed"], 10)

    def test_cli_exits_nonzero_even_when_inputs_parse(self) -> None:
        args = ["--repo-root", str(self.root), "--study-id", "dreamerv3-p1-synthetic",
                "--cell", "1:77"]
        for kind, flag in (("protocol", "protocol"), ("prior_seed_audit", "prior-audit"),
                           ("training_ledger", "training-ledger"),
                           ("prior_collection_ledger", "prior-collection-ledger")):
            for name, digest in self.sources[kind].items():
                args.extend((f"--{flag}", f"{name}={digest}"))
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(audit.main(args), 1)
        self.assertIs(json.loads(output.getvalue())["passed"], False)

    def test_missing_known_source_or_modified_ledger_is_rejected(self) -> None:
        del self.sources["protocol"][self.mix_path]
        with self.assertRaisesRegex(audit.SeedAuditError, "missing known cross-lane"):
            self.run_audit()
        self.add("protocol", self.mix_path, self.mix)
        ledger = "runs/drq-source/episodes.jsonl"
        (self.root / ledger).write_text('{"event":"reset","track_id":1,"seed":999}\n')
        with self.assertRaisesRegex(audit.SeedAuditError, "SHA-256 mismatch"):
            self.run_audit()

    def test_held_out_path_and_oversized_source_are_rejected(self) -> None:
        held_out = "runs/blind-episodes/episodes.jsonl"
        with self.assertRaisesRegex(audit.SeedAuditError, "outside ID-only allowlist"):
            audit._source(self.root, held_out, "0" * 64, "training_ledger")
        oversized = "runs/synthetic-training/episodes.jsonl"
        target = self.root / oversized
        target.parent.mkdir(parents=True)
        with target.open("wb") as stream:
            stream.truncate(32 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(audit.SeedAuditError, "exceeds 32 MiB"):
            audit._source(self.root, oversized, "0" * 64, "training_ledger")

    def test_protocol_with_duplicate_json_keys_is_rejected(self) -> None:
        path = self.root / self.dreamer_path
        path.write_text('{"format":"x","format":"y"}')
        self.sources["protocol"][self.dreamer_path] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(audit.SeedAuditError, "duplicate JSON key"):
            self.run_audit()

    def test_g0_consumed_cells_and_claim_are_exclusions(self) -> None:
        seed = self.add_g0()
        receipt = self.run_audit(seed)
        self.assertFalse(receipt["passed"])
        self.assertFalse(receipt["inventory_complete"])
        self.assertEqual([row["seed"] for row in receipt["matched_collisions"]], [seed])
        self.assertIn(audit.G0_PROTOCOL, [row["path"] for row in receipt["matched_collisions"][0]["sources"]])
        self.assertEqual(len(receipt["auxiliary_train_metadata"]), 3)

    def test_missing_g0_claim_or_mismatched_consumed_cells_stops(self) -> None:
        self.add_g0()
        claim = next((self.root / audit.G0_CLAIMS).glob("*.json"))
        claim.unlink()
        with self.assertRaisesRegex(audit.SeedAuditError, "claim inventory"):
            self.run_audit()
        self.add_g0()
        path = self.root / audit.G0_RUN / "cells.jsonl"
        path.write_text(path.read_text().replace('"geometry_seed": 4272000001', '"geometry_seed": 4272000099'))
        with self.assertRaisesRegex(audit.SeedAuditError, "cells manifest"):
            self.run_audit()

    def test_all_experiment_json_is_snapshotted_and_untyped_id_is_rejected(self) -> None:
        receipt = self.run_audit()
        self.assertEqual(len(receipt["experiment_content_inventory"]), len(list((self.root / "experiments").glob("*.json"))))
        self.add("prior_seed_audit", "experiments/synthetic-result.json", {"note": "other number"})
        del self.sources["prior_seed_audit"]["experiments/synthetic-result.json"]
        changed = self.run_audit()
        self.assertNotEqual(changed["experiment_content_inventory_sha256"],
                            receipt["experiment_content_inventory_sha256"])
        self.add("prior_seed_audit", "experiments/synthetic-result.json", {"note": "candidate 77"})
        del self.sources["prior_seed_audit"]["experiments/synthetic-result.json"]
        with self.assertRaisesRegex(audit.SeedAuditError, "candidate ID in untyped"):
            self.run_audit()

    def test_catalog_file_removal_and_content_drift_are_visible(self) -> None:
        original = self.run_audit()
        extra = "experiments/metadata-result.json"
        self.add("prior_seed_audit", extra, {"note": "recorded without candidate"})
        del self.sources["prior_seed_audit"][extra]
        added = self.run_audit()
        self.assertNotEqual(original["experiment_catalog_sha256"], added["experiment_catalog_sha256"])
        (self.root / extra).unlink()
        self.assertEqual(self.run_audit()["experiment_catalog_sha256"], original["experiment_catalog_sha256"])
        (self.root / self.mix_path).unlink()
        with self.assertRaisesRegex(audit.SeedAuditError, "declared experiment metadata missing"):
            self.run_audit()

    def test_held_out_episodes_are_never_in_train_inventory(self) -> None:
        heldout = self.root / "runs/blind-evaluation/episodes.jsonl"
        heldout.parent.mkdir(parents=True)
        heldout.write_text("not JSON; not a TRAIN metadata source")
        receipt = self.run_audit()
        self.assertFalse(receipt["passed"])
        self.assertNotIn("runs/blind-evaluation/episodes.jsonl", receipt["read_paths"])

    def test_omitted_protocol_or_train_ledger_and_ambiguous_schema_stop(self) -> None:
        self.add("protocol", "experiments/pixel-rlpd-offpolicy-pilot-v1.json", {
            "format": "haic-pixel-rlpd-study-v1", "reserved_training_seeds": [77]})
        del self.sources["protocol"]["experiments/pixel-rlpd-offpolicy-pilot-v1.json"]
        with self.assertRaisesRegex(audit.SeedAuditError, "undeclared current experiment protocol"):
            self.run_audit()
        (self.root / "experiments/pixel-rlpd-offpolicy-pilot-v1.json").unlink()
        self.add("training_ledger", "runs/new-train/episodes.jsonl",
                 '{"event":"reset","track_id":1,"seed":77}\n')
        del self.sources["training_ledger"]["runs/new-train/episodes.jsonl"]
        with self.assertRaisesRegex(audit.SeedAuditError, "omitted/new TRAIN source"):
            self.run_audit()
        (self.root / "runs/new-train/episodes.jsonl").unlink()
        altered = copy.deepcopy(self.mix)
        altered["unreviewed"] = {"geometry_seed_ids": [77]}
        self.add("protocol", self.mix_path, altered)
        with self.assertRaisesRegex(audit.SeedAuditError, "unknown geometry seed field"):
            self.run_audit()

    def test_forged_r5_receipt_does_not_get_an_exception(self) -> None:
        self.add("protocol", audit.R5_PROTOCOL, self.mix)
        self.add("training_ledger", f"{audit.R5_RUN}/learner-0-uniform/episodes.jsonl",
                 '{"event":"reset","track_id":1,"seed":77}\n')
        self.add("prior_seed_audit", audit.R5_ERRATUM, {"format": "haic-rlpd-g0-r5-receipt-erratum-v1"})
        del self.sources["prior_seed_audit"][audit.R5_ERRATUM]
        with self.assertRaisesRegex(audit.SeedAuditError, "r5 protocol or narrow historical erratum"):
            self.run_audit()

    def test_real_g0_and_r5_chains_are_primary_byte_checked(self) -> None:
        root = Path(__file__).resolve().parents[1]
        _, contents = audit._experiment_catalog(root)
        ids: dict[int, list[dict[str, str]]] = {}
        self.assertEqual(len(audit._g0_metadata(root, contents, ids)), 3)
        self.assertIn(4272000001, ids)
        self.assertIn(4272000012, ids)
        ledger = f"{audit.R5_RUN}/learner-0-uniform/episodes.jsonl"
        chain = audit._r5_chain(root, contents, {ledger: audit.R5_PINS[ledger]}, ids)
        self.assertEqual(len(chain), 5)
        self.assertIn(3910800035, ids)
        self.assertFalse(self.run_audit()["passed"])


if __name__ == "__main__":
    unittest.main()
