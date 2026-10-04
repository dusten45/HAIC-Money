"""Synthetic-only TD allocation audit tests. No simulator or protected outcomes."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import audit_tdmpc2_train_diag_seeds as audit


class TdDiagnosticInventoryTests(unittest.TestCase):
    CANDIDATE = "experiments/tdmpc2-future-diagnostic.json"

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cells = [
            {"partition": "TRAIN", "track_id": track, "geometry_seed": 3000 + (track - 1) * 6 + i,
             "obstacles": True}
            for track in range(1, 5) for i in range(6)
        ]
        self.write(self.CANDIDATE, {"format": audit.FORMAT,
                                    "purpose": "TRAIN-DIAGNOSTIC", "cells": self.cells})
        reserved = list(range(3910800001, 3910800513))
        self.write(audit.CATALOG_PROTOCOL, {"format": "haic-drq-training-geometry-protocol-v1",
                                             "candidate_seeds": reserved,
                                             "exclusion_seed_ids": {"blind": [88000],
                                                                    "heldout": [88001], "reserved": [88002]}})
        self.write(audit.CATALOG, {"protocol_sha256": self.sha(audit.CATALOG_PROTOCOL),
                                    "seed_audit": {"proposed_seeds": reserved},
                                    "train": [{"geometry_seed": seed} for seed in reserved[:120]],
                                    "train_diagnostic": [{"geometry_seed": seed} for seed in reserved[120:136]]})
        g0 = [{"partition": "TRAIN", "track_id": 1, "geometry_seed": 4272000001 + i,
               "obstacles": True} for i in range(12)]
        self.write(audit.G0_PROTOCOL, {"cells": g0})
        self.write(audit.G0_AUDIT, {"candidate_seeds": [c["geometry_seed"] for c in g0]})
        self.write(audit.G0_CELLS, self.rows([{"geometry_seed": c["geometry_seed"],
                                               "partition": "TRAIN", "track_id": 1} for c in g0]))
        self.write(f"{audit.G0_CLAIMS}/.lock", b"")
        self.write(f"{audit.G0_CLAIMS}/claim.json",
                   {"geometry_seeds": [c["geometry_seed"] for c in g0]})
        self.write(f"{audit.TRAIN_CLAIMS}/.gitkeep", b"")
        for path in audit.TD_JOURNALS:
            self.write(path, self.rows([{"event": "start", "protocol_sha256": "0" * 64},
                                         {"event": "reset_intent", "geometry_seed": 90001,
                                          "track_id": 1},
                                         {"event": "reset", "geometry_seed": 90001, "track_id": 1},
                                         {"event": "episode", "geometry_seed": 90001,
                                          "track_id": 1}]))
        self.write("runs/other-lane/episodes.jsonl", self.rows([
            {"event": "reset", "seed": 90002, "track_id": 2},
            {"event": "end", "seed": 90002, "track_id": 2},
        ]))
        self.write("experiments/pixel-rlpd-protocol-v1.json",
                   {"format": "haic-pixel-rlpd-study-v1",
                    "reserved_training_seeds": [90003],
                    "partitions": {"confirmation": {"seeds": [88003]},
                                   "blind": {"seeds": [88004]}}})

    def write(self, path: str, value: dict | bytes) -> None:
        dest = self.root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())

    def sha(self, path: str) -> str:
        return hashlib.sha256((self.root / path).read_bytes()).hexdigest()

    @staticmethod
    def rows(values: list[dict]) -> bytes:
        return b"".join(json.dumps(row).encode() + b"\n" for row in values)

    def audit(self, **kwargs) -> dict:
        return audit.audit_tdmpc2_train_diag_seeds(self.CANDIDATE, repo_root=self.root, **kwargs)

    def test_clean_synthetic_inventory_still_blocks_and_never_claims(self) -> None:
        report = self.audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["collisions"], [])
        self.assertEqual([b["field"] for b in report["blockers"]], ["coverage"])
        self.assertEqual(report["blind_episode_reads"], 0)
        self.assertFalse(report["reservation_or_claim"])
        self.assertEqual(len(report["cells"]), 24)
        self.assertEqual(len({c["geometry_seed"] for c in report["cells"]}), 24)
        self.assertEqual(report["candidate_sha256"], self.sha(self.CANDIDATE))
        self.assertFalse((self.root / audit.TRAIN_CLAIMS / "seed-3000.json").exists())

    def test_cross_track_geometry_alias_is_rejected_before_audit(self) -> None:
        cells = [dict(c) for c in self.cells]
        cells[6]["geometry_seed"] = cells[0]["geometry_seed"]
        self.write(self.CANDIDATE, {"format": audit.FORMAT,
                                    "purpose": "TRAIN-DIAGNOSTIC", "cells": cells})
        with self.assertRaisesRegex(audit.InventoryError, "cross-track geometry alias"):
            self.audit()

    def test_input_must_be_explicit_frozen_shape_not_seed_generation(self) -> None:
        for bad in ({"format": audit.FORMAT, "purpose": "TRAIN-DIAGNOSTIC",
                     "seed_start": 3000, "cells": self.cells},
                    {"format": audit.FORMAT, "purpose": "TRAIN-DIAGNOSTIC",
                     "cells": self.cells[:-1]},
                    {"format": audit.FORMAT, "purpose": "TRAIN-DIAGNOSTIC",
                     "cells": [dict(self.cells[0], obstacles=False)] + self.cells[1:]}):
            with self.subTest(bad=bad):
                self.write(self.CANDIDATE, bad)
                with self.assertRaises(audit.InventoryError):
                    self.audit()

    def test_drq_512_reservation_cross_track_and_selected_diagnostic(self) -> None:
        for seed in (3910800001, 3910800121, 3910800501):
            with self.subTest(seed=seed):
                cells = [dict(c) for c in self.cells]
                cells[18]["geometry_seed"] = seed  # track 4, reserved on other tracks.
                self.write(self.CANDIDATE, {"format": audit.FORMAT,
                                            "purpose": "TRAIN-DIAGNOSTIC", "cells": cells})
                report = self.audit()
                hit = next(c for c in report["collisions"] if c["geometry_seed"] == seed)
                self.assertEqual(hit["track_id"], 4)
                self.assertTrue(any(s["path"] == audit.CATALOG_PROTOCOL for s in hit["sources"]))

    def test_protected_id_metadata_only_and_no_protected_outcome_reads(self) -> None:
        cells = [dict(c) for c in self.cells]
        cells[0]["geometry_seed"] = 88003
        self.write(self.CANDIDATE, {"format": audit.FORMAT,
                                    "purpose": "TRAIN-DIAGNOSTIC", "cells": cells})
        self.write("runs/other-lane/evaluations/blind/episodes.jsonl", b"DO NOT READ\n")
        self.write("runs/other-lane/privateEpisodes/episodes.jsonl", b"DO NOT READ\n")
        self.write("experiments/other-lane-blind-result-v1.json", b"DO NOT READ\n")
        self.write("evaluations/protected/episodes.jsonl", b"DO NOT READ\n")
        original = Path.read_bytes

        def guarded(path: Path) -> bytes:
            self.assertFalse(any(marker in str(path).lower() for marker in
                                 ("privateepisodes", "/evaluations/", "blind-result")))
            return original(path)

        with patch.object(Path, "read_bytes", guarded):
            report = self.audit()
        hit = next(c for c in report["collisions"] if c["geometry_seed"] == 88003)
        self.assertTrue(any("pixel-rlpd-protocol-v1.json" in s["path"] for s in hit["sources"]))
        self.assertEqual(report["blind_episode_reads"], 0)

    def test_g0_claim_g1_registry_and_train_ledger_across_tracks(self) -> None:
        cells = [dict(c) for c in self.cells]
        cells[0]["geometry_seed"] = 4272000001
        cells[6]["geometry_seed"] = 90002
        cells[12]["geometry_seed"] = 90004
        self.write(self.CANDIDATE, {"format": audit.FORMAT,
                                    "purpose": "TRAIN-DIAGNOSTIC", "cells": cells})
        self.write("runs/other-lane/episodes.jsonl", self.rows([
            {"event": "reset", "seed": 90002, "track_id": 4},
            {"event": "end", "seed": 90002, "track_id": 4},
        ]))
        self.write(f"{audit.TRAIN_CLAIMS}/seed-90004.json", {
            "format": "haic-train-seed-claim-v1", "study_id": "rlpd-g1", "protocol_id": None,
            "protocol_path": None, "protocol_sha256": None, "partition": "TRAIN",
            "track_id": 1, "geometry_seed": 90004, "obstacles": True,
            "status": "reserved", "claimed_at_utc": "2026-09-28T00:00:00Z",
            "audit_source": "synthetic", "audit_digest_sha256": "0" * 64,
        })
        report = self.audit()
        sources = {row["geometry_seed"]: {s["path"] for s in row["sources"]}
                   for row in report["collisions"]}
        self.assertIn(audit.G0_PROTOCOL, sources[4272000001])
        self.assertIn("runs/other-lane/episodes.jsonl", sources[90002])
        self.assertIn(f"{audit.TRAIN_CLAIMS}/seed-90004.json", sources[90004])

    def test_td_partial_and_intent_without_reset_are_preserved(self) -> None:
        for suffix in ("v1", "v2"):
            with self.subTest(suffix=suffix):
                path = f"runs/tdmpc2-long-20260928-{suffix}/training.jsonl"
                self.write(path, self.rows([
                    {"event": "start", "protocol_sha256": "0" * 64},
                    {"event": "reset_intent", "geometry_seed": 3000, "track_id": 3},
                    *([{"event": "reset", "geometry_seed": 3000, "track_id": 3}]
                      if suffix == "v1" else []),
                    {"event": "partial", "decisions": 17},
                ]))
                report = self.audit()
                hit = next(c for c in report["collisions"] if c["geometry_seed"] == 3000)
                self.assertIn(path, {s["path"] for s in hit["sources"]})
                self.write(path, self.rows([{"event": "start", "protocol_sha256": "0" * 64},
                                            {"event": "reset_intent", "geometry_seed": 90001,
                                             "track_id": 1},
                                            {"event": "reset", "geometry_seed": 90001,
                                             "track_id": 1},
                                            {"event": "episode", "geometry_seed": 90001,
                                             "track_id": 1}]))

    def test_torn_td_journal_keeps_all_intact_reset_intents(self) -> None:
        path = audit.TD_JOURNALS[0]
        prefix = self.rows([{"event": "start", "protocol_sha256": "0" * 64},
                            {"event": "reset_intent", "geometry_seed": 3000, "track_id": 1},
                            {"event": "reset", "geometry_seed": 3000, "track_id": 1},
                            {"event": "episode", "geometry_seed": 3000, "track_id": 1},
                            {"event": "reset_intent", "geometry_seed": 3001, "track_id": 2}])
        for suffix in (b'{"event":"reset",', b'{"event":"reset","geometry_seed":3001,"track_id":2}'):
            with self.subTest(suffix=suffix):
                self.write(path, prefix + suffix)
                report = self.audit()
                self.assertEqual(report["status"], "BLOCKED")
                self.assertTrue({3000, 3001} <= {c["geometry_seed"] for c in report["collisions"]})
                self.assertTrue(any(b["path"] == path for b in report["blockers"]))

    def test_unknown_range_candidate_blocks_and_disjoint_warns(self) -> None:
        path = "experiments/other-train-allocation.json"
        for value in ({"road_seed_range": {"start": 2990, "end": 3000}},
                       {"road_seed_range": {"start": 2999, "count": 2}},
                       {"road_seed_range": [2990, 3000]},
                       {"road_seed_range": {"start": "unbounded", "end": 5000}},
                       {"road_seed_range": {"start": 2990, "end": 4000}},
                       {"road_seed_range": {"min": 2990, "max": 4000}},
                       {"road_seed_min": 2990, "road_seed_max": 4000},
                        {"training_seed_interval": {"from": 2990, "to": 4000}},
                        {"training_seed_interval": {"start": 5000, "end": 6000,
                                                     "from": 2990, "to": 4000}},
                        {"training_seed_span": {"from": 2990, "to": 4000}},
                        {"seedStart": 2990, "seedEnd": 4000},
                       {"training_seed_interval": {"START": 5000, "start": 2990, "end": 4000}},
                       {"road_seed_range": "unreviewed interval"},
                       {"road_seed_start": 2990},
                       {"road_seed_start": 5000},
                       {"road_seed_count": 10}):
            with self.subTest(value=value):
                self.write(path, value)
                report = self.audit()
                self.assertTrue(any(b["path"] == path for b in report["blockers"]))
        self.write(path, {"road_seed_range": {"start": 5000, "end": 6000}})
        report = self.audit()
        self.assertFalse(any(b["path"] == path for b in report["blockers"]))
        self.assertTrue(any(w["path"] == path for w in report["provenance_warnings"]))
        for value in ({"road_seed_min": 5000, "road_seed_max": 6000},
                      {"road_seed_start": 5000, "road_seed_end": 6000},
                       {"road_seed_start": 5000, "road_seed_count": 3000},
                       {"seedStart": 5000, "seedEnd": 6000},
                      {"road_seed_range": {"start": 5000, "count": 3000}},
                      {"training_seed_interval": {"from": 5000, "to": 6000}}):
            with self.subTest(disjoint=value):
                self.write(path, value)
                report = self.audit()
                self.assertFalse(any(b["path"] == path for b in report["blockers"]))
                self.assertTrue(any(w["path"] == path for w in report["provenance_warnings"]))

    def test_run_local_id_only_metadata_catches_unique_cross_track_road(self) -> None:
        for name, value in (("config.json", {"config": {"excluded_training_seeds": [3000],
                                                            "seed": 3001}}),
                            ("protocol.json", {"partitions": {"screen": {"seeds": [3000]}}}),
                            ("study_protocol.json", {"training_geometry_seeds": [3000]})):
            with self.subTest(name=name):
                path = f"runs/independent-training/{name}"
                self.write(path, value)
                report = self.audit()
                hit = next(c for c in report["collisions"] if c["geometry_seed"] == 3000)
                self.assertIn(path, {source["path"] for source in hit["sources"]})
                self.assertNotIn(3001, {c["geometry_seed"] for c in report["collisions"]})

    def test_run_local_sampler_rng_is_ambiguous_not_proven_road(self) -> None:
        for name in ("config.json", "protocol.json", "study_protocol.json",
                     "run-config.json", "run_config.json"):
            with self.subTest(name=name):
                path = f"runs/unreviewed-train-source/{name}"
                self.write(path, {"config": {"seed": 3000, "geometry_seed": 3001,
                                             "rng_seeds": {"track_seed": 3002}}})
                report = self.audit()
                self.assertEqual(report["collisions"], [])
                self.assertTrue(any(b["path"] == path and "unreviewed run-local" in b["reason"]
                                    for b in report["blockers"]))
                self.assertEqual(report["status"], "BLOCKED")

    def test_explicit_road_exclusion_survives_ambiguous_sampler_in_same_config(self) -> None:
        path = "runs/unreviewed-train-source/config.json"
        self.write(path, {"config": {"excluded_training_seeds": [3000],
                                     "seed": 3000, "geometry_seed": 3001}})
        report = self.audit()
        hit = next(c for c in report["collisions"] if c["geometry_seed"] == 3000)
        self.assertTrue(any(s["path"] == path and "TRAIN exclusion" in s["field"]
                            for s in hit["sources"]))
        self.assertNotIn(3001, {c["geometry_seed"] for c in report["collisions"]})
        self.assertTrue(any(b["path"] == path and "unreviewed run-local" in b["reason"]
                            for b in report["blockers"]))

    def test_run_local_declared_road_roles_and_contradictory_partitions(self) -> None:
        path = "runs/independent-training/study_protocol.json"
        self.write(path, {"training_geometry_seeds": [3000],
                          "partitions": {"screen": {"track_ids": [2], "seeds": [3001]}}})
        report = self.audit()
        for road, role in ((3000, "declared TRAIN road"), (3001, "declared screen")):
            hit = next(c for c in report["collisions"] if c["geometry_seed"] == road)
            self.assertTrue(any(s["path"] == path and role in s["field"] for s in hit["sources"]))
        self.assertFalse(any(b["path"] == path for b in report["blockers"]))
        self.write(path, {"partitions": {"screen": {"seeds": [3000], "partition": "BLIND"}}})
        report = self.audit()
        self.assertEqual(report["collisions"], [])
        self.assertTrue(any(b["path"] == path and "contradictory partition" in b["reason"]
                            for b in report["blockers"]))
        self.write(path, {"training_geometry_seeds": [3000], "seed_role": "SAMPLER RNG"})
        report = self.audit()
        self.assertEqual(report["collisions"], [])
        self.assertTrue(any(b["path"] == path and "contradictory run-local road" in b["reason"]
                            for b in report["blockers"]))

    def test_run_local_free_text_road_alias_is_not_silently_typed(self) -> None:
        path = "runs/unreviewed-train-source/config.json"
        self.write(path, {"note": "geometry_seed:3000 from unreviewed sampler log"})
        report = self.audit()
        self.assertEqual(report["collisions"], [])
        self.assertTrue(any(b["path"] == path and "unreviewed run-local" in b["reason"]
                            for b in report["blockers"]))

    def test_diagnostic_named_experiment_protocol_is_not_skipped(self) -> None:
        path = "experiments/other-train-diagnostic-protocol.json"
        self.write(path, {"partitions": {"train_diagnostic": {"seeds": [3000]},
                                          "blind": {"seeds": [3001]}}})
        report = self.audit()
        for road, role in ((3000, "train_diagnostic"), (3001, "blind")):
            hit = next(c for c in report["collisions"] if c["geometry_seed"] == road)
            self.assertTrue(any(s["path"] == path and role in s["field"] for s in hit["sources"]))
        self.assertFalse(any(b["path"] == path for b in report["blockers"]))

    def test_run_local_protocol_interior_interval_blocks_without_literal_id(self) -> None:
        path = "runs/independent-training/protocol.json"
        self.write(path, {"training_road_seed_range": {"start": 2990, "end": 4000}})
        report = self.audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["collisions"], [])
        self.assertTrue(any(b["path"] == path for b in report["blockers"]))

    def test_residual_partial_road_not_in_completed_episode_ledger(self) -> None:
        folder = f"{audit._RESIDUAL}/iteration-2"
        self.write(f"{folder}/episodes.jsonl", self.rows([
            {"episode": 1, "geometry_seed": 91000, "track_id": 2, "decisions": 8},
        ]))
        self.write(f"{folder}/result.json", {
            "completed_episodes": 1, "environment_decisions": 13,
            "option_action_counts": {"KEEP": 13, "STEER_MINUS": 0, "STEER_PLUS": 0,
                                     "COAST": 0, "BRAKE": 0},
            "status": "completed_bounded_training_pilot",
            "incomplete_final_episode": {"geometry_seed": 3000, "track_id": 4,
                                         "decisions": 5,
                                         "status": "budget_interrupted; not an episode outcome"},
        })
        report = self.audit()
        hit = next(c for c in report["collisions"] if c["geometry_seed"] == 3000)
        self.assertEqual(hit["track_id"], 1)  # ID aliases across tracks.
        self.assertTrue(any(s["path"] == f"{folder}/result.json" and "actual partial" in s["field"]
                            for s in hit["sources"]))
        self.assertTrue(any(b["path"] == f"{folder}/result.json" and b["field"] == "source binding"
                            for b in report["blockers"]))
        self.assertEqual(report["status"], "BLOCKED")

    def test_residual_retry_counter_correction_and_torn_ledger_preserve_partial(self) -> None:
        folder = f"{audit._RESIDUAL}/iteration-1-retry"
        self.write(f"{folder}/episodes.jsonl", self.rows([
            {"episode": 1, "geometry_seed": 91000, "track_id": 2, "decisions": 8},
        ]))
        self.write(f"{folder}/result.json", {
            "completed_episodes": 1, "environment_decisions": 14,
            "option_action_counts": {"KEEP": 13, "STEER_MINUS": 0, "STEER_PLUS": 0,
                                     "COAST": 0, "BRAKE": 0},
            "status": "completed_bounded_training_pilot",
            "incomplete_final_episode": {"geometry_seed": 3000, "track_id": 4,
                                         "decisions": 5,
                                         "status": "budget_interrupted; not an episode outcome"},
        })
        self.write(f"{folder}/counter-audit.json", {
            "status": "post-run-accounting-correction; original artifacts preserved",
            "completed_episode_decisions": 8, "incomplete_final_episode_decisions": 5,
            "actual_collector_step_calls": 13, "sum_of_per_option_action_counts": 13,
            "result_json_environment_decisions_field": 14,
        })
        self.assertFalse(any(b["path"] == f"{folder}/result.json" and b["field"] == "TRAIN ledger/marker"
                             for b in self.audit()["blockers"]))
        self.write(f"{folder}/episodes.jsonl", self.rows([
            {"episode": 1, "geometry_seed": 3002, "track_id": 2, "decisions": 8},
        ]) + b"{torn")
        report = self.audit()
        self.assertTrue({3000, 3002} <= {c["geometry_seed"] for c in report["collisions"]})
        self.assertTrue(any(b["path"] == f"{folder}/result.json" for b in report["blockers"]))

    def test_positive_interaction_without_any_road_is_ambiguous(self) -> None:
        folder = f"{audit._RESIDUAL}/iteration-1"
        self.write(f"{folder}/episodes.jsonl", b"")
        self.write(f"{folder}/failure.json", {
            "environment_decisions_completed": 2,
            "status": "implementation_smoke_failed; not a trained model or evaluation result",
        })
        report = self.audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["collisions"], [])
        self.assertTrue(any(b["path"] == f"{folder}/failure.json" and "no recorded road" in b["reason"]
                            for b in report["blockers"]))

    def test_zero_action_reset_remains_actual_road_not_released(self) -> None:
        path = "runs/tdmpc2-exploration-20260928-v1/failure.json"
        protocol = "experiments/tdmpc2-exploration-v1.json"
        self.write(protocol, {"format": "haic-tdmpc2-exploration-v1"})
        self.write(path, {"status": "aborted_after_first_reset_before_first_action",
                          "reset_cell": {"track_id": 4, "geometry_seed": 3000},
                          "attempted_episodes": 1, "decisions": 0,
                          "protocol_sha256": self.sha(protocol)})
        report = self.audit()
        hit = next(c for c in report["collisions"] if c["geometry_seed"] == 3000)
        self.assertTrue(any(s["path"] == path and "actual reset; 0 decisions" in s["field"]
                            for s in hit["sources"]))
        self.assertFalse(any(b["path"] == path for b in report["blockers"]))

    def final_source_fixture(self, decisions: int = 2, step_road: int = 3000) -> tuple[str, str, str, dict, dict]:
        folder = f"{audit._FINAL_SOURCE}/seed0"
        ledger = f"{folder}/episodes.jsonl"
        self.write(ledger, self.rows([
            {"event": "reset", "track_id": 4, "geometry_seed": 3000, "seed": 3000,
             "source_seed": 0, "episode_id": 0, "partition": "TRAIN", "collection_step": 0},
            {"event": "capped_partial", "track_id": 4, "geometry_seed": 3000, "seed": 3000,
             "source_seed": 0, "episode_id": 0, "collection_step": decisions, "steps": decisions},
        ]))
        steps = f"{folder}/steps.jsonl"
        self.write(steps, self.rows([{"episode_id": 0, "track_id": 4,
                                     "geometry_seed": step_road}]) * decisions)
        r6 = "experiments/drqv2-geometry-mix-v1-r6.json"
        self.write(r6, {"format": "haic-drq-geometry-mix-study-v1"})
        original_ledger = "runs/20260922-drq-augmentation-pad-v1-restart/control-seed0/episodes.jsonl"
        self.write(original_ledger, self.rows([
            {"event": "reset", "track_id": 1, "seed": 91000},
            {"event": "end", "track_id": 1, "seed": 91000},
        ]))
        self.write(audit._FINAL_SOURCE_R7, {
            "source_replay": {"0": {"episode_ledger_path": original_ledger,
                                    "episode_ledger_sha256": self.sha(original_ledger)}},
        })
        self.write(audit._FINAL_SOURCE_COLLECTION, {
            "r6_protocol_path": r6, "r6_protocol_sha256": self.sha(r6),
            "decisions": 100000, "capacity": 100000,
            "r7_protocol_path": audit._FINAL_SOURCE_R7,
            "r7_protocol_sha256": self.sha(audit._FINAL_SOURCE_R7),
            "catalog_sha256": self.sha(audit.CATALOG),
            "sources": {"0": {"original_ledger_sha256": self.sha(original_ledger), "schedule_sha256": "b",
                              "source_actor_sha256": "c", "source_checkpoint_sha256": "d",
                              "collection_rng_seeds": {"action_noise_seed": 3001}}},
        })
        receipt = {"format": "haic-drq-final-source-pool-v1", "partition": "TRAIN",
                   "source_seed": 0, "catalog_sha256": self.sha(audit.CATALOG),
                   "collection_protocol_sha256": self.sha(audit._FINAL_SOURCE_COLLECTION),
                   "episode_ledger_path": ledger, "episode_ledger_sha256": self.sha(ledger),
                   "step_ledger_path": steps, "step_ledger_sha256": self.sha(steps),
                   "completed": True, "decisions": decisions, "capacity": 100000,
                   "scheduled_episodes_consumed": 1,
                   "geometry_seeds": [3000], "original_ledger_sha256": self.sha(original_ledger),
                   "schedule_sha256": "b",
                   "source_actor_sha256": "c", "source_checkpoint_sha256": "d",
                   "collection_rng_seeds": {"action_noise_seed": 3001}}
        path = f"{folder}/receipt.json"
        self.write(path, receipt)
        source = {
            "r6_protocol_path": r6, "r6_protocol_sha256": self.sha(r6),
            "r7_protocol_path": audit._FINAL_SOURCE_R7,
            "r7_protocol_sha256": self.sha(audit._FINAL_SOURCE_R7),
            "collection_protocol_path": audit._FINAL_SOURCE_COLLECTION,
            "collection_protocol_sha256": self.sha(audit._FINAL_SOURCE_COLLECTION),
            "catalog_sha256": self.sha(audit.CATALOG),
            "source_replay": {"0": {"receipt_path": path, "receipt_sha256": self.sha(path)}},
            "runs": [{"rng_seeds": {key: 3001 for key in audit._FINAL_SOURCE_RNG}}],
        }
        self.write(audit._FINAL_SOURCE_PROTOCOL, source)
        return path, ledger, steps, receipt, source

    def test_final_source_partial_receipt_remains_unverified_and_preserves_reset(self) -> None:
        path, ledger, steps, receipt, source = self.final_source_fixture()
        report = self.audit()
        hit = next(c for c in report["collisions"] if c["geometry_seed"] == 3000)
        self.assertTrue(any(s["path"] == ledger and "actual" in s["field"]
                            for s in hit["sources"]))
        self.assertFalse(any(s["path"] == path for s in hit["sources"]))
        self.assertTrue(any(b["path"] == path and "fixed complete 100k" in b["reason"]
                            for b in report["blockers"]))
        self.assertNotIn(3001, {c["geometry_seed"] for c in report["collisions"]})
        self.assertFalse(any(b["path"] in (path, audit._FINAL_SOURCE_COLLECTION,
                                           audit._FINAL_SOURCE_PROTOCOL) and "sampler" in b["reason"]
                             for b in report["blockers"]))
        receipt["step_ledger_sha256"] = "0" * 64
        self.write(path, receipt)
        source["source_replay"]["0"]["receipt_sha256"] = self.sha(path)
        self.write(audit._FINAL_SOURCE_PROTOCOL, source)
        report = self.audit()
        self.assertTrue(any(b["path"] == path and "step ledger SHA/count mismatch" in b["reason"]
                            for b in report["blockers"]))
        hit = next(c for c in report["collisions"] if c["geometry_seed"] == 3000)
        self.assertFalse(any(s["path"] == path for s in hit["sources"]))
        receipt["step_ledger_sha256"] = self.sha(steps)
        collection = json.loads((self.root / audit._FINAL_SOURCE_COLLECTION).read_text())
        collection["decisions"] = 2
        collection["capacity"] = 2
        self.write(audit._FINAL_SOURCE_COLLECTION, collection)
        receipt["collection_protocol_sha256"] = self.sha(audit._FINAL_SOURCE_COLLECTION)
        self.write(path, receipt)
        source["collection_protocol_sha256"] = self.sha(audit._FINAL_SOURCE_COLLECTION)
        source["source_replay"]["0"]["receipt_sha256"] = self.sha(path)
        self.write(audit._FINAL_SOURCE_PROTOCOL, source)
        report = self.audit()
        self.assertTrue(any(b["path"] == path and "fixed complete 100k" in b["reason"]
                            for b in report["blockers"]))
        hit = next(c for c in report["collisions"] if c["geometry_seed"] == 3000)
        self.assertFalse(any(s["path"] == path for s in hit["sources"]))
        receipt["schedule_sha256"] = "changed-unreconciled-schedule"
        self.write(path, receipt)
        source["source_replay"]["0"]["receipt_sha256"] = self.sha(path)
        self.write(audit._FINAL_SOURCE_PROTOCOL, source)
        report = self.audit()
        self.assertTrue(any(b["path"] == path and "source/schedule binding mismatch" in b["reason"]
                            for b in report["blockers"]))

    def test_final_source_full_count_fake_step_roads_never_verify_visited_receipt(self) -> None:
        path, ledger, steps, _, _ = self.final_source_fixture(decisions=100000, step_road=3001)
        report = self.audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any(b["path"] == path and "schedule parity not independently verified" in b["reason"]
                            for b in report["blockers"]))
        hit = next(c for c in report["collisions"] if c["geometry_seed"] == 3000)
        self.assertTrue(any(s["path"] == ledger for s in hit["sources"]))
        self.assertFalse(any(s["path"] == path for s in hit["sources"]))
        self.assertTrue(any(b["path"] == steps for b in report["blockers"]))

    def test_residual_alias_and_action_counter_tamper_keep_known_tail(self) -> None:
        folder = f"{audit._RESIDUAL}/iteration-2"
        path = f"{folder}/result.json"
        self.write(f"{folder}/episodes.jsonl", self.rows([
            {"episode": 1, "geometry_seed": 91000, "track_id": 2, "decisions": 8,
             "option_counts": {"KEEP": 8}},
        ]))
        result = {
            "completed_episodes": 1, "environment_decisions": 13,
            "option_action_counts": {"KEEP": 13, "STEER_MINUS": 0, "STEER_PLUS": 0,
                                     "COAST": 0, "BRAKE": 0},
            "status": "completed_bounded_training_pilot",
            "incomplete_final_episode": {"geometry_seed": 3000, "track_id": 4,
                                         "decisions": 5,
                                         "status": "budget_interrupted; not an episode outcome"},
        }
        for change, reason in (({"unrecognized_road_ids": [3001]}, "unreviewed road identity alias"),
                               ({"option_action_counts": {"KEEP": 14}}, "option action counts")):
            with self.subTest(change=change):
                self.write(path, {**result, **change})
                report = self.audit()
                hit = next(c for c in report["collisions"] if c["geometry_seed"] == 3000)
                self.assertTrue(any(s["path"] == path for s in hit["sources"]))
                self.assertTrue(any(b["path"] == path and reason in b["reason"]
                                    for b in report["blockers"]))

    def test_result_with_second_unknown_road_alias_cannot_be_typed(self) -> None:
        folder = f"{audit._RESIDUAL}/iteration-2"
        self.write(f"{folder}/episodes.jsonl", self.rows([
            {"episode": 1, "geometry_seed": 91000, "track_id": 2, "decisions": 8},
        ]))
        self.write(f"{folder}/result.json", {
            "completed_episodes": 1, "environment_decisions": 13,
            "status": "completed_bounded_training_pilot", "unrecognized_road_ids": [3000],
            "incomplete_final_episode": {"geometry_seed": 91001, "track_id": 4,
                                         "decisions": 5,
                                         "status": "budget_interrupted; not an episode outcome"},
        })
        report = self.audit()
        self.assertTrue(any(b["path"] == f"{folder}/result.json" for b in report["blockers"]))
        self.assertEqual(report["status"], "BLOCKED")

    def test_unreviewed_result_files_are_not_opened_even_if_poisoned(self) -> None:
        forbidden = ("runs/other-lane/result.json", "runs/other-lane/failure.json",
                     "runs/other-lane/receipt.json", "runs/other-lane/blind/episodes.jsonl",
                     "experiments/other-lane-result.json")
        for path in forbidden:
            self.write(path, b"PROTECTED POISON 3000")
        original = Path.read_bytes

        def guarded(path: Path) -> bytes:
            self.assertNotIn(path.relative_to(self.root).as_posix(), forbidden)
            return original(path)

        with patch.object(Path, "read_bytes", guarded):
            report = self.audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["collisions"], [])

    def test_unrelated_metadata_drift_is_a_warning_not_a_collision(self) -> None:
        original = self.audit()
        self.write("experiments/other-lane-protocol.json", {"unrelated": "new metadata"})
        changed = self.audit(expected_source_inventory_sha256=original["source_inventory_sha256"])
        self.assertEqual(changed["collisions"], [])
        self.assertFalse(any(b["path"] == "source_inventory" for b in changed["blockers"]))
        self.assertTrue(any(w["path"] == "source_inventory" and w["field"] == "names+bytes"
                            for w in changed["provenance_warnings"]))
        self.assertEqual([b["field"] for b in changed["blockers"]], ["coverage"])

    def test_candidate_relevant_unknown_protocol_and_torn_td_record_block(self) -> None:
        self.write("experiments/unknown-lane-protocol.json", {"unknown_road_ids": [3000]})
        path = audit.TD_JOURNALS[0]
        self.write(path, self.rows([{"event": "start", "protocol_sha256": "0" * 64},
                                    {"event": "reset_intent", "geometry_seed": 3001,
                                     "track_id": 2}]) + b"{bad json\n")
        report = self.audit()
        self.assertTrue(any(b["path"] == "experiments/unknown-lane-protocol.json"
                            for b in report["blockers"]))
        self.assertTrue(any(b["path"] == path for b in report["blockers"]))
        self.assertIn(3001, {c["geometry_seed"] for c in report["collisions"]})

    def test_cli_has_no_reserve_or_seed_start_option_and_returns_blocked(self) -> None:
        with patch("sys.argv", ["audit", "--repo-root", str(self.root),
                                "--candidate-path", self.CANDIDATE]), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(audit.main(), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "BLOCKED")
        self.assertEqual(list((self.root / audit.TRAIN_CLAIMS).iterdir()),
                         [self.root / audit.TRAIN_CLAIMS / ".gitkeep"])
        with (patch("sys.argv", ["audit", "--repo-root", str(self.root),
                                 "--candidate-path", self.CANDIDATE, "--reserve"]),
              redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as rejected):
            audit.main()
        self.assertEqual(rejected.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
