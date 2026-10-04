"""Synthetic metadata-only coverage: no environment/model construction or reset."""
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from haic.train_seed_reservations import SeedUnavailableError
from scripts import audit_koi_steering_generalization as koi
from scripts.audit_rlpd_g1_coverage_seeds import InventoryError, _digest, _sha


class KoiTrainAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / koi.REGISTRY).mkdir(parents=True)
        (self.root / "runs").mkdir()

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")
        return path

    def check(self, **kwargs):
        return koi.audit(self.root, required_sources=(), **kwargs)

    def claim(self):
        return koi.claim(self.root, required_sources=())

    def test_fixed_uint32_24_roads_72_cells_144_slots(self):
        cells, slots = koi.proposed_cells(), koi.scheduled_slots()
        self.assertEqual(len(cells), 72)
        self.assertEqual(len(slots), 144)
        self.assertEqual({c["track_id"] for c in cells}, {1, 2, 3})
        self.assertEqual({c["geometry_seed"] for c in cells}, set(range(3184000001, 3184000025)))
        self.assertTrue(all(c["partition"] == "TRAIN" and c["obstacles"] is True for c in cells))
        self.assertEqual(len(koi.claim_cells()), 24)
        self.assertFalse({38300, 38301, 38302, 38303, 50300, 50301, 50302, 50303} & {c["geometry_seed"] for c in cells})

    def test_clear_deterministic_no_interaction(self):
        a, b = self.check(), self.check()
        self.assertEqual(a, b)
        self.assertEqual(a["status"], "clear")
        self.assertEqual(a["environment_constructions"], 0)
        self.assertEqual(a["environment_resets"], 0)
        self.assertEqual(a["blind_observation_reads"], 0)
        self.assertFalse(a["project_global_unseen_certified"])

    def test_all_track_and_protected_ids_excluded(self):
        self.put("experiments/other-protocol.json", {"partition": "blind", "track_id": 999,
                                                  "geometry_seed": koi.SEED_START + 23})
        self.assertEqual(len(self.check()["collisions"]), 1)
        with self.assertRaises(InventoryError):
            self.claim()
        self.assertEqual(list((self.root / koi.REGISTRY).iterdir()), [])

    def test_new_protocol_independent_hash_drift_is_not_collision(self):
        self.put("experiments/independent.json", {"cells": [{"geometry_seed": 111}]})
        self.assertEqual(self.check()["status"], "clear")
        self.put("experiments/independent.json", {"cells": [{"geometry_seed": 222}]})
        self.assertEqual(self.check()["status"], "clear")

    def test_interval_unknowns_and_missing_required_fail_closed(self):
        for value in ({"road_interval": "unknown"},
                      {"geometry_seed_range": {"start": koi.SEED_START - 20, "count": 100}},
                      {"geometry_seed": None}, {"geometry_seed": "unknown"}):
            with self.subTest(value=value):
                self.put("experiments/possible.json", value)
                self.assertEqual(self.check()["status"], "HOLD")
        self.assertEqual(koi.audit(self.root, required_sources=("runs/missing/episodes.jsonl",))["status"], "HOLD")

    def test_partial_reset_exposure_all_old_registry_preserved(self):
        for name, value in (("runs/other/reset-intents.jsonl", {"geometry_seed": koi.SEED_START}),
                            ("runs/other/reset-ledger.jsonl", {"seed": koi.SEED_START, "status": "reset_intent"}),
                            ("runs/other/exposure.json", {"geometry_seed": koi.SEED_START}),
                            ("runs/rlpd-g0-claims/batch.json", {"geometry_seeds": [koi.SEED_START]})):
            with self.subTest(name=name):
                path = self.put(name, value)
                self.assertEqual(self.check()["status"], "HOLD")
                path.unlink()

    def test_torn_ledger_keeps_known_candidate_prefix(self):
        path = self.root / "runs/episodes.jsonl"
        path.write_text(json.dumps({"geometry_seed": koi.SEED_START}) + '\n{"geometry_seed":')
        self.assertEqual(self.check()["status"], "HOLD")
        self.assertEqual(len(self.check()["collisions"]), 1)

    def test_protected_outcomes_never_read_but_protocols_read(self):
        self.put("runs/other-blind/episodes.jsonl", {"geometry_seed": koi.SEED_START, "reward": 9})
        self.put("evaluations/blind/summary.json", {"geometry_seed": koi.SEED_START})
        self.assertEqual(self.check()["sources"], [])
        self.put("evaluations/blind/protocol.json", {"geometry_seed": koi.SEED_START})
        self.assertEqual(self.check()["status"], "HOLD")

    def test_symlink_and_unknown_registry_entry_block(self):
        link = self.root / "runs/other"
        link.symlink_to(self.root / "experiments", target_is_directory=True)
        self.assertEqual(self.check()["status"], "HOLD")
        link.unlink()
        self.put(koi.REGISTRY + "/unclassified.json", {})
        self.assertEqual(self.check()["status"], "HOLD")

    def test_geometry_global_claims_bind_full_cohort_and_authenticate(self):
        receipt = self.claim()
        self.assertEqual(len(koi.verify_claims(receipt, root=self.root)), 24)
        self.assertEqual(receipt["claim_audit"]["source"], koi._scoped_source(receipt))
        self.assertEqual(self.check()["status"], "HOLD")
        self.assertEqual(self.check(authenticated_receipt=receipt)["status"], "clear")
        with self.assertRaises(SeedUnavailableError):
            self.claim()
        path = self.put(koi.AUDIT_PATH, receipt)
        self.assertEqual(koi.load_audit(path, _sha(path.read_bytes()), root=self.root), receipt)
        with self.assertRaises(InventoryError):
            koi.load_audit(path, "0" * 64, root=self.root)

    def test_actual_claim_wrapper_reaudits_under_shared_directory_lock(self):
        original = koi.audit
        def check_lock(*args, **kwargs):
            fd = os.open(self.root / koi.REGISTRY, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
            return original(*args, **kwargs)
        with patch.object(koi, "audit", side_effect=check_lock) as audited:
            receipt = self.claim()
        self.assertEqual(audited.call_count, 1)
        self.assertEqual(len(receipt["claims"]), 24)

    def test_new_collision_during_under_lock_reaudit_writes_no_claim(self):
        original = koi.audit
        def race(*args, **kwargs):
            self.put("runs/other/reset-intents.jsonl", {"geometry_seed": koi.SEED_START + 23})
            return original(*args, **kwargs)
        with patch.object(koi, "audit", side_effect=race), self.assertRaises(InventoryError):
            self.claim()
        self.assertEqual(list((self.root / koi.REGISTRY).iterdir()), [])

    def test_claim_tampering_and_scope_changes_block(self):
        receipt = self.claim()
        for field, value in (("cells", koi.claim_cells()), ("warnings", [{"unbound": True}]),
                             ("candidate_lineage_evidence", {}), ("project_global_unseen_certified", True),
                             ("environment_resets", 1), ("blind_observation_reads", 1), ("matched_cell_count", 24)):
            bad = deepcopy(receipt)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(InventoryError):
                koi.verify_claims(bad, root=self.root)
        path = self.root / receipt["claims"][0]["path"]
        row = json.loads(path.read_text())
        row["track_id"] = 3
        path.write_text(json.dumps(row))
        with self.assertRaises(InventoryError):
            koi.verify_claims(receipt, root=self.root)

    def protocol(self, receipt):
        audit = self.put(koi.AUDIT_PATH, receipt)
        obj = {"study": koi.STUDY_ID, "cells": koi.proposed_cells(), "episodes": 144,
               "schedule": koi.scheduled_slots(), "model_hashes": koi.MODEL_HASHES,
               "model_source_sha256": receipt["candidate_lineage_evidence"].get("model_source_sha256"),
               "candidate_manifest_sha256": koi.MANIFEST_SHA,
               "max_decisions": 1200, "frame_skip": 4, "warmup_ticks": 50, "raw_fps": 50,
               "official_action": False,
               "freshness": {"audit_receipt": {"path": koi.AUDIT_PATH, "sha256": _sha(audit.read_bytes())}}}
        path = self.put(koi.PROTOCOL_PATH, obj)
        copied = self.put(koi.RUN_PATH + "/protocol.json", obj)
        self.assertEqual(copied.read_bytes(), path.read_bytes())
        return path, obj

    def intent(self, slot_index=0):
        slot = koi.scheduled_slots()[slot_index]
        return {"status": "reset_intent", "arm": slot["mode"], "track": slot["track_id"],
                "seed": slot["seed"], "slot_id": f"{slot['track_id']}:{slot['seed']}:{slot['mode']}", "time": 1.5}

    def completed(self, slot_index=0):
        intent = self.intent(slot_index)
        t, s, a = intent["track"], intent["seed"], intent["arm"]
        return {"mode": a, "track_id": t, "seed": s, "completed": True, "lapTimeMs": 100,
                "progress": 1, "damage": 0, "collisions": 0, "retire_reason": "finished",
                "error": None, "invalid_actions": 0, "steps": 10, "raw_ticks": 40, "peak_rss_bytes": 1,
                "status": "completed", "file": f"{t}-{s}-{a}.json", "sha256": "a" * 64,
                "slot_id": intent["slot_id"], "process_file": f"{t}-{s}-{a}.bound-process.json", "process_sha256": "b" * 64}

    def test_exact_self_protocol_ledger_pending_child_and_completed_order(self):
        receipt = self.claim()
        path, _ = self.protocol(receipt)
        sha = _sha(path.read_bytes())
        def check():
            return self.check(authenticated_receipt=receipt, self_protocol_sha256=sha)
        self.assertEqual(check()["status"], "clear")
        exposure = self.root / koi.RUN_PATH / "exposure-audit.json"
        exposure.write_bytes((self.root / koi.AUDIT_PATH).read_bytes())
        self.assertEqual(check()["status"], "clear")
        exposure.write_text('{}\n')
        self.assertEqual(check()["status"], "HOLD")
        exposure.write_bytes((self.root / koi.AUDIT_PATH).read_bytes())
        ledger = self.put(koi.RUN_PATH + "/reset-ledger.jsonl", self.intent())
        self.assertEqual(check()["status"], "clear")
        rows = [self.intent(), self.completed(), self.intent(1)]
        ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.assertEqual(check()["status"], "clear")
        rows[-1]["arm"] = koi.ARMS[0] if rows[-1]["arm"] != koi.ARMS[0] else koi.ARMS[1]
        ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.assertEqual(check()["status"], "HOLD")

    def test_self_protocol_model_scope_count_and_receipt_tampering(self):
        receipt = self.claim()
        path, obj = self.protocol(receipt)
        sha = _sha(path.read_bytes())
        for field, value in (("episodes", 48), ("model_hashes", {}), ("cells", koi.proposed_cells()[:-1]),
                             ("candidate_manifest_sha256", "f" * 64), ("official_action", True)):
            with self.subTest(field=field):
                bad = deepcopy(obj)
                bad[field] = value
                self.put(koi.PROTOCOL_PATH, bad)
                self.assertEqual(self.check(authenticated_receipt=receipt, self_protocol_sha256=sha)["status"], "HOLD")
        self.put(koi.PROTOCOL_PATH, obj)
        self.put(koi.RUN_PATH + "/reset-intents.jsonl", self.intent())
        self.assertEqual(self.check(authenticated_receipt=receipt, self_protocol_sha256=sha)["status"], "HOLD")

    def test_foreign_study_does_not_become_self_by_name(self):
        receipt = self.claim()
        self.put(koi.RUN_PATH + "/reset-ledger.jsonl", self.intent())
        self.assertEqual(self.check(authenticated_receipt=receipt)["status"], "HOLD")
        self.put("runs/foreign/reset-ledger.jsonl", self.intent())
        path, _ = self.protocol(receipt)
        self.assertEqual(self.check(authenticated_receipt=receipt, self_protocol_sha256=_sha(path.read_bytes()))["status"], "HOLD")

    def test_supplied_protocol_hash_requires_actual_protocol(self):
        receipt = self.claim()
        self.assertEqual(self.check(authenticated_receipt=receipt, self_protocol_sha256="0" * 64)["status"], "HOLD")

    def test_unknown_legacy_sampler_is_not_namespace_disjoint_proof(self):
        self.put("runs/legacy/config.json", {"sampled_seed_range": [0, 4294967295]})
        self.assertEqual(self.check()["status"], "HOLD")

    def ppo(self, mode):
        cfg = {"algorithm": "PPO", "training_track_mode": mode, "track_ids": [1], "seeds": [42],
               "sampled_seed_range": [0, 4294967295], "track_sampler_seed": 917,
               "seed": 0, "n_envs": 1, "n_steps": 8, "total_timesteps": 32, "resume_from": None}
        self.put("runs/legacy/config.json", {"config": cfg, "git": {"commit": "0000000", "dirty": True},
                                           "command_line": f"train.py --training-track-mode {mode} --track-sampler-seed 917 --n-envs 1"})
        (self.root / "runs/legacy/train.log").write_text(
            f"track_ids: [1]\nseeds: [42]\ntraining track mode: {mode} sampler_seed: 917\n"
            "n_envs: 1  total_timesteps: 32\n| total_timesteps | 32 |\n")

    def test_legacy_fixed_mode_capability_is_not_allocation(self):
        self.ppo("fixed")
        self.assertEqual(self.check()["status"], "clear")
        self.assertEqual(self.check()["legacy_ppo_dispositions"][0]["status"], "resolved-fixed")

    def test_legacy_sampled_unknown_without_candidate_lineage_blocks(self):
        self.ppo("sampled")
        report = self.check()
        self.assertEqual(report["status"], "HOLD")
        self.assertEqual(report["legacy_uncertainties"], [])

    def test_malformed_explicit_road_unknown_blocks(self):
        path = self.root / "runs/episodes.jsonl"
        path.write_text('{"geometry_seed":')
        self.assertEqual(self.check()["status"], "HOLD")

    def test_new_interaction_after_claim_rechecked(self):
        receipt = self.claim()
        self.put("runs/lane/exposure.json", {"geometry_seed": koi.SEED_START})
        self.assertEqual(self.check(authenticated_receipt=receipt)["status"], "HOLD")

    def test_source_overrides_forbidden_for_actual_workspace(self):
        with self.assertRaises(InventoryError):
            koi.audit(Path(koi.__file__).resolve().parents[1], required_sources=())


if __name__ == "__main__":
    unittest.main()
