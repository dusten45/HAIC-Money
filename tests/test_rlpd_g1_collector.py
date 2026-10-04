"""Synthetic-only G1 preflight/journal tests; no actor weights or simulator."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import scripts.diagnose_rlpd_g1_coverage as g1


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class TestG1CollectorSkeleton(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "runs").mkdir()
        # Synthetic stand-ins for the actual pinned actor exports. The production
        # digests remain immutable; only this test's module lookup is patched.
        self.actors = {actor: f"test-actor-{i}".encode("ascii")
                       for i, actor in enumerate(g1.ACTORS)}
        self.hashes = {actor: sha(self.actors[actor]) for actor in g1.ACTORS}
        replacement = patch.dict(g1.ACTOR_HASHES, self.hashes)
        replacement.start()
        self.addCleanup(replacement.stop)
        sources = {path: self.write(path, path.encode("ascii")) for path in g1.SOURCE_FILES}
        for actor, path in zip(g1.ACTORS, g1.ACTOR_PATHS):
            self.write(path, self.actors[actor])
        self.cells = [{"partition": "TRAIN", "track_id": 1, "geometry_seed": 210000 + i,
                       "obstacles": True} for i in range(24)]
        self.protocol_path = "experiments/g1-synthetic-protocol.json"
        self.audit_path = "experiments/g1-synthetic-audit.json"
        self.claim_paths = []
        for cell in self.cells:
            seed = cell["geometry_seed"]
            path = f"experiments/train-seed-claims/seed-{seed}.json"
            self.claim_paths.append(path)
            self.write_json(path, {
                "format": "haic-train-seed-claim-v1", "study_id": "g1-synthetic",
                "protocol_id": "g1-synthetic", "protocol_path": None,
                "protocol_sha256": None, **cell, "status": "reserved",
                "claimed_at_utc": "2026-09-26T00:00:00Z",
                "audit_source": "haic-rlpd-g1-coverage-seed-inventory-v2:synthetic-test",
                "audit_digest_sha256": "a" * 64,
            })
        claim_sources = [{"path": path, "sha256": sha((self.root / path).read_bytes())}
                         for path in self.claim_paths]
        self.claim_sha = g1._digest(sorted(claim_sources, key=lambda entry: entry["path"]))
        self.protocol = {
            "format": "haic-rlpd-g1-coverage-protocol-v1", "status": "frozen",
            "partition": "TRAIN", "study_id": "g1-synthetic", "cells": self.cells,
            "source_hashes": sources,
            "source_actors": [{"id": actor, "path": path, "sha256": self.hashes[actor],
                               "source_sha256": "b" * 64, "export_protocol_sha256": "c" * 64,
                               "action_mode": "exported_tanh_mean"}
                              for actor, path in zip(g1.ACTORS, g1.ACTOR_PATHS)],
            "budget": {"max_decisions_per_episode": 2000, "max_total_decisions": 96000,
                       "max_total_raw_frames": 386448, "max_core_hours": 4.0},
            "runtime": {"frame_skip": 4, "reward_shaping": False, "collision_penalty": 0,
                        "interventions": False, "learner_updates": 0,
                        "reset_initial_raw_frames": 1, "reset_noop_raw_frames": 50},
            "event_rules": {"max_decisions": 2000, "negative_reward_limit": 100,
                            "stall_window": 20, "tile_window": 20,
                            "directed_delta_epsilon": 0.01,
                            "centerline_far_threshold_m": 10.0},
            "train_claims_sha256": self.claim_sha, "image_rubric_sha256": "d" * 64,
            "exclusions": {"training_geometry_seeds": [209999]},
        }
        self.protocol_sha = self.write_json(self.protocol_path, self.protocol)
        self.audit = {
            "format": "haic-rlpd-g1-coverage-seed-inventory-v2",
            "status": "no_known_recorded_overlap", "protocol_frozen": True,
            "self_claims_verified": True, "seed_start": self.cells[0]["geometry_seed"],
            "candidate_seeds": [row["geometry_seed"] for row in self.cells],
            "cells": self.cells, "train_claims_sha256": self.claim_sha,
            "collisions": [], "blockers": [], "provenance_warnings": [],
        }
        self.audit_sha = self.write_json(self.audit_path, self.audit)
        self.created = 0

    def write(self, relative: str, raw: bytes) -> str:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return sha(raw)

    def write_json(self, relative: str, value: object) -> str:
        return self.write(relative, json.dumps(value, sort_keys=True).encode("ascii"))

    def auditor(self, seed_start, **kwargs):
        self.assertEqual(seed_start, self.cells[0]["geometry_seed"])
        self.assertEqual(kwargs["repo_root"], self.root)
        self.assertEqual(kwargs["self_study_id"], "g1-synthetic")
        self.assertEqual(kwargs["self_protocol_sha256"], self.protocol_sha)
        return copy.deepcopy(self.audit)

    def preflight(self):
        return g1.preflight(self.root, self.protocol_path, self.protocol_sha,
                            self.audit_path, self.audit_sha, auditor=self.auditor)

    def fake_env(self, **kwargs):
        self.created += 1
        self.assertEqual(kwargs["track_id"], 1)
        return object()

    def fake_cell(self, *, row, actor, actor_info, actor_payload, rules,
                  centerline_far_threshold_m, env_factory, attempt):
        index = self.created
        self.assertTrue((self.root / "runs/synthetic/attempts" /
                         f"slot-{index:02d}.json").is_file())
        self.assertEqual(actor_payload, self.actors[actor_info["id"]])
        self.assertEqual(rules["max_decisions"], 2000)
        self.assertEqual(centerline_far_threshold_m, 10.0)
        env_factory(track_id=row["track_id"])
        attempt.update(phase="complete", decisions_completed=4, decision_calls=4,
                       raw_counters={"resets": 1, "reset_initial_raw_frames": 1,
                                     "reset_noop_raw_frames": 50, "driven_raw_frames": 16})
        return ({"partition": "TRAIN", "track_id": row["track_id"],
                 "geometry_seed": row["geometry_seed"], "actor_id": actor_info["id"],
                 "actor_sha256": actor_info["sha256"], "steps": 4,
                 "reset_initial_raw_frames": 1, "reset_noop_raw_frames": 50,
                 "driven_raw_frames": 16,
                 "road_centerline_sha256": sha(str(row["geometry_seed"]).encode()),
                 "summary": {"outcome": "off_track"}},
                {"synthetic_pixels_only": [1, 2, 3, 4]})

    def test_real_entry_point_refuses_all_injections_without_creating_env(self):
        context = self.preflight()
        result = g1.collect(context, env_factory=self.fake_env, run_cell=self.fake_cell)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("mid-episode", result["reason"])
        self.assertEqual(result["environment_creations"], 0)
        self.assertEqual(self.created, 0)

    def test_frozen_exclusions_fail_closed_before_environment_creation(self):
        for exclusions in (None, {"unreviewed_seed": 210000},
                           {"training_geometry_seeds": [210000]},
                           {"training_geometry_seeds": ["210000"]}):
            with self.subTest(exclusions=exclusions):
                self.setUp()
                if exclusions is None:
                    del self.protocol["exclusions"]
                else:
                    self.protocol["exclusions"] = exclusions
                self.protocol_sha = self.write_json(self.protocol_path, self.protocol)
                with self.assertRaises(ValueError):
                    self.preflight()
                self.assertEqual(self.created, 0)

    def test_bad_source_cell_actor_claim_and_audit_prevent_every_env_creation(self):
        changes = (
            lambda: self.write(next(iter(g1.SOURCE_FILES)), b"drift"),
            lambda: self.protocol["cells"][4].update(geometry_seed=42),
            lambda: self.protocol["source_actors"].reverse(),
            lambda: self.write_json(self.claim_paths[0], {"not": "a claim"}),
            lambda: self.audit.update(status="BLOCKED"),
            lambda: self.audit.update(provenance_warnings=[{"path": "runs/unknown",
                                                           "field": "candidate", "reason": "ambiguous"}]),
        )
        for change in changes:
            with self.subTest(change=change):
                # Rebuild fresh isolated fixtures rather than allowing a failed
                # case to influence the next fixture's source/claim inventory.
                self.setUp()
                change()
                self.protocol_sha = self.write_json(self.protocol_path, self.protocol)
                self.audit_sha = self.write_json(self.audit_path, self.audit)
                with self.assertRaises(ValueError):
                    self.preflight()
                self.assertEqual(self.created, 0)

    def test_fresh_auditor_block_and_unknown_warning_fail_closed(self):
        for mutation in ({"status": "BLOCKED"}, {"blockers": [{"path": "candidate"}]},
                         {"provenance_warnings": [{"path": "candidate", "field": "unknown"}]}):
            with self.subTest(mutation=mutation):
                def changed(*args, **kwargs):
                    return {**self.audit, **mutation}
                with self.assertRaisesRegex(ValueError, "BLOCKED"):
                    g1.preflight(self.root, self.protocol_path, self.protocol_sha,
                                 self.audit_path, self.audit_sha, auditor=changed)
                self.assertEqual(self.created, 0)

    def test_unrelated_provenance_warnings_do_not_false_block_train_cells(self):
        self.audit["provenance_warnings"] = [{
            "path": "experiments/dreamer-unrelated.json", "field": "source bytes",
            "reason": "unrelated development protocol changed",
        }]
        self.audit_sha = self.write_json(self.audit_path, self.audit)
        context = self.preflight()
        self.assertEqual(g1.collect(context, env_factory=self.fake_env,
                                    run_cell=self.fake_cell)["environment_creations"], 0)

        def changed_unrelated(*args, **kwargs):
            return {**self.audit, "provenance_warnings": [{
                "path": "experiments/drq-other.json", "field": "source bytes",
                "reason": "independent lane file changed",
            }]}

        context = g1.preflight(self.root, self.protocol_path, self.protocol_sha,
                               self.audit_path, self.audit_sha, auditor=changed_unrelated)
        self.assertEqual(context["protocol"]["cells"], self.cells)
        self.assertEqual(self.created, 0)

    def test_full_synthetic_journal_has_48_ordered_slots_and_no_image_attestation(self):
        context = self.preflight()
        result = g1.simulate_schedule(context, self.root / "runs/synthetic",
                                      env_factory=self.fake_env, run_cell=self.fake_cell,
                                      clock=lambda: 0.0)
        self.assertEqual(result["status"], "synthetic_only_complete")
        self.assertEqual((result["scheduled_slots"], result["complete_slots"],
                          result["censored_slots"], result["unrun_slots"]), (48, 48, 0, 0))
        self.assertEqual(result["decisions_completed"], 192)
        self.assertEqual(result["raw_frames_completed"], 48 * 67)
        self.assertEqual(self.created, 48)
        rows = json.loads((self.root / "runs/synthetic/slots.json").read_text())
        self.assertEqual([(row["geometry_seed"], row["actor_id"]) for row in rows],
                         [(cell["geometry_seed"], actor) for cell in self.cells
                          for actor in g1.ACTORS])
        self.assertTrue(all(row["image_flag"] is None for row in rows))
        self.assertNotIn("pass", result)
        self.assertEqual(len(list((self.root / "runs/synthetic/attempts").iterdir())), 48)

    def test_reset_or_step_failure_preserves_censor_partial_and_unrun(self):
        for phase, count in (("reset", 0), ("driving", 2)):
            with self.subTest(phase=phase):
                self.setUp()
                context = self.preflight()

                def failing(**kwargs):
                    attempt = kwargs["attempt"]
                    attempt.update(phase=phase, decision_calls=count + 1,
                                   decisions_completed=count, partial={"pixels": [1, 2][:count]},
                                   raw_counters={"resets": 1, "reset_initial_raw_frames": 1,
                                                 "reset_noop_raw_frames": 50,
                                                 "driven_raw_frames": count * 4})
                    kwargs["env_factory"](track_id=1)
                    raise RuntimeError(f"synthetic {phase} failure")

                result = g1.simulate_schedule(context, self.root / "runs/synthetic",
                                              env_factory=self.fake_env, run_cell=failing,
                                              clock=lambda: 0.0)
                self.assertEqual(result["status"], "synthetic_only_aborted")
                self.assertEqual((result["complete_slots"], result["censored_slots"],
                                  result["unrun_slots"]), (0, 1, 47))
                rows = json.loads((self.root / "runs/synthetic/slots.json").read_text())
                self.assertEqual(rows[0]["outcome"], "unknown")
                self.assertEqual(rows[0]["decisions"], count)
                self.assertTrue(all(row["status"] == "unrun" for row in rows[1:]))
                abort = json.loads((self.root / "runs/synthetic/aborts/slot-00.json").read_text())
                self.assertEqual(abort["phase"], phase)
                self.assertEqual(abort["partial"]["pixels"], [1, 2][:count])
                self.assertTrue(abort["no_retry_or_top_up"])
                self.assertEqual(self.created, 1)

    def test_storage_failure_after_fake_cell_preserves_attempt_and_unknown(self):
        context = self.preflight()

        def broken_storage(path, trace):
            raise OSError("synthetic storage failure")

        result = g1.simulate_schedule(context, self.root / "runs/synthetic",
                                      env_factory=self.fake_env, run_cell=self.fake_cell,
                                      clock=lambda: 0.0, store_trace=broken_storage)
        self.assertEqual(result["status"], "synthetic_only_aborted")
        self.assertEqual((result["censored_slots"], result["unrun_slots"]), (1, 47))
        self.assertEqual(self.created, 1)
        abort = json.loads((self.root / "runs/synthetic/aborts/slot-00.json").read_text())
        self.assertIn("storage failure", abort["error"])

    def test_pre_cell_source_drift_halts_without_second_environment(self):
        context = self.preflight()
        source_path = "scripts/diagnose_rlpd_g0.py"

        def drifting(**kwargs):
            result = self.fake_cell(**kwargs)
            self.write(source_path, b"synthetic mid-cell source drift")
            return result

        result = g1.simulate_schedule(context, self.root / "runs/synthetic",
                                      env_factory=self.fake_env, run_cell=drifting,
                                      clock=lambda: 0.0)
        self.assertEqual(result["status"], "synthetic_only_aborted")
        self.assertEqual(self.created, 1)
        abort = json.loads((self.root / "runs/synthetic/aborts/slot-00.json").read_text())
        self.assertIn("source/hash drift", abort["recheck_error"])

    def test_cpu_overrun_censors_fake_cell_but_real_is_blocked_even_under_cap(self):
        context = self.preflight()
        seconds = [0.0]

        def slow(**kwargs):
            result = self.fake_cell(**kwargs)
            seconds[0] = 4 * 3600
            return result

        result = g1.simulate_schedule(context, self.root / "runs/synthetic",
                                      env_factory=self.fake_env, run_cell=slow,
                                      clock=lambda: seconds[0])
        self.assertEqual(result["status"], "synthetic_only_aborted")
        self.assertEqual(result["censored_slots"], 1)
        self.assertEqual(self.created, 1)
        self.assertIn("mid-episode", g1.collect(context)["reason"])


if __name__ == "__main__":
    unittest.main()
