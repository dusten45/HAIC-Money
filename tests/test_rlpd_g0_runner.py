import copy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common_adapter import ActionSpec, ObservationSpec
from haic.algorithms.rlpd.g0_diagnostic import G0EventRules
from scripts.audit_rlpd_g0_seeds import R5_ERRATUM_PATH
from scripts.diagnose_rlpd_g0 import SOURCE_FILES, collect, preflight, run_cell
from scripts.prepare_rlpd_g0_protocol import freeze_g0_protocol
from scripts.summarize_rlpd_g0 import summarize_run


class SyntheticActor:
    def model(self, observation):
        return torch.zeros((len(observation), 3), dtype=torch.float32)

    def reset(self, observation):
        pass

    def act(self, observation):
        return np.array([0.0, 0.5, 0.5], dtype=np.float32)


class SyntheticEnvironment:
    def __init__(self, *, track_id, seed, **kwargs):
        self.track_id, self.track_seed = track_id, seed
        self.closed = False
        self.unwrapped = self
        self.max_off_track_steps = 100
        self.off_track_counter = 0
        self.warmup_steps = 0
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        self.track = [(0.0, 0.0, 30 * np.cos(theta), 30 * np.sin(theta)) for theta in angles]
        hull = type("Hull", (), {"position": (30.0, 0.0), "linearVelocity": (0.0, 0.0), "angle": 0.0})()
        self.car = type("Car", (), {"hull": hull})()
        self.tile_visited_count = 0
        self.finish_line_tracker = type("Tracker", (), {
            "candidate_crossing_time_s": None, "crossing_from_back": False,
        })()

    def reset(self, *, seed=None, options=None):
        self.t = 0.0
        self.step(None)
        image = np.zeros((4, 84, 84), dtype=np.float32)
        return image, {"seed": self.track_seed, "track_id": self.track_id}

    def step(self, action):
        self.t += 0.02
        if action is None:
            return np.zeros((4, 84, 84), dtype=np.float32), 0.0, False, False, {}
        self.tile_visited_count += 1
        image = np.zeros((4, 84, 84), dtype=np.float32)
        image[-1] = 1.0 / 255.0
        return image, 1.0, True, False, {
            "seed": self.track_seed, "track_id": self.track_id, "collision": False,
            "damage": 0.0, "progress": 1.0, "finish_qualified": True,
            "finished": True, "retire_reason": None,
        }

    def close(self):
        self.closed = True


class TestG0RunnerPreflight(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ("experiments", "runs", "scripts", "core/vendor", "haic/algorithms/rlpd"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        hashes = {}
        for relative in SOURCE_FILES:
            path = self.root / relative
            path.write_bytes(relative.encode("ascii"))
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        study_protocol = self.root / "experiments/synthetic.json"
        study_protocol.write_text("{}\n", encoding="utf-8")
        self.training_protocol_sha = hashlib.sha256(study_protocol.read_bytes()).hexdigest()
        self.actor_bytes = b"synthetic-rlpd-actor"
        self.actor_paths = []
        actors = []
        self.expected_actors = {}
        for seed, actor_id in enumerate(("first", "second"), start=1):
            run_dir = self.root / "runs" / actor_id
            checkpoint_dir = run_dir / "checkpoints/step-000131072"
            checkpoint_dir.mkdir(parents=True)
            path = checkpoint_dir / "actor.pt"
            path.write_bytes(self.actor_bytes + actor_id.encode("ascii"))
            self.actor_paths.append(path)
            actor_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            checkpoint = checkpoint_dir / "checkpoint.pt"
            checkpoint.write_bytes(b"synthetic-checkpoint-" + actor_id.encode("ascii"))
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            candidate_path = checkpoint_dir / "candidate.json"
            candidate_path.write_text(json.dumps({
                "format": "haic-rlpd-candidate-v1",
                "actor_path": "checkpoints/step-000131072/actor.pt",
                "actor_sha256": actor_sha, "protocol_sha256": self.training_protocol_sha,
                "checkpoint_path": "checkpoints/step-000131072/checkpoint.pt",
                "checkpoint_sha256": checkpoint_sha,
                "training_seed": seed, "environment_steps": 131072,
                "study_id": "synthetic", "arm": "rlpd",
            }), encoding="utf-8")
            self.expected_actors[actor_id] = {
                "path": path.relative_to(self.root).as_posix(),
                "sha256": actor_sha, "training_seed": seed,
                "study_id": "synthetic", "arm": "rlpd",
            }
            actors.append({
                "id": actor_id,
                "path": path.relative_to(self.root).as_posix(),
                "sha256": actor_sha,
                "source_sha256": "a" * 64,
                "export_protocol_sha256": self.training_protocol_sha,
                "action_mode": "exported_tanh_mean",
                "candidate_path": candidate_path.relative_to(self.root).as_posix(),
                "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                "checkpoint_path": checkpoint.relative_to(self.root).as_posix(),
                "checkpoint_sha256": checkpoint_sha,
                "source_study_protocol_path": "experiments/synthetic.json",
                "training_seed": seed, "environment_steps": 131072,
            })
        self.cells = [
            {"partition": "TRAIN", "track_id": 1 + i % 4, "geometry_seed": 41000 + i, "obstacles": True}
            for i in range(12)
        ]
        history = self.root / "experiments/history.json"
        history.write_text("{}\n", encoding="utf-8")
        inventory = [{"path": "experiments/history.json", "kind": "protocol",
                      "sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
                      "bytes": history.stat().st_size}]
        history2 = self.root / "experiments/history2.json"
        history2.write_text("{}\n", encoding="utf-8")
        inventory.append({"path": "experiments/history2.json", "kind": "protocol",
                          "sha256": hashlib.sha256(history2.read_bytes()).hexdigest(),
                          "bytes": history2.stat().st_size})
        erratum = self.root / R5_ERRATUM_PATH
        erratum.write_text("{}\n", encoding="utf-8")
        untyped = self.root / "experiments/existing-untyped.json"
        untyped.write_text("{}\n", encoding="utf-8")
        erratum_sha = hashlib.sha256(erratum.read_bytes()).hexdigest()
        inventory.append({"path": R5_ERRATUM_PATH, "kind": "erratum",
                          "sha256": erratum_sha, "bytes": erratum.stat().st_size})
        canonical = lambda value: hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        content_inventory = [
            {"path": file.relative_to(self.root).as_posix(),
             "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "bytes": file.stat().st_size}
            for file in sorted((self.root / "experiments").glob("*.json"))
        ]
        self.audit = {
            "format": "haic-rlpd-g0-seed-audit-v1", "status": "no_known_recorded_overlap",
            "passed": True, "cells": self.cells, "collision_count": 0,
            "ambiguities": [], "protocol_frozen": False,
            "candidate_rule": "seed_start + offset for offsets 0..11, in that order; no replacement on collision",
            "seed_start": 41000,
            "candidate_seeds": [row["geometry_seed"] for row in self.cells],
            "candidate_seeds_sha256": canonical([row["geometry_seed"] for row in self.cells]),
            "r5_erratum": {"path": R5_ERRATUM_PATH, "sha256": erratum_sha,
                           "original_ledger_bytes_attested": False},
            "source_inventory": inventory, "source_inventory_sha256": canonical(inventory),
            "experiment_content_inventory": content_inventory,
            "experiment_content_inventory_sha256": canonical(content_inventory),
            "excluded_inventory_sha256": "d" * 64,
        }
        self.canonical_audit = copy.deepcopy(self.audit)
        self.protocol = {
            "format": "haic-rlpd-g0-diagnostic-v1",
            "status": "frozen",
            "partition": "TRAIN", "frame_skip": 4, "max_steps": 2000, "decision_cap": 48000,
            "reward_shaping": False, "collision_penalty": 0.0, "interventions": False,
            "learner_updates": 0, "centerline_far_threshold_m": 9.0,
            "event_rules": {
                "max_decisions": 2000, "negative_reward_limit": 100, "stall_window": 20,
                "tile_window": 30, "directed_delta_epsilon": 0.0001,
            },
            "cells": self.cells, "source_hashes": hashes, "actors": actors,
            "geometry_audit_path": "experiments/audit.json",
            "r5_erratum_path": R5_ERRATUM_PATH, "r5_erratum_sha256": erratum_sha,
        }
        self.factory_calls = []
        self.save()

    def save(self):
        audit_path = self.root / "experiments/audit.json"
        audit_path.write_text(json.dumps(self.audit, sort_keys=True), encoding="utf-8")
        self.protocol["geometry_audit_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        protocol_path = self.root / "experiments/protocol.json"
        protocol_path.write_text(json.dumps(self.protocol, sort_keys=True), encoding="utf-8")
        self.expected = hashlib.sha256(protocol_path.read_bytes()).hexdigest()

    def payload(self, path):
        seed = self.expected_actors[path.parents[2].name]["training_seed"]
        return {
            "format": "haic-rlpd-pixel-actor-v1", "source_sha256": "a" * 64,
            "protocol_sha256": self.training_protocol_sha,
            "training_seed": seed, "environment_steps": 131072,
            "observation_spec": vars(ObservationSpec()),
            "action_spec": vars(ActionSpec()),
        }

    def factory(self, path):
        self.factory_calls.append(path)
        return SyntheticActor()

    def preflight(self, **options):
        return preflight(
            self.root, "experiments/protocol.json", self.expected,
            actor_factory=self.factory, payload_loader=self.payload,
            audit_verifier=lambda start, track, **kwargs: self.canonical_audit,
            expected_actors=self.expected_actors,
            **options,
        )

    def test_clean_synthetic_protocol_checks_two_cpu_reload_traces(self):
        result = self.preflight()
        self.assertEqual(len(result["cells"]), 12)
        self.assertEqual(len(self.factory_calls), 4)
        self.assertEqual(set(result["actors"]), {"first", "second"})

    def test_rejects_hash_drift_and_unallocated_geometry_before_loading(self):
        with self.assertRaisesRegex(ValueError, "frozen JSON hash changed"):
            preflight(
                self.root, "experiments/protocol.json", "0" * 64,
                actor_factory=self.factory, payload_loader=self.payload,
                audit_verifier=lambda start, track, **kwargs: self.canonical_audit,
                expected_actors=self.expected_actors,
            )
        self.protocol["cells"][0]["geometry_seed"] = 41001
        self.save()
        with self.assertRaisesRegex(ValueError, "12 independent"):
            self.preflight()
        self.assertEqual(self.factory_calls, [])

    def test_rejects_audit_mismatch_and_runtime_or_actor_drift_before_loading(self):
        self.audit["status"] = "incomplete"
        self.save()
        with self.assertRaisesRegex(ValueError, "clean frozen geometry audit"):
            self.preflight()
        self.audit["status"] = "no_known_recorded_overlap"
        self.save()
        (self.root / "experiments/history.json").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "audit source changed"):
            self.preflight()
        (self.root / "experiments/history.json").write_text("{}\n", encoding="utf-8")
        (self.root / "common_adapter.py").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source hash changed"):
            self.preflight()
        (self.root / "common_adapter.py").write_bytes(b"common_adapter.py")
        self.actor_paths[1].write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "actor bytes"):
            self.preflight()
        self.assertEqual(self.factory_calls, [])

    def test_rejects_nontrain_cells_and_bad_actor_metadata(self):
        self.protocol["cells"][0]["partition"] = "blind"
        self.save()
        with self.assertRaisesRegex(ValueError, "non-TRAIN"):
            self.preflight()
        self.assertEqual(self.factory_calls, [])

    def test_incomplete_self_consistent_seed_receipt_is_rejected_before_actor_load(self):
        self.audit["source_inventory"] = self.audit["source_inventory"][:1]
        self.audit["source_inventory_sha256"] = hashlib.sha256(
            json.dumps(self.audit["source_inventory"], sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        self.save()
        with self.assertRaisesRegex(ValueError, "complete fresh audit"):
            self.preflight()
        self.assertEqual(self.factory_calls, [])

    def test_existing_untyped_experiment_byte_mutation_is_rejected(self):
        (self.root / "experiments/existing-untyped.json").write_text(
            '{"new_training_geometry_seed":41000}\n', encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "experiment content changed"):
            self.preflight()
        self.assertEqual(self.factory_calls, [])

    def test_actor_copy_or_substituted_receipt_cannot_override_candidate_identity(self):
        copy_path = self.root / "runs/copied.pt"
        copy_path.write_bytes(self.actor_paths[0].read_bytes())
        self.protocol["actors"][0]["path"] = "runs/copied.pt"
        self.save()
        with self.assertRaisesRegex(ValueError, "independently designated"):
            self.preflight()
        self.assertEqual(self.factory_calls, [])

    def test_source_checkpoint_and_study_protocol_hash_drift_block_preflight(self):
        checkpoint = self.actor_paths[0].with_name("checkpoint.pt")
        checkpoint.write_bytes(b"different weights")
        with self.assertRaisesRegex(ValueError, "source checkpoint/study protocol provenance"):
            self.preflight()
        self.assertEqual(self.factory_calls, [])
        checkpoint.write_bytes(b"synthetic-checkpoint-first")
        (self.root / "experiments/synthetic.json").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "experiment content changed"):
            self.preflight()
        self.assertEqual(self.factory_calls, [])

    def test_one_synthetic_cell_keeps_pixels_and_actor_actions_separate(self):
        environment = []

        def factory(**kwargs):
            from train import build_env
            inspect.signature(build_env).bind(**kwargs)
            value = SyntheticEnvironment(**kwargs)
            environment.append(value)
            return value

        entry = self.protocol["actors"][0]
        result, trace = run_cell(
            row=self.cells[0], actor=SyntheticActor(), actor_info=entry,
            actor_payload=self.actor_paths[0].read_bytes(),
            rules=G0EventRules(2000, 100, 20, 30, 0.0001),
            centerline_far_threshold_m=9.0, env_factory=factory,
        )
        self.assertEqual(result["summary"]["outcome"], "finished")
        self.assertEqual(result["steps"], 1)
        self.assertEqual(result["driven_raw_frames"], 1)
        self.assertEqual(result["reset_initial_raw_frames"], 1)
        self.assertEqual(result["reset_noop_raw_frames"], 0)
        np.testing.assert_array_equal(trace["raw_official_action"][0], [0.0, 0.5, 0.5])
        self.assertEqual(trace["initial_stack"].shape, (4, 84, 84))
        self.assertEqual(trace["next_frame"].shape, (1, 84, 84))
        self.assertTrue(environment[0].closed)

    def test_ledger_drift_after_first_cell_stops_before_second_reset(self):
        context = self.preflight()
        environments = []

        def factory(**kwargs):
            value = SyntheticEnvironment(**kwargs)
            environments.append(value)
            original_close = value.close

            def close():
                original_close()
                (self.root / "experiments/history.json").write_text("new TRAIN reset\n", encoding="utf-8")

            value.close = close
            return value

        with self.assertRaisesRegex(ValueError, "audit source changed"):
            collect(self.root, context, "runs/g0-partial", env_factory=factory)
        self.assertEqual(len(environments), 1)
        output = self.root / "runs/g0-partial"
        self.assertEqual(len(list((output / "aborts").glob("*.json"))), 1)
        self.assertFalse((output / "cells.jsonl").exists())
        self.assertFalse((output / "manifest.json").exists())

    def test_mid_cell_failure_leaves_attempt_and_debited_partial_receipt(self):
        context = self.preflight()

        class FaultyEnvironment(SyntheticEnvironment):
            def step(self, action):
                if action is None:
                    return super().step(action)
                if self.tile_visited_count:
                    raise RuntimeError("synthetic fault after one decision")
                self.t += 0.02
                self.tile_visited_count += 1
                image = np.zeros((4, 84, 84), dtype=np.float32)
                image[-1] = 1.0 / 255.0
                return image, 0.0, False, False, {
                    "seed": self.track_seed, "track_id": self.track_id, "collision": False,
                    "damage": 0.0, "progress": 0.1, "finish_qualified": False,
                    "finished": False, "retire_reason": None,
                }

        with self.assertRaisesRegex(RuntimeError, "synthetic fault"):
            collect(self.root, context, "runs/g0-abort", env_factory=FaultyEnvironment)
        output = self.root / "runs/g0-abort"
        attempts = list((output / "attempts").glob("*.json"))
        aborts = list((output / "aborts").glob("*.json"))
        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(aborts), 1)
        abort = json.loads(aborts[0].read_text(encoding="utf-8"))
        self.assertEqual(abort["decision_reservation"], 2000)
        self.assertEqual(abort["decision_calls_at_least"], 2)
        self.assertEqual(abort["decisions_completed"], 1)
        self.assertGreaterEqual(abort["driven_raw_frames_observed"], 1)
        self.assertTrue(abort["no_retry_or_relabel"])
        self.assertIsNotNone(abort["partial_trace_sha256"])
        self.assertFalse((output / "manifest.json").exists())

    def test_rejects_raw_cell_drift_and_action_substitution(self):
        entry = self.protocol["actors"][0]
        arguments = dict(
            row=self.cells[0], actor=SyntheticActor(), actor_info=entry,
            actor_payload=self.actor_paths[0].read_bytes(),
            rules=G0EventRules(2000, 100, 20, 30, 0.0001),
            centerline_far_threshold_m=9.0,
        )

        def wrong_raw_cell(**kwargs):
            env = SyntheticEnvironment(**kwargs)
            env.track_seed += 1
            original_reset = env.reset

            def misleading_reset(**reset_kwargs):
                observation, info = original_reset(**reset_kwargs)
                info["seed"] = kwargs["seed"]
                return observation, info

            env.reset = misleading_reset
            return env

        with self.assertRaisesRegex(ValueError, "raw road differs"):
            run_cell(env_factory=wrong_raw_cell, **arguments)

        class TamperingWrapper:
            def __init__(self, **kwargs):
                self.env = SyntheticEnvironment(**kwargs)
                self.unwrapped = self.env

            def reset(self, *, seed=None, options=None):
                return self.env.reset(seed=seed, options=options)

            def step(self, action):
                altered = np.asarray(action).copy()
                altered[1] = 0.9
                return self.env.step(altered)

            def close(self):
                self.env.close()

        with self.assertRaisesRegex(ValueError, "raw CarRacing.step received a substituted action"):
            run_cell(env_factory=TamperingWrapper, **arguments)

    def test_reset_tick_count_survives_internal_warmup_reset(self):
        class ResetAgain(SyntheticEnvironment):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.warmup_steps = 2
                self.first_reset = True

            def reset(self, *, seed=None, options=None):
                observation, info = super().reset(seed=seed, options=options)
                if self.first_reset:
                    self.first_reset = False
                    self.step(np.zeros(3))
                    observation, info = self.reset(seed=seed, options=options)
                    self.step(np.zeros(3))
                self.tile_visited_count = 0
                return observation, info

            def step(self, action):
                if action is not None and np.array_equal(action, [0.0, 0.0, 0.0]):
                    self.t += 0.02
                    return np.zeros((4, 84, 84), dtype=np.float32), 0.0, False, False, {}
                return super().step(action)

        entry = self.protocol["actors"][0]
        result, _ = run_cell(
            row=self.cells[0], actor=SyntheticActor(), actor_info=entry,
            actor_payload=self.actor_paths[0].read_bytes(),
            rules=G0EventRules(2000, 100, 20, 30, 0.0001),
            centerline_far_threshold_m=9.0, env_factory=ResetAgain,
        )
        self.assertEqual(result["reset_initial_raw_frames"], 2)
        self.assertEqual(result["reset_noop_raw_frames"], 2)

    def test_paired_actor_road_hash_mismatch_is_not_a_comparison(self):
        context = self.preflight()
        created = []

        def factory(**kwargs):
            env = SyntheticEnvironment(**kwargs)
            if created:
                env.track[0] = (0.0, 0.0, 31.0, 0.0)
            created.append(env)
            return env

        with self.assertRaisesRegex(ValueError, "different road centerlines"):
            collect(self.root, context, "runs/g0-road-mismatch", env_factory=factory)
        self.assertEqual(len(created), 2)
        output = self.root / "runs/g0-road-mismatch"
        self.assertTrue((output / "cells.jsonl").is_file())
        self.assertEqual(len(list((output / "aborts").glob("*.json"))), 1)

    def test_protocol_freezer_pins_audited_cells_and_rejects_later_mutation(self):
        with self.assertRaisesRegex(ValueError, "experiments/\\*g0\\*"):
            freeze_g0_protocol(
                self.root, audit_path="experiments/audit.json",
                audit_sha256=self.protocol["geometry_audit_sha256"],
                output="experiments/failure-diagnostic.json",
                audit_verifier=lambda start, track, **kwargs: self.canonical_audit,
                payload_loader=self.payload, expected_actors=self.expected_actors,
            )
        protocol, digest = freeze_g0_protocol(
            self.root, audit_path="experiments/audit.json",
            audit_sha256=self.protocol["geometry_audit_sha256"],
            output="experiments/frozen-g0.json",
            audit_verifier=lambda start, track, **kwargs: self.canonical_audit,
            payload_loader=self.payload, expected_actors=self.expected_actors,
        )
        self.assertEqual(protocol["status"], "frozen")
        self.assertEqual(protocol["cells"], self.cells)
        self.assertEqual(len(protocol["source_hashes"]), len(SOURCE_FILES))
        self.assertEqual(len(digest), 64)
        verified = preflight(
            self.root, "experiments/frozen-g0.json", digest,
            actor_factory=self.factory, payload_loader=self.payload,
            audit_verifier=lambda start, track, **kwargs: self.canonical_audit,
            expected_actors=self.expected_actors,
        )
        self.assertEqual(len(verified["actors"]), 2)
        with self.assertRaisesRegex(ValueError, "new experiments"):
            freeze_g0_protocol(
                self.root, audit_path="experiments/audit.json",
                audit_sha256=self.protocol["geometry_audit_sha256"],
                output="experiments/frozen-g0.json",
                audit_verifier=lambda start, track, **kwargs: self.canonical_audit,
                payload_loader=self.payload, expected_actors=self.expected_actors,
            )
        self.assertTrue((self.root / "experiments/frozen-g0.json").is_file())

    def test_pool_claim_prevents_double_spend_through_another_output_path(self):
        context = self.preflight()
        created = []

        def factory(**kwargs):
            created.append(kwargs)
            return SyntheticEnvironment(**kwargs)

        manifest = collect(self.root, context, "runs/g0-first", env_factory=factory)
        self.assertEqual(manifest["cell_count"], 24)
        self.assertEqual(manifest["geometry_count"], 12)
        self.assertEqual(manifest["decisions_spent"], 24)
        summary = summarize_run(self.root, "runs/g0-first")
        self.assertEqual(summary["geometry_clusters"], 12)
        self.assertEqual(summary["diagnostic_episodes"], 24)
        self.assertEqual(summary["outcomes_by_actor"]["first"]["finished"], 12)
        (self.root / "experiments/audit.json").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen JSON hash changed"):
            summarize_run(self.root, "runs/g0-first")
        self.save()
        with self.assertRaisesRegex(ValueError, "already reserved"):
            collect(self.root, context, "runs/g0-second", env_factory=factory)
        self.assertEqual(len(created), 24)
        self.assertFalse((self.root / "runs/g0-second").exists())

    def test_post_drive_storage_failure_still_debits_first_attempt(self):
        context = self.preflight()
        with patch("scripts.diagnose_rlpd_g0.np.savez_compressed", side_effect=OSError("storage failure")):
            with self.assertRaisesRegex(OSError, "storage failure"):
                collect(self.root, context, "runs/g0-write-abort", env_factory=SyntheticEnvironment)
        output = self.root / "runs/g0-write-abort"
        aborts = list((output / "aborts").glob("*.json"))
        self.assertEqual(len(aborts), 1)
        abort = json.loads(aborts[0].read_text(encoding="utf-8"))
        self.assertEqual(abort["decisions_completed"], 1)
        self.assertEqual(abort["decision_reservation"], 2000)
        self.assertEqual(abort["phase"], "writing_trace_and_ledger")
        self.assertIn("storage failure", abort["partial_trace_error"])
        with self.assertRaisesRegex(ValueError, "already reserved"):
            collect(self.root, context, "runs/g0-second", env_factory=SyntheticEnvironment)

    def test_last_cell_source_drift_cannot_issue_complete_manifest(self):
        context = self.preflight()
        created = []

        def factory(**kwargs):
            env = SyntheticEnvironment(**kwargs)
            created.append(env)
            if len(created) == 24:
                original_close = env.close

                def close():
                    original_close()
                    (self.root / "experiments/history2.json").write_text("new allocation\n", encoding="utf-8")

                env.close = close
            return env

        with self.assertRaisesRegex(ValueError, "audit source changed"):
            collect(self.root, context, "runs/g0-last-cell-drift", env_factory=factory)
        output = self.root / "runs/g0-last-cell-drift"
        self.assertEqual(len(created), 24)
        self.assertEqual(len((output / "cells.jsonl").read_text(encoding="utf-8").splitlines()), 23)
        aborts = list((output / "aborts").glob("*.json"))
        self.assertEqual(len(aborts), 1)
        self.assertEqual(json.loads(aborts[0].read_text(encoding="utf-8"))["decisions_completed"], 1)
        self.assertFalse((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
