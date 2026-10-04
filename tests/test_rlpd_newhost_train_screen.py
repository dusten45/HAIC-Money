"""Synthetic, zero-real-reset checks for the paired in-sample RLPD screen."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common_adapter import ActionSpec, ObservationSpec
from scripts import diagnose_rlpd_newhost_train_screen as screen
from scripts.diagnose_rlpd_g0 import run_cell
from tests.test_rlpd_g0_runner import SyntheticActor, SyntheticEnvironment


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ScreenTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "experiments").mkdir()
        (self.root / "runs").mkdir()
        self.runtime = {"device": "cpu", "torch_num_threads": 1, "test_runtime": True}
        self.environments = []
        self.factory_calls = []
        for relative in screen.SOURCE_FILES | {"scripts/train_rlpd_newhost.py"}:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic source: {relative}".encode())
        source_map = {name: digest(self.root / name) for name in sorted(screen.G0_SOURCE_FILES)}
        self.cells = [{"partition": "TRAIN", "track_id": 1, "geometry_seed": 4272000001 + index,
                       "obstacles": True} for index in range(12)]
        self.old_actor_path = "runs/v5/checkpoints/step-000131072/actor.pt"
        old_sha = self.make_file(self.old_actor_path, b"synthetic V5 seed-50 actor")
        first_sha = self.make_file("runs/g0-first/actor.pt", b"synthetic first G0 actor")
        self.g0 = {
            "format": "haic-rlpd-g0-diagnostic-v1", "status": "frozen", "partition": "TRAIN",
            "frame_skip": 4, "max_steps": 2000, "decision_cap": 48000,
            "reward_shaping": False, "collision_penalty": 0.0, "interventions": False,
            "learner_updates": 0, "centerline_far_threshold_m": 9.0,
            "event_rules": {"max_decisions": 2000, "negative_reward_limit": 100,
                            "stall_window": 20, "tile_window": 30, "directed_delta_epsilon": 0.0001},
            "cells": self.cells, "source_hashes": source_map,
            "actors": [
                {"id": "long-horizon-seed11", "path": "runs/g0-first/actor.pt", "sha256": first_sha},
                {"id": "entropy-v5-author-seed50", "path": self.old_actor_path, "sha256": old_sha,
                 "source_sha256": "a" * 64, "export_protocol_sha256": "b" * 64,
                 "action_mode": "exported_tanh_mean", "environment_steps": 131072, "training_seed": 50},
            ],
        }
        g0_sha = self.make_json(screen.G0_PROTOCOL, self.g0)
        centerline = np.asarray(SyntheticEnvironment(track_id=1, seed=4272000001).track, dtype=np.float64)[:, 2:4]
        self.road_sha = hashlib.sha256(np.ascontiguousarray(centerline.astype("<f8")).tobytes()).hexdigest()
        ledger = []
        for cell in self.cells:
            for actor in self.g0["actors"]:
                ledger.append({"partition": "TRAIN", "track_id": 1,
                               "geometry_seed": cell["geometry_seed"], "actor_id": actor["id"],
                               "actor_sha256": actor["sha256"], "steps": 1,
                               "summary": {"outcome": "finished"},
                               "road_centerline_sha256": self.road_sha})
        ledger_path = self.root / screen.G0_LEDGER
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text("".join(json.dumps(row) + "\n" for row in ledger))
        ledger_sha = digest(ledger_path)
        manifest_sha = self.make_json(screen.G0_MANIFEST, {
            "format": "haic-rlpd-g0-diagnostic-result-v1", "protocol_sha256": g0_sha,
            "cells_sha256": ledger_sha, "cell_count": 24, "geometry_count": 12,
        })
        learner_sources = {**source_map, "scripts/train_rlpd_newhost.py": digest(self.root / "scripts/train_rlpd_newhost.py")}
        learner_protocol = {
            "format": "haic-rlpd-newhost-reused-train-v1", "status": "frozen",
            "cells": self.cells, "g0_protocol_sha256": g0_sha,
            "g0_manifest_sha256": manifest_sha, "g0_ledger_sha256": ledger_sha,
            "seed": 52, "total_steps": 131072, "candidate_steps": [65536, 131072],
            "frame_skip": 4, "max_steps": 2000, "obstacles": True,
            "reward_shaping": False, "official_performance_claim": False,
            "v5_prior_dataset_sha256": "c" * 64,
            "source_hashes": learner_sources,
        }
        learner_sha = self.make_json(screen.LEARNER_PROTOCOL, learner_protocol)
        self.make_file(screen.LEARNER_SNAPSHOT, (self.root / screen.LEARNER_PROTOCOL).read_bytes())
        new_actor_sha = self.make_file(screen.FINAL_ACTOR, b"synthetic new-host final actor")
        checkpoint_sha = self.make_file(screen.FINAL_CHECKPOINT, b"small synthetic final checkpoint")
        candidate = {"actor_path": "checkpoints/step-000131072/actor.pt", "actor_sha256": new_actor_sha,
                     "checkpoint_path": "checkpoints/step-000131072/checkpoint.pt",
                     "checkpoint_sha256": checkpoint_sha, "protocol_sha256": learner_sha,
                     "environment_steps": 131072, "gradient_steps": 130072}
        candidate_sha = self.make_json(screen.FINAL_CANDIDATE, candidate)
        result_sha = self.make_json(screen.LEARNER_RESULT, {
            "format": "haic-rlpd-newhost-result-v1", "status": "complete",
            "reused_train_only": True, "official_performance_claim": False,
            "protocol_sha256": learner_sha, "g0_ledger_sha256": ledger_sha,
            "environment_steps": 131072, "gradient_steps": 130072,
            "candidates": [{"environment_steps": 65536}, candidate],
        })
        constants = {"G0_SHA": g0_sha, "G0_MANIFEST_SHA": manifest_sha, "G0_LEDGER_SHA": ledger_sha,
                     "V5_ACTOR_SHA": old_sha, "LEARNER_PROTOCOL_SHA": learner_sha,
                     "LEARNER_RESULT_SHA": result_sha, "FINAL_CANDIDATE_SHA": candidate_sha,
                     "FINAL_ACTOR_SHA": new_actor_sha, "FINAL_CHECKPOINT_SHA": checkpoint_sha}
        for name, value in constants.items():
            patcher = patch.object(screen, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.protocol_path = "experiments/synthetic-in-sample-screen.json"
        self.run_dir = "runs/synthetic-paired-screen"

    def make_file(self, relative: str, content: bytes) -> str:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return digest(path)

    def make_json(self, relative: str, value: dict) -> str:
        return self.make_file(relative, (json.dumps(value, sort_keys=True) + "\n").encode())

    def actor_factory(self, path):
        self.factory_calls.append(path)
        return SyntheticActor()

    def payload_loader(self, path):
        if str(path).endswith(self.old_actor_path):
            source, protocol, seed = "a" * 64, "b" * 64, 50
        else:
            learner = json.loads((self.root / screen.LEARNER_PROTOCOL).read_text())
            source, protocol, seed = screen.canonical_sha(learner["source_hashes"]), screen.LEARNER_PROTOCOL_SHA, 52
        payload = {"format": "haic-rlpd-pixel-actor-v1", "source_sha256": source,
                "protocol_sha256": protocol, "training_seed": seed, "environment_steps": 131072,
                "observation_spec": asdict(ObservationSpec()), "action_spec": asdict(ActionSpec())}
        if seed == 52:
            payload["gradient_steps"] = 130072
            payload["environment_contract"] = {
                "partition": "TRAIN", "reused_g0_cells": self.cells,
                "v5_prior_dataset_sha256": "c" * 64, "obstacles": True,
                "reward_shaping": False, "frame_skip": 4, "max_steps": 2000,
            }
        return payload

    def synthetic_env(self, **kwargs):
        env = SyntheticEnvironment(**kwargs)
        self.environments.append(env)
        return env

    def ready(self, run_dir=None):
        self.run_dir = run_dir or self.run_dir
        frozen = screen.freeze(self.root, self.protocol_path, self.run_dir, runtime_fn=lambda: self.runtime)
        context = screen.preflight(self.root, self.protocol_path, frozen["sha256"],
                                   runtime_fn=lambda: self.runtime,
                                   actor_factory=self.actor_factory, payload_loader=self.payload_loader)
        return context

    def assert_advertised_paths_exist(self, output):
        for directory in ("failures", "partials"):
            for path in (output / directory).glob("*.json"):
                receipt = json.loads(path.read_text())
                for key, value in receipt.items():
                    if key.endswith("_path") and value is not None:
                        with self.subTest(receipt=path.name, field=key):
                            self.assertTrue((output / value).is_file())

    def test_freeze_and_cpu_preflight_are_zero_reset_and_source_bound(self):
        context = self.ready()
        self.assertEqual(len(self.factory_calls), 4)
        self.assertEqual(len(self.environments), 0)
        self.assertEqual(context["protocol"]["execution"]["core_hour_cap"], None)
        self.assertEqual(context["protocol"]["execution"]["cpu_nice_delta"], 10)
        self.assertEqual(context["protocol"]["output_root"], self.run_dir)
        self.assertEqual(context["protocol"]["execution"]["scheduled_slots"], 24)
        self.assertEqual(len(context["protocol"]["cells"]), 12)
        self.assertEqual(set(context["protocol"]["source_hashes"]), screen.SOURCE_FILES)
        self.assertEqual(context["protocol"]["actors"][0]["sha256"], screen.V5_ACTOR_SHA)
        with self.assertRaises(FileExistsError):
            screen.freeze(self.root, self.protocol_path, self.run_dir, runtime_fn=lambda: self.runtime)

    def test_tampered_lineage_and_sources_block_before_actor_or_reset(self):
        frozen = screen.freeze(self.root, self.protocol_path, self.run_dir, runtime_fn=lambda: self.runtime)
        for relative in (screen.LEARNER_RESULT, screen.FINAL_CANDIDATE, screen.G0_LEDGER,
                         screen.FINAL_ACTOR, screen.FINAL_CHECKPOINT, "train.py"):
            path = self.root / relative
            old = path.read_bytes()
            try:
                path.write_bytes(old + b"drift")
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    screen.preflight(self.root, self.protocol_path, frozen["sha256"],
                                     runtime_fn=lambda: self.runtime, actor_factory=self.actor_factory,
                                     payload_loader=self.payload_loader)
            finally:
                path.write_bytes(old)
        self.assertEqual(len(self.factory_calls), 0)
        self.assertEqual(len(self.environments), 0)

    def test_cpu_preflight_rejects_nonfinite_and_out_of_bounds_actions(self):
        frozen = screen.freeze(self.root, self.protocol_path, self.run_dir, runtime_fn=lambda: self.runtime)

        class BadActor(SyntheticActor):
            def __init__(self, action):
                self.action = action

            def act(self, observation):
                return self.action

        for values in ([np.nan, 0.5, 0.5], [0.0, -0.1, 0.5]):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "reload"):
                screen.preflight(self.root, self.protocol_path, frozen["sha256"],
                                 runtime_fn=lambda: self.runtime, payload_loader=self.payload_loader,
                                 actor_factory=lambda path: BadActor(np.asarray(values, dtype=np.float32)))
        self.assertEqual(len(self.environments), 0)

    def test_all_twelve_fake_env_pairs_have_exact_denominators_and_verified_trace_sha(self):
        context = self.ready()

        def checking_factory(**kwargs):
            slot = len(self.environments)
            receipt = self.root / f"runs/synthetic-paired-screen/attempts/slot-{slot:02d}.json"
            self.assertEqual(json.loads(receipt.read_text())["status"], "attempted-once")
            return self.synthetic_env(**kwargs)

        manifest = screen.collect(context, "runs/synthetic-paired-screen", env_factory=checking_factory)
        self.assertEqual(len(self.environments), 24)
        self.assertTrue(all(env.closed for env in self.environments))
        self.assertEqual(manifest["status"], "complete-in-sample-train-only")
        self.assertEqual(manifest["paired_roads"], 12)
        self.assertEqual(manifest["complete_slots"], 24)
        self.assertEqual(manifest["paired_finish_table"]["both"], 12)
        self.assertEqual(manifest["paired_finish_table"]["denominator"], 12)
        self.assertFalse((self.root / "runs/synthetic-paired-screen/manifest.json.pending").exists())
        for record in manifest["artifacts"]:
            slot = record["slot"]
            output = self.root / "runs/synthetic-paired-screen"
            self.assertEqual(record["cell_sha256"], digest(output / f"cells/slot-{slot:02d}.json"))
            self.assertEqual(record["trace_sha256"], digest(output / f"traces/slot-{slot:02d}.npz"))
        with self.assertRaises(FileExistsError):
            screen.collect(context, "runs/synthetic-paired-screen", env_factory=self.synthetic_env)
        self.assertEqual(len(self.environments), 24)

    def test_current_road_mismatch_is_censored_and_never_summarized(self):
        context = self.ready("runs/road-mismatch")
        calls = []

        def wrong_second_road(**kwargs):
            result, trace = run_cell(**kwargs)
            calls.append(kwargs["actor_info"]["id"])
            if len(calls) == 2:
                result["road_centerline_sha256"] = "f" * 64
            return result, trace

        with self.assertRaisesRegex(ValueError, "matched original G0"):
            screen.collect(context, "runs/road-mismatch", env_factory=self.synthetic_env, runner=wrong_second_road)
        output = self.root / "runs/road-mismatch"
        failure = json.loads((output / "failures/slot-01.json").read_text())
        self.assertEqual(failure["status"], "censored-invalid-attempt")
        self.assertTrue(failure["no_retry_or_topup"])
        self.assertEqual(len(self.environments), 2)
        self.assertEqual(len(list((output / "attempts").glob("*.json"))), 2)
        self.assertFalse((output / "manifest.json").exists())

    def test_mid_cell_failure_keeps_partial_and_no_retry(self):
        context = self.ready("runs/mid-cell-fault")

        class BrokenEnv(SyntheticEnvironment):
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

        def broken_factory(**kwargs):
            env = BrokenEnv(**kwargs)
            self.environments.append(env)
            return env

        with self.assertRaisesRegex(RuntimeError, "synthetic fault"):
            screen.collect(context, "runs/mid-cell-fault", env_factory=broken_factory)
        output = self.root / "runs/mid-cell-fault"
        failure = json.loads((output / "failures/slot-00.json").read_text())
        self.assertEqual(failure["decision_calls_at_least"], 2)
        self.assertEqual(failure["decisions_completed"], 1)
        self.assertIsNone(failure["partial_receipt_path"])
        partial = json.loads((output / "partials/slot-00.json").read_text())
        self.assertEqual(partial["decisions_recorded"], 1)
        self.assertEqual(partial["trace_sha256"], digest(output / "partials/slot-00.npz"))
        self.assertEqual(partial["failure_path"], "failures/slot-00.json")
        self.assert_advertised_paths_exist(output)
        self.assertEqual(len(self.environments), 1)
        self.assertTrue(self.environments[0].closed)
        self.assertFalse((output / "manifest.json").exists())

    def test_secondary_trace_write_failure_still_has_small_censor_receipt(self):
        context = self.ready("runs/trace-fault")
        with patch.object(screen.np, "savez_compressed", side_effect=OSError("synthetic disk full")):
            with self.assertRaisesRegex(OSError, "synthetic disk full"):
                screen.collect(context, "runs/trace-fault", env_factory=self.synthetic_env)
        output = self.root / "runs/trace-fault"
        receipt = json.loads((output / "failures/slot-00.json").read_text())
        self.assertTrue(receipt["no_retry_or_topup"])
        self.assertEqual(receipt["decisions_completed"], 1)
        self.assertIsNone(receipt["partial_receipt_path"])
        self.assertFalse((output / "partials/slot-00.json").exists())
        self.assert_advertised_paths_exist(output)
        self.assertEqual(len(list((output / "attempts").glob("*.json"))), 1)
        self.assertFalse((output / "manifest.json").exists())

    def test_failed_optional_partial_archive_advertises_no_broken_link(self):
        context = self.ready("runs/partial-storage-fault")

        def fail_after_one_fake_decision(**kwargs):
            kwargs["attempt"].update(phase="driving", decision_calls=2, decisions_completed=1,
                                     partial_arrays={"summed_reward": [1.0]})
            raise RuntimeError("synthetic partial episode fault")

        with patch.object(screen.np, "savez_compressed", side_effect=OSError("synthetic archive ENOSPC")):
            with self.assertRaisesRegex(RuntimeError, "partial episode fault"):
                screen.collect(context, "runs/partial-storage-fault",
                               env_factory=self.synthetic_env, runner=fail_after_one_fake_decision)
        output = self.root / "runs/partial-storage-fault"
        failure = json.loads((output / "failures/slot-00.json").read_text())
        self.assertEqual(failure["decisions_completed"], 1)
        self.assertIsNone(failure["partial_receipt_path"])
        self.assertFalse((output / "partials/slot-00.json").exists())
        self.assert_advertised_paths_exist(output)
        self.assertEqual(len(self.environments), 0)
        self.assertFalse((output / "manifest.json").exists())

    def test_pre_reset_source_drift_blocks_without_new_attempt(self):
        context = self.ready("runs/pre-reset-drift")
        path = self.root / "train.py"
        path.write_bytes(b"mutated after preflight")
        with self.assertRaisesRegex(ValueError, "G0 executable source changed"):
            screen.collect(context, "runs/pre-reset-drift", env_factory=self.synthetic_env)
        self.assertFalse((self.root / "runs/pre-reset-drift").exists())
        self.assertEqual(len(self.environments), 0)

    def test_between_slot_drift_halts_before_second_reset(self):
        context = self.ready("runs/between-slot-drift")
        original_recheck = screen._recheck
        checks = 0

        def recheck(*args, **kwargs):
            nonlocal checks
            checks += 1
            if checks == 4:
                raise ValueError("synthetic source drift between slots")
            return original_recheck(*args, **kwargs)

        with patch.object(screen, "_recheck", side_effect=recheck):
            with self.assertRaisesRegex(ValueError, "between slots"):
                screen.collect(context, "runs/between-slot-drift", env_factory=self.synthetic_env)
        output = self.root / "runs/between-slot-drift"
        self.assertEqual(len(self.environments), 1)
        self.assertEqual(len(list((output / "attempts").glob("*.json"))), 1)
        self.assertTrue((output / "cells/slot-00.json").is_file())
        self.assertEqual(json.loads((output / "halt.json").read_text())["next_slot"], 1)
        self.assertFalse((output / "manifest.json").exists())

    def test_changed_trace_blocks_complete_manifest(self):
        context = self.ready("runs/altered-trace")
        episodes = 0

        def corrupt_first_trace_at_end(**kwargs):
            nonlocal episodes
            result, trace = run_cell(**kwargs)
            episodes += 1
            if episodes == 24:
                (self.root / "runs/altered-trace/traces/slot-00.npz").write_bytes(b"corrupt")
            return result, trace

        with self.assertRaisesRegex(ValueError, "artifact changed"):
            screen.collect(context, "runs/altered-trace", env_factory=self.synthetic_env,
                           runner=corrupt_first_trace_at_end)
        output = self.root / "runs/altered-trace"
        self.assertEqual(len(self.environments), 24)
        self.assertEqual(json.loads((output / "halt.json").read_text())["status"], "post-collection-invalid")
        self.assertFalse((output / "manifest.json").exists())

    def test_checkpoint_changed_after_first_slot_blocks_final_manifest(self):
        context = self.ready("runs/checkpoint-drift")
        episodes = 0

        def change_checkpoint_after_first_slot(**kwargs):
            nonlocal episodes
            result, trace = run_cell(**kwargs)
            episodes += 1
            if episodes == 1:
                (self.root / screen.FINAL_CHECKPOINT).write_bytes(b"checkpoint changed after first slot")
            return result, trace

        with self.assertRaisesRegex(ValueError, "changed frozen evidence"):
            screen.collect(context, "runs/checkpoint-drift", env_factory=self.synthetic_env,
                           runner=change_checkpoint_after_first_slot)
        output = self.root / "runs/checkpoint-drift"
        self.assertEqual(len(self.environments), 24)
        self.assertEqual(len(list((output / "cells").glob("*.json"))), 24)
        self.assertEqual(json.loads((output / "halt.json").read_text())["status"], "post-collection-invalid")
        self.assertFalse((output / "manifest.json").exists())

    def test_manifest_write_failure_leaves_post_collection_halt(self):
        context = self.ready("runs/manifest-storage-fault")
        original_write = screen.write_json

        def fail_manifest_only(path, value):
            if path.name == "manifest.json.pending":
                path.write_bytes(b'{"truncated":')
                raise OSError("synthetic final ENOSPC")
            return original_write(path, value)

        with patch.object(screen, "write_json", side_effect=fail_manifest_only):
            with self.assertRaisesRegex(OSError, "synthetic final ENOSPC"):
                screen.collect(context, "runs/manifest-storage-fault", env_factory=self.synthetic_env)
        output = self.root / "runs/manifest-storage-fault"
        self.assertEqual(len(self.environments), 24)
        self.assertEqual(len(list((output / "attempts").glob("*.json"))), 24)
        self.assertEqual(len(list((output / "cells").glob("*.json"))), 24)
        self.assertFalse((output / "manifest.json").exists())
        self.assertTrue((output / "manifest.json.pending").exists())
        halt = json.loads((output / "halt.json").read_text())
        self.assertEqual(halt["status"], "post-collection-write-failure")
        self.assertEqual(halt["completed_slots"], 24)
        self.assertTrue(halt["no_retry"])

    def test_output_path_is_exclusive_and_cannot_escape_runs(self):
        context = self.ready()
        for path in ("runs/../old", "evaluations/new", "runs/nested/new", "runs/new_screen", "runs/alternate"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                screen.collect(context, path, env_factory=self.synthetic_env)
        with self.assertRaisesRegex(ValueError, "runs/<slug>"):
            screen.freeze(self.root, "experiments/other.json", "runs/../escape",
                          runtime_fn=lambda: self.runtime)
        self.assertEqual(len(self.environments), 0)


if __name__ == "__main__":
    unittest.main()
