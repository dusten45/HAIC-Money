"""Synthetic-only contract tests; no road simulator or real dataset is opened."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common_adapter import ActionAdapter
from dreamer_v3 import DreamerV3Agent, DreamerV3Config
from haic.algorithms.drq_v2.teacher_replay import TeacherDataset, TeacherEpisode
from scripts import train_dreamerv3_reused_train as runner


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestReusedTrainOffline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.collection_name = "experiments/dreamerv3-reused-train-diagnostic-v1.json"
        self.protocol_name = "experiments/dreamerv3-reused-train-offline-v1.json"
        self.output_name = "runs/synthetic-offline/teacher-seed-23"
        (self.root / "experiments").mkdir()
        (self.root / "runs/synthetic-collection/random").mkdir(parents=True)
        (self.root / "runs/synthetic-collection/teacher").mkdir(parents=True)
        (self.root / "runs/synthetic-offline").mkdir(parents=True)
        self.cgroup = {
            "limit_bytes": 4 * 1024**3, "used_bytes": 1024**3,
            "available_bytes": 3 * 1024**3,
        }
        cgroup_patch = patch.object(runner, "_cgroup_memory", return_value=self.cgroup)
        cgroup_patch.start()
        self.addCleanup(cgroup_patch.stop)
        r6 = {
            "study_id": "drqv2-geometry-mix-v1-r6",
            "training_pool": {"partition": "TRAIN", "track_ids": [1, 2, 3, 4],
                              "geometry_seeds": [111, 222]},
            "diagnostic_pool": {"partition": "TRAIN-DIAGNOSTIC", "geometry_seeds": [333]},
        }
        self.put(runner.R6_PATH, r6)
        pinned_r6 = patch.object(runner, "R6_SHA256", runner._sha256(self.root / runner.R6_PATH))
        pinned_r6.start()
        self.addCleanup(pinned_r6.stop)
        collector = self.root / runner.COLLECTOR_PATH
        collector.parent.mkdir(parents=True)
        collector.write_text("# synthetic collector executable identity\n", encoding="ascii")
        self.collection = {
            "format": runner.COLLECTION_FORMAT, "purpose": runner.PURPOSE,
            "study_id": "dreamerv3-reused-train-synthetic-v1",
            "r6_protocol": {"path": runner.R6_PATH, "sha256": runner._sha256(self.root / runner.R6_PATH)},
            "cells": [{"track_id": 1, "geometry_seed": 111}],
            "excluded_seeds": [333, 444],
            "frame_skip": 4, "max_steps": 2000,
            "budgets": {"random_decision_cap": 4, "teacher_decision_cap": 4},
            "source_sha256": {runner.COLLECTOR_PATH: runner._sha256(collector)},
            "random": {"source_id": "uniform-random-seed7", "source_actor_sha256": "b" * 64},
            "source_actor": {"source_id": "drq-source-0", "actor_sha256": "a" * 64},
        }
        self.put(self.collection_name, self.collection)
        sources = {}
        for name in runner.SOURCE_PATHS:
            source = self.root / name
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(runner.ROOT / name, source)
            sources[name] = runner._sha256(source)
        datasets = {}
        for arm in ("random", "teacher"):
            identity = self.collection["random" if arm == "random" else "source_actor"]
            actor_hash = identity["source_actor_sha256" if arm == "random" else "actor_sha256"]
            archive, digest = self.episode_bytes(identity["source_id"], actor_hash, arm == "teacher")
            prefix = f"runs/synthetic-collection/{arm}"
            archive_name = f"{prefix}/support-dataset.npz"
            receipt_name = f"{prefix}/collection-result.json"
            (self.root / archive_name).write_bytes(archive)
            dataset = {
                "receipt_path": receipt_name, "archive_path": archive_name,
                "archive_sha256": sha(archive), "dataset_digest": digest,
                "source_id": identity["source_id"], "source_actor_sha256": actor_hash,
                "stored_decisions": 3, "allowed_cells": self.collection["cells"],
                "excluded_seeds": self.collection["excluded_seeds"],
            }
            receipt = {
                "format": "haic-dreamerv3-reused-train-collection-result-v1",
                "purpose": runner.PURPOSE, "status": "completed", "schedule_exhausted": True,
                "schedule_attempts": 1, "complete_episode_count": 1,
                "partial_decisions": 0, "unresolved_decision_calls": 0,
                "study_id": self.collection["study_id"],
                "protocol_sha256": runner._sha256(self.root / self.collection_name),
                "arm": arm, "source_id": identity["source_id"], "source_actor_sha256": actor_hash,
                "archive_sha256": dataset["archive_sha256"], "dataset_digest": digest,
                "allowed_cells": self.collection["cells"], "excluded_seeds": self.collection["excluded_seeds"],
                "stored_decisions": 3, "decisions_spent": 3, "dataset_path": "support-dataset.npz",
                "decision_cap": 4,
                "complete_episodes": 1,
            }
            self.put(receipt_name, receipt)
            dataset["receipt_sha256"] = runner._sha256(self.root / receipt_name)
            datasets[arm] = dataset
        config = asdict(DreamerV3Config(
            device="cpu", embed_dim=64, hidden_dim=32, num_categoricals=4, num_classes=4,
            batch_size=1, seq_len=4, burnin_steps=1, short_episode_fraction=1.0,
            replay_capacity=8, imagination_horizon=2, warmup_steps=0,
        ))
        self.protocol = {
            "format": runner.FORMAT, "purpose": runner.PURPOSE,
            "study_id": self.collection["study_id"],
            "collection_protocol": {"path": self.collection_name,
                                    "sha256": runner._sha256(self.root / self.collection_name)},
            "source_sha256": sources, "datasets": datasets,
            "learner": {"seeds": [23, 24], "updates": 2, "config": config},
            "resources": {"max_archive_bytes": 8 * 1024**2,
                          "max_cgroup_memory_bytes": 8 * 1024**3,
                          "min_cgroup_available_bytes": 512 * 1024**2},
        }
        self.freeze()

    def put(self, name, payload):
        (self.root / name).write_text(json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8")

    def freeze(self):
        self.put(self.protocol_name, self.protocol)
        return runner._sha256(self.root / self.protocol_name)

    def run_arm(self, arm="teacher", seed=23):
        return runner.train_arm(self.protocol_name, self.freeze(), arm, seed, self.output_name,
                                repo_root=self.root)

    @staticmethod
    def episode_bytes(source_id, actor_hash, finished):
        frames = np.stack([np.full((84, 84), i + 10, dtype=np.uint8) for i in range(4)])
        actions = np.tile([0.25, -0.4, 0.15], (3, 1)).astype(np.float32)
        episode = TeacherEpisode(
            frames=frames, actions=actions,
            applied_actions=np.stack([ActionAdapter().to_official(action) for action in actions]),
            rewards=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            terminated=[False] * 3, truncated=[False, False, True],
            terminal=[False, False, finished], finished=[False, False, finished],
            episode_id=0, source_id=source_id, source_actor_sha256=actor_hash,
            geometry_id="111", track_id=1, complete=True,
        )
        dataset = TeacherDataset([episode])
        digest = dataset.seal()
        return dataset.to_bytes(), digest

    def test_model_only_exact_budget_preserves_untrained_actor_critic_and_zero_environment_steps(self):
        calls = []
        original = DreamerV3Agent.update

        def spy(agent, *, model_only=False):
            calls.append(model_only)
            return original(agent, model_only=model_only)

        with patch.object(DreamerV3Agent, "update", spy):
            result = self.run_arm()
        self.assertEqual(calls, [True, True])
        self.assertEqual(result["model_only_updates"], 2)
        self.assertEqual(result["environment_steps"], 0)
        self.assertEqual(result["dataset_evidence"]["decisions"], 3)
        self.assertEqual(result["dataset_evidence"]["terminal_events"], 1)
        self.assertEqual(result["dataset_evidence"]["distinct_finished_cells"], 1)
        self.assertEqual(result["dataset_evidence"]["finished_cells"], [{"track_id": 1, "geometry_seed": 111}])
        self.assertEqual(result["collection_decisions_spent"], 3)
        self.assertTrue(result["actor_critic_target_unchanged"])
        self.assertFalse(result["actor_trained"])
        self.assertFalse(result["fresh_claim"])
        self.assertFalse(result["p1b_claim"])
        self.assertFalse(result["promotion_eligible"])
        output = self.root / self.output_name
        self.assertEqual(runner._sha256(output / result["checkpoint_path"]), result["checkpoint_sha256"])
        self.assertEqual(runner._sha256(output / result["metrics_path"]), result["metrics_sha256"])
        rows = (output / result["metrics_path"]).read_text().splitlines()
        self.assertEqual([json.loads(row)["update"] for row in rows], [1, 2])
        payload = torch.load(output / result["checkpoint_path"], map_location="cpu", weights_only=False)
        self.assertEqual(payload["environment_steps"], 0)
        self.assertEqual(payload["gradient_steps"], 2)
        self.assertEqual(payload["trainer_state"], None)
        self.assertFalse(payload["run_metadata"]["promotion_eligible"])
        self.assertFalse(payload["actor_optimizer"]["state"])
        self.assertFalse(payload["critic_optimizer"]["state"])
        with patch.object(Path, "read_bytes", side_effect=AssertionError("no archive read")):
            with self.assertRaises(FileExistsError):
                self.run_arm()

    def test_wrong_source_receipt_sha_and_cell_fail_before_archive_load(self):
        changes = (
            ("source identity", lambda: self.protocol["datasets"]["teacher"].update(source_id="foreign"), "source identity"),
            ("receipt SHA", lambda: self.protocol["datasets"]["random"].update(receipt_sha256="0" * 64), "SHA-256"),
            ("collection SHA", lambda: self.protocol["collection_protocol"].update(sha256="0" * 64), "SHA-256"),
            ("executable SHA", lambda: self.protocol["source_sha256"].update({"dreamer_v3.py": "0" * 64}), "SHA-256"),
        )
        for label, change, message in changes:
            with self.subTest(label=label):
                saved_protocol = json.loads(json.dumps(self.protocol))
                saved_collection = json.loads(json.dumps(self.collection))
                change()
                with patch.object(Path, "read_bytes", side_effect=AssertionError("archive load before preflight")):
                    with self.assertRaisesRegex(ValueError, message):
                        self.run_arm()
                self.assertFalse((self.root / self.output_name).exists())
                self.protocol = saved_protocol
                self.collection = saved_collection

    def test_reused_train_membership_and_exclusions_are_pinned(self):
        self.collection["cells"] = [{"track_id": 1, "geometry_seed": 333}]
        self.put(self.collection_name, self.collection)
        self.protocol["collection_protocol"]["sha256"] = runner._sha256(self.root / self.collection_name)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("archive load before membership")):
            with self.assertRaisesRegex(ValueError, "known-reused r6 TRAIN"):
                self.run_arm()
        self.assertFalse((self.root / self.output_name).exists())

    def test_aborted_or_partial_collection_never_trains(self):
        receipt_name = self.protocol["datasets"]["teacher"]["receipt_path"]
        original = json.loads((self.root / receipt_name).read_text())
        for change in ({"status": "aborted"}, {"schedule_exhausted": False},
                       {"partial_decisions": 1, "decisions_spent": 4}):
            with self.subTest(change=change):
                receipt = {**original, **change}
                self.put(receipt_name, receipt)
                self.protocol["datasets"]["teacher"]["receipt_sha256"] = runner._sha256(
                    self.root / receipt_name,
                )
                with patch.object(Path, "read_bytes", side_effect=AssertionError("archive loaded")):
                    with self.assertRaisesRegex(ValueError, "collection receipt|incomplete collection"):
                        self.run_arm()
                self.assertFalse((self.root / self.output_name).exists())

    def test_replaced_r6_hash_is_rejected_before_archive_load(self):
        self.collection["r6_protocol"]["sha256"] = "0" * 64
        self.put(self.collection_name, self.collection)
        self.protocol["collection_protocol"]["sha256"] = runner._sha256(self.root / self.collection_name)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("archive load before r6 check")):
            with self.assertRaisesRegex(ValueError, "known reused r6 TRAIN"):
                self.run_arm()
        self.assertFalse((self.root / self.output_name).exists())

    def test_archive_and_cgroup_caps_fail_before_archive_bytes(self):
        for change, message in (
            (lambda: self.protocol["resources"].update(max_archive_bytes=1), "archive size"),
            (lambda: self.cgroup.update(available_bytes=1), "cgroup memory"),
        ):
            with self.subTest(message=message):
                saved_protocol = json.loads(json.dumps(self.protocol))
                saved_cgroup = self.cgroup.copy()
                change()
                with patch.object(Path, "read_bytes", side_effect=AssertionError("archive load before resource guard")):
                    with self.assertRaisesRegex(ValueError, message):
                        self.run_arm()
                self.assertFalse((self.root / self.output_name).exists())
                self.protocol = saved_protocol
                self.cgroup.clear()
                self.cgroup.update(saved_cgroup)

    def test_archive_sha_abort_is_preserved_without_overwrite_or_resume(self):
        with (self.root / self.protocol["datasets"]["teacher"]["archive_path"]).open("ab") as stream:
            stream.write(b"modified-after-freeze")
        with patch.object(runner, "replay_from_dataset_bytes", side_effect=AssertionError("must not deserialize")):
            with self.assertRaisesRegex(ValueError, "archive SHA-256"):
                self.run_arm()
        output = self.root / self.output_name
        abort = json.loads((output / "abort.json").read_text())
        self.assertEqual(abort["phase"], "archive verification")
        self.assertEqual(abort["completed_model_only_updates"], 0)
        self.assertFalse(abort["promotion_eligible"])
        with self.assertRaises(FileExistsError):
            self.run_arm()

    def test_symlink_and_unsafe_archive_paths_fail_before_archive_bytes(self):
        archive = self.root / self.protocol["datasets"]["teacher"]["archive_path"]
        copy = archive.with_name("immutable-copy.npz")
        archive.rename(copy)
        archive.symlink_to(copy)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("archive load before path check")):
            with self.assertRaisesRegex(ValueError, "symlink path"):
                self.run_arm()
        self.assertFalse((self.root / self.output_name).exists())
        archive.unlink()
        copy.rename(archive)
        self.protocol["datasets"]["teacher"]["archive_path"] = "runs/synthetic-collection/teacher/../support-dataset.npz"
        with self.assertRaisesRegex(ValueError, "normalized repository-relative"):
            self.run_arm()

    def test_unavailable_sequence_is_named_and_keeps_abort_evidence(self):
        self.protocol["learner"]["config"].update(seq_len=4, burnin_steps=1, short_episode_fraction=0.0)
        with self.assertRaisesRegex(runner.UnavailableSequenceError, "unavailable replay sequence"):
            self.run_arm()
        abort = json.loads((self.root / self.output_name / "abort.json").read_text())
        self.assertEqual(abort["error_type"], "UnavailableSequenceError")
        self.assertEqual(abort["completed_model_only_updates"], 0)
        self.assertFalse((self.root / self.output_name / "world-model-checkpoint.pt").exists())

    def test_nonfinite_update_keeps_abort_evidence(self):
        original = DreamerV3Agent.update

        def bad_metric(agent, *, model_only=False):
            metrics = original(agent, model_only=model_only)
            metrics["loss_wm"] = float("nan")
            return metrics

        with patch.object(DreamerV3Agent, "update", bad_metric):
            with self.assertRaisesRegex(ValueError, "invalid metrics"):
                self.run_arm()
        abort = json.loads((self.root / self.output_name / "abort.json").read_text())
        self.assertEqual(abort["completed_model_only_updates"], 0)
        self.assertEqual((self.root / self.output_name / "update-metrics.jsonl").read_text(), "")

    def test_actor_mutation_and_cgroup_pressure_abort_before_success(self):
        original = DreamerV3Agent.update

        def bad_actor(agent, *, model_only=False):
            metrics = original(agent, model_only=model_only)
            with torch.no_grad():
                next(agent.actor.parameters()).add_(0.01)
            return metrics

        with patch.object(DreamerV3Agent, "update", bad_actor):
            with self.assertRaisesRegex(ValueError, "actor/critic update"):
                self.run_arm()
        abort = json.loads((self.root / self.output_name / "abort.json").read_text())
        self.assertEqual(abort["completed_model_only_updates"], 0)
        self.output_name = "runs/synthetic-offline/teacher-seed-24"
        low = dict(self.cgroup, available_bytes=1)
        with patch.object(runner, "_cgroup_memory", side_effect=[self.cgroup, self.cgroup, self.cgroup, low]):
            with self.assertRaisesRegex(ValueError, "frozen training resource floor"):
                self.run_arm(seed=24)
        output = self.root / self.output_name
        abort = json.loads((output / "abort.json").read_text())
        self.assertEqual(abort["completed_model_only_updates"], 1)
        self.assertEqual(len((output / "update-metrics.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
