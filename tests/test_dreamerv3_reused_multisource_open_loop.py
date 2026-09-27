"""Synthetic archives/checkpoints only; no road reset or real development read."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from common_adapter import ActionAdapter, Transition
from dreamer_v3 import DreamerV3Agent, DreamerV3Config
from haic.algorithms.drq_v2.teacher_replay import TeacherDataset, TeacherEpisode
from scripts.diagnose import dreamerv3_reused_multisource_open_loop as score


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Encoder:
    def __call__(self, stacks):
        return torch.zeros((len(stacks), 2))


class _RSSM:
    hidden_dim = stoch_dim = 1

    def observe_sequence(self, embeds, actions, first):
        return torch.zeros((1, embeds.shape[1], 2)), None, None

    def step_prior(self, h, z, action):
        return h, z, None, None


class _Decoder:
    def __init__(self, delta=1 / 255):
        self.delta = delta

    def __call__(self, state):
        return torch.full((1, 1, 84, 84), self.delta)


class _Reward:
    def pred(self, state):
        return torch.full((1,), 0.5)


class _Continue:
    def __call__(self, state):
        return torch.zeros((1, 1))


def _modules(delta=1 / 255):
    return _Encoder(), _RSSM(), _Decoder(delta), _Reward(), _Continue()


class SyntheticFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.roads = tuple(range(101, 113))
        self.dev_roads = (201, 202, 203, 204)
        self.sources = {}
        for name, value in (("ROADS", self.roads), ("DEV_ROADS", self.dev_roads),
                            ("TRAIN_DECISIONS", 24), ("REPLAY_CAPACITY", 128),
                            ("R6_SHA256", "0" * 64),
                            ("TRAIN_ROOT", "runs/synthetic-multisource"),
                            ("DEV_ROOT", "runs/synthetic-multisource/development"),
                            ("OUTPUT", "runs/synthetic-multisource/open-loop-v1")):
            patcher = mock.patch.object(score, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.output = self.root / score.OUTPUT
        self.output.parent.mkdir(parents=True)
        self.config = asdict(DreamerV3Config(device="cpu", embed_dim=8, hidden_dim=8,
                                             num_categoricals=2, num_classes=2, twohot_bins=3,
                                             replay_capacity=128))
        for name in score.SOURCE_PATHS | set(("scripts/train_dreamerv3_reused_multisource.py",
                                              "scripts/train_dreamerv3_reused_train.py",
                                              "haic/algorithms/dreamer_v3/offline.py")):
            source = score.RUNTIME_SOURCES.get(name)
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if source is None:
                dest.write_bytes(b"# synthetic unexecuted training source\n")
            else:
                shutil.copyfile(source, dest)
            self.sources[name] = _digest(dest)
        collector = "scripts/diagnose/collect_dreamerv3_reused_train.py"
        self.collector_sha = self.put_bytes(collector, b"# synthetic collector, never invoked\n")
        self.r6 = {"training_pool": {"partition": "TRAIN", "track_ids": [1, 2, 3, 4],
                                      "geometry_seeds": list(self.roads + self.dev_roads)},
                   "diagnostic_pool": {"partition": "TRAIN-DIAGNOSTIC", "geometry_seeds": [999]}}
        self.r6_sha = self.put_json("experiments/drqv2-geometry-mix-v1-r6.json", self.r6)
        self.patch_value("R6_SHA256", self.r6_sha)
        self.train = {"format": "haic-dreamerv3-reused-train-multisource-v1",
                      "purpose": "reused-TRAIN-engineering-diagnostic", "study_id": "synthetic-mixed",
                      "r6_protocol": {"path": "experiments/drqv2-geometry-mix-v1-r6.json", "sha256": self.r6_sha},
                      "cells": self.cells(self.roads), "learner": {"seeds": [0, 1], "updates": 256,
                                                               "config": self.config},
                      "source_sha256": {key: self.sources[key] for key in self.sources if key !=
                                        "scripts/diagnose/dreamerv3_reused_multisource_open_loop.py" and key !=
                                        "scripts/diagnose/dreamerv3_reused_train_open_loop.py"},
                      "datasets": {}}
        self.train_receipts = {}
        self.lineage = []
        for index, road in enumerate(self.roads):
            for ordinal, source in enumerate(("source0", "source1")):
                key = 2 * index + ordinal
                self.lineage.append({"source_id": f"drq-source-{ordinal}",
                                     "actor_sha256": ("a" if ordinal == 0 else "b") * 64,
                                     "archive_sha256": None, "dataset_digest": None,
                                     "road": {"track_id": 1, "geometry_seed": road},
                                     "original_episode_id": str(index), "replay_episode_id": key,
                                     "sequence_id_start": key, "sequence_id_end": key})
        for ordinal, label in enumerate(("source0", "source1")):
            prefix = f"runs/synthetic-training/{label}/teacher"
            source_id = f"drq-source-{ordinal}"
            actor_hash = ("a" if ordinal == 0 else "b") * 64
            digest, archive_sha = self.archive(prefix + "/support-dataset.npz", source_id, actor_hash,
                                               self.roads, "synthetic-" + label, "teacher", 1)
            row = {"receipt_path": prefix + "/collection-result.json",
                   "archive_path": prefix + "/support-dataset.npz", "archive_sha256": archive_sha,
                   "dataset_digest": digest, "source_id": source_id, "source_actor_sha256": actor_hash,
                   "source_checkpoint_sha256": ("c" if ordinal == 0 else "d") * 64,
                   "stored_decisions": 12}
            receipt = {"status": "completed", "arm": "teacher", "study_id": "synthetic-" + label,
                       "source_id": source_id, "source_actor_sha256": actor_hash,
                       "archive_sha256": archive_sha, "dataset_digest": digest, "stored_decisions": 12,
                       "allowed_cells": self.cells(self.roads), "complete_episode_count": 12,
                       "schedule_exhausted": True, "partial_decisions": 0, "unresolved_decision_calls": 0,
                       "episode_rows": [{"attempt": i, "episode_id": i, "status": "complete", "complete": True,
                                         "track_id": 1, "geometry_seed": road, "decisions": 1,
                                         "terminal": i % 2 == 0, "finished": False,
                                         "terminated": i % 2 == 0, "truncated": i % 2 == 1}
                                        for i, road in enumerate(self.roads)]}
            row["receipt_sha256"] = self.put_json(row["receipt_path"], receipt)
            self.train_receipts[label] = receipt
            self.train["datasets"][label] = row
            for entry in self.lineage[ordinal::2]:
                entry["archive_sha256"] = archive_sha
                entry["dataset_digest"] = digest
        self.train_sha = self.put_json(score.TRAIN_PATH, self.train)
        self.patch_value("TRAIN_SHA256", self.train_sha)
        policy = {"policy": "uniform-native-random-v1", "collector_sha256": self.collector_sha,
                  "rng_seed": 7395, "native_bounds": [-1.0, 1.0], "action_dim": 3}
        self.random_sha = hashlib.sha256(json.dumps(policy, sort_keys=True,
                                                    separators=(",", ":")).encode()).hexdigest()
        self.dev = {"format": "haic-dreamerv3-reused-train-diagnostic-v1",
                    "study_id": "dreamerv3-reused-train-multisource-development-v1",
                    "purpose": self.train["purpose"],
                    "freshness_claim": "reused r6 TRAIN cells; not fresh P1/P1b, evaluation, or promotion",
                    "r6_protocol": self.train["r6_protocol"], "cells": self.cells(self.dev_roads),
                    "episode_schedule": [0, 1, 2, 3], "frame_skip": 4, "max_steps": 2000,
                    "budgets": {"random_decision_cap": 8000, "teacher_decision_cap": 8000},
                    "source_actor": {"source_id": "drq-source-0", "actor_sha256": "a" * 64,
                                     "checkpoint_sha256": "c" * 64},
                    "random": {"rng_seed": 7395, "source_id": "uniform-native-random-seed-7395",
                               "source_actor_sha256": self.random_sha},
                    "source_sha256": {collector: self.collector_sha,
                                      "common_adapter.py": self.sources["common_adapter.py"],
                                      "haic/algorithms/drq_v2/teacher_replay.py": self.sources[
                                          "haic/algorithms/drq_v2/teacher_replay.py"]}}
        self.dev_sha = self.put_json(score.DEV_PATH, self.dev)
        self.patch_value("DEV_SHA256", self.dev_sha)
        self.development = {}
        for label, arm in (("source0", "teacher"), ("random", "random")):
            prefix = f"{score.DEV_ROOT}/{arm}"
            source_id = self.dev["source_actor"]["source_id"] if label == "source0" else self.dev["random"]["source_id"]
            actor_hash = "a" * 64 if label == "source0" else self.random_sha
            digest, archive_sha = self.archive(prefix + "/support-dataset.npz", source_id, actor_hash,
                                               self.dev_roads, self.dev["study_id"], arm, 41)
            row = {"receipt_path": prefix + "/collection-result.json",
                   "archive_path": prefix + "/support-dataset.npz", "archive_sha256": archive_sha,
                   "dataset_digest": digest, "source_id": source_id, "source_actor_sha256": actor_hash}
            receipt = {"format": "haic-dreamerv3-reused-train-collection-result-v1",
                       "status": "completed", "study_id": self.dev["study_id"], "purpose": self.dev["purpose"],
                       "protocol_sha256": self.dev_sha, "r6_protocol_sha256": self.r6_sha, "arm": arm,
                       "source_id": source_id, "source_actor_sha256": actor_hash,
                       "source_checkpoint_sha256": "c" * 64 if label == "source0" else None,
                       "random_rng_seed": 7395 if label == "random" else None,
                       "archive_sha256": archive_sha, "dataset_digest": digest,
                       "dataset_path": "support-dataset.npz", "allowed_cells": self.cells(self.dev_roads),
                       "excluded_seeds": [999], "decision_cap": 8000, "complete_episode_count": 4,
                       "schedule_attempts": 4, "schedule_exhausted": True, "partial_decisions": 0,
                       "unresolved_decision_calls": 0, "stored_decisions": 164, "decisions_spent": 164,
                       "distinct_finished_geometries": [],
                       "episode_rows": [{"attempt": i, "episode_id": i, "complete": True, "status": "complete",
                                         "track_id": 1, "geometry_seed": road, "decisions": 41,
                                         "terminal": True, "finished": False,
                                         "terminated": True, "truncated": False}
                                        for i, road in enumerate(self.dev_roads)]}
            row["receipt_sha256"] = self.put_json(row["receipt_path"], receipt)
            self.development[label] = row
        self.training = self.checkpoints()
        self.protocol = {"format": score.FORMAT, "purpose": score.PURPOSE, "study_id": score.STUDY_ID,
                         "training_protocol": {"path": score.TRAIN_PATH, "sha256": self.train_sha},
                         "development_collection_protocol": {"path": score.DEV_PATH, "sha256": self.dev_sha},
                         "training": self.training, "development": self.development,
                         "source_sha256": {key: self.sources[key] for key in score.SOURCE_PATHS},
                         "scoring": {"context_decisions": 8, "window_decisions": 32,
                                     "latent_seed": 120927, "context_mode": "reset-origin-full-prefix-terminal",
                                     "baselines": score.original.BASELINES, "min_complete_roads_per_stratum": 4},
                         "resources": {"max_archive_bytes": 4 * 1024**2, "max_uncompressed_bytes": 16 * 1024**2,
                                       "max_decisions_per_archive": 1024, "max_checkpoint_bytes": 64 * 1024**2,
                                       "max_cgroup_memory_bytes": 32 * 1024**3,
                                       "min_cgroup_available_bytes": 8 * 1024**3,
                                       "min_disk_available_bytes": 2 * 1024**3}}
        self.freeze()
        cgroup = {"limit_bytes": 32 * 1024**3, "available_bytes": 16 * 1024**3}
        cg = mock.patch.object(score.original, "_cgroup", return_value=cgroup)
        disk = mock.patch.object(score.os, "statvfs", return_value=SimpleNamespace(f_bavail=16, f_frsize=1024**3))
        cg.start()
        disk.start()
        self.addCleanup(cg.stop)
        self.addCleanup(disk.stop)

    def cells(self, roads):
        return [{"track_id": 1, "geometry_seed": road} for road in roads]

    def patch_value(self, name, value):
        patcher = mock.patch.object(score, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def put_bytes(self, name, contents):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return _digest(path)

    def put_json(self, name, payload):
        return self.put_bytes(name, (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode())

    def archive(self, name, source_id, actor_hash, roads, study_id, arm, length):
        episodes = []
        for i, road in enumerate(roads):
            frames = np.broadcast_to(np.arange(length + 1, dtype=np.uint8)[:, None, None],
                                     (length + 1, 84, 84)).copy()
            actions = np.zeros((length, 3), dtype=np.float32)
            applied = np.stack([ActionAdapter().to_official(action) for action in actions])
            terminated = np.zeros(length, dtype=bool)
            terminated[-1] = length != 1 or i % 2 == 0
            truncated = np.zeros(length, dtype=bool)
            truncated[-1] = not terminated[-1]
            episodes.append(TeacherEpisode(
                frames=frames, actions=actions, applied_actions=applied,
                rewards=np.full(length, 0.5 if length == 1 else 0.25, dtype=np.float32),
                terminated=terminated, truncated=truncated, terminal=terminated.copy(),
                finished=np.zeros(length, dtype=bool), episode_id=i, source_id=source_id,
                source_actor_sha256=actor_hash, geometry_id=str(road), track_id=1,
                metadata={"study_id": study_id, "arm": arm, "attempt": i}))
        dataset = TeacherDataset(episodes)
        digest = dataset.seal()
        archive_sha = self.put_bytes(name, dataset.to_bytes())
        return digest, archive_sha

    def checkpoints(self):
        rows = []
        lineage_shas = []
        for seed in (0, 1):
            prefix = f"{score.TRAIN_ROOT}/learner-{seed}"
            lineage_name = prefix + "/lineage.json"
            lineage_sha = self.put_json(lineage_name, {"format": "haic-dreamerv3-reused-train-multisource-v1-lineage",
                                                       "range_convention": "inclusive", "transition_count": 24,
                                                       "episodes": self.lineage})
            lineage_shas.append(lineage_sha)
            metadata = {"purpose": self.train["purpose"], "study_id": self.train["study_id"],
                        "seed": seed, "protocol_sha256": self.train_sha, "lineage_path": lineage_name,
                        "lineage_sha256": lineage_sha, "sources": self.train["datasets"],
                        "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
                        "promotion_eligible": False}
            agent = DreamerV3Agent(DreamerV3Config(**self.config), seed=seed)
            blank = np.zeros((4, 84, 84), dtype=np.uint8)
            for i in range(24):
                terminal = i // 2 % 2 == 0
                agent.replay.add(Transition(observation=blank, action=np.zeros(3, dtype=np.float32),
                                            applied_action=np.asarray([0, 0.5, 0.5]), reward=0.5,
                                            next_observation=blank, terminated=terminal, truncated=not terminal,
                                            terminal=terminal, episode_id=i, step=0))
            agent.gradient_steps = 256  # A synthetic checkpoint marker, not a learner update.
            checkpoint_name = prefix + "/world-model-checkpoint.pt"
            agent.save_checkpoint(self.root / checkpoint_name, run_metadata=metadata)
            checkpoint_sha = _digest(self.root / checkpoint_name)
            result = {"format": "haic-dreamerv3-reused-train-multisource-v1-result", "status": "complete",
                      "seed": seed, "purpose": self.train["purpose"], "study_id": self.train["study_id"],
                      "protocol_sha256": self.train_sha, "sources": self.train["datasets"],
                      "checkpoint_path": "world-model-checkpoint.pt", "checkpoint_sha256": checkpoint_sha,
                      "lineage_path": lineage_name, "lineage_sha256": lineage_sha,
                      "model_only_updates": 256, "environment_steps": 0, "distinct_training_roads": 12,
                      "episodes": 24, "decisions": 24, "actor_critic_target_optimizers_unchanged": True,
                      "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
                      "promotion_eligible": False}
            result_name = prefix + "/training-result.json"
            rows.append({"seed": seed, "result_path": result_name,
                         "result_sha256": self.put_json(result_name, result),
                         "checkpoint_path": checkpoint_name, "checkpoint_sha256": checkpoint_sha,
                         "lineage_path": lineage_name, "lineage_sha256": lineage_sha})
        self.patch_value("RESULT_SHAS", tuple(row["result_sha256"] for row in rows))
        self.patch_value("CHECKPOINT_SHAS", tuple(row["checkpoint_sha256"] for row in rows))
        self.patch_value("LINEAGE_SHA256", lineage_shas[0])
        return rows

    def freeze(self):
        self.protocol_sha = self.put_json(score.PROTOCOL_PATH, self.protocol)
        return self.protocol_sha

    def preflight(self):
        return score.preflight(score.PROTOCOL_PATH, self.freeze(), score.OUTPUT, repo_root=self.root)

    def score(self):
        return score.score(score.PROTOCOL_PATH, self.freeze(), score.OUTPUT, repo_root=self.root)


class MixedScoreContracts(SyntheticFixture):
    def test_sha_pinned_preflight_and_seeded_invariant(self):
        checked = self.preflight()
        self.assertEqual(len(checked["lineage"]), 24)
        self.assertEqual(checked["lineage"][-1]["sequence_id_end"], 23)
        model = checked["training"][0]
        modules = score._checkpoint(model["checkpoint"], config=self.config, seed=0,
                                    metadata=model["metadata"], lineage=checked["lineage"])
        self.assertEqual(len(modules), 5)
        self.assertFalse(self.output.exists())

    def test_both_seeds_both_action_sources_gate_and_denominators(self):
        with mock.patch.object(score, "_checkpoint", return_value=_modules()):
            report = self.score()
        self.assertTrue(report["gate"]["passed"])
        self.assertFalse(report["promotion_eligible"])
        self.assertEqual(report["training_baselines"]["training_decisions"], 24)
        self.assertEqual(report["training_baselines"]["terminal_prevalence"], 0.5)
        self.assertEqual(set(report["strata"]), {"source0", "random"})
        for stratum in report["strata"].values():
            self.assertEqual((stratum["collected_complete_episodes"], stratum["collected_independent_roads"]), (4, 4))
            self.assertEqual(len(stratum["models"]), 2)
            for model in stratum["models"]:
                self.assertEqual(len(model["roads"]), 4)
                self.assertEqual((model["aggregate"]["window_count"], model["aggregate"]["window_label_uses"],
                                  model["aggregate"]["duplicate_label_uses"]), (8, 256, 124))
                self.assertEqual(model["aggregate"]["terminal_positive_label_uses"], 4)
                self.assertEqual(model["episodes"][0]["windows"][1]["reset_origin_prefix_decisions"], 9)
                self.assertTrue(model["image_gate_passed"])
        self.assertTrue((self.output / "score-result.json").is_file())

    def test_strict_image_gate_failure(self):
        with mock.patch.object(score, "_checkpoint", return_value=_modules(0)):
            report = self.score()
        self.assertFalse(report["gate"]["passed"])
        self.assertEqual(report["status"], "engineering_proxy_only")

    def test_four_complete_but_short_episodes_cannot_pass(self):
        row = self.development["random"]
        digest, archive_sha = self.archive(row["archive_path"], row["source_id"],
                                           row["source_actor_sha256"], self.dev_roads,
                                           self.dev["study_id"], "random", 40)
        receipt = json.loads((self.root / row["receipt_path"]).read_text())
        receipt.update(dataset_digest=digest, archive_sha256=archive_sha,
                       stored_decisions=160, decisions_spent=160)
        for episode in receipt["episode_rows"]:
            episode["decisions"] = 40
        row.update(dataset_digest=digest, archive_sha256=archive_sha)
        row["receipt_sha256"] = self.put_json(row["receipt_path"], receipt)
        with mock.patch.object(score, "_checkpoint", return_value=_modules()):
            report = self.score()
        self.assertEqual(report["status"], "coverage_inconclusive")
        self.assertFalse(report["gate"]["passed"])
        self.assertEqual(report["strata"]["random"]["collected_independent_roads"], 4)
        self.assertEqual(report["strata"]["random"]["models"][0]["aggregate"]["window_count"], 0)

    def test_lineage_forgery_with_recomputed_sha_rejected_before_zip(self):
        for seed in (0, 1):
            path = self.root / self.training[seed]["lineage_path"]
            payload = json.loads(path.read_text())
            payload["episodes"][1]["source_id"] = payload["episodes"][0]["source_id"]
            self.training[seed]["lineage_sha256"] = self.put_json(self.training[seed]["lineage_path"], payload)
        score.LINEAGE_SHA256 = self.training[0]["lineage_sha256"]
        for seed in (0, 1):
            result_name = self.training[seed]["result_path"]
            result = json.loads((self.root / result_name).read_text())
            result["lineage_sha256"] = score.LINEAGE_SHA256
            self.training[seed]["result_sha256"] = self.put_json(result_name, result)
        score.RESULT_SHAS = tuple(row["result_sha256"] for row in self.training)
        with mock.patch.object(score.original, "_load_dataset", side_effect=AssertionError("ZIP opened")):
            with self.assertRaisesRegex(ValueError, "lineage source/road/episode"):
                self.preflight()
        self.assertFalse(self.output.exists())

    def test_wrong_dev_cells_and_source_hash_rejected_before_zip(self):
        self.dev["cells"][0]["geometry_seed"] = self.roads[0]
        self.protocol["development_collection_protocol"]["sha256"] = self.put_json(score.DEV_PATH, self.dev)
        score.DEV_SHA256 = self.protocol["development_collection_protocol"]["sha256"]
        with mock.patch.object(score.original, "_load_dataset", side_effect=AssertionError("ZIP opened")):
            with self.assertRaisesRegex(ValueError, "training-excluded"):
                self.preflight()
        self.dev["cells"][0]["geometry_seed"] = self.dev_roads[0]
        self.protocol["development_collection_protocol"]["sha256"] = self.put_json(score.DEV_PATH, self.dev)
        score.DEV_SHA256 = self.protocol["development_collection_protocol"]["sha256"]
        self.development["random"]["source_actor_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "forged development source"):
            self.preflight()

    def test_rejects_zip_size_low_headroom_and_output_collision_before_read(self):
        with mock.patch.object(score.original, "_dataset_bytes", side_effect=AssertionError("ZIP opened")):
            self.protocol["resources"]["max_archive_bytes"] = 1
            with self.assertRaisesRegex(ValueError, "oversized input"):
                self.preflight()
            self.protocol["resources"]["max_archive_bytes"] = 4 * 1024**2
            with mock.patch.object(score.original, "_cgroup", return_value={
                    "limit_bytes": 32 * 1024**3, "available_bytes": 1}):
                with self.assertRaisesRegex(ValueError, "cgroup lacks"):
                    self.preflight()
            with mock.patch.object(score.os, "statvfs", return_value=SimpleNamespace(f_bavail=1, f_frsize=1)):
                with self.assertRaisesRegex(ValueError, "disk lacks"):
                    self.preflight()
            self.output.mkdir()
            with self.assertRaises(FileExistsError):
                self.preflight()

    def test_seeded_actor_mutation_and_checkpoint_replay_loss_rejected(self):
        checked = self.preflight()
        model = checked["training"][0]
        path = model["checkpoint"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["actor"][next(iter(payload["actor"]))].add_(1)
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "actor/critic seeded initialization"):
            score._checkpoint(path, config=self.config, seed=0, metadata=model["metadata"],
                              lineage=checked["lineage"])
        payload["replay"]["sequence_ids"][3] = 0
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "lost or overwrote lineage"):
            score._checkpoint(path, config=self.config, seed=0, metadata=model["metadata"],
                              lineage=checked["lineage"])

    def test_forged_checkpoint_metadata_rejected_in_preflight_before_zip(self):
        row = self.training[0]
        checkpoint = self.root / row["checkpoint_path"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload["run_metadata"]["promotion_eligible"] = True
        torch.save(payload, checkpoint)
        row["checkpoint_sha256"] = _digest(checkpoint)
        result = json.loads((self.root / row["result_path"]).read_text())
        result["checkpoint_sha256"] = row["checkpoint_sha256"]
        row["result_sha256"] = self.put_json(row["result_path"], result)
        score.CHECKPOINT_SHAS = tuple(item["checkpoint_sha256"] for item in self.training)
        score.RESULT_SHAS = tuple(item["result_sha256"] for item in self.training)
        with mock.patch.object(score.original, "_load_dataset", side_effect=AssertionError("ZIP opened")):
            with self.assertRaisesRegex(ValueError, "checkpoint metadata/config/counters"):
                self.preflight()
        self.assertFalse(self.output.exists())

    def test_archive_sha_and_zip_bomb_guard(self):
        checked = self.preflight()
        row = checked["development"]["random"]["row"]
        archive = checked["development"]["random"]["archive"]
        resources = checked["protocol"]["resources"].copy()
        resources["max_uncompressed_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "ZIP entries"):
            score._dataset(archive, row, resources, self.dev_roads, self.dev["study_id"], "random",
                           checked["development"]["random"]["receipt"])
        with archive.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "archive size or SHA-256"):
            score._dataset(archive, row, checked["protocol"]["resources"], self.dev_roads,
                           self.dev["study_id"], "random", checked["development"]["random"]["receipt"])


if __name__ == "__main__":
    unittest.main()
