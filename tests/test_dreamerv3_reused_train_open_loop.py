"""Synthetic, simulator-free contracts for the sealed reused-TRAIN scorer."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from dreamer_v3 import (CategoricalRSSM, ContinueHead, ConvDecoder, ConvEncoder,
                        DreamerV3Config, RewardHead)
from haic.algorithms.drq_v2.teacher_replay import TeacherDataset, TeacherEpisode
from scripts.diagnose import dreamerv3_reused_train_open_loop as scorer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Encoder:
    def __call__(self, stacks):
        return torch.zeros((len(stacks), 2))


class _RSSM:
    hidden_dim = stoch_dim = 1

    def __init__(self):
        self.prefixes = []

    def observe_sequence(self, embeds, actions, first):
        self.prefixes.append((actions.shape[1], first.tolist()))
        return torch.zeros((1, embeds.shape[1], 2)), None, None

    def step_prior(self, h, z, action):
        return h, z, None, None


class _Decoder:
    def __call__(self, state):
        return torch.zeros((1, 1, 84, 84))


class _Reward:
    def pred(self, state):
        return torch.ones((1,))


class _Continue:
    def __call__(self, state):
        return torch.zeros((1, 1))


def _episode(*, length: int, road: int, arm: str, study_id: str, attempt: int,
             source_id: str | None = None, actor_hash: str | None = None) -> TeacherEpisode:
    values = np.arange(length + 1, dtype=np.uint8).reshape(-1, 1, 1)
    frames = np.broadcast_to(values, (length + 1, 84, 84)).copy()
    actions = np.zeros((length, 3), dtype=np.float32)
    applied = np.full((length, 3), 0.5, dtype=np.float32)
    applied[:, 0] = 0
    terminated = np.zeros(length, dtype=bool)
    terminated[-1] = True
    return TeacherEpisode(
        frames=frames, actions=actions, applied_actions=applied,
        rewards=np.full(length, 0.25 if study_id.endswith("diagnostic-v1") else 0.5, dtype=np.float32),
        terminated=terminated, truncated=np.zeros(length, dtype=bool),
        finished=np.zeros(length, dtype=bool), terminal=terminated.copy(),
        episode_id=attempt, source_id=source_id or f"{arm}-source",
        source_actor_sha256=actor_hash or ("a" if arm == "random" else "b") * 64,
        geometry_id=str(road), track_id=1,
        metadata={"study_id": study_id, "arm": arm, "attempt": attempt},
    )


class ScorerFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "runs" / "diagnostic-score" / "new-result"
        self.output.parent.mkdir(parents=True)
        self.train_id = "dreamerv3-reused-train-diagnostic-v1"
        self.dev_id = scorer.DEVELOPMENT_ID
        self.train_roads = (3910800001, 3910800004, 3910800034, 3910800085)
        self.development_roads = scorer.DEVELOPMENT_ROADS
        self.actor = {"source_id": "teacher-source", "actor_sha256": "b" * 64}
        self.config = asdict(DreamerV3Config(device="cpu", embed_dim=8, hidden_dim=8,
                                            num_categoricals=2, num_classes=2, twohot_bins=3,
                                            replay_capacity=256))
        self._source_files()
        self._protocols_and_artifacts()
        self.offline_sha_patch = mock.patch.object(scorer, "OFFLINE_SHA256", self.offline_sha)
        self.r6_sha_patch = mock.patch.object(scorer, "R6_SHA256", self.r6_sha)
        self.offline_sha_patch.start()
        self.r6_sha_patch.start()
        self.addCleanup(self.offline_sha_patch.stop)
        self.addCleanup(self.r6_sha_patch.stop)

    def _write(self, relative: str, contents: bytes) -> str:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return _sha(path)

    def _json(self, relative: str, payload: dict) -> str:
        return self._write(relative, (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode())

    def _source_files(self):
        self.sources = {}
        self.collector_sha = self._write(scorer.COLLECTOR, b"# frozen synthetic collector\n")
        for name in scorer.SOURCE_PATHS:
            if name == "scripts/diagnose/dreamerv3_reused_train_open_loop.py":
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(scorer.__file__, path)
                self.sources[name] = _sha(path)
            else:
                self.sources[name] = self._write(name, b"# synthetic executable source\n")

    @staticmethod
    def _cells(roads):
        return [{"track_id": 1, "geometry_seed": road} for road in roads]

    def _archive(self, relative: str, *, arm: str, study_id: str, roads: tuple[int, ...], length: int):
        original = study_id == self.train_id
        source_id = (f"uniform-native-random-seed-{7391 if original else 7392}"
                     if arm == "random" else "teacher-source")
        source_hash = (self.training_random_hash if original else self.dev_random_hash) if arm == "random" else "b" * 64
        episodes = [_episode(length=length, road=road, arm=arm, study_id=study_id, attempt=index,
                             source_id=source_id, actor_hash=source_hash)
                    for index, road in enumerate(roads)]
        dataset = TeacherDataset(episodes)
        dataset.seal()
        digest = dataset.digest
        archive_sha = self._write(relative, dataset.to_bytes())
        return digest, archive_sha

    def _protocols_and_artifacts(self):
        def random_hash(seed):
            identity = {"policy": "uniform-native-random-v1", "collector_sha256": self.collector_sha,
                        "rng_seed": seed, "native_bounds": [-1.0, 1.0], "action_dim": 3}
            return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

        self.training_random_hash = random_hash(7391)
        self.dev_random_hash = random_hash(7392)
        r6 = {"training_pool": {"partition": "TRAIN", "track_ids": [1, 2, 3, 4],
                                "geometry_seeds": list(self.train_roads + self.development_roads)},
              "diagnostic_pool": {"partition": "TRAIN-DIAGNOSTIC", "geometry_seeds": [3910800123]}}
        self.r6_sha = self._json("experiments/drqv2-geometry-mix-v1-r6.json", r6)
        r6_ref = {"path": "experiments/drqv2-geometry-mix-v1-r6.json", "sha256": self.r6_sha}
        training_collection = {"r6_protocol": r6_ref, "cells": self._cells(self.train_roads),
                               "source_actor": self.actor, "source_sha256": {scorer.COLLECTOR: self.collector_sha},
                               "random": {"rng_seed": 7391, "source_id": "uniform-native-random-seed-7391",
                                          "source_actor_sha256": self.training_random_hash}}
        collection_name = "experiments/dreamerv3-reused-train-diagnostic-v1.json"
        collection_sha = self._json(collection_name, training_collection)
        self.dev_name = "experiments/dreamerv3-reused-train-development-v1.json"
        self.dev_protocol = {
            "format": "haic-dreamerv3-reused-train-diagnostic-v1",
            "purpose": "reused-TRAIN-engineering-diagnostic", "study_id": self.dev_id,
            "freshness_claim": "reused r6 TRAIN cells; not fresh P1/P1b, evaluation, or promotion",
            "r6_protocol": r6_ref, "cells": self._cells(self.development_roads),
            "episode_schedule": [0, 1, 2, 3], "frame_skip": 4, "max_steps": 2000,
            "budgets": {"random_decision_cap": 8000, "teacher_decision_cap": 8000},
            "random": {"rng_seed": 7392, "source_id": "uniform-native-random-seed-7392",
                       "source_actor_sha256": self.dev_random_hash},
            "source_actor": self.actor, "source_sha256": training_collection["source_sha256"],
        }
        dev_sha = self._json(self.dev_name, self.dev_protocol)
        self.offline = {
            "format": "haic-dreamerv3-reused-train-offline-v1",
            "purpose": "reused-TRAIN-engineering-diagnostic", "study_id": self.train_id,
            "collection_protocol": {"path": collection_name, "sha256": collection_sha},
            "learner": {"seeds": [0, 1], "updates": 64, "config": self.config},
            "datasets": {}, "source_sha256": {name: self.sources[name] for name in self.sources if name !=
                                              "scripts/diagnose/dreamerv3_reused_train_open_loop.py"},
        }
        self.training = {}
        self.development = {}
        for arm in ("random", "teacher"):
            train_prefix = f"runs/20260926-dreamerv3-reused-train-diagnostic-v1/collection/{arm}"
            archive_path = f"{train_prefix}/support-dataset.npz"
            digest, archive_sha = self._archive(archive_path, arm=arm, study_id=self.train_id,
                                                roads=self.train_roads, length=5)
            train_receipt_path = f"{train_prefix}/collection-result.json"
            receipt_sha = self._json(train_receipt_path, {"archive_sha256": archive_sha})
            self.offline["datasets"][arm] = {
                "archive_path": archive_path, "archive_sha256": archive_sha, "dataset_digest": digest,
                "source_id": ("uniform-native-random-seed-7391" if arm == "random" else "teacher-source"),
                "source_actor_sha256": self.training_random_hash if arm == "random" else "b" * 64,
                "allowed_cells": self._cells(self.train_roads), "stored_decisions": 20,
                "receipt_path": train_receipt_path, "receipt_sha256": receipt_sha,
            }
            prefix = f"runs/20260926-dreamerv3-reused-train-diagnostic-v1/development/{arm}"
            archive_path = f"{prefix}/support-dataset.npz"
            digest, archive_sha = self._archive(archive_path, arm=arm, study_id=self.dev_id,
                                                roads=self.development_roads, length=41)
            receipt_path = f"{prefix}/collection-result.json"
            receipt = {
                "format": "haic-dreamerv3-reused-train-collection-result-v1", "status": "completed",
                "purpose": self.dev_protocol["purpose"], "study_id": self.dev_id, "protocol_sha256": dev_sha,
                "r6_protocol_sha256": self.r6_sha, "arm": arm,
                "source_id": ("uniform-native-random-seed-7392" if arm == "random" else "teacher-source"),
                "source_actor_sha256": self.dev_random_hash if arm == "random" else "b" * 64,
                "dataset_path": "support-dataset.npz", "archive_sha256": archive_sha, "dataset_digest": digest,
                "decision_cap": 8000, "allowed_cells": self._cells(self.development_roads),
                "excluded_seeds": [3910800123], "stored_decisions": 164,
                "complete_episode_count": 4, "unresolved_decision_calls": 0,
                "episode_rows": [{"status": "complete", "attempt": index, "episode_id": index,
                                  "track_id": 1, "geometry_seed": road, "decisions": 41,
                                  "terminal": True, "finished": False}
                                 for index, road in enumerate(self.development_roads)],
            }
            receipt_sha = self._json(receipt_path, receipt)
            self.development[arm] = {"receipt_path": receipt_path, "receipt_sha256": receipt_sha,
                                     "archive_path": archive_path, "archive_sha256": archive_sha,
                                     "dataset_digest": digest,
                                     "source_id": ("uniform-native-random-seed-7392" if arm == "random" else "teacher-source"),
                                     "source_actor_sha256": self.dev_random_hash if arm == "random" else "b" * 64}
        self.offline_sha = self._json(scorer.OFFLINE_PATH, self.offline)
        self._checkpoints()
        self.protocol = {
            "format": scorer.FORMAT, "purpose": scorer.PURPOSE, "study_id": self.dev_id,
            "offline_protocol": {"path": scorer.OFFLINE_PATH, "sha256": self.offline_sha},
            "development_collection_protocol": {"path": self.dev_name, "sha256": dev_sha},
            "training": self.training, "development": self.development, "source_sha256": self.sources,
            "scoring": {"context_decisions": 8, "window_decisions": 32,
                        "min_terminal_episodes_per_stratum": 4, "min_geometries_per_stratum": 4,
                        "latent_seed": 123, "baselines": scorer.BASELINES,
                        "context_mode": "reset-origin-full-prefix-terminal"},
            "resources": {"max_archive_bytes": 4 * 1024**2, "max_uncompressed_bytes": 32 * 1024**2,
                          "max_decisions_per_archive": 1024, "max_checkpoint_bytes": 64 * 1024**2,
                          "max_cgroup_memory_bytes": 128 * 1024**3,
                          "min_cgroup_available_bytes": 16 * 1024**2},
        }
        self.protocol_path = self.root / "experiments/dreamerv3-reused-train-score-v1.json"
        self.freeze()

    def _checkpoints(self):
        torch.manual_seed(0)
        encoder = ConvEncoder(in_channels=4, embed_dim=self.config["embed_dim"])
        rssm = CategoricalRSSM(action_dim=3, embed_dim=self.config["embed_dim"],
                               hidden_dim=self.config["hidden_dim"], num_categoricals=self.config["num_categoricals"],
                               num_classes=self.config["num_classes"], unimix=self.config["unimix"])
        decoder = ConvDecoder(in_features=rssm.state_dim, out_channels=1)
        reward = RewardHead(in_features=rssm.state_dim, bins=self.config["twohot_bins"])
        continuation = ContinueHead(in_features=rssm.state_dim)
        for arm in ("random", "teacher"):
            self.training[arm] = []
            for seed in (0, 1):
                prefix = f"{scorer.TRAIN_ROOT}/{arm}-seed-{seed}"
                checkpoint_path = f"{prefix}/world-model-checkpoint.pt"
                metadata = {"purpose": self.offline["purpose"], "study_id": self.train_id, "arm": arm,
                            "seed": seed, "offline_protocol_sha256": self.offline_sha,
                            "dataset_digest": self.offline["datasets"][arm]["dataset_digest"],
                            "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
                            "promotion_eligible": False}
                checkpoint = {"format": "haic-dreamerv3-checkpoint-v3", "run_metadata": metadata,
                              "environment_steps": 0, "gradient_steps": 64, "config": self.config,
                              "encoder": encoder.state_dict(), "rssm": rssm.state_dict(),
                              "decoder": decoder.state_dict(), "reward_head": reward.state_dict(),
                              "continue_head": continuation.state_dict()}
                checkpoint_file = self.root / checkpoint_path
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, checkpoint_file)
                checkpoint_sha = _sha(checkpoint_file)
                offline = self.offline["datasets"][arm]
                result = {"format": "haic-dreamerv3-reused-train-offline-result-v1", "status": "complete",
                          "purpose": self.offline["purpose"], "study_id": self.train_id, "arm": arm, "seed": seed,
                          "offline_protocol_sha256": self.offline_sha,
                          "collection_protocol_sha256": self.offline["collection_protocol"]["sha256"],
                          "checkpoint_path": "world-model-checkpoint.pt", "checkpoint_sha256": checkpoint_sha,
                          "collection_receipt_path": offline["receipt_path"],
                          "collection_receipt_sha256": offline["receipt_sha256"],
                          "model_only_updates": 64, "environment_steps": 0, "actor_trained": False,
                          "actor_critic_target_unchanged": True, "fresh_claim": False, "p1b_claim": False,
                          "promotion_eligible": False, **{name: offline[name] for name in (
                              "archive_path", "archive_sha256", "dataset_digest", "source_id", "source_actor_sha256")}}
                result_path = f"{prefix}/training-result.json"
                result_sha = self._json(result_path, result)
                self.training[arm].append({"seed": seed, "result_path": result_path,
                                            "result_sha256": result_sha, "checkpoint_path": checkpoint_path,
                                            "checkpoint_sha256": checkpoint_sha})

    def freeze(self):
        self.protocol_sha = self._json("experiments/dreamerv3-reused-train-score-v1.json", self.protocol)

    def preflight(self):
        return scorer.preflight(self.protocol_path, self.protocol_sha, self.output, repo_root=self.root)


class ProvenanceTests(ScorerFixture):
    def test_sealed_preflight_and_read_only_checkpoint_metadata(self):
        checked = self.preflight()
        self.assertEqual(checked["development_cells"], tuple((1, road) for road in self.development_roads))
        checkpoint = checked["training"]["teacher"][1]["checkpoint"]
        modules = scorer._checkpoint_modules(checkpoint, arm="teacher", seed=1,
                                             dataset_digest=self.offline["datasets"]["teacher"]["dataset_digest"],
                                             config=self.config)
        self.assertEqual(len(modules), 5)
        self.assertFalse(self.output.exists())

    def test_real_synthetic_world_model_scores_prior_not_actor(self):
        checked = self.preflight()
        modules = scorer._checkpoint_modules(checked["training"]["teacher"][0]["checkpoint"],
                                             arm="teacher", seed=0,
                                             dataset_digest=self.offline["datasets"]["teacher"]["dataset_digest"],
                                             config=self.config)
        episode = _episode(length=41, road=self.development_roads[0], arm="random",
                           study_id=self.dev_id, attempt=0)
        row = scorer._score_episode(episode, modules, 0.25, 0.2, 123)
        self.assertEqual((len(row["windows"]), row["duplicate_label_uses"]), (2, 31))
        self.assertTrue(all(np.isfinite(value) for value in row["metrics"].values()))
        self.assertEqual(row["unique_terminal_positive_labels"], 1)

    def test_forged_development_random_source_is_rejected(self):
        self.dev_protocol["random"]["source_actor_sha256"] = "c" * 64
        self.protocol["development_collection_protocol"]["sha256"] = self._json(self.dev_name, self.dev_protocol)
        self.freeze()
        with self.assertRaisesRegex(ValueError, "independently source-pinned"):
            self.preflight()

    def test_disjoint_reused_train_and_exact_pins(self):
        self.dev_protocol["cells"][0]["geometry_seed"] = self.train_roads[0]
        self.protocol["development_collection_protocol"]["sha256"] = self._json(self.dev_name, self.dev_protocol)
        self.freeze()
        with self.assertRaisesRegex(ValueError, "disjoint reused r6 TRAIN"):
            self.preflight()

    def test_rejects_forged_training_result_and_checkpoint_metadata(self):
        path = self.root / self.training["random"][0]["result_path"]
        result = json.loads(path.read_text())
        result["actor_trained"] = True
        self.training["random"][0]["result_sha256"] = self._json(self.training["random"][0]["result_path"], result)
        self.freeze()
        with self.assertRaisesRegex(ValueError, "training result actor_trained"):
            self.preflight()
        result["actor_trained"] = False
        self.training["random"][0]["result_sha256"] = self._json(self.training["random"][0]["result_path"], result)
        checkpoint_file = self.root / self.training["random"][0]["checkpoint_path"]
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        payload["run_metadata"]["seed"] = 1
        torch.save(payload, checkpoint_file)
        self.training["random"][0]["checkpoint_sha256"] = _sha(checkpoint_file)
        result["checkpoint_sha256"] = self.training["random"][0]["checkpoint_sha256"]
        self.training["random"][0]["result_sha256"] = self._json(self.training["random"][0]["result_path"], result)
        self.freeze()
        with self.assertRaisesRegex(ValueError, "checkpoint metadata"):
            self.preflight()

    def test_source_hash_output_collision_and_zip_bound(self):
        self.sources["dreamer_v3.py"] = "c" * 64
        self.freeze()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.preflight()
        self.sources["dreamer_v3.py"] = self.offline["source_sha256"]["dreamer_v3.py"]
        self.freeze()
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.preflight()
        archive = self.root / self.development["random"]["archive_path"]
        data = archive.read_bytes()
        with self.assertRaisesRegex(ValueError, "compressed size"):
            scorer._dataset_bytes(archive, hashlib.sha256(data).hexdigest(), len(data) - 1, 32 * 1024**2, 1024)
        with self.assertRaisesRegex(ValueError, "ZIP entries"):
            scorer._dataset_bytes(archive, hashlib.sha256(data).hexdigest(), len(data), 256, 1024)

    def test_coverage_and_overlapping_windows_without_model_ranking(self):
        models = (_Encoder(), _RSSM(), _Decoder(), _Reward(), _Continue())
        with mock.patch.object(scorer, "_checkpoint_modules", return_value=models):
            report = scorer.score(self.protocol_path, self.protocol_sha, self.output, repo_root=self.root)
        self.assertEqual(report["status"], "descriptive_only")
        self.assertFalse(report["promotion_eligible"])
        self.assertEqual(set(report["strata"]), {"random", "teacher"})
        for stratum in report["strata"].values():
            self.assertEqual(stratum["coverage"], "descriptive_sufficient_minimum")
            self.assertEqual(stratum["scored_independent_terminal_episodes"], 4)
            self.assertEqual(stratum["scored_independent_geometries"], 4)
            self.assertEqual([(model["model_arm"], model["learner_seed"]) for model in stratum["models"]],
                             [("random", 0), ("random", 1), ("teacher", 0), ("teacher", 1)])
            for model in stratum["models"]:
                totals = model["aggregate"]
                self.assertEqual((totals["window_count"], totals["window_label_uses"],
                                  totals["unique_scored_decisions"], totals["duplicate_label_uses"]),
                                 (8, 256, 132, 124))
                self.assertEqual((totals["terminal_positive_label_uses"], totals["unique_terminal_positive_labels"]),
                                 (4, 4))
                self.assertEqual(len(model["roads"]), 4)
                self.assertEqual(model["episodes"][0]["windows"][1]["reset_origin_prefix_decisions"], 9)
                self.assertEqual(model["training_baselines"]["constant_reward"], 0.25)
                self.assertAlmostEqual(model["training_baselines"]["terminal_prevalence"], 0.2)
        self.assertTrue((self.output / "score-result.json").is_file())
        with self.assertRaises(FileExistsError):
            self.preflight()

    def test_insufficient_episode_coverage_cannot_be_called_pass(self):
        receipt_path = self.development["random"]["receipt_path"]
        receipt = json.loads((self.root / receipt_path).read_text())
        receipt["episode_rows"] = receipt["episode_rows"][:3]
        receipt["complete_episode_count"] = 3
        receipt["stored_decisions"] = 123
        digest, archive_sha = self._archive(self.development["random"]["archive_path"], arm="random",
                                            study_id=self.dev_id, roads=self.development_roads[:3], length=41)
        receipt["dataset_digest"] = digest
        receipt["archive_sha256"] = archive_sha
        self.development["random"].update(dataset_digest=digest, archive_sha256=archive_sha)
        self.development["random"]["receipt_sha256"] = self._json(receipt_path, receipt)
        self.freeze()
        with mock.patch.object(scorer, "_checkpoint_modules",
                               return_value=(_Encoder(), _RSSM(), _Decoder(), _Reward(), _Continue())):
            result = scorer.score(self.protocol_path, self.protocol_sha, self.output, repo_root=self.root)
        self.assertEqual(result["status"], "coverage_inconclusive")
        self.assertEqual(len(result["strata"]["random"]["coverage_reasons"]), 2)


class WindowTests(unittest.TestCase):
    def test_short_episode_and_reset_origin_terminal_context(self):
        modules = (_Encoder(), _RSSM(), _Decoder(), _Reward(), _Continue())
        short = _episode(length=40, road=3910800002, arm="random",
                         study_id=scorer.DEVELOPMENT_ID, attempt=0)
        self.assertIsNone(scorer._score_episode(short, modules, 0.25, 0.2, 5)["metrics"])
        long = _episode(length=90, road=3910800002, arm="random",
                        study_id=scorer.DEVELOPMENT_ID, attempt=0)
        result = scorer._score_episode(long, modules, 0.25, 0.2, 5)
        self.assertEqual((result["unique_scored_decisions"], result["duplicate_label_uses"]), (64, 0))
        self.assertEqual(modules[1].prefixes[-1][0], 58)
        self.assertEqual(result["windows"][1]["reset_origin_prefix_decisions"], 58)
        self.assertAlmostEqual(result["metrics"]["reward_mse"], 0.25)
        self.assertAlmostEqual(result["metrics"]["training_constant_reward_mse"], 0.0625)
        self.assertAlmostEqual(result["metrics"]["image_mse"], result["metrics"]["shifted_repeat_mse"])


if __name__ == "__main__":
    unittest.main()
