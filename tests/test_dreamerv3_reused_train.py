"""Synthetic-only reused-TRAIN collector contracts; never opens the simulator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common_adapter import ActionAdapter, EpisodeCollector, ObservationSpec
from haic.algorithms.dreamer_v3.offline import replay_from_dataset_bytes
from haic.algorithms.drq_v2.teacher_replay import TeacherDataset
from scripts.diagnose import collect_dreamerv3_reused_train as reused


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.sources = {
            name: self.write(name, (Path(reused.__file__).read_bytes() if name == reused.COLLECTOR_PATH
                                    else f"synthetic executable {name}\n".encode()))
            for name in sorted(reused.REQUIRED_SOURCES)
        }
        actor_path = "runs/source-pad4/actor.pt"
        checkpoint_path = "runs/source-pad4/checkpoint.pt"
        self.source = {
            "source_id": "drq-source-0", "learner_seed": 0, "source_revision": "synthetic-revision",
            "checkpoint_path": checkpoint_path, "checkpoint_sha256": self.write(checkpoint_path, b"checkpoint"),
            "actor_path": actor_path, "actor_sha256": self.write(actor_path, b"actor"),
            "actor_weights_sha256": digest(b"weights"),
        }
        self.r6 = {
            "format": "haic-drq-geometry-mix-study-v1", "study_id": "drqv2-geometry-mix-v1-r6",
            "training_pool": {"partition": "TRAIN", "track_ids": [1, 2, 3, 4], "geometry_seeds": [101, 102]},
            "diagnostic_pool": {"partition": "TRAIN-DIAGNOSTIC", "geometry_seeds": [201]},
            "environment": {"partition": "TRAIN", "track_ids": [1, 2, 3, 4], "frame_skip": 4,
                            "max_steps": 2000, "reward_shaping": False, "obstacles": True},
            "learner": {"padding": 4, "source_environment_steps": 131072},
            "source_actors": [
                {"source_seed": 0, "weight_only_fork": True, "source_revision": self.source["source_revision"],
                 "source_checkpoint_path": checkpoint_path,
                 "source_checkpoint_sha256": self.source["checkpoint_sha256"],
                 "source_actor_path": actor_path, "source_actor_sha256": self.source["actor_sha256"],
                 "source_actor_weights_sha256": self.source["actor_weights_sha256"]},
                {"source_seed": 1},
            ],
        }
        self.r6_sha = self.write_json(reused.R6_PATH, self.r6)
        self.protocol = {
            "format": reused.FORMAT, "study_id": "dreamerv3-reused-train-synthetic",
            "purpose": reused.PURPOSE, "freshness_claim": reused.LIMITATION,
            "r6_protocol": {"path": reused.R6_PATH, "sha256": self.r6_sha},
            "source_sha256": self.sources,
            "cells": [{"track_id": 1, "geometry_seed": 101}, {"track_id": 2, "geometry_seed": 102}],
            "episode_schedule": [0, 1], "frame_skip": 4, "max_steps": 5,
            "budgets": {"random_decision_cap": 5, "teacher_decision_cap": 5},
            "random": {"rng_seed": 77, "source_id": "uniform-native-random-seed-77",
                       "source_actor_sha256": reused.random_source_hash(self.sources[reused.COLLECTOR_PATH], 77)},
            "source_actor": self.source,
            "spec_fingerprints": {"action": ActionAdapter().spec.fingerprint,
                                  "observation": ObservationSpec().fingerprint},
        }
        self.protocol_path = "experiments/dreamerv3-reused-train-synthetic.json"
        self.save()
        (root / "runs/synthetic-only").mkdir(parents=True)

    def write(self, relative: str, content: bytes) -> str:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return digest(content)

    def write_json(self, relative: str, payload: dict) -> str:
        return self.write(relative, (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode())

    def save(self) -> None:
        self.protocol_sha = self.write_json(self.protocol_path, self.protocol)

    def pair(self) -> dict:
        return {"learner_seed": 0, "source_revision": self.source["source_revision"],
                "checkpoint_sha256": self.source["checkpoint_sha256"],
                "actor_sha256": self.source["actor_sha256"],
                "actor_weights_sha256": self.source["actor_weights_sha256"],
                "action_fingerprint": self.protocol["spec_fingerprints"]["action"],
                "observation_fingerprint": self.protocol["spec_fingerprints"]["observation"],
                "cpu_smoke_observations": 3}


class FakeActor:
    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        assert tuple(observations.shape) == (1, 4, 84, 84)
        return torch.tensor([[2.0, -0.5, 0.25]], dtype=torch.float32)


class FakeEnv:
    def __init__(self, track: int, seed: int, *, length: int = 2, ending: str = "finished"):
        self.track, self.seed, self.length, self.ending = track, seed, length, ending
        self.steps = 0
        self.actions: list[np.ndarray] = []
        self.closed = False

    def observation(self, step: int) -> np.ndarray:
        frames = np.stack([np.full((84, 84), 10 + max(0, step - 3 + channel), dtype=np.uint8)
                           for channel in range(4)])
        return ObservationSpec().from_uint8(frames)

    def reset(self, *, seed=None, options=None):
        assert self.steps == 0
        return self.observation(0), {"track_id": self.track, "seed": self.seed}

    def step(self, official_action):
        self.actions.append(np.asarray(official_action).copy())
        self.steps += 1
        last = self.steps == self.length
        finished = last and self.ending == "finished"
        terminated = last and self.ending == "retired"
        truncated = last and self.ending in ("finished", "timeout")
        return self.observation(self.steps), float(self.steps), terminated, truncated, {
            "track_id": self.track, "seed": self.seed, "finished": finished,
            "progress": self.steps / self.length, "damage": 0.25,
            "retire_reason": "off_track" if terminated else None,
        }

    def close(self):
        self.closed = True


class ReusedTrainTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="dreamer-reused-synthetic-")
        self.addCleanup(temp.cleanup)
        self.fixture = Fixture(Path(temp.name))
        self.envs: list[FakeEnv] = []
        pinned_r6 = patch.object(reused, "R6_SHA256", self.fixture.r6_sha)
        pinned_r6.start()
        self.addCleanup(pinned_r6.stop)

    def builder(self, track, geometry_seed, max_steps, frame_skip, *, reward_shaping, obstacles, collision_penalty):
        self.assertEqual((max_steps, frame_skip, reward_shaping, obstacles, collision_penalty),
                         (5, 4, False, True, 0.0))
        env = FakeEnv(track, geometry_seed)
        self.envs.append(env)
        return env

    def mocked(self, *, builder=None, collector=EpisodeCollector, pair=None):
        for target, replacement in (
            ("audit_source_actor_pair", pair or self.fixture.pair()),
            ("load_exported_actor", (FakeActor(), ActionAdapter(), ObservationSpec())),
        ):
            context = patch.object(reused, target, return_value=replacement)
            context.start()
            self.addCleanup(context.stop)
        for target, replacement in (("build_env", builder or self.builder), ("EpisodeCollector", collector)):
            context = patch.object(reused, target, side_effect=replacement)
            context.start()
            self.addCleanup(context.stop)

    def output(self, name="collection") -> Path:
        return self.fixture.root / "runs/synthetic-only" / name

    def collect(self, name="collection") -> dict:
        return reused.collect(self.fixture.root / self.fixture.protocol_path, self.fixture.protocol_sha,
                              self.output(name), repo_root=self.fixture.root)

    def test_pinned_protocol_executable_r6_and_cpu_identity_precede_every_reset(self):
        self.mocked()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            reused.collect(self.fixture.root / self.fixture.protocol_path, "0" * 64,
                           self.output(), repo_root=self.fixture.root)
        self.fixture.write("train.py", b"changed training runtime")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.collect()
        self.fixture.write("train.py", b"synthetic executable train.py\n")
        (self.fixture.root / reused.R6_PATH).write_bytes(b"changed r6")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.collect()
        self.fixture.write_json(reused.R6_PATH, self.fixture.r6)
        (self.fixture.root / self.fixture.source["checkpoint_path"]).write_bytes(b"forged checkpoint")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.collect()
        self.fixture.write(self.fixture.source["checkpoint_path"], b"checkpoint")
        (self.fixture.root / self.fixture.source["actor_path"]).write_bytes(b"forged actor")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.collect()
        self.fixture.write(self.fixture.source["actor_path"], b"actor")
        with patch.object(reused, "audit_source_actor_pair", return_value={**self.fixture.pair(),
                                                                            "actor_weights_sha256": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "CPU source identity mismatch"):
                self.collect()
        self.assertEqual(self.envs, [])
        self.assertFalse(self.output().exists())

    def test_proposed_source_inventory_cannot_forge_executing_collector(self):
        self.mocked()
        changed = b"synthetic modified collector"
        self.fixture.sources[reused.COLLECTOR_PATH] = self.fixture.write(reused.COLLECTOR_PATH, changed)
        self.fixture.protocol["random"]["source_actor_sha256"] = reused.random_source_hash(
            self.fixture.sources[reused.COLLECTOR_PATH], 77,
        )
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "executing collector differs"):
            self.collect()
        self.assertEqual(self.envs, [])

    def test_reused_r6_train_accepts_only_exact_unique_cells_not_old_diagnostic_heldout(self):
        self.mocked()
        self.assertEqual(reused.preflight(self.fixture.root / self.fixture.protocol_path,
                                         self.fixture.protocol_sha, repo_root=self.fixture.root)["excluded_seeds"], [201])
        for seed in (201, 301, 99):
            self.fixture.protocol["cells"][0]["geometry_seed"] = seed
            self.fixture.save()
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "r6 training_pool"):
                self.collect()
        self.fixture.protocol["cells"][0]["geometry_seed"] = 101
        self.fixture.protocol["cells"][1]["geometry_seed"] = 101
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "unique cells"):
            self.collect()
        self.fixture.protocol["cells"][1]["geometry_seed"] = 102
        self.fixture.protocol["episode_schedule"] = [0, 0]
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "permutation"):
            self.collect()
        self.fixture.protocol["episode_schedule"] = [0, 1]
        self.fixture.r6["diagnostic_pool"]["geometry_seeds"] = [101]
        self.fixture.r6_sha = self.fixture.write_json(reused.R6_PATH, self.fixture.r6)
        self.fixture.protocol["r6_protocol"]["sha256"] = self.fixture.r6_sha
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "exact frozen r6"):
            self.collect()
        with patch.object(reused, "R6_SHA256", self.fixture.r6_sha):
            with self.assertRaisesRegex(ValueError, "overlaps diagnostic"):
                self.collect()
        self.assertEqual(self.envs, [])

    def test_forged_random_identity_or_protocol_purpose_and_unbounded_cap_block_reset(self):
        self.mocked()
        for field, value in (("source_actor_sha256", "a" * 64), ("rng_seed", True)):
            old = self.fixture.protocol["random"][field]
            self.fixture.protocol["random"][field] = value
            self.fixture.save()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.collect()
            self.fixture.protocol["random"][field] = old
        self.fixture.protocol["purpose"] = "TRAIN-only"
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "unsupported reused"):
            self.collect()
        self.fixture.protocol["purpose"] = reused.PURPOSE
        self.fixture.protocol["budgets"]["random_decision_cap"] = 32769
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "bounded"):
            self.collect()
        self.assertEqual(self.envs, [])

    def test_no_symlinks_or_heldout_output_or_duplicate_output(self):
        self.mocked()
        link = self.fixture.root / "experiments/dreamerv3-reused-train-link.json"
        link.symlink_to(self.fixture.root / self.fixture.protocol_path)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            reused.preflight(link, self.fixture.protocol_sha, repo_root=self.fixture.root)
        with self.assertRaisesRegex(ValueError, "TRAIN run root"):
            self.collect("blind-cell")
        output = self.output()
        output.mkdir()
        (output / "marker").write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            self.collect()
        self.assertEqual((output / "marker").read_bytes(), b"preserve")
        self.assertEqual(self.envs, [])

    def test_two_arms_store_executed_actions_raw_rewards_real_tplusone_and_replay_handoff(self):
        self.mocked()
        result = self.collect()
        self.assertEqual(result["purpose"], reused.PURPOSE)
        self.assertEqual(result["freshness_claim"], reused.LIMITATION)
        self.assertEqual(result["allowed_cells"], self.fixture.protocol["cells"])
        self.assertEqual(result["excluded_seeds"], [201])
        self.assertEqual([(env.track, env.seed) for env in self.envs], [(1, 101), (2, 102)] * 2)
        for arm in ("random", "teacher"):
            row = result["arms"][arm]
            arm_dir = self.output() / arm
            self.assertEqual(json.loads((arm_dir / "collection-result.json").read_text()), row)
            self.assertEqual((row["decisions_spent"], row["stored_decisions"], row["partial_decisions"]),
                             (4, 4, 0))
            self.assertEqual([attempt["raw_reward_sum"] for attempt in row["episode_rows"]], [3.0, 3.0])
            self.assertEqual([attempt["finished"] for attempt in row["episode_rows"]], [True, True])
            archive = (arm_dir / row["dataset_path"]).read_bytes()
            self.assertEqual(row["archive_sha256"], digest(archive))
            dataset = TeacherDataset.from_bytes(archive, expected_digest=row["dataset_digest"])
            self.assertEqual(dataset.transition_count, 4)
            self.assertEqual([episode.episode_id for episode in dataset.episodes], ["0", "1"])
            self.assertTrue(all(episode.source_id == row["source_id"] and
                                episode.source_actor_sha256 == row["source_actor_sha256"] for episode in dataset.episodes))
            self.assertEqual(dataset.episodes[0].frames[:, 0, 0].tolist(), [10, 11, 12])
            self.assertEqual(dataset.episodes[0].truncated.tolist(), [False, True])
            self.assertEqual(dataset.episodes[0].terminal.tolist(), [False, True])
            self.assertEqual(dataset.episodes[0].finished.tolist(), [False, True])
            np.testing.assert_allclose(self.envs[0 if arm == "random" else 2].actions,
                                       dataset.episodes[0].applied_actions)
            replay, receipt = replay_from_dataset_bytes(
                archive, archive_sha256=row["archive_sha256"], dataset_digest=row["dataset_digest"],
                source_id=row["source_id"], source_actor_sha256=row["source_actor_sha256"],
                allowed_cells=[(cell["track_id"], cell["geometry_seed"]) for cell in row["allowed_cells"]],
                excluded_seeds=row["excluded_seeds"], capacity=row["stored_decisions"],
            )
            self.assertEqual((receipt["decisions"], receipt["distinct_finished_cells"]), (4, 2))
            batch = replay.sample_sequence(1, 32, short_episode_fraction=1.0)
            self.assertEqual(batch["effective_seq_len"], 2)
            self.assertTrue(batch["is_terminal"][0, -1].item())
            self.assertAlmostEqual(batch["observations"][0, -1, -1, 0, 0].item(), 12 / 255.0)
        self.assertNotEqual(result["arms"]["random"]["source_actor_sha256"],
                            result["arms"]["teacher"]["source_actor_sha256"])
        random_episode = TeacherDataset.from_bytes((self.output() / "random/support-dataset.npz").read_bytes()).episodes[0]
        proposed = np.random.default_rng(77).uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32)
        np.testing.assert_allclose(random_episode.actions, proposed, atol=1e-6)
        np.testing.assert_allclose(random_episode.applied_actions,
                                   [ActionAdapter().to_official(row) for row in proposed])
        teacher_episode = TeacherDataset.from_bytes((self.output() / "teacher/support-dataset.npz").read_bytes()).episodes[0]
        np.testing.assert_allclose(teacher_episode.actions, [[1.0, -0.5, 0.25]] * 2)
        np.testing.assert_allclose(teacher_episode.applied_actions, [[1.0, 0.25, 0.625]] * 2)
        self.assertEqual(teacher_episode.metadata["proposed_native_actions"], [[2.0, -0.5, 0.25]] * 2)
        self.assertTrue(all(env.closed for env in self.envs))

    def test_real_timeout_is_not_terminal_and_retirement_is_terminal(self):
        self.mocked(builder=lambda track, seed, *args, **kwargs: self._env(track, seed,
                                                                           "timeout" if seed == 101 else "retired"))
        result = self.collect()
        for arm in ("random", "teacher"):
            row = result["arms"][arm]
            dataset = TeacherDataset.from_bytes((self.output() / arm / "support-dataset.npz").read_bytes())
            self.assertEqual(row["distinct_finished_geometries"], [])
            self.assertEqual((dataset.episodes[0].truncated[-1], dataset.episodes[0].terminal[-1]), (True, False))
            self.assertEqual((dataset.episodes[1].terminated[-1], dataset.episodes[1].terminal[-1]), (True, True))

    def _env(self, track, seed, ending="finished", length=2):
        env = FakeEnv(track, seed, ending=ending, length=length)
        self.envs.append(env)
        return env

    def test_cap_partial_uses_no_synthetic_terminal_and_no_adaptive_replacement(self):
        self.fixture.protocol["budgets"] = {"random_decision_cap": 3, "teacher_decision_cap": 3}
        self.fixture.save()
        self.mocked(builder=lambda t, s, *a, **k: self._env(t, s, length=2 if s == 101 else 4))
        result = self.collect()
        for arm in ("random", "teacher"):
            row = result["arms"][arm]
            self.assertEqual((row["decisions_spent"], row["stored_decisions"], row["partial_decisions"]), (3, 2, 1))
            self.assertEqual([item["status"] for item in row["episode_rows"]], ["complete", "cap_partial"])
            self.assertEqual(row["episode_rows"][1]["finished"], False)
            self.assertTrue(row["schedule_exhausted"])
            dataset = TeacherDataset.from_bytes((self.output() / arm / "support-dataset.npz").read_bytes())
            self.assertEqual(len(dataset.episodes), 1)
            self.assertEqual(dataset.episodes[0].geometry_id, "101")
        self.assertEqual(len(self.envs), 4)
        self.assertTrue(all(env.closed for env in self.envs))

    def test_zero_complete_writes_no_archive_and_preserves_spent_decisions(self):
        self.fixture.protocol["cells"] = self.fixture.protocol["cells"][:1]
        self.fixture.protocol["episode_schedule"] = [0]
        self.fixture.protocol["budgets"] = {"random_decision_cap": 1, "teacher_decision_cap": 1}
        self.fixture.save()
        self.mocked()
        result = self.collect()
        for arm in ("random", "teacher"):
            row = result["arms"][arm]
            self.assertEqual((row["decisions_spent"], row["stored_decisions"], row["partial_decisions"]), (1, 0, 1))
            self.assertIsNone(row["dataset_path"])
            self.assertIsNone(row["archive_sha256"])
            self.assertFalse((self.output() / arm / "support-dataset.npz").exists())

    def test_runtime_failure_records_actual_attempts_and_keeps_prior_arm_archive(self):
        class BadEnv(FakeEnv):
            def step(self, action):
                observation, reward, terminated, truncated, info = super().step(action)
                return observation, float("nan") if self.steps == 1 else reward, terminated, truncated, info

        def builder(track, seed, *args, **kwargs):
            env = (FakeEnv(track, seed) if len(self.envs) < 2 else BadEnv(track, seed))
            self.envs.append(env)
            return env

        self.mocked(builder=builder)
        with self.assertRaisesRegex(ValueError, "raw reward"):
            self.collect()
        abort = json.loads((self.output() / "abort-receipt.json").read_text())
        self.assertEqual(abort["failed_arm"], "teacher")
        self.assertEqual(abort["completed_arms"]["random"]["decisions_spent"], 4)
        self.assertTrue((self.output() / "random/support-dataset.npz").is_file())
        teacher = json.loads((self.output() / "teacher/collection-result.json").read_text())
        self.assertEqual((teacher["decisions_spent"], teacher["decision_calls"], teacher["stored_decisions"]),
                         (1, 1, 0))
        self.assertEqual(teacher["episode_rows"][0]["status"], "aborted")
        self.assertEqual([(env.seed, env.steps) for env in self.envs], [(101, 2), (102, 2), (101, 1)])
        self.assertTrue(all(env.closed for env in self.envs))

    def test_step_exception_logs_unresolved_call_without_refill(self):
        class FailingStep(FakeEnv):
            def step(self, action):
                super().step(action)
                raise RuntimeError("synthetic step interrupted after consumption")

        self.mocked(builder=lambda t, s, *a, **k: self._failing_env(t, s, FailingStep))
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.collect()
        receipt = json.loads((self.output() / "random/collection-result.json").read_text())
        self.assertEqual((receipt["decisions_spent"], receipt["decision_calls"],
                          receipt["unresolved_decision_calls"]), (0, 1, 1))
        self.assertEqual(receipt["episode_rows"][0]["decision_calls"], 1)
        self.assertEqual(len(self.envs), 1)
        self.assertTrue(self.envs[0].closed)

    def _failing_env(self, track, seed, cls):
        env = cls(track, seed)
        self.envs.append(env)
        return env

    def test_max_steps_without_real_boundary_aborts_not_synthesizes_finish(self):
        self.fixture.protocol["max_steps"] = 2
        self.fixture.save()
        self.mocked(builder=lambda t, s, *a, **k: self._env(t, s, length=4))
        with self.assertRaisesRegex(ValueError, "without a real end"):
            self.collect()
        receipt = json.loads((self.output() / "random/collection-result.json").read_text())
        self.assertEqual((receipt["decisions_spent"], receipt["stored_decisions"]), (2, 0))
        self.assertFalse(receipt["episode_rows"][0]["complete"])
        self.assertFalse((self.output() / "random/support-dataset.npz").exists())


if __name__ == "__main__":
    unittest.main()
