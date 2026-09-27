"""Source1 complement contracts using only synthetic actor, files and environment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common_adapter import ActionAdapter, ObservationSpec
from haic.algorithms.drq_v2.teacher_replay import TeacherDataset
from scripts.diagnose import collect_dreamerv3_reused_source1 as source1
from scripts.diagnose import collect_dreamerv3_reused_train as reused


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeActor:
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        assert tuple(frames.shape) == (1, 4, 84, 84)
        return torch.tensor([[2.0, -0.5, 0.25]], dtype=torch.float32)


class FakeEnv:
    def __init__(self, track: int, seed: int, *, finish: bool, bad_terminal: bool = False,
                 interrupt: bool = False):
        self.track, self.seed = track, seed
        self.finish, self.bad_terminal, self.interrupt = finish, bad_terminal, interrupt
        self.actions: list[np.ndarray] = []
        self.closed = False

    def frame(self, step: int) -> np.ndarray:
        frames = np.stack([np.full((84, 84), 10 + max(0, step - 3 + channel), dtype=np.uint8)
                           for channel in range(4)])
        return ObservationSpec().from_uint8(frames)

    def reset(self, *, seed=None, options=None):
        return self.frame(0), {"track_id": self.track, "seed": self.seed}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        if self.interrupt:
            raise RuntimeError("synthetic step interrupted after decision call")
        return self.frame(1), 1.25, not self.finish and not self.bad_terminal, self.finish, {
            "track_id": self.track, "seed": self.seed, "finished": self.finish or self.bad_terminal,
            "progress": 1.0 if self.finish else 0.3, "damage": 0.0,
            "retire_reason": "off_track" if not self.finish else None,
        }

    def close(self):
        self.closed = True


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.sources = {name: self.write(name, (Path(reused.__file__).read_bytes()
                                                  if name == reused.COLLECTOR_PATH
                                                  else f"synthetic {name}\n".encode()))
                        for name in sorted(reused.REQUIRED_SOURCES)}
        self.sources[source1.WRAPPER_PATH] = self.write(source1.WRAPPER_PATH, Path(source1.__file__).read_bytes())
        self.actor0 = {
            "source_id": "drq-source-0", "learner_seed": 0, "source_revision": source1.SOURCE_REVISION,
            "checkpoint_path": "runs/synthetic-actor0/checkpoint.pt",
            "actor_path": "runs/synthetic-actor0/actor.pt",
            "actor_weights_sha256": digest(b"weights0"),
        }
        self.actor1 = {
            "source_id": "drq-source-1", "learner_seed": 1, "source_revision": source1.SOURCE_REVISION,
            "checkpoint_path": "runs/synthetic-actor1/checkpoint.pt",
            "actor_path": "runs/synthetic-actor1/actor.pt",
            "actor_weights_sha256": digest(b"weights1"),
        }
        for actor in (self.actor0, self.actor1):
            actor["checkpoint_sha256"] = self.write(actor["checkpoint_path"], actor["source_id"].encode() + b" checkpoint")
            actor["actor_sha256"] = self.write(actor["actor_path"], actor["source_id"].encode() + b" actor")
        self.cells = [{"track_id": 1, "geometry_seed": seed} for seed in source1.FAMILIES]
        self.r6 = {
            "format": "haic-drq-geometry-mix-study-v1", "study_id": "drqv2-geometry-mix-v1-r6",
            "training_pool": {"partition": "TRAIN", "geometry_seeds": list(source1.FAMILIES),
                              "track_ids": [1, 2, 3, 4]},
            "diagnostic_pool": {"partition": "TRAIN-DIAGNOSTIC", "geometry_seeds": [999]},
            "environment": {"partition": "TRAIN", "frame_skip": 4, "max_steps": 2000,
                            "reward_shaping": False, "obstacles": True},
            "learner": {"padding": 4, "source_environment_steps": 131072},
            "source_actors": [self.r6_actor(actor) for actor in (self.actor0, self.actor1)],
        }
        self.r6_sha = self.write_json(reused.R6_PATH, self.r6)
        self.original = {
            "format": reused.FORMAT, "study_id": "dreamerv3-reused-train-diversity-v1",
            "purpose": reused.PURPOSE, "freshness_claim": reused.LIMITATION,
            "r6_protocol": {"path": reused.R6_PATH, "sha256": self.r6_sha},
            "source_sha256": {name: self.sources[name] for name in reused.REQUIRED_SOURCES},
            "cells": self.cells, "episode_schedule": list(range(12)), "frame_skip": 4,
            "max_steps": 2000, "budgets": {"random_decision_cap": 24000, "teacher_decision_cap": 24000},
            "source_actor": self.actor0,
        }
        self.original_sha = self.write_json(source1.SOURCE0_PROTOCOL, self.original)
        self.receipt = {
            "status": "completed", "arm": "teacher", "study_id": self.original["study_id"],
            "protocol_sha256": self.original_sha, "r6_protocol_sha256": self.r6_sha,
            "source_id": "drq-source-0", "source_actor_sha256": self.actor0["actor_sha256"],
            "source_checkpoint_sha256": self.actor0["checkpoint_sha256"],
            "allowed_cells": self.cells, "decision_cap": 24000, "schedule_attempts": 12,
            "complete_episode_count": 12, "schedule_exhausted": True, "partial_decisions": 0,
            "unresolved_decision_calls": 0, "archive_sha256": digest(b"immutable source0 archive"),
            "distinct_finished_geometries": list(source1.SOURCE0_FINISHES),
            "episode_rows": [{"attempt": i, "episode_id": i, **cell, "status": "complete",
                              "complete": True, "terminal": True, "decisions": 1,
                              "finished": cell["geometry_seed"] in source1.SOURCE0_FINISHES}
                             for i, cell in enumerate(self.cells)],
        }
        self.receipt_sha = self.write_json(source1.SOURCE0_RECEIPT, self.receipt)
        self.catalog = {"format": "haic-drq-training-geometry-catalog-v1",
                        "train": [{"geometry_seed": seed, "family": family}
                                  for seed, family in source1.FAMILIES.items()]}
        self.catalog_sha = self.write_json(source1.CATALOG_PATH, self.catalog)
        self.historical = {
            "format": "haic-drq-training-geometry-diagnostic-summary-v1",
            "catalog_sha256": self.catalog_sha,
            "geometry": [{"geometry_seed": seed, "family": family, "partition": "train",
                          "by_source": {"1": {"finished": seed in source1.HISTORICAL_SOURCE1_FINISHES}}}
                         for seed, family in source1.FAMILIES.items()],
        }
        self.historical_sha = self.write_json(source1.HISTORICAL_PATH, self.historical)
        self.protocol = {
            "format": source1.FORMAT, "study_id": source1.STUDY_ID,
            "purpose": reused.PURPOSE, "freshness_claim": reused.LIMITATION,
            "source0_protocol": {"path": source1.SOURCE0_PROTOCOL, "sha256": self.original_sha},
            "source0_teacher_receipt": {"path": source1.SOURCE0_RECEIPT, "sha256": self.receipt_sha},
            "r6_protocol": {"path": reused.R6_PATH, "sha256": self.r6_sha},
            "catalog": {"path": source1.CATALOG_PATH, "sha256": self.catalog_sha},
            "historical_train_summary": {"path": source1.HISTORICAL_PATH, "sha256": self.historical_sha},
            "source_sha256": self.sources, "cells": self.cells, "episode_schedule": list(range(12)),
            "frame_skip": 4, "max_steps": 2000, "budgets": {"teacher_decision_cap": 24000},
            "source_actor": self.actor1,
            "spec_fingerprints": {"action": ActionAdapter().spec.fingerprint,
                                  "observation": ObservationSpec().fingerprint},
        }
        self.save()
        (root / source1.OUTPUT_PATH).parent.mkdir(parents=True)

    @staticmethod
    def r6_actor(actor: dict) -> dict:
        return {"source_seed": actor["learner_seed"], "weight_only_fork": True,
                "source_revision": actor["source_revision"],
                "source_checkpoint_path": actor["checkpoint_path"],
                "source_checkpoint_sha256": actor["checkpoint_sha256"],
                "source_actor_path": actor["actor_path"],
                "source_actor_sha256": actor["actor_sha256"],
                "source_actor_weights_sha256": actor["actor_weights_sha256"]}

    def write(self, name: str, content: bytes) -> str:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return digest(content)

    def write_json(self, name: str, value: dict) -> str:
        return self.write(name, (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())

    def save(self) -> None:
        self.protocol_sha = self.write_json(source1.PROTOCOL_PATH, self.protocol)

    def pair(self) -> dict:
        return {"learner_seed": 1, "source_revision": source1.SOURCE_REVISION,
                "checkpoint_sha256": self.actor1["checkpoint_sha256"],
                "actor_sha256": self.actor1["actor_sha256"],
                "actor_weights_sha256": self.actor1["actor_weights_sha256"],
                "action_fingerprint": self.protocol["spec_fingerprints"]["action"],
                "observation_fingerprint": self.protocol["spec_fingerprints"]["observation"],
                "cpu_smoke_observations": 3}


class Source1Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="source1-synthetic-")
        self.addCleanup(temp.cleanup)
        self.fx = Fixture(Path(temp.name))
        self.envs: list[FakeEnv] = []
        for module, field, value in (
            (reused, "R6_SHA256", self.fx.r6_sha), (source1, "SOURCE0_PROTOCOL_SHA256", self.fx.original_sha),
            (source1, "SOURCE0_RECEIPT_SHA256", self.fx.receipt_sha),
            (source1, "CATALOG_SHA256", self.fx.catalog_sha),
            (source1, "HISTORICAL_SHA256", self.fx.historical_sha),
            (source1, "ACTOR_SHA256", self.fx.actor1["actor_sha256"]),
            (source1, "CHECKPOINT_SHA256", self.fx.actor1["checkpoint_sha256"]),
            (source1, "WEIGHTS_SHA256", self.fx.actor1["actor_weights_sha256"]),
        ):
            context = patch.object(module, field, value)
            context.start()
            self.addCleanup(context.stop)
        for module, field, result in (
            (reused, "audit_source_actor_pair", self.fx.pair()),
            (reused, "load_exported_actor", (FakeActor(), ActionAdapter(), ObservationSpec())),
        ):
            context = patch.object(module, field, return_value=result)
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(reused, "build_env", side_effect=self.builder)
        context.start()
        self.addCleanup(context.stop)

    def builder(self, track, geometry_seed, max_steps, frame_skip, *, reward_shaping, obstacles, collision_penalty):
        self.assertEqual((track, max_steps, frame_skip, reward_shaping, obstacles, collision_penalty),
                         (1, 2000, 4, False, True, 0.0))
        env = FakeEnv(track, geometry_seed, finish=geometry_seed in (3910800011, 3910800008))
        self.envs.append(env)
        return env

    def collect(self):
        return source1.collect(self.fx.root / source1.PROTOCOL_PATH, self.fx.protocol_sha,
                               self.fx.root / source1.OUTPUT_PATH, repo_root=self.fx.root)

    def reject_before_reset(self, pattern: str):
        with self.assertRaisesRegex((ValueError, FileExistsError), pattern):
            self.collect()
        self.assertFalse(self.envs)
        self.assertFalse((self.fx.root / source1.OUTPUT_PATH).exists())

    def test_real_input_sha_constants_are_exact_and_static(self):
        self.assertEqual(reused.sha256(source1.ROOT / source1.SOURCE0_PROTOCOL),
                         "a2f3e92fd1b2e95f883b24e3ef28d58b3e70938a0dd867d2c9ec5da5bd46df83")
        self.assertEqual(reused.sha256(source1.ROOT / source1.SOURCE0_RECEIPT),
                         "00a388f5eabdd9707b096dfb735b2d717b309b86a9daa1423d2c16177fa2c353")
        self.assertEqual(reused.sha256(source1.ROOT / source1.CATALOG_PATH),
                         "a8cbe5d2445563f9065a944254f6410486135b20eb8574ec1cf9eb611ebbf00b")
        self.assertEqual(reused.sha256(source1.ROOT / source1.HISTORICAL_PATH),
                         "29fe9d2b8d4eff04879cc9c1390b4701e4f235af710856f39acf6592f65015ee")
        self.assertEqual(reused.sha256(Path(reused.__file__)),
                         source1.ORIGINAL_COLLECTOR_SHA256)
        catalog = json.loads((source1.ROOT / source1.CATALOG_PATH).read_text())
        actual = {item["geometry_seed"]: item["family"] for item in catalog["train"]
                  if item["geometry_seed"] in source1.FAMILIES}
        self.assertEqual(actual, source1.FAMILIES)
        historical = json.loads((source1.ROOT / source1.HISTORICAL_PATH).read_text())
        prior = sorted(item["geometry_seed"] for item in historical["geometry"]
                       if item["geometry_seed"] in source1.FAMILIES and item["by_source"]["1"]["finished"])
        self.assertEqual(prior, list(source1.HISTORICAL_SOURCE1_FINISHES))

    def test_source1_teacher_only_archive_and_complement_gate(self):
        row = self.collect()
        output = self.fx.root / source1.OUTPUT_PATH
        self.assertEqual(row["source_id"], "drq-source-1")
        self.assertEqual(row["source0_reference"]["teacher_receipt_sha256"], self.fx.receipt_sha)
        self.assertFalse(row["source0_reference"]["original_learner_gate_passed"])
        self.assertEqual(row["source1_finished_geometries"], [3910800008, 3910800011])
        self.assertEqual(row["new_finished_geometries"], [3910800008])
        self.assertEqual(row["union_finished_geometries"], [3910800008, 3910800011, 3910800041])
        self.assertEqual(row["historical_source1_reference"]["sha256"], self.fx.historical_sha)
        self.assertEqual(row["historical_source1_reference"]["finished_geometries_on_same_reused_train_roads"],
                         list(source1.HISTORICAL_SOURCE1_FINISHES))
        self.assertIn("not historical source1", row["new_finished_geometries_relative_to"])
        self.assertEqual(len(row["union_finished_families"]), 3)
        self.assertTrue(row["support_gate"]["passed"])
        self.assertFalse(row["support_gate"]["authorizes_training"])
        self.assertFalse(row["support_gate"]["merges_archives"])
        teacher = row["teacher"]
        self.assertEqual((teacher["schedule_attempts"], teacher["complete_episode_count"],
                          teacher["decisions_spent"], teacher["stored_decisions"]), (12, 12, 12, 12))
        self.assertEqual(teacher["source_actor_sha256"], self.fx.actor1["actor_sha256"])
        archive = (output / "teacher/support-dataset.npz").read_bytes()
        self.assertEqual(digest(archive), teacher["archive_sha256"])
        dataset = TeacherDataset.from_bytes(archive, expected_digest=teacher["dataset_digest"])
        episode = dataset.episodes[0]
        self.assertEqual((episode.source_id, episode.source_actor_sha256),
                         ("drq-source-1", self.fx.actor1["actor_sha256"]))
        self.assertEqual(episode.frames[:, 0, 0].tolist(), [10, 11])
        self.assertEqual((episode.truncated[-1], episode.terminal[-1]), (True, True))
        np.testing.assert_allclose(episode.actions[0], [1.0, -0.5, 0.25])
        np.testing.assert_allclose(episode.applied_actions[0], [1.0, 0.25, 0.625])
        self.assertEqual(dataset.episodes[1].finished.tolist(), [True])
        self.assertFalse((output / "random").exists())
        self.assertTrue(all(env.closed for env in self.envs))

    def test_no_new_road_does_not_release_learner_or_erase_source0(self):
        def only_old(track, seed, *args, **kwargs):
            env = FakeEnv(track, seed, finish=seed in source1.SOURCE0_FINISHES)
            self.envs.append(env)
            return env

        with patch.object(reused, "build_env", side_effect=only_old):
            row = self.collect()
        self.assertEqual(row["new_finished_geometries"], [])
        self.assertEqual(row["union_finished_geometries"], list(source1.SOURCE0_FINISHES))
        self.assertFalse(row["support_gate"]["passed"])
        self.assertEqual(row["teacher"]["schedule_attempts"], 12)

    def test_source0_forged_receipt_and_wrong_identity_refused(self):
        self.fx.write(source1.SOURCE0_RECEIPT, b"forged old receipt")
        self.reject_before_reset("pinned reference SHA-256")
        self.fx.receipt_sha = self.fx.write_json(source1.SOURCE0_RECEIPT, self.fx.receipt)
        self.fx.protocol["source0_teacher_receipt"]["sha256"] = self.fx.receipt_sha
        self.fx.receipt["episode_rows"][0]["finished"] = True
        forged_sha = self.fx.write_json(source1.SOURCE0_RECEIPT, self.fx.receipt)
        self.fx.protocol["source0_teacher_receipt"]["sha256"] = forged_sha
        self.fx.save()
        with patch.object(source1, "SOURCE0_RECEIPT_SHA256", forged_sha):
            self.reject_before_reset("source0 teacher rows")
        self.fx.receipt["episode_rows"][0]["finished"] = False
        self.fx.receipt_sha = self.fx.write_json(source1.SOURCE0_RECEIPT, self.fx.receipt)
        self.fx.protocol["source0_teacher_receipt"]["sha256"] = self.fx.receipt_sha
        self.fx.protocol["source_actor"] = self.fx.actor0
        self.fx.save()
        self.reject_before_reset("source1 identity")

    def test_rejects_wrong_cells_order_cap_and_frame_contract(self):
        cases = (("cells", self.fx.cells[:-1]), ("episode_schedule", [0] * 12),
                 ("budgets", {"teacher_decision_cap": 24001}), ("max_steps", 2001),
                 ("frame_skip", 1))
        for field, value in cases:
            with self.subTest(field=field):
                original = self.fx.protocol[field]
                self.fx.protocol[field] = value
                self.fx.save()
                self.reject_before_reset("same twelve")
                self.fx.protocol[field] = original

    def test_rejects_catalog_bytes_family_and_pinned_source_map(self):
        self.fx.write(source1.CATALOG_PATH, b"changed catalog")
        self.reject_before_reset("pinned reference SHA-256")
        self.fx.catalog["train"][1]["family"] = "easy-curvature-anchor"
        self.fx.catalog_sha = self.fx.write_json(source1.CATALOG_PATH, self.fx.catalog)
        self.fx.protocol["catalog"]["sha256"] = self.fx.catalog_sha
        self.fx.save()
        with patch.object(source1, "CATALOG_SHA256", self.fx.catalog_sha):
            self.reject_before_reset("catalog TRAIN families")
        self.fx.catalog["train"][1]["family"] = source1.FAMILIES[3910800011]
        self.fx.catalog_sha = self.fx.write_json(source1.CATALOG_PATH, self.fx.catalog)
        self.fx.protocol["catalog"]["sha256"] = self.fx.catalog_sha
        self.fx.protocol["source_sha256"]["train.py"] = "a" * 64
        self.fx.save()
        self.reject_before_reset("exact original runtime source map")

    def test_rejects_historical_source1_reference_and_rewritten_outcomes(self):
        self.fx.write(source1.HISTORICAL_PATH, b"replaced historical source1 result")
        self.reject_before_reset("pinned reference SHA-256")
        self.fx.historical["geometry"][0]["by_source"]["1"]["finished"] = False
        forged_sha = self.fx.write_json(source1.HISTORICAL_PATH, self.fx.historical)
        self.fx.protocol["historical_train_summary"]["sha256"] = forged_sha
        self.fx.save()
        with patch.object(source1, "HISTORICAL_SHA256", forged_sha):
            self.reject_before_reset("historical source1 finishes")

    def test_rejects_actor_checkpoint_and_cpu_smoke_mismatch(self):
        self.fx.write(self.fx.actor1["checkpoint_path"], b"changed checkpoint")
        self.reject_before_reset("SHA-256 mismatch")
        self.fx.write(self.fx.actor1["checkpoint_path"], b"drq-source-1 checkpoint")
        self.fx.write(self.fx.actor1["actor_path"], b"changed actor")
        self.reject_before_reset("SHA-256 mismatch")
        self.fx.write(self.fx.actor1["actor_path"], b"drq-source-1 actor")
        with patch.object(reused, "audit_source_actor_pair", return_value={**self.fx.pair(),
                                                                              "cpu_smoke_observations": 0}):
            self.reject_before_reset("CPU actor/checkpoint parity")

    def test_wrapper_is_pinned_against_executing_bytes_and_symlinks(self):
        self.fx.sources[source1.WRAPPER_PATH] = self.fx.write(source1.WRAPPER_PATH, b"forged wrapper")
        self.fx.protocol["source_sha256"] = self.fx.sources
        self.fx.save()
        self.reject_before_reset("executing source1 wrapper differs")
        self.fx.sources[source1.WRAPPER_PATH] = self.fx.write(source1.WRAPPER_PATH,
                                                               Path(source1.__file__).read_bytes())
        self.fx.save()
        target = self.fx.root / self.fx.actor1["actor_path"]
        target.unlink()
        target.symlink_to(self.fx.root / self.fx.actor0["actor_path"])
        self.reject_before_reset("symlinks")

    def test_wrong_protocol_hash_output_overwrite_and_heldout_path(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            source1.collect(self.fx.root / source1.PROTOCOL_PATH, "0" * 64,
                            self.fx.root / source1.OUTPUT_PATH, repo_root=self.fx.root)
        with self.assertRaisesRegex(ValueError, "single new source1-only"):
            source1.collect(self.fx.root / source1.PROTOCOL_PATH, self.fx.protocol_sha,
                            self.fx.root / "runs/heldout/collection", repo_root=self.fx.root)
        output = self.fx.root / source1.OUTPUT_PATH
        output.mkdir()
        (output / "marker").write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            self.collect()
        self.assertEqual((output / "marker").read_bytes(), b"preserve")
        self.assertFalse(self.envs)

    def test_reset_guard_rejects_wrapper_catalog_source0_or_history_drift(self):
        for name in (source1.WRAPPER_PATH, source1.CATALOG_PATH,
                     source1.SOURCE0_RECEIPT, source1.HISTORICAL_PATH):
            with self.subTest(name=name):
                if (self.fx.root / source1.OUTPUT_PATH).exists():
                    temp = tempfile.TemporaryDirectory(prefix="source1-guard-second-")
                    self.addCleanup(temp.cleanup)
                    self.fx = Fixture(Path(temp.name))
                    (self.fx.root / source1.OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

                def builder(track, seed, *args, **kwargs):
                    env = self.builder(track, seed, *args, **kwargs)
                    if len(self.envs) == 1:
                        original_close = env.close

                        def close():
                            original_close()
                            self.fx.write(name, b"source changed between attempts")

                        env.close = close
                    return env

                with patch.object(reused, "build_env", side_effect=builder):
                    with self.assertRaisesRegex(ValueError, "pinned reference SHA-256"):
                        self.collect()
                self.assertEqual(len(self.envs), 1)
                output = self.fx.root / source1.OUTPUT_PATH
                arm = json.loads((output / "teacher/collection-result.json").read_text())
                abort = json.loads((output / "abort-receipt.json").read_text())
                self.assertEqual((arm["decision_calls"], arm["decisions_spent"],
                                  arm["stored_decisions"], arm["schedule_attempts"]), (1, 1, 1, 2))
                self.assertEqual((abort["decision_calls"], abort["decisions_spent"],
                                  abort["stored_decisions"]), (1, 1, 1))
                self.assertTrue(abort["consumption_known"])
                self.envs.clear()

    def test_original_collector_checks_runtime_and_actor_per_reset(self):
        for name in ("train.py", "env_wrapper.py", self.fx.actor1["actor_path"],
                     self.fx.actor1["checkpoint_path"]):
            with self.subTest(name=name):
                if (self.fx.root / source1.OUTPUT_PATH).exists():
                    temp = tempfile.TemporaryDirectory(prefix="source1-actor-drift-")
                    self.addCleanup(temp.cleanup)
                    self.fx = Fixture(Path(temp.name))
                    (self.fx.root / source1.OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

                def builder(track, seed, *args, **kwargs):
                    env = self.builder(track, seed, *args, **kwargs)
                    if len(self.envs) == 1:
                        close = env.close

                        def drift():
                            close()
                            self.fx.write(name, b"source changed after first completed cell")

                        env.close = drift
                    return env

                with patch.object(reused, "build_env", side_effect=builder):
                    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                        self.collect()
                self.assertEqual(len(self.envs), 1)
                abort = json.loads((self.fx.root / source1.OUTPUT_PATH / "abort-receipt.json").read_text())
                self.assertEqual((abort["decision_calls"], abort["decisions_spent"]), (1, 1))
                self.envs.clear()

    def test_terminal_framing_failure_logs_actual_calls_and_no_false_finish(self):
        def builder(track, seed, *args, **kwargs):
            env = FakeEnv(track, seed, finish=False, bad_terminal=True)
            self.envs.append(env)
            return env

        with patch.object(reused, "build_env", side_effect=builder):
            with self.assertRaisesRegex(ValueError, "finished flag disagrees"):
                self.collect()
        output = self.fx.root / source1.OUTPUT_PATH
        arm = json.loads((output / "teacher/collection-result.json").read_text())
        abort = json.loads((output / "abort-receipt.json").read_text())
        self.assertEqual((arm["decisions_spent"], arm["decision_calls"], arm["stored_decisions"]), (1, 1, 0))
        self.assertEqual(arm["distinct_finished_geometries"], [])
        self.assertEqual(abort["decisions_spent"], 1)
        self.assertTrue(self.envs[0].closed)

    def test_step_exception_does_not_falsely_report_zero_consumption(self):
        def builder(track, seed, *args, **kwargs):
            env = FakeEnv(track, seed, finish=False, interrupt=True)
            self.envs.append(env)
            return env

        with patch.object(reused, "build_env", side_effect=builder):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.collect()
        arm = json.loads((self.fx.root / source1.OUTPUT_PATH / "teacher/collection-result.json").read_text())
        abort = json.loads((self.fx.root / source1.OUTPUT_PATH / "abort-receipt.json").read_text())
        self.assertEqual((arm["decision_calls"], arm["decisions_spent"],
                          arm["unresolved_decision_calls"]), (1, 0, 1))
        self.assertEqual((abort["decision_calls"], abort["decisions_spent"],
                          abort["unresolved_decision_calls"]), (1, 0, 1))


if __name__ == "__main__":
    unittest.main()
