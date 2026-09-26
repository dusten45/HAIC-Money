"""Pure synthetic P1 sidecar tests; no simulator, roads, or real models are opened."""

from __future__ import annotations

import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common_adapter import ActionAdapter, EpisodeCollector, ObservationSpec
from haic.algorithms.drq_v2.teacher_replay import TeacherDataset
from haic.algorithms.dreamer_v3.offline import replay_from_dataset_bytes
from scripts import audit_dreamerv3_p1_seeds as seed_auditor
from scripts import collect_dreamerv3_support as support


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.source_hashes = {
            path: self.write(path, f"synthetic source {path}\n".encode())
            for path in sorted(support.REQUIRED_SOURCES)
        }
        self.auditor_hash = self.write(
            "scripts/audit_dreamerv3_p1_seeds.py", Path(seed_auditor.__file__).read_bytes(),
        )
        self.audit_sources = {
            "protocol": {
                path: self.write(path, b"{\"synthetic\": true}\n")
                for path in sorted(support.KNOWN_AUDIT_INPUTS["protocol"])
            },
            "prior_seed_audit": {
                "experiments/other-geometry-audit.json": self.write(
                    "experiments/other-geometry-audit.json", b"{\"synthetic\": true}\n",
                ),
            },
            "training_ledger": {
                "runs/training-only/episodes.jsonl": self.write(
                    "runs/training-only/episodes.jsonl", b'{"track_id":1,"seed":4}\n',
                ),
            },
            "prior_collection_ledger": {
                path: self.write(path, b'{"geometry_seed":5,"track_id":2}\n')
                for path in sorted(support.KNOWN_AUDIT_INPUTS["prior_collection_ledger"])
            },
        }
        self.actor_hash = self.write("runs/source-actor/actor.pt", b"synthetic actor")
        self.checkpoint_hash = self.write("runs/source-actor/checkpoint.pt", b"synthetic checkpoint")
        self.source_weights_hash = digest(b"synthetic actor weights")
        self.protocol = {
            "format": support.FORMAT,
            "study_id": "dreamerv3-p1-synthetic-only",
            "purpose": "TRAIN-only",
            "source_sha256": self.source_hashes,
            "seed_audit": {
                "path": "runs/synthetic-only/seed-audit.json",
                "sha256": "0" * 64,
                "sources": self.audit_sources,
                "auditor": {"path": "scripts/audit_dreamerv3_p1_seeds.py", "sha256": self.auditor_hash},
            },
            "source_actor": {
                "source_id": "drq-source-0",
                "learner_seed": 0,
                "source_revision": "synthetic-revision",
                "checkpoint_path": "runs/source-actor/checkpoint.pt",
                "checkpoint_sha256": self.checkpoint_hash,
                "actor_path": "runs/source-actor/actor.pt",
                "actor_sha256": self.actor_hash,
                "actor_weights_sha256": self.source_weights_hash,
                "training_ledger_path": "runs/training-only/episodes.jsonl",
                "training_ledger_sha256": self.audit_sources["training_ledger"]["runs/training-only/episodes.jsonl"],
            },
            "training_pool": {
                "cells": [{"track_id": 1, "geometry_seed": 101}],
                "episode_schedule": [0],
            },
            "training_development": {"seeds": [201]},
            "partitions": {part: {"seeds": [seed]} for part, seed in (
                ("screen", 301), ("confirmation", 401), ("blind", 501),
            )},
            "reserved_training_seeds": [201, 301, 401, 501],
            "frame_skip": 4,
            "max_steps": 5,
            "budgets": {"decision_cap": 4, "minimum_distinct_finished_geometries": 1},
            "spec_fingerprints": {
                "action": ActionAdapter().spec.fingerprint, "observation": ObservationSpec().fingerprint,
            },
        }
        self.audit = {}
        self.write_audit()
        self.write_protocol()

    def write(self, relative: str, content: bytes) -> str:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return digest(content)

    def write_audit(self) -> None:
        seeds = sorted(cell["geometry_seed"] for cell in self.protocol["training_pool"]["cells"])
        evidence = [
            {"kind": kind, "path": path, "sha256": sha, "schema_checked": True}
            for kind, declarations in self.audit_sources.items() for path, sha in declarations.items()
        ]
        self.audit = {
            "format": support.AUDIT_FORMAT,
            "study_id": self.protocol["study_id"],
            "passed": True,
            "purpose": "Dreamer P1 TRAIN-only geometry seed IDs",
            "inventory_complete": True,
            "schema_validation": "pass",
            "auditor_source_sha256": self.auditor_hash,
            "proposed_seeds": seeds,
            "proposed_seed_count": len(seeds),
            "proposed_seeds_sha256": digest(json.dumps(seeds, separators=(",", ":")).encode()),
            "matched_collisions": [],
            "parse_errors": [],
            "retired_pool_collisions": [],
            "source_evidence": evidence,
            "read_paths": [row["path"] for row in evidence],
            "read_scope": support.AUDIT_SCOPE,
            "freshness_claim": support.FRESHNESS_LIMITATION,
            "blind_data_access": "none; partition seed IDs are exclusion-only",
            "structural_blind_geometry_comparison": "not performed",
        }
        self.flush_audit()

    def flush_audit(self) -> None:
        self.audit_sha = self.write(self.protocol["seed_audit"]["path"], json_bytes(self.audit))
        self.protocol["seed_audit"]["sha256"] = self.audit_sha

    def write_protocol(self) -> None:
        self.protocol_sha = self.write("experiments/dreamerv3-p1-synthetic-only.json", json_bytes(self.protocol))

    @property
    def protocol_path(self) -> Path:
        return self.root / "experiments/dreamerv3-p1-synthetic-only.json"

    @property
    def audit_path(self) -> Path:
        return self.root / self.protocol["seed_audit"]["path"]

    def args(self) -> tuple[Path, str, Path, str]:
        return self.protocol_path, self.protocol_sha, self.audit_path, self.audit_sha

    def pair(self) -> dict:
        source = self.protocol["source_actor"]
        return {
            "source_revision": source["source_revision"],
            "learner_seed": 0,
            "checkpoint_sha256": self.checkpoint_hash,
            "actor_sha256": self.actor_hash,
            "actor_weights_sha256": self.source_weights_hash,
            "action_fingerprint": self.protocol["spec_fingerprints"]["action"],
            "observation_fingerprint": self.protocol["spec_fingerprints"]["observation"],
            "cpu_smoke_observations": 3,
        }


class FakeActor:
    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        assert tuple(observations.shape) == (1, 4, 84, 84)
        return torch.tensor([[2.0, -0.5, 0.25]], dtype=torch.float32)


class SyntheticCollector(EpisodeCollector):
    """Run the real Transition contract only against in-memory fake observations."""


class FakeEnv:
    def __init__(self, track_id: int, seed: int, *, length: int = 2, ending: str = "finished"):
        self.track_id, self.seed = track_id, seed
        self.length, self.ending = length, ending
        self.steps = 0
        self.actions: list[np.ndarray] = []
        self.closed = False
        self.spec = ObservationSpec()

    def observation(self, step: int) -> np.ndarray:
        latest = [np.full((84, 84), 10 + index, dtype=np.uint8) for index in range(step + 1)]
        stack = np.stack([latest[max(0, step - 3 + channel)] for channel in range(4)])
        return self.spec.from_uint8(stack)

    def reset(self, *, seed=None, options=None):
        assert self.steps == 0
        return self.observation(0), {"track_id": self.track_id, "seed": self.seed}

    def step(self, official_action):
        self.actions.append(np.asarray(official_action).copy())
        self.steps += 1
        last = self.steps == self.length
        is_finish = last and self.ending in ("finished", "finished-timeout")
        truncated = last and self.ending in ("finished", "timeout", "finished-timeout")
        terminated = last and self.ending == "retired"
        return self.observation(self.steps), float(self.steps), terminated, truncated, {
            "track_id": self.track_id, "seed": self.seed, "finished": is_finish,
            "progress": float(self.steps) / self.length, "damage": 0.25,
            "retire_reason": "off_track" if terminated and self.ending == "retired" else None,
        }

    def close(self):
        self.closed = True


class DreamerSupportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dreamer-support-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.fixture = Fixture(Path(temporary.name))
        self.envs: list[FakeEnv] = []

    def builder(self, track_id, geometry_seed, max_steps, frame_skip, *, reward_shaping, obstacles, collision_penalty):
        self.assertEqual((frame_skip, reward_shaping, obstacles, collision_penalty), (4, False, True, 0.0))
        env = FakeEnv(track_id, geometry_seed)
        self.envs.append(env)
        return env

    def checked(self, *args):
        return support.preflight(*(args or self.fixture.args()), repo_root=self.fixture.root)

    def collect(self, output: Path | None = None):
        output = output or self.fixture.root / "runs/synthetic-only/collection"
        return support.collect(*self.fixture.args(), output, repo_root=self.fixture.root)

    def mocks(self, *, builder=None, collector=None, actor=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        # Model a future independently passing auditor only for synthetic collector contracts.
        stack.enter_context(patch.object(
            seed_auditor, "audit_dreamerv3_p1_seeds",
            side_effect=lambda *args, **kwargs: copy.deepcopy(self.fixture.audit),
        ))
        stack.enter_context(patch.object(support, "audit_source_actor_pair", return_value=self.fixture.pair()))
        stack.enter_context(patch.object(support, "build_env", side_effect=builder or self.builder))
        stack.enter_context(patch.object(support, "load_exported_actor", return_value=(
            actor or FakeActor(), ActionAdapter(), ObservationSpec(),
        )))
        stack.enter_context(patch.object(support, "EpisodeCollector", collector or SyntheticCollector))
        return stack

    def test_self_declared_passing_receipt_cannot_open_collector(self):
        with self.assertRaisesRegex(ValueError, "pinned seed auditor rejected"):
            self.checked()
        with self.assertRaisesRegex(ValueError, "pinned seed auditor rejected"):
            self.collect()
        self.assertEqual(self.envs, [])
        self.assertFalse((self.fixture.root / "runs/synthetic-only/collection").exists())

    def test_disagreeing_auditor_output_cannot_open_collector(self):
        with patch.object(seed_auditor, "audit_dreamerv3_p1_seeds", return_value={"passed": False}):
            with self.assertRaisesRegex(ValueError, "not reproduced as a passing"):
                self.checked()

    def test_preflight_requires_new_exact_protocol_and_independent_audit_source(self):
        self.mocks()
        checked = self.checked()
        self.assertEqual(checked["protocol_sha256"], self.fixture.protocol_sha)
        self.assertEqual(
            support.preflight(
                Path("experiments/dreamerv3-p1-synthetic-only.json"), self.fixture.protocol_sha,
                Path("runs/synthetic-only/seed-audit.json"), self.fixture.audit_sha,
                repo_root=self.fixture.root,
            )["seed_audit_sha256"],
            self.fixture.audit_sha,
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.checked(self.fixture.protocol_path, "0" * 64, self.fixture.audit_path, self.fixture.audit_sha)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.checked(self.fixture.protocol_path, self.fixture.protocol_sha, self.fixture.audit_path, "0" * 64)
        self.fixture.protocol["seed_audit"]["auditor"]["sha256"] = "0" * 64
        self.fixture.write_protocol()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.checked()

    def test_input_symlinks_and_untrusted_output_paths_are_rejected(self):
        self.mocks()
        symlink = self.fixture.root / "experiments/dreamerv3-p1-link.json"
        symlink.symlink_to(self.fixture.protocol_path)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.checked(symlink, self.fixture.protocol_sha, self.fixture.audit_path, self.fixture.audit_sha)
        with self.assertRaisesRegex(ValueError, "TRAIN run root"):
            self.collect(self.fixture.root / "evaluations/new-output")
        with self.assertRaisesRegex(ValueError, "TRAIN run root"):
            self.collect(self.fixture.root / "runs/blind-synthetic/new-output")

    def test_old_drq_receipt_cannot_certify_dreamer_and_missing_inputs_block_reset(self):
        self.mocks()
        self.fixture.audit["format"] = "haic-drq-training-seed-audit-v1"
        self.fixture.flush_audit()
        self.fixture.write_protocol()
        with self.assertRaisesRegex(ValueError, "does not certify"):
            self.collect()
        self.assertEqual(self.envs, [])
        self.assertFalse((self.fixture.root / "runs/synthetic-only/collection").exists())
        self.fixture.write_audit()
        self.fixture.protocol["seed_audit"]["sources"]["prior_collection_ledger"] = {}
        self.fixture.write_protocol()
        with self.assertRaisesRegex(ValueError, "known cross-lane|no prior_collection_ledger"):
            self.checked()

    def test_schema_unchecked_forged_colliding_or_foreign_audit_rejected(self):
        self.mocks()
        for field, value in (
            ("schema_validation", "unknown"), ("matched_collisions", [{"seed": 101}]),
            ("parse_errors", ["opaque input"]), ("inventory_complete", False),
            ("study_id", "another-study"), ("proposed_seeds", [999]),
        ):
            original = self.fixture.audit[field]
            self.fixture.audit[field] = value
            self.fixture.flush_audit()
            self.fixture.write_protocol()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "does not certify"):
                self.checked()
            self.fixture.audit[field] = original
        self.fixture.write_audit()
        self.fixture.audit["source_evidence"][0]["schema_checked"] = False
        self.fixture.flush_audit()
        self.fixture.write_protocol()
        with self.assertRaisesRegex(ValueError, "evidence is incomplete"):
            self.checked()

    def test_stale_id_sources_or_actor_weights_block_preflight(self):
        self.mocks()
        ledger = self.fixture.root / "runs/training-only/episodes.jsonl"
        ledger.write_text('{"seed":999}\n')
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.checked()
        self.fixture.write("runs/training-only/episodes.jsonl", b'{"track_id":1,"seed":4}\n')
        checkpoint = self.fixture.root / self.fixture.protocol["source_actor"]["checkpoint_path"]
        checkpoint.write_bytes(b"unapproved checkpoint mutation")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.checked()
        self.fixture.write("runs/source-actor/checkpoint.pt", b"synthetic checkpoint")
        actor = self.fixture.root / self.fixture.protocol["source_actor"]["actor_path"]
        actor.write_bytes(b"unapproved actor mutation")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.checked()
        self.fixture.write("runs/source-actor/actor.pt", b"synthetic actor")
        self.fixture.protocol["source_actor"]["actor_weights_sha256"] = "f" * 64
        self.fixture.write_protocol()
        with self.assertRaisesRegex(ValueError, "actor_weights_sha256"):
            self.checked()

    def test_actor_training_ledger_must_be_in_exact_audited_inventory(self):
        self.mocks()
        self.fixture.protocol["source_actor"]["training_ledger_sha256"] = "a" * 64
        self.fixture.write_protocol()
        with self.assertRaisesRegex(ValueError, "source actor training ledger"):
            self.checked()

    def test_unpinned_collector_source_blocks_actor_and_environment(self):
        self.mocks()
        (self.fixture.root / "scripts/collect_dreamerv3_support.py").write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.collect()
        self.assertEqual(self.envs, [])

    def test_rejects_pool_evaluation_overlap_or_missing_reserved_development(self):
        self.mocks()
        self.fixture.protocol["training_pool"]["cells"][0]["geometry_seed"] = 301
        self.fixture.write_protocol()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.checked()
        self.fixture.protocol["training_pool"]["cells"][0]["geometry_seed"] = 101
        self.fixture.protocol["reserved_training_seeds"].remove(201)
        self.fixture.write_protocol()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.checked()

    def test_completed_episode_seals_exact_executed_native_official_tplusone(self):
        self.mocks()
        result = self.collect()
        self.assertTrue(result["coverage_pass"])
        self.assertEqual((result["decisions_spent"], result["stored_decisions"]), (2, 2))
        self.assertEqual(result["source_id"], "drq-source-0")
        self.assertEqual(result["source_actor_sha256"], self.fixture.actor_hash)
        self.assertEqual(result["source_checkpoint_sha256"], self.fixture.checkpoint_hash)
        archive = self.fixture.root / "runs/synthetic-only/collection/support-dataset.npz"
        self.assertEqual(result["archive_sha256"], support.sha256(archive))
        dataset = TeacherDataset.from_bytes(archive.read_bytes(), expected_digest=result["dataset_digest"])
        self.assertTrue(dataset.sealed)
        episode = dataset.episodes[0]
        self.assertEqual(episode.source_id, result["source_id"])
        self.assertEqual((episode.steps, len(episode.frames)), (2, 3))
        self.assertEqual(episode.frames[:, 0, 0].tolist(), [10, 11, 12])
        np.testing.assert_allclose(episode.actions, [[1.0, -0.5, 0.25]] * 2)
        np.testing.assert_allclose(episode.applied_actions, [[1.0, 0.25, 0.625]] * 2)
        np.testing.assert_allclose(self.envs[0].actions, episode.applied_actions)
        self.assertEqual(episode.metadata["proposed_native_actions"], [[2.0, -0.5, 0.25]] * 2)
        self.assertAlmostEqual(episode.observation(2)[-1, 0, 0], 12 / 255.0)
        self.assertEqual(episode.finished.tolist(), [False, True])
        self.assertEqual(episode.terminal.tolist(), [False, True])
        self.assertEqual(episode.truncated.tolist(), [False, True])
        self.assertEqual(episode.terminated.tolist(), [False, False])
        self.assertTrue(self.envs[0].closed)
        with self.assertRaises(ValueError):
            episode.frames[0, 0, 0] = 123

        replay, receipt = replay_from_dataset_bytes(
            archive.read_bytes(), archive_sha256=result["archive_sha256"],
            dataset_digest=result["dataset_digest"], source_id=result["source_id"],
            source_actor_sha256=result["source_actor_sha256"],
            allowed_cells=[(cell["track_id"], cell["geometry_seed"])
                           for cell in result["allowed_cells"]],
            excluded_seeds=result["excluded_seeds"], capacity=result["stored_decisions"],
        )
        self.assertEqual(receipt["distinct_finished_cells"], 1)
        batch = replay.sample_sequence(1, 32, short_episode_fraction=1.0)
        self.assertEqual(batch["effective_seq_len"], 2)
        self.assertTrue(batch["is_last"][0, -1].item())
        self.assertTrue(batch["is_terminal"][0, -1].item())
        self.assertAlmostEqual(batch["observations"][0, -1, -1, 0, 0].item(), 12 / 255, places=6)

    def test_real_truncation_is_boundary_but_not_terminal(self):
        self.mocks(builder=lambda t, s, *a, **k: FakeEnv(t, s, ending="timeout"))
        result = self.collect()
        self.assertFalse(result["coverage_pass"])
        episode = TeacherDataset.from_bytes((self.fixture.root / "runs/synthetic-only/collection/support-dataset.npz").read_bytes()).episodes[0]
        self.assertEqual((episode.terminated[-1], episode.truncated[-1], episode.terminal[-1], episode.finished[-1]),
                         (False, True, False, False))
        self.assertEqual(result["episode_rows"][0]["complete"], True)
        self.assertEqual(result["episode_rows"][0]["finished"], False)

    def test_real_finished_timeout_is_terminal_and_truncated(self):
        self.mocks(builder=lambda t, s, *a, **k: FakeEnv(t, s, ending="finished-timeout"))
        result = self.collect()
        episode = TeacherDataset.from_bytes((self.fixture.root / "runs/synthetic-only/collection/support-dataset.npz").read_bytes()).episodes[0]
        self.assertTrue(result["coverage_pass"])
        self.assertEqual((episode.terminated[-1], episode.truncated[-1], episode.terminal[-1], episode.finished[-1]),
                         (False, True, True, True))

    def test_strict_cap_disposes_only_partial_attempt_without_fabricated_finish(self):
        self.fixture.protocol["training_pool"]["episode_schedule"] = [0, 0]
        self.fixture.protocol["budgets"]["decision_cap"] = 3
        self.fixture.write_protocol()

        def builder(track, seed, *args, **kwargs):
            env = FakeEnv(track, seed, length=2 if not self.envs else 4)
            self.envs.append(env)
            return env

        self.mocks(builder=builder)
        result = self.collect()
        self.assertEqual((result["decisions_spent"], result["stored_decisions"], result["discarded_decisions"]),
                         (3, 2, 1))
        self.assertEqual(result["complete_episode_count"], 1)
        self.assertEqual([row["complete"] for row in result["episode_rows"]], [True, False])
        self.assertEqual([row["finished"] for row in result["episode_rows"]], [True, False])
        self.assertEqual(len(TeacherDataset.from_bytes(
            (self.fixture.root / "runs/synthetic-only/collection/support-dataset.npz").read_bytes(),
        ).episodes), 1)
        self.assertTrue(all(env.closed for env in self.envs))

    def test_two_completed_attempts_have_distinct_ids_under_one_source(self):
        self.fixture.protocol["training_pool"]["episode_schedule"] = [0, 0]
        self.fixture.write_protocol()
        self.mocks()
        result = self.collect()
        dataset = TeacherDataset.from_bytes(
            (self.fixture.root / "runs/synthetic-only/collection/support-dataset.npz").read_bytes(),
        )
        self.assertEqual(result["complete_episode_count"], 2)
        self.assertEqual(result["distinct_finished_geometries"], [101])
        self.assertEqual(result["decisions_spent"], 4)
        self.assertEqual([episode.episode_id for episode in dataset.episodes], ["0", "1"])
        self.assertEqual([episode.metadata["collector_episode_id"] for episode in dataset.episodes], [0, 0])
        self.assertTrue(result["schedule_exhausted"])

    def test_cap_with_zero_complete_rows_writes_no_dataset_and_no_false_ending(self):
        self.fixture.protocol["budgets"]["decision_cap"] = 1
        self.fixture.write_protocol()
        self.mocks()
        result = self.collect()
        self.assertEqual(result["stored_decisions"], 0)
        self.assertIsNone(result["archive_sha256"])
        self.assertIsNone(result["dataset_digest"])
        self.assertFalse(result["coverage_pass"])
        self.assertFalse(result["episode_rows"][0]["complete"])
        self.assertFalse((self.fixture.root / "runs/synthetic-only/collection/support-dataset.npz").exists())

    def test_nonfinite_reward_and_missing_max_steps_boundary_fail_before_archive(self):
        class BadRewardEnv(FakeEnv):
            def step(self, official_action):
                observation, _, terminated, truncated, info = super().step(official_action)
                return observation, float("nan"), terminated, truncated, info

        self.mocks(builder=lambda t, s, *a, **k: BadRewardEnv(t, s))
        with self.assertRaisesRegex(ValueError, "raw reward"):
            self.collect()
        self.assertFalse((self.fixture.root / "runs/synthetic-only/collection/support-dataset.npz").exists())

        self.fixture.protocol["max_steps"] = 2
        self.fixture.write_protocol()
        self.mocks(builder=lambda t, s, *a, **k: FakeEnv(t, s, length=4))
        with self.assertRaisesRegex(ValueError, "without a real end or truncation"):
            self.collect(self.fixture.root / "runs/synthetic-only/second-attempt")
        self.assertFalse((self.fixture.root / "runs/synthetic-only/second-attempt/support-dataset.npz").exists())

    def test_refuses_output_collision_before_reset_and_preserves_existing_bytes(self):
        self.mocks()
        output = self.fixture.root / "runs/synthetic-only/collection"
        output.mkdir()
        marker = output / "marker"
        marker.write_bytes(b"do-not-overwrite")
        with self.assertRaises(FileExistsError):
            self.collect(output)
        self.assertEqual(marker.read_bytes(), b"do-not-overwrite")
        self.assertEqual(self.envs, [])

    def test_collector_constructor_failure_closes_synthetic_environment(self):
        class BadCollector:
            def __init__(self, *args, **kwargs):
                raise ValueError("synthetic collector constructor failed")

        self.mocks(collector=BadCollector)
        with self.assertRaisesRegex(ValueError, "constructor failed"):
            self.collect()
        self.assertEqual(len(self.envs), 1)
        self.assertTrue(self.envs[0].closed)
        self.assertFalse((self.fixture.root / "runs/synthetic-only/collection/support-dataset.npz").exists())

    def test_mismatched_action_or_reset_or_frame_stack_rejects_dataset(self):
        for defect in ("applied", "reset", "next_frame", "terminal"):
            with self.subTest(defect=defect):
                class DefectiveCollector(EpisodeCollector):
                    def reset(self, **kwargs):
                        observation, info = super().reset(**kwargs)
                        if defect == "reset":
                            info["seed"] = 999
                        return observation, info

                    def step(self, action):
                        row = super().step(action)
                        if defect == "applied":
                            row.applied_action[1] = 0.99
                        if defect == "next_frame" and row.done:
                            row.next_observation[0] = 0.0
                        if defect == "terminal" and row.step == 0:
                            row.terminal = True
                        return row

                self.mocks(collector=DefectiveCollector)
                output = self.fixture.root / "runs/synthetic-only" / defect
                with self.assertRaisesRegex(ValueError, "action parity|reset did not|frame-aligned|terminal flag"):
                    self.collect(output)
                self.assertFalse((output / "support-dataset.npz").exists())


if __name__ == "__main__":
    unittest.main()
