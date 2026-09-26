from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import random
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import torch

from common_adapter import ActionAdapter
from drq_v2 import DrQv2Agent, DrQv2Config
from haic.algorithms.drq_v2.teacher_replay import (
    GeometryPoolRNG,
    RNGStreams,
    TeacherDataset,
    TeacherEpisode,
    fork_from_source,
    file_sha256,
    find_sampled_track_env,
    learner_rng_state,
    restore_learner_rng_state,
    update_from_mixture,
    TwoSourceReplay,
)
from scripts.train_drq_teacher_replay import _load_collection


def _small_config() -> DrQv2Config:
    return DrQv2Config(
        feature_dim=8,
        hidden_dim=8,
        replay_capacity=128,
        batch_size=64,
        warmup_steps=8,
        device="cpu",
    )


def _batch(teacher_count: int) -> dict[str, np.ndarray]:
    count = 64
    source = np.zeros(count, dtype=np.int8)
    source[:teacher_count] = 1
    return {
        "observation": np.zeros((count, 4, 84, 84), dtype=np.uint8),
        "next_observation": np.full((count, 4, 84, 84), 128, dtype=np.uint8),
        "action": np.zeros((count, 3), dtype=np.float32),
        "reward": np.linspace(-1.0, 1.0, count, dtype=np.float32),
        "discount": np.full(count, 0.99**3, dtype=np.float32),
        "terminated": np.zeros(count, dtype=np.uint8),
        "truncated": np.zeros(count, dtype=np.uint8),
        "terminal": np.zeros(count, dtype=np.uint8),
        "horizon": np.full(count, 3, dtype=np.int64),
        "source": source,
        "episode_id": np.arange(count, dtype=np.int64),
        "geometry_seed": np.full(count, 1000, dtype=np.int64),
    }


def _collection_protocol_fixture(root: Path) -> tuple[dict, dict, Path]:
    run_dir = root / "runs/test/teacher-data/learner-0"
    run_dir.mkdir(parents=True)
    gate_path = root / "runs/test/a0-collection-gate.json"
    dataset = TeacherDataset()
    actor_sha = "a" * 64
    native_actions = np.zeros((3, 3), dtype=np.float32)
    action_adapter = ActionAdapter()
    dataset.add_episode(TeacherEpisode(
        frames=np.zeros((4, 84, 84), dtype=np.uint8),
        actions=native_actions,
        applied_actions=np.stack([action_adapter.to_official(action) for action in native_actions]),
        rewards=np.ones(3, dtype=np.float32),
        progress=np.linspace(0.1, 0.3, 3, dtype=np.float32),
        damage=np.zeros(3, dtype=np.float32),
        terminated=np.asarray([False, False, True]),
        truncated=np.zeros(3, dtype=np.bool_),
        finished=np.zeros(3, dtype=np.bool_),
        terminal=np.asarray([False, False, True]),
        retire_reasons=[None, None, None],
        episode_id=0,
        source_id="source-learner-0",
        source_actor_sha256=actor_sha,
        geometry_id="100",
        track_id=1,
        complete=True,
        metadata={"synthetic": True},
    ))
    digest = dataset.seal()
    dataset_path = run_dir / "teacher-dataset.npz"
    dataset_path.write_bytes(dataset.to_bytes())
    dataset_file_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    protocol_sha = "b" * 64
    candidates = [100, 101, 102, 103, 200, 201, 202, 203, 300]
    known = [10]
    live = {
        "passed": True,
        "candidate_seeds": candidates,
        "candidate_hits": {str(seed): [] for seed in candidates},
        "parse_errors": [],
        "known_excluded_geometry_seeds": known,
    }
    gate = {
        "format": "haic-drq-teacher-replay-a0-collection-gate-v1",
        "status": "pass",
        "study_id": "drqv2-teacher-replay-v1-r2",
        "study_protocol_sha256": protocol_sha,
        "geometry_audit_path": "experiments/audit.json",
        "geometry_audit_sha256": "c" * 64,
        "live_geometry_audit": live,
    }
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    result_path = run_dir / "collection-result.json"
    result_path.write_text(json.dumps({
        "study_protocol_sha256": protocol_sha,
        "learner_seed": 0,
        "source_actor_sha256": actor_sha,
        "a0_collection_gate_path": gate_path.relative_to(root).as_posix(),
        "a0_collection_gate_sha256": gate_sha,
        "coverage_pass": True,
        "coverage_gate": "pass",
        "finished_geometry_seed_count": 4,
        "decisions": 16384,
        "decision_cap": 16384,
        "dataset_path": dataset_path.relative_to(root).as_posix(),
        "dataset_file_sha256": dataset_file_sha,
        "dataset_digest": digest,
        "training_pool": {"track_ids": [1, 2, 3, 4], "seeds": [100, 101, 102, 103]},
        "finished_geometry_seeds": [100, 101, 102, 103],
    }), encoding="utf-8")
    protocol = {
        "study_id": "drqv2-teacher-replay-v1-r2",
        "geometry_audit": {"report_path": "experiments/audit.json", "report_sha256": "c" * 64},
        "training_pools": {
            "teacher_training": {"seeds": [100, 101, 102, 103]},
            "online_training": {"seeds": [200, 201, 202, 203]},
        },
        "partitions": {"screen": {"seeds": [300]}},
        "known_excluded_geometry_seeds": known,
    }
    source = {"learner_seed": 0, "actor_sha256": actor_sha}
    return protocol, source, result_path


class _FakeOnlineReplay:
    n_step = 3
    gamma = 0.99
    size = 96
    oldest_sequence = 0
    _next_sequence = 96

    def valid_indices(self) -> list[int]:
        return list(range(self.size))

    def _build_n_step(self, sequence: int):
        return {} if sequence < self._next_sequence else None

    def sample(self, batch_size: int, indices: list[int]) -> dict[str, np.ndarray]:
        values = _batch(0)
        for name in ("observation", "next_observation", "action", "reward", "discount",
                     "terminated", "truncated", "terminal", "horizon"):
            values[name] = values[name][:batch_size].copy()
        values["episode_id"] = np.asarray(indices, dtype=np.int64)
        values["sequence"] = np.asarray(indices, dtype=np.int64)
        return {name: values[name] for name in (
            "observation", "next_observation", "action", "reward", "discount",
            "terminated", "truncated", "terminal", "horizon", "episode_id", "sequence"
        )}


class DrQTeacherTrainingTests(unittest.TestCase):
    def test_collection_result_is_bound_to_gate_protocol_and_sealed_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, source, result_path = _collection_protocol_fixture(root)
            protocol_sha = hashlib.sha256(b"frozen protocol").hexdigest()
            # The synthetic result and gate use this same immutable protocol digest.
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["study_protocol_sha256"] = protocol_sha
            result_path.write_text(json.dumps(result), encoding="utf-8")
            gate_path = root / result["a0_collection_gate_path"]
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["study_protocol_sha256"] = protocol_sha
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            result["a0_collection_gate_sha256"] = hashlib.sha256(gate_path.read_bytes()).hexdigest()
            result_path.write_text(json.dumps(result), encoding="utf-8")
            collection, dataset = _load_collection(
                root,
                protocol,
                result_path,
                root / result["dataset_path"],
                protocol_sha,
                source,
                load_dataset=True,
            )
            self.assertTrue(collection["coverage_pass"])
            self.assertEqual(collection["result_sha256"], hashlib.sha256(result_path.read_bytes()).hexdigest())
            self.assertIsNotNone(dataset)
            self.assertEqual(dataset.digest, result["dataset_digest"])
            self.assertEqual(dataset.transition_count, 3)

    def test_weight_only_fork_preserves_all_networks_but_resets_mutable_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = DrQv2Agent(_small_config(), seed=5)
            source.environment_steps = 131072
            checkpoint = source.save_checkpoint(root / "checkpoint.pt")
            actor = source.export_actor(root / "actor.pt")
            fork, identity = fork_from_source(
                checkpoint,
                actor,
                _small_config(),
                learner_seed=0,
                study_seed=55,
                expected_checkpoint_sha256=file_sha256(checkpoint),
                expected_actor_sha256=file_sha256(actor),
            )
            self.assertEqual(identity["source_environment_steps"], 131072)
            self.assertEqual(fork.environment_steps, 131072)
            self.assertEqual(fork.gradient_steps, 0)
            self.assertEqual(fork.replay.size, 0)
            self.assertEqual(fork.actor_optimizer.state, {})
            self.assertEqual(fork.critic_optimizer.state, {})
            for target, expected in (
                (fork.actor, source.actor),
                (fork.critic_one, source.critic_one),
                (fork.critic_two, source.critic_two),
                (fork.target_one, source.target_one),
                (fork.target_two, source.target_two),
            ):
                for actual_value, expected_value in zip(target.state_dict().values(), expected.state_dict().values()):
                    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)

    def test_weight_only_fork_rejects_checkpoint_or_actor_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = DrQv2Agent(_small_config(), seed=6)
            source.environment_steps = 131072
            checkpoint = source.save_checkpoint(root / "checkpoint.pt")
            actor = source.export_actor(root / "actor.pt")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
                fork_from_source(checkpoint, actor, _small_config(), learner_seed=0,
                                 study_seed=1, expected_checkpoint_sha256="0" * 64,
                                 expected_actor_sha256=file_sha256(actor))

    def test_mixture_update_uses_separate_rngs_and_fixed_quotas(self) -> None:
        agent = DrQv2Agent(_small_config(), seed=7)
        rng = RNGStreams(70, torch.device("cpu"))
        batch = _batch(16)
        torch_before = torch.get_rng_state().clone()
        metrics = update_from_mixture(agent, batch, rng)
        self.assertEqual(agent.gradient_steps, 1)
        self.assertEqual(metrics["sampled_teacher_count"], 16.0)
        self.assertEqual(metrics["sampled_online_count"], 48.0)
        self.assertEqual(metrics["sampled_teacher_fraction"], 0.25)
        self.assertIn("td_error_teacher", metrics)
        self.assertEqual(metrics["actor_updated"], 0.0)
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        second = update_from_mixture(agent, batch, rng)
        self.assertEqual(agent.gradient_steps, 2)
        self.assertEqual(second["actor_updated"], 1.0)
        self.assertIn("teacher_state_actor_abs_delta_mean", second)
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))

    def test_control_mixture_is_exactly_online_only(self) -> None:
        agent = DrQv2Agent(_small_config(), seed=8)
        metrics = update_from_mixture(agent, _batch(0), RNGStreams(80, torch.device("cpu")))
        self.assertEqual(metrics["sampled_teacher_count"], 0.0)
        self.assertEqual(metrics["sampled_online_count"], 64.0)
        self.assertEqual(metrics["sampled_teacher_fraction"], 0.0)
        self.assertNotIn("td_error_teacher", metrics)

    def test_source_checkpoint_retains_final_declared_exploration_noise(self) -> None:
        agent = DrQv2Agent(_small_config(), seed=80)
        agent.environment_steps = 131072
        self.assertAlmostEqual(agent.exploration_std(), 0.05)

    def test_mixture_update_rejects_wrong_teacher_quota(self) -> None:
        agent = DrQv2Agent(_small_config(), seed=9)
        with self.assertRaisesRegex(ValueError, "exactly online-only"):
            update_from_mixture(agent, _batch(15), RNGStreams(90, torch.device("cpu")))

    def test_geometry_pool_rng_cycles_without_replacement_and_restores(self) -> None:
        seeds = [100, 200, 300, 400]
        rng = GeometryPoolRNG([1, 2, 3, 4], seeds, seed=42)
        first_cycle = [rng.integers(0, 2**32, dtype=np.uint64) for _ in seeds]
        self.assertEqual(set(first_cycle), set(seeds))
        state = rng.state_dict()
        expected_next = rng.integers(0, 2**32, dtype=np.uint64)
        restored = GeometryPoolRNG([1, 2, 3, 4], seeds, seed=100)
        restored.load_state_dict(state)
        self.assertEqual(restored.integers(0, 2**32, dtype=np.uint64), expected_next)
        self.assertIn(int(rng.choice([1, 2, 3, 4])), [1, 2, 3, 4])

    def test_sampler_injection_walks_wrappers_without_unwrapping_environment(self) -> None:
        sampler = SimpleNamespace(_track_ids=(1, 2, 3, 4), _rng=object(), _excluded_seeds=frozenset())
        environment = SimpleNamespace(env=SimpleNamespace(env=sampler))
        self.assertIs(find_sampled_track_env(environment), sampler)
        with self.assertRaisesRegex(RuntimeError, "no audited SampledHaicTrack"):
            find_sampled_track_env(SimpleNamespace(env=object()))

    def test_two_source_sampler_checkpoint_restores_rng_and_valid_rows(self) -> None:
        replay = TwoSourceReplay(_FakeOnlineReplay(), None, mode="online-only", seed=21)
        replay.sample()
        state = replay.state_dict()
        expected = replay.sample()
        restored = TwoSourceReplay(_FakeOnlineReplay(), None, mode="online-only", seed=999)
        restored.load_state_dict(state)
        actual = restored.sample()
        np.testing.assert_array_equal(actual["source_indices"], expected["source_indices"])
        np.testing.assert_array_equal(actual["source"], expected["source"])

    def test_full_episode_boundary_checkpoint_restores_all_independent_rng_streams(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _small_config()
            agent = DrQv2Agent(config, seed=100)
            sampler = GeometryPoolRNG([1, 2, 3, 4], [100, 200, 300, 400], seed=101)
            mixture = TwoSourceReplay(agent.replay, None, mode="online-only", seed=102)
            rng = RNGStreams(103, torch.device("cpu"))
            trainer_state = {
                "format": "haic-drq-teacher-trainer-v1",
                "resume_allowed": True,
                "checkpoint_at_episode_boundary": True,
                "next_episode_id": 1,
                "learner_update_rng": rng.state_dict(),
                "geometry_pool_rng": sampler.state_dict(),
                "two_source_replay": mixture.state_dict(),
                "global_learner_rng": learner_rng_state(agent),
            }
            checkpoint = agent.save_checkpoint(root / "boundary.pt", trainer_state=trainer_state)

            random_expected = random.random()
            torch_expected = torch.rand(5)
            action_rng_expected = agent.rng.random()
            replay_rng_expected = agent.replay.rng.random()
            mixture_rng_expected = mixture.rng.random()
            geometry_expected = sampler.integers(0, 2**32, dtype=np.uint64)
            observation = torch.arange(2 * 4 * 84 * 84, dtype=torch.float32).reshape(2, 4, 84, 84)
            shift_expected = rng.shift(observation, "critic_current", pad=4)

            restored = DrQv2Agent(config, seed=999)
            restored_state = restored.load_checkpoint(checkpoint)
            restored_sampler = GeometryPoolRNG([1, 2, 3, 4], [100, 200, 300, 400], seed=999)
            restored_sampler.load_state_dict(restored_state["geometry_pool_rng"])
            restored_mixture = TwoSourceReplay(restored.replay, None, mode="online-only", seed=999)
            restored_mixture.load_state_dict(restored_state["two_source_replay"])
            restored_rng = RNGStreams(999, torch.device("cpu"))
            restored_rng.load_state_dict(restored_state["learner_update_rng"])
            restore_learner_rng_state(restored, restored_state["global_learner_rng"])

            self.assertEqual(random.random(), random_expected)
            torch.testing.assert_close(torch.rand(5), torch_expected, rtol=0, atol=0)
            self.assertEqual(restored.rng.random(), action_rng_expected)
            self.assertEqual(restored.replay.rng.random(), replay_rng_expected)
            self.assertEqual(restored_mixture.rng.random(), mixture_rng_expected)
            self.assertEqual(restored_sampler.integers(0, 2**32, dtype=np.uint64), geometry_expected)
            torch.testing.assert_close(
                restored_rng.shift(observation, "critic_current", pad=4), shift_expected, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
