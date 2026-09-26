from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from common_adapter import Transition
from drq_v2 import DrQv2Agent, DrQv2Config, Uint8Replay
from haic.algorithms.drq_v2.teacher_replay import TwoSourceReplay
from haic.algorithms.drq_v2.teacher_study import learner_rng_state, restore_learner_rng_state
from scripts import train_drq_geometry_mix as trainer


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _protocol() -> dict:
    variants = {
        name: {"family_tiers": dict(tiers), "tier_weights": dict(weights)}
        for name, (tiers, weights) in trainer.EXPECTED_VARIANT_DISTRIBUTIONS.items()
    }
    runs = []
    for source_seed in (0, 1):
        for variant_index, variant in enumerate(trainer.VARIANTS):
            base = 500_000 + source_seed * 10_000 + variant_index * 100
            runs.append({
                "source_seed": source_seed,
                "variant": variant,
                "run_dir": f"runs/geometry-mix-test/learner-{source_seed}-{variant}",
                "track_seed": base + 1,
                "geometry_seed": base + 2,
                "actor_rng_seed": base + 3,
                "replay_rng_seed": base + 4,
                "target_noise_seed": base + 5,
                "update_rng_seed": base + 6,
            })
    learner = {
        "algorithm": "DrQ-v2",
        **trainer.EXPECTED_LEARNER,
        "reward_shaping": False,
        "reward_normalization": False,
    }
    actors = [
        {
            "learner_seed": source_seed,
            "source_revision": f"source-revision-{source_seed}",
            "checkpoint_path": f"runs/source-{source_seed}/checkpoint.pt",
            "checkpoint_sha256": f"{source_seed + 1:064x}",
            "checkpoint_manifest_path": f"runs/source-{source_seed}/checkpoint.manifest.json",
            "checkpoint_manifest_sha256": f"{source_seed + 11:064x}",
            "actor_path": f"runs/source-{source_seed}/actor.pt",
            "actor_sha256": f"{source_seed + 21:064x}",
        }
        for source_seed in (0, 1)
    ]
    return {
        "format": trainer.PROTOCOL_FORMAT,
        "study_id": trainer.STUDY_ID,
        "run_root": "runs/geometry-mix-test",
        "catalog": {
            "path": "runs/catalog/catalog.json",
            "sha256": "",
            "protocol_path": "experiments/catalog-protocol.json",
            "protocol_sha256": "",
        },
        "environment": {
            **trainer.EXPECTED_ENVIRONMENT,
            "max_steps": 2000,
            "obstacles": True,
            "observation_shape": [4, 84, 84],
        },
        "training_pool": {
            "partition": "TRAIN",
            "geometry_seeds": [
                100_000 + family_index * 100 + row_index
                for family_index in range(len(trainer.FAMILIES))
                for row_index in range(20)
            ],
            "track_ids": [1, 2, 3, 4],
        },
        "budgets": dict(trainer.EXPECTED_BUDGETS),
        "learner": learner,
        "runtime": {"training_device": "cpu"},
        "source_actors": actors,
        "variants": variants,
        "runs": runs,
    }


def _catalog_fixture(root: Path, protocol: dict) -> tuple[dict, set[int], set[int]]:
    family_rules = [{"name": name} for name in trainer.FAMILIES]
    catalog_protocol = {
        "format": "haic-drq-training-geometry-protocol-v1",
        "train_target": 120,
        "diagnostic_target": 16,
        "family_rules": family_rules,
    }
    generation_path = root / protocol["catalog"]["protocol_path"]
    generation_path.parent.mkdir(parents=True, exist_ok=True)
    generation_raw = json.dumps(catalog_protocol, sort_keys=True).encode()
    generation_path.write_bytes(generation_raw)
    train = []
    for family_index, family in enumerate(trainer.FAMILIES):
        for row_index in range(20):
            train.append({
                "geometry_seed": 100_000 + family_index * 100 + row_index,
                "track_id": 1,
                "family": family,
                "stage": "representative" if row_index < 10 else "variant",
            })
    diagnostic = [
        {
            "geometry_seed": 200_000 + index,
            "track_id": 1,
            "family": trainer.FAMILIES[index % len(trainer.FAMILIES)],
            "stage": "diagnostic",
        }
        for index in range(16)
    ]
    catalog = {
        "format": "haic-drq-training-geometry-catalog-v1",
        "protocol_sha256": _sha(generation_raw),
        "train": train,
        "train_diagnostic": diagnostic,
    }
    catalog_path = root / protocol["catalog"]["path"]
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_raw = json.dumps(catalog, sort_keys=True).encode()
    catalog_path.write_bytes(catalog_raw)
    protocol["catalog"]["sha256"] = _sha(catalog_raw)
    protocol["catalog"]["protocol_sha256"] = _sha(generation_raw)
    return protocol, {row["geometry_seed"] for row in train}, {
        row["geometry_seed"] for row in diagnostic
    }


def test_protocol_fixes_all_three_distributions_and_drq_training_budget():
    protocol = _protocol()
    trainer._validate_protocol(protocol)
    config = trainer._agent_config(protocol)
    assert config.observation_shape == (4, 84, 84)
    assert config.replay_capacity == 100_000
    assert config.batch_size == 64
    assert config.n_step == 3
    assert config.augmentation_pad == 4
    assert config.actor_learning_rate == config.critic_learning_rate == 1e-4
    assert config.target_policy_noise == 0.2
    assert config.target_policy_noise_clip == 0.5

    protocol["variants"]["failure_weighted"]["tier_weights"]["difficult"] = 0.56
    with pytest.raises(ValueError, match="tier weights changed"):
        trainer._validate_protocol(protocol)


def test_catalog_index_returns_train_only_and_sampler_never_receives_seed_pools(tmp_path):
    protocol, train_seeds, diagnostic_seeds = _catalog_fixture(tmp_path, _protocol())
    allowed, families = trainer._validate_catalog(protocol, tmp_path)
    checkpoint_catalog, checkpoint_families = trainer._catalog_train_index(tmp_path, protocol)
    assert set(allowed.values()) == train_seeds
    assert checkpoint_catalog == allowed
    assert checkpoint_families == families
    assert set(allowed.values()).isdisjoint(diagnostic_seeds)
    assert set(families) == set(trainer.FAMILIES)

    arm = trainer._arm(protocol, 0, "uniform")
    with patch.object(trainer, "build_catalog_env", return_value=object()) as build:
        env = trainer._build_training_env(protocol, tmp_path, "uniform", arm)
    assert env is build.return_value
    kwargs = build.call_args.kwargs
    assert kwargs["track_ids"] == [1, 2, 3, 4]
    assert kwargs["expected_catalog_sha256"] == protocol["catalog"]["sha256"]
    assert kwargs["expected_protocol_sha256"] == protocol["catalog"]["protocol_sha256"]
    assert kwargs["family_tiers"] == protocol["variants"]["uniform"]["family_tiers"]
    assert kwargs["tier_weights"] == protocol["variants"]["uniform"]["tier_weights"]
    assert not (set(kwargs) & {"train", "train_diagnostic", "seeds", "reserved_training_seeds"})
    assert kwargs["geometry_seed"] not in train_seeds | diagnostic_seeds
    assert kwargs["geometry_seed"] not in {10_001, 10_002, 10_003}  # prior baseline IDs


def test_update_schedule_has_exact_startup_and_one_update_per_later_decision():
    warmup = trainer.EXPECTED_BUDGETS["warmup_steps"]
    total = trainer.EXPECTED_BUDGETS["additional_online_steps"]
    scheduled = [step for step in range(1, total + 1) if trainer._updates_due(step, warmup)]
    assert len(scheduled) == total - warmup == 22_768
    assert scheduled[0] == warmup + 1
    assert scheduled[-1] == total
    assert [trainer._expected_gradient_steps(step, warmup) for step in (1, warmup, warmup + 1, total)] == [
        0, 0, 1, 22_768,
    ]

    class FakeReplay:
        gamma, n_step, size, oldest_sequence, _next_sequence = 0.99, 3, 96, 0, 96

        def valid_indices(self):
            return list(range(self.size))

        def _build_n_step(self, sequence):
            return {} if sequence < self._next_sequence else None

        def sample(self, batch_size, *, indices):
            return {
                "observation": np.zeros((batch_size, 4, 84, 84), dtype=np.uint8),
                "next_observation": np.zeros((batch_size, 4, 84, 84), dtype=np.uint8),
                "action": np.zeros((batch_size, 3), dtype=np.float32),
                "reward": np.zeros(batch_size, dtype=np.float32),
                "terminated": np.zeros(batch_size, dtype=np.uint8),
                "truncated": np.zeros(batch_size, dtype=np.uint8),
                "terminal": np.zeros(batch_size, dtype=np.uint8),
                "discount": np.full(batch_size, .99**3, dtype=np.float32),
                "horizon": np.full(batch_size, 3, dtype=np.uint8),
                "episode_id": np.asarray(indices, dtype=np.int64),
                "sequence": np.asarray(indices, dtype=np.int64),
            }

    batch = trainer._online_batch(
        TwoSourceReplay(FakeReplay(), None, mode="online-only", seed=19), 64
    )
    assert batch["source"].shape == (64,)
    assert np.count_nonzero(batch["source"]) == 0
    assert len(batch["episode_id"]) == len(batch["sequence"]) == 64


def test_checkpoint_resume_guard_rejects_mid_episode_and_completed_checkpoints():
    trainer._require_episode_boundary_resume({
        "resume_allowed": True,
        "checkpoint_at_episode_boundary": True,
        "next_episode_id": 4,
    })
    for state in (
        {"resume_allowed": False, "checkpoint_at_episode_boundary": True, "next_episode_id": 4},
        {"resume_allowed": True, "checkpoint_at_episode_boundary": False, "next_episode_id": 4},
        {"resume_allowed": True, "checkpoint_at_episode_boundary": True, "next_episode_id": None},
    ):
        with pytest.raises(ValueError, match="episode boundary"):
            trainer._require_episode_boundary_resume(state)


def test_full_checkpoint_restores_independent_actor_replay_target_and_update_rngs(tmp_path):
    config = DrQv2Config(
        feature_dim=8,
        hidden_dim=8,
        replay_capacity=128,
        batch_size=64,
        warmup_steps=8,
        device="cpu",
    )
    arm = {
        "actor_rng_seed": 101,
        "replay_rng_seed": 102,
        "target_noise_seed": 103,
        "update_rng_seed": 104,
        "track_seed": 105,
        "geometry_seed": 106,
    }
    agent = DrQv2Agent(config, seed=11)
    agent.environment_steps = 131_072
    trainer._assert_online_only_fork(agent)
    streams = trainer._make_update_rngs(agent, arm)
    sampler_state = {"episode_count": 7, "schedule": [11, 12, 13]}
    state = {
        "format": trainer.TRAINER_FORMAT,
        "resume_allowed": True,
        "checkpoint_at_episode_boundary": True,
        "next_episode_id": 8,
        "catalog_sampler_state": sampler_state,
        "rng_streams": streams.state_dict(),
        "global_learner_rng": learner_rng_state(agent),
    }
    checkpoint = agent.save_checkpoint(tmp_path / "checkpoint.pt", trainer_state=state)

    expected_actor = agent.rng.normal(size=4)
    expected_replay = agent.replay.rng.random(4)
    expected_target = torch.rand(4, generator=streams.target_noise)
    expected_update = streams.shift_rngs["critic_current"].random(4)
    expected_global_torch = torch.rand(4)

    restored = DrQv2Agent(config, seed=999)
    restored_streams = trainer._make_update_rngs(restored, {**arm, "actor_rng_seed": 999})
    restored_state = restored.load_checkpoint(checkpoint)
    restored_streams.load_state_dict(restored_state["rng_streams"])
    restore_learner_rng_state(restored, restored_state["global_learner_rng"])

    np.testing.assert_array_equal(restored.rng.normal(size=4), expected_actor)
    np.testing.assert_array_equal(restored.replay.rng.random(4), expected_replay)
    torch.testing.assert_close(
        torch.rand(4, generator=restored_streams.target_noise), expected_target, rtol=0, atol=0
    )
    np.testing.assert_array_equal(
        restored_streams.shift_rngs["critic_current"].random(4), expected_update
    )
    torch.testing.assert_close(torch.rand(4), expected_global_torch, rtol=0, atol=0)
    assert restored_state["catalog_sampler_state"] == sampler_state


def test_episode_ledger_record_contains_catalog_identity_fields():
    transition = SimpleNamespace(
        episode_id=3,
        step=8,
        reward=-0.25,
        terminated=False,
        truncated=True,
        terminal=False,
        info={
            "track_id": 2,
            "seed": 12345,
            "geometry_seed": 12345,
            "geometry_family": trainer.FAMILY_ANCHOR,
            "finished": False,
            "progress": 0.6,
            "damage": 0.0,
            "retire_reason": None,
        },
    )
    record = trainer._episode_event(
        transition, [np.zeros(3, dtype=np.float32) for _ in range(9)], -2.25, 0,
        "easy_retention", 900,
    )
    assert record["event"] == "end"
    assert record["seed"] == 12345
    assert record["geometry_family"] == trainer.FAMILY_ANCHOR
    assert record["source_seed"] == 0
    assert record["variant"] == "easy_retention"


def test_online_only_incremental_sampler_matches_valid_rows_through_ring_wrap():
    replay = Uint8Replay(capacity=128, n_step=3, gamma=0.99, seed=33)
    sampler = TwoSourceReplay(replay, None, mode="online-only", seed=44)
    for index in range(300):
        episode_id, step = divmod(index, 8)
        terminated = step == 7 and episode_id % 2 == 0
        truncated = step == 7 and episode_id % 2 == 1
        current = np.full((4, 84, 84), index % 255, dtype=np.uint8)
        following = np.full((4, 84, 84), (index + 1) % 255, dtype=np.uint8)
        replay.add(Transition(
            observation=current,
            action=np.zeros(3, dtype=np.float32),
            reward=float(index % 3),
            next_observation=following,
            terminated=terminated,
            truncated=truncated,
            terminal=terminated,
            info={},
            episode_id=episode_id,
            step=step,
            applied_action=np.zeros(3, dtype=np.float32),
        ))
        cached = sampler._refresh_online_valid_indices()
        assert cached == replay.valid_indices()
        if len(cached) >= 64:
            batch = sampler.sample(64)
            assert len(batch["source_indices"]) == 64
            assert np.count_nonzero(batch["source"]) == 0
            assert len(batch["source_tags"]) == 64
