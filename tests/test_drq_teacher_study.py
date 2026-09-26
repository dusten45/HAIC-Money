from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_drq_teacher_protocol import (
    ProtocolError,
    _agent_drq_compatibility_record,
    _adapter_compatibility_record,
    _evaluation_harness_record,
    validate_protocol,
)


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _agent_source(*, extra_dispatch: bool) -> bytes:
    extension = "        elif model_format == 'other':\n            self.reset(None)\n" if extra_dispatch else ""
    return (
        "class DrQFeatures:\n"
        "    def __init__(self, feature_dim):\n        self.feature_dim = feature_dim\n"
        "class DrQActor:\n"
        "    def __init__(self, feature_dim, hidden_dim):\n"
        "        self.feature_dim = feature_dim\n        self.hidden_dim = hidden_dim\n"
        "class Agent:\n"
        "    def __init__(self, payload):\n"
        "        model_format = payload.get('format')\n"
        "        if model_format == DRQ_ACTOR_FORMAT:\n            self._init_drq(payload)\n"
        + extension
        + "    def _init_drq(self, payload):\n        self._runtime_mode = 'drq'\n"
        "    def reset(self, seed):\n"
        "        if self._runtime_mode == 'drq':\n            return\n"
        "    def act(self, observation):\n"
        "        if self._runtime_mode == 'drq':\n            return self.model(observation)\n"
    ).encode()


def _protocol(root: Path) -> dict:
    source_adapter = (
        "class Transition:\n"
        "    marker = 1\n"
    ).encode()
    study_adapter = (
        "class Transition:\n"
        "    marker = 1\n"
        "    @property\n"
        "    def is_first(self):\n"
        "        return self.step == 0\n"
        "    @property\n"
        "    def is_last(self):\n"
        "        return self.terminated or self.truncated\n"
        "    @property\n"
        "    def is_terminal(self):\n"
        "        return bool(self.terminal)\n"
    ).encode()
    source_agent = _agent_source(extra_dispatch=False)
    study_agent = _agent_source(extra_dispatch=True)
    source_snapshots = {
        "agent.py": _write(root / "agent.py", study_agent),
        "drq_v2.py": _write(root / "drq_v2.py", b"learner source"),
        "common_adapter.py": _write(root / "common_adapter.py", study_adapter),
        "evaluate_policy.py": _write(root / "evaluate_policy.py", b"current evaluator"),
    }
    actors = []
    for learner_seed in (0, 1):
        checkpoint_path = f"source/seed{learner_seed}/checkpoint.pt"
        checkpoint_manifest_path = f"source/seed{learner_seed}/checkpoint.manifest.json"
        source_snapshot_dir = f"source/seed{learner_seed}/source"
        training_source_hashes = {
            "drq_v2.py": _write(root / source_snapshot_dir / "drq_v2.py", b"learner source"),
            "common_adapter.py": _write(root / source_snapshot_dir / "common_adapter.py", source_adapter),
        }
        checkpoint_source_hashes_relative = {
            **training_source_hashes,
            "agent.py": _write(root / source_snapshot_dir / "agent.py", source_agent),
            "evaluate_policy.py": _write(root / source_snapshot_dir / "evaluate_policy.py", b"source evaluator"),
        }
        checkpoint_source_hashes = {
            str((root / source_snapshot_dir / name).resolve()): value
            for name, value in checkpoint_source_hashes_relative.items()
        }
        actor_path = f"source/seed{learner_seed}/actor.pt"
        run_config_path = f"source/seed{learner_seed}/config.json"
        episodes_path = f"source/seed{learner_seed}/episodes.jsonl"
        checkpoint_hash = _write(root / checkpoint_path, f"checkpoint{learner_seed}".encode())
        checkpoint_manifest_hash = _write(root / checkpoint_manifest_path, json.dumps({
            "algorithm": "drq-v2", "source_hashes": checkpoint_source_hashes,
            "extra": {"environment_steps": 131072,
                      "training_source_sha256": training_source_hashes},
        }).encode())
        actor_hash = _write(root / actor_path, f"actor{learner_seed}".encode())
        run_config_hash = _write(root / run_config_path, json.dumps({
            "git": {"commit": "3ea51d6", "dirty": True},
            "config": {
                "algorithm": "drq-v2", "seed": learner_seed, "total_steps": 131072,
                "track_ids": [1, 2, 3, 4],
                "drq_config": {"augmentation_pad": 4},
                "training_source_sha256": training_source_hashes,
            },
        }).encode())
        episode_record = {"track_id": 1, "geometry_seed": 1000 + learner_seed}
        episode_bytes = (json.dumps(episode_record) + "\n").encode()
        episodes_hash = _write(root / episodes_path, episode_bytes)
        actors.append({
            "learner_seed": learner_seed,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_manifest_path": checkpoint_manifest_path,
            "checkpoint_manifest_sha256": checkpoint_manifest_hash,
            "actor_path": actor_path,
            "actor_sha256": actor_hash,
            "run_config_path": run_config_path,
            "run_config_sha256": run_config_hash,
            "episodes_path": episodes_path,
            "episodes_sha256": episodes_hash,
            "source_training_snapshot_dir": source_snapshot_dir,
            "source_training_hashes": training_source_hashes,
            "checkpoint_source_hashes": checkpoint_source_hashes_relative,
            "source_revision": "3ea51d6",
            "source_dirty": True,
            "training_steps": 131072,
            "training_track_ids": [1, 2, 3, 4],
            "training_geometry_seeds": [1000 + learner_seed],
            "source_pair_audit": {
                "learner_seed": learner_seed,
                "checkpoint_sha256": checkpoint_hash,
                "actor_sha256": actor_hash,
                "actor_weights_sha256": "2" * 64,
                "action_fingerprint": "3" * 64,
                "observation_fingerprint": "4" * 64,
                "cpu_smoke_observations": 3,
                "agent_source_parity": {
                    "source_sha256": checkpoint_source_hashes_relative["agent.py"],
                    "study_sha256": source_snapshots["agent.py"],
                    "observations": 3,
                    "actions_exactly_equal": True,
                },
            },
        })

    def matrix(track_ids: list[int], seeds: list[int]) -> dict:
        return {"track_ids": track_ids, "seeds": seeds, "repeats": 2}

    candidate_seeds = sorted({
        2000, 3000,
        *range(4000, 4008), *range(5000, 5008), *range(6000, 6008),
    })
    audit_path = "audit/geometry.json"
    audit_report = {
        "passed": True,
        "parse_errors": [],
        "candidate_seeds": candidate_seeds,
        "candidate_hits": {str(seed): [] for seed in candidate_seeds},
        "known_excluded_geometry_seeds": [7000, 7001],
        "source_snapshot_count": 1,
        "global_freshness_claim": "no known recorded overlap; historical pilot schedules are incomplete",
    }
    audit_hash = _write(root / audit_path, json.dumps(audit_report, sort_keys=True).encode())
    compatibility = sorted([
        _agent_drq_compatibility_record(root / "source/seed0/source/agent.py", root / "agent.py"),
        _adapter_compatibility_record(
            root / "source/seed0/source/common_adapter.py", root / "common_adapter.py"
        ),
        _evaluation_harness_record(
            root / "source/seed0/source/evaluate_policy.py", root / "evaluate_policy.py"
        ),
    ], key=lambda row: row["path"])

    return {
        "name": "drqv2-teacher-replay-v1",
        "purpose": "research",
        "frame_skip": 4,
        "max_steps": 2000,
        "reserved_training_seeds": sorted({
            7000, 7001, 1000, 1001,
            *range(4000, 4008), *range(5000, 5008), *range(6000, 6008),
        }),
        "schema_version": 1,
        "study_id": "drqv2-teacher-replay-v1",
        "run_root": "runs/drqv2-teacher-replay-v1-test",
        "hypothesis": "Teacher replay improves repeated unseen-track completion.",
        "source_revision": "0123456789abcdef",
        "working_tree_dirty": True,
        "working_tree_status_sha256": "5" * 64,
        "spec_fingerprints": {"action": "3" * 64, "observation": "4" * 64},
        "source_snapshots": source_snapshots,
        "source_compatibility": compatibility,
        "source_actors": actors,
        "partitions": {
            "screen": matrix([101, 102, 103], list(range(4000, 4008))),
            "confirmation": matrix([111, 112, 113, 114], list(range(5000, 5008))),
            "blind": matrix([121, 122, 123], list(range(6000, 6008))),
        },
        "training_pools": {
            "teacher_training": {"track_ids": [1, 2, 3, 4], "seeds": [2000], "sampler_seed": 801},
            "online_training": {"track_ids": [1, 2, 3, 4], "seeds": [3000], "sampler_seed": 802},
        },
        "geometry_audit": {
            "report_path": audit_path,
            "report_sha256": audit_hash,
            "candidate_seeds": candidate_seeds,
            "source_snapshot_count": 1,
            "known_excluded_geometry_seed_count": 2,
            "global_freshness_claim": audit_report["global_freshness_claim"],
        },
        "known_excluded_geometry_seeds": [7000, 7001],
        "environment": {
            "observation_shape": [4, 84, 84],
            "observation_dtype": "float32",
            "observation_range": [0.0, 1.0],
            "frame_skip": 4,
            "action_axes": ["steer", "gas", "brake"],
            "native_action_range": [-1.0, 1.0],
            "raw_reward": True,
        },
        "learner": {
            "algorithm": "DrQ-v2", "padding": 4, "feature_dim": 256, "hidden_dim": 256,
            "n_step": 3, "gamma": 0.99, "actor_lr": 0.0001, "critic_lr": 0.0001,
            "tau": 0.01, "target_update_frequency": 2, "actor_update_frequency": 2,
            "target_noise_std": 0.2, "target_noise_clip": 0.5, "steering_logit_l2": 0.0,
            "reward_shaping": False, "reward_normalization": False,
        },
        "budgets": {
            "teacher_decisions_per_source_cap": 16384,
            "additional_online_decisions_per_arm_source": 32768,
            "online_startup_decisions_without_updates": 10000,
            "critic_updates_per_learning_decision": 1,
            "batch_size": 64,
            "teacher_rows_per_treatment_batch": 16,
            "online_rows_per_treatment_batch": 48,
            "checkpoint_online_steps": [16384, 32768],
            "online_replay_capacity": 100000,
            "teacher_replay_capacity_per_source": 16384,
        },
        "runtime": {
            "training_device": "cuda:0",
            "training_executable": "/workspace/HAIC-Money/.venv/bin/python",
            "training_python": "3.11.14",
            "dependencies": {"torch": "2.11.0+cu128"},
            "hardware": {"gpu": "RTX 5070 Ti"},
            "evaluation_executable": "/tmp/kilo/haic-cpu21/bin/python",
            "evaluation_runtime": {"torch": "2.1.0+cpu", "cuda_available": False},
            "worker_limits": {
                "actor_load_seconds_max": 10, "reset_act_seconds_max": 5,
                "peak_rss_bytes_max": 1073741824, "workers": 1,
                "frame_skip": 4, "max_steps": 2000,
            },
        },
        "selection": {
            "screen_cells": 24, "confirmation_cells": 32, "blind_cells": 24,
            "repeats_per_cell": 2, "canonical_repeat": 0,
            "checkpoint_tie_break": "earlier",
            "checkpoint_candidate_steps": [16384, 32768],
            "screen_sort": "completions-desc,canonical-mean-progress-desc,completed-lap-time-asc",
            "screen_requires_nonzero_teacher_finishes_per_source": True,
            "screen_teacher_finishes_at_least_online_control_per_source": True,
            "confirmation_minimum_gain_vs_online_only_per_source": 2,
            "confirmation_at_least_unchanged_source_per_source": True,
            "confirmation_winning_geometry_minimum": 2,
            "promotion_requires_both_source_actors": True,
            "blind_finalist_learner_zero_breaks_exact_tie": True,
            "blind_opens_only_after_confirmation_pass": True,
            "blind_minimum_canonical_finishes": 1,
        },
    }


class DrQTeacherProtocolTests(unittest.TestCase):
    def test_accepts_fresh_geometry_disjoint_from_source_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = validate_protocol(_protocol(root), root)
            self.assertEqual(report["source_actor_count"], 2)
            self.assertEqual(report["partition_cell_counts"]["screen"], 24)

    def test_rejects_screen_geometry_seen_in_source_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _protocol(root)
            protocol["partitions"]["screen"]["seeds"][0] = 1000
            protocol["reserved_training_seeds"].append(1000)
            protocol["reserved_training_seeds"].sort()
            with self.assertRaisesRegex(ProtocolError, "source-training"):
                validate_protocol(protocol, root)

    def test_rejects_evaluation_geometry_in_any_training_track(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _protocol(root)
            protocol["training_pools"]["teacher_training"]["seeds"][0] = 4000
            with self.assertRaisesRegex(ProtocolError, "overlaps evaluation"):
                validate_protocol(protocol, root)

    def test_rejects_reused_seed_across_teacher_and_online_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _protocol(root)
            protocol["training_pools"]["online_training"]["seeds"][0] = 2000
            with self.assertRaisesRegex(ProtocolError, "training geometry pools must be disjoint"):
                validate_protocol(protocol, root)

    def test_rejects_source_snapshot_or_ledger_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _protocol(root)
            protocol["source_snapshots"]["agent.py"] = "0" * 64
            with self.assertRaisesRegex(ProtocolError, "snapshot hash mismatch"):
                validate_protocol(protocol, root)

    def test_rejects_unknown_top_level_or_partition_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _protocol(root)
            protocol["unfrozen_option"] = True
            with self.assertRaisesRegex(ProtocolError, "unsupported top-level"):
                validate_protocol(protocol, root)


if __name__ == "__main__":
    unittest.main()
