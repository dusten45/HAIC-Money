import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.rlpd_common import (
    DATASET_FORMAT,
    ROOT,
    canonical_sha256,
    geometry_seeds_from_ledger,
    load_offline_replay,
    read_protocol,
    sha256_file,
    verify_dataset,
    write_json,
)
from scripts.run_rlpd_pilot import _rank_for_run
from scripts.run_rlpd_followup import _finalist_screen_rank


TEACHER_ACTOR = "c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954"
TEACHER_CHECKPOINT = "c806c0998d4a9241917dd368581b9625e293c16ff45e0e1fb7bd5759dd20b3d4"


def study_protocol():
    future_training_seeds = list(range(4001, 4065))
    return {
        "format": "haic-pixel-rlpd-study-v1",
        "name": "test-rlpd-pilot",
        "status": "frozen",
        "frame_skip": 4,
        "max_steps": 2000,
        "training_track_ids": [1, 2, 3, 4],
        "training_geometry_seeds": [2001, 2002],
        "teacher_data_cells": [
            {"track_id": 1, "geometry_seed": 2001, "obstacles": True},
            {"track_id": 2, "geometry_seed": 2002, "obstacles": True},
        ],
        "teacher_data_budget": {"decisions": 8192, "minimum_distinct_finishes": 2},
        "student_training": {
            "learner_seeds": [0, 1],
            "steps_per_run": 16384,
            "candidate_steps": [8192, 16384],
            "first_update_step": 1000,
            "random_decisions": 1000,
            "policy_takeover_step": 2000,
            "updates_per_decision": 1,
            "expected_gradient_steps_per_run": 15384,
            "batch_size": 64,
            "offline_batch_size": 32,
            "online_batch_size": 32,
            "backup_entropy": False,
            "replay_capacity": 100000,
            "offline_capacity": 8192,
            "learner": {
                "actor_lr": 0.0003,
                "critic_lr": 0.0003,
                "temperature_lr": 0.0003,
                "gamma": 0.99,
                "tau": 0.005,
                "num_qs": 10,
                "num_min_qs": 1,
                "target_entropy": -1.5,
                "initial_alpha": 0.1,
                "augmentation_pad": 4,
                "n_step": 1,
                "weight_decay": 0.0,
            },
        },
        "partitions": {
            "screen": {"track_ids": [101], "seeds": [3001], "repeats": 2},
            "confirmation": {"track_ids": [111], "seeds": [3011], "repeats": 2},
            "blind": {"track_ids": [121], "seeds": [3021], "repeats": 2},
        },
        "future_full_reservation": {
            "status": "reserved_not_opened; requires new full protocol and successful pilot gate",
            "training_track_ids": [1, 2, 3, 4],
            "training_geometry_seeds": future_training_seeds,
            "teacher_data_cells": [
                {"track_id": 1 + index % 4, "geometry_seed": seed, "obstacles": True}
                for index, seed in enumerate(future_training_seeds)
            ],
            "teacher_data_budget": {"decisions": 16384, "minimum_distinct_finishes": 4},
            "screen": {"track_ids": [131], "seeds": [3031], "repeats": 2},
            "confirmation": {"track_ids": [141], "seeds": [3041], "repeats": 2},
            "blind": {"track_ids": [151], "seeds": [3051], "repeats": 2},
        },
        "reserved_training_seeds": [3001, 3011, 3021, 3031, 3041, 3051],
        "evaluation": {"screen_candidate_steps": [8192, 16384]},
        "teacher": {
            "actor_sha256": TEACHER_ACTOR,
            "checkpoint_sha256": TEACHER_CHECKPOINT,
        },
        "geometry_audit": {
            "candidate_seed_token_audit": {
                "known_recorded_uses": [],
                "candidate_seed_count": 72,
                "candidate_seeds_sha256": canonical_sha256(
                    sorted([2001, 2002, 3001, 3011, 3021, 3031, 3041, 3051] + future_training_seeds)
                ),
                "searched_artifacts_sha256": "0" * 64,
            },
            "teacher_source_ledgers": [{
                "path": "runs/20260922-drq-augmentation-pad-v1-restart/control-seed1/episodes.jsonl",
                "sha256": sha256_file(
                    ROOT / "runs/20260922-drq-augmentation-pad-v1-restart/control-seed1/episodes.jsonl"
                ),
                "geometry_seed_set_sha256": canonical_sha256(sorted(geometry_seeds_from_ledger(
                    ROOT / "runs/20260922-drq-augmentation-pad-v1-restart/control-seed1/episodes.jsonl"
                ))),
            }],
        },
        "source_hashes": {"requirements.txt": sha256_file(ROOT / "requirements.txt")},
    }


def followup_protocol():
    protocol = study_protocol()
    training_seeds = list(range(4_000_012_001, 4_000_012_065))
    pilot_eval_seeds = sorted(
        list(range(4_000_013_001, 4_000_013_009))
        + list(range(4_000_013_011, 4_000_013_019))
        + list(range(4_000_013_021, 4_000_013_029))
    )
    protocol["name"] = "pixel-rlpd-long-horizon-followup-test"
    protocol["training_geometry_seeds"] = training_seeds
    protocol["teacher_data_cells"] = [
        {"track_id": 1 + index % 4, "geometry_seed": seed, "obstacles": True}
        for index, seed in enumerate(training_seeds)
    ]
    protocol["teacher_data_budget"] = {"decisions": 16384, "minimum_distinct_finishes": 4}
    protocol["student_training"].update({
        "steps_per_run": 131072,
        "candidate_steps": [65536, 131072],
        "expected_gradient_steps_per_run": 130072,
        "offline_capacity": 16384,
    })
    protocol["partitions"] = {
        "screen": {"track_ids": [201, 202, 203], "seeds": list(range(4_000_013_001, 4_000_013_009)), "repeats": 2},
        "confirmation": {"track_ids": [211, 212, 213, 214], "seeds": list(range(4_000_013_011, 4_000_013_019)), "repeats": 2},
        "blind": {"track_ids": [221, 222, 223], "seeds": list(range(4_000_013_021, 4_000_013_029)), "repeats": 2},
    }
    protocol["reserved_training_seeds"] = sorted(set(pilot_eval_seeds + [
        4_000_000_000 + value for value in range(1, 100)
    ]))
    protocol.pop("future_full_reservation")
    return protocol


def pixel_stack(value):
    return np.full((4, 84, 84), value, dtype=np.uint8)


def make_episode(directory: Path, *, episode_id: int, track_id: int, seed: int, finished=True):
    path = directory / "episodes" / f"episode-{episode_id:04d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.full((1, 84, 84), 20 + episode_id, dtype=np.uint8)
    np.savez_compressed(
        path,
        frames=frames,
        initial_observation=np.repeat(frames[0:1], 4, axis=0),
        final_observation=pixel_stack(90 + episode_id),
        proposed_actions=np.zeros((1, 3), dtype=np.float32),
        executed_actions=np.zeros((1, 3), dtype=np.float32),
        applied_actions=np.zeros((1, 3), dtype=np.float32),
        rewards=np.asarray([1.0], dtype=np.float32),
        terminated=np.asarray([finished], dtype=np.bool_),
        truncated=np.asarray([not finished], dtype=np.bool_),
        terminal=np.asarray([finished], dtype=np.bool_),
    )
    return {
        "episode_id": episode_id,
        "path": f"episodes/{path.name}",
        "sha256": sha256_file(path),
        "track_id": track_id,
        "geometry_seed": seed,
        "steps": 1,
        "finished": finished,
    }


def write_dataset(directory: Path, protocol_path: Path, protocol: dict):
    episodes = [
        make_episode(directory, episode_id=0, track_id=1, seed=2001),
        make_episode(directory, episode_id=1, track_id=2, seed=2002),
    ]
    manifest = {
        "format": DATASET_FORMAT,
        "study_id": protocol["name"],
        "study_protocol_sha256": sha256_file(protocol_path),
        "teacher_actor_sha256": TEACHER_ACTOR,
        "teacher_checkpoint_sha256": TEACHER_CHECKPOINT,
        "decisions_spent": 2,
        "stored_decisions": 2,
        "discarded_decisions": 0,
        "distinct_finish_geometries": 2,
        "episodes": episodes,
    }
    manifest["dataset_sha256"] = canonical_sha256(manifest)
    write_json(directory / "manifest.json", manifest)
    return manifest


class TestRLPDRunContracts(unittest.TestCase):
    def test_blind_finalist_is_fixed_from_screen_before_confirmation(self):
        selected = {
            "rlpd-seed0": {
                "canonical_finishes": 1,
                "avg_progress": 0.5,
                "avg_lap_time_ms": 42000.0,
                "environment_steps": 65536,
                "training_seed": 0,
                "actor_sha256": "a" * 64,
                "arm": "rlpd",
            },
            "rlpd-seed1": {
                "canonical_finishes": 1,
                "avg_progress": 0.6,
                "avg_lap_time_ms": 41000.0,
                "environment_steps": 131072,
                "training_seed": 1,
                "actor_sha256": "b" * 64,
                "arm": "rlpd",
            },
        }
        finalist = _finalist_screen_rank(selected, [0, 1])
        self.assertEqual(finalist["actor_sha256"], "b" * 64)
        selected["rlpd-seed1"].update({
            "avg_progress": 0.5,
            "avg_lap_time_ms": 42000.0,
            "environment_steps": 65536,
        })
        finalist = _finalist_screen_rank(selected, [0, 1])
        self.assertEqual(finalist["training_seed"], 0)

    def test_screen_selection_is_canonical_and_exact_ties_keep_earlier_checkpoint(self):
        candidates = [
            {"candidate_id": "early", "archive_sha256": "a" * 64, "source_path": "early.pt"},
            {"candidate_id": "late", "archive_sha256": "b" * 64, "source_path": "late.pt"},
        ]
        summaries = [
            {
                "candidate_id": name,
                "eligible": True,
                "determinism_audited": True,
                "cpu_reload_matches": True,
                "operational_failures": 0,
                "canonical_episodes": 12,
                "summary": {"avg_progress": 0.4, "avg_lap_time_ms": None},
            }
            for name in ("early", "late")
        ]
        identities = {
            "a" * 64: {"arm": "rlpd", "seed": 0, "environment_steps": 8192, "checkpoint_sha256": "1" * 64},
            "b" * 64: {"arm": "rlpd", "seed": 0, "environment_steps": 16384, "checkpoint_sha256": "2" * 64},
        }
        selected = _rank_for_run(
            candidates,
            summaries,
            {"early": 2, "late": 2},
            {"arm": "rlpd", "seed": 0},
            identities,
        )
        self.assertEqual(selected["environment_steps"], 8192)
        self.assertEqual(selected["learner_checkpoint_sha256"], "1" * 64)

        summaries[1]["summary"]["avg_progress"] = 0.5
        selected = _rank_for_run(
            candidates,
            summaries,
            {"early": 2, "late": 2},
            {"arm": "rlpd", "seed": 0},
            identities,
        )
        self.assertEqual(selected["environment_steps"], 16384)

    def test_screen_selection_prioritizes_finish_count_before_progress(self):
        candidates = [
            {"candidate_id": "early", "archive_sha256": "c" * 64, "source_path": "early.pt"},
            {"candidate_id": "late", "archive_sha256": "d" * 64, "source_path": "late.pt"},
        ]
        summaries = [
            {
                "candidate_id": name,
                "eligible": True,
                "determinism_audited": True,
                "cpu_reload_matches": True,
                "operational_failures": 0,
                "canonical_episodes": 12,
                "summary": {"avg_progress": progress, "avg_lap_time_ms": None},
            }
            for name, progress in (("early", 0.9), ("late", 0.1))
        ]
        identities = {
            "c" * 64: {"arm": "sac", "seed": 1, "environment_steps": 8192, "checkpoint_sha256": "3" * 64},
            "d" * 64: {"arm": "sac", "seed": 1, "environment_steps": 16384, "checkpoint_sha256": "4" * 64},
        }
        selected = _rank_for_run(
            candidates,
            summaries,
            {"early": 0, "late": 1},
            {"arm": "sac", "seed": 1},
            identities,
        )
        self.assertEqual(selected["environment_steps"], 16384)

    def test_screen_selection_uses_protocol_specific_canonical_cell_count(self):
        candidates = [{"candidate_id": "full-screen", "archive_sha256": "e" * 64, "source_path": "full.pt"}]
        summaries = [{
            "candidate_id": "full-screen",
            "eligible": True,
            "determinism_audited": True,
            "cpu_reload_matches": True,
            "operational_failures": 0,
            "canonical_episodes": 24,
            "summary": {"avg_progress": 0.2, "avg_lap_time_ms": None},
        }]
        identity = {
            "e" * 64: {
                "arm": "rlpd",
                "seed": 10,
                "environment_steps": 65536,
                "checkpoint_sha256": "5" * 64,
            }
        }
        selected = _rank_for_run(
            candidates,
            summaries,
            {"full-screen": 1},
            {"arm": "rlpd", "seed": 10},
            identity,
            expected_canonical_episodes=24,
        )
        self.assertEqual(selected["canonical_finishes"], 1)
        self.assertEqual(selected["environment_steps"], 65536)

    def test_frozen_protocol_accepts_disjoint_uint32_partitions_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "study.json"
            write_json(path, study_protocol())
            loaded = read_protocol(path)
            self.assertEqual(loaded["name"], "test-rlpd-pilot")
            self.assertEqual(loaded["_path"], str(path.resolve()))

    def test_protocol_rejects_training_overlap_and_unreserved_evaluation_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "study.json"
            protocol = study_protocol()
            protocol["partitions"]["screen"]["seeds"] = [2001]
            write_json(path, protocol)
            with self.assertRaisesRegex(ValueError, "training"):
                read_protocol(path, verify_sources=False)

            protocol = study_protocol()
            protocol["reserved_training_seeds"].remove(3021)
            write_json(path, protocol)
            with self.assertRaisesRegex(ValueError, "reserved_training_seeds"):
                read_protocol(path, verify_sources=False)

    def test_followup_protocol_allows_longer_horizon_with_fresh_closed_partitions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "followup.json"
            write_json(path, followup_protocol())
            loaded = read_protocol(path, verify_sources=False)
            self.assertEqual(loaded["teacher_data_budget"], {
                "decisions": 16384,
                "minimum_distinct_finishes": 4,
            })
            self.assertEqual(loaded["student_training"]["steps_per_run"], 131072)
            self.assertEqual(loaded["student_training"]["expected_gradient_steps_per_run"], 130072)
            self.assertEqual(loaded["student_training"]["candidate_steps"], [65536, 131072])
            self.assertEqual(len(loaded["partitions"]["confirmation"]["seeds"]), 8)
            self.assertEqual(len(loaded["partitions"]["blind"]["seeds"]), 8)

    def test_dataset_hash_and_teacher_geometry_gate_then_restore_offline_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol_path = root / "study.json"
            write_json(protocol_path, study_protocol())
            protocol = read_protocol(protocol_path, verify_sources=False)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            manifest = write_dataset(dataset_dir, protocol_path, protocol)

            verified = verify_dataset(dataset_dir, protocol)
            self.assertEqual(verified["verified_finish_geometries"], 2)
            replay, digest, loaded = load_offline_replay(dataset_dir, protocol, seed=41)
            self.assertEqual(digest, manifest["dataset_sha256"])
            self.assertEqual(len(replay), 2)
            self.assertEqual(replay.valid_count, 2)
            replay_batch = replay.sample(8)
            self.assertTrue(np.all(replay_batch["source"] == "offline"))
            self.assertTrue(np.isin(replay_batch["geometry_seed"], [2001, 2002]).all())
            self.assertTrue(np.all(replay_batch["terminal"]))
            self.assertIn("verified_stored_decisions", loaded)

    def test_dataset_rejects_missing_finish_or_modified_episode_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol_path = root / "study.json"
            write_json(protocol_path, study_protocol())
            protocol = read_protocol(protocol_path, verify_sources=False)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            manifest = write_dataset(dataset_dir, protocol_path, protocol)
            manifest["episodes"][1]["finished"] = False
            manifest["distinct_finish_geometries"] = 1
            manifest["dataset_sha256"] = canonical_sha256({
                key: value for key, value in manifest.items() if key != "dataset_sha256"
            })
            write_json(dataset_dir / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "too few distinct-geometry"):
                verify_dataset(dataset_dir, protocol)

            write_dataset(dataset_dir, protocol_path, protocol)
            with (dataset_dir / "episodes" / "episode-0000.npz").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "content hash"):
                verify_dataset(dataset_dir, protocol)


if __name__ == "__main__":
    unittest.main()
