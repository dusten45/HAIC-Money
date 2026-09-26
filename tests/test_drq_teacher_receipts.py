"""Synthetic contract tests for the independent teacher-replay receipt joiner."""

from __future__ import annotations

import hashlib
import contextlib
import io
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

from scripts import compare_drq_teacher_replay as compare
from scripts.validate_drq_teacher_protocol import (
    _adapter_compatibility_record,
    _agent_drq_compatibility_record,
    _evaluation_harness_record,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _synthetic_agent_source(*, extra_dispatch: bool) -> bytes:
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


class ReceiptFixture:
    def __init__(self, root: Path):
        self.root = root
        self.protocol_path = root / "study-protocol.json"
        self.protocol = self._build_protocol()
        self.protocol_bytes = json.dumps(self.protocol, indent=2, sort_keys=True).encode() + b"\n"
        self.protocol_path.write_bytes(self.protocol_bytes)
        self.protocol_sha = digest(self.protocol_bytes)
        self.screen: dict[tuple[int, str, int | None], Path] = {}
        self.confirmation: dict[tuple[int, str], Path] = {}
        self.blind: Path | None = None
        self.reports: dict[Path, dict[str, object]] = {}
        self._make_screen_receipts()
        self._make_confirmation_receipts()
        self._make_blind_receipt()

    def _build_protocol(self) -> dict:
        snapshot = self.root / "protocol-source.py"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(b"frozen source snapshot\n")
        source_adapter = b"class Transition:\n    marker = 1\n"
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
        source_agent = _synthetic_agent_source(extra_dispatch=False)
        study_agent = _synthetic_agent_source(extra_dispatch=True)
        (self.root / "agent.py").write_bytes(study_agent)
        (self.root / "common_adapter.py").write_bytes(study_adapter)
        (self.root / "drq_v2.py").write_bytes(b"frozen drq source\n")
        root_source_hashes = {
            "drq_v2.py": digest(b"frozen drq source\n"),
            "agent.py": digest(study_agent),
            "common_adapter.py": digest(study_adapter),
            "evaluate_policy.py": digest(b"current evaluator source\n"),
        }
        (self.root / "evaluate_policy.py").write_bytes(b"current evaluator source\n")
        spec_fingerprints = {"action": "d" * 64, "observation": "e" * 64}
        prior_seed = 99
        training_seeds = {0: [900], 1: [901]}
        actors = []
        for learner in (0, 1):
            base = self.root / "source-actors" / f"learner-{learner}"
            checkpoint = base / "checkpoints" / "step-000131072" / "checkpoint.pt"
            checkpoint_manifest = base / "checkpoints" / "step-000131072" / "checkpoint.manifest.json"
            source_snapshot_dir = base / "source"
            actor = base / "actor.pt"
            run_config = base / "run-config.json"
            ledger = base / "episodes.jsonl"
            checkpoint_manifest.parent.mkdir(parents=True, exist_ok=True)
            run_config.parent.mkdir(parents=True, exist_ok=True)
            source_snapshot_dir.mkdir(parents=True, exist_ok=True)
            (source_snapshot_dir / "drq_v2.py").write_bytes(b"frozen drq source\n")
            (source_snapshot_dir / "agent.py").write_bytes(source_agent)
            (source_snapshot_dir / "common_adapter.py").write_bytes(source_adapter)
            (source_snapshot_dir / "evaluate_policy.py").write_bytes(b"source evaluator source\n")
            training_source_hashes = {
                "drq_v2.py": digest(b"frozen drq source\n"),
                "common_adapter.py": digest(source_adapter),
            }
            checkpoint_source_hashes_relative = {
                **training_source_hashes,
                "agent.py": digest(source_agent),
                "evaluate_policy.py": digest(b"source evaluator source\n"),
            }
            checkpoint_source_hashes = {
                str((source_snapshot_dir / name).resolve()): value
                for name, value in checkpoint_source_hashes_relative.items()
            }
            checkpoint_manifest.write_text(json.dumps({
                "algorithm": "drq-v2", "source_hashes": checkpoint_source_hashes,
                "extra": {"environment_steps": 131072,
                          "training_source_sha256": training_source_hashes},
            }))
            run_config.write_text(json.dumps({
                "git": {"commit": "synthetic-test-revision", "dirty": False},
                "config": {"seed": learner, "total_steps": 131072, "algorithm": "drq-v2", "track_ids": [1, 2, 3, 4],
                           "drq_config": {"augmentation_pad": 4},
                           "training_source_sha256": training_source_hashes},
            }))
            for path, content in (
                (checkpoint, f"source-checkpoint-{learner}".encode()),
                (actor, f"source-actor-{learner}".encode()),
                (ledger, json.dumps({"seed": training_seeds[learner][0]}).encode() + b"\n"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            actors.append({
                "learner_seed": learner,
                "checkpoint_path": checkpoint.relative_to(self.root).as_posix(),
                "checkpoint_sha256": digest(checkpoint.read_bytes()),
                "checkpoint_manifest_path": checkpoint_manifest.relative_to(self.root).as_posix(),
                "checkpoint_manifest_sha256": digest(checkpoint_manifest.read_bytes()),
                "actor_path": actor.relative_to(self.root).as_posix(),
                "actor_sha256": digest(actor.read_bytes()),
                "run_config_path": run_config.relative_to(self.root).as_posix(),
                "run_config_sha256": digest(run_config.read_bytes()),
                "episodes_path": ledger.relative_to(self.root).as_posix(),
                "episodes_sha256": digest(ledger.read_bytes()),
                "source_training_snapshot_dir": source_snapshot_dir.relative_to(self.root).as_posix(),
                "source_training_hashes": training_source_hashes,
                "checkpoint_source_hashes": checkpoint_source_hashes_relative,
                "source_revision": "synthetic-test-revision",
                "source_dirty": False,
                "training_steps": 131072,
                "training_track_ids": [1, 2, 3, 4],
                "training_geometry_seeds": training_seeds[learner],
                "source_pair_audit": {
                    "learner_seed": learner,
                    "checkpoint_sha256": digest(checkpoint.read_bytes()),
                    "actor_sha256": digest(actor.read_bytes()),
                    "actor_weights_sha256": "b" * 64,
                    "action_fingerprint": spec_fingerprints["action"],
                    "observation_fingerprint": spec_fingerprints["observation"],
                    "cpu_smoke_observations": 3,
                    "agent_source_parity": {
                        "source_sha256": checkpoint_source_hashes_relative["agent.py"],
                        "study_sha256": root_source_hashes["agent.py"],
                        "observations": 3,
                        "actions_exactly_equal": True,
                    },
                },
            })
        screen_seeds = list(range(11001, 11009))
        confirmation_seeds = list(range(12001, 12009))
        blind_seeds = list(range(13001, 13009))
        excluded = [prior_seed, 900, 901, *screen_seeds, *confirmation_seeds, *blind_seeds]
        candidate_seeds = sorted({5001, 5002, 6001, 6002, *screen_seeds, *confirmation_seeds, *blind_seeds})
        freshness_claim = "no known recorded overlap; historical pilot schedules are incomplete"
        geometry_report = {
            "passed": True,
            "parse_errors": [],
            "candidate_hits": {str(seed): [] for seed in candidate_seeds},
            "candidate_seeds": candidate_seeds,
            "known_excluded_geometry_seeds": [prior_seed],
            "source_snapshot_count": 1,
            "global_freshness_claim": freshness_claim,
        }
        geometry_report_path = self.root / "geometry-audit.json"
        geometry_report_path.write_text(json.dumps(geometry_report, indent=2, sort_keys=True) + "\n")
        compatibility = sorted([
            _agent_drq_compatibility_record(
                self.root / "source-actors/learner-0/source/agent.py", self.root / "agent.py"
            ),
            _adapter_compatibility_record(
                self.root / "source-actors/learner-0/source/common_adapter.py",
                self.root / "common_adapter.py",
            ),
            _evaluation_harness_record(
                self.root / "source-actors/learner-0/source/evaluate_policy.py",
                self.root / "evaluate_policy.py",
            ),
        ], key=lambda row: row["path"])
        return {
            "name": "drqv2-teacher-replay-v1",
            "purpose": "research",
            "frame_skip": 4,
            "max_steps": 2000,
            "partitions": {
                "screen": {"track_ids": [101, 102, 103], "seeds": screen_seeds, "repeats": 2},
                "confirmation": {"track_ids": [111, 112, 113, 114], "seeds": confirmation_seeds, "repeats": 2},
                "blind": {"track_ids": [121, 122, 123], "seeds": blind_seeds, "repeats": 2},
            },
            "reserved_training_seeds": sorted(excluded),
            "schema_version": 1,
            "study_id": "drqv2-teacher-replay-v1",
            "run_root": "runs/drqv2-teacher-replay-v1-test",
            "hypothesis": "teacher replay improves repeated unseen-track finishes",
            "source_revision": "synthetic-test-revision",
            "source_compatibility": compatibility,
            "working_tree_dirty": False,
            "working_tree_status_sha256": "f" * 64,
            "spec_fingerprints": spec_fingerprints,
            "source_snapshots": {
                snapshot.relative_to(self.root).as_posix(): digest(snapshot.read_bytes()),
                **root_source_hashes,
            },
            "source_actors": actors,
            "training_pools": {
                "teacher_training": {"track_ids": [1, 2, 3, 4], "seeds": [5001, 5002], "sampler_seed": 1},
                "online_training": {"track_ids": [1, 2, 3, 4], "seeds": [6001, 6002], "sampler_seed": 2},
            },
            "known_excluded_geometry_seeds": [prior_seed],
            "geometry_audit": {
                "report_path": geometry_report_path.relative_to(self.root).as_posix(),
                "report_sha256": digest(geometry_report_path.read_bytes()),
                "candidate_seeds": candidate_seeds,
                "source_snapshot_count": 1,
                "known_excluded_geometry_seed_count": 1,
                "global_freshness_claim": freshness_claim,
            },
            "environment": {
                "observation_shape": [4, 84, 84], "observation_dtype": "float32",
                "observation_range": [0.0, 1.0], "frame_skip": 4,
                "action_axes": ["steer", "gas", "brake"], "native_action_range": [-1.0, 1.0],
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
                "critic_updates_per_learning_decision": 1, "batch_size": 64,
                "teacher_rows_per_treatment_batch": 16, "online_rows_per_treatment_batch": 48,
                "checkpoint_online_steps": [16384, 32768], "online_replay_capacity": 100000,
                "teacher_replay_capacity_per_source": 16384,
            },
            "runtime": {
                "training_device": "cpu", "training_executable": "python -m train_drq_teacher",
                "training_python": "python", "dependencies": {"torch": "2.1.0"},
                "hardware": {"synthetic": True},
                "evaluation_executable": "python -m evaluate_policy",
                "evaluation_runtime": {"torch": "2.1.0+cpu", "cuda_available": False},
                "worker_limits": {"actor_load_seconds_max": 10, "reset_act_seconds_max": 5,
                                  "peak_rss_bytes_max": 1073741824, "workers": 1,
                                  "frame_skip": 4, "max_steps": 2000},
            },
            "selection": {
                "screen_cells": 24, "confirmation_cells": 32, "blind_cells": 24,
                "repeats_per_cell": 2, "canonical_repeat": 0,
                "checkpoint_tie_break": "earlier", "confirmation_minimum_gain_vs_online_only_per_source": 2,
                "confirmation_at_least_unchanged_source_per_source": True,
                "checkpoint_candidate_steps": [16384, 32768],
                "screen_sort": "completions-desc,canonical-mean-progress-desc,completed-lap-time-asc",
                "screen_requires_nonzero_teacher_finishes_per_source": True,
                "screen_teacher_finishes_at_least_online_control_per_source": True,
                "confirmation_winning_geometry_minimum": 2,
                "promotion_requires_both_source_actors": True,
                "blind_finalist_learner_zero_breaks_exact_tie": True,
                "blind_opens_only_after_confirmation_pass": True,
                "blind_minimum_canonical_finishes": 1,
            },
        }

    def _actor(self, learner: int, arm: str, step: int | None) -> tuple[Path, str, str | None]:
        if arm == "unchanged-source":
            source = self.protocol["source_actors"][learner]
            return self.root / source["actor_path"], source["actor_sha256"], None
        actor_path = self.root / "student-actors" / f"{learner}-{arm}-{step}.pt"
        actor_path.parent.mkdir(parents=True, exist_ok=True)
        if not actor_path.exists():
            actor_path.write_bytes(f"actor-{learner}-{arm}-{step}".encode())
        checkpoint_sha = digest(f"checkpoint-{learner}-{arm}-{step}".encode())
        return actor_path, digest(actor_path.read_bytes()), checkpoint_sha

    @staticmethod
    def identity(learner: int, arm: str, actor_sha: str, checkpoint_sha: str | None, step: int | None) -> dict:
        return {
            "source_learner_seed": learner,
            "arm": arm,
            "actor_sha256": actor_sha,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_online_step": step,
        }

    def _finishes(self, partition: str, learner: int, arm: str, step: int | None) -> set[tuple[int, int]]:
        matrix = self.protocol["partitions"][partition]
        cells = [(track, seed) for track in matrix["track_ids"] for seed in matrix["seeds"]]
        if partition == "confirmation":
            count = 1 if arm == "unchanged-source" else 0 if arm == "online-only" else 3
            return set(cells[:count])
        if partition == "blind":
            return {cells[0]} if arm == "teacher-replay" else set()
        if arm == "unchanged-source":
            count = 3
        elif arm == "online-only":
            count = 2 if step == 16384 else 3
        else:
            count = 1 if step == 16384 else 3
        return set(cells[:count])

    def _make_screen_receipts(self) -> None:
        for learner in (0, 1):
            for arm in ("unchanged-source", "online-only", "teacher-replay"):
                steps = (None,) if arm == "unchanged-source" else (16384, 32768)
                for step in steps:
                    actor_path, actor_sha, checkpoint_sha = self._actor(learner, arm, step)
                    identity = self.identity(learner, arm, actor_sha, checkpoint_sha, step)
                    finishes = self._finishes("screen", learner, arm, step)
                    report = self._write_evaluator("screen", identity, actor_path, finishes)
                    wrapper_path = self.root / "wrappers" / f"screen-{learner}-{arm}-{step}.json"
                    wrapper = {
                        "study_protocol_sha256": self.protocol_sha,
                        **identity,
                        "partition": "screen",
                        "evaluator_receipt_path": str(report["pointer_path"]),
                        "evaluator_receipt_sha256": digest(report["pointer_path"].read_bytes()),
                    }
                    dump_json(wrapper_path, wrapper)
                    self.screen[(learner, arm, step)] = wrapper_path
                    self.reports[wrapper_path] = report

    def _selected_screen(self, learner: int, arm: str) -> Path:
        step = None if arm == "unchanged-source" else 16384 if arm == "online-only" else 32768
        return self.screen[(learner, arm, step)]

    def _make_confirmation_receipts(self) -> None:
        for learner in (0, 1):
            for arm in ("unchanged-source", "online-only", "teacher-replay"):
                screen_path = self._selected_screen(learner, arm)
                screen_wrapper = json.loads(screen_path.read_text())
                identity = self.identity(
                    learner, arm, screen_wrapper["actor_sha256"], screen_wrapper["checkpoint_sha256"],
                    screen_wrapper["checkpoint_online_step"],
                )
                actor_path, _, _ = self._actor(learner, arm, identity["checkpoint_online_step"])
                report = self._write_evaluator(
                    "confirmation", identity, actor_path, self._finishes("confirmation", learner, arm,
                                                                          identity["checkpoint_online_step"]),
                    previous=self.reports[screen_path],
                )
                wrapper_path = self.root / "wrappers" / f"confirmation-{learner}-{arm}.json"
                wrapper = {
                    "study_protocol_sha256": self.protocol_sha,
                    **identity,
                    "partition": "confirmation",
                    "evaluator_receipt_path": str(report["pointer_path"]),
                    "evaluator_receipt_sha256": digest(report["pointer_path"].read_bytes()),
                    "screen_receipt_sha256": digest(screen_path.read_bytes()),
                    "screen_candidate_identity": identity,
                }
                dump_json(wrapper_path, wrapper)
                self.confirmation[(learner, arm)] = wrapper_path
                self.reports[wrapper_path] = report

    def _make_blind_receipt(self) -> None:
        learner, arm = 0, "teacher-replay"
        screen_path = self._selected_screen(learner, arm)
        confirmation_path = self.confirmation[(learner, arm)]
        screen_wrapper = json.loads(screen_path.read_text())
        identity = self.identity(learner, arm, screen_wrapper["actor_sha256"], screen_wrapper["checkpoint_sha256"],
                                 screen_wrapper["checkpoint_online_step"])
        actor_path, _, _ = self._actor(learner, arm, identity["checkpoint_online_step"])
        report = self._write_evaluator(
            "blind", identity, actor_path, self._finishes("blind", learner, arm, identity["checkpoint_online_step"]),
            previous=self.reports[confirmation_path],
        )
        self.blind = self.root / "wrappers" / "blind-0-teacher.json"
        dump_json(self.blind, {
            "study_protocol_sha256": self.protocol_sha,
            **identity,
            "partition": "blind",
            "evaluator_receipt_path": str(report["pointer_path"]),
            "evaluator_receipt_sha256": digest(report["pointer_path"].read_bytes()),
            "screen_receipt_sha256": digest(screen_path.read_bytes()),
            "screen_candidate_identity": identity,
        })
        self.reports[self.blind] = report

    def _write_evaluator(
        self,
        partition: str,
        identity: dict,
        actor_path: Path,
        finishes: set[tuple[int, int]],
        *,
        previous: dict | None = None,
        suffix: str = "",
    ) -> dict:
        matrix = self.protocol["partitions"][partition]
        candidate_id = f"{identity['actor_sha256'][:16]}-candidate"
        evaluation_dir = self.root / "evaluations" / partition / f"{identity['source_learner_seed']}-{identity['arm']}-{identity['checkpoint_online_step']}{suffix}"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        run_dir = self.root / "student-runs" / f"{identity['source_learner_seed']}-{identity['arm']}-{identity['checkpoint_online_step']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        if not config_path.exists():
            config_path.write_text('{"config": {"frame_skip": 4, "max_steps": 2000}}\n')
        config_sha = digest(config_path.read_bytes())
        candidate_dir = evaluation_dir / "candidates"
        candidate_dir.mkdir(exist_ok=True)
        archive_relative = f"candidates/{candidate_id}.pt"
        archive_path = evaluation_dir / archive_relative
        archive_path.write_bytes(actor_path.read_bytes())
        config_relative = f"candidates/{candidate_id}.config.json"
        (evaluation_dir / config_relative).write_bytes(config_path.read_bytes())
        runtime = {
            "python_version": [3, 11], "sys_platform": "linux",
            "packages": {"torch": "2.1.0+cpu", "numpy": "1.26.0", "gymnasium": "0.29.1",
                         "opencv-python": "4.8.1.78", "stable-baselines3": "2.2.1"},
            "torch_cuda": None, "cuda_available": False, "torch_threads": 1,
            "torch_interop_threads": 1,
        }
        action = [0.1, 0.2, 0.3]
        trace_sha = digest(struct.pack("<3f", *action))
        canonical_rows = []
        episodes = []
        for track in matrix["track_ids"]:
            for seed in matrix["seeds"]:
                finished = (track, seed) in finishes
                for repeat in range(2):
                    row = {
                        "status": "ok", "track_id": track, "seed": seed, "repeat": repeat,
                        "steps": 1, "reward": 3.0, "progress": 0.5,
                        "finished": finished, "terminated": True, "truncated": False,
                        "retire_reason": None if finished else "off_track", "finish_qualified": finished,
                        "finish_time_s": 1.234 if finished else None,
                        "lap_time_ms": 1234.0 if finished else None, "damage": 0.1,
                        "termination_class": "finished" if finished else "off_track",
                        "collision_actions": 0, "steering_delta_abs_mean": 0.25,
                        "action_smoothing": {}, "action_smoothing_fingerprint": "a" * 64,
                        "action_control": {}, "action_control_fingerprint": "b" * 64,
                        "action_representation": {}, "action_representation_fingerprint": "c" * 64,
                        "raw_time_s": 1.0, "action_trace_sha256": trace_sha, "actions": [action],
                        "model_load_seconds": 0.1, "process_initialization_seconds": 1.0,
                        "reset_seconds": 0.1, "agent_reset_seconds": 0.1, "first_action_seconds": 0.1,
                        "mean_action_seconds": 0.1, "p95_action_seconds": 0.1, "max_action_seconds": 0.1,
                        "actor_load_peak_rss_bytes": 1000000, "peak_rss_bytes": 2000000,
                        "rss_scope": "whole isolated evaluator", "episode_wall_seconds": 1.0,
                        "loaded_archive_sha256": identity["actor_sha256"], "runtime": runtime,
                        "worker_pid": 1000 + repeat, "candidate_id": candidate_id,
                        "source_path": str(actor_path), "parent_wall_seconds": 1.0,
                    }
                    episodes.append(row)
                    if repeat == 0:
                        canonical_rows.append(row)
        finish_count = sum(row["finished"] for row in canonical_rows)
        lap_times = [row["lap_time_ms"] for row in canonical_rows if row["finished"]]
        summary = {
            "action_control": {}, "action_control_fingerprint": "b" * 64, "action_control_present": False,
            "action_representation": {}, "action_representation_fingerprint": "c" * 64,
            "action_representation_present": False, "action_smoothing": {},
            "action_smoothing_fingerprint": "a" * 64, "action_smoothing_present": True,
            "algorithm": "drq-v2", "aliases": [str(actor_path)], "archive_sha256": identity["actor_sha256"],
            "by_track": {}, "candidate_id": candidate_id, "canonical_episodes": len(canonical_rows),
            "cpu_reload_matches": True, "determinism_audited": True, "diagnostic_only": False,
            "eligible": True, "evaluation_archive_path": archive_relative,
            "evaluation_archive_sha256": identity["actor_sha256"],
            "evaluation_run_config_path": config_relative,
            "expected_cells": len(canonical_rows), "expected_results": len(episodes),
            "export_metadata": {"format": "haic-drq-v2-actor-v1", "action_spec": {"frame_skip": 4}},
            "export_spec_fingerprints": {}, "is_comparator": False, "norm_obs": False,
            "operational_failures": 0, "policy_sha256": identity["actor_sha256"],
            "resources": {"process_initialization_seconds": 1.0, "agent_reset_seconds": 0.1,
                          "max_action_seconds": 0.1, "peak_rss_bytes": 2000000},
            "run_config_path": str(config_path), "run_config_sha256": config_sha,
            "run_frame_skip": 4, "run_max_steps": 2000, "source_path": str(actor_path),
            "summary": {"n_episodes": len(canonical_rows),
                        "finish_rate": finish_count / len(canonical_rows), "avg_progress": 0.5,
                        "avg_lap_time_ms": sum(lap_times) / len(lap_times) if lap_times else None},
            "vecnormalize_path": None, "vecnormalize_sha256": None,
        }
        previous_eval = None
        if previous is not None:
            prior_pointer = previous["pointer_path"]
            prior_dir = previous["evaluation_dir"]
            previous_eval = {"evaluation_dir": str(prior_dir),
                             "partition": "screen" if partition == "confirmation" else "confirmation",
                             "actor_sha256": identity["actor_sha256"], "files": {}}
            for name, source in (
                ("previous_evaluation.json", prior_pointer),
                ("previous_summary.json", prior_dir / "summary.json"),
                ("previous_manifest.json", prior_dir / "manifest.json"),
            ):
                snapshot_path = evaluation_dir / name
                shutil.copyfile(source, snapshot_path)
                previous_eval["files"][name] = {
                    "source_path": str(source.resolve()), "sha256": digest(source.read_bytes()),
                    "snapshot_path": name,
                }
        audit_rows = [
            {"audited": True, "candidate_id": candidate_id, "matches_canonical": True,
             "repeats": 2, "seed": seed, "track_id": track}
            for track in matrix["track_ids"] for seed in matrix["seeds"]
        ]
        manifest = {
            "schema_version": 2, "protocol": f"{self.protocol['name']}-{partition}",
            "protocol_sha256": self.protocol_sha, "partition": partition, "diagnostic_only": False,
            "run_dir": str(run_dir), "previous_evaluation": previous_eval,
            "started_at_utc": "2026-01-01T00:00:00Z" if partition == "screen" else
                               "2026-01-02T00:00:00Z" if partition == "confirmation" else "2026-01-03T00:00:00Z",
            "cell_matrix": matrix, "frame_skip": 4, "max_steps": 2000,
            "action_smoothing": {}, "action_smoothing_fingerprints": ["a" * 64],
            "action_smoothing_by_candidate": {candidate_id: {}}, "timeout_seconds": 300,
            "workers": 1,
            "limits": {"initialization_seconds": 10.0, "reset_seconds": 5.0,
                       "action_seconds": 5.0, "peak_rss_bytes": 1024**3},
            "git": {"commit": "test", "dirty": False}, "coordinator_runtime": runtime,
            "worker_runtime": [runtime], "runtime": runtime,
            "finished_at_utc": "2026-01-01T00:01:00Z",
        }
        dump_json(evaluation_dir / "manifest.json", manifest)
        dump_json(evaluation_dir / "summary.json", [summary])
        with (evaluation_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
            for row in episodes:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        dump_json(evaluation_dir / "determinism.json", audit_rows)
        dump_json(evaluation_dir / "protocol.json", matrix)
        (evaluation_dir / "protocol_spec.json").write_bytes(self.protocol_bytes)
        dump_json(evaluation_dir / "candidates.json", [{"candidate_id": candidate_id}])
        pointer_path = self.root / "evaluator-pointers" / f"{partition}-{candidate_id}-{suffix or 'base'}.json"
        dump_json(pointer_path, {
            "evaluation_dir": str(evaluation_dir.resolve()), "protocol_name": f"{self.protocol['name']}-{partition}",
            "protocol_sha256": self.protocol_sha, "partition": partition, "diagnostic_only": False,
            "ranked": [summary],
        })
        return {"pointer_path": pointer_path, "evaluation_dir": evaluation_dir,
                "identity": identity, "candidate_id": candidate_id, "summary": summary,
                "finishes": finish_count}


class DrqTeacherReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="drq-teacher-receipts-")
        cls.root = Path(cls.temporary.name)
        cls.fixture = ReceiptFixture(cls.root)
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.protocol, cls.protocol_sha = compare.load_protocol(cls.fixture.protocol_path, cls.root)
        cls.screen_receipts = list(cls.fixture.screen.values())
        cls.confirmation_receipts = list(cls.fixture.confirmation.values())
        cls.screen_lineage = [cls.fixture.screen[(learner, arm, step)]
                              for learner in (0, 1)
                              for arm, step in (("unchanged-source", None), ("online-only", 16384),
                                                ("teacher-replay", 32768))]

    def copy_wrapper(self, source: Path, name: str, mutate) -> Path:
        value = json.loads(source.read_text(encoding="utf-8"))
        mutate(value)
        destination = self.root / "mutated-wrappers" / name
        dump_json(destination, value)
        return destination

    def copy_evaluator(self, source: Path, name: str, mutate) -> Path:
        original = self.fixture.reports[source]
        report = original
        new_dir = self.root / "mutated-evaluations" / name
        shutil.copytree(original["evaluation_dir"], new_dir)
        mutate(new_dir)
        pointer = json.loads(Path(original["pointer_path"]).read_text())
        pointer["evaluation_dir"] = str(new_dir.resolve())
        pointer_path = self.root / "mutated-pointers" / f"{name}.json"
        dump_json(pointer_path, pointer)
        wrapper = json.loads(source.read_text())
        wrapper["evaluator_receipt_path"] = str(pointer_path.resolve())
        wrapper["evaluator_receipt_sha256"] = digest(pointer_path.read_bytes())
        wrapper_path = self.root / "mutated-wrappers" / f"{name}.json"
        dump_json(wrapper_path, wrapper)
        return wrapper_path

    def write_decision(self, name: str, decision: dict) -> Path:
        path = self.root / "decisions" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        compare.write_sealed(path, decision)
        return path

    def valid_screen_decision(self) -> dict:
        return compare.compare_receipts(self.protocol, self.protocol_sha, "screen", self.screen_receipts)

    def valid_confirmation_decision(self) -> dict:
        return compare.compare_receipts(
            self.protocol, self.protocol_sha, "confirmation", self.confirmation_receipts, self.screen_lineage
        )

    def test_valid_screen_join_reports_each_candidate_without_selection(self) -> None:
        before = {path: digest(path.read_bytes()) for path in self.screen_receipts}
        decision = self.valid_screen_decision()
        self.assertFalse(decision["selection_performed"])
        self.assertEqual(decision["result"]["selection_performed"], False)
        self.assertEqual(len(decision["result"]["sources"]), 2)
        self.assertEqual(len(decision["input_receipts"]), 10)
        learner0 = decision["result"]["sources"][0]
        self.assertEqual(learner0["canonical_cells"], 24)
        teacher_eligibility = {
            item["checkpoint_online_step"]: item["screen_only_eligible"]
            for item in learner0["candidates"] if item["arm"] == "teacher-replay"
        }
        self.assertEqual(teacher_eligibility, {16384: False, 32768: True})
        self.assertEqual(before, {path: digest(path.read_bytes()) for path in self.screen_receipts})

    def test_cli_writes_a_distinct_sealed_decision_receipt(self) -> None:
        output = self.root / "cli-output" / "screen-decision.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        args = [
            "--protocol-file", str(self.fixture.protocol_path),
            "--partition", "screen",
            "--output", str(output),
            "--repo-root", str(self.root),
        ]
        for path in self.screen_receipts:
            args.extend(("--receipt", str(path)))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(compare.main(args), 0)
        decision, decision_sha = compare._sealed_json(output, "test output")
        self.assertEqual(decision["receipt_type"], compare.DECISION_TYPE)
        self.assertEqual(output.with_suffix(".json.sha256").read_text().strip(), decision_sha)
        self.assertIn(str(output), stdout.getvalue())

    def test_valid_confirmation_applies_fixed_per_source_hurdles(self) -> None:
        decision = self.valid_confirmation_decision()
        self.assertTrue(decision["result"]["passed"])
        for source in decision["result"]["sources"]:
            self.assertEqual(source["finish_counts"]["teacher_replay"], 3)
            self.assertTrue(source["hurdles"]["teacher_at_least_online_only_plus_2"])
            self.assertTrue(source["hurdles"]["teacher_at_least_unchanged_source"])
            self.assertGreaterEqual(len(source["gain_geometry_seeds_vs_online_only"]), 2)
            comparison = source["teacher_vs_online_only"]
            self.assertEqual(comparison["paired_wins"] + comparison["paired_losses"]
                             + comparison["paired_ties"], 32)
            self.assertIn("not a significance test", comparison["uncertainty_interpretation"])

    def test_screen_requires_exact_actor_receipt_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 10"):
            compare.compare_receipts(self.protocol, self.protocol_sha, "screen", self.screen_receipts[:-1])
        with self.assertRaisesRegex(ValueError, "exactly 10"):
            compare.compare_receipts(self.protocol, self.protocol_sha, "screen", self.screen_receipts + [self.screen_receipts[0]])

    def test_rejects_duplicate_roles_and_wrong_screen_grid(self) -> None:
        duplicate = self.screen_receipts[:-1] + [self.screen_receipts[0]]
        with self.assertRaisesRegex(ValueError, "duplicate actor receipt file paths"):
            compare.compare_receipts(self.protocol, self.protocol_sha, "screen", duplicate)
        wrong = self.copy_wrapper(self.screen_receipts[-1], "wrong-role.json",
                                  lambda value: value.update(source_learner_seed=0))
        with self.assertRaisesRegex(ValueError, "duplicate source/arm/checkpoint"):
            compare.compare_receipts(self.protocol, self.protocol_sha, "screen", [wrong, *self.screen_receipts[1:]])

    def test_rejects_protocol_hash_evaluator_hash_and_schema_mutations(self) -> None:
        stale_protocol = self.copy_wrapper(
            self.screen_receipts[0], "stale-protocol.json",
            lambda value: value.update(study_protocol_sha256="0" * 64),
        )
        with self.assertRaisesRegex(ValueError, "protocol SHA differs"):
            compare.compare_receipts(self.protocol, self.protocol_sha, "screen",
                                     [stale_protocol, *self.screen_receipts[1:]])
        stale_evaluator = self.copy_wrapper(
            self.screen_receipts[0], "stale-evaluator.json",
            lambda value: value.update(evaluator_receipt_sha256="0" * 64),
        )
        with self.assertRaisesRegex(ValueError, "evaluator receipt hash mismatch"):
            compare.compare_receipts(self.protocol, self.protocol_sha, "screen",
                                     [stale_evaluator, *self.screen_receipts[1:]])
        unsupported = self.copy_wrapper(self.screen_receipts[0], "unsupported-field.json",
                                        lambda value: value.update(unexpected_field=True))
        with self.assertRaisesRegex(ValueError, "unsupported schema fields"):
            compare.compare_receipts(self.protocol, self.protocol_sha, "screen",
                                     [unsupported, *self.screen_receipts[1:]])

    def test_rejects_mismatched_evaluator_protocol_and_unsupported_manifest(self) -> None:
        source = self.screen_receipts[0]

        def change_protocol(directory: Path) -> None:
            path = directory / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["protocol_sha256"] = "0" * 64
            dump_json(path, manifest)

        mismatched = self.copy_evaluator(source, "wrong-evaluator-protocol", change_protocol)
        with self.assertRaisesRegex(ValueError, "manifest identity mismatch"):
            compare.compare_receipts(self.protocol, self.protocol_sha, "screen",
                                     [mismatched, *self.screen_receipts[1:]])

        def add_manifest_field(directory: Path) -> None:
            path = directory / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["future_field"] = 1
            dump_json(path, manifest)

        unsupported = self.copy_evaluator(source, "unsupported-manifest", add_manifest_field)
        with self.assertRaisesRegex(ValueError, "unsupported evaluator manifest schema"):
            compare.compare_receipts(self.protocol, self.protocol_sha, "screen",
                                     [unsupported, *self.screen_receipts[1:]])

    def test_rejects_missing_extra_and_duplicate_evaluator_cells(self) -> None:
        source = self.screen_receipts[0]

        def drop_row(directory: Path) -> None:
            path = directory / "episodes.jsonl"
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n")

        missing = self.copy_evaluator(source, "missing-cell", drop_row)
        with self.assertRaisesRegex(ValueError, "incomplete evaluator cell matrix"):
            compare.compare_receipts(self.protocol, self.protocol_sha, "screen",
                                     [missing, *self.screen_receipts[1:]])

        def duplicate_row(directory: Path) -> None:
            path = directory / "episodes.jsonl"
            lines = path.read_text().splitlines()
            path.write_text("\n".join([*lines, lines[0]]) + "\n")

        duplicate = self.copy_evaluator(source, "duplicate-cell", duplicate_row)
        with self.assertRaisesRegex(ValueError, "extra/duplicate evaluator cell"):
            compare.compare_receipts(self.protocol, self.protocol_sha, "screen",
                                     [duplicate, *self.screen_receipts[1:]])

    def test_rejects_repeat_outcome_trace_mismatch(self) -> None:
        source = self.screen_receipts[0]

        def mismatch_repeat(directory: Path) -> None:
            path = directory / "episodes.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]["finished"] = not rows[1]["finished"]
            rows[1]["termination_class"] = "finished" if rows[1]["finished"] else "off_track"
            rows[1]["lap_time_ms"] = 1234.0 if rows[1]["finished"] else None
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

        mismatched = self.copy_evaluator(source, "repeat-mismatch", mismatch_repeat)
        with self.assertRaisesRegex(ValueError, "reload outcome/trace mismatch"):
            compare.compare_receipts(self.protocol, self.protocol_sha, "screen",
                                     [mismatched, *self.screen_receipts[1:]])

    def test_confirmation_rejects_reselected_actor_and_wrong_lineage(self) -> None:
        selected = self.confirmation_receipts[0]
        wrong_lineage = self.copy_wrapper(
            selected, "wrong-screen-hash.json", lambda value: value.update(screen_receipt_sha256="0" * 64)
        )
        with self.assertRaisesRegex(ValueError, "wrong screen lineage"):
            compare.compare_receipts(
                self.protocol, self.protocol_sha, "confirmation",
                [wrong_lineage, *self.confirmation_receipts[1:]], self.screen_lineage,
            )
        with self.assertRaisesRegex(ValueError, "exactly 6 lineage"):
            compare.compare_receipts(
                self.protocol, self.protocol_sha, "confirmation", self.confirmation_receipts,
                self.screen_lineage[:-1],
            )

    def test_confirmation_rejects_duplicate_source_arm_role(self) -> None:
        duplicate = self.copy_wrapper(
            self.confirmation_receipts[-1], "confirmation-duplicate-role.json",
            lambda value: value.update(source_learner_seed=0),
        )
        receipts = [duplicate, *self.confirmation_receipts[1:]]
        with self.assertRaisesRegex(ValueError, "duplicate source/arm/checkpoint"):
            compare.compare_receipts(
                self.protocol, self.protocol_sha, "confirmation", receipts, self.screen_lineage
            )

    def test_confirmation_is_fail_pass_when_fixed_hurdle_is_not_met(self) -> None:
        source = self.fixture.confirmation[(1, "teacher-replay")]
        prior_screen = self._screen_lineage_by_role()[(1, "teacher-replay")]
        wrapper = json.loads(source.read_text())
        identity = compare._identity(wrapper)
        actor_path, _, _ = self.fixture._actor(1, "teacher-replay", identity["checkpoint_online_step"])
        report = self.fixture._write_evaluator(
            "confirmation", identity, actor_path, set(), previous=self.fixture.reports[prior_screen],
            suffix="-zero-teacher-finish",
        )
        wrapper["evaluator_receipt_path"] = str(report["pointer_path"])
        wrapper["evaluator_receipt_sha256"] = digest(report["pointer_path"].read_bytes())
        replacement = self.root / "mutated-wrappers" / "confirmation-zero-teacher.json"
        dump_json(replacement, wrapper)
        reports = [replacement if path == source else path for path in self.confirmation_receipts]
        decision = compare.compare_receipts(
            self.protocol, self.protocol_sha, "confirmation", reports, self.screen_lineage
        )
        self.assertFalse(decision["result"]["passed"])
        learner1 = next(item for item in decision["result"]["sources"] if item["source_learner_seed"] == 1)
        self.assertFalse(learner1["passed"])
        self.assertFalse(learner1["hurdles"]["nonzero_teacher_finishes"])

    def _screen_lineage_by_role(self) -> dict[tuple[int, str], Path]:
        result = {}
        for (learner, arm, step), path in self.fixture.screen.items():
            selected = None if arm == "unchanged-source" else 16384 if arm == "online-only" else 32768
            if step == selected:
                result[(learner, arm)] = path
        return result

    def test_blind_accepts_only_screen_fixed_candidate_after_confirmation(self) -> None:
        screen_path = self._screen_lineage_by_role()[(0, "teacher-replay")]
        screen_decision_path = self.write_decision("screen-paired.json", self.valid_screen_decision())
        confirmation_decision_path = self.write_decision("confirmation-paired.json", self.valid_confirmation_decision())
        finalist_path = self._write_finalist(screen_path, screen_decision_path)
        lineages = [confirmation_decision_path, finalist_path]
        decision = compare.compare_receipts(self.protocol, self.protocol_sha, "blind", [self.fixture.blind], lineages)
        self.assertTrue(decision["result"]["accepted"])
        self.assertEqual(decision["result"]["canonical_blind_finishes"], 1)

    def test_blind_rejects_reselection_and_wrong_confirmation_gate(self) -> None:
        screen_path = self._screen_lineage_by_role()[(0, "teacher-replay")]
        screen_decision_path = self.write_decision("screen-paired-invalid.json", self.valid_screen_decision())
        valid_confirmation_path = self.write_decision("confirmation-paired-valid.json", self.valid_confirmation_decision())
        confirmation = self.valid_confirmation_decision()
        confirmation["result"]["passed"] = False
        failed_confirmation_path = self.write_decision("confirmation-paired-invalid.json", confirmation)
        finalist = self._write_finalist(screen_path, screen_decision_path)
        reselected = self.copy_wrapper(
            self.fixture.blind, "blind-reselected.json",
            lambda value: value.update(screen_receipt_sha256="0" * 64),
        )
        with self.assertRaisesRegex(ValueError, "wrong screen finalist lineage"):
            compare.compare_receipts(
                self.protocol, self.protocol_sha, "blind", [reselected],
                [valid_confirmation_path, finalist],
            )
        with self.assertRaisesRegex(ValueError, "passed paired confirmation"):
            compare.compare_receipts(
                self.protocol, self.protocol_sha, "blind", [self.fixture.blind],
                [failed_confirmation_path, finalist],
            )

    def test_blind_rejects_altered_selected_candidates_hash(self) -> None:
        screen_path = self._screen_lineage_by_role()[(0, "teacher-replay")]
        screen_decision_path = self.write_decision("screen-paired-hash.json", self.valid_screen_decision())
        confirmation_path = self.write_decision(
            "confirmation-paired-hash.json", self.valid_confirmation_decision()
        )
        finalist_path = self._write_finalist(screen_path, screen_decision_path)
        finalist_value = json.loads(finalist_path.read_text())
        finalist_value["selection_file_sha256"] = "0" * 64
        altered = self.root / "finalists" / "altered-selection-hash.json"
        compare.write_sealed(altered, finalist_value)
        with self.assertRaisesRegex(ValueError, "selected-candidates file hash mismatch"):
            compare.compare_receipts(
                self.protocol, self.protocol_sha, "blind", [self.fixture.blind],
                [confirmation_path, altered],
            )

    def test_blind_without_a_canonical_finish_is_not_accepted(self) -> None:
        screen_path = self._screen_lineage_by_role()[(0, "teacher-replay")]
        confirmation_path = self.fixture.confirmation[(0, "teacher-replay")]
        screen_decision_path = self.write_decision("screen-paired-zero-blind.json", self.valid_screen_decision())
        confirmation_decision_path = self.write_decision(
            "confirmation-paired-zero-blind.json", self.valid_confirmation_decision()
        )
        finalist_path = self._write_finalist(screen_path, screen_decision_path)
        source_wrapper = json.loads(self.fixture.blind.read_text())
        identity = compare._identity(source_wrapper)
        actor_path, _, _ = self.fixture._actor(0, "teacher-replay", identity["checkpoint_online_step"])
        report = self.fixture._write_evaluator(
            "blind", identity, actor_path, set(), previous=self.fixture.reports[confirmation_path],
            suffix="-zero-finish",
        )
        source_wrapper["evaluator_receipt_path"] = str(report["pointer_path"])
        source_wrapper["evaluator_receipt_sha256"] = digest(report["pointer_path"].read_bytes())
        blind_path = self.root / "mutated-wrappers" / "blind-zero-finish.json"
        dump_json(blind_path, source_wrapper)
        decision = compare.compare_receipts(
            self.protocol, self.protocol_sha, "blind", [blind_path],
            [confirmation_decision_path, finalist_path],
        )
        self.assertFalse(decision["result"]["accepted"])
        self.assertEqual(decision["result"]["canonical_blind_finishes"], 0)
        self.assertEqual(len(decision["result"]["canonical_outcomes"]), 24)

    def _write_finalist(self, screen_path: Path, screen_decision_path: Path) -> Path:
        wrapper = json.loads(screen_path.read_text())
        screen_decision_sha = digest(screen_decision_path.read_bytes())
        selection_path = self.root / "finalists" / f"{screen_decision_path.stem}-selected-candidates.json"
        candidate_receipts = []
        for actor_path in self.screen_receipts:
            actor = json.loads(actor_path.read_text())
            candidate_receipts.append({
                "candidate_identity": compare._identity(actor),
                "wrapper_path": str(actor_path.resolve()),
                "wrapper_sha256": digest(actor_path.read_bytes()),
            })
        selected_identities = []
        for learner in (0, 1):
            for arm in ("unchanged-source", "online-only", "teacher-replay"):
                selected_path = self._screen_lineage_by_role()[(learner, arm)]
                selected_wrapper = json.loads(selected_path.read_text())
                score_tuple = [self.fixture.reports[selected_path]["finishes"], 0.5, -1234.0]
                selected_identities.append({
                    **compare._identity(selected_wrapper),
                    "screen_wrapper_path": str(selected_path.resolve()),
                    "screen_wrapper_sha256": digest(selected_path.read_bytes()),
                    "score_tuple": score_tuple,
                })
        blind_finalist = next(item for item in selected_identities
                              if item["source_learner_seed"] == 0 and item["arm"] == "teacher-replay")
        selection = {
            "format": compare.SELECTION_FORMAT,
            "study_protocol_sha256": self.protocol_sha,
            "screen_decision_path": str(screen_decision_path.resolve()),
            "screen_decision_sha256": screen_decision_sha,
            "candidate_receipts": candidate_receipts,
            "selected_identities": selected_identities,
            "selection_rule": "screen tuple, 16384 on checkpoint ties, learner 0 on finalist ties",
            "blind_finalist": blind_finalist,
        }
        dump_json(selection_path, selection)
        path = self.root / "finalists" / f"{screen_decision_path.stem}-{wrapper['source_learner_seed']}-finalist.json"
        value = {
            "schema_version": 1, "receipt_type": compare.FINALIST_TYPE,
            "study_protocol_sha256": self.protocol_sha, "partition": "screen",
            "source_learner_seed": wrapper["source_learner_seed"], "arm": wrapper["arm"],
            "actor_sha256": wrapper["actor_sha256"], "checkpoint_sha256": wrapper["checkpoint_sha256"],
            "checkpoint_online_step": wrapper["checkpoint_online_step"],
            "screen_receipt_sha256": digest(screen_path.read_bytes()),
            "screen_decision_path": str(screen_decision_path.resolve()),
            "screen_decision_sha256": screen_decision_sha,
            "selection_file_path": str(selection_path.resolve()),
            "selection_file_sha256": digest(selection_path.read_bytes()),
            "fixed_at_utc": "2026-01-01T12:00:00Z",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        compare.write_sealed(path, value)
        return path

    def test_load_protocol_rejects_unsupported_fields(self) -> None:
        protocol = json.loads(self.fixture.protocol_path.read_text())
        protocol["unrecognized"] = True
        path = self.root / "bad-protocol.json"
        dump_json(path, protocol)
        with self.assertRaisesRegex(ValueError, "unsupported|missing"):
            compare.load_protocol(path, self.root)


if __name__ == "__main__":
    unittest.main()
