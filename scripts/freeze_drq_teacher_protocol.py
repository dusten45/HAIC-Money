"""Freeze the teacher-replay study only after source and geometry gates pass."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import gymnasium
import numpy as np
import torch

from common_adapter import ActionSpec, ObservationSpec
from haic.algorithms.drq_v2.teacher_replay import audit_source_actor_pair
from scripts.audit_drq_teacher_geometry import audit_geometry
from scripts.validate_drq_teacher_protocol import (
    ProtocolError,
    _agent_drq_compatibility_record,
    _adapter_compatibility_record,
    _evaluation_harness_record,
    _geometry_seeds_from_ledger,
    sha256_file,
    validate_protocol_file,
)


STUDY_ID = "drqv2-teacher-replay-v1-r3"
SOURCE_RUN = Path("runs/20260922-drq-augmentation-pad-v1-restart")
PROTOCOL_PATH = Path("experiments/drqv2-teacher-replay-v1-r3.json")
AUDIT_PATH = Path("experiments/drqv2-teacher-replay-v1-r3-geometry-audit.json")
EVALUATION_PYTHON = Path("/tmp/kilo/haic-cpu21/bin/python")
RUN_ROOT = Path("runs/20260924-drqv2-teacher-replay-v1-r3")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_actor_paths(learner_seed: int) -> dict[str, Path]:
    run = SOURCE_RUN / f"control-seed{learner_seed}"
    checkpoint_dir = run / "checkpoints/step-000131072"
    return {
        "run_config": run / "config.json",
        "checkpoint": checkpoint_dir / "checkpoint.pt",
        "checkpoint_manifest": checkpoint_dir / "checkpoint.manifest.json",
        "actor": checkpoint_dir / "actor.pt",
        "episodes": run / "episodes.jsonl",
    }


def _audit_source_agent_actions(source_agent_path: Path, study_agent_path: Path,
                               actor_path: Path) -> dict[str, Any]:
    from agent import Agent as StudyAgent

    spec = importlib.util.spec_from_file_location("drq_source_agent_snapshot", source_agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load original Agent source snapshot: {source_agent_path}")
    source_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source_module)
    payload = torch.load(actor_path, map_location="cpu", weights_only=False)
    source_agent = object.__new__(source_module.Agent)
    study_agent = object.__new__(StudyAgent)
    source_agent._init_drq(payload)
    study_agent._init_drq(payload)
    observations = [
        np.zeros((4, 84, 84), dtype=np.float32),
        np.full((4, 84, 84), 0.5, dtype=np.float32),
        np.ones((4, 84, 84), dtype=np.float32),
    ]
    for observation in observations:
        source_action = source_agent.act(observation)
        study_action = study_agent.act(observation)
        if not np.array_equal(source_action, study_action):
            raise ValueError("source-era and study Agent produce different DrQ CPU actions")
    return {
        "source_sha256": sha256_file(source_agent_path),
        "study_sha256": sha256_file(study_agent_path),
        "observations": len(observations),
        "actions_exactly_equal": True,
    }


def _source_records(repo_root: Path, fingerprints: dict[str, str]) -> list[dict[str, Any]]:
    records = []
    for learner_seed in (0, 1):
        paths = _source_actor_paths(learner_seed)
        resolved = {name: (repo_root / path).resolve() for name, path in paths.items()}
        for name, path in resolved.items():
            if not path.is_file():
                raise FileNotFoundError(f"source learner {learner_seed} missing {name}: {path}")
        config = json.loads(resolved["run_config"].read_text(encoding="utf-8"))
        git = config.get("git", {})
        run_config = config.get("config", {})
        drq = run_config.get("drq_config", {})
        if run_config.get("algorithm") != "drq-v2" or run_config.get("seed") != learner_seed:
            raise ValueError(f"source learner {learner_seed} run-config identity mismatch")
        if run_config.get("total_steps") != 131072 or run_config.get("track_ids") != [1, 2, 3, 4]:
            raise ValueError(f"source learner {learner_seed} budget/track contract mismatch")
        if drq.get("augmentation_pad") != 4:
            raise ValueError(f"source learner {learner_seed} is not the pad-4 control")
        if not isinstance(git.get("commit"), str) or type(git.get("dirty")) is not bool:
            raise ValueError(f"source learner {learner_seed} has no frozen code revision")

        checkpoint_manifest = json.loads(resolved["checkpoint_manifest"].read_text(encoding="utf-8"))
        checkpoint_hash = sha256_file(resolved["checkpoint"])
        actor_hash = sha256_file(resolved["actor"])
        manifest_extra = checkpoint_manifest.get("extra", {})
        if (
            checkpoint_manifest.get("algorithm") != "drq-v2"
            or not checkpoint_manifest.get("source_hashes")
            or manifest_extra.get("environment_steps") != 131072
        ):
            raise ValueError(f"source learner {learner_seed} checkpoint manifest is incomplete")
        source_dir = SOURCE_RUN / "source"
        source_dir_resolved = (repo_root / source_dir).resolve()
        training_hashes = dict(run_config.get("training_source_sha256", {}))
        checkpoint_source_hashes = {
            Path(path).resolve().relative_to(source_dir_resolved).as_posix(): value
            for path, value in checkpoint_manifest.get("source_hashes", {}).items()
        }
        if (
            not training_hashes
            or not checkpoint_source_hashes
            or manifest_extra.get("training_source_sha256") != training_hashes
        ):
            raise ValueError(f"source learner {learner_seed} training source hashes are incomplete")
        combined_source_hashes = dict(training_hashes)
        for relative_path, value in checkpoint_source_hashes.items():
            if relative_path in combined_source_hashes and combined_source_hashes[relative_path] != value:
                raise ValueError(f"source learner {learner_seed} has conflicting code hashes: {relative_path}")
            combined_source_hashes[relative_path] = value
        for relative_path, expected_hash in combined_source_hashes.items():
            source_copy = (source_dir_resolved / relative_path).resolve()
            root_copy = (repo_root / relative_path).resolve()
            compatible_source_difference = False
            if relative_path == "common_adapter.py" and source_copy.is_file() and root_copy.is_file():
                compatibility = _adapter_compatibility_record(source_copy, root_copy)
                compatible_source_difference = (
                    compatibility["source_sha256"] == expected_hash
                    and compatibility["study_sha256"] == sha256_file(root_copy)
                )
            elif relative_path == "agent.py" and source_copy.is_file() and root_copy.is_file():
                compatibility = _agent_drq_compatibility_record(source_copy, root_copy)
                compatible_source_difference = (
                    compatibility["source_sha256"] == expected_hash
                    and compatibility["study_sha256"] == sha256_file(root_copy)
                )
            elif relative_path == "evaluate_policy.py" and source_copy.is_file() and root_copy.is_file():
                compatibility = _evaluation_harness_record(source_copy, root_copy)
                compatible_source_difference = (
                    compatibility["source_sha256"] == expected_hash
                    and compatibility["study_sha256"] == sha256_file(root_copy)
                )
            if (
                not source_copy.is_relative_to(source_dir_resolved)
                or not source_copy.is_file()
                or sha256_file(source_copy) != expected_hash
                or not root_copy.is_file()
                or (sha256_file(root_copy) != expected_hash and not compatible_source_difference)
            ):
                raise ValueError(
                    f"source learner {learner_seed} code snapshot differs from current study code: {relative_path}"
                )
        pair_audit = audit_source_actor_pair(
            resolved["checkpoint"],
            resolved["actor"],
            learner_seed=learner_seed,
            source_revision=git["commit"],
            expected_checkpoint_sha256=checkpoint_hash,
            expected_actor_sha256=actor_hash,
        )
        pair_audit["agent_source_parity"] = _audit_source_agent_actions(
            repo_root / SOURCE_RUN / "source/agent.py",
            repo_root / "agent.py",
            resolved["actor"],
        )
        if (
            pair_audit["action_fingerprint"] != fingerprints["action"]
            or pair_audit["observation_fingerprint"] != fingerprints["observation"]
        ):
            raise ValueError(f"source learner {learner_seed} actor fingerprint mismatch")
        records.append({
            "learner_seed": learner_seed,
            "checkpoint_path": paths["checkpoint"].as_posix(),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_manifest_path": paths["checkpoint_manifest"].as_posix(),
            "checkpoint_manifest_sha256": sha256_file(resolved["checkpoint_manifest"]),
            "actor_path": paths["actor"].as_posix(),
            "actor_sha256": actor_hash,
            "run_config_path": paths["run_config"].as_posix(),
            "run_config_sha256": sha256_file(resolved["run_config"]),
            "episodes_path": paths["episodes"].as_posix(),
            "episodes_sha256": sha256_file(resolved["episodes"]),
            "source_training_snapshot_dir": source_dir.as_posix(),
            "source_training_hashes": training_hashes,
            "checkpoint_source_hashes": checkpoint_source_hashes,
            "source_revision": git["commit"],
            "source_dirty": git["dirty"],
            "training_steps": 131072,
            "training_track_ids": [1, 2, 3, 4],
            "training_geometry_seeds": sorted(_geometry_seeds_from_ledger(resolved["episodes"])),
            "source_pair_audit": pair_audit,
        })
    return records


def _source_snapshot_paths(repo_root: Path) -> list[str]:
    paths = {
        "agent.py", "action_representation.py", "action_smoothing.py", "common_adapter.py",
        "damage.py", "drq_v2.py", "env_wrapper.py", "evaluate_policy.py",
        "package_submission.py", "train.py", "train_drqv2.py", "tracking.py",
        "requirements.txt", "AGENTS.md", "docs/plans/drqv2-teacher-replay-plan.md",
        "docs/evaluation/protocol.md", "docs/evaluation/generalization-policy.md",
        "haic/__init__.py", "haic/algorithms/__init__.py",
        "haic/algorithms/drq_v2/__init__.py",
        "haic/algorithms/drq_v2/teacher_replay.py",
        "haic/algorithms/drq_v2/teacher_study.py",
        "scripts/audit_drq_teacher_geometry.py",
        "scripts/collect_drq_teacher.py",
        "scripts/validate_drq_teacher_protocol.py",
        "scripts/freeze_drq_teacher_protocol.py",
        "scripts/train_drq_teacher_replay.py",
        "scripts/compare_drq_teacher_replay.py",
        "scripts/run_drq_teacher_evaluations.py",
        "tests/test_drq_teacher_geometry_audit.py",
        "tests/test_drq_teacher_study.py",
        "tests/test_drq_teacher_training.py",
        "tests/test_drq_teacher_replay.py",
        "tests/test_drq_teacher_receipts.py",
        "tests/test_drq_teacher_orchestrator.py",
    }
    core = repo_root / "core"
    if core.is_dir():
        paths.update(path.relative_to(repo_root).as_posix() for path in core.rglob("*.py"))
    missing = sorted(path for path in paths if not (repo_root / path).is_file())
    if missing:
        raise FileNotFoundError(f"study source snapshots are missing: {missing}")
    return sorted(paths)


def _hash_source_snapshots(repo_root: Path) -> dict[str, str]:
    return {path: sha256_file(repo_root / path) for path in _source_snapshot_paths(repo_root)}


def _package_versions() -> dict[str, str]:
    names = (
        "torch", "gymnasium", "numpy", "opencv-python", "stable-baselines3",
        "box2d-py", "pygame",
    )
    versions = {name: importlib.metadata.version(name) for name in names}
    versions["python"] = platform.python_version()
    return versions


def _evaluation_runtime() -> dict[str, Any]:
    if not EVALUATION_PYTHON.is_file():
        raise FileNotFoundError(f"pinned CPU evaluation interpreter is missing: {EVALUATION_PYTHON}")
    code = (
        "import importlib.metadata as m,json,platform,sys,torch; "
        "names=('torch','gymnasium','numpy','opencv-python','stable-baselines3','box2d-py','pygame'); "
        "print(json.dumps({'python':platform.python_version(),'executable':sys.executable,"
        "'platform':sys.platform,'cuda_available':torch.cuda.is_available(),"
        "'torch_cuda':torch.version.cuda,'dependencies':{n:m.version(n) for n in names}}))"
    )
    output = subprocess.run(
        [str(EVALUATION_PYTHON), "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = json.loads(output.stdout)
    if runtime["dependencies"].get("torch") != "2.1.0+cpu" or runtime["cuda_available"]:
        raise ValueError("CPU evaluator environment is not the pinned Torch 2.1 CPU runtime")
    runtime["torch"] = runtime["dependencies"]["torch"]
    return runtime


def _git_identity(repo_root: Path) -> tuple[str, bool, str]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--branch"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout
    return revision, bool(status.strip()), _sha256_bytes(status.encode("utf-8"))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with partial.open("xb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    os.replace(partial, path)


def build_protocol(repo_root: Path) -> dict[str, Any]:
    if torch.cuda.device_count() < 1 or not torch.cuda.is_available():
        raise RuntimeError("frozen GPU training runtime is unavailable")
    if not (repo_root / EVALUATION_PYTHON).is_file():
        raise FileNotFoundError(EVALUATION_PYTHON)
    if (repo_root / AUDIT_PATH).exists() or (repo_root / PROTOCOL_PATH).exists():
        raise FileExistsError("refusing to overwrite an existing frozen audit or study protocol")
    screen_seeds = list(range(4_260_300_001, 4_260_300_009))
    confirmation_seeds = list(range(4_260_400_001, 4_260_400_009))
    blind_seeds = list(range(4_260_500_001, 4_260_500_009))
    teacher_seeds = list(range(4_260_100_001, 4_260_100_009))
    online_seeds = list(range(4_260_200_001, 4_260_200_009))
    candidates = sorted({*teacher_seeds, *online_seeds, *screen_seeds,
                         *confirmation_seeds, *blind_seeds})
    exclusions = {AUDIT_PATH.as_posix(), PROTOCOL_PATH.as_posix()}
    report = audit_geometry(repo_root, candidates, exclude_paths=exclusions)
    if not report["passed"]:
        raise ValueError(f"fresh geometry audit failed: {report['parse_errors']} {report['candidate_hits']}")
    action_fingerprint = ActionSpec().fingerprint
    observation_fingerprint = ObservationSpec().fingerprint
    fingerprints = {"action": action_fingerprint, "observation": observation_fingerprint}
    sources = _source_records(repo_root, fingerprints)
    prior_seeds = set(report["known_excluded_geometry_seeds"])
    source_training_seeds = {
        seed for source in sources for seed in source["training_geometry_seeds"]
    }
    eval_seeds = set(screen_seeds + confirmation_seeds + blind_seeds)
    train_seeds = set(teacher_seeds + online_seeds)
    if candidates != sorted(train_seeds | eval_seeds):
        raise RuntimeError("candidate seed registry does not match the five frozen study pools")
    if candidates and any(seed in prior_seeds for seed in candidates):
        raise ValueError("a selected study geometry is present in recorded prior evidence")
    if eval_seeds & source_training_seeds or eval_seeds & train_seeds:
        raise ValueError("evaluation geometries overlap source or study training geometries")
    if set(teacher_seeds) & set(online_seeds):
        raise ValueError("teacher and online training geometry pools overlap")

    source_revision, worktree_dirty, status_sha = _git_identity(repo_root)
    evaluation_runtime = _evaluation_runtime()
    training_dependencies = _package_versions()
    hardware = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "cuda_runtime": str(torch.version.cuda),
        "host_platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    runtime = {
        "training_device": "cuda:0",
        "training_executable": sys.executable,
        "training_python": platform.python_version(),
        "dependencies": training_dependencies,
        "hardware": hardware,
        "evaluation_executable": str(EVALUATION_PYTHON),
        "evaluation_runtime": evaluation_runtime,
        "worker_limits": {
            "actor_load_seconds_max": 10,
            "reset_act_seconds_max": 5,
            "peak_rss_bytes_max": 1073741824,
            "workers": 1,
            "frame_skip": 4,
            "max_steps": 2000,
        },
    }
    source_snapshots = _hash_source_snapshots(repo_root)
    source_compatibility = sorted([
        _agent_drq_compatibility_record(
            repo_root / SOURCE_RUN / "source/agent.py", repo_root / "agent.py"
        ),
        _adapter_compatibility_record(
            repo_root / SOURCE_RUN / "source/common_adapter.py", repo_root / "common_adapter.py"
        ),
        _evaluation_harness_record(
            repo_root / SOURCE_RUN / "source/evaluate_policy.py", repo_root / "evaluate_policy.py"
        ),
    ], key=lambda row: row["path"])
    known_excluded = sorted(prior_seeds)
    reserved = sorted(prior_seeds | source_training_seeds | eval_seeds)
    _atomic_json(repo_root / AUDIT_PATH, report)
    audit_sha = sha256_file(repo_root / AUDIT_PATH)
    return {
        "name": STUDY_ID,
        "purpose": "research",
        "frame_skip": 4,
        "max_steps": 2000,
        "partitions": {
            "screen": {"track_ids": [101, 102, 103], "seeds": screen_seeds, "repeats": 2},
            "confirmation": {
                "track_ids": [111, 112, 113, 114], "seeds": confirmation_seeds, "repeats": 2,
            },
            "blind": {"track_ids": [121, 122, 123], "seeds": blind_seeds, "repeats": 2},
        },
        "reserved_training_seeds": reserved,
        "schema_version": 1,
        "study_id": STUDY_ID,
        "run_root": RUN_ROOT.as_posix(),
        "hypothesis": (
            "A fixed mixture of complete pad-4 DrQ source-actor transitions improves both "
            "source-seed actors on one shared fresh screen and independent confirmation pool, "
            "compared with a matched online-only DrQ fine-tune."
        ),
        "source_revision": source_revision,
        "working_tree_dirty": worktree_dirty,
        "working_tree_status_sha256": status_sha,
        "spec_fingerprints": fingerprints,
        "source_snapshots": source_snapshots,
        "source_compatibility": source_compatibility,
        "source_actors": sources,
        "training_pools": {
            "teacher_training": {
                "track_ids": [1, 2, 3, 4], "seeds": teacher_seeds, "sampler_seed": 917001,
            },
            "online_training": {
                "track_ids": [1, 2, 3, 4], "seeds": online_seeds, "sampler_seed": 917002,
            },
        },
        "known_excluded_geometry_seeds": known_excluded,
        "geometry_audit": {
            "report_path": AUDIT_PATH.as_posix(),
            "report_sha256": audit_sha,
            "candidate_seeds": candidates,
            "source_snapshot_count": report["source_snapshot_count"],
            "known_excluded_geometry_seed_count": len(known_excluded),
            "global_freshness_claim": report["global_freshness_claim"],
        },
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
            "algorithm": "DrQ-v2",
            "padding": 4,
            "feature_dim": 256,
            "hidden_dim": 256,
            "n_step": 3,
            "gamma": 0.99,
            "actor_lr": 0.0001,
            "critic_lr": 0.0001,
            "tau": 0.01,
            "target_update_frequency": 2,
            "actor_update_frequency": 2,
            "target_noise_std": 0.2,
            "target_noise_clip": 0.5,
            "steering_logit_l2": 0.0,
            "reward_shaping": False,
            "reward_normalization": False,
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
        "runtime": runtime,
        "selection": {
            "screen_cells": 24,
            "confirmation_cells": 32,
            "blind_cells": 24,
            "repeats_per_cell": 2,
            "canonical_repeat": 0,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    protocol_path = repo_root / PROTOCOL_PATH
    gate_path = repo_root / RUN_ROOT / "a0-collection-gate.json"
    try:
        if protocol_path.exists():
            raise FileExistsError(f"refusing to overwrite frozen protocol: {protocol_path}")
        if (repo_root / RUN_ROOT).exists() or gate_path.exists():
            raise FileExistsError(f"study run root already exists: {repo_root / RUN_ROOT}")
        protocol = build_protocol(repo_root)
        _atomic_json(protocol_path, protocol)
        validation = validate_protocol_file(
            protocol_path, repo_root, audit_live_geometry=True
        )
        live_audit = validation.get("live_geometry_audit")
        if not isinstance(live_audit, dict) or live_audit.get("passed") is not True:
            raise ValueError("A0 live geometry gate did not return a sealed pass")
        gate = {
            "format": "haic-drq-teacher-replay-a0-collection-gate-v1",
            "status": "pass",
            "study_id": protocol["study_id"],
            "study_protocol_path": PROTOCOL_PATH.as_posix(),
            "study_protocol_sha256": sha256_file(protocol_path),
            "geometry_audit_path": protocol["geometry_audit"]["report_path"],
            "geometry_audit_sha256": protocol["geometry_audit"]["report_sha256"],
            "known_excluded_geometry_seeds_sha256": _sha256_bytes(
                json.dumps(protocol["known_excluded_geometry_seeds"], separators=(",", ":")).encode()
            ),
            "live_geometry_audit": live_audit,
        }
        _atomic_json(gate_path, gate)
    except (OSError, ValueError, ProtocolError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "protocol_path": PROTOCOL_PATH.as_posix(),
        "protocol_sha256": sha256_file(protocol_path),
        "a0_collection_gate_path": gate_path.relative_to(repo_root).as_posix(),
        "a0_collection_gate_sha256": sha256_file(gate_path),
        "source_revision": protocol["source_revision"],
        "source_snapshot_count": len(protocol["source_snapshots"]),
        "known_excluded_geometry_seed_count": len(protocol["known_excluded_geometry_seeds"]),
        "partition_cell_counts": validation["partition_cell_counts"],
        "training_pool_geometry_seed_counts": validation["training_pool_geometry_seed_counts"],
        "live_geometry_audit": "pass",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
