"""Run one frozen, matched DrQ-v2 online-only or teacher-replay arm."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from common_adapter import EpisodeCollector
from drq_v2 import DrQv2Agent, DrQv2Config
from haic.algorithms.drq_v2.teacher_replay import (
    GeometryPoolRNG,
    RNGStreams,
    TeacherDataset,
    TwoSourceReplay,
    audit_source_actor_pair,
    find_sampled_track_env,
    fork_from_source,
    learner_rng_state,
    restore_learner_rng_state,
    update_from_mixture,
)
from scripts.validate_drq_teacher_protocol import ProtocolError, validate_protocol_file
from train import build_sampled_env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_file(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError("protocol source paths must be repository-relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"invalid source path in protocol: {relative_path}")
    return resolved


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as destination:
        destination.write(data)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    _atomic_bytes(path, data)


def _agent_config(protocol: dict[str, Any]) -> DrQv2Config:
    learner = protocol["learner"]
    budgets = protocol["budgets"]
    return DrQv2Config(
        observation_shape=tuple(protocol["environment"]["observation_shape"]),
        action_dim=3,
        feature_dim=learner["feature_dim"],
        hidden_dim=learner["hidden_dim"],
        replay_capacity=budgets["online_replay_capacity"],
        batch_size=budgets["batch_size"],
        warmup_steps=budgets["online_startup_decisions_without_updates"],
        n_step=learner["n_step"],
        gamma=learner["gamma"],
        actor_learning_rate=learner["actor_lr"],
        critic_learning_rate=learner["critic_lr"],
        steering_logit_l2=learner["steering_logit_l2"],
        tau=learner["tau"],
        actor_update_frequency=learner["actor_update_frequency"],
        target_update_frequency=learner["target_update_frequency"],
        augmentation_pad=learner["padding"],
        exploration_initial_std=0.2,
        exploration_final_std=0.05,
        exploration_duration=100_000,
        target_policy_noise=learner["target_noise_std"],
        target_policy_noise_clip=learner["target_noise_clip"],
        device=protocol["runtime"]["training_device"],
    )


def _derived_seeds(study_seed: int) -> dict[str, int]:
    children = np.random.SeedSequence(study_seed).spawn(3)
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(("exploration", "online_replay", "mixture"), children)
    }


def _load_collection(root: Path, protocol: dict[str, Any], collection_result_path: Path,
                     teacher_dataset_path: Path | None, protocol_sha: str,
                     source: dict[str, Any], *, load_dataset: bool
                     ) -> tuple[dict[str, Any], TeacherDataset | None]:
    result_path = collection_result_path if collection_result_path.is_absolute() else root / collection_result_path
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("study_protocol_sha256") != protocol_sha:
        raise ValueError("teacher collection belongs to another frozen protocol")
    if result.get("a0_collection_gate_path") is None or result.get("a0_collection_gate_sha256") is None:
        raise ValueError("teacher collection has no frozen A0 freshness-gate lineage")
    gate_path = _repo_file(root, result["a0_collection_gate_path"])
    if _sha256(gate_path) != result["a0_collection_gate_sha256"]:
        raise ValueError("teacher collection A0 gate file hash mismatch")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("study_protocol_sha256") != protocol_sha or gate.get("status") != "pass":
        raise ValueError("teacher collection A0 gate is not a pass for this protocol")
    live_audit = gate.get("live_geometry_audit", {})
    candidate_geometry = sorted({
        seed for pool in protocol["training_pools"].values() for seed in pool["seeds"]
    } | {
        seed for partition in protocol["partitions"].values() for seed in partition["seeds"]
    })
    if (
        live_audit.get("passed") is not True
        or live_audit.get("candidate_seeds") != candidate_geometry
        or live_audit.get("parse_errors")
        or any(live_audit.get("candidate_hits", {}).values())
        or live_audit.get("known_excluded_geometry_seeds")
        != protocol["known_excluded_geometry_seeds"]
    ):
        raise ValueError("collection result A0 gate does not cover every frozen geometry")
    if result.get("learner_seed") != source["learner_seed"]:
        raise ValueError("teacher collection source learner does not match this arm")
    if result.get("source_actor_sha256") != source["actor_sha256"]:
        raise ValueError("teacher collection actor identity does not match this source")
    if not result.get("coverage_pass") or result.get("coverage_gate") != "pass":
        raise ValueError("teacher finish-coverage gate did not pass; training is prohibited")
    if result.get("finished_geometry_seed_count", 0) < 4:
        raise ValueError("teacher collection has fewer than four distinct finished geometries")
    if result.get("decisions") != result.get("decision_cap"):
        raise ValueError("teacher collection did not spend the exact preregistered decision budget")
    if result.get("dataset_path") is None or result.get("dataset_digest") is None:
        raise ValueError("teacher collection has no sealed dataset")
    data_path = _repo_file(root, result["dataset_path"])
    if teacher_dataset_path is not None:
        requested_dataset = teacher_dataset_path if teacher_dataset_path.is_absolute() else root / teacher_dataset_path
        if requested_dataset.resolve() != data_path.resolve():
            raise ValueError("--teacher-dataset path differs from its sealed collection result")
    if load_dataset and teacher_dataset_path is None:
        raise ValueError("teacher-replay requires --teacher-dataset")
    if _sha256(data_path) != result.get("dataset_file_sha256"):
        raise ValueError("sealed teacher dataset file hash does not match collection result")
    result["result_sha256"] = _sha256(result_path)
    dataset = (
        TeacherDataset.from_bytes(data_path.read_bytes(), expected_digest=result["dataset_digest"])
        if load_dataset else None
    )
    if dataset is None:
        return result, None
    for episode in dataset.episodes:
        if episode.source_actor_sha256 != source["actor_sha256"]:
            raise ValueError("teacher dataset contains a foreign source actor")
        if episode.track_id not in (1, 2, 3, 4):
            raise ValueError("teacher dataset contains a non-training track ID")
        if int(episode.geometry_id) not in set(result["training_pool"]["seeds"]):
            raise ValueError("teacher dataset contains a geometry outside its frozen pool")
    return result, dataset


def _reset_metadata(run_dir: Path) -> dict[int, dict[str, int]]:
    path = run_dir / "episodes.jsonl"
    metadata: dict[int, dict[str, int]] = {}
    if not path.exists():
        return metadata
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") in {"reset", "resume-reset"}:
                metadata[int(row["episode_id"])] = {
                    "geometry_seed": int(row["seed"]),
                    "track_id": int(row["track_id"]),
                }
    return metadata


def _empty_sample_trace(total_updates: int, batch_size: int) -> dict[str, np.ndarray]:
    return {
        "source_codes": np.zeros((total_updates, batch_size), dtype=np.uint8),
        "source_indices": np.zeros((total_updates, batch_size), dtype=np.int64),
        "source_episode_ids": np.full((total_updates, batch_size), -1, dtype=np.int64),
        "geometry_seeds": np.zeros((total_updates, batch_size), dtype=np.uint32),
        "track_ids": np.zeros((total_updates, batch_size), dtype=np.uint8),
        "source_steps": np.full((total_updates, batch_size), -1, dtype=np.int32),
    }


def _record_sample_trace(trace: dict[str, np.ndarray], offset: int, batch: dict[str, Any],
                         episode_metadata: dict[int, dict[str, int]]) -> None:
    trace["source_codes"][offset] = batch["source"]
    trace["source_indices"][offset] = batch["source_indices"]
    for row, tag in enumerate(batch["source_tags"]):
        if tag["source"] == "online":
            episode_id = int(tag["episode_id"])
            metadata = episode_metadata.get(episode_id)
            if metadata is None:
                raise ValueError(f"sampled online replay row has no reset ledger: episode {episode_id}")
            trace["source_episode_ids"][offset, row] = episode_id
            trace["geometry_seeds"][offset, row] = metadata["geometry_seed"]
            trace["track_ids"][offset, row] = metadata["track_id"]
            trace["source_steps"][offset, row] = int(tag["source_index"])
        elif tag["source"] == "teacher":
            trace["source_episode_ids"][offset, row] = int(tag["episode_id"])
            trace["geometry_seeds"][offset, row] = int(tag["geometry_id"])
            trace["track_ids"][offset, row] = int(tag["track_id"])
            trace["source_steps"][offset, row] = int(tag["step"])
        else:
            raise ValueError(f"unsupported replay source tag: {tag}")


def _save_sample_trace(run_dir: Path, trace: dict[str, np.ndarray], count: int,
                       protocol_sha: str, dataset_digest: str | None,
                       additional_step: int) -> tuple[Path, str]:
    path = run_dir / f"replay-sample-trace-step-{additional_step:09d}.npz"
    stream = __import__("io").BytesIO()
    arrays = {name: value[:count] for name, value in trace.items()}
    arrays["study_protocol_sha256"] = np.frombuffer(protocol_sha.encode("ascii"), dtype=np.uint8)
    arrays["teacher_dataset_digest"] = np.frombuffer((dataset_digest or "").encode("ascii"), dtype=np.uint8)
    np.savez_compressed(stream, **arrays)
    data = stream.getvalue()
    _atomic_bytes(path, data)
    return path, _sha256(path)


def _load_sample_trace(path: Path, total_updates: int, protocol_sha: str,
                       dataset_digest: str | None) -> tuple[dict[str, np.ndarray], int]:
    trace = _empty_sample_trace(total_updates, 64)
    if not path.is_file():
        return trace, 0
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(trace) | {"study_protocol_sha256", "teacher_dataset_digest"}:
            raise ValueError("sample trace has missing or unsupported fields")
        if archive["study_protocol_sha256"].tobytes().decode("ascii") != protocol_sha:
            raise ValueError("sample trace belongs to another study protocol")
        if archive["teacher_dataset_digest"].tobytes().decode("ascii") != (dataset_digest or ""):
            raise ValueError("sample trace teacher dataset identity does not match")
        rows = int(archive["source_codes"].shape[0])
        if rows > total_updates:
            raise ValueError("sample trace exceeds preregistered gradient-update count")
        for name in trace:
            values = np.asarray(archive[name])
            if values.shape != (rows, 64) or values.dtype != trace[name].dtype:
                raise ValueError(f"sample trace field {name} has invalid shape or dtype")
            trace[name][:rows] = values
    return trace, rows


def _save_candidate(agent: DrQv2Agent, *, protocol: dict[str, Any], protocol_sha: str,
                    source: dict[str, Any], arm: str, run_dir: Path, root: Path, additional_step: int,
                    collection_result: dict[str, Any], dataset: TeacherDataset | None,
                    mixture: TwoSourceReplay, rng_streams: RNGStreams,
                    pool_rng: GeometryPoolRNG,
                    transition_done: bool, next_episode_id: int, trace_sha256: str,
                    trace_path: str) -> dict[str, Any]:
    online_step_name = f"step-{additional_step:09d}"
    checkpoint_dir = run_dir / "checkpoints" / online_step_name
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    dataset_digest = dataset.digest if dataset is not None else None
    resume_allowed = bool(
        transition_done
        and additional_step < protocol["budgets"]["additional_online_decisions_per_arm_source"]
    )
    trainer_state = {
        "format": "haic-drq-teacher-trainer-v1",
        "study_protocol_sha256": protocol_sha,
        "source_learner_seed": source["learner_seed"],
        "source_revision": source["source_revision"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_actor_sha256": source["actor_sha256"],
        "arm": arm,
        "additional_online_steps": additional_step,
        "study_gradient_steps": agent.gradient_steps,
        "teacher_dataset_digest": dataset_digest,
        "teacher_collection_result_sha256": collection_result["result_sha256"],
        "two_source_replay": mixture.state_dict(),
        "learner_update_rng": rng_streams.state_dict(),
        "geometry_pool_rng": pool_rng.state_dict(),
        "global_learner_rng": learner_rng_state(agent),
        "replay_sample_trace_sha256": trace_sha256,
        "replay_sample_trace_path": trace_path,
        "resume_allowed": resume_allowed,
        "checkpoint_at_episode_boundary": bool(transition_done),
        "next_episode_id": int(next_episode_id) if resume_allowed else None,
        "environment_snapshot": None,
        "resume_contract": "restart the wrapper at the next episode boundary only",
    }
    source_paths = [_repo_file(root, path) for path in protocol["source_snapshots"]]
    run_metadata = {
        "study_protocol_sha256": protocol_sha,
        "study_id": protocol["study_id"],
        "source_learner_seed": source["learner_seed"],
        "source_revision": source["source_revision"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_actor_sha256": source["actor_sha256"],
        "arm": arm,
        "additional_online_step": additional_step,
        "online_pool": protocol["training_pools"]["online_training"],
        "teacher_dataset_digest": dataset_digest,
        "reward_contract": {"raw_reward": True, "shaping": False, "normalization": False},
        "max_steps": protocol["max_steps"],
        "seeds": protocol["training_pools"]["online_training"]["seeds"],
        "weight_only_fork": True,
        "source_environment_steps_inherited_for_exploration_schedule": 131072,
    }
    checkpoint_path = agent.save_checkpoint(
        checkpoint_dir / "checkpoint.pt",
        source_paths=source_paths,
        run_metadata=run_metadata,
        trainer_state=trainer_state,
    )
    actor_path = agent.export_actor(checkpoint_dir / "actor.pt")
    checkpoint_sha256 = _sha256(checkpoint_path)
    actor_sha256 = _sha256(actor_path)
    checkpoint_manifest_path = checkpoint_path.with_suffix(".manifest.json")
    checkpoint_manifest_sha256 = _sha256(checkpoint_manifest_path)
    return {
        "source_learner_seed": source["learner_seed"],
        "arm": arm,
        "checkpoint_online_step": additional_step,
        "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_manifest_path": checkpoint_manifest_path.relative_to(root).as_posix(),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "actor_path": actor_path.relative_to(root).as_posix(),
        "actor_sha256": actor_sha256,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_actor_sha256": source["actor_sha256"],
        "study_protocol_sha256": protocol_sha,
        "teacher_dataset_digest": dataset_digest,
        "study_gradient_steps": agent.gradient_steps,
        "resume_allowed": resume_allowed,
        "replay_sample_trace_path": trace_path,
        "replay_sample_trace_sha256": trace_sha256,
    }


def train_arm(protocol: dict[str, Any], protocol_path: Path, root: Path, learner_seed: int,
              arm: str, collection_result_path: Path, teacher_dataset_path: Path | None,
              run_dir: Path,
              resume_checkpoint: Path | None = None) -> dict[str, Any]:
    protocol_sha = _sha256(protocol_path)
    source = next(item for item in protocol["source_actors"] if item["learner_seed"] == learner_seed)
    if arm not in {"online-only", "teacher-replay"}:
        raise ValueError("arm must be online-only or teacher-replay")
    collection_result, collected_dataset = _load_collection(
        root, protocol, collection_result_path, teacher_dataset_path, protocol_sha,
        source, load_dataset=arm == "teacher-replay"
    )
    mode = "online-only" if arm == "online-only" else "teacher-replay"
    dataset = collected_dataset if arm == "teacher-replay" else None
    if arm == "teacher-replay" and dataset is None:
        raise ValueError("teacher-replay arm requires its sealed source dataset")

    source_checkpoint = _repo_file(root, source["checkpoint_path"])
    source_actor = _repo_file(root, source["actor_path"])
    config = _agent_config(protocol)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("frozen GPU training runtime is unavailable")
    free_gpu_bytes, total_gpu_bytes = torch.cuda.mem_get_info(torch.device(config.device))
    if free_gpu_bytes < 8 * 1024**3:
        raise RuntimeError(
            f"insufficient isolated GPU memory: {free_gpu_bytes} free of {total_gpu_bytes} bytes"
        )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    audit = audit_source_actor_pair(
        source_checkpoint,
        source_actor,
        learner_seed=learner_seed,
        source_revision=source["source_revision"],
        expected_checkpoint_sha256=source["checkpoint_sha256"],
        expected_actor_sha256=source["actor_sha256"],
    )
    if audit["source_revision"] != source["source_revision"]:
        raise ValueError("source revision changed after study freeze")

    study_seed = 17_000 + learner_seed
    derived = _derived_seeds(study_seed)
    agent, fork_identity = fork_from_source(
        source_checkpoint,
        source_actor,
        config,
        learner_seed=learner_seed,
        study_seed=study_seed,
        expected_checkpoint_sha256=source["checkpoint_sha256"],
        expected_actor_sha256=source["actor_sha256"],
    )
    agent.rng = np.random.default_rng(derived["exploration"])
    agent.replay.rng = np.random.default_rng(derived["online_replay"])
    rng_streams = RNGStreams(study_seed + 1000, agent.device)
    pool = protocol["training_pools"]["online_training"]
    pool_rng = GeometryPoolRNG(pool["track_ids"], pool["seeds"], seed=pool["sampler_seed"])
    mixture = TwoSourceReplay(
        agent.replay,
        dataset,
        mode=mode,
        seed=derived["mixture"],
        online_source_id=f"online-{arm}-seed{learner_seed}",
    )
    env = build_sampled_env(
        track_ids=pool["track_ids"],
        sampler_seed=pool["sampler_seed"],
        max_steps=protocol["max_steps"],
        frame_skip=protocol["frame_skip"],
        reward_shaping=False,
        excluded_seeds=protocol["reserved_training_seeds"],
        obstacles=True,
        collision_penalty=0.0,
    )
    try:
        sampled = find_sampled_track_env(env)
    except RuntimeError:
        env.close()
        raise
    sampled._rng = pool_rng
    collector = EpisodeCollector(env, action_adapter=agent.action_adapter, gamma=config.gamma)
    run_dir = run_dir if run_dir.is_absolute() else root / run_dir
    run_dir.mkdir(parents=True, exist_ok=resume_checkpoint is not None)
    config_path = run_dir / "run-config.json"
    evaluator_config_path = run_dir / "config.json"
    resume_state: dict[str, Any] | None = None

    if resume_checkpoint is None:
        if any(run_dir.iterdir()):
            env.close()
            raise FileExistsError(f"new training run directory is not empty: {run_dir}")
        run_config = {
            "schema_version": 1,
            "study_id": protocol["study_id"],
            "study_protocol_sha256": protocol_sha,
            "source_learner_seed": learner_seed,
            "source_revision": source["source_revision"],
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "source_actor_sha256": source["actor_sha256"],
            "source_pair_audit": audit,
            "weight_only_fork": fork_identity,
            "arm": arm,
            "replay_mode": mode,
            "study_seed": study_seed,
            "derived_rng_seeds": derived,
            "agent_config": config.__dict__,
            "training_pool": pool,
            "teacher_dataset_digest": dataset.digest if dataset is not None else None,
            "teacher_collection_result_sha256": _sha256(
                collection_result_path if collection_result_path.is_absolute()
                else root / collection_result_path
            ),
            "runtime": protocol["runtime"],
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
        evaluator_config = {
            "name": f"{protocol['study_id']}-{arm}-seed{learner_seed}",
            "config": {
                "algorithm": "drq-v2",
                "seed": study_seed,
                "total_steps": 131072 + protocol["budgets"]["additional_online_decisions_per_arm_source"],
                "track_ids": pool["track_ids"],
                "track_sampler_seed": pool["sampler_seed"],
                "training_track_mode": "sampled",
                "seeds": pool["seeds"],
                "max_steps": protocol["max_steps"],
                "frame_skip": protocol["frame_skip"],
                "reward_shaping": False,
                "collision_penalty": 0.0,
                "action_smoothing": None,
                "action_control": None,
                "action_representation": None,
                "study_id": protocol["study_id"],
                "study_protocol_sha256": protocol_sha,
                "source_learner_seed": learner_seed,
                "arm": arm,
            },
        }
        _atomic_json(config_path, run_config)
        _atomic_json(evaluator_config_path, evaluator_config)
        additional_steps = 0
        next_episode_id = 0
        replay_trace, trace_rows = _empty_sample_trace(
            protocol["budgets"]["additional_online_decisions_per_arm_source"]
            - protocol["budgets"]["online_startup_decisions_without_updates"], 64
        ), 0
        if trace_rows:
            raise ValueError("new training run unexpectedly contains a sample trace")
        observation, reset_info = collector.reset(seed=None)
    else:
        resume_checkpoint = resume_checkpoint.resolve()
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        resume_state = payload.get("trainer_state")
        if not isinstance(resume_state, dict) or resume_state.get("format") != "haic-drq-teacher-trainer-v1":
            env.close()
            raise ValueError("resume checkpoint lacks full teacher-study trainer state")
        resume_state = copy.deepcopy(resume_state)
        del payload
        if not resume_state.get("resume_allowed") or not resume_state.get("checkpoint_at_episode_boundary"):
            env.close()
            raise ValueError("mid-episode checkpoints cannot be resumed")
        if (
            resume_state.get("study_protocol_sha256") != protocol_sha
            or resume_state.get("source_learner_seed") != learner_seed
            or resume_state.get("arm") != arm
            or resume_state.get("source_checkpoint_sha256") != source["checkpoint_sha256"]
            or resume_state.get("source_actor_sha256") != source["actor_sha256"]
            or resume_state.get("teacher_dataset_digest") != (dataset.digest if dataset else None)
        ):
            env.close()
            raise ValueError("resume checkpoint belongs to another frozen source, arm or dataset")
        if not config_path.is_file():
            env.close()
            raise FileNotFoundError("resume run is missing its frozen run-config.json")
        saved_run = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            saved_run.get("study_protocol_sha256") != protocol_sha
            or saved_run.get("arm") != arm
            or saved_run.get("source_learner_seed") != learner_seed
            or saved_run.get("source_actor_sha256") != source["actor_sha256"]
            or saved_run.get("teacher_dataset_digest") != (dataset.digest if dataset else None)
        ):
            env.close()
            raise ValueError("resume run-config does not match the current study")
        evaluator_config = json.loads(evaluator_config_path.read_text(encoding="utf-8"))
        if (
            evaluator_config.get("config", {}).get("study_protocol_sha256") != protocol_sha
            or evaluator_config.get("config", {}).get("source_learner_seed") != learner_seed
            or evaluator_config.get("config", {}).get("arm") != arm
        ):
            env.close()
            raise ValueError("resume evaluator config does not match the current study")
        trainer_state = agent.load_checkpoint(resume_checkpoint)
        if not isinstance(trainer_state, dict) or any(
            trainer_state.get(key) != resume_state.get(key)
            for key in (
                "format", "study_protocol_sha256", "source_learner_seed", "arm",
                "additional_online_steps", "study_gradient_steps", "teacher_dataset_digest",
            )
        ):
            env.close()
            raise ValueError("trainer state changed while loading the learner checkpoint")
        pool_rng.load_state_dict(trainer_state["geometry_pool_rng"])
        mixture.load_state_dict(trainer_state["two_source_replay"])
        rng_streams.load_state_dict(trainer_state["learner_update_rng"])
        restore_learner_rng_state(agent, trainer_state["global_learner_rng"])
        additional_steps = int(trainer_state["additional_online_steps"])
        if agent.environment_steps != 131072 + additional_steps:
            env.close()
            raise ValueError("resume learner environment step does not match study offset")
        if agent.gradient_steps != int(trainer_state["study_gradient_steps"]):
            env.close()
            raise ValueError("resume learner gradient step does not match trainer state")
        next_episode_id = int(trainer_state["next_episode_id"])
        collector.episode_id = next_episode_id - 1
        reset_info_seed = sampled._rng
        if reset_info_seed is not pool_rng:
            raise ValueError("restored reset sampler is not attached to the sampled environment")
        observation, reset_info = collector.reset(seed=None)
        trace_path = run_dir / trainer_state["replay_sample_trace_path"]
        if _sha256(trace_path) != trainer_state["replay_sample_trace_sha256"]:
            raise ValueError("resume sample-trace SHA-256 differs from checkpoint provenance")
        replay_trace, trace_rows = _load_sample_trace(
            trace_path,
            protocol["budgets"]["additional_online_decisions_per_arm_source"]
            - protocol["budgets"]["online_startup_decisions_without_updates"],
            protocol_sha,
            dataset.digest if dataset is not None else None,
        )
        expected_trace_rows = max(
            0, additional_steps - protocol["budgets"]["online_startup_decisions_without_updates"]
        )
        if trace_rows != expected_trace_rows:
            raise ValueError("resume sample trace is not complete through the episode-boundary checkpoint")

    reset_map = _reset_metadata(run_dir)
    reset_map[int(collector.episode_id)] = {
        "geometry_seed": int(reset_info["seed"]),
        "track_id": int(reset_info["track_id"]),
    }
    if reset_info["track_id"] not in pool["track_ids"] or reset_info["seed"] not in pool["seeds"]:
        env.close()
        raise ValueError("online reset sampler produced a cell outside its frozen training pool")
    episode_reward = 0.0
    episode_actions: list[np.ndarray] = []
    episode_started_step = additional_steps
    episode_started_info = dict(reset_info)
    latest_metrics: dict[str, float] = {}
    candidate_records: list[dict[str, Any]] = []
    for step in protocol["budgets"]["checkpoint_online_steps"]:
        checkpoint_dir = run_dir / "checkpoints" / f"step-{step:09d}"
        if checkpoint_dir.is_dir():
            existing = checkpoint_dir / "actor.pt"
            if existing.is_file():
                payload_actor = torch.load(existing, map_location="cpu", weights_only=False)
                candidate_records.append({
                    "source_learner_seed": learner_seed,
                    "arm": arm,
                    "checkpoint_online_step": step,
                    "actor_path": existing.relative_to(root).as_posix(),
                    "actor_sha256": _sha256(existing),
                    "checkpoint_path": (checkpoint_dir / "checkpoint.pt").relative_to(root).as_posix(),
                    "checkpoint_sha256": _sha256(checkpoint_dir / "checkpoint.pt"),
                    "checkpoint_manifest_path": (checkpoint_dir / "checkpoint.manifest.json").relative_to(root).as_posix(),
                    "checkpoint_manifest_sha256": _sha256(checkpoint_dir / "checkpoint.manifest.json"),
                    "study_protocol_sha256": protocol_sha,
                    "source_checkpoint_sha256": source["checkpoint_sha256"],
                    "source_actor_sha256": source["actor_sha256"],
                    "teacher_dataset_digest": dataset.digest if dataset else None,
                    "resume_allowed": None,
                    "actor_format": payload_actor.get("format"),
                })

    if resume_state is None:
        log_mode = "x"
    else:
        log_mode = "a"
    episode_path = run_dir / "episodes.jsonl"
    update_path = run_dir / "update-metrics.jsonl"
    reset_event = {
        "event": "reset",
        "episode_id": collector.episode_id,
        "additional_online_step": additional_steps,
        "source_learner_seed": learner_seed,
        "arm": arm,
        **reset_info,
    }
    started = time.monotonic()
    total_steps = protocol["budgets"]["additional_online_decisions_per_arm_source"]
    warmup = protocol["budgets"]["online_startup_decisions_without_updates"]
    expected_updates = total_steps - warmup
    checkpoints = set(protocol["budgets"]["checkpoint_online_steps"])
    if additional_steps >= total_steps:
        env.close()
        raise ValueError("completed training runs cannot be resumed")
    try:
        with episode_path.open(log_mode, encoding="utf-8") as episodes, update_path.open(log_mode, encoding="utf-8") as updates:
            episodes.write(json.dumps({**reset_event, "event": "resume-reset" if resume_state else "reset"},
                                      sort_keys=True) + "\n")
            episodes.flush()
            transition = None
            for additional_step in range(additional_steps + 1, total_steps + 1):
                action = agent.act(observation, deterministic=False)
                transition = collector.step(action)
                agent.observe(transition)
                episode_reward += transition.reward
                episode_actions.append(transition.action.copy())
                previous_gradients = agent.gradient_steps
                batch = None
                if additional_step > warmup:
                    batch = mixture.sample(batch_size=config.batch_size)
                    _record_sample_trace(replay_trace, trace_rows, batch, reset_map)
                    latest_metrics = update_from_mixture(agent, batch, rng_streams)
                    if agent.gradient_steps != previous_gradients + 1:
                        raise RuntimeError("scheduled learner update did not advance exactly one gradient step")
                    trace_rows += 1
                    update_record = {
                        "additional_online_step": additional_step,
                        "source_learner_seed": learner_seed,
                        "arm": arm,
                        **latest_metrics,
                        "exploration_std": agent.exploration_std(),
                        "teacher_dataset_digest": dataset.digest if dataset else None,
                    }
                    updates.write(json.dumps(update_record, sort_keys=True, allow_nan=False) + "\n")
                elif agent.gradient_steps != previous_gradients:
                    raise RuntimeError("startup interaction unexpectedly updated the learner")

                if transition.done:
                    if collector.current_observation is not None:
                        raise RuntimeError("terminal transition did not close the collector episode")
                    actions = np.asarray(episode_actions, dtype=np.float32)
                    end_record = {
                        "event": "end",
                        "episode_id": transition.episode_id,
                        "additional_online_step": additional_step,
                        "source_learner_seed": learner_seed,
                        "arm": arm,
                        "track_id": int(transition.info["track_id"]),
                        "seed": int(transition.info["seed"]),
                        "steps": transition.step + 1,
                        "reward": episode_reward,
                        "terminated": transition.terminated,
                        "truncated": transition.truncated,
                        "terminal": transition.terminal,
                        **{key: transition.info.get(key) for key in (
                            "finished", "progress", "damage", "retire_reason"
                        )},
                        "native_action_mean": actions.mean(axis=0).tolist(),
                        "native_saturation_fraction": (np.abs(actions) >= 0.99).mean(axis=0).tolist(),
                        "steering_abs_ge_0_46_fraction": float((np.abs(actions[:, 0]) >= 0.46).mean()),
                    }
                    episodes.write(json.dumps(end_record, sort_keys=True, allow_nan=False) + "\n")
                    episodes.flush()
                    episode_reward = 0.0
                    episode_actions = []
                    episode_started_step = additional_step
                    episode_started_info = {}

                if additional_step in checkpoints:
                    trace_path, trace_sha = _save_sample_trace(
                        run_dir, replay_trace, trace_rows, protocol_sha,
                        dataset.digest if dataset else None, additional_step,
                    )
                    candidate = _save_candidate(
                        agent,
                        protocol=protocol,
                        protocol_sha=protocol_sha,
                        source=source,
                        arm=arm,
                        run_dir=run_dir,
                        root=root,
                        additional_step=additional_step,
                        collection_result=collection_result,
                        dataset=dataset,
                        mixture=mixture,
                        rng_streams=rng_streams,
                        pool_rng=pool_rng,
                        transition_done=bool(transition.done),
                        next_episode_id=int(transition.episode_id) + 1,
                        trace_sha256=trace_sha,
                        trace_path=trace_path.relative_to(run_dir).as_posix(),
                    )
                    candidate_records.append(candidate)

                if transition.done and additional_step < total_steps:
                    observation, reset_info = collector.reset(seed=None)
                    reset_map[int(collector.episode_id)] = {
                        "geometry_seed": int(reset_info["seed"]),
                        "track_id": int(reset_info["track_id"]),
                    }
                    if reset_info["track_id"] not in pool["track_ids"] or reset_info["seed"] not in pool["seeds"]:
                        raise ValueError("online reset sampler produced a cell outside its frozen pool")
                    episodes.write(json.dumps({
                        "event": "reset",
                        "episode_id": collector.episode_id,
                        "additional_online_step": additional_step,
                        "source_learner_seed": learner_seed,
                        "arm": arm,
                        **reset_info,
                    }, sort_keys=True) + "\n")
                    episodes.flush()
                    episode_started_info = dict(reset_info)
                elif not transition.done:
                    observation = transition.next_observation

                if additional_step % 1000 == 0 or additional_step == total_steps:
                    progress_record = {
                        "source_learner_seed": learner_seed,
                        "arm": arm,
                        "additional_online_step": additional_step,
                        "environment_steps": agent.environment_steps,
                        "gradient_steps": agent.gradient_steps,
                        "expected_gradient_steps": max(0, additional_step - warmup),
                        "exploration_std": agent.exploration_std(),
                        "online_replay_size": agent.replay.size,
                        "online_replay_memory_bytes": agent.replay.memory_bytes,
                        "teacher_replay_rows": dataset.transition_count if dataset else 0,
                        "teacher_dataset_digest": dataset.digest if dataset else None,
                        "last_update": latest_metrics,
                        "episode_started_step": episode_started_step,
                        "episode_started_info": episode_started_info,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    print(json.dumps(progress_record, sort_keys=True), flush=True)
            if transition is None:
                raise RuntimeError("training loop did not execute an online decision")
            incomplete_episode = None
            if not transition.done:
                actions = np.asarray(episode_actions, dtype=np.float32)
                incomplete_episode = {
                    "episode_id": transition.episode_id,
                    "track_id": int(transition.info["track_id"]),
                    "geometry_seed": int(transition.info["seed"]),
                    "steps": transition.step + 1,
                    "reward": episode_reward,
                    "terminated": transition.terminated,
                    "truncated": transition.truncated,
                    "terminal": transition.terminal,
                    "budget_interrupted": True,
                }
                episodes.write(json.dumps({
                    "event": "budget-stop",
                    "additional_online_step": total_steps,
                    "source_learner_seed": learner_seed,
                    "arm": arm,
                    **incomplete_episode,
                    "native_action_mean": actions.mean(axis=0).tolist(),
                    "native_saturation_fraction": (np.abs(actions) >= 0.99).mean(axis=0).tolist(),
                    "steering_abs_ge_0_46_fraction": float((np.abs(actions[:, 0]) >= 0.46).mean()),
                }, sort_keys=True, allow_nan=False) + "\n")
            updates.flush()
            episodes.flush()
    finally:
        env.close()

    if agent.gradient_steps != expected_updates:
        raise RuntimeError(
            f"completed {agent.gradient_steps} gradient updates; expected exactly {expected_updates}"
        )
    if trace_rows != expected_updates:
        raise RuntimeError(f"recorded {trace_rows} replay batches; expected exactly {expected_updates}")
    trace_path = run_dir / f"replay-sample-trace-step-{total_steps:09d}.npz"
    trace_sha = _sha256(trace_path)
    checkpoint_catalog = {
        "schema_version": 1,
        "study_protocol_sha256": protocol_sha,
        "source_learner_seed": learner_seed,
        "source_actor_sha256": source["actor_sha256"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "arm": arm,
        "teacher_dataset_digest": dataset.digest if dataset else None,
        "candidates": sorted(candidate_records, key=lambda record: record["checkpoint_online_step"]),
    }
    _atomic_json(run_dir / "checkpoint-catalog.json", checkpoint_catalog)
    result = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "study_protocol_sha256": protocol_sha,
        "source_learner_seed": learner_seed,
        "source_revision": source["source_revision"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_actor_sha256": source["actor_sha256"],
        "source_pair_audit": audit,
        "weight_only_fork": fork_identity,
        "arm": arm,
        "replay_mode": mode,
        "teacher_dataset_digest": dataset.digest if dataset else None,
        "teacher_collection_result_sha256": _sha256(
            collection_result_path if collection_result_path.is_absolute()
            else root / collection_result_path
        ),
        "additional_online_steps": total_steps,
        "source_environment_steps": 131072,
        "final_environment_steps": agent.environment_steps,
        "study_gradient_steps": agent.gradient_steps,
        "expected_gradient_steps": expected_updates,
        "replay_sample_trace_path": trace_path.relative_to(run_dir).as_posix(),
        "replay_sample_trace_sha256": trace_sha,
        "replay_sample_trace_rows": trace_rows,
        "checkpoint_catalog_path": "checkpoint-catalog.json",
        "final_optimizer_state_entries": {
            "actor": len(agent.actor_optimizer.state),
            "critics": len(agent.critic_optimizer.state),
        },
        "completed": True,
        "incomplete_budget_episode": incomplete_episode,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(run_dir / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-file", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--learner-seed", required=True, type=int, choices=(0, 1))
    parser.add_argument("--arm", required=True, choices=("online-only", "teacher-replay"))
    parser.add_argument("--collection-result", required=True, type=Path)
    parser.add_argument("--teacher-dataset", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    protocol_path = args.protocol_file.resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    try:
        validate_protocol_file(protocol_path, root)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        result = train_arm(
            protocol,
            protocol_path,
            root,
            args.learner_seed,
            args.arm,
            args.collection_result,
            args.teacher_dataset,
            run_dir,
            args.resume_checkpoint,
        )
    except (OSError, ValueError, ProtocolError, RuntimeError, FloatingPointError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
