"""Train one predeclared RLPD/SAC student arm and seed."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from common_adapter import EpisodeCollector, ObservationSpec
from haic.algorithms.rlpd.agent import PixelRLPDAgent, RLPDConfig, sample_balanced_batch
from haic.algorithms.rlpd.replay import FrameStackReplay, PixelTransition
from scripts.rlpd_common import (
    canonical_sha256,
    load_offline_replay,
    read_protocol,
    runtime_metadata,
    sha256_file,
    snapshot_sources,
    write_json,
)
from train import build_env


ONLINE_REPLAY_CAPACITY = 100_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--offline-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("rlpd", "sac"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def vm_hwm_kib() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def create_episode(protocol, rng, episode_id):
    cells = list(itertools.product(
        protocol["training_track_ids"], protocol["training_geometry_seeds"]
    ))
    track_id, geometry_seed = cells[int(rng.integers(0, len(cells)))]
    environment = build_env(
        track_id,
        geometry_seed,
        protocol["max_steps"],
        protocol["frame_skip"],
        reward_shaping=False,
        obstacles=True,
    )
    collector = EpisodeCollector(environment)
    observation, reset_info = collector.reset()
    return environment, collector, observation, {
        "episode_id": episode_id,
        "track_id": int(track_id),
        "geometry_seed": int(geometry_seed),
        "reset_observation_sha256": array_sha256(ObservationSpec().to_uint8(observation)),
        "reset_info": reset_info,
    }


def transition_to_replay(transition, proposed_action, episode_id, geometry_seed):
    return PixelTransition(
        observation=ObservationSpec().to_uint8(transition.observation),
        proposed_action=np.asarray(proposed_action, dtype=np.float32),
        executed_action=np.asarray(transition.action, dtype=np.float32),
        applied_action=np.asarray(transition.applied_action, dtype=np.float32),
        reward=transition.reward,
        next_observation=ObservationSpec().to_uint8(transition.next_observation),
        terminated=transition.terminated,
        truncated=transition.truncated,
        terminal=transition.terminal,
        episode_id=episode_id,
        step=transition.step,
        track_id=int(transition.info["track_id"]),
        geometry_seed=geometry_seed,
    )


def verify_reconstructed_prefix(protocol, trainer_state):
    cell = trainer_state.get("current_cell")
    if not isinstance(cell, dict):
        raise ValueError("resume checkpoint is missing the active environment cell")
    environment = build_env(
        cell["track_id"], cell["geometry_seed"], protocol["max_steps"], protocol["frame_skip"],
        reward_shaping=False, obstacles=True,
    )
    collector = EpisodeCollector(environment)
    try:
        observation, reset_info = collector.reset()
        if array_sha256(ObservationSpec().to_uint8(observation)) != trainer_state["reset_observation_sha256"]:
            raise RuntimeError("resume reset observation does not match the checkpoint trajectory")
        for index, expected in enumerate(trainer_state["episode_prefix"]):
            transition = collector.step(np.asarray(expected["proposed_action"], dtype=np.float32))
            checks = {
                "observation_sha256": array_sha256(ObservationSpec().to_uint8(transition.observation)),
                "next_observation_sha256": array_sha256(ObservationSpec().to_uint8(transition.next_observation)),
                "reward": float(transition.reward),
                "executed_action": np.asarray(transition.action, dtype=np.float32).tolist(),
                "applied_action": np.asarray(transition.applied_action, dtype=np.float32).tolist(),
                "terminated": bool(transition.terminated),
                "truncated": bool(transition.truncated),
                "terminal": bool(transition.terminal),
            }
            if checks != {key: expected[key] for key in checks}:
                raise RuntimeError(f"resume episode prefix diverged at action {index}: {checks}")
            if transition.done:
                raise RuntimeError("resume action prefix crosses an episode boundary")
            observation = transition.next_observation
        if array_sha256(ObservationSpec().to_uint8(observation)) != trainer_state["observation_sha256"]:
            raise RuntimeError("resume collector observation mismatch")
        return environment, collector, observation, reset_info
    except BaseException:
        environment.close()
        raise


def checkpoint_state(agent, offline, online, trainer_state, *, dataset_sha256, protocol_sha256, run_config):
    state = agent.checkpoint_state(
        offline_replay=offline,
        online_replay=online,
        trainer_state=trainer_state,
    )
    state["study"] = {
        "protocol_sha256": protocol_sha256,
        "offline_dataset_sha256": dataset_sha256,
        "run_config": run_config,
    }
    return state


def save_candidate(
    run_dir,
    agent,
    offline,
    online,
    trainer_state,
    *,
    protocol,
    protocol_path,
    dataset_sha256,
    run_config,
):
    step = agent.environment_steps
    checkpoint_dir = run_dir / "checkpoints" / f"step-{step:09d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    source_sha = canonical_sha256(protocol["source_hashes"])
    environment_contract = {
        "arm": run_config["arm"],
        "training_seed": run_config["training_seed"],
        "offline_dataset_sha256": dataset_sha256,
        "training_track_ids": protocol["training_track_ids"],
        "training_geometry_seeds": protocol["training_geometry_seeds"],
        "frame_skip": protocol["frame_skip"],
        "obstacles": True,
        "reward_shaping": False,
        "action_smoothing": None,
        "action_control": None,
    }
    actor_path = agent.export_actor(
        checkpoint_dir / "actor.pt",
        source_sha256=source_sha,
        protocol_sha256=sha256_file(protocol_path),
        training_seed=run_config["training_seed"],
        environment_contract=environment_contract,
    )
    payload = checkpoint_state(
        agent,
        offline,
        online,
        trainer_state,
        dataset_sha256=dataset_sha256,
        protocol_sha256=sha256_file(protocol_path),
        run_config=run_config,
    )
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    temporary = checkpoint_path.with_name("checkpoint.pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    record = {
        "format": "haic-rlpd-candidate-v1",
        "study_id": protocol["name"],
        "arm": run_config["arm"],
        "training_seed": run_config["training_seed"],
        "environment_steps": step,
        "gradient_steps": agent.gradient_steps,
        "actor_path": str(actor_path.relative_to(run_dir)),
        "actor_sha256": sha256_file(actor_path),
        "checkpoint_path": str(checkpoint_path.relative_to(run_dir)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "protocol_sha256": sha256_file(protocol_path),
        "offline_dataset_sha256": dataset_sha256,
        "resume_episode_steps": len(trainer_state["episode_prefix"]),
        "vm_hwm_kib": vm_hwm_kib(),
        "gpu_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(agent.device) if agent.device.type == "cuda" else 0
        ),
    }
    write_json(checkpoint_dir / "candidate.json", record, exclusive=True)
    return record


def main():
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = read_protocol(protocol_path)
    if args.seed not in protocol["student_training"]["learner_seeds"]:
        raise ValueError("learner seed is not predeclared in the protocol")
    if args.total_steps is None:
        total_steps = protocol["student_training"]["steps_per_run"]
    else:
        total_steps = args.total_steps
    if total_steps != protocol["student_training"]["steps_per_run"]:
        raise ValueError("total_steps must exactly match the frozen matched-run budget")
    checkpoint_steps = tuple(protocol["student_training"].get(
        "candidate_steps",
        protocol["evaluation"]["screen_candidate_steps"],
    ))
    if not checkpoint_steps or total_steps < max(checkpoint_steps):
        raise ValueError("protocol does not include required candidate checkpoints")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen pilot learner must use its pinned CUDA training runtime")
    offline, dataset_sha, _dataset_manifest = load_offline_replay(
        args.offline_dataset.resolve(), protocol, seed=args.seed + 0x0FF1
    )
    learner = protocol["student_training"]["learner"]
    config = RLPDConfig(
        actor_lr=learner["actor_lr"],
        critic_lr=learner["critic_lr"],
        temperature_lr=learner["temperature_lr"],
        gamma=learner["gamma"],
        tau=learner["tau"],
        batch_size=protocol["student_training"]["batch_size"],
        num_qs=learner["num_qs"],
        num_min_qs=learner["num_min_qs"],
        target_entropy=learner["target_entropy"],
        initial_alpha=learner["initial_alpha"],
        backup_entropy=protocol["student_training"]["backup_entropy"],
        augmentation_pad=learner["augmentation_pad"],
    )
    actual_runtime = runtime_metadata(device=args.device)
    expected_runtime = protocol["runtime"]["training"]
    if (
        actual_runtime["installed_distributions_sha256"]
        != expected_runtime["installed_distributions_sha256"]
        or actual_runtime["torch"] != expected_runtime["torch"]
        or actual_runtime["numpy"] != expected_runtime["numpy"]
        or actual_runtime.get("gpu") != expected_runtime.get("gpu")
    ):
        raise RuntimeError("effective training dependencies or accelerator differ from the frozen protocol")
    run_dir = args.run_dir.resolve()
    resuming = args.resume is not None
    if resuming:
        resume_path = args.resume.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        run_dir = resume_path.parent.parent.parent
        if run_dir != args.run_dir.resolve():
            raise ValueError("resume checkpoint is not inside the requested run directory")
    elif run_dir.exists():
        raise FileExistsError(run_dir)

    run_config = {
        "format": "haic-rlpd-run-config-v1",
        "study_id": protocol["name"],
        "arm": args.arm,
        "algorithm": "rlpd" if args.arm == "rlpd" else "sac-pixel-online-control",
        "training_seed": args.seed,
        "protocol_sha256": sha256_file(protocol_path),
        "offline_dataset_sha256": dataset_sha,
        "offline_collection_manifest_sha256": sha256_file(args.offline_dataset / "manifest.json"),
        "environment_steps": total_steps,
        "gradient_steps_expected": total_steps - protocol["student_training"]["first_update_step"],
        "frame_skip": protocol["frame_skip"],
        "max_steps": protocol["max_steps"],
        "reward_shaping": False,
        "obstacles": True,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "device": str(args.device),
        "learner_config": asdict(config),
        "installed_distributions_sha256": actual_runtime["installed_distributions_sha256"],
        "official_performance_claim": False,
    }

    if not resuming:
        run_dir.mkdir(parents=True, exist_ok=False)
        snapshot_sources(protocol, run_dir / "source")
        (run_dir / "study_protocol.json").write_bytes(protocol_path.read_bytes())
        write_json(run_dir / "config.json", run_config, exclusive=True)
        write_json(run_dir / "runtime.json", actual_runtime, exclusive=True)
    else:
        saved_config = json.loads((run_dir / "config.json").read_text())
        if saved_config != run_config:
            raise ValueError("resume run configuration differs from the original study run")

    seed_everything(args.seed)
    agent = PixelRLPDAgent(config, seed=args.seed, device=args.device)
    online = FrameStackReplay(
        ONLINE_REPLAY_CAPACITY,
        seed=args.seed + 0x0A11,
        source="online",
    )
    sampler_rng = np.random.default_rng(args.seed + 0x5A11)
    action_rng = np.random.default_rng(args.seed + 0xAC71)
    episode_id = 0
    episode_reward = 0.0
    episode_prefix = []
    episode_initial_hash = None
    current_cell = None
    environment = None
    collector = None
    observation = None
    reset_info = None
    start_step = 0
    if resuming:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        study = checkpoint.get("study", {})
        if (
            study.get("protocol_sha256") != run_config["protocol_sha256"]
            or study.get("offline_dataset_sha256") != dataset_sha
            or study.get("run_config") != run_config
        ):
            raise ValueError("checkpoint study identity differs from protocol/dataset/run config")
        trainer = agent.load_checkpoint_state(
            checkpoint,
            offline_replay=offline,
            online_replay=online,
        )
        if trainer.get("offline_dataset_sha256") != dataset_sha:
            raise ValueError("restored trainer data hash differs from the frozen dataset")
        sampler_rng.bit_generator.state = trainer["sampler_rng_state"]
        action_rng.bit_generator.state = trainer["action_rng_state"]
        episode_id = trainer["episode_id"]
        episode_reward = trainer["episode_reward"]
        episode_prefix = trainer["episode_prefix"]
        episode_initial_hash = trainer["reset_observation_sha256"]
        current_cell = trainer["current_cell"]
        start_step = agent.environment_steps
        environment, collector, observation, reset_info = verify_reconstructed_prefix(protocol, trainer)
        if array_sha256(ObservationSpec().to_uint8(observation)) != trainer["observation_sha256"]:
            environment.close()
            raise RuntimeError("exact resume failed: current observation mismatch")
    else:
        environment, collector, observation, cell_state = create_episode(protocol, sampler_rng, episode_id)
        current_cell = {key: cell_state[key] for key in ("track_id", "geometry_seed")}
        episode_initial_hash = cell_state["reset_observation_sha256"]
        reset_info = cell_state["reset_info"]

    started = time.perf_counter()
    last_metrics = {}
    checkpoints = []
    metrics_path = run_dir / "metrics.jsonl"
    episodes_path = run_dir / "episodes.jsonl"
    metrics_file = metrics_path.open("a" if resuming else "x", encoding="utf-8")
    episodes_file = episodes_path.open("a" if resuming else "x", encoding="utf-8")
    try:
        if not resuming:
            episodes_file.write(json.dumps({
                "event": "reset", "episode_id": episode_id, **current_cell,
                "reset_observation_sha256": episode_initial_hash, **reset_info,
            }, sort_keys=True) + "\n")
            episodes_file.flush()
        for step in range(start_step, total_steps):
            if step < protocol["student_training"]["policy_takeover_step"]:
                proposed_action = action_rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
                pre_tanh_action = None
                behavior_mode = "random"
            else:
                proposed_action, pre_tanh_action, _policy_log_prob = agent.act_with_details(
                    observation, deterministic=False
                )
                behavior_mode = "policy"
            transition = collector.step(proposed_action)
            online.add(transition_to_replay(
                transition,
                proposed_action,
                episode_id,
                current_cell["geometry_seed"],
            ))
            episode_reward += float(transition.reward)
            next_observation = transition.next_observation
            episode_prefix.append({
                "proposed_action": np.asarray(proposed_action, dtype=np.float32).tolist(),
                "pre_tanh_action": (
                    None if pre_tanh_action is None
                    else np.asarray(pre_tanh_action, dtype=np.float32).tolist()
                ),
                "behavior_mode": behavior_mode,
                "observation_sha256": array_sha256(ObservationSpec().to_uint8(transition.observation)),
                "next_observation_sha256": array_sha256(ObservationSpec().to_uint8(next_observation)),
                "reward": float(transition.reward),
                "executed_action": np.asarray(transition.action, dtype=np.float32).tolist(),
                "applied_action": np.asarray(transition.applied_action, dtype=np.float32).tolist(),
                "terminated": bool(transition.terminated),
                "truncated": bool(transition.truncated),
                "terminal": bool(transition.terminal),
            })
            if step >= protocol["student_training"]["first_update_step"]:
                if args.arm == "rlpd":
                    batch = sample_balanced_batch(
                        offline,
                        online,
                        batch_size=config.batch_size,
                        offline_count=protocol["student_training"]["offline_batch_size"],
                    )
                else:
                    batch = online.sample(config.batch_size)
                last_metrics = agent.update(batch)
            agent.environment_steps = step + 1

            if transition.done:
                episode_array = np.asarray(
                    [item["executed_action"] for item in episode_prefix], dtype=np.float32
                )
                episodes_file.write(json.dumps({
                    "event": "end",
                    "episode_id": episode_id,
                    "global_step": step + 1,
                    **current_cell,
                    "steps": len(episode_prefix),
                    "reward": episode_reward,
                    "terminated": transition.terminated,
                    "truncated": transition.truncated,
                    "terminal": transition.terminal,
                    "finished": bool(transition.info.get("finished", False)),
                    "progress": float(transition.info.get("progress", 0.0)),
                    "damage": float(transition.info.get("damage", 0.0)),
                    "retire_reason": transition.info.get("retire_reason"),
                "native_action_mean": episode_array.mean(axis=0).tolist(),
                "native_saturation_fraction": (np.abs(episode_array) >= 0.99).mean(axis=0).tolist(),
                "policy_pre_tanh_abs_max": max(
                    (
                        max(abs(value) for value in item["pre_tanh_action"])
                        for item in episode_prefix
                        if item["pre_tanh_action"] is not None
                    ),
                    default=None,
                ),
                "behavior_mode_counts": {
                    "random": sum(item["behavior_mode"] == "random" for item in episode_prefix),
                    "policy": sum(item["behavior_mode"] == "policy" for item in episode_prefix),
                },
            }, sort_keys=True) + "\n")
                episodes_file.flush()
                environment.close()
                episode_id += 1
                episode_reward = 0.0
                episode_prefix = []
                environment, collector, observation, cell_state = create_episode(
                    protocol, sampler_rng, episode_id
                )
                current_cell = {key: cell_state[key] for key in ("track_id", "geometry_seed")}
                episode_initial_hash = cell_state["reset_observation_sha256"]
                reset_info = cell_state["reset_info"]
                episodes_file.write(json.dumps({
                    "event": "reset", "episode_id": episode_id, **current_cell,
                    "reset_observation_sha256": episode_initial_hash, **reset_info,
                }, sort_keys=True) + "\n")
                episodes_file.flush()
            else:
                observation = next_observation

            if (step + 1) % 1000 == 0 or step + 1 in checkpoint_steps:
                record = {
                    "step": step + 1,
                    **last_metrics,
                    "online_replay_size": len(online),
                    "online_replay_valid": online.valid_count,
                    "offline_replay_size": len(offline),
                    "offline_dataset_sha256": dataset_sha,
                    "vm_hwm_kib": vm_hwm_kib(),
                    "cuda_allocated_bytes": (
                        torch.cuda.memory_allocated(agent.device) if agent.device.type == "cuda" else 0
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                metrics_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                metrics_file.flush()
                print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)

            if step + 1 in checkpoint_steps:
                trainer_state = {
                    "offline_dataset_sha256": dataset_sha,
                    "episode_id": episode_id,
                    "episode_reward": episode_reward,
                    "episode_prefix": list(episode_prefix),
                    "current_cell": dict(current_cell),
                    "reset_observation_sha256": episode_initial_hash,
                    "observation_sha256": array_sha256(ObservationSpec().to_uint8(observation)),
                    "sampler_rng_state": sampler_rng.bit_generator.state,
                    "action_rng_state": action_rng.bit_generator.state,
                    "reset_info": reset_info,
                }
                candidate = save_candidate(
                    run_dir,
                    agent,
                    offline,
                    online,
                    trainer_state,
                    protocol=protocol,
                    protocol_path=protocol_path,
                    dataset_sha256=dataset_sha,
                    run_config=run_config,
                )
                checkpoints.append(candidate)
                index_path = run_dir / "frozen_candidates.json"
                previous = json.loads(index_path.read_text()) if index_path.exists() else {
                    "format": "haic-rlpd-frozen-candidates-v1",
                    "study_id": protocol["name"],
                    "protocol_sha256": run_config["protocol_sha256"],
                    "offline_dataset_sha256": dataset_sha,
                    "candidates": [],
                }
                previous["candidates"].append(candidate)
                write_json(index_path, previous)

        if agent.environment_steps != total_steps:
            raise RuntimeError("training terminated before the exact frozen decision budget")
        if agent.gradient_steps != total_steps - protocol["student_training"]["first_update_step"]:
            raise RuntimeError("gradient-step count does not match the frozen warmup schedule")
        write_json(run_dir / "result.json", {
            "format": "haic-rlpd-run-result-v1",
            "study_id": protocol["name"],
            "arm": args.arm,
            "training_seed": args.seed,
            "environment_steps": agent.environment_steps,
            "gradient_steps": agent.gradient_steps,
            "offline_dataset_sha256": dataset_sha,
            "candidate_count": len(checkpoints),
            "candidates": checkpoints,
            "wall_seconds": time.perf_counter() - started,
            "official_performance_claim": False,
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(args.device),
            },
        }, exclusive=True)
    finally:
        metrics_file.close()
        episodes_file.close()
        if environment is not None:
            environment.close()
    print(f"RLPD/SAC student run complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
