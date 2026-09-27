"""Train one matched RLPD target-entropy variant and learner seed."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from haic.algorithms.rlpd.agent import PixelRLPDAgent, RLPDConfig, sample_balanced_batch
from haic.algorithms.rlpd.replay import FrameStackReplay
from scripts.rlpd_common import (
    canonical_sha256,
    load_offline_replay,
    read_protocol,
    runtime_metadata,
    sha256_file,
    snapshot_sources,
    write_json,
)
from scripts.train_rlpd import (
    array_sha256,
    checkpoint_state,
    create_episode,
    save_candidate,
    seed_everything,
    transition_to_replay,
    verify_reconstructed_prefix,
    vm_hwm_kib,
)


ONLINE_REPLAY_CAPACITY = 100_000
FOLLOWUP_NAME = "pixel-rlpd-entropy-target-ablation-v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--offline-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def _config(protocol, arm: str):
    specification = protocol["student_training"]["arm_specs"].get(arm)
    if not isinstance(specification, dict):
        raise ValueError("arm is not declared by the frozen entropy-ablation protocol")
    if specification.get("algorithm") != "rlpd" or specification.get("use_offline") is not True:
        raise ValueError("both entropy-ablation arms must use the same offline RLPD treatment")
    if specification.get("target_entropy") not in {-1.5, 1.5}:
        raise ValueError("target entropy must be one of the two predeclared values")
    learner = protocol["student_training"]["learner"]
    return RLPDConfig(
        actor_lr=learner["actor_lr"],
        critic_lr=learner["critic_lr"],
        temperature_lr=learner["temperature_lr"],
        gamma=learner["gamma"],
        tau=learner["tau"],
        batch_size=protocol["student_training"]["batch_size"],
        num_qs=learner["num_qs"],
        num_min_qs=learner["num_min_qs"],
        target_entropy=float(specification["target_entropy"]),
        initial_alpha=learner["initial_alpha"],
        backup_entropy=protocol["student_training"]["backup_entropy"],
        augmentation_pad=learner["augmentation_pad"],
    ), specification


def _save_entropy_candidate(run_dir, agent, offline, online, trainer_state, *, protocol, protocol_path, dataset_sha256, run_config):
    record = save_candidate(
        run_dir,
        agent,
        offline,
        online,
        trainer_state,
        protocol=protocol,
        protocol_path=protocol_path,
        dataset_sha256=dataset_sha256,
        run_config=run_config,
    )
    # The common candidate writer uses run_config.arm in the actor payload; the
    # study-specific target remains explicitly bound in the candidate/run record.
    record["target_entropy"] = run_config["learner_config"]["target_entropy"]
    record["arm_spec"] = run_config["arm_spec"]
    candidate_path = Path(record["actor_path"])
    candidate_json = candidate_path.parent / "candidate.json"
    write_json(candidate_json, record)
    return record


def main():
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = read_protocol(protocol_path)
    if protocol.get("name") != FOLLOWUP_NAME:
        raise ValueError("entropy ablation requires its independently frozen protocol")
    student = protocol["student_training"]
    if args.arm not in student["arms"] or args.seed not in student["learner_seeds"]:
        raise ValueError("arm/learner seed was not predeclared")
    total_steps = student["steps_per_run"]
    checkpoint_steps = tuple(student["candidate_steps"])
    if checkpoint_steps[-1] != total_steps:
        raise ValueError("the final predeclared checkpoint must equal the full budget")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen learner must use the pinned CUDA runtime")
    config, arm_spec = _config(protocol, args.arm)
    actual_runtime = runtime_metadata(device=args.device)
    expected_runtime = protocol["runtime"]["training"]
    if (
        actual_runtime["installed_distributions_sha256"]
        != expected_runtime["installed_distributions_sha256"]
        or actual_runtime["torch"] != expected_runtime["torch"]
        or actual_runtime["numpy"] != expected_runtime["numpy"]
        or actual_runtime.get("gpu") != expected_runtime.get("gpu")
    ):
        raise RuntimeError("effective training dependency/accelerator inventory differs from the frozen protocol")

    offline, dataset_sha, dataset_manifest = load_offline_replay(
        args.offline_dataset.resolve(), protocol, seed=args.seed + 0x0FF1
    )
    run_dir = args.run_dir.resolve()
    resuming = args.resume is not None
    if resuming:
        checkpoint_path = args.resume.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        run_dir = checkpoint_path.parent.parent.parent
        if run_dir != args.run_dir.resolve():
            raise ValueError("resume checkpoint lies outside the requested run root")
    elif run_dir.exists():
        raise FileExistsError(run_dir)

    run_config = {
        "format": "haic-rlpd-entropy-ablation-run-v1",
        "study_id": protocol["name"],
        "arm": args.arm,
        "algorithm": "rlpd",
        "arm_spec": dict(arm_spec),
        "target_entropy": float(arm_spec["target_entropy"]),
        "training_seed": args.seed,
        "protocol_sha256": sha256_file(protocol_path),
        "offline_dataset_sha256": dataset_sha,
        "offline_collection_manifest_sha256": sha256_file(args.offline_dataset / "manifest.json"),
        "environment_steps": total_steps,
        "gradient_steps_expected": total_steps - student["first_update_step"],
        "frame_skip": protocol["frame_skip"],
        "max_steps": protocol["max_steps"],
        "reward_shaping": False,
        "obstacles": True,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "device": args.device,
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
    elif json.loads((run_dir / "config.json").read_text()) != run_config:
        raise ValueError("resume run config differs from its original arm/target/dataset")

    seed_everything(args.seed)
    agent = PixelRLPDAgent(config, seed=args.seed, device=args.device)
    online = FrameStackReplay(ONLINE_REPLAY_CAPACITY, seed=args.seed + 0x0A11, source="online")
    sampler_rng = np.random.default_rng(args.seed + 0x5A11)
    action_rng = np.random.default_rng(args.seed + 0xAC71)
    episode_id = 0
    episode_reward = 0.0
    episode_prefix = []
    environment = collector = observation = reset_info = None
    current_cell = None
    episode_initial_hash = None
    start_step = 0
    if resuming:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        study_record = checkpoint.get("study", {})
        if (
            study_record.get("protocol_sha256") != run_config["protocol_sha256"]
            or study_record.get("offline_dataset_sha256") != dataset_sha
            or study_record.get("run_config") != run_config
        ):
            raise ValueError("resume checkpoint identity differs from frozen protocol/arm/data")
        trainer = agent.load_checkpoint_state(checkpoint, offline_replay=offline, online_replay=online)
        sampler_rng.bit_generator.state = trainer["sampler_rng_state"]
        action_rng.bit_generator.state = trainer["action_rng_state"]
        episode_id = trainer["episode_id"]
        episode_reward = trainer["episode_reward"]
        episode_prefix = trainer["episode_prefix"]
        current_cell = trainer["current_cell"]
        episode_initial_hash = trainer["reset_observation_sha256"]
        start_step = agent.environment_steps
        environment, collector, observation, reset_info = verify_reconstructed_prefix(protocol, trainer)
        if array_sha256(ObservationSpec().to_uint8(observation)) != trainer["observation_sha256"]:
            environment.close()
            raise RuntimeError("exact resume collector observation mismatch")
    else:
        environment, collector, observation, cell_state = create_episode(protocol, sampler_rng, episode_id)
        current_cell = {key: cell_state[key] for key in ("track_id", "geometry_seed")}
        episode_initial_hash = cell_state["reset_observation_sha256"]
        reset_info = cell_state["reset_info"]

    started = time.perf_counter()
    metrics_path, episodes_path = run_dir / "metrics.jsonl", run_dir / "episodes.jsonl"
    metrics_file = metrics_path.open("a" if resuming else "x", encoding="utf-8")
    episodes_file = episodes_path.open("a" if resuming else "x", encoding="utf-8")
    checkpoints = []
    last_metrics = {}
    try:
        if not resuming:
            episodes_file.write(json.dumps({
                "event": "reset", "episode_id": episode_id, **current_cell,
                "reset_observation_sha256": episode_initial_hash, **reset_info,
            }, sort_keys=True) + "\n")
            episodes_file.flush()
        for step in range(start_step, total_steps):
            if step < student["policy_takeover_step"]:
                proposed = action_rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
                pre_tanh, behavior = None, "random"
            else:
                proposed, pre_tanh, _ = agent.act_with_details(observation, deterministic=False)
                behavior = "policy"
            transition = collector.step(proposed)
            online.add(transition_to_replay(
                transition, proposed, episode_id, current_cell["geometry_seed"]
            ))
            episode_reward += float(transition.reward)
            next_observation = transition.next_observation
            episode_prefix.append({
                "proposed_action": np.asarray(proposed, dtype=np.float32).tolist(),
                "pre_tanh_action": None if pre_tanh is None else np.asarray(pre_tanh, dtype=np.float32).tolist(),
                "behavior_mode": behavior,
                "observation_sha256": array_sha256(ObservationSpec().to_uint8(transition.observation)),
                "next_observation_sha256": array_sha256(ObservationSpec().to_uint8(next_observation)),
                "reward": float(transition.reward),
                "executed_action": np.asarray(transition.action, dtype=np.float32).tolist(),
                "applied_action": np.asarray(transition.applied_action, dtype=np.float32).tolist(),
                "terminated": bool(transition.terminated),
                "truncated": bool(transition.truncated),
                "terminal": bool(transition.terminal),
            })
            if step >= student["first_update_step"]:
                batch = sample_balanced_batch(
                    offline,
                    online,
                    batch_size=config.batch_size,
                    offline_count=student["offline_batch_size"],
                )
                last_metrics = agent.update(batch)
            agent.environment_steps = step + 1

            if transition.done:
                executed = np.asarray([item["executed_action"] for item in episode_prefix], dtype=np.float32)
                episodes_file.write(json.dumps({
                    "event": "end", "episode_id": episode_id, "global_step": step + 1,
                    **current_cell, "steps": len(episode_prefix), "reward": episode_reward,
                    "terminated": transition.terminated, "truncated": transition.truncated,
                    "terminal": transition.terminal,
                    "finished": bool(transition.info.get("finished", False)),
                    "progress": float(transition.info.get("progress", 0.0)),
                    "damage": float(transition.info.get("damage", 0.0)),
                    "retire_reason": transition.info.get("retire_reason"),
                    "native_action_mean": executed.mean(axis=0).tolist(),
                    "native_saturation_fraction": (np.abs(executed) >= 0.99).mean(axis=0).tolist(),
                    "target_entropy": config.target_entropy,
                }, sort_keys=True) + "\n")
                episodes_file.flush()
                environment.close()
                episode_id += 1
                episode_reward = 0.0
                episode_prefix = []
                environment, collector, observation, cell_state = create_episode(protocol, sampler_rng, episode_id)
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
                metric = {
                    "step": step + 1,
                    **last_metrics,
                    "arm": args.arm,
                    "target_entropy": config.target_entropy,
                    "offline_replay_size": len(offline),
                    "online_replay_size": len(online),
                    "online_replay_valid": online.valid_count,
                    "offline_dataset_sha256": dataset_sha,
                    "vm_hwm_kib": vm_hwm_kib(),
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(agent.device),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                metrics_file.write(json.dumps(metric, sort_keys=True, allow_nan=False) + "\n")
                metrics_file.flush()
                print(json.dumps(metric, sort_keys=True, allow_nan=False), flush=True)

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
                candidate["target_entropy"] = config.target_entropy
                candidate["arm_spec"] = dict(arm_spec)
                write_json(run_dir / "checkpoints" / f"step-{agent.environment_steps:09d}" / "candidate.json", candidate)
                checkpoints.append(candidate)
                index_path = run_dir / "frozen_candidates.json"
                index = json.loads(index_path.read_text()) if index_path.exists() else {
                    "format": "haic-rlpd-entropy-candidates-v1",
                    "study_id": protocol["name"],
                    "protocol_sha256": run_config["protocol_sha256"],
                    "offline_dataset_sha256": dataset_sha,
                    "arm": args.arm,
                    "target_entropy": config.target_entropy,
                    "candidates": [],
                }
                index["candidates"].append(candidate)
                write_json(index_path, index)

        if agent.environment_steps != total_steps or agent.gradient_steps != total_steps - student["first_update_step"]:
            raise RuntimeError("entropy-ablation run missed its frozen student/update budget")
        write_json(run_dir / "result.json", {
            "format": "haic-rlpd-entropy-run-result-v1",
            "study_id": protocol["name"],
            "arm": args.arm,
            "target_entropy": config.target_entropy,
            "training_seed": args.seed,
            "environment_steps": agent.environment_steps,
            "gradient_steps": agent.gradient_steps,
            "offline_dataset_sha256": dataset_sha,
            "candidate_count": len(checkpoints),
            "candidates": checkpoints,
            "wall_seconds": time.perf_counter() - started,
            "official_performance_claim": False,
        }, exclusive=True)
    finally:
        metrics_file.close()
        episodes_file.close()
        if environment is not None:
            environment.close()
    print(f"RLPD entropy ablation run complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
