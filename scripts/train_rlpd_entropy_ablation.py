"""Train one from-scratch, one-factor pixel RLPD target-entropy arm."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from common_adapter import EpisodeCollector, ObservationSpec
from haic.algorithms.rlpd.agent import PixelRLPDAgent, RLPDConfig, sample_balanced_batch
from haic.algorithms.rlpd.replay import FrameStackReplay
from scripts.rlpd_common import (
    load_offline_replay,
    runtime_metadata,
    sha256_file,
    snapshot_sources,
    write_json,
)
from scripts.rlpd_entropy_common import read_entropy_protocol
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


def main():
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = read_entropy_protocol(protocol_path)
    student = protocol["student_training"]
    if args.arm not in student["arms"] or args.seed not in student["learner_seeds"]:
        raise ValueError("arm/learner seed was not declared by the frozen ablation")
    arm_spec = student["arm_specs"][args.arm]
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen entropy-ablation training runtime must use CUDA")
    total_steps = student["steps_per_run"]
    checkpoint_steps = tuple(student["candidate_steps"])
    if checkpoint_steps[-1] != total_steps:
        raise ValueError("the final screen checkpoint must equal the frozen decision budget")
    learner = student["learner"]
    config = RLPDConfig(
        actor_lr=learner["actor_lr"],
        critic_lr=learner["critic_lr"],
        temperature_lr=learner["temperature_lr"],
        gamma=learner["gamma"],
        tau=learner["tau"],
        batch_size=student["batch_size"],
        num_qs=learner["num_qs"],
        num_min_qs=learner["num_min_qs"],
        target_entropy=float(arm_spec["target_entropy"]),
        initial_alpha=learner["initial_alpha"],
        backup_entropy=student["backup_entropy"],
        augmentation_pad=learner["augmentation_pad"],
    )
    if not arm_spec["use_offline"]:
        raise ValueError("both entropy arms must preserve the same offline/online mixture")
    actual_runtime = runtime_metadata(device=args.device)
    expected_runtime = protocol["runtime"]["training"]
    if (
        actual_runtime["installed_distributions_sha256"] != expected_runtime["installed_distributions_sha256"]
        or actual_runtime["torch"] != expected_runtime["torch"]
        or actual_runtime["numpy"] != expected_runtime["numpy"]
        or actual_runtime.get("gpu") != expected_runtime.get("gpu")
    ):
        raise RuntimeError("effective training runtime differs from the frozen inventory")
    offline, dataset_sha, _dataset_manifest = load_offline_replay(
        args.offline_dataset.resolve(), protocol, seed=args.seed + 0x0FF1
    )

    run_dir = args.run_dir.resolve()
    resuming = args.resume is not None
    if resuming:
        resume_path = args.resume.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        run_dir = resume_path.parent.parent.parent
        if run_dir != args.run_dir.resolve():
            raise ValueError("resume checkpoint must belong to the requested arm/seed directory")
    elif run_dir.exists():
        raise FileExistsError(run_dir)

    run_config = {
        "format": "haic-rlpd-entropy-ablation-run-v1",
        "study_id": protocol["name"],
        "arm": args.arm,
        "algorithm": "rlpd",
        "arm_spec": dict(arm_spec),
        "target_entropy": config.target_entropy,
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
        raise ValueError("resume config differs in target, arm, dataset, or runtime")

    seed_everything(args.seed)
    agent = PixelRLPDAgent(config, seed=args.seed, device=args.device)
    online = FrameStackReplay(ONLINE_REPLAY_CAPACITY, seed=args.seed + 0x0A11, source="online")
    sampler_rng = np.random.default_rng(args.seed + 0x5A11)
    action_rng = np.random.default_rng(args.seed + 0xAC71)
    episode_id, episode_reward, episode_prefix = 0, 0.0, []
    environment = collector = observation = reset_info = current_cell = None
    episode_initial_hash = None
    start_step = 0
    if resuming:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        study = checkpoint.get("study", {})
        if (
            study.get("protocol_sha256") != run_config["protocol_sha256"]
            or study.get("offline_dataset_sha256") != dataset_sha
            or study.get("run_config") != run_config
        ):
            raise ValueError("resume checkpoint does not match this target arm/data/protocol")
        trainer = agent.load_checkpoint_state(checkpoint, offline_replay=offline, online_replay=online)
        if trainer.get("offline_dataset_sha256") != dataset_sha:
            raise ValueError("restored trainer dataset hash mismatch")
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
            raise RuntimeError("exact resume observation differs from the saved transition prefix")
    else:
        environment, collector, observation, cell_state = create_episode(protocol, sampler_rng, episode_id)
        current_cell = {key: cell_state[key] for key in ("track_id", "geometry_seed")}
        episode_initial_hash, reset_info = cell_state["reset_observation_sha256"], cell_state["reset_info"]

    started = time.perf_counter()
    metrics_file = (run_dir / "metrics.jsonl").open("a" if resuming else "x", encoding="utf-8")
    episodes_file = (run_dir / "episodes.jsonl").open("a" if resuming else "x", encoding="utf-8")
    checkpoints, last_metrics = [], {}
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
                last_metrics = agent.update(sample_balanced_batch(
                    offline,
                    online,
                    batch_size=config.batch_size,
                    offline_count=student["offline_batch_size"],
                ))
            agent.environment_steps = step + 1
            if transition.done:
                action_array = np.asarray([item["executed_action"] for item in episode_prefix], dtype=np.float32)
                episodes_file.write(json.dumps({
                    "event": "end", "episode_id": episode_id, "global_step": step + 1,
                    **current_cell, "steps": len(episode_prefix), "reward": episode_reward,
                    "terminated": transition.terminated, "truncated": transition.truncated,
                    "terminal": transition.terminal,
                    "finished": bool(transition.info.get("finished", False)),
                    "progress": float(transition.info.get("progress", 0.0)),
                    "damage": float(transition.info.get("damage", 0.0)),
                    "retire_reason": transition.info.get("retire_reason"),
                    "target_entropy": config.target_entropy,
                    "native_action_mean": action_array.mean(axis=0).tolist(),
                    "native_saturation_fraction": (np.abs(action_array) >= 0.99).mean(axis=0).tolist(),
                }, sort_keys=True) + "\n")
                episodes_file.flush()
                environment.close()
                episode_id += 1
                episode_reward, episode_prefix = 0.0, []
                environment, collector, observation, cell_state = create_episode(protocol, sampler_rng, episode_id)
                current_cell = {key: cell_state[key] for key in ("track_id", "geometry_seed")}
                episode_initial_hash, reset_info = cell_state["reset_observation_sha256"], cell_state["reset_info"]
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
                    "alpha": float(agent.alpha.detach().cpu().item()),
                    "beta": float(agent.log_alpha.detach().cpu().item()),
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
                candidate_path = run_dir / "checkpoints" / f"step-{agent.environment_steps:09d}" / "candidate.json"
                write_json(candidate_path, candidate)
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
            raise RuntimeError("entropy-ablation student missed its frozen decision/update budget")
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
            "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "device": args.device},
        }, exclusive=True)
    finally:
        metrics_file.close()
        episodes_file.close()
        if environment is not None:
            environment.close()
    print(f"RLPD entropy-ablation run complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
