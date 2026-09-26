"""Bounded training and development evaluation for frozen-DrQ residual options."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch
from gymnasium.wrappers import TimeLimit

from common_adapter import EpisodeCollector
from drq_v2 import load_exported_actor
from haic.algorithms.drq_v2 import (
    FeatureOptionReplay,
    OptionTransitionAccumulator,
    ResidualOption,
    ResidualOptionQ,
    ResidualOptionState,
    polyak_update_option_target,
    select_greedy_option,
    train_option_q_step,
)
from train import HaicTrack, SampledHaicTrack


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("name") not in {
        "drqv2-residual-options-pilot-v1",
        "drqv2-residual-options-pilot-v2",
    }:
        raise ValueError("unsupported residual option pilot protocol")
    return protocol


def reserved_geometry_seeds(protocol: dict, root: Path) -> tuple[set[int], dict[str, str]]:
    reserved = set()
    source_hashes = {}
    for relative in protocol["exclusion_protocols"]:
        path = root / relative
        document = json.loads(path.read_text(encoding="utf-8"))
        source_hashes[relative] = sha256_file(path)
        values = document.get("reserved_training_seeds", [])
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in values):
            raise ValueError(f"invalid reserved training seed in {relative}")
        reserved.update(values)
        for partition in document.get("partitions", {}).values():
            seeds = partition.get("seeds", [])
            if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
                raise ValueError(f"invalid partition seed in {relative}")
            reserved.update(seeds)
        prior_development = document.get("development_screen", {}).get("geometry_seeds", [])
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in prior_development):
            raise ValueError(f"invalid prior development seed in {relative}")
        reserved.update(prior_development)
    development_seeds = protocol["development_screen"]["geometry_seeds"]
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in development_seeds):
        raise ValueError("development geometry seed is not uint32")
    reused_development = protocol.get("reused_development_geometry_seeds", [])
    if any(type(seed) is not int or seed not in development_seeds for seed in reused_development):
        raise ValueError("reused development seeds must be members of this development grid")
    overlap = reserved.intersection(development_seeds)
    if overlap != set(reused_development):
        raise ValueError("development grid overlaps a prior seed without an exact reuse declaration")
    reserved.update(development_seeds)
    return reserved, source_hashes


def validate_base_actor(path: Path, expected_sha256: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"frozen DrQ actor hash mismatch: expected {expected_sha256}, got {actual}")
    return actual


@torch.inference_mode()
def infer_base_and_features(actor, observation: np.ndarray, device: torch.device):
    tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    features = actor.encoder(tensor)
    base_action = torch.tanh(actor.policy(actor.trunk(features)))
    return (
        features[0].detach().cpu().numpy().astype(np.float32, copy=True),
        base_action[0].detach().cpu().numpy().astype(np.float32, copy=True),
    )


@torch.inference_mode()
def q_values_for(head: ResidualOptionQ, features: np.ndarray, base_action: np.ndarray, device):
    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    action_tensor = torch.as_tensor(base_action, dtype=torch.float32, device=device).unsqueeze(0)
    return head(feature_tensor, action_tensor)[0].detach().cpu().numpy()


def exploration_values(option: ResidualOption) -> np.ndarray:
    values = np.full(len(ResidualOption), -1e6, dtype=np.float32)
    values[int(option)] = 1e6
    return values


def train(protocol_path: Path, run_dir: Path, base_actor_path: Path) -> Path:
    root = Path.cwd()
    protocol_path = protocol_path.resolve()
    run_dir = run_dir.resolve()
    base_actor_path = base_actor_path.resolve()
    protocol = load_protocol(protocol_path)
    training = protocol["training"]
    development = protocol["development_screen"]
    base_hash = validate_base_actor(base_actor_path, protocol["base_driver"]["sha256"])
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    if base_actor_path == run_dir / "frozen_actor.pt":
        raise ValueError("base actor source and run output must differ")

    excluded_seeds, exclusion_hashes = reserved_geometry_seeds(protocol, root)
    device = torch.device("cuda" if training["device"] == "cuda-if-available" and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    seed = int(training["initial_iteration_seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    option_rng = np.random.default_rng(seed)

    run_dir.mkdir(parents=True)
    copied_actor = run_dir / "frozen_actor.pt"
    shutil.copy2(base_actor_path, copied_actor)
    if sha256_file(copied_actor) != base_hash:
        raise RuntimeError("copied frozen actor hash changed")
    protocol_copy = run_dir / "protocol.json"
    shutil.copy2(protocol_path, protocol_copy)
    protocol_hash = sha256_file(protocol_copy)
    actor, _, _ = load_exported_actor(copied_actor, device=str(device))
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)

    q_config = training["q_head"]
    online = ResidualOptionQ(q_config["feature_dim"], q_config["hidden_dim"]).to(device)
    target = ResidualOptionQ(q_config["feature_dim"], q_config["hidden_dim"]).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=q_config["learning_rate"])
    replay = FeatureOptionReplay(
        training["replay"]["capacity"],
        q_config["feature_dim"],
        base_hash,
        seed=seed,
    )

    environment = SampledHaicTrack(
        track_ids=protocol["environment"]["track_ids"],
        sampler_seed=training["environment_sampler_seed"],
        max_steps=protocol["environment"]["max_decisions_per_episode"],
        frame_skip=protocol["environment"]["frame_skip"],
        excluded_seeds=excluded_seeds,
        obstacles=protocol["environment"]["obstacles"],
    )
    environment = TimeLimit(environment, max_episode_steps=protocol["environment"]["max_decisions_per_episode"])
    collector = EpisodeCollector(environment, gamma=training["double_q"]["gamma_per_decision"])
    observation, initial_info = collector.reset()
    intervention_margin = float(training.get("intervention_margin_q_minus_keep", 0.0))
    option_state = ResidualOptionState(
        training["option_duration_decisions"],
        intervention_margin=intervention_margin,
        steering_delta=training["steering_delta_native"],
        brake_floor_official=training["brake_floor_official"],
    )
    accumulator = None
    option_counts = Counter()
    update_count = 0
    environment_steps = 0
    completed_episodes = 0
    episode_steps = 0
    episode_reward = 0.0
    episode_option_counts = Counter()
    losses = []
    last_info = initial_info
    started = time.perf_counter()
    manifest = {
        "schema_version": 1,
        "protocol": str(protocol_path.relative_to(root)),
        "protocol_sha256": protocol_hash,
        "base_actor_path": str(base_actor_path),
        "base_actor_sha256": base_hash,
        "source_sha256": {
            "scripts/drqv2_residual_options.py": sha256_file(root / "scripts/drqv2_residual_options.py"),
            "haic/algorithms/drq_v2/residual_options.py": sha256_file(root / "haic/algorithms/drq_v2/residual_options.py"),
            **exclusion_hashes,
        },
        "training_seed": seed,
        "environment_sampler_seed": training["environment_sampler_seed"],
        "excluded_geometry_seed_count": len(excluded_seeds),
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "training",
    }
    write_json(run_dir / "manifest.json", manifest)
    episodes_path = run_dir / "episodes.jsonl"
    metrics_path = run_dir / "metrics.jsonl"

    try:
        with episodes_path.open("x", encoding="utf-8") as episodes_file, metrics_path.open("x", encoding="utf-8") as metrics_file:
            budget = int(training["environment_decisions_per_iteration"])
            coverage_steps = int(training["uniform_option_coverage_decisions"])
            epsilon = float(training["epsilon_after_coverage"])
            for decision_index in range(budget):
                environment_steps = decision_index + 1
                features, base_action = infer_base_and_features(actor, observation, device)
                at_boundary = option_state.at_boundary
                values = None
                if at_boundary:
                    greedy_values = q_values_for(online, features, base_action, device)
                    if decision_index < coverage_steps or option_rng.random() < epsilon:
                        selected_for_exploration = ResidualOption(int(option_rng.integers(len(ResidualOption))))
                        values = exploration_values(selected_for_exploration)
                    else:
                        values = greedy_values
                native_action, selected_option = option_state.act(base_action, values)
                if at_boundary:
                    accumulator = OptionTransitionAccumulator(
                        features,
                        base_action,
                        selected_option,
                        gamma=training["double_q"]["gamma_per_decision"],
                        max_duration=training["option_duration_decisions"],
                    )
                if accumulator is None:
                    raise RuntimeError("missing option-boundary transition accumulator")

                transition = collector.step(native_action)
                episode_steps += 1
                episode_reward += transition.reward
                episode_option_counts[selected_option.name] += 1
                option_counts[selected_option.name] += 1
                last_info = transition.info
                option_complete = option_state.at_boundary
                transition_complete = option_complete or transition.done
                if transition_complete:
                    next_features = next_base_action = None
                    if not transition.terminal:
                        next_features, next_base_action = infer_base_and_features(
                            actor, transition.next_observation, device,
                        )
                else:
                    next_features = next_base_action = None
                option_transition = accumulator.append(
                    transition.reward,
                    next_features=next_features,
                    next_base_action=next_base_action,
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                    terminal=transition.terminal,
                )
                if transition_complete:
                    if option_transition is None:
                        raise RuntimeError("completed option did not emit its transition")
                    replay.add(option_transition)
                    accumulator = None
                    replay_settings = training["replay"]
                    if replay.size >= replay_settings["minimum_size_before_update"]:
                        batch = replay.sample(replay_settings["batch_size"])
                        losses.append(train_option_q_step(
                            online,
                            target,
                            optimizer,
                            batch,
                            frozen_actor_sha256=base_hash,
                            gamma=training["double_q"]["gamma_per_decision"],
                            max_grad_norm=training["double_q"]["gradient_clip_norm"],
                        ))
                        polyak_update_option_target(
                            target, online, training["double_q"]["target_tau"],
                        )
                        update_count += 1

                if transition.done:
                    completed_episodes += 1
                    episode_row = {
                        "episode": completed_episodes,
                        "track_id": transition.info.get("track_id"),
                        "geometry_seed": transition.info.get("seed"),
                        "decisions": episode_steps,
                        "raw_reward": episode_reward,
                        "finished": bool(transition.info.get("finished", False)),
                        "progress": float(transition.info.get("progress", 0.0)),
                        "damage": float(transition.info.get("damage", 0.0)),
                        "retire_reason": transition.info.get("retire_reason"),
                        "terminated": transition.terminated,
                        "truncated": transition.truncated,
                        "option_counts": dict(episode_option_counts),
                    }
                    episodes_file.write(json.dumps(episode_row, sort_keys=True) + "\n")
                    observation, _ = collector.reset()
                    option_state.reset()
                    episode_steps = 0
                    episode_reward = 0.0
                    episode_option_counts = Counter()
                else:
                    observation = transition.next_observation

                if environment_steps % 2048 == 0 or environment_steps == budget:
                    metric_row = {
                        "environment_decisions": environment_steps,
                        "completed_episodes": completed_episodes,
                        "replay_size": replay.size,
                        "gradient_steps": update_count,
                        "mean_recent_loss": float(np.mean([row["loss"] for row in losses[-256:]])) if losses else None,
                        "option_selections": dict(option_counts),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                    metrics_file.write(json.dumps(metric_row, sort_keys=True) + "\n")
                    metrics_file.flush()
    except Exception as exc:
        write_json(run_dir / "failure.json", {
            "status": "failed_before_training_completion",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "environment_decisions_completed": environment_steps,
            "option_transitions_in_replay": replay.size,
            "gradient_steps": update_count,
            "occurred_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise
    finally:
        environment.close()

    interrupted_episode = None
    if episode_steps:
        interrupted_episode = {
            "track_id": last_info.get("track_id"),
            "geometry_seed": last_info.get("seed"),
            "decisions": episode_steps,
            "raw_reward": episode_reward,
            "status": "budget_interrupted; not an episode outcome",
            "incomplete_option_discarded": accumulator is not None,
        }
    checkpoint_path = run_dir / "option_q.pt"
    torch.save({
        "format": "haic-drq-v2-residual-option-q-v1",
        "feature_dim": q_config["feature_dim"],
        "hidden_dim": q_config["hidden_dim"],
        "base_actor_sha256": base_hash,
        "option_q": online.cpu().state_dict(),
        "option_spec": {
            "duration": training["option_duration_decisions"],
            "steering_delta_native": training["steering_delta_native"],
            "brake_floor_official": training["brake_floor_official"],
            "intervention_margin_q_minus_keep": intervention_margin,
            "names": [option.name for option in ResidualOption],
        },
        "training_seed": seed,
        "gradient_steps": update_count,
        "replay_size": replay.size,
    }, checkpoint_path)
    torch.save(replay.state_dict(), run_dir / "option_replay.pt")
    result = {
        "status": "completed_bounded_training_pilot",
        "environment_decisions": environment_steps,
        "completed_episodes": completed_episodes,
        "completed_episode_finishes": sum(
            json.loads(line).get("finished", False)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
        ),
        "incomplete_final_episode": interrupted_episode,
        "option_transitions": replay.size,
        "gradient_steps": update_count,
        "option_action_counts": dict(option_counts),
        "mean_last_256_q_losses": float(np.mean([row["loss"] for row in losses[-256:]])) if losses else None,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "replay_snapshot": "option_replay.pt",
        "replay_sha256": sha256_file(run_dir / "option_replay.pt"),
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": "single-driver, single-learner internal feasibility run; no matched multi-seed or confirmation evidence",
    }
    write_json(run_dir / "result.json", result)
    manifest["status"] = "completed"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "manifest.completed.json", manifest)
    return checkpoint_path


def load_option_models(run_dir: Path):
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "option_q.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("format") != "haic-drq-v2-residual-option-q-v1":
        raise ValueError("unsupported residual option checkpoint")
    if payload.get("base_actor_sha256") != manifest["base_actor_sha256"]:
        raise ValueError("option Q checkpoint is bound to another frozen driver")
    actor_path = run_dir / "frozen_actor.pt"
    actor, _, _ = load_exported_actor(actor_path, device="cpu")
    if sha256_file(actor_path) != manifest["base_actor_sha256"]:
        raise ValueError("frozen actor copy does not match run manifest")
    online = ResidualOptionQ(payload["feature_dim"], payload["hidden_dim"])
    online.load_state_dict(payload["option_q"])
    online.eval()
    option_spec = payload["option_spec"]
    return actor, online, option_spec, manifest


def evaluate_cell(actor, head, option_spec, track_id, seed, protocol, composite: bool) -> dict:
    frame_skip = protocol["environment"]["frame_skip"]
    max_steps = protocol["environment"]["max_decisions_per_episode"]
    raw_env = HaicTrack(track_id, seed, max_steps, frame_skip, protocol["environment"]["obstacles"])
    env = TimeLimit(raw_env, max_episode_steps=max_steps)
    collector = EpisodeCollector(env, gamma=protocol["training"]["double_q"]["gamma_per_decision"])
    observation, _ = collector.reset()
    start_time = env.unwrapped.t
    option_state = ResidualOptionState(
        option_spec["duration"],
        intervention_margin=option_spec.get("intervention_margin_q_minus_keep", 0.0),
        steering_delta=option_spec["steering_delta_native"],
        brake_floor_official=option_spec["brake_floor_official"],
    )
    option_counts = Counter()
    interventions = 0
    total_reward = 0.0
    done = False
    steps = 0
    transition = None
    try:
        while not done and steps < max_steps:
            features, base_action = infer_base_and_features(actor, observation, torch.device("cpu"))
            if composite:
                values = q_values_for(head, features, base_action, torch.device("cpu")) if option_state.at_boundary else None
                native_action, option = option_state.act(base_action, values)
                option_counts[option.name] += 1
                interventions += int(not np.array_equal(native_action, base_action))
            else:
                native_action = base_action
            transition = collector.step(native_action)
            total_reward += transition.reward
            steps += 1
            observation = transition.next_observation
            done = transition.done
    finally:
        env.close()
    info = transition.info if transition is not None else {}
    finish_time = info.get("finish_time_s")
    return {
        "track_id": track_id,
        "geometry_seed": seed,
        "decisions": steps,
        "raw_reward": total_reward,
        "progress": float(info.get("progress", 0.0)),
        "finished": bool(info.get("finished", False)),
        "finish_time_s": finish_time,
        "lap_time_ms": round((finish_time - start_time) * 1000) if finish_time is not None else None,
        "damage": float(info.get("damage", 0.0)),
        "retire_reason": info.get("retire_reason"),
        "terminated": transition.terminated if transition is not None else False,
        "truncated": transition.truncated if transition is not None else False,
        "option_counts": dict(option_counts),
        "intervention_fraction": interventions / steps if steps and composite else 0.0,
        "done": done,
    }


def summarize_cells(rows: list[dict]) -> dict:
    return {
        "cells": len(rows),
        "finish_count": sum(row["finished"] for row in rows),
        "mean_progress": float(np.mean([row["progress"] for row in rows])) if rows else 0.0,
        "mean_raw_reward": float(np.mean([row["raw_reward"] for row in rows])) if rows else 0.0,
        "mean_damage": float(np.mean([row["damage"] for row in rows])) if rows else 0.0,
        "mean_lap_time_ms_finishers": (
            float(np.mean([row["lap_time_ms"] for row in rows if row["finished"]]))
            if any(row["finished"] for row in rows) else None
        ),
    }


def evaluate(protocol_path: Path, run_dir: Path, output_path: Path) -> Path:
    root = Path.cwd()
    protocol = load_protocol(protocol_path.resolve())
    run_dir = run_dir.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"evaluation output already exists: {output_path}")
    actor, head, option_spec, manifest = load_option_models(run_dir)
    if manifest["protocol_sha256"] != sha256_file(run_dir / "protocol.json"):
        raise ValueError("run protocol snapshot no longer matches manifest")
    if sha256_file(protocol_path.resolve()) != manifest["protocol_sha256"]:
        raise ValueError("evaluation protocol differs from training protocol")
    training = protocol["training"]
    expected_option_spec = {
        "duration": training["option_duration_decisions"],
        "steering_delta_native": training["steering_delta_native"],
        "brake_floor_official": training["brake_floor_official"],
        "intervention_margin_q_minus_keep": training.get("intervention_margin_q_minus_keep", 0.0),
    }
    if any(option_spec.get(key, 0.0) != value for key, value in expected_option_spec.items()):
        raise ValueError("option checkpoint action spec differs from evaluation protocol")
    actor.eval()
    head.eval()
    torch.set_num_threads(1)
    rows = []
    for track_id in protocol["development_screen"]["track_ids"]:
        for seed in protocol["development_screen"]["geometry_seeds"]:
            control = evaluate_cell(actor, head, option_spec, track_id, seed, protocol, composite=False)
            treatment = evaluate_cell(actor, head, option_spec, track_id, seed, protocol, composite=True)
            rows.append({
                "track_id": track_id,
                "geometry_seed": seed,
                "control": control,
                "residual_options": treatment,
                "paired_finish_delta": int(treatment["finished"]) - int(control["finished"]),
            })
    control_rows = [row["control"] for row in rows]
    treatment_rows = [row["residual_options"] for row in rows]
    control_summary = summarize_cells(control_rows)
    treatment_summary = summarize_cells(treatment_rows)
    paired_gains = sum(row["paired_finish_delta"] > 0 for row in rows)
    paired_losses = sum(row["paired_finish_delta"] < 0 for row in rows)
    paired_ties = len(rows) - paired_gains - paired_losses
    result = {
        "schema_version": 1,
        "status": "development_screen_only",
        "protocol": str(protocol_path.resolve().relative_to(root)),
        "protocol_sha256": sha256_file(protocol_path.resolve()),
        "base_actor_sha256": manifest["base_actor_sha256"],
        "option_checkpoint_sha256": sha256_file(run_dir / "option_q.pt"),
        "evaluation_runner_sha256": sha256_file(root / "scripts/drqv2_residual_options.py"),
        "option_core_sha256": sha256_file(root / "haic/algorithms/drq_v2/residual_options.py"),
        "device": "cpu",
        "torch_version": torch.__version__,
        "cells_per_policy": len(rows),
        "control": control_summary,
        "residual_options": treatment_summary,
        "paired_finish_cells": {"wins": paired_gains, "ties": paired_ties, "losses": paired_losses},
        "screen_gate": {
            "nonzero_candidate_finishes": treatment_summary["finish_count"] > 0,
            "candidate_finish_count_at_least_control": treatment_summary["finish_count"] >= control_summary["finish_count"],
            "passes_development_continuation_signal": (
                treatment_summary["finish_count"] > 0
                and treatment_summary["finish_count"] >= control_summary["finish_count"]
            ),
            "interpretation": "exploratory continuation signal only; not matched training-seed replication, confirmation, blind, or promotion",
        },
        "cells": rows,
        "limitations": [
            "One frozen base actor and one option learner; this does not establish independent-seed replication.",
            "The development seeds had no exact textual match in audited local artifacts, but incomplete historical pilot scheduling prevents a global freshness claim.",
            "This bespoke CPU development runner is not the tagged submission/evaluator path and does not establish package or official eligibility.",
        ],
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--protocol", type=Path, default=Path("experiments/drqv2-residual-options-pilot-v1.json"))
    train_parser.add_argument("--base-actor", type=Path)
    train_parser.add_argument("--run-dir", type=Path)
    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--protocol", type=Path, default=Path("experiments/drqv2-residual-options-pilot-v1.json"))
    eval_parser.add_argument("--run-dir", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = load_protocol(args.protocol)
    if args.command == "train":
        base_actor = args.base_actor or Path(protocol["base_driver"]["path"])
        run_dir = args.run_dir or Path("runs/20260924-drqv2-residual-options-pilot/iteration-1")
        result = train(args.protocol, run_dir, base_actor)
    else:
        result = evaluate(args.protocol, args.run_dir, args.output)
    print(result)


if __name__ == "__main__":
    main()
