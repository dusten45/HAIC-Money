"""Reproducible CPU training entry point for the pixel-only PPO policy."""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from haic_agent.networks import VisualActorCritic
from training.env_factory import (
    CollectedTransition,
    TrainingEpisode,
    TrainingSplit,
    create_episode_environment,
    create_training_environment,
    describe_split,
    split_track_seeds,
)
from training.imitation import behavioral_cloning_warmup, collect_teacher_demonstrations
from training.ppo import PPOConfig, PPOUpdater
from training.rollout import RolloutStorage
from training.site_maps import load_site_map_split


DEFAULT_SPLIT = split_track_seeds(
    train=((1, 101), (2, 102)), tune=((3, 201),), held_out=((4, 301),)
)
REWARD_SCALE = 0.1
AUXILIARY_TARGET_SCALES = (40.0, 100.0, 100.0, 100.0, 100.0, 0.4, 3.0, 1.0, 1.0, 1.0)
ROAD_TRACKING_LATERAL_WEIGHT = 0.5
ROAD_TRACKING_HEADING_WEIGHT = 0.5
MAX_NORMALIZED_LATERAL_ERROR = 2.0


def close_training_environment(environment: Any) -> None:
    """Close either a direct Gym environment or Task 1's collecting wrapper."""
    close = getattr(environment, "close", None)
    if callable(close):
        close()
        return
    environment.environment.close()


def set_reproducible_seed(seed: int) -> None:
    """Set every random source used by the CPU smoke and full training paths."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def auxiliary_targets(transition: CollectedTransition) -> torch.Tensor:
    """Normalize labels for the exact pixels used to choose the action."""
    labels = transition.observation_labels or transition.labels
    half_width = max(float(labels.road_half_width), 1e-6)
    normalized_lateral = float(
        np.clip(float(labels.lateral_error) / half_width, -2.0, 2.0)
    )
    values = (
        labels.speed,
        *labels.wheel_omega,
        labels.steering_angle,
        labels.yaw_rate,
        normalized_lateral,
        math.sin(float(labels.heading_error)),
        math.cos(float(labels.heading_error)),
    )
    return torch.tensor(values, dtype=torch.float32) / torch.tensor(
        AUXILIARY_TARGET_SCALES, dtype=torch.float32
    )


def shape_transition_reward(transition: CollectedTransition, previous_progress: float) -> float:
    """Add progress, finish, collision, off-track, damage, and time shaping to raw reward."""
    labels = transition.labels
    new_road = max(0.0, labels.tile_progress - previous_progress)
    shaped = (
        float(transition.reward)
        + 10.0 * new_road
        - 0.01
        + route_tracking_reward(labels)
    )
    if labels.finished:
        shaped += 100.0
    if labels.collision:
        shaped -= 5.0
    if labels.off_track:
        shaped -= 10.0
    shaped -= 0.1 * labels.damage
    return shaped * REWARD_SCALE


def route_tracking_reward(labels: Any) -> float:
    """Give PPO dense training feedback for route alignment and road centering."""
    half_width = max(float(labels.road_half_width), 1e-6)
    normalized_lateral_error = float(
        np.clip(
            float(labels.lateral_error) / half_width,
            -MAX_NORMALIZED_LATERAL_ERROR,
            MAX_NORMALIZED_LATERAL_ERROR,
        )
    )
    heading_error = float(labels.heading_error)
    return -ROAD_TRACKING_LATERAL_WEIGHT * normalized_lateral_error**2 - (
        ROAD_TRACKING_HEADING_WEIGHT * (1.0 - math.cos(heading_error))
    )


def _create_environment_for_episode(episode: TrainingEpisode, *, max_decisions: int):
    if isinstance(episode, tuple):
        track_id, seed = episode
        return create_training_environment(
            track_id=int(track_id), seed=int(seed), max_decisions=max_decisions
        )
    return create_episode_environment(episode, max_decisions=max_decisions)


def collect_rollout(
    model: VisualActorCritic,
    episodes: Iterable[TrainingEpisode],
    *,
    total_steps: int,
    max_decisions: int,
) -> RolloutStorage:
    """Collect a fixed number of four-raw-tick training transitions on CPU."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    episode_list = tuple(episodes)
    if not episode_list:
        raise ValueError("at least one train episode is required")
    storage = RolloutStorage()
    episode_index = 0
    while len(storage) < total_steps:
        episode = episode_list[episode_index % len(episode_list)]
        episode_index += 1
        environment = _create_environment_for_episode(episode, max_decisions=max_decisions)
        observation, _ = environment.reset()
        previous_progress = 0.0
        try:
            for decision_index in range(max_decisions):
                observation_tensor = torch.from_numpy(observation).unsqueeze(0)
                with torch.no_grad():
                    output = model(observation_tensor)
                    action, log_probability, pretransform_action = model.sample_actions_with_pretransform(output)
                transition = environment.step_transition(action.squeeze(0).cpu().numpy())
                with torch.no_grad():
                    next_value = model(
                        torch.from_numpy(transition.next_observation).unsqueeze(0)
                    ).value.item()
                collector_boundary = (
                    decision_index + 1 >= max_decisions or len(storage) + 1 >= total_steps
                )
                rollout_truncated = transition.truncated or (
                    collector_boundary and not transition.terminated
                )
                storage.add(
                    observation=observation_tensor.squeeze(0),
                    action=action.squeeze(0),
                    pretransform_action=pretransform_action.squeeze(0),
                    log_probability=log_probability.item(),
                    value=output.value.item(),
                    next_value=next_value,
                    reward=shape_transition_reward(transition, previous_progress),
                    terminated=transition.terminated,
                    truncated=rollout_truncated,
                    auxiliary_targets=auxiliary_targets(transition),
                )
                previous_progress = transition.labels.tile_progress
                observation = transition.next_observation
                if transition.terminated or rollout_truncated:
                    break
        finally:
            close_training_environment(environment)
    storage.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
    return storage


def save_checkpoint(
    path: Path, model: VisualActorCritic, updater: PPOUpdater, *, step: int, metadata: dict[str, Any]
) -> None:
    """Save a resumable CPU checkpoint under the configured artifacts path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": updater.optimizer.state_dict(),
            "step": int(step),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: Path, model: VisualActorCritic, updater: PPOUpdater) -> dict[str, Any]:
    """Load a checkpoint without changing the fixed inference model contract."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    updater.optimizer.load_state_dict(checkpoint["optimizer_state"])
    return {"step": int(checkpoint["step"]), "metadata": dict(checkpoint["metadata"])}


def save_best_checkpoint(
    path: Path,
    model: VisualActorCritic,
    updater: PPOUpdater,
    *,
    step: int,
    tune_metrics: dict[str, float | None],
    metadata: dict[str, Any],
) -> bool:
    """Persist a candidate only when its tune rank beats the stored checkpoint."""
    if path.is_file():
        stored = torch.load(path, map_location="cpu")
        stored_metrics = stored.get("metadata", {}).get("tune_metrics")
        if stored_metrics is not None and selection_score(tune_metrics) <= selection_score(stored_metrics):
            return False
    checkpoint_metadata = dict(metadata)
    checkpoint_metadata["tune_metrics"] = tune_metrics
    save_checkpoint(path, model, updater, step=step, metadata=checkpoint_metadata)
    return True


def evaluate_policy(
    model: VisualActorCritic,
    episodes: Iterable[TrainingEpisode],
    *,
    max_decisions: int,
) -> dict[str, float | None]:
    """Measure tune/held-out driving success; completion rate is the first metric."""
    results: list[dict[str, float | bool | None]] = []
    for episode in episodes:
        environment = _create_environment_for_episode(episode, max_decisions=max_decisions)
        observation, _ = environment.reset()
        total_reward = 0.0
        auxiliary_squared_errors: list[float] = []
        final_transition: CollectedTransition | None = None
        finish_time: float | None = None
        try:
            for _ in range(max_decisions):
                with torch.no_grad():
                    policy_output = model(torch.from_numpy(observation).unsqueeze(0))
                    action = model.deterministic_actions(policy_output)
                final_transition = environment.step_transition(action.squeeze(0).cpu().numpy())
                total_reward += final_transition.reward
                with torch.no_grad():
                    auxiliary_squared_errors.append(
                        torch.nn.functional.mse_loss(
                            policy_output.auxiliary_predictions.squeeze(0),
                            auxiliary_targets(final_transition),
                        ).item()
                    )
                observation = final_transition.next_observation
                if final_transition.terminated or final_transition.truncated:
                    break
            raw_finish_time = getattr(environment.unwrapped, "finish_time_s", None)
            finish_time = float(raw_finish_time) if raw_finish_time is not None else None
        finally:
            close_training_environment(environment)
        labels = final_transition.labels if final_transition is not None else None
        results.append(
            {
                "finished": bool(labels.finished) if labels else False,
                "progress": float(labels.tile_progress) if labels else 0.0,
                "reward": total_reward,
                "lap_time_s": finish_time,
                "auxiliary_mse": float(np.mean(auxiliary_squared_errors))
                if auxiliary_squared_errors
                else 0.0,
            }
        )
    finished = [result for result in results if result["finished"]]
    finish_times = [result["lap_time_s"] for result in finished if result["lap_time_s"] is not None]
    return {
        "finish_rate": float(np.mean([result["finished"] for result in results])) if results else 0.0,
        "median_finished_lap_time_s": float(np.median(finish_times)) if finish_times else None,
        "p90_finished_lap_time_s": float(np.percentile(finish_times, 90)) if finish_times else None,
        "mean_progress": float(np.mean([result["progress"] for result in results])) if results else 0.0,
        "mean_reward": float(np.mean([result["reward"] for result in results])) if results else 0.0,
        "auxiliary_mse": float(np.mean([result["auxiliary_mse"] for result in results])) if results else 0.0,
        "episodes": float(len(results)),
        "completed_episodes": float(len(finished)),
    }


def selection_score(metrics: dict[str, float | None]) -> tuple[float, float, float, float]:
    """Order checkpoints by completion, median/p90 finished lap time, then progress."""
    lap_time = metrics["median_finished_lap_time_s"]
    p90_lap_time = metrics.get("p90_finished_lap_time_s", lap_time)
    lap_component = -float(lap_time) if lap_time is not None else float("-inf")
    p90_component = -float(p90_lap_time) if p90_lap_time is not None else float("-inf")
    return float(metrics["finish_rate"]), lap_component, p90_component, float(metrics["mean_progress"])


def hud_ablation(
    model: VisualActorCritic, tune_episodes: Iterable[TrainingEpisode], *, max_decisions: int
) -> dict[str, dict[str, float | None]]:
    """Measure the HUD branch contribution on the tune split using the same weights."""
    enabled = evaluate_policy(model, tune_episodes, max_decisions=max_decisions)
    previous = model.use_hud
    model.use_hud = False
    try:
        disabled = evaluate_policy(model, tune_episodes, max_decisions=max_decisions)
    finally:
        model.use_hud = previous
    return {"hud_enabled": enabled, "hud_disabled": disabled}


def distribute_training_steps(total_steps: int, updates: int) -> tuple[int, ...]:
    """Split one run's environment budget into positive on-policy updates."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if updates < 1 or updates > total_steps:
        raise ValueError("updates must be between 1 and total_steps")
    steps_per_update, remainder = divmod(total_steps, updates)
    return tuple(
        steps_per_update + (1 if update_index < remainder else 0)
        for update_index in range(updates)
    )


def rollout_action_metrics(rollout: RolloutStorage) -> dict[str, float]:
    """Summarize actual sampled simulator actions for exploration diagnosis."""
    if not len(rollout):
        raise ValueError("cannot summarize an empty rollout")
    actions = torch.stack([step.action for step in rollout.steps]).float()
    return {
        "steer_mean": float(actions[:, 0].mean()),
        "steer_std": float(actions[:, 0].std(unbiased=False)),
        "mean_abs_steer": float(actions[:, 0].abs().mean()),
        "gas_mean": float(actions[:, 1].mean()),
        "gas_max": float(actions[:, 1].max()),
        "brake_mean": float(actions[:, 2].mean()),
        "brake_fraction": float((actions[:, 2] > 0.001).float().mean()),
        "pedal_overlap_fraction": float(
            ((actions[:, 1] > 0.0) & (actions[:, 2] > 0.0)).float().mean()
        ),
    }


def train(
    *,
    output_directory: Path,
    total_steps: int,
    max_decisions: int,
    seed: int,
    updates: int = 4,
    evaluation_max_decisions: int | None = None,
    learning_rate: float = 3e-4,
    resume: Path | None = None,
    split: TrainingSplit = DEFAULT_SPLIT,
    teacher_warmup_epochs: int = 3,
    teacher_max_decisions: int = 800,
    teacher_warmup_batch_size: int = 32,
    teacher_warmup_learning_rate: float = 1e-3,
) -> dict[str, Any]:
    """Warm-start the pixel actor from train-only demonstrations, then run PPO."""
    started = time.perf_counter()
    selection_max_decisions = (
        max_decisions if evaluation_max_decisions is None else evaluation_max_decisions
    )
    if max_decisions < 1 or selection_max_decisions < 1:
        raise ValueError("max_decisions and evaluation_max_decisions must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if teacher_warmup_epochs < 0:
        raise ValueError("teacher_warmup_epochs must be non-negative")
    if teacher_max_decisions < 1:
        raise ValueError("teacher_max_decisions must be positive")
    if teacher_warmup_batch_size < 1:
        raise ValueError("teacher_warmup_batch_size must be positive")
    if not np.isfinite(teacher_warmup_learning_rate) or teacher_warmup_learning_rate <= 0:
        raise ValueError("teacher_warmup_learning_rate must be finite and positive")
    set_reproducible_seed(seed)
    update_step_counts = distribute_training_steps(total_steps, updates)
    model = VisualActorCritic()
    updater = PPOUpdater(
        model,
        PPOConfig(
            learning_rate=learning_rate,
            minibatch_size=min(32, min(update_step_counts)),
        ),
    )
    start_step = 0
    resume_metadata: dict[str, Any] = {}
    if resume is not None:
        loaded = load_checkpoint(resume, model, updater)
        start_step = loaded["step"]
        resume_metadata = loaded["metadata"]
        for parameter_group in updater.optimizer.param_groups:
            parameter_group["lr"] = learning_rate
    checkpoint = output_directory / "policy.pt"
    update_results: list[dict[str, Any]] = []
    best_tune_metrics: dict[str, float | None] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    checkpoint_updated = False
    completed_steps = 0
    timings_s = {
        "teacher_data_collection_s": 0.0,
        "teacher_warmup_s": 0.0,
        "rollout_s": 0.0,
        "ppo_update_s": 0.0,
        "tune_selection_eval_s": 0.0,
        "full_tune_eval_s": 0.0,
        "hud_ablation_s": 0.0,
    }
    previous_warmup = resume_metadata.get("teacher_warmup")
    teacher_decision_limit = min(teacher_max_decisions, max_decisions)
    if isinstance(previous_warmup, dict) and previous_warmup.get("enabled"):
        teacher_warmup_metrics = dict(previous_warmup)
        teacher_warmup_metrics["reused_from_checkpoint"] = True
    elif teacher_warmup_epochs == 0:
        teacher_warmup_metrics = {
            "enabled": False,
            "teacher": "vision_corridor_training_only",
            "epochs": 0,
            "demonstration_episodes": 0,
            "demonstration_steps": 0,
            "teacher_loss_used_during_ppo": False,
        }
    else:
        collection_started = time.perf_counter()
        demonstrations = collect_teacher_demonstrations(
            split, max_decisions=teacher_decision_limit
        )
        timings_s["teacher_data_collection_s"] = (
            time.perf_counter() - collection_started
        )
        warmup_started = time.perf_counter()
        teacher_warmup_metrics = behavioral_cloning_warmup(
            model,
            demonstrations,
            epochs=teacher_warmup_epochs,
            batch_size=teacher_warmup_batch_size,
            learning_rate=teacher_warmup_learning_rate,
        )
        timings_s["teacher_warmup_s"] = time.perf_counter() - warmup_started
        teacher_warmup_metrics["data_collection_s"] = timings_s[
            "teacher_data_collection_s"
        ]
        teacher_warmup_metrics["warmup_s"] = timings_s["teacher_warmup_s"]
        teacher_warmup_metrics["max_decisions_per_episode"] = teacher_decision_limit

    for update_number, update_steps in enumerate(update_step_counts, start=1):
        rollout_started = time.perf_counter()
        rollout = collect_rollout(
            model, split.train, total_steps=update_steps, max_decisions=max_decisions
        )
        action_metrics = rollout_action_metrics(rollout)
        rollout_time = time.perf_counter() - rollout_started
        timings_s["rollout_s"] += rollout_time
        update_started = time.perf_counter()
        losses = dict(updater.update(rollout))
        ppo_update_time = time.perf_counter() - update_started
        timings_s["ppo_update_s"] += ppo_update_time
        completed_steps += len(rollout)
        tune_started = time.perf_counter()
        tune_metrics = evaluate_policy(
            model, split.tune, max_decisions=selection_max_decisions
        )
        tune_time = time.perf_counter() - tune_started
        timings_s["tune_selection_eval_s"] += tune_time
        metadata = {
            "seed": seed,
            "split": describe_split(split),
            "training_steps_this_run": completed_steps,
            "training_steps_requested": total_steps,
            "updates_completed": update_number,
            "updates_requested": updates,
            "learning_rate": learning_rate,
            "teacher_warmup": teacher_warmup_metrics,
            "reward_scale": REWARD_SCALE,
            "auxiliary_target_scales": AUXILIARY_TARGET_SCALES,
            "selection_max_decisions": selection_max_decisions,
            "full_evaluation_max_decisions": max_decisions,
            "training_action_metrics": action_metrics,
            "tune_metrics": tune_metrics,
            "selection_metric": "finish_rate, negative_median_finished_lap_time_s, negative_p90_finished_lap_time_s, mean_progress",
        }
        updated = save_best_checkpoint(
            checkpoint,
            model,
            updater,
            step=start_step + completed_steps,
            tune_metrics=tune_metrics,
            metadata=metadata,
        )
        checkpoint_updated = checkpoint_updated or updated
        if updated:
            best_tune_metrics = tune_metrics
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        update_results.append(
            {
                "update": update_number,
                "steps": len(rollout),
                "step": start_step + completed_steps,
                "losses": losses,
                "action_metrics": action_metrics,
                "tune_metrics": tune_metrics,
                "checkpoint_updated": updated,
                "timings_s": {
                    "rollout_s": rollout_time,
                    "ppo_update_s": ppo_update_time,
                    "tune_selection_eval_s": tune_time,
                },
            }
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    elif checkpoint.is_file():
        stored_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(stored_checkpoint["model_state"])
        best_tune_metrics = stored_checkpoint.get("metadata", {}).get("tune_metrics")
    if best_tune_metrics is None:
        best_tune_metrics = update_results[-1]["tune_metrics"]
    full_tune_started = time.perf_counter()
    full_tune_metrics = evaluate_policy(model, split.tune, max_decisions=max_decisions)
    timings_s["full_tune_eval_s"] = time.perf_counter() - full_tune_started
    ablation_started = time.perf_counter()
    ablation = hud_ablation(
        model, split.tune, max_decisions=selection_max_decisions
    )
    timings_s["hud_ablation_s"] = time.perf_counter() - ablation_started
    timings_s["wall_s"] = time.perf_counter() - started
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_updated": checkpoint_updated,
        "losses": update_results[-1]["losses"],
        "teacher_warmup": teacher_warmup_metrics,
        "tune_metrics": best_tune_metrics,
        "full_tune_metrics": full_tune_metrics,
        "ablation": ablation,
        "updates": update_results,
        "updates_completed": len(update_results),
        "training_steps": completed_steps,
        "step": start_step + completed_steps,
        "timings_s": timings_s,
        "wall_time_s": timings_s["wall_s"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/haic/visual-ppo"))
    parser.add_argument("--total-steps", type=int, default=2_048)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    parser.add_argument("--evaluation-max-decisions", type=int, default=800)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--teacher-warmup-epochs", type=int, default=3)
    parser.add_argument("--teacher-max-decisions", type=int, default=800)
    parser.add_argument("--teacher-warmup-batch-size", type=int, default=32)
    parser.add_argument("--teacher-warmup-learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--site-map-split", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_steps = 8 if args.smoke else args.total_steps
    max_decisions = 4 if args.smoke else args.max_decisions
    evaluation_max_decisions = 4 if args.smoke else args.evaluation_max_decisions
    updates = 1 if args.smoke else args.updates
    teacher_warmup_epochs = min(args.teacher_warmup_epochs, 1) if args.smoke else args.teacher_warmup_epochs
    split = load_site_map_split(args.site_map_split) if args.site_map_split else DEFAULT_SPLIT
    result = train(
        output_directory=args.output,
        total_steps=total_steps,
        max_decisions=max_decisions,
        seed=args.seed,
        updates=updates,
        evaluation_max_decisions=evaluation_max_decisions,
        learning_rate=args.learning_rate,
        resume=args.resume,
        split=split,
        teacher_warmup_epochs=teacher_warmup_epochs,
        teacher_max_decisions=args.teacher_max_decisions,
        teacher_warmup_batch_size=args.teacher_warmup_batch_size,
        teacher_warmup_learning_rate=args.teacher_warmup_learning_rate,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
