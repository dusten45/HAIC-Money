"""Reproducible CPU training entry point for the pixel-only PPO policy."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from haic_agent.networks import VisualActorCritic
from training.env_factory import CollectedTransition, TrackSeedSplit, create_training_environment, split_track_seeds
from training.ppo import PPOConfig, PPOUpdater
from training.rollout import RolloutStorage


DEFAULT_SPLIT = split_track_seeds(
    train=((1, 101), (2, 102)), tune=((3, 201),), held_out=((4, 301),)
)


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
    """Convert collector-only state labels into the seven visual auxiliary targets."""
    labels = transition.labels
    return torch.tensor(
        (labels.speed, *labels.wheel_omega, labels.steering_angle, labels.yaw_rate),
        dtype=torch.float32,
    )


def shape_transition_reward(transition: CollectedTransition, previous_progress: float) -> float:
    """Add progress, finish, collision, off-track, damage, and time shaping to raw reward."""
    labels = transition.labels
    new_road = max(0.0, labels.tile_progress - previous_progress)
    shaped = float(transition.reward) + 10.0 * new_road - 0.01
    if labels.finished:
        shaped += 100.0
    if labels.collision:
        shaped -= 5.0
    if labels.off_track:
        shaped -= 10.0
    shaped -= 0.1 * labels.damage
    return shaped


def collect_rollout(
    model: VisualActorCritic,
    episodes: Iterable[tuple[int, int]],
    *,
    total_steps: int,
    max_decisions: int,
) -> RolloutStorage:
    """Collect a fixed number of four-raw-tick training transitions on CPU."""
    episode_list = tuple(episodes)
    if not episode_list:
        raise ValueError("at least one train episode is required")
    storage = RolloutStorage()
    episode_index = 0
    while len(storage) < total_steps:
        track_id, seed = episode_list[episode_index % len(episode_list)]
        episode_index += 1
        environment = create_training_environment(
            track_id=track_id, seed=seed, max_decisions=max_decisions
        )
        observation, _ = environment.reset()
        previous_progress = 0.0
        try:
            for _ in range(max_decisions):
                observation_tensor = torch.from_numpy(observation).unsqueeze(0)
                with torch.no_grad():
                    output = model(observation_tensor)
                    action, log_probability = model.sample_actions(output)
                transition = environment.step_transition(action.squeeze(0).cpu().numpy())
                with torch.no_grad():
                    next_value = model(
                        torch.from_numpy(transition.next_observation).unsqueeze(0)
                    ).value.item()
                storage.add(
                    observation=observation_tensor.squeeze(0),
                    action=action.squeeze(0),
                    log_probability=log_probability.item(),
                    value=output.value.item(),
                    next_value=next_value,
                    reward=shape_transition_reward(transition, previous_progress),
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                    auxiliary_targets=auxiliary_targets(transition),
                )
                previous_progress = transition.labels.tile_progress
                observation = transition.next_observation
                if transition.terminated or transition.truncated or len(storage) >= total_steps:
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


def evaluate_policy(
    model: VisualActorCritic,
    episodes: Iterable[tuple[int, int]],
    *,
    max_decisions: int,
) -> dict[str, float | None]:
    """Measure tune/held-out driving success; completion rate is the first metric."""
    results: list[dict[str, float | bool | None]] = []
    for track_id, seed in episodes:
        environment = create_training_environment(
            track_id=track_id, seed=seed, max_decisions=max_decisions
        )
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
        "mean_progress": float(np.mean([result["progress"] for result in results])) if results else 0.0,
        "mean_reward": float(np.mean([result["reward"] for result in results])) if results else 0.0,
        "auxiliary_mse": float(np.mean([result["auxiliary_mse"] for result in results])) if results else 0.0,
        "episodes": float(len(results)),
        "completed_episodes": float(len(finished)),
    }


def selection_score(metrics: dict[str, float | None]) -> tuple[float, float, float]:
    """Order checkpoints by completion, finished-lap time, then progress only."""
    lap_time = metrics["median_finished_lap_time_s"]
    lap_component = -float(lap_time) if lap_time is not None else float("-inf")
    return float(metrics["finish_rate"]), lap_component, float(metrics["mean_progress"])


def hud_ablation(
    model: VisualActorCritic, tune_episodes: Iterable[tuple[int, int]], *, max_decisions: int
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


def train(
    *,
    output_directory: Path,
    total_steps: int,
    max_decisions: int,
    seed: int,
    resume: Path | None = None,
    split: TrackSeedSplit = DEFAULT_SPLIT,
) -> dict[str, Any]:
    """Run a reproducible PPO update, tune evaluation, ablation, and checkpoint save."""
    set_reproducible_seed(seed)
    model = VisualActorCritic()
    updater = PPOUpdater(model, PPOConfig(minibatch_size=min(32, total_steps)))
    start_step = 0
    if resume is not None:
        start_step = load_checkpoint(resume, model, updater)["step"]
    rollout = collect_rollout(
        model, split.train, total_steps=total_steps, max_decisions=max_decisions
    )
    losses = dict(updater.update(rollout))
    tune_metrics = evaluate_policy(model, split.tune, max_decisions=max_decisions)
    ablation = hud_ablation(model, split.tune, max_decisions=max_decisions)
    checkpoint = output_directory / "policy.pt"
    metadata = {
        "seed": seed,
        "split": {"train": split.train, "tune": split.tune, "held_out": split.held_out},
        "tune_metrics": tune_metrics,
        "hud_ablation": ablation,
        "selection_metric": "finish_rate, negative_median_finished_lap_time_s, mean_progress",
    }
    save_checkpoint(checkpoint, model, updater, step=start_step + total_steps, metadata=metadata)
    return {"checkpoint": str(checkpoint), "losses": losses, "tune_metrics": tune_metrics, "ablation": ablation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/haic/visual-ppo"))
    parser.add_argument("--total-steps", type=int, default=2_048)
    parser.add_argument("--max-decisions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_steps = 8 if args.smoke else args.total_steps
    max_decisions = 4 if args.smoke else args.max_decisions
    result = train(
        output_directory=args.output,
        total_steps=total_steps,
        max_decisions=max_decisions,
        seed=args.seed,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
