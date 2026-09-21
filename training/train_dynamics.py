"""Train and evaluate the small latent-dynamics ensemble at decision cadence."""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from haic_agent.dynamics import DynamicsEnsembleOutput, LatentDynamicsEnsemble
from haic_agent.networks import ACTION_SIZE, LATENT_SIZE, VisualActorCritic
from training.env_factory import (
    CollectedTransition,
    TrainingEpisode,
    TrainingSplit,
    create_episode_environment,
    create_training_environment,
    describe_split,
    split_track_seeds,
)
from training.labels import collect_labels
from training.site_maps import load_site_map_split


DEFAULT_SPLIT = split_track_seeds(
    train=((1, 401), (2, 402)), tune=((3, 501),), held_out=((4, 601),)
)


class LatentEncoder(Protocol):
    def encode_observation(self, observations: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class DynamicsBatch:
    """Decision-level targets; risk values are train-only simulator labels."""

    latent: Tensor
    action: Tensor
    next_latent: Tensor
    progress_delta: Tensor
    reward: Tensor
    collision: Tensor
    off_track: Tensor

    def __post_init__(self) -> None:
        if self.latent.ndim != 2 or self.latent.shape[1] != LATENT_SIZE:
            raise ValueError("latent must have shape (B, 128)")
        batch_size = self.latent.shape[0]
        if self.action.shape != (batch_size, ACTION_SIZE):
            raise ValueError("action must have shape (B, 3)")
        if self.next_latent.shape != self.latent.shape:
            raise ValueError("next_latent must match latent")
        for name in ("progress_delta", "reward", "collision", "off_track"):
            if getattr(self, name).shape != (batch_size,):
                raise ValueError(f"{name} must have shape (B,)")


@dataclass(frozen=True)
class DynamicsLoss:
    total: Tensor
    latent: Tensor
    progress: Tensor
    reward: Tensor
    collision: Tensor
    off_track: Tensor


def build_transition_batch(
    encoder: LatentEncoder,
    transitions: Sequence[CollectedTransition],
    *,
    previous_progress: Sequence[float],
) -> DynamicsBatch:
    """Encode decision observations and use only the following decision's labels."""
    if not transitions:
        raise ValueError("at least one collected transition is required")
    if len(previous_progress) != len(transitions):
        raise ValueError("previous_progress must align one-to-one with transitions")
    if any(transition.label_timing != "next_decision" for transition in transitions):
        raise ValueError("dynamics targets require labels from the next decision boundary")
    observations = torch.from_numpy(np.stack([transition.observation for transition in transitions])).float()
    next_observations = torch.from_numpy(np.stack([transition.next_observation for transition in transitions])).float()
    with torch.no_grad():
        latent = encoder.encode_observation(observations).detach().cpu().float()
        next_latent = encoder.encode_observation(next_observations).detach().cpu().float()
    labels = [transition.labels for transition in transitions]
    return DynamicsBatch(
        latent=latent,
        action=torch.from_numpy(np.stack([transition.action for transition in transitions])).float(),
        next_latent=next_latent,
        progress_delta=torch.tensor(
            [label.tile_progress - progress for label, progress in zip(labels, previous_progress)], dtype=torch.float32
        ),
        reward=torch.tensor([transition.reward for transition in transitions], dtype=torch.float32),
        collision=torch.tensor([float(label.collision) for label in labels], dtype=torch.float32),
        off_track=torch.tensor([float(label.off_track) for label in labels], dtype=torch.float32),
    )


def rare_event_positive_weight(target: Tensor) -> Tensor:
    """Return a class-balanced positive weight for rare binary risk labels."""
    positives = target.sum()
    negatives = target.numel() - positives
    if positives <= 0:
        return torch.ones((), device=target.device, dtype=target.dtype)
    return torch.clamp(negatives / positives, min=1.0)


def member_bootstrap_indices(
    num_members: int,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Draw an independent same-size bootstrap sample for each ensemble member."""
    if num_members < 1 or batch_size < 1:
        raise ValueError("num_members and batch_size must both be positive")
    return torch.randint(
        batch_size,
        (num_members, batch_size),
        generator=generator,
        device=device,
    )


def _bootstrap_indices_for(output: DynamicsEnsembleOutput, batch: DynamicsBatch, indices: Tensor | None) -> Tensor:
    member_count, batch_size = output.progress_delta.shape
    if indices is None:
        return torch.arange(batch_size, device=batch.latent.device).repeat(member_count, 1)
    if indices.shape != (member_count, batch_size):
        raise ValueError("member bootstrap indices must have shape (num_members, batch_size)")
    return indices.to(device=batch.latent.device, dtype=torch.long)


def _memberwise_mse(prediction: Tensor, target: Tensor, indices: Tensor) -> Tensor:
    return torch.stack(
        [F.mse_loss(prediction[member, index], target[index]) for member, index in enumerate(indices)]
    ).mean()


def _memberwise_risk_loss(logits: Tensor, target: Tensor, indices: Tensor) -> Tensor:
    return torch.stack(
        [
            F.binary_cross_entropy_with_logits(
                logits[member, index],
                target[index],
                pos_weight=rare_event_positive_weight(target[index]),
            )
            for member, index in enumerate(indices)
        ]
    ).mean()


def dynamics_loss(
    output: DynamicsEnsembleOutput, batch: DynamicsBatch, *, member_indices: Tensor | None = None
) -> DynamicsLoss:
    """Score every member against aligned one-decision targets."""
    indices = _bootstrap_indices_for(output, batch, member_indices)
    target_latent_residual = batch.next_latent - batch.latent
    latent = _memberwise_mse(output.next_latent_residual, target_latent_residual, indices)
    progress = _memberwise_mse(output.progress_delta, batch.progress_delta, indices)
    reward = _memberwise_mse(output.reward, batch.reward, indices)
    collision = _memberwise_risk_loss(output.collision_logits, batch.collision, indices)
    off_track = _memberwise_risk_loss(output.off_track_logits, batch.off_track, indices)
    total = latent + progress + reward + collision + off_track
    return DynamicsLoss(total, latent, progress, reward, collision, off_track)


def train_dynamics_step(
    model: LatentDynamicsEnsemble, optimizer: torch.optim.Optimizer, batch: DynamicsBatch
) -> dict[str, float]:
    """Perform one finite, class-balanced supervised update."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    indices = member_bootstrap_indices(
        len(model.members), batch.latent.shape[0], device=batch.latent.device
    )
    loss = dynamics_loss(model(batch.latent, batch.action), batch, member_indices=indices)
    loss.total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return {
        "total_loss": float(loss.total.detach()),
        "latent_loss": float(loss.latent.detach()),
        "progress_loss": float(loss.progress.detach()),
        "reward_loss": float(loss.reward.detach()),
        "collision_loss": float(loss.collision.detach()),
        "off_track_loss": float(loss.off_track.detach()),
    }


def evaluate_dynamics(model: LatentDynamicsEnsemble, batch: DynamicsBatch) -> dict[str, float]:
    """Return held-out next-latent and train-only risk metrics."""
    model.eval()
    with torch.no_grad():
        prediction = model.predict(batch.latent, batch.action)
        collision_target = batch.collision
        off_track_target = batch.off_track
        return {
            "latent_mse": float(F.mse_loss(prediction.next_latent, batch.next_latent)),
            "progress_mse": float(F.mse_loss(prediction.progress_delta, batch.progress_delta)),
            "reward_mse": float(F.mse_loss(prediction.reward, batch.reward)),
            "collision_brier": float(F.mse_loss(prediction.collision_probability, collision_target)),
            "off_track_brier": float(F.mse_loss(prediction.off_track_probability, off_track_target)),
            "collision_accuracy": float(((prediction.collision_probability >= 0.5).float() == collision_target).float().mean()),
            "off_track_accuracy": float(((prediction.off_track_probability >= 0.5).float() == off_track_target).float().mean()),
            "mean_epistemic_uncertainty": float(prediction.uncertainty.mean()),
        }


def save_dynamics_checkpoint(
    path: Path,
    model: LatentDynamicsEnsemble,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    metadata: dict[str, Any],
) -> None:
    """Save CPU-portable weights and the training state needed to resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": int(step), "metadata": metadata},
        path,
    )


def load_dynamics_checkpoint(
    path: Path, model: LatentDynamicsEnsemble, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    """Load a dynamics checkpoint on CPU for training or planner packaging."""
    saved = torch.load(path, map_location="cpu")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    return saved


def _close_environment(environment: Any) -> None:
    close = getattr(environment, "close", None)
    if callable(close):
        close()
    else:
        environment.environment.close()


def collect_dynamics_batch(
    policy: VisualActorCritic,
    episodes: Iterable[TrainingEpisode],
    *,
    max_decisions: int,
) -> DynamicsBatch:
    """Collect policy transitions at the four-raw-tick decision cadence."""
    transitions: list[CollectedTransition] = []
    previous_progress: list[float] = []
    for episode in episodes:
        if isinstance(episode, tuple):
            track_id, seed = episode
            environment = create_training_environment(
                track_id=int(track_id), seed=int(seed), max_decisions=max_decisions
            )
        else:
            environment = create_episode_environment(episode, max_decisions=max_decisions)
        observation, reset_info = environment.reset()
        progress = collect_labels(environment, reset_info).tile_progress
        try:
            for _ in range(max_decisions):
                with torch.no_grad():
                    output = policy(torch.from_numpy(observation).unsqueeze(0))
                    action, _ = policy.sample_actions(output)
                    action = action.squeeze(0).cpu().numpy()
                transition = environment.step_transition(action)
                transitions.append(transition)
                previous_progress.append(progress)
                progress = transition.labels.tile_progress
                observation = transition.next_observation
                if transition.terminated or transition.truncated:
                    break
        finally:
            _close_environment(environment)
    return build_transition_batch(policy, transitions, previous_progress=previous_progress)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def collection_policy(policy_checkpoint: Path | None, *, allow_random_policy: bool = False) -> tuple[VisualActorCritic, str]:
    """Load the PPO policy whose latent/action distribution feeds dynamics data."""
    if policy_checkpoint is None:
        if not allow_random_policy:
            raise ValueError("a policy checkpoint is required outside explicit smoke development")
        return VisualActorCritic().eval(), "random_init_smoke"
    checkpoint = torch.load(policy_checkpoint, map_location="cpu")
    if "model_state" not in checkpoint:
        raise ValueError("policy checkpoint does not contain Task 2 model_state weights")
    policy = VisualActorCritic()
    policy.load_state_dict(checkpoint["model_state"])
    return policy.eval(), str(policy_checkpoint)


def train(
    *,
    output_directory: Path,
    steps: int,
    max_decisions: int,
    seed: int,
    policy_checkpoint: Path | None = None,
    allow_random_policy: bool = False,
    split: TrainingSplit = DEFAULT_SPLIT,
) -> dict[str, Any]:
    """Collect train transitions, fit the ensemble, and report held-out metrics."""
    set_reproducible_seed(seed)
    policy, policy_source = collection_policy(
        policy_checkpoint, allow_random_policy=allow_random_policy
    )
    train_batch = collect_dynamics_batch(policy, split.train, max_decisions=max_decisions)
    model = LatentDynamicsEnsemble()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics: dict[str, float] = {}
    for _ in range(steps):
        metrics = train_dynamics_step(model, optimizer, train_batch)
    held_out_batch = collect_dynamics_batch(policy, split.held_out, max_decisions=max_decisions)
    held_out_metrics = evaluate_dynamics(model, held_out_batch)
    checkpoint = output_directory / "dynamics.pt"
    save_dynamics_checkpoint(
        checkpoint,
        model,
        optimizer,
        step=steps,
        metadata={
            "seed": seed,
            "split": describe_split(split),
            "held_out_metrics": held_out_metrics,
            "target_cadence": "next_decision_after_four_raw_ticks",
            "policy_source": policy_source,
        },
    )
    return {"checkpoint": str(checkpoint), "train_losses": metrics, "held_out_metrics": held_out_metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/haic/latent-dynamics"))
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--max-decisions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--site-map-split", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(
        output_directory=args.output,
        steps=3 if args.smoke else args.steps,
        max_decisions=2 if args.smoke else args.max_decisions,
        seed=args.seed,
        policy_checkpoint=args.policy_checkpoint,
        allow_random_policy=args.smoke,
        split=load_site_map_split(args.site_map_split) if args.site_map_split else DEFAULT_SPLIT,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
