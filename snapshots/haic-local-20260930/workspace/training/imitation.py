"""Training-only visual teacher demonstrations and actor warm-start."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional as F

from haic_agent.networks import (
    DEFAULT_PEDAL_EXPANSION,
    VisualActorCritic,
)
from training.env_factory import TrainingSplit, create_episode_environment
from training.vision_teacher import VisionCorridorAgent

TEACHER_POLICY_CAP_FRACTION = 0.75


def teacher_pedal_reference(teacher: Any) -> tuple[float, float]:
    """Return the pedal range the given teacher instance actually commands.

    The class constants only describe the default profile. Faster benchmark
    profiles raise ``max_gas`` (``sprint_guarded`` uses 0.24), and normalizing
    by the class constant silently clipped every demonstration above 0.12,
    which capped the imitation target far below the teacher it copies.
    """
    max_gas = float(getattr(teacher, "max_gas", VisionCorridorAgent.MAX_TEACHER_GAS))
    max_brake = float(
        getattr(teacher, "max_brake", VisionCorridorAgent.MAX_TEACHER_BRAKE)
    )
    if not math.isfinite(max_gas) or max_gas <= 0.0:
        raise ValueError("teacher max_gas must be finite and positive")
    if not math.isfinite(max_brake) or max_brake <= 0.0:
        raise ValueError("teacher max_brake must be finite and positive")
    return max_gas, max_brake


@dataclass(frozen=True)
class TeacherDemonstrations:
    """CPU tensors containing train-split pixels and unconstrained actor targets."""

    observations: torch.Tensor
    pretransform_actions: torch.Tensor
    episodes: int
    steps: int
    collection_method: str = "expert"
    teacher_action_fraction: float = 1.0


def _close_environment(environment: Any) -> None:
    close = getattr(environment, "close", None)
    if callable(close):
        close()
        return
    environment.environment.close()


def _teacher_action_tensor(
    action: Any,
    *,
    teacher_pedal_range: tuple[float, float] = (
        VisionCorridorAgent.MAX_TEACHER_GAS,
        VisionCorridorAgent.MAX_TEACHER_BRAKE,
    ),
    pedal_expansion: float = DEFAULT_PEDAL_EXPANSION,
    cap_fraction: float = TEACHER_POLICY_CAP_FRACTION,
) -> torch.Tensor:
    try:
        values = np.asarray(action, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("teacher action must be finite [steer, gas, brake]") from error
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("teacher action must be finite [steer, gas, brake]")
    if not math.isfinite(cap_fraction) or not 0.0 < cap_fraction <= 1.0:
        raise ValueError("cap_fraction must be finite and in (0, 1]")
    teacher_max_gas, teacher_max_brake = teacher_pedal_range
    throttle_limit, brake_limit = VisualActorCritic.pedal_limits(pedal_expansion)
    bounded_values = values.copy()
    bounded_values[0] = np.clip(bounded_values[0], -1.0, 1.0)
    bounded_values[1] = (
        np.clip(bounded_values[1], 0.0, teacher_max_gas)
        / teacher_max_gas
        * throttle_limit
        * cap_fraction
    )
    bounded_values[2] = (
        np.clip(bounded_values[2], 0.0, teacher_max_brake)
        / teacher_max_brake
        * brake_limit
        * cap_fraction
    )
    bounded = torch.from_numpy(bounded_values).unsqueeze(0)
    pretransform, _ = VisualActorCritic._unbound_actions(bounded, pedal_expansion)
    return pretransform.squeeze(0)


def collect_teacher_demonstrations(
    split: TrainingSplit,
    *,
    max_decisions: int,
    teacher_factory: Callable[[], Any] = VisionCorridorAgent,
    pedal_expansion: float = DEFAULT_PEDAL_EXPANSION,
    cap_fraction: float = TEACHER_POLICY_CAP_FRACTION,
) -> TeacherDemonstrations:
    """Collect teacher actions from the training episodes only.

    Tune and held-out episodes are deliberately inaccessible to this collector;
    it accepts a split object and iterates only ``split.train``.
    """
    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    episodes = tuple(split.train)
    if not episodes:
        raise ValueError("at least one train episode is required for teacher warm-up")

    observations: list[torch.Tensor] = []
    pretransform_actions: list[torch.Tensor] = []
    collected_episodes = 0
    for episode in episodes:
        environment = create_episode_environment(episode, max_decisions=max_decisions)
        teacher = teacher_factory()
        try:
            observation, _ = environment.reset()
            reset = getattr(teacher, "reset", None)
            if callable(reset):
                reset(observation)
            for _ in range(max_decisions):
                pixels = np.asarray(observation, dtype=np.float32)
                if (
                    pixels.shape != (4, 84, 84)
                    or not np.all(np.isfinite(pixels))
                    or float(pixels.min()) < 0.0
                    or float(pixels.max()) > 1.0
                ):
                    raise ValueError("teacher observation must be finite normalized 4x84x84 pixels")
                action = np.asarray(teacher.act(observation), dtype=np.float32)
                target = _teacher_action_tensor(
                    action,
                    teacher_pedal_range=teacher_pedal_reference(teacher),
                    pedal_expansion=pedal_expansion,
                    cap_fraction=cap_fraction,
                )
                policy_action = VisualActorCritic._bound_actions(
                    target.unsqueeze(0), pedal_expansion
                )[0].numpy()
                observations.append(
                    torch.from_numpy(np.rint(pixels * 255.0).astype(np.uint8))
                )
                pretransform_actions.append(target)
                transition = environment.step_transition(policy_action)
                observation = transition.next_observation
                if transition.terminated or transition.truncated:
                    break
            collected_episodes += 1
        finally:
            _close_environment(environment)

    if not observations:
        raise RuntimeError("teacher collection produced no transitions")
    return TeacherDemonstrations(
        observations=torch.stack(observations),
        pretransform_actions=torch.stack(pretransform_actions).float(),
        episodes=collected_episodes,
        steps=len(observations),
        collection_method="expert",
        teacher_action_fraction=1.0,
    )


def collect_dagger_demonstrations(
    policy: VisualActorCritic,
    split: TrainingSplit,
    *,
    max_decisions: int,
    teacher_action_probability: float = 0.5,
    seed: int = 0,
    teacher_factory: Callable[[], Any] = VisionCorridorAgent,
    cap_fraction: float = TEACHER_POLICY_CAP_FRACTION,
) -> TeacherDemonstrations:
    """Label learner-visited training states while mixing in teacher actions.

    The student supplies visited observations; the teacher supplies the labels.
    Environment interaction is restricted to ``split.train`` and ``tune`` and
    ``held_out`` are never passed to the teacher collector.
    """
    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    if (
        not math.isfinite(teacher_action_probability)
        or teacher_action_probability < 0.0
        or teacher_action_probability > 1.0
    ):
        raise ValueError("teacher_action_probability must be between 0 and 1")
    episodes = tuple(split.train)
    if not episodes:
        raise ValueError("at least one train episode is required for DAgger collection")

    rng = np.random.default_rng(seed)
    observations: list[torch.Tensor] = []
    pretransform_actions: list[torch.Tensor] = []
    collected_episodes = 0
    teacher_executed_steps = 0
    was_training = policy.training
    policy.eval()
    try:
        for episode in episodes:
            environment = create_episode_environment(episode, max_decisions=max_decisions)
            teacher = teacher_factory()
            try:
                observation, _ = environment.reset()
                reset = getattr(teacher, "reset", None)
                if callable(reset):
                    reset(observation)
                for _ in range(max_decisions):
                    pixels = np.asarray(observation, dtype=np.float32)
                    if (
                        pixels.shape != (4, 84, 84)
                        or not np.all(np.isfinite(pixels))
                        or float(pixels.min()) < 0.0
                        or float(pixels.max()) > 1.0
                    ):
                        raise ValueError("DAgger observation must be finite normalized 4x84x84 pixels")
                    teacher_action = np.asarray(teacher.act(observation), dtype=np.float32)
                    target = _teacher_action_tensor(
                        teacher_action,
                        teacher_pedal_range=teacher_pedal_reference(teacher),
                        pedal_expansion=policy.pedal_expansion,
                        cap_fraction=cap_fraction,
                    )
                    bounded_teacher_action = (
                        VisualActorCritic._bound_actions(
                            target.unsqueeze(0), policy.pedal_expansion
                        )[0]
                        .cpu()
                        .numpy()
                    )
                    with torch.no_grad():
                        output = policy(torch.from_numpy(pixels.copy()).unsqueeze(0))
                        learner_action = (
                            policy.deterministic_actions(output)[0].cpu().numpy().astype(np.float32)
                        )
                    observations.append(
                        torch.from_numpy(np.rint(pixels * 255.0).astype(np.uint8))
                    )
                    pretransform_actions.append(target)
                    use_teacher = rng.random() < teacher_action_probability
                    executed_action = bounded_teacher_action if use_teacher else learner_action
                    teacher_executed_steps += int(use_teacher)
                    transition = environment.step_transition(executed_action)
                    observation = transition.next_observation
                    if transition.terminated or transition.truncated:
                        break
                collected_episodes += 1
            finally:
                _close_environment(environment)
    finally:
        policy.train(was_training)

    if not observations:
        raise RuntimeError("DAgger collection produced no transitions")
    return TeacherDemonstrations(
        observations=torch.stack(observations),
        pretransform_actions=torch.stack(pretransform_actions).float(),
        episodes=collected_episodes,
        steps=len(observations),
        collection_method="dagger",
        teacher_action_fraction=teacher_executed_steps / len(observations),
    )


def combine_demonstrations(
    demonstrations: tuple[TeacherDemonstrations, ...] | list[TeacherDemonstrations],
) -> TeacherDemonstrations:
    """Append expert and learner-state labels while keeping a compact pixel store."""
    items = tuple(demonstrations)
    if not items:
        raise ValueError("at least one demonstration set is required")
    for item in items:
        _validate_demonstrations(item)
    observations = []
    for item in items:
        if item.observations.dtype == torch.uint8:
            observations.append(item.observations)
        else:
            observations.append(
                torch.round(item.observations.float().clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            )
    total_steps = sum(item.steps for item in items)
    teacher_steps = sum(
        item.steps * float(item.teacher_action_fraction) for item in items
    )
    return TeacherDemonstrations(
        observations=torch.cat(observations, dim=0),
        pretransform_actions=torch.cat(
            [item.pretransform_actions.float() for item in items], dim=0
        ),
        episodes=sum(item.episodes for item in items),
        steps=total_steps,
        collection_method="aggregated",
        teacher_action_fraction=teacher_steps / total_steps,
    )


def _validate_demonstrations(demonstrations: TeacherDemonstrations) -> int:
    observations = demonstrations.observations
    actions = demonstrations.pretransform_actions
    if observations.ndim != 4 or tuple(observations.shape[1:]) != (4, 84, 84):
        raise ValueError("demonstration observations must have shape (N, 4, 84, 84)")
    if actions.ndim != 2 or tuple(actions.shape[1:]) != (2,):
        raise ValueError("pretransform actions must have shape (N, 2)")
    if observations.shape[0] < 1 or observations.shape[0] != actions.shape[0]:
        raise ValueError("demonstrations must contain matching non-empty observations and actions")
    if observations.device.type != "cpu" or actions.device.type != "cpu":
        raise ValueError("demonstrations must be stored on CPU")
    if not torch.isfinite(actions).all():
        raise ValueError("demonstration actions must contain only finite values")
    if observations.dtype == torch.uint8:
        pass
    elif observations.is_floating_point():
        if not torch.isfinite(observations).all():
            raise ValueError("demonstration observations must contain only finite values")
        if torch.any(observations < 0.0) or torch.any(observations > 1.0):
            raise ValueError("demonstration observations must be normalized to [0, 1]")
    else:
        raise ValueError("demonstration observations must be uint8 or normalized floats")
    return int(observations.shape[0])


def _normalized_observation_batch(observations: torch.Tensor) -> torch.Tensor:
    if observations.dtype == torch.uint8:
        return observations.float().div_(255.0)
    return observations.float()


def _mean_action_mse(
    model: VisualActorCritic,
    demonstrations: TeacherDemonstrations,
    *,
    batch_size: int,
) -> float:
    count = _validate_demonstrations(demonstrations)
    total = 0.0
    with torch.no_grad():
        for start in range(0, count, batch_size):
            end = min(count, start + batch_size)
            output = model(
                _normalized_observation_batch(demonstrations.observations[start:end])
            )
            batch_loss = F.mse_loss(
                output.action_mean,
                demonstrations.pretransform_actions[start:end],
                reduction="mean",
            )
            total += float(batch_loss) * (end - start)
    return total / count


def behavioral_cloning_warmup(
    model: VisualActorCritic,
    demonstrations: TeacherDemonstrations,
    *,
    epochs: int,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    cap_fraction: float = TEACHER_POLICY_CAP_FRACTION,
) -> dict[str, float | int | bool | str]:
    """Fit the visual actor mean briefly, before PPO takes over optimization."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    count = _validate_demonstrations(demonstrations)
    actor_parameters = (
        list(model.full_frame_encoder.parameters())
        + list(model.hud_encoder.parameters())
        + list(model.fusion.parameters())
        + list(model.policy_mean.parameters())
    )
    for optional_encoder in (
        model.visual_feature_encoder,
        model.temporal_feature_encoder,
    ):
        if optional_encoder is not None:
            actor_parameters.extend(optional_encoder.parameters())
    optimizer = torch.optim.Adam(actor_parameters, lr=learning_rate)
    was_training = model.training
    model.train()
    initial_loss = _mean_action_mse(model, demonstrations, batch_size=batch_size)
    for _ in range(epochs):
        permutation = torch.randperm(count)
        for start in range(0, count, batch_size):
            indices = permutation[start : start + batch_size]
            output = model(
                _normalized_observation_batch(demonstrations.observations[indices])
            )
            loss = F.mse_loss(
                output.action_mean, demonstrations.pretransform_actions[indices]
            )
            if not torch.isfinite(loss):
                raise RuntimeError("teacher warm-up produced a non-finite action loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, max_norm=1.0)
            optimizer.step()
    final_loss = _mean_action_mse(model, demonstrations, batch_size=batch_size)
    model.train(was_training)
    return {
        "enabled": True,
        "teacher": "vision_corridor_training_only",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "demonstration_episodes": int(demonstrations.episodes),
        "demonstration_steps": int(count),
        "demonstration_collection_method": demonstrations.collection_method,
        "teacher_action_fraction": float(demonstrations.teacher_action_fraction),
        "teacher_policy_cap_fraction": float(cap_fraction),
        "pedal_expansion": float(model.pedal_expansion),
        "throttle_limit": float(model.throttle_limit),
        "observation_storage_dtype": str(demonstrations.observations.dtype),
        "demonstration_storage_bytes": int(
            demonstrations.observations.nelement()
            * demonstrations.observations.element_size()
            + demonstrations.pretransform_actions.nelement()
            * demonstrations.pretransform_actions.element_size()
        ),
        "initial_action_mse": initial_loss,
        "final_action_mse": final_loss,
        "trained_visual_feature_encoder": model.visual_feature_encoder is not None,
        "trained_temporal_feature_encoder": model.temporal_feature_encoder is not None,
        "teacher_loss_used_during_ppo": False,
    }


__all__ = [
    "TeacherDemonstrations",
    "teacher_pedal_reference",
    "behavioral_cloning_warmup",
    "collect_dagger_demonstrations",
    "collect_teacher_demonstrations",
    "combine_demonstrations",
]
