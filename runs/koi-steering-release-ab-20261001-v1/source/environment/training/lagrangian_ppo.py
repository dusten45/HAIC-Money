"""Training-only PPO interfaces for an episodic Lagrangian cost critic.

The submitted actor remains ``VisualActorCritic``. This module wraps it only
while training and exports the actor's original state dict without the cost
value head.
"""

from collections import OrderedDict
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from haic_agent.networks import LATENT_SIZE, PolicyOutput, VisualActorCritic
from training.lagrangian import (
    CompletedEpisodeCost,
    EpisodeBoundary,
    IncompleteEpisodeCost,
    LagrangeMultiplier,
)
from training.ppo import PPOConfig, PPOUpdater


@dataclass(frozen=True)
class LagrangianPolicyOutput:
    """Actor output plus the training-only cost-value estimate."""

    policy_output: PolicyOutput
    cost_value: Tensor

    @property
    def latent(self) -> Tensor:
        return self.policy_output.latent

    @property
    def action_mean(self) -> Tensor:
        return self.policy_output.action_mean

    @property
    def action_log_std(self) -> Tensor:
        return self.policy_output.action_log_std

    @property
    def value(self) -> Tensor:
        return self.policy_output.value

    @property
    def auxiliary_predictions(self) -> Tensor:
        return self.policy_output.auxiliary_predictions


class LagrangianActorCritic(nn.Module):
    """Attach a cost critic to the existing image-only actor latent."""

    def __init__(self, policy: VisualActorCritic) -> None:
        super().__init__()
        if not isinstance(policy, VisualActorCritic):
            raise TypeError("policy must be a VisualActorCritic")
        self.policy = policy
        self.cost_value_head = nn.Linear(LATENT_SIZE, 1)
        nn.init.zeros_(self.cost_value_head.weight)
        nn.init.zeros_(self.cost_value_head.bias)

    @property
    def throttle_limit(self) -> float:
        return self.policy.throttle_limit

    @property
    def pedal_expansion(self) -> float:
        return self.policy.pedal_expansion

    @property
    def brake_limit(self) -> float:
        return self.policy.brake_limit

    @property
    def pedal_scale(self):
        return self.policy.pedal_scale

    @property
    def use_hud(self) -> bool:
        return self.policy.use_hud

    @use_hud.setter
    def use_hud(self, value: bool) -> None:
        self.policy.use_hud = bool(value)

    @property
    def last_visual_features(self) -> Tensor | None:
        """Preserve the actor's train-only visual-feature recorder interface."""
        return self.policy.last_visual_features

    @property
    def last_temporal_features(self) -> Tensor | None:
        return self.policy.last_temporal_features

    def forward(self, observation_tensor: Tensor) -> LagrangianPolicyOutput:
        policy_output = self.policy(observation_tensor)
        cost_value = self.cost_value_head(policy_output.latent).squeeze(-1)
        return LagrangianPolicyOutput(policy_output, cost_value)

    def sample_actions_with_pretransform(
        self, output: LagrangianPolicyOutput
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.policy.sample_actions_with_pretransform(output.policy_output)

    def deterministic_actions(self, output: LagrangianPolicyOutput) -> Tensor:
        return self.policy.deterministic_actions(output.policy_output)

    def log_probability_from_pretransform(
        self,
        pretransform_actions: Tensor,
        action_mean: Tensor,
        action_log_std: Tensor,
    ) -> Tensor:
        return self.policy.log_probability_from_pretransform(
            pretransform_actions, action_mean, action_log_std
        )

    @staticmethod
    def entropy(action_log_std: Tensor) -> Tensor:
        return VisualActorCritic.entropy(action_log_std)

    def actor_state_dict(self) -> OrderedDict[str, Tensor]:
        """Return a strict-loadable actor-only copy for inference packaging."""
        actor_state = self.policy.state_dict()
        return OrderedDict(
            (name, tensor.detach().clone()) for name, tensor in actor_state.items()
        )

    def load_actor_state_dict(self, state_dict, *, strict: bool = True):
        """Load actor-only weights without touching the training cost head."""
        return self.policy.load_state_dict(state_dict, strict=strict)


@dataclass
class LagrangianRolloutStep:
    observation: Tensor
    action: Tensor
    pretransform_action: Tensor
    old_log_probability: float
    reward: float
    reward_value: float
    next_reward_value: float
    cost: float
    cost_value: float
    next_cost_value: float
    boundary: EpisodeBoundary
    auxiliary_targets: Tensor
    reward_advantage: float = 0.0
    reward_return: float = 0.0
    cost_advantage: float = 0.0
    cost_return: float = 0.0


@dataclass(frozen=True)
class LagrangianRolloutBatch:
    observations: Tensor
    pretransform_actions: Tensor
    old_log_probabilities: Tensor
    advantages: Tensor
    raw_policy_advantages: Tensor
    reward_returns: Tensor
    cost_returns: Tensor
    auxiliary_targets: Tensor
    lagrange_multiplier: float


def _finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _discount(name: str, value: float) -> float:
    result = _finite_scalar(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _multiplier_value(value: float | LagrangeMultiplier) -> float:
    result = value.value if isinstance(value, LagrangeMultiplier) else value
    result = _finite_scalar("lagrange_multiplier", result)
    if result < 0.0:
        raise ValueError("lagrange_multiplier must be non-negative")
    return result


class LagrangianRolloutStorage:
    """Store separate reward/cost values and compute boundary-safe GAE."""

    def __init__(self) -> None:
        self.steps: list[LagrangianRolloutStep] = []
        self.completed_episode_costs: list[CompletedEpisodeCost] = []
        self.incomplete_episode_costs: list[IncompleteEpisodeCost] = []
        self.train_episode_outcomes: list[dict] = []
        self.incomplete_train_episode_outcomes: list[dict] = []
        self._advantages_ready = False

    def __len__(self) -> int:
        return len(self.steps)

    def add(
        self,
        *,
        observation: Tensor,
        action: Tensor,
        pretransform_action: Tensor,
        old_log_probability: float,
        reward: float,
        reward_value: float,
        next_reward_value: float,
        cost: float,
        cost_value: float,
        next_cost_value: float,
        boundary: EpisodeBoundary,
        auxiliary_targets: Tensor,
    ) -> None:
        if observation.shape != (4, 84, 84):
            raise ValueError("observation must have shape (4, 84, 84)")
        if action.shape != (3,):
            raise ValueError("action must have shape (3,)")
        if pretransform_action.shape != (2,):
            raise ValueError("pretransform_action must have shape (2,)")
        if auxiliary_targets.shape != (10,):
            raise ValueError("auxiliary_targets must have shape (10,)")
        if not isinstance(boundary, EpisodeBoundary):
            raise TypeError("boundary must be an EpisodeBoundary")
        checked_cost = _finite_scalar("cost", cost)
        if checked_cost < 0.0:
            raise ValueError("cost must be non-negative")
        scalar_values = {
            "old_log_probability": old_log_probability,
            "reward": reward,
            "reward_value": reward_value,
            "next_reward_value": next_reward_value,
            "cost_value": cost_value,
            "next_cost_value": next_cost_value,
        }
        checked = {
            name: _finite_scalar(name, value) for name, value in scalar_values.items()
        }
        if not torch.isfinite(observation).all():
            raise ValueError("observation must be finite")
        if not torch.isfinite(action).all():
            raise ValueError("action must be finite")
        if not torch.isfinite(pretransform_action).all():
            raise ValueError("pretransform_action must be finite")
        if not torch.isfinite(auxiliary_targets).all():
            raise ValueError("auxiliary_targets must be finite")
        self.steps.append(
            LagrangianRolloutStep(
                observation=observation.detach().cpu().float().clone(),
                action=action.detach().cpu().float().clone(),
                pretransform_action=pretransform_action.detach().cpu().float().clone(),
                old_log_probability=checked["old_log_probability"],
                reward=checked["reward"],
                reward_value=checked["reward_value"],
                next_reward_value=checked["next_reward_value"],
                cost=checked_cost,
                cost_value=checked["cost_value"],
                next_cost_value=checked["next_cost_value"],
                boundary=boundary,
                auxiliary_targets=auxiliary_targets.detach().cpu().float().clone(),
            )
        )
        self._advantages_ready = False

    def compute_returns_and_advantages(
        self,
        *,
        reward_gamma: float,
        cost_gamma: float = 1.0,
        gae_lambda: float = 0.95,
    ) -> None:
        if not self.steps:
            raise ValueError("cannot compute GAE for an empty rollout")
        reward_discount = _discount("reward_gamma", reward_gamma)
        cost_discount = _discount("cost_gamma", cost_gamma)
        trace_decay = _discount("gae_lambda", gae_lambda)
        self._compute_stream(
            signal_name="reward",
            value_name="reward_value",
            next_value_name="next_reward_value",
            advantage_name="reward_advantage",
            return_name="reward_return",
            gamma=reward_discount,
            gae_lambda=trace_decay,
        )
        self._compute_stream(
            signal_name="cost",
            value_name="cost_value",
            next_value_name="next_cost_value",
            advantage_name="cost_advantage",
            return_name="cost_return",
            gamma=cost_discount,
            gae_lambda=trace_decay,
        )
        self._advantages_ready = True

    def _compute_stream(
        self,
        *,
        signal_name: str,
        value_name: str,
        next_value_name: str,
        advantage_name: str,
        return_name: str,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        advantage = 0.0
        for step in reversed(self.steps):
            boundary = step.boundary
            if boundary.episode_done:
                bootstrap_value = 0.0
                continuation = 0.0
            elif boundary.collector_only:
                bootstrap_value = getattr(step, next_value_name)
                continuation = 0.0
            else:
                bootstrap_value = getattr(step, next_value_name)
                continuation = 1.0
            delta = (
                getattr(step, signal_name)
                + gamma * bootstrap_value
                - getattr(step, value_name)
            )
            advantage = delta + gamma * gae_lambda * continuation * advantage
            setattr(step, advantage_name, advantage)
            setattr(step, return_name, advantage + getattr(step, value_name))

    def batch(
        self,
        *,
        lagrange_multiplier: float | LagrangeMultiplier,
        normalize_advantages: bool = True,
    ) -> LagrangianRolloutBatch:
        if not self.steps:
            raise ValueError("cannot create a PPO batch from an empty rollout")
        if not self._advantages_ready:
            raise RuntimeError("compute_returns_and_advantages must run before batch")
        multiplier = _multiplier_value(lagrange_multiplier)
        reward_advantages = torch.tensor(
            [step.reward_advantage for step in self.steps], dtype=torch.float32
        )
        cost_advantages = torch.tensor(
            [step.cost_advantage for step in self.steps], dtype=torch.float32
        )
        raw_policy_advantages = reward_advantages - multiplier * cost_advantages
        advantages = raw_policy_advantages.clone()
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        return LagrangianRolloutBatch(
            observations=torch.stack([step.observation for step in self.steps]),
            pretransform_actions=torch.stack(
                [step.pretransform_action for step in self.steps]
            ),
            old_log_probabilities=torch.tensor(
                [step.old_log_probability for step in self.steps], dtype=torch.float32
            ),
            advantages=advantages,
            raw_policy_advantages=raw_policy_advantages,
            reward_returns=torch.tensor(
                [step.reward_return for step in self.steps], dtype=torch.float32
            ),
            cost_returns=torch.tensor(
                [step.cost_return for step in self.steps], dtype=torch.float32
            ),
            auxiliary_targets=torch.stack(
                [step.auxiliary_targets for step in self.steps]
            ),
            lagrange_multiplier=multiplier,
        )


class LagrangianPPOUpdater:
    """PPO update with reward/cost critics and a combined actor advantage."""

    def __init__(
        self, model: LagrangianActorCritic, config: PPOConfig
    ) -> None:
        if not isinstance(model, LagrangianActorCritic):
            raise TypeError("model must be a LagrangianActorCritic")
        self.model = model
        self.config = config
        # Reuse the repository PPO optimizer construction, including its
        # shared/policy-mean learning-rate groups; only the minibatch objective
        # is extended here with the cost-value loss.
        self._policy_updater = PPOUpdater(model.policy, config)
        self.optimizer = self._policy_updater.optimizer
        self.optimizer.add_param_group(
            {
                "params": model.cost_value_head.parameters(),
                "lr": config.learning_rate,
                "group_name": "cost_value",
            }
        )

    def set_learning_rates(
        self,
        learning_rate: float,
        policy_mean_learning_rate: float | None = None,
    ) -> None:
        self._policy_updater.set_learning_rates(
            learning_rate, policy_mean_learning_rate
        )
        for group in self.optimizer.param_groups:
            if group.get("group_name") == "cost_value":
                group["lr"] = learning_rate

    def update(
        self,
        storage: LagrangianRolloutStorage,
        *,
        lagrange_multiplier: float | LagrangeMultiplier,
    ) -> dict[str, float]:
        batch = storage.batch(lagrange_multiplier=lagrange_multiplier)
        count = len(storage)
        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "cost_value_loss": 0.0,
            "auxiliary_loss": 0.0,
            "entropy": 0.0,
            "grad_norm": 0.0,
        }
        updates = 0
        for _ in range(self.config.epochs):
            permutation = torch.randperm(count)
            for indices in permutation.split(self.config.minibatch_size):
                observations = batch.observations[indices]
                output = self.model(observations)
                new_log_probabilities = self.model.log_probability_from_pretransform(
                    batch.pretransform_actions[indices],
                    output.action_mean,
                    output.action_log_std,
                )
                ratio = (
                    new_log_probabilities
                    - batch.old_log_probabilities[indices]
                ).exp()
                advantages = batch.advantages[indices]
                unclipped = ratio * advantages
                clipped = ratio.clamp(
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(
                    output.value, batch.reward_returns[indices]
                )
                cost_value_loss = F.mse_loss(
                    output.cost_value, batch.cost_returns[indices]
                )
                auxiliary_loss = F.mse_loss(
                    output.auxiliary_predictions,
                    batch.auxiliary_targets[indices],
                )
                entropy = self.model.entropy(output.action_log_std).mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    + self.config.value_coefficient * cost_value_loss
                    + self.config.auxiliary_coefficient * auxiliary_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                metrics = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "cost_value_loss": cost_value_loss,
                    "auxiliary_loss": auxiliary_loss,
                    "entropy": entropy,
                    "grad_norm": grad_norm,
                }
                for name, value in metrics.items():
                    totals[name] += float(value.detach())
                updates += 1
        if updates == 0:
            raise ValueError("PPO config must produce at least one minibatch update")
        averaged = {name: value / updates for name, value in totals.items()}
        averaged["lagrange_multiplier"] = batch.lagrange_multiplier
        averaged["raw_policy_advantage_abs_mean"] = float(
            batch.raw_policy_advantages.abs().mean()
        )
        return averaged


__all__ = [
    "LagrangianActorCritic",
    "LagrangianPolicyOutput",
    "LagrangianPPOUpdater",
    "LagrangianRolloutBatch",
    "LagrangianRolloutStep",
    "LagrangianRolloutStorage",
]
