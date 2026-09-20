"""Minimal CPU PPO update for the visual actor-critic."""

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from haic_agent.networks import VisualActorCritic
from training.rollout import RolloutStorage


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    auxiliary_coefficient: float = 0.1
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 32


class PPOUpdater:
    """Apply clipped PPO policy, value, entropy, and visual auxiliary losses."""

    def __init__(self, model: VisualActorCritic, config: PPOConfig) -> None:
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def update(self, storage: RolloutStorage) -> Mapping[str, float]:
        batch = storage.batch(normalize_advantages=True)
        count = len(storage)
        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "auxiliary_loss": 0.0, "entropy": 0.0, "grad_norm": 0.0}
        updates = 0
        for _ in range(self.config.epochs):
            permutation = torch.randperm(count)
            for indices in permutation.split(self.config.minibatch_size):
                observations = batch.observations[indices]
                actions = batch.actions[indices]
                output = self.model(observations)
                new_log_probabilities = self.model.log_probability(
                    actions, output.action_mean, output.action_log_std
                )
                ratio = (new_log_probabilities - batch.old_log_probabilities[indices]).exp()
                advantages = batch.advantages[indices]
                unclipped = ratio * advantages
                clipped = ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(output.value, batch.returns[indices])
                auxiliary_loss = nn.functional.mse_loss(
                    output.auxiliary_predictions, batch.auxiliary_targets[indices]
                )
                entropy = self.model.entropy(output.action_log_std).mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    + self.config.auxiliary_coefficient * auxiliary_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                metrics = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "auxiliary_loss": auxiliary_loss,
                    "entropy": entropy,
                    "grad_norm": grad_norm,
                }
                for name, value in metrics.items():
                    totals[name] += float(value.detach())
                updates += 1
        return {name: total / updates for name, total in totals.items()}
