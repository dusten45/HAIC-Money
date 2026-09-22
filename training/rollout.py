"""PPO rollout storage at the four-tick decision cadence."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RolloutStep:
    observation: Tensor
    action: Tensor
    pretransform_action: Tensor
    log_probability: float
    value: float
    next_value: float
    reward: float
    terminated: bool
    truncated: bool
    auxiliary_targets: Tensor
    advantage: float = 0.0
    return_: float = 0.0


@dataclass(frozen=True)
class RolloutBatch:
    observations: Tensor
    actions: Tensor
    pretransform_actions: Tensor
    old_log_probabilities: Tensor
    advantages: Tensor
    returns: Tensor
    auxiliary_targets: Tensor


class RolloutStorage:
    """Store policy decisions and compute episode-safe generalized advantages."""

    def __init__(self) -> None:
        self.steps: list[RolloutStep] = []

    def __len__(self) -> int:
        return len(self.steps)

    def add(
        self,
        *,
        observation: Tensor,
        action: Tensor,
        pretransform_action: Tensor,
        log_probability: float,
        value: float,
        next_value: float,
        reward: float,
        terminated: bool,
        truncated: bool,
        auxiliary_targets: Tensor,
    ) -> None:
        self.steps.append(
            RolloutStep(
                observation=observation.detach().cpu().float().clone(),
                action=action.detach().cpu().float().clone(),
                pretransform_action=pretransform_action.detach().cpu().float().clone(),
                log_probability=float(log_probability),
                value=float(value),
                next_value=float(next_value),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                auxiliary_targets=auxiliary_targets.detach().cpu().float().clone(),
            )
        )

    def compute_returns_and_advantages(self, *, gamma: float, gae_lambda: float) -> None:
        """Compute GAE without carrying rewards across terminal or truncated episodes.

        A time-limit truncation uses its stored next-state value once, then
        stops the recursive GAE term before the next episode's first step.
        """
        advantage = 0.0
        for step in reversed(self.steps):
            if step.terminated:
                bootstrap_value = 0.0
                continuation = 0.0
            elif step.truncated:
                bootstrap_value = step.next_value
                continuation = 0.0
            else:
                bootstrap_value = step.next_value
                continuation = 1.0
            delta = step.reward + gamma * bootstrap_value - step.value
            advantage = delta + gamma * gae_lambda * continuation * advantage
            step.advantage = advantage
            step.return_ = advantage + step.value

    def batch(self, *, normalize_advantages: bool = True) -> RolloutBatch:
        if not self.steps:
            raise ValueError("cannot create a PPO batch from an empty rollout")
        advantages = torch.tensor([step.advantage for step in self.steps], dtype=torch.float32)
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return RolloutBatch(
            observations=torch.stack([step.observation for step in self.steps]),
            actions=torch.stack([step.action for step in self.steps]),
            pretransform_actions=torch.stack([step.pretransform_action for step in self.steps]),
            old_log_probabilities=torch.tensor(
                [step.log_probability for step in self.steps], dtype=torch.float32
            ),
            advantages=advantages,
            returns=torch.tensor([step.return_ for step in self.steps], dtype=torch.float32),
            auxiliary_targets=torch.stack([step.auxiliary_targets for step in self.steps]),
        )
