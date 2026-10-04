"""Inference-safe learned latent dynamics for short-horizon action planning."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from haic_agent.networks import ACTION_SIZE, LATENT_SIZE


@dataclass(frozen=True)
class DynamicsEnsembleOutput:
    """Per-member predictions with member dimension first.

    Every tensor begins with ``num_members`` and preserves all input leading
    dimensions after it.  The residual is added to the input latent by
    :meth:`LatentDynamicsEnsemble.predict`.
    """

    next_latent_residual: Tensor
    progress_delta: Tensor
    reward: Tensor
    collision_logits: Tensor
    off_track_logits: Tensor


@dataclass(frozen=True)
class DynamicsPrediction:
    """Mean prediction and one ensemble-disagreement score per input item."""

    next_latent: Tensor
    progress_delta: Tensor
    reward: Tensor
    collision_probability: Tensor
    off_track_probability: Tensor
    uncertainty: Tensor


class _DynamicsMember(nn.Module):
    """One independently initialized MLP in the bootstrap-style ensemble."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(LATENT_SIZE + ACTION_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.next_latent_residual = nn.Linear(hidden_size, LATENT_SIZE)
        self.progress_delta = nn.Linear(hidden_size, 1)
        self.reward = nn.Linear(hidden_size, 1)
        self.collision_logits = nn.Linear(hidden_size, 1)
        self.off_track_logits = nn.Linear(hidden_size, 1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        hidden = self.trunk(features)
        return (
            self.next_latent_residual(hidden),
            self.progress_delta(hidden).squeeze(-1),
            self.reward(hidden).squeeze(-1),
            self.collision_logits(hidden).squeeze(-1),
            self.off_track_logits(hidden).squeeze(-1),
        )


class LatentDynamicsEnsemble(nn.Module):
    """Predict one decision-step of latent dynamics without simulator access."""

    def __init__(self, *, num_members: int = 5, hidden_size: int = 192) -> None:
        super().__init__()
        if num_members < 2:
            raise ValueError("num_members must be at least 2 for epistemic uncertainty")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.members = nn.ModuleList(_DynamicsMember(int(hidden_size)) for _ in range(int(num_members)))

    @staticmethod
    def _validate_inputs(latent: Tensor, action: Tensor) -> None:
        if latent.ndim < 2 or latent.shape[-1] != LATENT_SIZE:
            raise ValueError("latent must have leading batch dimensions followed by width 128")
        if action.ndim != latent.ndim or action.shape[:-1] != latent.shape[:-1] or action.shape[-1] != ACTION_SIZE:
            raise ValueError("action must match latent leading dimensions and have width 3")
        if not torch.isfinite(latent).all() or not torch.isfinite(action).all():
            raise ValueError("latent and action must be finite")

    def forward(self, latent: Tensor, action: Tensor) -> DynamicsEnsembleOutput:
        """Return each member's next-decision predictions.

        ``latent`` and ``action`` may be ``(B, ...)`` or ``(B, horizon, ...)``.
        The CEM planner can therefore evaluate all candidates and every horizon
        step in one call while retaining member-level uncertainty.
        """
        self._validate_inputs(latent, action)
        leading_shape = latent.shape[:-1]
        features = torch.cat((latent, action), dim=-1).reshape(-1, LATENT_SIZE + ACTION_SIZE)
        member_outputs = [member(features) for member in self.members]

        def reshape_and_stack(index: int) -> Tensor:
            return torch.stack(
                [values[index].reshape(*leading_shape, *values[index].shape[1:]) for values in member_outputs], dim=0
            )

        return DynamicsEnsembleOutput(
            next_latent_residual=reshape_and_stack(0),
            progress_delta=reshape_and_stack(1),
            reward=reshape_and_stack(2),
            collision_logits=reshape_and_stack(3),
            off_track_logits=reshape_and_stack(4),
        )

    def predict(self, latent: Tensor, action: Tensor) -> DynamicsPrediction:
        """Aggregate members while retaining their disagreement as uncertainty."""
        output = self(latent, action)
        member_next_latent = latent.unsqueeze(0) + output.next_latent_residual
        collision_probabilities = torch.sigmoid(output.collision_logits)
        off_track_probabilities = torch.sigmoid(output.off_track_logits)
        latent_spread = member_next_latent.std(dim=0, unbiased=False).pow(2).mean(dim=-1).sqrt()
        progress_spread = output.progress_delta.std(dim=0, unbiased=False)
        reward_spread = output.reward.std(dim=0, unbiased=False)
        collision_spread = collision_probabilities.std(dim=0, unbiased=False)
        off_track_spread = off_track_probabilities.std(dim=0, unbiased=False)
        uncertainty = latent_spread + progress_spread + reward_spread + collision_spread + off_track_spread
        return DynamicsPrediction(
            next_latent=member_next_latent.mean(dim=0),
            progress_delta=output.progress_delta.mean(dim=0),
            reward=output.reward.mean(dim=0),
            collision_probability=collision_probabilities.mean(dim=0),
            off_track_probability=off_track_probabilities.mean(dim=0),
            uncertainty=uncertainty,
        )
