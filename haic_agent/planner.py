"""Deadline-bounded CEM planning over learned latent dynamics only."""

from dataclasses import dataclass
import time
from typing import Callable

import torch
from torch import Tensor

from haic_agent.dynamics import LatentDynamicsEnsemble
from haic_agent.networks import ACTION_SIZE, LATENT_SIZE


@dataclass(frozen=True)
class PlanResult:
    """A complete valid CEM candidate ready for one environment action."""

    action: Tensor
    unconstrained_sequence: Tensor
    score: float


class CEMPlanner:
    """Anytime cross-entropy planning using policy prior and dynamics risk.

    The stored sequence stays in unconstrained action coordinates.  This makes
    a shifted warm start consistent with the policy's Normal prior, while only
    bounded actions ever leave the planner.
    """

    def __init__(
        self,
        *,
        horizon: int = 8,
        population: int = 64,
        iterations: int = 4,
        candidate_batch_size: int = 16,
        elite_fraction: float = 0.2,
        progress_weight: float = 1.0,
        reward_weight: float = 0.5,
        time_cost: float = 0.02,
        collision_cost: float = 3.0,
        off_track_cost: float = 3.0,
        uncertainty_cost: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if horizon < 1 or population < 2 or iterations < 1 or candidate_batch_size < 1:
            raise ValueError("horizon, population, iterations, and candidate_batch_size must be positive")
        if not 0.0 < elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")
        self.horizon = int(horizon)
        self.population = int(population)
        self.iterations = int(iterations)
        self.candidate_batch_size = int(candidate_batch_size)
        self.elite_count = max(1, int(self.population * elite_fraction))
        self.progress_weight = float(progress_weight)
        self.reward_weight = float(reward_weight)
        self.time_cost = float(time_cost)
        self.collision_cost = float(collision_cost)
        self.off_track_cost = float(off_track_cost)
        self.uncertainty_cost = float(uncertainty_cost)
        self.clock = clock
        self._cached_unconstrained: Tensor | None = None

    @staticmethod
    def _bounded(unconstrained: Tensor) -> Tensor:
        return torch.cat((torch.tanh(unconstrained[..., :1]), torch.sigmoid(unconstrained[..., 1:])), dim=-1)

    @staticmethod
    def _valid_actions(actions: Tensor) -> bool:
        return bool(
            actions.shape[-1] == ACTION_SIZE
            and torch.isfinite(actions).all()
            and torch.all(actions[..., :1] >= -1.0)
            and torch.all(actions[..., :1] <= 1.0)
            and torch.all(actions[..., 1:] >= 0.0)
            and torch.all(actions[..., 1:] <= 1.0)
        )

    def reset(self) -> None:
        """Forget a previous episode's action sequence."""
        self._cached_unconstrained = None

    def warm_start(self) -> Tensor | None:
        """Shift the prior optimum one decision and repeat its final action."""
        if self._cached_unconstrained is None:
            return None
        cached = self._cached_unconstrained
        if cached.shape != (self.horizon, ACTION_SIZE) or not torch.isfinite(cached).all():
            return None
        return torch.cat((cached[1:], cached[-1:]), dim=0)

    @staticmethod
    def _valid_inputs(latent: Tensor, policy_mean: Tensor, policy_log_std: Tensor) -> bool:
        return bool(
            latent.shape == (1, LATENT_SIZE)
            and policy_mean.shape == (1, ACTION_SIZE)
            and policy_log_std.shape == (1, ACTION_SIZE)
            and torch.isfinite(latent).all()
            and torch.isfinite(policy_mean).all()
            and torch.isfinite(policy_log_std).all()
        )

    def score_sequences(
        self, latent: Tensor, actions: Tensor, dynamics: LatentDynamicsEnsemble, *, deadline: float
    ) -> Tensor | None:
        """Score candidates by learned forward progress, risk and uncertainty.

        Rollout feeds the ensemble mean next latent into the next horizon
        prediction, which keeps each candidate autoregressive.
        """
        if actions.ndim != 3 or actions.shape[1:] != (self.horizon, ACTION_SIZE):
            raise ValueError("actions must have shape (population, horizon, 3)")
        if not self._valid_actions(actions) or self.clock() >= deadline:
            return None
        current_latent = latent.expand(actions.shape[0], -1)
        scores = torch.zeros(actions.shape[0], dtype=actions.dtype, device=actions.device)
        try:
            for step in range(self.horizon):
                if self.clock() >= deadline:
                    return None
                prediction = dynamics.predict(current_latent, actions[:, step, :])
                values = (
                    prediction.next_latent,
                    prediction.progress_delta,
                    prediction.reward,
                    prediction.collision_probability,
                    prediction.off_track_probability,
                    prediction.uncertainty,
                )
                if not all(torch.isfinite(value).all() for value in values):
                    return None
                if self.clock() >= deadline:
                    return None
                scores = scores + (
                    self.progress_weight * prediction.progress_delta
                    + self.reward_weight * prediction.reward
                    - self.time_cost
                    - self.collision_cost * prediction.collision_probability
                    - self.off_track_cost * prediction.off_track_probability
                    - self.uncertainty_cost * prediction.uncertainty
                )
                current_latent = prediction.next_latent
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return None
        return scores if torch.isfinite(scores).all() else None

    def _candidate_scores(
        self, latent: Tensor, candidates: Tensor, dynamics: LatentDynamicsEnsemble, *, deadline: float
    ) -> tuple[Tensor, Tensor] | None:
        valid_candidates: list[Tensor] = []
        valid_scores: list[Tensor] = []
        for start in range(0, candidates.shape[0], self.candidate_batch_size):
            if self.clock() >= deadline:
                break
            chunk = candidates[start : start + self.candidate_batch_size]
            score = self.score_sequences(latent, self._bounded(chunk), dynamics, deadline=deadline)
            if score is None:
                if self.clock() >= deadline:
                    break
                continue
            if self.clock() >= deadline:
                break
            valid_candidates.append(chunk)
            valid_scores.append(score)
        if not valid_scores:
            return None
        combined = torch.cat(valid_candidates, dim=0), torch.cat(valid_scores, dim=0)
        return None if self.clock() >= deadline else combined

    def plan(
        self,
        latent: Tensor,
        policy_mean: Tensor,
        policy_log_std: Tensor,
        dynamics: LatentDynamicsEnsemble,
        *,
        deadline: float,
    ) -> PlanResult | None:
        """Improve the actor's Normal prior until ``deadline`` and return best."""
        if self.clock() >= deadline or not self._valid_inputs(latent, policy_mean, policy_log_std):
            return None
        mean = policy_mean.detach().reshape(1, 1, ACTION_SIZE).expand(self.population, self.horizon, -1).clone()
        log_std = policy_log_std.detach().clamp(-5.0, 2.0).reshape(1, 1, ACTION_SIZE)
        std = log_std.exp().expand_as(mean)
        warm = self.warm_start()
        best_sequence: Tensor | None = None
        best_score: Tensor | None = None
        for _ in range(self.iterations):
            if self.clock() >= deadline:
                break
            candidates = mean + torch.randn_like(mean) * std
            if warm is not None:
                candidates[0] = warm.to(dtype=candidates.dtype, device=candidates.device)
            scored = self._candidate_scores(latent, candidates, dynamics, deadline=deadline)
            if scored is None:
                break
            if self.clock() >= deadline:
                break
            valid_candidates, scores = scored
            top_score, top_index = scores.max(dim=0)
            if self.clock() >= deadline:
                break
            if best_score is None or top_score > best_score:
                best_score = top_score.detach()
                best_sequence = valid_candidates[int(top_index)].detach().clone()
            elite_count = min(self.elite_count, valid_candidates.shape[0])
            elite_indices = scores.topk(elite_count).indices
            if self.clock() >= deadline:
                break
            elites = valid_candidates[elite_indices]
            mean = elites.mean(dim=0, keepdim=True).expand(self.population, -1, -1).clone()
            std = elites.std(dim=0, unbiased=False, keepdim=True).clamp_min(0.05).expand_as(mean).clone()
            if self.clock() >= deadline:
                break
            warm = None
        if self.clock() >= deadline or best_sequence is None or best_score is None:
            return None
        action = self._bounded(best_sequence[:1]).squeeze(0)
        if not self._valid_actions(action):
            return None
        if self.clock() >= deadline:
            return None
        self._cached_unconstrained = best_sequence.cpu()
        return PlanResult(action=action.cpu(), unconstrained_sequence=best_sequence.cpu(), score=float(best_score.cpu()))
