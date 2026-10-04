"""Isolated Q-bootstrap weight for the original TD-MPC2 planner."""

import math
from numbers import Real
from typing import Any, cast

import torch

from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner


class QWeightedPlanner(TDMPC2Planner):
    """Retain MPPI and sampled-Q RNG while weighting only terminal bootstrap Q."""

    def __init__(self, model: object, config: PlannerConfig, *, q_weight: float):
        if isinstance(q_weight, bool) or not isinstance(q_weight, Real) or not 0 <= q_weight <= 1 or not math.isfinite(q_weight):
            raise ValueError("q_weight must be finite and in [0, 1]")
        super().__init__(model, config)
        self.q_weight = q_weight

    def _estimate_value(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Score trajectories, retaining reward on a terminal transition."""
        model = cast(Any, self.model)
        cfg = self.config
        batch = cfg.num_samples
        total = torch.zeros((batch, 1), device=z.device, dtype=z.dtype)
        alive = torch.ones((batch, 1), device=z.device, dtype=torch.bool)
        valid = torch.isfinite(actions).all(dim=0).all(dim=-1).unsqueeze(-1)
        discount = 1.0

        for t in range(cfg.horizon):
            if not alive.any():
                break
            reward = self._reward(model.reward(z, actions[t], None), batch)
            next_z = self._tensor(model.next(z, actions[t], None), z.shape, "next")
            finite = torch.isfinite(reward) & torch.isfinite(next_z).all(-1, keepdim=True)
            valid = valid & (~alive | finite)
            total = total + discount * torch.where(alive & finite, reward, 0)
            z = torch.where(torch.isfinite(next_z), next_z, 0)
            discount *= cfg.discount

            if cfg.episodic:
                prob = self._tensor(model.termination(z, None), (batch, 1), "termination")
                finite_prob = torch.isfinite(prob) & (prob >= 0) & (prob <= 1)
                valid = valid & (~alive | finite_prob)
                alive = alive & ~(prob > 0.5)
            if discount == 0:
                break

        if discount != 0 and alive.any():
            action, _ = model.pi(z, None)
            action = self._tensor(action, (batch, cfg.action_dim), "bootstrap policy")
            finite_action = torch.isfinite(action).all(-1, keepdim=True)
            valid = valid & (~alive | finite_action)
            action = torch.where(torch.isfinite(action), action, 0).clamp(-1, 1)
            q = self._tensor(model.Q(z, action, None, return_type="avg"), (batch, 1), "Q")
            finite_q = torch.isfinite(q)
            valid = valid & (~alive | finite_q)
            if self.q_weight != 1:
                q = self.q_weight * q
            total = total + discount * torch.where(alive & finite_q, q, 0)

        return torch.where(valid & torch.isfinite(total), total, -torch.inf)
