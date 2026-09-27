"""Single-task TD-MPC2 MPPI planner.

Follows nicklashansen/tdmpc2 ``tdmpc2/tdmpc2.py`` at
e9f59321933cbc8e11a002b842adc7d4ffae8ff1. The official single-task
configuration fixes the planning horizon at 3; it does not anneal the horizon.
The planner is independent of the model implementation and environment.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PlannerConfig:
    """Official single-task planning defaults, with explicit action/discount inputs.

    ``discount`` is the scalar produced by the caller's episode-length rule; zero
    is permitted. ``num_bins`` and ``vmin/vmax`` must match the model's reward head.
    Set ``episodic=True`` only when the model has a termination head.
    """

    action_dim: int
    discount: float
    horizon: int = 3
    iterations: int = 6
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5
    episodic: bool = False
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0

    def __post_init__(self) -> None:
        if self.action_dim < 1 or self.horizon < 1 or self.iterations < 1:
            raise ValueError("action_dim, horizon and iterations must be positive")
        if not 1 <= self.num_elites <= self.num_samples:
            raise ValueError("num_elites must be between 1 and num_samples")
        if not 0 <= self.num_pi_trajs <= self.num_samples:
            raise ValueError("num_pi_trajs must be between 0 and num_samples")
        if not 0 < self.min_std <= self.max_std:
            raise ValueError("std bounds must be positive and ordered")
        if not 0 <= self.discount <= 1 or not 0 <= self.temperature:
            raise ValueError("discount and temperature must be finite and nonnegative")
        if not torch.isfinite(torch.tensor((self.discount, self.temperature, self.min_std, self.max_std))).all():
            raise ValueError("planner scalars must be finite")
        if self.num_bins < 0 or (self.num_bins > 1 and not self.vmin < self.vmax):
            raise ValueError("invalid reward bins or value range")


class TDMPC2Planner:
    """Plan one action from a single, batched observation.

    Required duck-typed model methods (``task=None`` for this single-task lane):

    * ``encode(obs, None) -> z[1, D]``
    * ``pi(z, None) -> (sampled_action[B, A], info)``
    * ``next(z, action, None) -> next_z[B, D]``
    * ``reward(z, action, None) -> logits[B, num_bins]``; if num_bins=0,
      reward is scalar [B, 1], or symlog scalar [B, 1] if num_bins=1
    * ``Q(z, action, None, return_type='avg') -> scalar[B, 1]``
    * if episodic, ``termination(next_z, None) -> probability[B, 1]``

    The stochastic prior must return sampled, tanh-squashed actions, not its
    mean. ``plan`` returns an unbatched action on the model's device; ``t0=True``
    suppresses the shifted previous-mean warm start at an episode boundary.
    """

    def __init__(self, model: object, config: PlannerConfig):
        self.model = model
        self.config = config
        self.prev_mean = torch.zeros(config.horizon, config.action_dim)

    def reset(self) -> None:
        """Clear the warm start without invoking the model."""
        self.prev_mean = torch.zeros_like(self.prev_mean)

    @staticmethod
    def _tensor(value: torch.Tensor, shape: tuple[int, ...], name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.shape != shape or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating tensor with shape {shape}")
        return value

    def _reward(self, logits: torch.Tensor, batch: int) -> torch.Tensor:
        cfg = self.config
        bins = max(cfg.num_bins, 1)
        logits = self._tensor(logits, (batch, bins), "reward")
        if cfg.num_bins == 0:
            return logits
        if cfg.num_bins == 1:
            return logits.sign() * torch.expm1(logits.abs())
        bounds = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, dtype=logits.dtype, device=logits.device)
        decoded = (F.softmax(logits, dim=-1) * bounds).sum(-1, keepdim=True)
        return decoded.sign() * torch.expm1(decoded.abs())

    def _estimate_value(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Score trajectories, retaining reward on a terminal transition."""
        cfg = self.config
        batch = cfg.num_samples
        total = torch.zeros((batch, 1), device=z.device, dtype=z.dtype)
        alive = torch.ones((batch, 1), device=z.device, dtype=torch.bool)
        valid = torch.isfinite(actions).all(dim=0).all(dim=-1).unsqueeze(-1)
        discount = 1.0

        for t in range(cfg.horizon):
            if not alive.any():
                break
            reward = self._reward(self.model.reward(z, actions[t], None), batch)
            next_z = self._tensor(self.model.next(z, actions[t], None), z.shape, "next")
            finite = torch.isfinite(reward) & torch.isfinite(next_z).all(-1, keepdim=True)
            valid = valid & (~alive | finite)
            total = total + discount * torch.where(alive & finite, reward, 0)
            z = torch.where(torch.isfinite(next_z), next_z, 0)
            discount *= cfg.discount

            if cfg.episodic:
                prob = self._tensor(self.model.termination(z, None), (batch, 1), "termination")
                finite_prob = torch.isfinite(prob) & (prob >= 0) & (prob <= 1)
                valid = valid & (~alive | finite_prob)
                alive = alive & ~(prob > 0.5)
            if discount == 0:
                break

        if discount != 0 and alive.any():
            action, _ = self.model.pi(z, None)
            action = self._tensor(action, (batch, cfg.action_dim), "bootstrap policy")
            finite_action = torch.isfinite(action).all(-1, keepdim=True)
            valid = valid & (~alive | finite_action)
            action = torch.where(torch.isfinite(action), action, 0).clamp(-1, 1)
            q = self._tensor(self.model.Q(z, action, None, return_type="avg"), (batch, 1), "Q")
            finite_q = torch.isfinite(q)
            valid = valid & (~alive | finite_q)
            total = total + discount * torch.where(alive & finite_q, q, 0)

        return torch.where(valid & torch.isfinite(total), total, -torch.inf)

    @torch.no_grad()
    def plan(self, obs: torch.Tensor, *, t0: bool = False, eval_mode: bool = False) -> torch.Tensor:
        """Sample Gaussian and policy-prior trajectories, refine MPPI, act."""
        cfg = self.config
        z = self.model.encode(obs, None)
        if not isinstance(z, torch.Tensor) or z.ndim != 2 or z.shape[0] != 1 or not z.is_floating_point():
            raise ValueError("encode must return floating latent [1, D]")
        if not torch.isfinite(z).all():
            raise FloatingPointError("nonfinite encoded state")

        actions = torch.empty((cfg.horizon, cfg.num_samples, cfg.action_dim), device=z.device, dtype=z.dtype)
        if cfg.num_pi_trajs:
            prior_z = z.repeat(cfg.num_pi_trajs, 1)
            for t in range(cfg.horizon):
                prior_action, _ = self.model.pi(prior_z, None)
                prior_action = self._tensor(prior_action, (cfg.num_pi_trajs, cfg.action_dim), "policy prior")
                if not torch.isfinite(prior_action).all():
                    raise FloatingPointError("nonfinite policy prior action")
                actions[t, :cfg.num_pi_trajs] = prior_action.clamp(-1, 1)
                if t < cfg.horizon - 1:
                    prior_z = self._tensor(self.model.next(prior_z, actions[t, :cfg.num_pi_trajs], None), prior_z.shape, "prior next")
                    if not torch.isfinite(prior_z).all():
                        raise FloatingPointError("nonfinite policy prior state")

        mean = torch.zeros((cfg.horizon, cfg.action_dim), device=z.device, dtype=z.dtype)
        std = torch.full_like(mean, cfg.max_std)
        if not t0:
            mean[:-1].copy_(self.prev_mean.to(device=z.device, dtype=z.dtype)[1:])

        for _ in range(cfg.iterations):
            noise = torch.randn((cfg.horizon, cfg.num_samples - cfg.num_pi_trajs, cfg.action_dim), device=z.device, dtype=z.dtype)
            actions[:, cfg.num_pi_trajs:] = (mean[:, None] + std[:, None] * noise).clamp(-1, 1)
            values = self._estimate_value(z.repeat(cfg.num_samples, 1), actions).squeeze(-1)
            if torch.isfinite(values).sum() < cfg.num_elites:
                raise FloatingPointError("insufficient finite trajectories for MPPI elites")
            elite_value, elite_idx = values.topk(cfg.num_elites)
            elite_actions = actions[:, elite_idx]
            score = torch.exp(cfg.temperature * (elite_value - elite_value.max()))
            score = score / score.sum()
            mean = (score[None, :, None] * elite_actions).sum(dim=1) / (score.sum() + 1e-9)
            std = ((score[None, :, None] * (elite_actions - mean[:, None]).square()).sum(dim=1) / (score.sum() + 1e-9)).sqrt()
            std = std.clamp(cfg.min_std, cfg.max_std)
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not torch.isfinite(score).all():
                raise FloatingPointError("nonfinite MPPI distribution")

        # Same categorical elite draw as the official Gumbel-Softmax sampler.
        gumbel = -torch.empty_like(score).exponential_().log()
        idx = (score.log() + gumbel).argmax()
        action = elite_actions[0, idx]
        if not eval_mode:
            action = action + std[0] * torch.randn(cfg.action_dim, device=z.device, dtype=z.dtype)
        self.prev_mean = mean.detach().clone()
        return action.clamp(-1, 1)
