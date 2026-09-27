"""Single-task, H-step TD-MPC2 learner.

Training equations follow nicklashansen/tdmpc2 at
e9f59321933cbc8e11a002b842adc7d4ffae8ff1, tdmpc2/tdmpc2.py.
HAIC finishes are reported as truncations; the replay's semantic terminal flag
includes finishes, while time-limit truncations still bootstrap.
"""

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .model import soft_ce


@dataclass(frozen=True)
class TDMPC2LearnerConfig:
    episode_length: int
    horizon: int = 3
    batch_size: int = 256
    lr: float = 3e-4
    enc_lr_scale: float = 0.3
    grad_clip_norm: float = 20.0
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    termination_coef: float = 1.0
    rho: float = 0.5
    entropy_coef: float = 1e-4
    discount_denom: float = 5.0
    discount_min: float = 0.95
    discount_max: float = 0.995


class TDMPC2Learner(nn.Module):
    """Learns from episode-contained replay samples of H transitions.

    ``update`` accepts a mapping or a replay with ``sample(batch_size)``.
    Samples contain obs[H+1,B,...] and action/reward/terminated[H,B,...];
    reward and flags have a trailing singleton dimension. ``terminal`` is the
    semantic MDP ending (raw termination OR HAIC finish), not a time limit.
    The model owns its target Q ensemble and tau.
    """

    def __init__(self, model: nn.Module, cfg: TDMPC2LearnerConfig):
        super().__init__()
        if cfg.episode_length <= 0 or cfg.horizon <= 0 or cfg.batch_size <= 0:
            raise ValueError("episode_length, horizon and batch_size must be positive")
        if cfg.discount_denom <= 0 or not 0 <= cfg.rho <= 1:
            raise ValueError("discount_denom must be positive and rho in [0, 1]")
        if model.cfg.num_bins < 2:
            raise ValueError("5M distributional TD-MPC2 requires at least two value bins")

        self.model = model
        self.cfg = cfg
        frac = cfg.episode_length / cfg.discount_denom
        self.discount = min(max((frac - 1) / frac, cfg.discount_min), cfg.discount_max)
        self.register_buffer("q_scale", torch.ones(1, device=next(model.parameters()).device))

        groups = [
            {"params": model._encoder.parameters(), "lr": cfg.lr * cfg.enc_lr_scale},
            {"params": model._dynamics.parameters()},
            {"params": model._reward.parameters()},
            {"params": model._Qs.parameters()},
        ]
        if model.cfg.episodic:
            groups.append({"params": model._termination.parameters()})
        self.optim = torch.optim.Adam(groups, lr=cfg.lr)
        self.pi_optim = torch.optim.Adam(model._pi.parameters(), lr=cfg.lr, eps=1e-5)
        self.model.eval()

    @torch.no_grad()
    def _td_target(
        self, next_z: torch.Tensor, reward: torch.Tensor, terminal: torch.Tensor,
        truncated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Online policy action, minimum of two sampled *target* Q heads.

        ``terminal`` is the semantic terminal label, including HAIC finishes
        (which the simulator labels truncated). A time limit ends replay
        sampling but not the MDP and requires a real final observation here.
        """
        if next_z.shape[:-1] != reward.shape[:-1] or reward.shape != terminal.shape:
            raise ValueError("TD target requires matching [H,B,1] reward/terminal and next_z")
        if truncated is not None and truncated.shape != terminal.shape:
            raise ValueError("truncated must have the same shape as terminal")
        action, _ = self.model.pi(next_z, None)
        target_q = self.model.Q(next_z, action, None, return_type="min", target=True)
        if target_q.shape != reward.shape:
            raise ValueError("target Q shape differs from reward shape")
        return reward + self.discount * (1.0 - terminal) * target_q

    def update(self, batch: Mapping[str, torch.Tensor] | object) -> dict[str, float]:
        """Perform one joint model/critic update, then one policy and Q EMA update."""
        if not isinstance(batch, Mapping):
            batch = batch.sample(self.cfg.batch_size)
        if "weights" in batch or "priorities" in batch:
            raise ValueError("upstream uniform replay does not support importance weights")
        device = self.q_scale.device
        obs = torch.as_tensor(batch["obs"], device=device)
        action = torch.as_tensor(batch["action"], device=device, dtype=torch.float32)
        reward = torch.as_tensor(batch["reward"], device=device, dtype=torch.float32)
        terminated = torch.as_tensor(batch["terminated"], device=device, dtype=torch.float32)
        terminal = torch.as_tensor(batch.get("terminal", terminated), device=device, dtype=torch.float32)
        truncated = torch.as_tensor(batch.get("truncated", torch.zeros_like(terminated)), device=device)

        h, b = self.cfg.horizon, action.shape[1]
        if obs.shape[:2] != (h + 1, b) or action.shape[:2] != (h, b):
            raise ValueError("expected time-major H+1 observations and H actions")
        if reward.shape != (h, b, 1) or terminated.shape != reward.shape or terminal.shape != reward.shape or truncated.shape != reward.shape:
            raise ValueError("expected [H,B,1] rewards, terminal labels and truncations")
        if not torch.isfinite(reward).all() or not torch.isfinite(action).all():
            raise ValueError("nonfinite replay reward or action")
        if not torch.all((terminated == 0) | (terminated == 1)) or not torch.all((terminal == 0) | (terminal == 1)) or not torch.all((truncated == 0) | (truncated == 1)):
            raise ValueError("termination and truncation flags must be binary")
        if torch.any(terminated > terminal):
            raise ValueError("raw environment termination cannot be nonterminal")
        if "bootstrap_mask" in batch:
            mask = torch.as_tensor(batch["bootstrap_mask"], device=device)
            if mask.shape != reward.shape or not torch.equal(mask, 1 - terminal):
                raise ValueError("bootstrap mask must equal 1-semantic terminal (including finish)")
        if not self.model.cfg.episodic and bool(terminal.any()):
            raise ValueError("episodic=True is required when replay contains terminal transitions")

        # Targets use the online encoder and policy with stopped gradients and
        # the target Q ensemble; no separate target encoder or actor is used.
        self.model.eval()
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], None)
            targets = self._td_target(next_z, reward, terminal, truncated)

        self.model.train()
        try:
            z = self.model.encode(obs[0], None)
            zs = [z]
            consistency_loss = torch.zeros((), device=device)
            for t in range(h):
                z = self.model.next(z, action[t], None)
                consistency_loss = consistency_loss + self.cfg.rho ** t * F.mse_loss(z, next_z[t])
                zs.append(z)
            zs = torch.stack(zs)

            qs = self.model.Q(zs[:-1], action, None, return_type="all")
            reward_logits = self.model.reward(zs[:-1], action, None)
            if qs.shape[:3] != (self.model.cfg.num_q, h, b) or reward_logits.shape[:2] != (h, b):
                raise ValueError("expected Q logits [num_q,H,B,bins] and reward logits [H,B,bins]")
            reward_loss = torch.zeros((), device=device)
            value_loss = torch.zeros((), device=device)
            for t in range(h):
                weight = self.cfg.rho ** t
                reward_loss = reward_loss + weight * soft_ce(reward_logits[t], reward[t], self.model.cfg).mean()
                for q in qs[:, t]:
                    value_loss = value_loss + weight * soft_ce(q, targets[t], self.model.cfg).mean()
            consistency_loss = consistency_loss / h
            reward_loss = reward_loss / h
            value_loss = value_loss / (h * self.model.cfg.num_q)

            if self.model.cfg.episodic:
                termination_logits = self.model.termination(zs[1:], None, unnormalized=True)
                termination_loss = F.binary_cross_entropy_with_logits(termination_logits, terminal)
            else:
                termination_loss = torch.zeros((), device=device)
            total_loss = (
                self.cfg.consistency_coef * consistency_loss
                + self.cfg.reward_coef * reward_loss
                + self.cfg.value_coef * value_loss
                + self.cfg.termination_coef * termination_loss
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError("nonfinite TD-MPC2 model loss")
            self.optim.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("nonfinite TD-MPC2 model gradient")
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)

            # Frozen critic weights still transmit gradients through the sampled
            # action; the rollout latents are detached from the actor objective.
            pi_action, pi_info = self.model.pi(zs.detach(), None)
            pi_q = self.model.Q(zs.detach(), pi_action, None, return_type="avg", detach=True)
            with torch.no_grad():
                quantiles = torch.quantile(pi_q[0].detach().float().flatten(), torch.tensor([0.05, 0.95], device=device))
                self.q_scale.lerp_((quantiles[1] - quantiles[0]).clamp(min=1).reshape_as(self.q_scale), self.model.cfg.tau)
            rho = self.cfg.rho ** torch.arange(h + 1, device=device)
            pi_loss = ((-self.cfg.entropy_coef * pi_info["scaled_entropy"] - pi_q / self.q_scale).mean(dim=(1, 2)) * rho).mean()
            if not torch.isfinite(pi_loss):
                raise FloatingPointError("nonfinite TD-MPC2 policy loss")
            self.pi_optim.zero_grad(set_to_none=True)
            pi_loss.backward()
            pi_grad_norm = nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
            if not torch.isfinite(pi_grad_norm):
                raise FloatingPointError("nonfinite TD-MPC2 policy gradient")
            self.pi_optim.step()
            self.pi_optim.zero_grad(set_to_none=True)
            self.model.soft_update_target_Q()

            return {
                "consistency_loss": consistency_loss.detach().item(),
                "reward_loss": reward_loss.detach().item(),
                "value_loss": value_loss.detach().item(),
                "termination_loss": termination_loss.detach().item(),
                "total_loss": total_loss.detach().item(),
                "grad_norm": grad_norm.detach().item(),
                "pi_loss": pi_loss.detach().item(),
                "pi_grad_norm": pi_grad_norm.detach().item(),
                "pi_entropy": pi_info["entropy"].detach().mean().item(),
                "pi_scaled_entropy": pi_info["scaled_entropy"].detach().mean().item(),
                "pi_scale": self.q_scale.detach().item(),
            }
        finally:
            self.model.eval()
