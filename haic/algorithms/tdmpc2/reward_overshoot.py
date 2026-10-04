"""Isolated H3 TD-MPC2 reward-target overshooting on logged episode suffixes.

Only imagined rewards at steps 4 and 5 are added. The original model, replay
sampling, critic, termination, actor, and target-Q update remain unchanged.
"""

from typing import Any, Mapping, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from .learner import TDMPC2Learner, TDMPC2LearnerConfig
from .model import soft_ce
from .replay import EpisodeReplay


class OvershootReplay(EpisodeReplay):
    """Append two same-episode logged transitions to an unchanged H3 sample."""

    def __init__(self, capacity: int, horizon: int = 3, action_dim: int = 3, **kwargs):
        if horizon != 3 or action_dim != 3:
            raise ValueError("reward overshooting requires H3 and three-dimensional actions")
        super().__init__(capacity, horizon, action_dim, **kwargs)

    def sample(
        self,
        batch_size: int,
        *,
        include_partial: bool | None = None,
        augment: bool = False,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = super().sample(
            batch_size, include_partial=include_partial, augment=augment,
            device=device, generator=generator,
        )
        actions = np.zeros((2, batch_size, 3), dtype=np.float32)
        rewards = np.zeros((2, batch_size, 1), dtype=np.float32)
        masks = np.zeros((2, batch_size, 1), dtype=np.float32)
        episodes = {ep.episode_id: ep for ep in self._episodes}
        if self._active is not None:
            episodes[self._active.episode_id] = self._active
        for column, (episode_id, start_step) in enumerate(zip(
            batch["episode_id"].tolist(), batch["start_step"].tolist()
        )):
            episode = episodes[int(episode_id)]
            start = int(start_step) - episode.start_step
            if start < 0 or start + 3 > len(episode):
                raise ValueError("sampled H3 window falls outside its retained episode")
            # The base window can end at a finish, raw termination, or time limit.
            # None permits a suffix, even though an ordinary time limit bootstraps.
            for offset in range(2):
                index = start + 3 + offset
                if index >= len(episode):
                    break
                if any(episode.terminated[start:index]) or any(episode.truncated[start:index]):
                    raise ValueError("overshoot suffix crosses an episode boundary")
                actions[offset, column] = episode.actions[index]
                rewards[offset, column, 0] = episode.rewards[index]
                masks[offset, column, 0] = 1.0
        batch["overshoot_action"] = torch.from_numpy(actions).to(device)
        batch["overshoot_reward"] = torch.from_numpy(rewards).to(device)
        batch["overshoot_mask"] = torch.from_numpy(masks).to(device)
        return batch


class OvershootLearner(TDMPC2Learner):
    """Original joint update plus masked reward CE at imagined steps 4 and 5."""

    def __init__(
        self, model: nn.Module, cfg: TDMPC2LearnerConfig, *, aux_enabled: bool = True,
    ) -> None:
        super().__init__(model, cfg)
        if type(aux_enabled) is not bool:
            raise TypeError("aux_enabled must be bool")
        if aux_enabled and (cfg.horizon != 3 or cfg.reward_coef != 0.1 or model.cfg.action_dim != 3):
            raise ValueError("reward overshooting requires H3, reward_coef=0.1 and 3D actions")
        self.aux_enabled = aux_enabled

    def _overshoot_reward_loss(
        self, z3: torch.Tensor, action: torch.Tensor,
        reward: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Original batch mean and /3, including zero-masked batch columns."""
        loss = torch.zeros((), device=z3.device)
        if bool(mask[0].any()):
            logits4 = self.model.reward(z3, action[0], None)
            loss = loss + self.cfg.rho ** 3 * (soft_ce(logits4, reward[0], self.model.cfg) * mask[0]).mean()
        if bool(mask[1].any()):
            z4 = self.model.next(z3, action[0], None)
            logits5 = self.model.reward(z4, action[1], None)
            loss = loss + self.cfg.rho ** 4 * (soft_ce(logits5, reward[1], self.model.cfg) * mask[1]).mean()
        return loss / 3

    def update(self, batch: Mapping[str, torch.Tensor] | object) -> dict[str, float]:
        if not self.aux_enabled:
            return super().update(batch)
        if not isinstance(batch, Mapping):
            batch = cast(Any, batch).sample(self.cfg.batch_size)
        batch = cast(Mapping[str, torch.Tensor], batch)
        if "weights" in batch or "priorities" in batch:
            raise ValueError("upstream uniform replay does not support importance weights")
        device = self.q_scale.device
        obs = torch.as_tensor(batch["obs"], device=device)
        action = torch.as_tensor(batch["action"], device=device, dtype=torch.float32)
        reward = torch.as_tensor(batch["reward"], device=device, dtype=torch.float32)
        terminated = torch.as_tensor(batch["terminated"], device=device, dtype=torch.float32)
        terminal = torch.as_tensor(batch["terminal"], device=device, dtype=torch.float32)
        truncated = torch.as_tensor(batch["truncated"], device=device, dtype=torch.float32)
        extra_action = torch.as_tensor(batch["overshoot_action"], device=device, dtype=torch.float32)
        extra_reward = torch.as_tensor(batch["overshoot_reward"], device=device, dtype=torch.float32)
        extra_mask = torch.as_tensor(batch["overshoot_mask"], device=device, dtype=torch.float32)

        h, b = self.cfg.horizon, action.shape[1]
        if obs.shape[:2] != (h + 1, b) or action.shape != (h, b, 3):
            raise ValueError("expected time-major H+1 observations and H three-dimensional actions")
        if reward.shape != (h, b, 1) or terminated.shape != reward.shape or terminal.shape != reward.shape or truncated.shape != reward.shape:
            raise ValueError("expected [H,B,1] rewards, terminal labels and truncations")
        if extra_action.shape != (2, b, 3) or extra_reward.shape != (2, b, 1) or extra_mask.shape != extra_reward.shape:
            raise ValueError("expected two [B,3] suffix actions and [B,1] rewards/masks")
        if not torch.isfinite(reward).all() or not torch.isfinite(action).all() or not torch.isfinite(extra_reward).all() or not torch.isfinite(extra_action).all():
            raise ValueError("nonfinite replay reward or action")
        if not torch.all((terminated == 0) | (terminated == 1)) or not torch.all((terminal == 0) | (terminal == 1)) or not torch.all((truncated == 0) | (truncated == 1)) or not torch.all((extra_mask == 0) | (extra_mask == 1)):
            raise ValueError("termination, truncation and overshoot masks must be binary")
        if torch.any(terminated > terminal) or torch.any(terminal > torch.maximum(terminated, truncated)):
            raise ValueError("semantic terminal must include raw termination and mark finishes, not time limits")
        if torch.any(torch.maximum(terminated[:-1], truncated[:-1]) != 0):
            raise ValueError("H3 windows must not cross episode boundaries")
        if torch.any(extra_mask[1] > extra_mask[0]) or torch.any(extra_mask[0] > 1 - torch.maximum(terminated[-1], truncated[-1])):
            raise ValueError("overshoot masks must stop at the first episode boundary")
        if torch.any(extra_action * (1 - extra_mask) != 0) or torch.any(extra_reward * (1 - extra_mask) != 0):
            raise ValueError("masked overshoot suffix must be zero")
        if "bootstrap_mask" in batch:
            mask = torch.as_tensor(batch["bootstrap_mask"], device=device)
            if mask.shape != reward.shape or not torch.equal(mask, 1 - terminal):
                raise ValueError("bootstrap mask must equal 1-semantic terminal (including finish)")
        if not self.model.cfg.episodic and bool(terminal.any()):
            raise ValueError("episodic=True is required when replay contains terminal transitions")

        # The remainder mirrors the frozen H3 update; only reward_loss receives
        # the extra two masked terms, before the same joint optimizer step.
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
            overshoot_reward_loss = self._overshoot_reward_loss(zs[-1], extra_action, extra_reward, extra_mask)
            reward_loss = reward_loss + overshoot_reward_loss
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
            grad_norm = clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("nonfinite TD-MPC2 model gradient")
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)

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
            pi_grad_norm = clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
            if not torch.isfinite(pi_grad_norm):
                raise FloatingPointError("nonfinite TD-MPC2 policy gradient")
            self.pi_optim.step()
            self.pi_optim.zero_grad(set_to_none=True)
            self.model.soft_update_target_Q()

            return {
                "consistency_loss": consistency_loss.detach().item(),
                "reward_loss": reward_loss.detach().item(),
                "overshoot_reward_loss": overshoot_reward_loss.detach().item(),
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
