"""Objective-only raw decoded reward MSE control of the frozen head adapter.

Constructor, architecture, freeze checks, snapshots and Adam settings are inherited
from RewardHeadAdapter c2a1626b3ab321100ae22579d1f665a2b3fdc69661d3cdcce796f4df715fc615.
The copied step changes only categorical CE to standard decoded raw scalar MSE.
"""

from collections.abc import Mapping

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from .model import two_hot_inv
from .reward_head_adapt import AUGMENTATION_SEED, RewardHeadAdapter


class RawRewardMSEAdapter(RewardHeadAdapter):
    """One full observed-frame B256 head-only raw reward MSE Adam update."""

    def step(self, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        if not isinstance(batch, Mapping) or set(batch) != {"obs", "action", "reward"}:
            raise ValueError("batch must contain only obs, action, reward")
        self.verify_only_reward_changed()
        device = self._head_params[0].device
        obs = torch.as_tensor(batch["obs"], device=device)
        action = torch.as_tensor(batch["action"], device=device)
        reward = torch.as_tensor(batch["reward"], device=device)
        if obs.shape != (256, 4, 64, 64) or obs.dtype != torch.uint8:
            raise ValueError("obs must be observed uint8 [256,4,64,64] frames")
        if action.shape != (256, 3) or action.dtype != torch.float32:
            raise ValueError("action must be logged float32 [256,3] commands")
        if reward.shape != (256, 1) or reward.dtype != torch.float32:
            raise ValueError("reward must be raw float32 [256,1] labels")
        if not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(reward).all()):
            raise ValueError("nonfinite action or reward")
        if bool((action.abs() > 1).any()):
            raise ValueError("logged actions must be in [-1,1]")

        # Preserve the parent's augmentation schedule and the caller's RNG streams.
        rng_devices = []
        if device.type == "cuda":
            rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=rng_devices):
            torch.default_generator.manual_seed(AUGMENTATION_SEED + self.updates)
            if rng_devices:
                torch.cuda.default_generators[rng_devices[0]].manual_seed(AUGMENTATION_SEED + self.updates)
            with torch.no_grad():
                z = self.model.encode(obs)
        if z.shape != (256, self.model.cfg.latent_dim) or not bool(torch.isfinite(z).all()):
            raise FloatingPointError("nonfinite or malformed encoded observation")
        self.optim.zero_grad(set_to_none=True)
        logits = self.model.reward(z, action.detach())
        if logits.shape != (256, 101) or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("nonfinite or malformed reward logits")
        loss = (two_hot_inv(logits, self.model.cfg) - reward.detach()).square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite reward loss")
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self._head_params):
            raise FloatingPointError("missing or nonfinite reward-head gradient")
        grad_norm = clip_grad_norm_(self._head_params, 10.0, error_if_nonfinite=True)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError("nonfinite reward-head gradient norm")
        self.optim.step()
        if any(not bool(torch.isfinite(p).all()) for p in self._head_params):
            raise FloatingPointError("nonfinite reward-head parameter after optimizer step")
        if any(not bool(torch.isfinite(value).all()) for state in self.optim.state.values()
               for value in state.values() if isinstance(value, torch.Tensor)):
            raise FloatingPointError("nonfinite reward-head optimizer state after step")
        self.optim.zero_grad(set_to_none=True)
        self.updates += 1
        self.verify_only_reward_changed()
        return {"loss": float(loss.detach().item()), "grad_norm": float(grad_norm.detach().item())}
