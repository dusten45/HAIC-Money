"""Head-only reward calibration on observed TD-MPC2 pixel/action/reward triples.

The caller loads and provenance-checks the frozen model and constructs observed
single-transition batches. This module does not load replay or checkpoints and
does not own the experiment's update count.
"""

import hashlib
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from .model import NormedLinear, WorldModel, soft_ce


AUGMENTATION_SEED = 20260929


class RewardHeadAdapter:
    """Perform one categorical reward-head update per observed B256 batch."""

    def __init__(self, model: WorldModel) -> None:
        if not isinstance(model, WorldModel):
            raise TypeError("expected a loaded TD-MPC2 WorldModel")
        if model.cfg.action_dim != 3 or model.cfg.num_bins != 101:
            raise ValueError("reward adaptation requires action_dim=3 and num_bins=101")
        if (not isinstance(model._reward, nn.Sequential) or len(model._reward) != 3
                or not all(isinstance(layer, NormedLinear) for layer in model._reward[:2])
                or not isinstance(model._reward[-1], nn.Linear)):
            raise ValueError("reward adaptation requires the original reward MLP")
        if model._reward[-1].out_features != 101:
            raise ValueError("reward MLP must output 101 categorical logits")
        if tuple(model.cfg.obs_shape["rgb"]) != (4, 64, 64):
            raise ValueError("reward adaptation requires four raw 64x64 pixel frames")
        self.model = model
        model.requires_grad_(False)
        model._reward.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model.eval()
        self._head_params = tuple(model._reward.parameters())
        if not self._head_params or not all(p.requires_grad for p in self._head_params):
            raise ValueError("reward MLP has no trainable parameters")
        self.optim = torch.optim.Adam(self._head_params, lr=3e-5, weight_decay=0)
        self._initial = self.snapshot()
        self.updates = 0

    def snapshot(self) -> dict[str, str]:
        """Fingerprint every model parameter and buffer, including target Qs."""
        result = {}
        for key, value in self.model.state_dict().items():
            tensor = value.detach().to(device="cpu").contiguous()
            digest = hashlib.sha256()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            result[key] = digest.hexdigest()
        return result

    def verify_only_reward_changed(self) -> bool:
        """Raise on any non-head byte drift; return whether the head changed."""
        current = self.snapshot()
        if current.keys() != self._initial.keys():
            raise RuntimeError("model state_dict keys changed during head adaptation")
        for key, original in self._initial.items():
            if not key.startswith("_reward.") and current[key] != original:
                raise RuntimeError(f"non-head model state changed: {key}")
        if self.model.training or any(module.training for module in self.model.modules()):
            raise RuntimeError("model must remain in eval mode")
        if any(p.requires_grad or p.grad is not None for name, p in self.model.named_parameters()
               if not name.startswith("_reward.")):
            raise RuntimeError("frozen model parameter has a gradient or is trainable")
        return any(current[key] != original for key, original in self._initial.items()
                   if key.startswith("_reward."))

    def step(self, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        """Update only _reward from current raw observations and logged actions."""
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

        # WorldModel.encode retains its random ShiftAug in eval mode. Forking
        # the device RNG leaves the caller's CPU/CUDA streams bitwise unchanged.
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
        loss = soft_ce(logits, reward.detach(), self.model.cfg).mean()
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
