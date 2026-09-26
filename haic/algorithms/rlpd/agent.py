"""Independent pixel RLPD learner; no DrQ-v2 learner/checkpoint dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from common_adapter import ActionAdapter, ObservationSpec
from haic.algorithms.rlpd.augment import random_shift
from haic.algorithms.rlpd.model import (
    ACTION_DIM,
    NUM_QS,
    PixelActor,
    PixelCritic,
    select_target_q_values,
)


ACTOR_FORMAT = "haic-rlpd-pixel-actor-v1"


@dataclass(frozen=True)
class RLPDConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    num_qs: int = NUM_QS
    num_min_qs: int = 1
    target_entropy: float = -ACTION_DIM / 2
    initial_alpha: float = 0.1
    backup_entropy: bool = False
    augmentation_pad: int = 4

    def __post_init__(self) -> None:
        for name in ("actor_lr", "critic_lr", "temperature_lr", "gamma", "tau", "initial_alpha"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.gamma > 1.0 or self.tau > 1.0:
            raise ValueError("gamma and tau must not exceed one")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.num_qs != NUM_QS or self.num_min_qs != 1:
            raise ValueError("the pinned pixel contract requires 10 Qs and a 1-Q target subset")
        if type(self.backup_entropy) is not bool:
            raise ValueError("backup_entropy must be boolean")
        if type(self.augmentation_pad) is not int or self.augmentation_pad < 0:
            raise ValueError("augmentation_pad must be a non-negative integer")


def sample_balanced_batch(offline, online, *, batch_size: int, offline_count: int):
    """Concatenate fixed per-source sample counts without silent fallback."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(offline_count) is not int or not 0 < offline_count < batch_size:
        raise ValueError("offline_count must leave a positive count for both sources")
    if len(offline) == 0 or len(online) == 0:
        raise ValueError("both frozen offline and student online replay must be non-empty")
    offline_batch = offline.sample(offline_count)
    online_batch = online.sample(batch_size - offline_count)
    keys = set(offline_batch) - {"source"}
    if keys != set(online_batch) - {"source"}:
        raise ValueError("offline and online replay batch fields do not match")
    result = {key: np.concatenate((offline_batch[key], online_batch[key]), axis=0) for key in keys}
    result["source"] = np.concatenate((offline_batch["source"], online_batch["source"]))
    result["source_counts"] = {
        "offline": int(len(offline_batch["source"])),
        "online": int(len(online_batch["source"])),
    }
    return result


def sac_bellman_target(
    reward: torch.Tensor,
    terminal: torch.Tensor,
    target_q: torch.Tensor,
    *,
    gamma: float,
    backup_entropy: bool = False,
    alpha: torch.Tensor | None = None,
    next_log_prob: torch.Tensor | None = None,
) -> torch.Tensor:
    """One-step SAC target; truncation itself does not suppress bootstrapping."""
    if reward.ndim != 2 or terminal.shape != reward.shape or target_q.shape != reward.shape:
        raise ValueError("reward, terminal, and target_q must have matching (batch, 1) shapes")
    if type(backup_entropy) is not bool or not 0.0 < gamma <= 1.0:
        raise ValueError("invalid entropy-backup flag or discount")
    if (
        not torch.isfinite(reward).all()
        or not torch.isfinite(terminal).all()
        or not torch.isfinite(target_q).all()
        or torch.any((terminal < 0.0) | (terminal > 1.0))
    ):
        raise ValueError("Bellman target inputs must be finite with terminal in [0, 1]")
    if backup_entropy:
        if alpha is None or next_log_prob is None or next_log_prob.shape != reward.shape:
            raise ValueError("entropy backup requires alpha and matching next_log_prob")
        target_q = target_q - alpha * next_log_prob
    return reward + gamma * (1.0 - terminal.to(dtype=reward.dtype)) * target_q


class PixelRLPDAgent:
    """Q10 pixel SAC with one random target head and optional frozen prior data."""

    def __init__(
        self,
        config: RLPDConfig | None = None,
        *,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.config = config or RLPDConfig()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.observation_spec = ObservationSpec()
        self.action_adapter = ActionAdapter()
        self.actor = PixelActor().to(self.device)
        self.critic = PixelCritic(num_qs=self.config.num_qs).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.actor.encoder.convolution.load_state_dict(self.critic.encoder.convolution.state_dict())
        for parameter in self.actor.encoder.convolution.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(self.config.initial_alpha), dtype=torch.float32, device=self.device)
        )
        self.actor_optimizer = torch.optim.Adam(
            (parameter for parameter in self.actor.parameters() if parameter.requires_grad),
            lr=self.config.actor_lr,
            weight_decay=0.0,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr, weight_decay=0.0
        )
        self.temperature_optimizer = torch.optim.Adam(
            (self.log_alpha,), lr=self.config.temperature_lr, weight_decay=0.0
        )
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed + 0x524C5044)
        self.environment_steps = 0
        self.gradient_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.inference_mode()
    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        action, _, _ = self.act_with_details(observation, deterministic=deterministic)
        return action

    @torch.inference_mode()
    def act_with_details(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, float | None]:
        value = self.observation_spec.validate(observation)
        tensor = torch.from_numpy(value).to(self.device).unsqueeze(0)
        action, log_prob, pre_tanh = self.actor.sample(
            tensor, deterministic=deterministic, generator=None if deterministic else self.rng
        )
        result = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if result.shape != (ACTION_DIM,) or not np.isfinite(result).all():
            raise RuntimeError("RLPD actor produced an invalid native action")
        result = np.clip(result, -1.0, 1.0)
        pre_tanh_result = (
            None if pre_tanh is None else pre_tanh.squeeze(0).detach().cpu().numpy().astype(np.float32)
        )
        log_prob_result = None if log_prob is None else float(log_prob.squeeze().detach().item())
        return result, pre_tanh_result, log_prob_result

    def reset(self, _observation=None) -> None:
        """The pixel actor is feed-forward and has no episode carry."""

    def _batch_tensors(self, batch: dict[str, Any]):
        observation = torch.as_tensor(batch["observation"], device=self.device)
        next_observation = torch.as_tensor(batch["next_observation"], device=self.device)
        action = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device).reshape(-1, 1)
        terminal = torch.as_tensor(batch["terminal"], dtype=torch.float32, device=self.device).reshape(-1, 1)
        if observation.dtype != torch.uint8 or next_observation.dtype != torch.uint8:
            raise TypeError("pixel replay must return uint8 observations")
        if (
            observation.shape[0] != self.config.batch_size
            or next_observation.shape != observation.shape
            or action.shape != (self.config.batch_size, ACTION_DIM)
            or reward.shape != (self.config.batch_size, 1)
            or terminal.shape != (self.config.batch_size, 1)
        ):
            raise ValueError("sampled replay batch does not match the frozen learner contract")
        return observation, action, reward, next_observation, terminal

    @staticmethod
    def _finite_grad_norm(parameters) -> float:
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients:
            raise RuntimeError("learner update produced no gradients")
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("learner update produced non-finite gradients")
        return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach()) for g in gradients])).item())

    @torch.no_grad()
    def _update_target(self) -> None:
        tau = self.config.tau
        for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.lerp_(source, tau)
        for target, source in zip(self.target_critic.buffers(), self.critic.buffers()):
            target.copy_(source)

    def update(self, batch: dict[str, Any]) -> dict[str, float | int]:
        observation, action, reward, next_observation, terminal = self._batch_tensors(batch)
        current = observation.float().div(255.0)
        following = next_observation.float().div(255.0)
        current = random_shift(current, pad=self.config.augmentation_pad, generator=self.rng)
        following = random_shift(following, pad=self.config.augmentation_pad, generator=self.rng)

        # Match the author pixel update: synchronize only the actor CNN before
        # critic optimization; its latent projection remains independently trained.
        self.actor.encoder.convolution.load_state_dict(
            self.critic.encoder.convolution.state_dict()
        )

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(following, generator=self.rng)
            all_target_q = self.target_critic(following, next_action)
            subset = torch.randperm(
                self.config.num_qs, device=self.device, generator=self.rng
            )[: self.config.num_min_qs]
            next_q = select_target_q_values(all_target_q, subset)
            target = sac_bellman_target(
                reward,
                terminal,
                next_q,
                gamma=self.config.gamma,
                backup_entropy=self.config.backup_entropy,
                alpha=self.alpha.detach() if self.config.backup_entropy else None,
                next_log_prob=next_log_prob if self.config.backup_entropy else None,
            )

        current_q = self.critic(current, action)
        critic_loss = (current_q - target).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = self._finite_grad_norm(self.critic.parameters())
        self.critic_optimizer.step()
        self._update_target()

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        try:
            self.actor_optimizer.zero_grad(set_to_none=True)
            proposed, log_prob, _ = self.actor.sample(current, generator=self.rng)
            with torch.no_grad():
                critic_features = self.critic.encoder(current)
            policy_q = self.critic(current, proposed, features=critic_features)
            actor_loss = (self.alpha.detach() * log_prob - policy_q.mean(dim=1, keepdim=True)).mean()
            actor_loss.backward()
            actor_parameters = [parameter for parameter in self.actor.parameters() if parameter.requires_grad]
            actor_grad_norm = self._finite_grad_norm(actor_parameters)
            self.actor_optimizer.step()
        finally:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)

        entropy = -log_prob.detach()
        # Faithful to pinned SACLearner.update_temperature(): target_entropy=-1.5
        # and differential entropy is reported as -log_prob, despite its unusual
        # alpha direction for ordinary positive differential entropy.
        temperature_loss = (self.alpha * (entropy - self.config.target_entropy)).mean()
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        if self.log_alpha.grad is None or not torch.isfinite(self.log_alpha.grad):
            raise FloatingPointError("temperature update produced an invalid alpha gradient")
        alpha_gradient = float(self.log_alpha.grad.detach().item())
        self.temperature_optimizer.step()

        self.gradient_steps += 1
        source_values, source_counts = np.unique(batch["source"], return_counts=True)
        source_metrics = {str(key): int(value) for key, value in zip(source_values, source_counts)}
        target_error = (current_q.detach() - target).abs().mean()
        q_variance = current_q.detach().var(dim=1, unbiased=False).mean()
        executed = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        proposed_batch = torch.as_tensor(batch["proposed_action"], dtype=torch.float32, device=self.device)
        metrics = {
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "temperature_loss": float(temperature_loss.detach().item()),
            "alpha": float(self.alpha.detach().item()),
            "alpha_gradient": alpha_gradient,
            "entropy": float(entropy.mean().item()),
            "target_q": float(target.mean().item()),
            "target_error": float(target_error.item()),
            "q_mean": float(current_q.detach().mean().item()),
            "q_max": float(current_q.detach().max().item()),
            "q_ensemble_variance": float(q_variance.item()),
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "log_std_min": float(self.actor.distribution(current)[1].detach().min().item()),
            "log_std_max": float(self.actor.distribution(current)[1].detach().max().item()),
            "action_execution_mismatch_fraction": float(
                (executed - proposed_batch).abs().gt(1e-6).any(dim=1).float().mean().item()
            ),
            "target_q_head": int(subset[0].item()),
            "offline_samples": source_metrics.get("offline", 0),
            "online_samples": source_metrics.get("online", 0),
            "gradient_steps": self.gradient_steps,
        }
        if any(not math.isfinite(float(value)) for key, value in metrics.items() if key not in {
            "target_q_head", "offline_samples", "online_samples", "gradient_steps"
        }):
            raise FloatingPointError("learner produced a non-finite metric")
        return metrics

    def export_actor(
        self,
        path: str | Path,
        *,
        source_sha256: str,
        protocol_sha256: str,
        training_seed: int,
        environment_contract: dict[str, Any],
    ) -> Path:
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        if not all(
            isinstance(value, str) and len(value) == 64
            for value in (source_sha256, protocol_sha256)
        ):
            raise ValueError("source and protocol hashes must be SHA-256 strings")
        actor_state = {key: value.detach().cpu().clone() for key, value in self.actor.state_dict().items()}
        payload = {
            "format": ACTOR_FORMAT,
            "config": {"latent_dim": self.actor.encoder.latent_dim},
            "observation_spec": asdict(self.observation_spec),
            "action_spec": asdict(self.action_adapter.spec),
            "actor_state_dict": actor_state,
            "training_seed": int(training_seed),
            "environment_steps": int(self.environment_steps),
            "gradient_steps": int(self.gradient_steps),
            "source_sha256": source_sha256,
            "protocol_sha256": protocol_sha256,
            "environment_contract": dict(environment_contract),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path

    def checkpoint_state(self, *, offline_replay, online_replay, trainer_state: dict[str, Any]):
        """Full local resume snapshot; offline and online replay stay separate."""
        return {
            "format": "haic-rlpd-training-checkpoint-v1",
            "config": asdict(self.config),
            "device_type": self.device.type,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "torch_generator_state": self.rng.get_state(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
            "offline_replay": offline_replay.state_dict(),
            "online_replay": online_replay.state_dict(),
            "trainer_state": dict(trainer_state),
        }

    def load_checkpoint_state(self, state: dict[str, Any], *, offline_replay, online_replay):
        if state.get("format") != "haic-rlpd-training-checkpoint-v1":
            raise ValueError("unsupported RLPD training checkpoint")
        if state.get("config") != asdict(self.config):
            raise ValueError("RLPD checkpoint configuration differs from the current run")
        self.actor.load_state_dict(state["actor"], strict=True)
        self.critic.load_state_dict(state["critic"], strict=True)
        self.target_critic.load_state_dict(state["target_critic"], strict=True)
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.temperature_optimizer.load_state_dict(state["temperature_optimizer"])
        for optimizer in (self.actor_optimizer, self.critic_optimizer, self.temperature_optimizer):
            for values in optimizer.state.values():
                for key, value in values.items():
                    if isinstance(value, torch.Tensor):
                        values[key] = value.to(self.device)
        self.rng.set_state(state["torch_generator_state"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.random.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_initialized() and state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        self.environment_steps = int(state["environment_steps"])
        self.gradient_steps = int(state["gradient_steps"])
        offline_replay.load_state_dict(state["offline_replay"])
        online_replay.load_state_dict(state["online_replay"])
        return dict(state["trainer_state"])
