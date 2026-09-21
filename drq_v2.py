"""Small native DrQ-v2 implementation for the frozen HAIC pixel contract."""

from __future__ import annotations

import copy
import json
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common_adapter import (
    ActionAdapter,
    ActionSpec,
    ObservationSpec,
    Transition,
    build_checkpoint_manifest,
    write_checkpoint_manifest,
)


def _as_uint8_observation(observation: np.ndarray, spec: ObservationSpec) -> np.ndarray:
    array = np.asarray(observation)
    if array.dtype == np.uint8:
        if array.shape != spec.shape:
            raise ValueError(f"observation must have shape {spec.shape}")
        return np.ascontiguousarray(array)
    return spec.to_uint8(array)


class Uint8Replay:
    """Compact frame replay with terminal-safe n-step sampling.

    A transition stores one newest frame rather than duplicating a four-frame
    stack.  Stacks are reconstructed from chronological frames and episode
    metadata; only terminal/truncation boundary observations require a small
    full-stack side table.  This keeps the default 100k replay below the
    memory cost of two full observations per transition.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        *,
        observation_spec: ObservationSpec | None = None,
        action_dim: int = 3,
        n_step: int = 3,
        gamma: float = 0.99,
        seed: int = 0,
    ):
        if capacity < 2:
            raise ValueError("replay capacity must be at least 2")
        if n_step < 1:
            raise ValueError("n_step must be positive")
        if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be finite and in (0, 1]")
        self.capacity = int(capacity)
        self.observation_spec = observation_spec or ObservationSpec()
        self.action_dim = int(action_dim)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        height, width = self.observation_spec.shape[1:]
        self.frames = np.zeros((self.capacity, height, width), dtype=np.uint8)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.bool_)
        self.truncated = np.zeros(self.capacity, dtype=np.bool_)
        self.terminal = np.zeros(self.capacity, dtype=np.bool_)
        self.episode_ids = np.full(self.capacity, -1, dtype=np.int64)
        self.episode_steps = np.full(self.capacity, -1, dtype=np.int64)
        self.sequence_ids = np.full(self.capacity, -1, dtype=np.int64)
        self._boundary_observations: dict[int, np.ndarray] = {}
        self._next_sequence = 0
        self._size = 0
        self.rng = np.random.default_rng(seed)

    @property
    def size(self) -> int:
        return self._size

    @property
    def memory_bytes(self) -> int:
        arrays = (
            self.frames,
            self.actions,
            self.rewards,
            self.terminated,
            self.truncated,
            self.terminal,
            self.episode_ids,
            self.episode_steps,
            self.sequence_ids,
        )
        return int(sum(array.nbytes for array in arrays))

    @property
    def oldest_sequence(self) -> int:
        return max(0, self._next_sequence - self._size)

    @property
    def newest_sequence(self) -> int:
        return self._next_sequence - 1

    def _record(self, sequence: int) -> tuple[int, int] | None:
        if sequence < self.oldest_sequence or sequence >= self._next_sequence:
            return None
        slot = int(sequence % self.capacity)
        if int(self.sequence_ids[slot]) != int(sequence):
            return None
        return slot, int(self.episode_ids[slot])

    def add(self, transition: Transition) -> None:
        converted = transition.as_uint8(self.observation_spec)
        if converted.action.shape != (self.action_dim,) or not np.isfinite(converted.action).all():
            raise ValueError("replay action has the wrong shape or is non-finite")
        if np.any(converted.action < -1.000001) or np.any(converted.action > 1.000001):
            raise ValueError("replay expects symmetric normalized actions")
        if converted.observation.shape != self.observation_spec.shape:
            raise ValueError("replay observation shape does not match the contract")
        sequence = self._next_sequence
        slot = int(sequence % self.capacity)
        old_sequence = int(self.sequence_ids[slot])
        if old_sequence >= 0:
            self._boundary_observations.pop(old_sequence, None)
        self.frames[slot] = converted.observation[-1]
        self.actions[slot] = converted.action
        self.rewards[slot] = converted.reward
        self.terminated[slot] = converted.terminated
        self.truncated[slot] = converted.truncated
        self.terminal[slot] = bool(converted.terminal)
        self.episode_ids[slot] = int(converted.episode_id)
        self.episode_steps[slot] = int(converted.step)
        self.sequence_ids[slot] = sequence
        if converted.done:
            self._boundary_observations[sequence] = converted.next_observation.copy()
        self._next_sequence += 1
        self._size = min(self.capacity, self._size + 1)

    def _stack(self, sequence: int) -> np.ndarray | None:
        record = self._record(sequence)
        if record is None:
            return None
        slot, episode_id = record
        current_step = int(self.episode_steps[slot])
        if current_step < 0:
            return None
        channels = self.observation_spec.shape[0]
        result = np.empty(self.observation_spec.shape, dtype=np.uint8)
        for channel in range(channels):
            # Clamp only reset padding; later stacks do not need the episode start.
            offset = max(channel - channels + 1, -current_step)
            frame_record = self._record(sequence + offset)
            if frame_record is None:
                return None
            frame_slot, frame_episode = frame_record
            if frame_episode != episode_id or self.episode_steps[frame_slot] != current_step + offset:
                return None
            result[channel] = self.frames[frame_slot]
        return result

    def _build_n_step(self, start: int) -> dict[str, Any] | None:
        first = self._record(start)
        if first is None:
            return None
        first_slot, episode_id = first
        first_step = int(self.episode_steps[first_slot])
        total_reward = 0.0
        horizon = 0
        endpoint = None
        for offset in range(self.n_step):
            record = self._record(start + offset)
            if record is None or record[1] != episode_id:
                return None
            slot, _ = record
            if self.episode_steps[slot] != first_step + offset:
                return None
            total_reward += (self.gamma**offset) * float(self.rewards[slot])
            horizon = offset + 1
            endpoint = slot
            if self.terminal[slot] or self.terminated[slot] or self.truncated[slot]:
                break
        if endpoint is None:
            return None
        endpoint_sequence = start + horizon - 1
        endpoint_done = bool(
            self.terminal[endpoint] or self.terminated[endpoint] or self.truncated[endpoint]
        )
        if endpoint_done:
            next_observation = self._boundary_observations.get(endpoint_sequence)
            if next_observation is None:
                return None
            discount = self.gamma**horizon if not self.terminal[endpoint] else 0.0
        else:
            next_record = self._record(start + horizon)
            if (
                next_record is None
                or next_record[1] != episode_id
                or self.episode_steps[next_record[0]] != first_step + horizon
            ):
                return None
            next_observation = self._stack(start + horizon)
            if next_observation is None:
                return None
            discount = self.gamma**horizon
        observation = self._stack(start)
        if observation is None:
            return None
        return {
            "observation": observation,
            "action": self.actions[first[0]].copy(),
            "reward": np.float32(total_reward),
            "next_observation": np.asarray(next_observation, dtype=np.uint8).copy(),
            "terminated": np.uint8(self.terminated[endpoint]),
            "truncated": np.uint8(self.truncated[endpoint]),
            "terminal": np.uint8(self.terminal[endpoint]),
            "discount": np.float32(discount),
            "horizon": np.int64(horizon),
            "episode_id": np.int64(episode_id),
            "sequence": np.int64(start),
        }

    def valid_indices(self) -> list[int]:
        return [
            sequence
            for sequence in range(self.oldest_sequence, self._next_sequence)
            if self._build_n_step(sequence) is not None
        ]

    def sample(self, batch_size: int, indices: Iterable[int] | None = None) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if indices is None:
            chosen = []
            attempts = 0
            while len(chosen) < batch_size and attempts < max(batch_size * 32, 256):
                attempts += 1
                candidate = int(self.rng.integers(self.oldest_sequence, self._next_sequence))
                if candidate in chosen or self._build_n_step(candidate) is None:
                    continue
                chosen.append(candidate)
            if len(chosen) < batch_size:
                raise ValueError("not enough terminal-safe replay transitions")
        else:
            chosen = [int(index) for index in indices]
            if len(chosen) != batch_size:
                raise ValueError("indices length must equal batch_size")
        rows = []
        for index in chosen:
            row = self._build_n_step(index)
            if row is None:
                raise ValueError(f"replay index {index} is not terminal-safe")
            rows.append(row)
        return {
            key: np.stack([row[key] for row in rows])
            for key in rows[0]
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "action_dim": self.action_dim,
            "n_step": self.n_step,
            "gamma": self.gamma,
            "next_sequence": self._next_sequence,
            "size": self._size,
            "frames": self.frames.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "terminated": self.terminated.copy(),
            "truncated": self.truncated.copy(),
            "terminal": self.terminal.copy(),
            "episode_ids": self.episode_ids.copy(),
            "episode_steps": self.episode_steps.copy(),
            "sequence_ids": self.sequence_ids.copy(),
            "boundary_observations": {
                int(key): value.copy() for key, value in self._boundary_observations.items()
            },
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity or int(state["action_dim"]) != self.action_dim:
            raise ValueError("replay checkpoint shape does not match configured replay")
        if int(state["n_step"]) != self.n_step or float(state["gamma"]) != self.gamma:
            raise ValueError("replay checkpoint return configuration does not match")
        for name in (
            "frames",
            "actions",
            "rewards",
            "terminated",
            "truncated",
            "terminal",
            "episode_ids",
            "episode_steps",
            "sequence_ids",
        ):
            target = getattr(self, name)
            value = np.asarray(state[name], dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(f"replay checkpoint array {name} has the wrong shape")
            target[...] = value
        self._next_sequence = int(state["next_sequence"])
        self._size = int(state["size"])
        self._boundary_observations = {
            int(key): np.asarray(value, dtype=np.uint8).copy()
            for key, value in state.get("boundary_observations", {}).items()
        }
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])


def random_shift(
    observations: torch.Tensor,
    *,
    pad: int = 4,
    seed: int | None = None,
) -> torch.Tensor:
    """Apply DrQ's random integer translation independently per batch item."""

    if observations.ndim != 4:
        raise ValueError("random_shift expects BCHW observations")
    if pad < 0:
        raise ValueError("padding must be non-negative")
    if pad == 0:
        return observations.clone()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=observations.device)
        generator.manual_seed(int(seed))
    padded = F.pad(observations, (pad, pad, pad, pad), mode="replicate")
    height, width = observations.shape[-2:]
    result = torch.empty_like(observations)
    for index in range(observations.shape[0]):
        top = int(torch.randint(0, 2 * pad + 1, (), generator=generator, device=observations.device))
        left = int(torch.randint(0, 2 * pad + 1, (), generator=generator, device=observations.device))
        result[index] = padded[index, :, top : top + height, left : left + width]
    return result


def _normalize_tensor(observation: torch.Tensor) -> torch.Tensor:
    value = observation.float()
    if observation.dtype == torch.uint8:
        value = value / 255.0
    return value - 0.5


class DrQEncoder(nn.Module):
    def __init__(self, input_channels: int = 4, feature_dim: int = 256):
        super().__init__()
        self.input_channels = int(input_channels)
        self.feature_dim = int(feature_dim)
        self.convolution = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            flattened = int(self.convolution(torch.zeros(1, input_channels, 84, 84)).numel())
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.linear(self.convolution(_normalize_tensor(observation)))


class DrQActor(nn.Module):
    def __init__(self, input_channels: int = 4, action_dim: int = 3, feature_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.encoder = DrQEncoder(input_channels, feature_dim)
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.policy(self.trunk(self.encoder(observation))))

    @torch.inference_mode()
    def act(self, observation: np.ndarray, deterministic: bool = True, noise_std: float = 0.0) -> np.ndarray:
        value = np.asarray(observation)
        tensor = torch.as_tensor(value).unsqueeze(0)
        action = self(tensor).squeeze(0).cpu().numpy()
        if not deterministic and noise_std > 0.0:
            action = action + np.random.default_rng().normal(0.0, noise_std, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


class DrQCritic(nn.Module):
    def __init__(self, input_channels: int = 4, action_dim: int = 3, feature_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.encoder = DrQEncoder(input_channels, feature_dim)
        self.q = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat((self.encoder(observation), action), dim=-1)).squeeze(-1)


@dataclass
class DrQv2Config:
    observation_shape: tuple[int, int, int] = (4, 84, 84)
    action_dim: int = 3
    feature_dim: int = 256
    hidden_dim: int = 256
    replay_capacity: int = 100_000
    batch_size: int = 256
    warmup_steps: int = 1_000
    n_step: int = 3
    gamma: float = 0.99
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4
    tau: float = 0.01
    actor_update_frequency: int = 2
    target_update_frequency: int = 2
    augmentation_pad: int = 4
    exploration_initial_std: float = 0.2
    exploration_final_std: float = 0.05
    exploration_duration: int = 100_000
    target_policy_noise: float = 0.2
    target_policy_noise_clip: float = 0.5
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.observation_shape = tuple(self.observation_shape)
        if tuple(self.observation_shape) != (4, 84, 84):
            raise ValueError("DrQ-v2 is frozen to four 84x84 channels")
        if self.action_dim != 3:
            raise ValueError("DrQ-v2 is frozen to three continuous actions")
        if self.batch_size <= 0 or self.n_step <= 0 or self.replay_capacity < 2:
            raise ValueError("invalid replay/update sizes")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("gamma and tau must be in (0, 1]")


class DrQv2Agent:
    def __init__(
        self,
        config: DrQv2Config | None = None,
        *,
        seed: int = 0,
        action_spec: ActionSpec | None = None,
        observation_spec: ObservationSpec | None = None,
    ):
        self.config = config or DrQv2Config()
        self.observation_spec = observation_spec or ObservationSpec()
        if self.observation_spec.channel_order != "CHW" or self.observation_spec.control_plane_fingerprint is not None:
            raise ValueError("DrQ-v2 requires the frozen CHW observation without a control plane")
        self.action_adapter = ActionAdapter(action_spec or ActionSpec())
        self.device = torch.device(self.config.device)
        if self.device.type != "cpu" and not torch.cuda.is_available():
            raise ValueError("requested DrQ device is unavailable")
        torch.manual_seed(int(seed))
        self.rng = np.random.default_rng(seed)
        random.seed(int(seed))
        self.actor = DrQActor(
            self.config.observation_shape[0],
            self.config.action_dim,
            self.config.feature_dim,
            self.config.hidden_dim,
        ).to(self.device)
        self.critic_one = DrQCritic(
            self.config.observation_shape[0],
            self.config.action_dim,
            self.config.feature_dim,
            self.config.hidden_dim,
        ).to(self.device)
        self.critic_two = DrQCritic(
            self.config.observation_shape[0],
            self.config.action_dim,
            self.config.feature_dim,
            self.config.hidden_dim,
        ).to(self.device)
        self.target_one = copy.deepcopy(self.critic_one).to(self.device)
        self.target_two = copy.deepcopy(self.critic_two).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_one.parameters()) + list(self.critic_two.parameters()),
            lr=self.config.critic_learning_rate,
        )
        self.replay = Uint8Replay(
            self.config.replay_capacity,
            observation_spec=self.observation_spec,
            action_dim=self.config.action_dim,
            n_step=self.config.n_step,
            gamma=self.config.gamma,
            seed=seed,
        )
        self.environment_steps = 0
        self.gradient_steps = 0
        self._hard_update_targets()

    def _hard_update_targets(self) -> None:
        self.target_one.load_state_dict(self.critic_one.state_dict())
        self.target_two.load_state_dict(self.critic_two.state_dict())

    def _soft_update_targets(self) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for target, source in zip(self.target_one.parameters(), self.critic_one.parameters()):
                target.mul_(1.0 - tau).add_(source, alpha=tau)
            for target, source in zip(self.target_two.parameters(), self.critic_two.parameters()):
                target.mul_(1.0 - tau).add_(source, alpha=tau)

    def observe(self, transition: Transition) -> None:
        self.replay.add(transition)
        self.environment_steps += 1

    def exploration_std(self) -> float:
        fraction = min(1.0, self.environment_steps / max(1, self.config.exploration_duration))
        return float(
            self.config.exploration_initial_std
            + fraction * (self.config.exploration_final_std - self.config.exploration_initial_std)
        )

    @torch.inference_mode()
    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        validated = self.observation_spec.validate(observation)
        tensor = torch.as_tensor(validated, device=self.device).unsqueeze(0)
        action = self.actor(tensor).squeeze(0).cpu().numpy()
        if not deterministic:
            action = action + self.rng.normal(0.0, self.exploration_std(), size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _batch_tensors(self, batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            "observation": torch.as_tensor(batch["observation"], device=self.device),
            "next_observation": torch.as_tensor(batch["next_observation"], device=self.device),
            "action": torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device),
            "reward": torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device),
            "discount": torch.as_tensor(batch["discount"], dtype=torch.float32, device=self.device),
        }

    def update(self) -> dict[str, float]:
        zero = {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "q1_mean": 0.0,
            "q2_mean": 0.0,
            "target_mean": 0.0,
            "replay_size": float(self.replay.size),
            "gradient_steps": float(self.gradient_steps),
        }
        if self.replay.size < max(self.config.batch_size, self.config.warmup_steps):
            return zero
        try:
            batch = self.replay.sample(self.config.batch_size)
        except ValueError:
            return zero
        values = self._batch_tensors(batch)
        observation = random_shift(values["observation"], pad=self.config.augmentation_pad)
        next_observation = random_shift(values["next_observation"], pad=self.config.augmentation_pad)
        with torch.no_grad():
            next_action = self.actor(next_observation)
            noise = torch.randn_like(next_action) * self.config.target_policy_noise
            noise = noise.clamp(-self.config.target_policy_noise_clip, self.config.target_policy_noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            target_q = torch.minimum(
                self.target_one(next_observation, next_action),
                self.target_two(next_observation, next_action),
            )
            target = values["reward"] + values["discount"] * target_q
        q1 = self.critic_one(observation, values["action"])
        q2 = self.critic_two(observation, values["action"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_one.parameters()) + list(self.critic_two.parameters()), 10.0
        )
        self.critic_optimizer.step()
        self.gradient_steps += 1
        actor_loss = torch.zeros((), device=self.device)
        if self.gradient_steps % self.config.actor_update_frequency == 0:
            for parameter in list(self.critic_one.parameters()) + list(self.critic_two.parameters()):
                parameter.requires_grad_(False)
            actor_observation = random_shift(values["observation"], pad=self.config.augmentation_pad)
            actor_loss = -self.critic_one(actor_observation, self.actor(actor_observation)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()
            for parameter in list(self.critic_one.parameters()) + list(self.critic_two.parameters()):
                parameter.requires_grad_(True)
        if self.gradient_steps % self.config.target_update_frequency == 0:
            self._soft_update_targets()
        metrics = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "q1_mean": float(q1.detach().mean().cpu()),
            "q2_mean": float(q2.detach().mean().cpu()),
            "target_mean": float(target.detach().mean().cpu()),
            "replay_size": float(self.replay.size),
            "gradient_steps": float(self.gradient_steps),
        }
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"non-finite DrQ-v2 update: {metrics}")
        return metrics

    def _payload(self) -> dict[str, Any]:
        return {
            "format": "haic-drq-v2-checkpoint-v1",
            "config": asdict(self.config),
            "observation_spec": asdict(self.observation_spec),
            "action_spec": asdict(self.action_adapter.spec),
            "actor": self.actor.state_dict(),
            "critic_one": self.critic_one.state_dict(),
            "critic_two": self.critic_two.state_dict(),
            "target_one": self.target_one.state_dict(),
            "target_two": self.target_two.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
            "numpy_rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": (
                torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
            ),
            "python_rng_state": random.getstate(),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        source_paths: Iterable[str | Path] = (),
        run_metadata: dict[str, Any] | None = None,
        trainer_state: dict[str, Any] | None = None,
    ) -> Path:
        """Save learner state plus opaque trainer data, not a live simulator snapshot."""
        path = Path(path)
        manifest = build_checkpoint_manifest(
            algorithm="drq-v2",
            reward_contract=(run_metadata or {}).get("reward_contract", {}),
            frame_skip=self.action_adapter.spec.frame_skip,
            max_steps=int((run_metadata or {}).get("max_steps", 0)),
            seeds=(run_metadata or {}).get("seeds", []),
            model_path=path,
            optimizer_path=path,
            replay_path=path,
            rng_path=path,
            source_paths=source_paths,
            action_spec=self.action_adapter.spec,
            observation_spec=self.observation_spec,
            extra={"environment_steps": self.environment_steps, **(run_metadata or {})},
        )
        payload = self._payload()
        payload["manifest"] = manifest
        if trainer_state is not None:
            payload["trainer_state"] = copy.deepcopy(trainer_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        write_checkpoint_manifest(path.with_suffix(".manifest.json"), manifest)
        return path

    def load_checkpoint(self, path: str | Path) -> dict[str, Any] | None:
        """Restore the learner and return trainer data without restoring an environment."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "haic-drq-v2-checkpoint-v1":
            raise ValueError("unsupported DrQ-v2 checkpoint format")
        saved_config = asdict(DrQv2Config(**payload["config"]))
        current_config = asdict(self.config)
        mismatches = [
            key for key in current_config
            if key != "device" and saved_config[key] != current_config[key]
        ]
        if mismatches:
            raise ValueError(f"checkpoint configuration does not match agent: {', '.join(mismatches)}")
        saved_observation = ObservationSpec(**payload["observation_spec"])
        saved_action = ActionSpec(**payload["action_spec"])
        if saved_observation.fingerprint != self.observation_spec.fingerprint:
            raise ValueError("checkpoint observation specification does not match agent")
        if saved_action.fingerprint != self.action_adapter.spec.fingerprint:
            raise ValueError("checkpoint action specification does not match agent")
        manifest = payload.get("manifest")
        if manifest is None and Path(path).with_suffix(".manifest.json").is_file():
            manifest = json.loads(Path(path).with_suffix(".manifest.json").read_text())
        if manifest is not None:
            expected = build_checkpoint_manifest(
                algorithm="drq-v2", reward_contract={}, frame_skip=saved_action.frame_skip,
                max_steps=0, seeds=[], dependency_lockfile=None,
                observation_spec=saved_observation, action_spec=saved_action,
            )
            for key in ("schema_version", "algorithm", "observation", "action", "frame_skip"):
                if manifest.get(key) != expected[key]:
                    raise ValueError(f"checkpoint manifest {key} does not match the saved contract")
        saved_device = torch.device(saved_config["device"])
        cuda_rng_state = payload.get("torch_cuda_rng_state")
        if saved_device.type == "cuda" and cuda_rng_state is None:
            warnings.warn(
                "Legacy CUDA checkpoint has no CUDA RNG state; exact stochastic continuation is unavailable",
                RuntimeWarning, stacklevel=2,
            )
        elif saved_device.type != self.device.type:
            warnings.warn(
                "Checkpoint device type changed; exact stochastic continuation is not guaranteed",
                RuntimeWarning, stacklevel=2,
            )
        self.actor.load_state_dict(payload["actor"])
        self.critic_one.load_state_dict(payload["critic_one"])
        self.critic_two.load_state_dict(payload["critic_two"])
        self.target_one.load_state_dict(payload["target_one"])
        self.target_two.load_state_dict(payload["target_two"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.replay.load_state_dict(payload["replay"])
        self.environment_steps = int(payload["environment_steps"])
        self.gradient_steps = int(payload["gradient_steps"])
        self.rng.bit_generator.state = copy.deepcopy(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if self.device.type == "cuda" and cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state.cpu(), device=self.device)
        random.setstate(payload["python_rng_state"])
        return copy.deepcopy(payload.get("trainer_state"))

    def export_actor(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "haic-drq-v2-actor-v1",
                "config": asdict(self.config),
                "observation_spec": asdict(self.observation_spec),
                "action_spec": asdict(self.action_adapter.spec),
                "state_dict": self.actor.state_dict(),
            },
            path,
        )
        return path


def load_exported_actor(path: str | Path, *, device: str = "cpu") -> tuple[DrQActor, ActionAdapter, ObservationSpec]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "haic-drq-v2-actor-v1":
        raise ValueError("unsupported DrQ-v2 actor export")
    config = DrQv2Config(**payload["config"])
    observation_spec = ObservationSpec(**payload["observation_spec"])
    action_spec = ActionSpec(**payload["action_spec"])
    # Initialization is overwritten by saved weights and must not advance training RNG.
    with torch.random.fork_rng(devices=[]):
        actor = DrQActor(
            config.observation_shape[0],
            config.action_dim,
            config.feature_dim,
            config.hidden_dim,
        )
    actor.load_state_dict(payload["state_dict"])
    actor.to(device)
    actor.eval()
    return actor, ActionAdapter(action_spec), observation_spec
