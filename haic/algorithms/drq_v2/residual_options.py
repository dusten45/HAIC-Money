"""Model-free short-horizon residual options over a frozen DrQ-v2 actor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ResidualOption(IntEnum):
    KEEP = 0
    STEER_MINUS = 1
    STEER_PLUS = 2
    COAST = 3
    BRAKE = 4


OPTION_COUNT = len(ResidualOption)
_ACTION_DIM = 3


def _native_action(values, *, name: str = "base_action") -> np.ndarray:
    action = np.asarray(values, dtype=np.float32)
    if action.shape != (_ACTION_DIM,) or not np.isfinite(action).all():
        raise ValueError(f"{name} must be a finite shape-(3,) action")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ValueError(f"{name} must be inside native [-1, 1] bounds")
    return action


def apply_native_residual(
    base_action,
    option: ResidualOption | int,
    *,
    steering_delta: float = 0.15,
    brake_floor_official: float = 0.25,
) -> np.ndarray:
    """Apply one fixed option to a symmetric native ``[steer, gas, brake]`` action.

    Native gas and brake values in ``[-1, 1]`` map to official ``[0, 1]``.
    ``KEEP`` is an identity residual, not a zero native action.
    """
    action = _native_action(base_action)
    try:
        option = ResidualOption(option)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown residual option: {option!r}") from exc
    if not math.isfinite(steering_delta) or not 0.0 <= steering_delta <= 2.0:
        raise ValueError("steering_delta must be finite and in [0, 2]")
    if not math.isfinite(brake_floor_official) or not 0.0 <= brake_floor_official <= 1.0:
        raise ValueError("brake_floor_official must be finite and in [0, 1]")

    result = action.copy()
    if option is ResidualOption.STEER_MINUS:
        result[0] = np.clip(result[0] - steering_delta, -1.0, 1.0)
    elif option is ResidualOption.STEER_PLUS:
        result[0] = np.clip(result[0] + steering_delta, -1.0, 1.0)
    elif option is ResidualOption.COAST:
        result[1:] = -1.0
    elif option is ResidualOption.BRAKE:
        result[1] = -1.0
        native_brake_floor = 2.0 * brake_floor_official - 1.0
        result[2] = max(result[2], native_brake_floor)
    return result.astype(np.float32, copy=False)


def select_greedy_option(values, *, intervention_margin: float = 0.0) -> ResidualOption:
    """Select a non-KEEP option only when it beats KEEP by a fixed margin."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    q_values = np.asarray(values, dtype=np.float64)
    if q_values.shape != (OPTION_COUNT,) or not np.isfinite(q_values).all():
        raise ValueError(f"q_values must be a finite shape-({OPTION_COUNT},) vector")
    if not math.isfinite(intervention_margin) or intervention_margin < 0.0:
        raise ValueError("intervention_margin must be finite and non-negative")
    best_intervention = int(np.argmax(q_values[1:])) + 1
    if q_values[best_intervention] - q_values[ResidualOption.KEEP] <= intervention_margin:
        return ResidualOption.KEEP
    return ResidualOption(best_intervention)


class ResidualOptionQ(nn.Module):
    """Small discrete Q head over frozen DrQ encoder features and base action."""

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim + _ACTION_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, OPTION_COUNT),
        )

    def forward(self, features: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        if features.ndim not in (1, 2) or features.shape[-1] != self.feature_dim:
            raise ValueError("features must have shape (feature_dim,) or (batch, feature_dim)")
        if base_action.shape != (*features.shape[:-1], _ACTION_DIM):
            raise ValueError("base_action must match feature batch dimensions and have 3 values")
        return self.network(torch.cat((features, base_action), dim=-1))


class ResidualOptionState:
    """Inference-time fixed-duration hold state; caller recomputes base actions."""

    def __init__(
        self,
        duration: int = 2,
        *,
        intervention_margin: float = 0.0,
        steering_delta: float = 0.15,
        brake_floor_official: float = 0.25,
    ):
        if type(duration) is not int or duration <= 0:
            raise ValueError("duration must be a positive integer")
        if not math.isfinite(intervention_margin) or intervention_margin < 0.0:
            raise ValueError("intervention_margin must be finite and non-negative")
        if not math.isfinite(steering_delta) or not 0.0 <= steering_delta <= 2.0:
            raise ValueError("steering_delta must be finite and in [0, 2]")
        if not math.isfinite(brake_floor_official) or not 0.0 <= brake_floor_official <= 1.0:
            raise ValueError("brake_floor_official must be finite and in [0, 1]")
        self.duration = duration
        self.intervention_margin = float(intervention_margin)
        self.steering_delta = float(steering_delta)
        self.brake_floor_official = float(brake_floor_official)
        self.reset()

    @property
    def at_boundary(self) -> bool:
        return self.remaining == 0

    def reset(self) -> None:
        self.option: ResidualOption | None = None
        self.remaining = 0

    def act(self, base_action, q_values=None) -> tuple[np.ndarray, ResidualOption]:
        """Apply one decision, selecting an option only at an option boundary."""
        if self.at_boundary:
            if q_values is None:
                raise ValueError("q_values are required at an option boundary")
            self.option = select_greedy_option(
                q_values,
                intervention_margin=self.intervention_margin,
            )
            self.remaining = self.duration
        assert self.option is not None
        action = apply_native_residual(
            base_action,
            self.option,
            steering_delta=self.steering_delta,
            brake_floor_official=self.brake_floor_official,
        )
        selected = self.option
        self.remaining -= 1
        if self.at_boundary:
            self.option = None
        return action, selected


def _feature_vector(values, *, name: str) -> np.ndarray:
    feature = np.asarray(values, dtype=np.float32)
    if feature.ndim != 1 or feature.size == 0 or not np.isfinite(feature).all():
        raise ValueError(f"{name} must be a finite non-empty feature vector")
    return feature.copy()


@dataclass(frozen=True)
class OptionTransition:
    """One option-boundary transition with actual duration and bootstrap flags."""

    features: np.ndarray
    base_action: np.ndarray
    option: ResidualOption | int
    reward: float
    duration: int
    next_features: np.ndarray | None
    next_base_action: np.ndarray | None
    terminated: bool
    truncated: bool
    terminal: bool

    def __post_init__(self) -> None:
        features = _feature_vector(self.features, name="features")
        base_action = _native_action(self.base_action, name="base_action").copy()
        try:
            option = ResidualOption(self.option)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown residual option: {self.option!r}") from exc
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")
        if type(self.duration) is not int or self.duration <= 0:
            raise ValueError("duration must be a positive integer")
        terminated = bool(self.terminated)
        truncated = bool(self.truncated)
        terminal = bool(self.terminal)
        if terminated and not terminal:
            raise ValueError("terminated transitions must be terminal")
        if terminal and not (terminated or truncated):
            raise ValueError("terminal transitions must end an episode")
        next_features = self.next_features
        next_base_action = self.next_base_action
        if terminal:
            next_features = None
            next_base_action = None
        else:
            if next_features is None or next_base_action is None:
                raise ValueError("bootstrappable option transitions require final-state features and action")
            next_features = _feature_vector(next_features, name="next_features")
            if next_features.shape != features.shape:
                raise ValueError("next_features shape must match features")
            next_base_action = _native_action(next_base_action, name="next_base_action").copy()
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "base_action", base_action)
        object.__setattr__(self, "option", option)
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "terminated", terminated)
        object.__setattr__(self, "truncated", truncated)
        object.__setattr__(self, "terminal", terminal)
        object.__setattr__(self, "next_features", next_features)
        object.__setattr__(self, "next_base_action", next_base_action)


class OptionTransitionAccumulator:
    """Aggregate raw rewards until option duration or episode boundary."""

    def __init__(
        self,
        features,
        base_action,
        option: ResidualOption | int,
        *,
        gamma: float = 0.99,
        max_duration: int = 2,
    ):
        if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be finite and in (0, 1]")
        if type(max_duration) is not int or max_duration <= 0:
            raise ValueError("max_duration must be a positive integer")
        try:
            self.option = ResidualOption(option)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown residual option: {option!r}") from exc
        self.features = _feature_vector(features, name="features")
        self.base_action = _native_action(base_action, name="base_action").copy()
        self.gamma = float(gamma)
        self.max_duration = max_duration
        self.reward = 0.0
        self.duration = 0
        self._finished = False

    def append(
        self,
        reward: float,
        *,
        next_features=None,
        next_base_action=None,
        terminated: bool = False,
        truncated: bool = False,
        terminal: bool = False,
    ) -> OptionTransition | None:
        if self._finished:
            raise RuntimeError("option transition is already complete")
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        terminated = bool(terminated)
        truncated = bool(truncated)
        terminal = bool(terminal or terminated)
        if terminal and not (terminated or truncated):
            raise ValueError("terminal transitions must end an episode")
        self.reward += (self.gamma ** self.duration) * float(reward)
        self.duration += 1
        complete = terminated or truncated or self.duration == self.max_duration
        if not complete:
            return None
        self._finished = True
        return OptionTransition(
            features=self.features,
            base_action=self.base_action,
            option=self.option,
            reward=self.reward,
            duration=self.duration,
            next_features=next_features,
            next_base_action=next_base_action,
            terminated=terminated,
            truncated=truncated,
            terminal=terminal,
        )


class FeatureOptionReplay:
    """Ring replay for frozen-encoder features, bound to one actor hash."""

    def __init__(self, capacity: int, feature_dim: int, frozen_actor_sha256: str, *, seed: int = 0):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if type(feature_dim) is not int or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        if not re.fullmatch(r"[0-9a-f]{64}", frozen_actor_sha256):
            raise ValueError("frozen_actor_sha256 must be a lowercase SHA-256 digest")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.frozen_actor_sha256 = frozen_actor_sha256
        self.features = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.base_actions = np.zeros((capacity, _ACTION_DIM), dtype=np.float32)
        self.options = np.zeros(capacity, dtype=np.uint8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.durations = np.zeros(capacity, dtype=np.int32)
        self.next_features = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.next_base_actions = np.zeros((capacity, _ACTION_DIM), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool_)
        self.truncated = np.zeros(capacity, dtype=np.bool_)
        self.terminal = np.zeros(capacity, dtype=np.bool_)
        self.cursor = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, transition: OptionTransition) -> None:
        if not isinstance(transition, OptionTransition):
            raise TypeError("transition must be an OptionTransition")
        if transition.features.shape != (self.feature_dim,):
            raise ValueError("transition feature dimension does not match replay")
        index = self.cursor
        self.features[index] = transition.features
        self.base_actions[index] = transition.base_action
        self.options[index] = int(transition.option)
        self.rewards[index] = transition.reward
        self.durations[index] = transition.duration
        if not transition.terminal:
            self.next_features[index] = transition.next_features
            self.next_base_actions[index] = transition.next_base_action
        else:
            self.next_features[index].fill(0.0)
            self.next_base_actions[index].fill(0.0)
        self.terminated[index] = transition.terminated
        self.truncated[index] = transition.truncated
        self.terminal[index] = transition.terminal
        self.cursor = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray | str]:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.size == 0:
            raise ValueError("cannot sample an empty option replay")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {
            "features": self.features[indices].copy(),
            "base_action": self.base_actions[indices].copy(),
            "option": self.options[indices].astype(np.int64),
            "reward": self.rewards[indices].copy(),
            "duration": self.durations[indices].copy(),
            "next_features": self.next_features[indices].copy(),
            "next_base_action": self.next_base_actions[indices].copy(),
            "terminated": self.terminated[indices].copy(),
            "truncated": self.truncated[indices].copy(),
            "terminal": self.terminal[indices].copy(),
            "frozen_actor_sha256": self.frozen_actor_sha256,
        }

    def state_dict(self) -> dict:
        """Return a complete replay snapshot, including sample RNG state."""
        return {
            "capacity": self.capacity,
            "feature_dim": self.feature_dim,
            "frozen_actor_sha256": self.frozen_actor_sha256,
            "cursor": self.cursor,
            "size": self.size,
            "rng_state": self.rng.bit_generator.state,
            **{
                name: getattr(self, name).copy()
                for name in (
                    "features", "base_actions", "options", "rewards", "durations",
                    "next_features", "next_base_actions", "terminated", "truncated", "terminal",
                )
            },
        }

    def load_state_dict(self, state: dict) -> None:
        if (
            state.get("capacity") != self.capacity
            or state.get("feature_dim") != self.feature_dim
            or state.get("frozen_actor_sha256") != self.frozen_actor_sha256
        ):
            raise ValueError("option replay snapshot contract does not match")
        size, cursor = state.get("size"), state.get("cursor")
        if type(size) is not int or not 0 <= size <= self.capacity:
            raise ValueError("invalid option replay size")
        if type(cursor) is not int or not 0 <= cursor < self.capacity:
            raise ValueError("invalid option replay cursor")
        names = (
            "features", "base_actions", "options", "rewards", "durations",
            "next_features", "next_base_actions", "terminated", "truncated", "terminal",
        )
        arrays = {}
        for name in names:
            expected = getattr(self, name)
            value = np.asarray(state.get(name))
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"invalid option replay array: {name}")
            if value.dtype.kind in "fiu" and not np.isfinite(value).all():
                raise ValueError(f"non-finite option replay array: {name}")
            arrays[name] = value
        if size < self.capacity and cursor != size:
            raise ValueError("invalid option replay cursor for a partially filled buffer")
        if size and (
            np.any(arrays["options"][:size] >= OPTION_COUNT)
            or np.any(arrays["durations"][:size] <= 0)
        ):
            raise ValueError("invalid option replay transition metadata")
        rng = np.random.default_rng()
        try:
            rng.bit_generator.state = state["rng_state"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid option replay RNG state") from exc
        for name, value in arrays.items():
            getattr(self, name)[:] = value
        self.cursor = cursor
        self.size = size
        self.rng = rng


def train_option_q_step(
    online: ResidualOptionQ,
    target: ResidualOptionQ,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, np.ndarray | str | torch.Tensor],
    *,
    frozen_actor_sha256: str,
    gamma: float = 0.99,
    max_grad_norm: float = 10.0,
) -> dict[str, float]:
    """Apply one Double-Q regression step to an option-transition batch."""
    if not re.fullmatch(r"[0-9a-f]{64}", frozen_actor_sha256):
        raise ValueError("frozen_actor_sha256 must be a lowercase SHA-256 digest")
    if batch.get("frozen_actor_sha256") != frozen_actor_sha256:
        raise ValueError("option replay was generated by a different frozen actor")
    if online.feature_dim != target.feature_dim:
        raise ValueError("online and target option heads must share feature dimensions")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be finite and positive")
    device = next(online.parameters()).device

    def tensor(name, *, dtype):
        return torch.as_tensor(batch[name], dtype=dtype, device=device)

    features = tensor("features", dtype=torch.float32)
    base_action = tensor("base_action", dtype=torch.float32)
    option = tensor("option", dtype=torch.int64)
    reward = tensor("reward", dtype=torch.float32)
    duration = tensor("duration", dtype=torch.int64)
    next_features = tensor("next_features", dtype=torch.float32)
    next_base_action = tensor("next_base_action", dtype=torch.float32)
    terminal = tensor("terminal", dtype=torch.bool)
    if features.ndim != 2:
        raise ValueError("features must be batched")
    batch_size = features.shape[0]
    if option.shape != (batch_size,) or torch.any(option < 0) or torch.any(option >= OPTION_COUNT):
        raise ValueError("option indices must be valid and match the batch")
    with torch.no_grad():
        online_next_q = online(next_features, next_base_action)
        target_next_q = target(next_features, next_base_action)
        targets = double_q_targets(reward, duration, terminal, online_next_q, target_next_q, gamma=gamma)
    selected_q = online(features, base_action).gather(1, option.unsqueeze(1)).squeeze(1)
    loss = F.mse_loss(selected_q, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("option Q loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(online.parameters(), max_grad_norm)
    if not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("option Q gradient norm is non-finite")
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "q_mean": float(selected_q.detach().mean().cpu()),
        "target_mean": float(targets.mean().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
    }


@torch.no_grad()
def polyak_update_option_target(target: ResidualOptionQ, online: ResidualOptionQ, tau: float) -> None:
    if target.feature_dim != online.feature_dim:
        raise ValueError("online and target option heads must share feature dimensions")
    if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be finite and in [0, 1]")
    for target_parameter, online_parameter in zip(target.parameters(), online.parameters(), strict=True):
        target_parameter.lerp_(online_parameter, tau)


def double_q_targets(
    reward: torch.Tensor,
    duration: torch.Tensor,
    terminal: torch.Tensor,
    online_next_q: torch.Tensor,
    target_next_q: torch.Tensor,
    *,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Build detached Double-Q targets with option-duration discounting."""
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be finite and in (0, 1]")
    if online_next_q.ndim != 2 or target_next_q.shape != online_next_q.shape:
        raise ValueError("online and target next Q values must share shape (batch, options)")
    if online_next_q.shape[1] != OPTION_COUNT:
        raise ValueError(f"next Q values must have {OPTION_COUNT} options")
    device = online_next_q.device
    reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
    raw_duration = torch.as_tensor(duration, device=device)
    if raw_duration.dtype.is_floating_point:
        if not torch.isfinite(raw_duration).all() or torch.any(raw_duration != raw_duration.round()):
            raise ValueError("durations must be finite integers")
    duration = raw_duration.to(dtype=torch.int64)
    terminal = torch.as_tensor(terminal, dtype=torch.bool, device=device)
    batch_size = online_next_q.shape[0]
    if reward.shape != (batch_size,) or duration.shape != (batch_size,) or terminal.shape != (batch_size,):
        raise ValueError("reward, duration and terminal must have shape (batch,)")
    if torch.any(duration <= 0):
        raise ValueError("durations must be positive")
    if not torch.isfinite(reward).all() or not torch.isfinite(online_next_q).all() or not torch.isfinite(target_next_q).all():
        raise ValueError("rewards and next Q values must be finite")
    with torch.no_grad():
        greedy = online_next_q.argmax(dim=-1, keepdim=True)
        next_value = target_next_q.gather(-1, greedy).squeeze(-1)
        discount = torch.pow(torch.as_tensor(gamma, device=device), duration)
        discount = torch.where(terminal, torch.zeros_like(discount), discount)
        return reward + discount * next_value
