"""Algorithm-neutral contracts shared by native HAIC training stacks.

The module intentionally contains no Stable-Baselines3 or environment-specific
training code.  DrQ-v2 consumes the same observation, action, transition, and
checkpoint contracts that a future recurrent/world-model trainer can use.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class ObservationSpec:
    shape: tuple[int, int, int] = (4, 84, 84)
    dtype: str = "float32"
    channel_order: str = "CHW"
    low: float = 0.0
    high: float = 1.0
    uint8_scale: int = 255
    control_plane_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if tuple(self.shape) != (4, 84, 84):
            raise ValueError("HAIC observations must have shape (4, 84, 84)")
        if self.channel_order not in {"CHW", "HWC"}:
            raise ValueError("channel_order must be CHW or HWC")
        if np.dtype(self.dtype) != np.dtype(np.float32):
            raise ValueError("the frozen observation contract uses float32")
        if not np.isfinite([self.low, self.high]).all() or self.low >= self.high:
            raise ValueError("observation bounds must be finite and ordered")
        if int(self.uint8_scale) != 255:
            raise ValueError("uint8_scale must remain 255 for the HAIC contract")

    @property
    def fingerprint(self) -> str:
        payload = {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "channel_order": self.channel_order,
            "low": self.low,
            "high": self.high,
            "uint8_scale": self.uint8_scale,
            "control_plane_fingerprint": self.control_plane_fingerprint,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def _as_chw(
        self,
        observation: np.ndarray,
        source_channel_order: str | None = None,
    ) -> np.ndarray:
        array = np.asarray(observation)
        source_channel_order = source_channel_order or self.channel_order
        if source_channel_order == "HWC":
            if array.shape != (self.shape[1], self.shape[2], self.shape[0]):
                raise ValueError("HWC observation has the wrong shape")
            array = np.transpose(array, (2, 0, 1))
        elif source_channel_order != "CHW":
            raise ValueError("source_channel_order must be CHW or HWC")
        if array.shape != self.shape:
            raise ValueError(f"observation must have shape {self.shape}")
        return array

    def validate(
        self,
        observation: np.ndarray,
        *,
        source_channel_order: str | None = None,
    ) -> np.ndarray:
        array = self._as_chw(observation, source_channel_order)
        if np.dtype(array.dtype) != np.dtype(self.dtype):
            raise ValueError(f"observation must have dtype {self.dtype}")
        if not np.isfinite(array).all() or np.any(array < self.low) or np.any(array > self.high):
            raise ValueError("observation contains values outside the frozen range")
        return np.ascontiguousarray(array)

    def to_uint8(
        self,
        observation: np.ndarray,
        *,
        source_channel_order: str | None = None,
    ) -> np.ndarray:
        array = self.validate(observation, source_channel_order=source_channel_order)
        return np.rint((array - self.low) / (self.high - self.low) * self.uint8_scale).astype(
            np.uint8,
            copy=False,
        )

    def from_uint8(self, observation: np.ndarray) -> np.ndarray:
        array = np.asarray(observation)
        if array.shape != self.shape or array.dtype != np.uint8:
            raise ValueError(f"uint8 observation must have shape {self.shape}")
        return np.ascontiguousarray(
            self.low + array.astype(np.float32) / self.uint8_scale * (self.high - self.low)
        )


@dataclass(frozen=True)
class ActionSpec:
    native_low: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    native_high: tuple[float, float, float] = (1.0, 1.0, 1.0)
    official_low: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    official_high: tuple[float, float, float] = (1.0, 1.0, 1.0)
    frame_skip: int = 4
    order: tuple[str, str, str] = ("steer", "gas", "brake")
    method: str = "symmetric-native-to-haic-box"

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["native_low"] = list(self.native_low)
        payload["native_high"] = list(self.native_high)
        payload["official_low"] = list(self.official_low)
        payload["official_high"] = list(self.official_high)
        payload["order"] = list(self.order)
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class ActionAdapter:
    """Map a symmetric algorithm action to the official HAIC Box action."""

    def __init__(self, spec: ActionSpec | None = None):
        self.spec = spec or ActionSpec()
        self._native_low = np.asarray(self.spec.native_low, dtype=np.float32)
        self._native_high = np.asarray(self.spec.native_high, dtype=np.float32)
        self._official_low = np.asarray(self.spec.official_low, dtype=np.float32)
        self._official_high = np.asarray(self.spec.official_high, dtype=np.float32)

    @staticmethod
    def _validate(values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError("action must be a finite shape-(3,) vector")
        return array

    def to_official(self, native_action: Any, *, clip: bool = True) -> np.ndarray:
        values = self._validate(native_action)
        if clip:
            values = np.clip(values, self._native_low, self._native_high)
        elif np.any(values < self._native_low) or np.any(values > self._native_high):
            raise ValueError("native action is outside symmetric bounds")
        official = values.copy()
        official[1:] = (values[1:] + 1.0) * 0.5
        return np.clip(official, self._official_low, self._official_high).astype(np.float32)

    def to_native(self, official_action: Any, *, clip: bool = True) -> np.ndarray:
        values = self._validate(official_action)
        if clip:
            values = np.clip(values, self._official_low, self._official_high)
        elif np.any(values < self._official_low) or np.any(values > self._official_high):
            raise ValueError("official action is outside HAIC bounds")
        native = values.copy()
        native[1:] = values[1:] * 2.0 - 1.0
        return np.clip(native, self._native_low, self._native_high).astype(np.float32)


@dataclass
class Transition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    terminal: bool | None = None
    discount: float | None = None
    info: dict[str, Any] = field(default_factory=dict)
    episode_id: int = 0
    step: int = 0
    applied_action: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.observation = np.asarray(self.observation)
        self.next_observation = np.asarray(self.next_observation)
        self.action = np.asarray(self.action, dtype=np.float32)
        if self.applied_action is not None:
            self.applied_action = np.asarray(self.applied_action, dtype=np.float32)
        self.reward = float(self.reward)
        self.terminated = bool(self.terminated)
        self.truncated = bool(self.truncated)
        self.terminal = self.terminated or bool(self.terminal)
        if self.discount is not None:
            self.discount = float(self.discount)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def bootstrap_allowed(self) -> bool:
        return not bool(self.terminal)

    def as_uint8(self, observation_spec: ObservationSpec | None = None) -> "Transition":
        spec = observation_spec or ObservationSpec()
        observation = self.observation
        next_observation = self.next_observation
        if observation.dtype != np.uint8:
            observation = spec.to_uint8(observation)
        if next_observation.dtype != np.uint8:
            next_observation = spec.to_uint8(next_observation)
        return Transition(
            observation=observation,
            action=self.action,
            reward=self.reward,
            next_observation=next_observation,
            terminated=self.terminated,
            truncated=self.truncated,
            terminal=self.terminal,
            discount=self.discount,
            info=dict(self.info),
            episode_id=self.episode_id,
            step=self.step,
            applied_action=self.applied_action,
        )


class EpisodeCollector:
    """Collect exactly one transition per high-level ``env.step`` call."""

    def __init__(
        self,
        env: Any,
        *,
        observation_spec: ObservationSpec | None = None,
        action_adapter: ActionAdapter | None = None,
        gamma: float = 0.99,
    ):
        if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be finite and in (0, 1]")
        self.env = env
        self.observation_spec = observation_spec or ObservationSpec()
        self.action_adapter = action_adapter or ActionAdapter()
        self.gamma = float(gamma)
        self.current_observation: np.ndarray | None = None
        self.current_info: dict[str, Any] = {}
        self.episode_id = -1
        self.step_index = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        observation = self.observation_spec.validate(observation)
        self.current_observation = observation
        self.current_info = dict(info or {})
        self.episode_id += 1
        self.step_index = 0
        return observation.copy(), dict(self.current_info)

    @staticmethod
    def _is_terminal(terminated: bool, truncated: bool, info: dict[str, Any]) -> bool:
        if terminated:
            return True
        if info.get("finished"):
            return True
        return info.get("retire_reason") in {"crash", "off_track", "out_of_bounds"}

    def step(self, native_action: Any) -> Transition:
        if self.current_observation is None:
            raise RuntimeError("reset must be called before step")
        native_action = self.action_adapter._validate(native_action)
        applied_action = self.action_adapter.to_official(native_action)
        native_action = self.action_adapter.to_native(applied_action)
        next_observation, reward, terminated, truncated, info = self.env.step(applied_action)
        next_observation = self.observation_spec.validate(next_observation)
        info = {**self.current_info, **dict(info or {})}
        terminal = self._is_terminal(bool(terminated), bool(truncated), info)
        transition = Transition(
            observation=self.current_observation.copy(),
            action=native_action.copy(),
            reward=reward,
            next_observation=next_observation.copy(),
            terminated=terminated,
            truncated=truncated,
            terminal=terminal,
            discount=self.gamma if not terminal else 0.0,
            info=info,
            episode_id=self.episode_id,
            step=self.step_index,
            applied_action=applied_action.copy(),
        )
        self.step_index += 1
        self.current_observation = None if (terminated or truncated) else next_observation
        self.current_info = info
        return transition


class ReplayAdapter(Protocol):
    def add(self, transition: Transition) -> None: ...

    def sample(self, batch_size: int): ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class PolicyAdapter(Protocol):
    def reset_episode(self) -> None: ...

    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray: ...


class CPUActorAdapter:
    """Bridge a native symmetric actor to official CPU actions."""

    def __init__(
        self,
        actor: Any,
        action_adapter: ActionAdapter | None = None,
        observation_spec: ObservationSpec | None = None,
    ):
        self.actor = actor
        self.action_adapter = action_adapter or ActionAdapter()
        self.observation_spec = observation_spec or ObservationSpec()
        if hasattr(self.actor, "eval"):
            self.actor.eval()

    def reset_episode(self) -> None:
        reset = getattr(self.actor, "reset_episode", None)
        if reset is not None:
            reset()

    def _native_action(self, observation: np.ndarray, deterministic: bool) -> np.ndarray:
        validated = self.observation_spec.validate(observation)
        act = getattr(self.actor, "act", None)
        if act is not None:
            value = act(validated, deterministic=deterministic)
        else:
            import torch

            with torch.inference_mode():
                value = self.actor(torch.as_tensor(validated).unsqueeze(0))
                if isinstance(value, (tuple, list)):
                    value = value[0]
                value = value.squeeze(0).detach().cpu().numpy()
        return self.action_adapter._validate(value)

    def native_action(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self._native_action(observation, deterministic)

    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self.action_adapter.to_official(self._native_action(observation, deterministic))

    def deterministic_trace(self, observations: Iterable[np.ndarray]) -> dict[str, Any]:
        self.reset_episode()
        actions = [self.act(observation, deterministic=True) for observation in observations]
        array = np.asarray(actions, dtype=np.float32)
        return {
            "actions": array.tolist(),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checkpoint_manifest(
    *,
    algorithm: str,
    reward_contract: dict[str, Any],
    frame_skip: int,
    max_steps: int,
    seeds: Iterable[int],
    model_path: str | Path | None = None,
    optimizer_path: str | Path | None = None,
    replay_path: str | Path | None = None,
    rng_path: str | Path | None = None,
    source_paths: Iterable[str | Path] = (),
    dependency_lockfile: str | Path | None = "requirements.txt",
    action_spec: ActionSpec | None = None,
    observation_spec: ObservationSpec | None = None,
    hardware: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observation_spec = observation_spec or ObservationSpec()
    action_spec = action_spec or ActionSpec(frame_skip=frame_skip)
    source_hashes = {}
    for path in source_paths:
        path = Path(path)
        if path.is_file():
            source_hashes[str(path)] = file_sha256(path)
    lock_hash = None
    if dependency_lockfile is not None and Path(dependency_lockfile).is_file():
        lock_hash = file_sha256(dependency_lockfile)
    paths = {
        "model": str(model_path) if model_path is not None else None,
        "optimizer": str(optimizer_path) if optimizer_path is not None else None,
        "replay": str(replay_path) if replay_path is not None else None,
        "rng": str(rng_path) if rng_path is not None else None,
    }
    manifest = {
        "schema_version": 1,
        "algorithm": algorithm,
        "observation": {
            "shape": list(observation_spec.shape),
            "dtype": observation_spec.dtype,
            "channel_order": observation_spec.channel_order,
            "normalization": [observation_spec.low, observation_spec.high],
            "fingerprint": observation_spec.fingerprint,
        },
        "action": {
            "native_low": list(action_spec.native_low),
            "native_high": list(action_spec.native_high),
            "official_low": list(action_spec.official_low),
            "official_high": list(action_spec.official_high),
            "order": list(action_spec.order),
            "frame_skip": action_spec.frame_skip,
            "fingerprint": action_spec.fingerprint,
        },
        "reward_contract": reward_contract,
        "frame_skip": int(frame_skip),
        "max_steps": int(max_steps),
        "seeds": [int(seed) for seed in seeds],
        "paths": paths,
        "source_hashes": source_hashes,
        "dependency_lockfile": {
            "path": str(dependency_lockfile) if dependency_lockfile is not None else None,
            "sha256": lock_hash,
        },
        "hardware": hardware or {"platform": platform.platform()},
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_checkpoint_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
