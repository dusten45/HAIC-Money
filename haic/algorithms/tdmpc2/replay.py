"""Episode-contained online pixel replay for single-task TD-MPC2.

Upstream stores an initial observation with dummy action/reward, then samples
H+1 rows and drops row zero from the transition targets. Here the same layout
is built from H actual transitions, without storing dummy NaNs. The caller
controls seed-step/update timing; replay does not start training on its own.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def random_shift(
    observations: torch.Tensor,
    *,
    pad: int = 3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Upstream ShiftAug: one independent shift per image, shared by its channels.

    Accepts [B,C,S,S] or time-major [T,B,C,S,S], with independent draws at
    each time index. Returns float32 in the input's pixel scale (not /255).
    The upstream pixel encoder already applies this augmentation, so do not
    augment replay samples a second time when using that encoder.
    """
    if not isinstance(observations, torch.Tensor):
        raise TypeError("observations must be a torch.Tensor")
    if type(pad) is not int or pad < 0:
        raise ValueError("pad must be a non-negative integer")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None")
    if observations.ndim not in (4, 5) or observations.shape[-1] != observations.shape[-2]:
        raise ValueError("observations must have shape [B,C,S,S] or [T,B,C,S,S]")
    if not observations.is_floating_point() and observations.dtype != torch.uint8:
        raise TypeError("observations must be uint8 or floating-point pixels")
    shape = observations.shape
    x = observations.reshape(-1, *shape[-3:]).float()
    n, _, height, _ = x.shape
    if pad == 0:
        return x.reshape(shape)
    padded = F.pad(x, (pad,) * 4, mode="replicate")
    eps = 1.0 / (height + 2 * pad)
    coords = torch.linspace(
        -1.0 + eps, 1.0 - eps, height + 2 * pad, device=x.device, dtype=x.dtype
    )[:height]
    horizontal = coords[None, :].expand(height, -1)
    vertical = horizontal.T
    grid = torch.stack((horizontal, vertical), dim=-1)[None].expand(n, -1, -1, -1)
    offsets = torch.randint(
        0, 2 * pad + 1, (n, 1, 1, 2), device=x.device, generator=generator
    ).to(x.dtype)
    grid = grid + offsets * (2.0 / (height + 2 * pad))
    return F.grid_sample(padded, grid, padding_mode="zeros", align_corners=False).reshape(shape)


@dataclass
class _Episode:
    episode_id: int
    observations: list[np.ndarray]
    start_step: int = 0
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    terminal: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def trim_prefix(self, steps: int) -> None:
        del self.observations[:steps]
        for values in (self.actions, self.rewards, self.terminated, self.truncated, self.terminal):
            del values[:steps]
        self.start_step += steps


class EpisodeReplay:
    """Bounded replay of entire reset-separated episodes and an in-flight episode.

    Call start_episode(reset_observation) once per reset and add_step with the
    *actual* following observation on each environment step. A terminated or
    truncated step closes its episode automatically. In-flight data is excluded
    by default, as in upstream OnlineTrainer; use include_partial=True to make
    full H-step windows in the current episode eligible. Calling start_episode
    again without a boundary or explicit discard_partial() is an error.

    Defaults bootstrap across truncation but not termination. For HAIC, finish
    is a truncation with info['finished']=True: pass terminal=terminated or
    info['finished'] on that step to disable bootstrap and train the termination
    target; ordinary time-limit truncations should retain terminal=False.
    """

    def __init__(
        self,
        capacity: int,
        horizon: int,
        action_dim: int = 3,
        *,
        seed: int = 0,
        include_partial: bool = False,
        bootstrap_on_truncation: bool = True,
        augmentation_pad: int = 3,
    ) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if type(horizon) is not int or not 0 < horizon <= capacity:
            raise ValueError("horizon must be positive and at most capacity")
        if type(action_dim) is not int or action_dim <= 0:
            raise ValueError("action_dim must be a positive integer")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if type(include_partial) is not bool or type(bootstrap_on_truncation) is not bool:
            raise TypeError("include_partial and bootstrap_on_truncation must be bool")
        if type(augmentation_pad) is not int or augmentation_pad < 0:
            raise ValueError("augmentation_pad must be a non-negative integer")
        self.capacity = capacity
        self.horizon = horizon
        self.action_dim = action_dim
        self.include_partial = include_partial
        self.bootstrap_on_truncation = bootstrap_on_truncation
        self.augmentation_pad = augmentation_pad
        self.rng = np.random.default_rng(seed)
        self._episodes: deque[_Episode] = deque()
        self._active: _Episode | None = None
        self._observation_shape: tuple[int, int, int] | None = None
        self._next_episode_id = 0
        self._size = 0

    def __len__(self) -> int:
        """Number of stored transitions in completed episodes (not in-flight)."""
        return self._size

    @property
    def active_length(self) -> int:
        return len(self._active) if self._active is not None else 0

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def _pixel(self, observation: np.ndarray | torch.Tensor) -> np.ndarray:
        value = observation.detach().cpu().numpy() if isinstance(observation, torch.Tensor) else np.asarray(observation)
        if value.ndim != 3 or value.shape[0] < 1 or value.shape[1] < 1 or value.shape[1] != value.shape[2]:
            raise ValueError("observation must be a [C,S,S] frame stack")
        if self._observation_shape is not None and value.shape != self._observation_shape:
            raise ValueError("observation shape differs from the replay's first observation")
        if value.dtype == np.uint8:
            return np.array(value, copy=True, order="C")
        if value.dtype != np.float32 or not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
            raise ValueError("observation must be uint8 or finite float32 pixels in [0,1]")
        return np.rint(value * 255.0).astype(np.uint8)

    def start_episode(self, observation: np.ndarray | torch.Tensor) -> int:
        """Register a reset image; a previous live episode must be closed or discarded."""
        if self._active is not None:
            raise RuntimeError("previous episode has no terminated/truncated boundary")
        image = self._pixel(observation)
        if self._observation_shape is None:
            self._observation_shape = image.shape
        episode_id = self._next_episode_id
        self._next_episode_id += 1
        self._active = _Episode(episode_id=episode_id, observations=[image])
        return episode_id

    def discard_partial(self) -> int:
        """Explicitly discard an unfinished episode, e.g. at a training budget cut."""
        if self._active is None:
            raise RuntimeError("no in-flight episode to discard")
        steps = len(self._active)
        self._active = None
        return steps

    def add_step(
        self,
        next_observation: np.ndarray | torch.Tensor,
        action: np.ndarray | torch.Tensor,
        reward: float,
        terminated: bool = False,
        truncated: bool = False,
        *,
        terminal: bool | None = None,
    ) -> None:
        """Append one transition and, on a real boundary, commit the episode."""
        if self._active is None:
            raise RuntimeError("call start_episode before adding a step")
        if type(terminated) not in (bool, np.bool_) or type(truncated) not in (bool, np.bool_):
            raise TypeError("terminated and truncated must be bool")
        if terminal is not None and type(terminal) not in (bool, np.bool_):
            raise TypeError("terminal must be bool or None")
        terminal = bool(terminated) if terminal is None else bool(terminal)
        if (terminated and not terminal) or (terminal and not (terminated or truncated)):
            raise ValueError("terminal must include terminations and be at an episode boundary")
        image = self._pixel(next_observation)
        action_value = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
        if action_value.shape != (self.action_dim,) or action_value.dtype.kind not in "fi":
            raise ValueError(f"action must have shape ({self.action_dim},) and numeric dtype")
        action_value = np.asarray(action_value, dtype=np.float32)
        if not np.isfinite(action_value).all() or np.any(action_value < -1) or np.any(action_value > 1):
            raise ValueError("action must be finite and within [-1,1]")
        reward_value = reward.detach().cpu().numpy() if isinstance(reward, torch.Tensor) else np.asarray(reward)
        if reward_value.ndim != 0 or reward_value.dtype.kind not in "fiu":
            raise ValueError("reward must be a finite scalar")
        scalar_reward = float(np.float32(reward_value))
        if not np.isfinite(scalar_reward):
            raise ValueError("reward must be a finite scalar")

        episode = self._active
        episode.observations.append(image)
        episode.actions.append(action_value.copy())
        episode.rewards.append(scalar_reward)
        episode.terminated.append(bool(terminated))
        episode.truncated.append(bool(truncated))
        episode.terminal.append(terminal)
        if len(episode) > self.capacity:
            episode.trim_prefix(len(episode) - self.capacity)
        if terminated or truncated:
            self._active = None
            self._episodes.append(episode)
            self._size += len(episode)
            while self._size > self.capacity:
                oldest = self._episodes[0]
                overflow = min(self._size - self.capacity, len(oldest))
                self._size -= overflow
                if overflow == len(oldest):
                    self._episodes.popleft()
                else:
                    oldest.trim_prefix(overflow)

    def eligible_windows(self, *, include_partial: bool | None = None) -> int:
        """Count full H-transition windows, for caller-controlled seed-step gates."""
        if include_partial is None:
            include_partial = self.include_partial
        if type(include_partial) is not bool:
            raise TypeError("include_partial must be bool")
        episodes = list(self._episodes)
        if include_partial and self._active is not None:
            episodes.append(self._active)
        return sum(max(0, len(episode) - self.horizon + 1) for episode in episodes)

    def sample(
        self,
        batch_size: int,
        *,
        include_partial: bool | None = None,
        augment: bool = False,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample uniform, time-major [H+1,B] observations and [H,B] targets.

        bootstrap_mask is the TD multiplier before discount; terminal marks
        semantic end for the termination head and may additionally include a
        HAIC finish. Raw terminated/truncated flags are returned unchanged.
        """
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(augment) is not bool:
            raise TypeError("augment must be bool")
        if include_partial is None:
            include_partial = self.include_partial
        if type(include_partial) is not bool:
            raise TypeError("include_partial must be bool")
        episodes = list(self._episodes)
        if include_partial and self._active is not None:
            episodes.append(self._active)
        counts = np.array([max(0, len(episode) - self.horizon + 1) for episode in episodes], dtype=np.int64)
        total = int(counts.sum())
        if total == 0:
            raise ValueError("replay contains no full, eligible episode-contained windows")
        if self._observation_shape is None:
            raise RuntimeError("nonempty replay has no observation shape")
        obs = np.empty((self.horizon + 1, batch_size, *self._observation_shape), dtype=np.uint8)
        actions = np.empty((self.horizon, batch_size, self.action_dim), dtype=np.float32)
        rewards = np.empty((self.horizon, batch_size, 1), dtype=np.float32)
        terminated = np.empty((self.horizon, batch_size, 1), dtype=np.float32)
        truncated = np.empty_like(terminated)
        terminal = np.empty_like(terminated)
        episode_ids = np.empty(batch_size, dtype=np.int64)
        start_steps = np.empty(batch_size, dtype=np.int64)
        boundaries = np.cumsum(counts)
        for column, index in enumerate(self.rng.integers(total, size=batch_size)):
            episode_index = int(np.searchsorted(boundaries, index, side="right"))
            episode = episodes[episode_index]
            start = int(index - (boundaries[episode_index] - counts[episode_index]))
            stop = start + self.horizon
            obs[:, column] = episode.observations[start : stop + 1]
            actions[:, column] = episode.actions[start:stop]
            rewards[:, column, 0] = episode.rewards[start:stop]
            terminated[:, column, 0] = episode.terminated[start:stop]
            truncated[:, column, 0] = episode.truncated[start:stop]
            terminal[:, column, 0] = episode.terminal[start:stop]
            episode_ids[column] = episode.episode_id
            start_steps[column] = episode.start_step + start
        mask = 1.0 - terminal if self.bootstrap_on_truncation else 1.0 - np.maximum(terminal, truncated)
        result = {
            "obs": torch.from_numpy(obs).to(device),
            "action": torch.from_numpy(actions).to(device),
            "reward": torch.from_numpy(rewards).to(device),
            "terminated": torch.from_numpy(terminated).to(device),
            "truncated": torch.from_numpy(truncated).to(device),
            "terminal": torch.from_numpy(terminal).to(device),
            "bootstrap_mask": torch.from_numpy(mask).to(device),
            "episode_id": torch.from_numpy(episode_ids).to(device),
            "start_step": torch.from_numpy(start_steps).to(device),
        }
        if augment:
            result["obs"] = random_shift(result["obs"], pad=self.augmentation_pad, generator=generator)
        return result

    @staticmethod
    def _episode_state(episode: _Episode, action_dim: int) -> dict[str, Any]:
        return {
            "episode_id": episode.episode_id,
            "start_step": episode.start_step,
            "observations": np.stack(episode.observations),
            "actions": np.stack(episode.actions) if episode.actions else np.empty((0, action_dim), np.float32),
            "rewards": np.asarray(episode.rewards, dtype=np.float32),
            "terminated": np.asarray(episode.terminated, dtype=np.bool_),
            "truncated": np.asarray(episode.truncated, dtype=np.bool_),
            "terminal": np.asarray(episode.terminal, dtype=np.bool_),
        }

    def state_dict(self) -> dict[str, Any]:
        """Copy full replay, including an in-flight episode and exact sampler RNG."""
        return {
            "format": "haic-tdmpc2-episode-replay-v1",
            "capacity": self.capacity,
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "include_partial": self.include_partial,
            "bootstrap_on_truncation": self.bootstrap_on_truncation,
            "augmentation_pad": self.augmentation_pad,
            "observation_shape": self._observation_shape,
            "next_episode_id": self._next_episode_id,
            "size": self._size,
            "episodes": [self._episode_state(ep, self.action_dim) for ep in self._episodes],
            "active": None if self._active is None else self._episode_state(self._active, self.action_dim),
            "rng_state": deepcopy(self.rng.bit_generator.state),
        }

    def _restore_episode(
        self, state: dict[str, Any], shape: tuple[int, int, int], *, completed: bool
    ) -> _Episode:
        if not isinstance(state, dict):
            raise ValueError("replay episode checkpoint must be a dictionary")
        episode_id, start_step = state.get("episode_id"), state.get("start_step")
        if type(episode_id) is not int or episode_id < 0 or type(start_step) is not int or start_step < 0:
            raise ValueError("replay episode ID and start step must be non-negative integers")
        arrays = {}
        expected = {
            "observations": (np.uint8, (None, *shape)),
            "actions": (np.float32, (None, self.action_dim)),
            "rewards": (np.float32, (None,)),
            "terminated": (np.bool_, (None,)),
            "truncated": (np.bool_, (None,)),
            "terminal": (np.bool_, (None,)),
        }
        for name, (dtype, dimensions) in expected.items():
            array = np.asarray(state.get(name))
            if (
                array.dtype != dtype
                or array.ndim != len(dimensions)
                or array.shape[1:] != dimensions[1:]
            ):
                raise ValueError(f"replay checkpoint {name} has invalid shape or dtype")
            arrays[name] = array
        length = len(arrays["actions"])
        if (completed and length == 0) or length > self.capacity or len(arrays["observations"]) != length + 1:
            raise ValueError("replay checkpoint episode has invalid transition count")
        if any(len(arrays[name]) != length for name in expected if name != "observations"):
            raise ValueError("replay checkpoint episode has inconsistent field lengths")
        if not np.isfinite(arrays["actions"]).all() or np.any(np.abs(arrays["actions"]) > 1):
            raise ValueError("replay checkpoint contains invalid actions")
        if not np.isfinite(arrays["rewards"]).all():
            raise ValueError("replay checkpoint contains invalid rewards")
        terminated, truncated, terminal = (arrays[name] for name in ("terminated", "truncated", "terminal"))
        if np.any(terminated & ~terminal) or np.any(terminal & ~(terminated | truncated)):
            raise ValueError("replay checkpoint has invalid semantic terminal labels")
        if np.any(terminated[:-1] | truncated[:-1] | terminal[:-1]):
            raise ValueError("replay checkpoint has an internal reset boundary")
        if completed and not (terminated[-1] or truncated[-1]):
            raise ValueError("completed replay episode is missing its final boundary")
        if not completed and length and (terminated[-1] or truncated[-1] or terminal[-1]):
            raise ValueError("in-flight replay episode already has a boundary")
        return _Episode(
            episode_id=episode_id,
            start_step=start_step,
            observations=[value.copy() for value in arrays["observations"]],
            actions=[value.copy() for value in arrays["actions"]],
            rewards=arrays["rewards"].tolist(),
            terminated=terminated.tolist(),
            truncated=truncated.tolist(),
            terminal=terminal.tolist(),
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore only a shape/config-consistent, episode-safe replay snapshot."""
        if not isinstance(state, dict) or state.get("format") != "haic-tdmpc2-episode-replay-v1":
            raise ValueError("unsupported replay checkpoint format")
        for name in (
            "capacity", "horizon", "action_dim", "include_partial",
            "bootstrap_on_truncation", "augmentation_pad",
        ):
            value = state.get(name)
            expected = getattr(self, name)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"replay checkpoint {name} differs from this replay")
        shape = state.get("observation_shape")
        if shape is not None and (
            not isinstance(shape, tuple)
            or len(shape) != 3
            or any(type(x) is not int or x <= 0 for x in shape)
            or shape[1] != shape[2]
        ):
            raise ValueError("replay checkpoint observation shape is invalid")
        if self._observation_shape is not None and shape != self._observation_shape:
            raise ValueError("replay checkpoint observation shape differs from this replay")
        snapshots = state.get("episodes")
        if not isinstance(snapshots, list) or (shape is None and (snapshots or state.get("active") is not None)):
            raise ValueError("replay checkpoint episodes are malformed")
        episodes = [self._restore_episode(ep, shape, completed=True) for ep in snapshots]
        active_state = state.get("active")
        active = None if active_state is None else self._restore_episode(active_state, shape, completed=False)
        ids = [ep.episode_id for ep in episodes] + ([] if active is None else [active.episode_id])
        next_id, size = state.get("next_episode_id"), state.get("size")
        if (
            ids != sorted(set(ids))
            or type(next_id) is not int or next_id < 0
            or (ids and next_id <= ids[-1])
            or type(size) is not int or size != sum(len(ep) for ep in episodes)
            or not 0 <= size <= self.capacity
        ):
            raise ValueError("replay checkpoint episode ordering or size is inconsistent")
        test_rng = np.random.default_rng()
        try:
            test_rng.bit_generator.state = deepcopy(state["rng_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("replay checkpoint sampler RNG state is invalid") from exc
        self._episodes = deque(episodes)
        self._active = active
        self._observation_shape = shape
        self._next_episode_id = next_id
        self._size = size
        self.rng.bit_generator.state = test_rng.bit_generator.state
