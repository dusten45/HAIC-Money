"""Frame-efficient one-step pixel replay with explicit episode boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FRAME_SHAPE = (84, 84)
STACK_SHAPE = (4, 84, 84)
ACTION_DIM = 3


@dataclass(frozen=True)
class PixelTransition:
    observation: np.ndarray
    proposed_action: np.ndarray
    executed_action: np.ndarray
    applied_action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    terminal: bool
    episode_id: int
    step: int
    track_id: int
    geometry_seed: int

    def __post_init__(self) -> None:
        for name in ("observation", "next_observation"):
            value = np.asarray(getattr(self, name))
            if value.shape != STACK_SHAPE or value.dtype not in (np.uint8, np.float32):
                raise ValueError(f"{name} must be a 4x84x84 uint8 or float32 stack")
            if value.dtype == np.float32 and (
                not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0)
            ):
                raise ValueError(f"{name} must contain finite values in [0, 1]")
        for name in ("proposed_action", "executed_action", "applied_action"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite 3-action vector")
            low = np.asarray((-1.0, 0.0, 0.0) if name == "applied_action" else (-1.0, -1.0, -1.0))
            high = np.ones(ACTION_DIM, dtype=np.float32)
            if np.any(value < low) or np.any(value > high):
                raise ValueError(f"{name} is outside its declared coordinate bounds")
        if not np.isfinite(self.reward):
            raise ValueError("reward must be finite")
        if type(self.episode_id) is not int or self.episode_id < 0:
            raise ValueError("episode_id must be a non-negative integer")
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if type(self.track_id) is not int or self.track_id <= 0:
            raise ValueError("track_id must be a positive integer")
        if type(self.geometry_seed) is not int or not 0 <= self.geometry_seed < 2**32:
            raise ValueError("geometry_seed must be an unsigned 32-bit integer")
        terminated, truncated, terminal = bool(self.terminated), bool(self.truncated), bool(self.terminal)
        if terminated and not terminal:
            raise ValueError("terminated transitions must disable bootstrapping")
        if terminal and not (terminated or truncated):
            raise ValueError("terminal transitions must end an episode")
        object.__setattr__(self, "observation", self._as_uint8(self.observation))
        object.__setattr__(self, "next_observation", self._as_uint8(self.next_observation))
        for name in ("proposed_action", "executed_action", "applied_action"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float32).copy())
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "terminated", terminated)
        object.__setattr__(self, "truncated", truncated)
        object.__setattr__(self, "terminal", terminal)

    @staticmethod
    def _as_uint8(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        if value.dtype == np.uint8:
            return np.ascontiguousarray(value).copy()
        return np.rint(value * 255.0).astype(np.uint8)


class FrameStackReplay:
    """Store one grayscale frame per transition and sparse boundary stacks."""

    def __init__(
        self,
        capacity: int,
        *,
        seed: int = 0,
        source: str,
        immutable: bool = False,
    ):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.capacity = capacity
        self.source = source
        self.immutable = bool(immutable)
        self.frames = np.zeros((capacity, *FRAME_SHAPE), dtype=np.uint8)
        self.proposed_actions = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self.executed_actions = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self.applied_actions = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool_)
        self.truncated = np.zeros(capacity, dtype=np.bool_)
        self.terminal = np.zeros(capacity, dtype=np.bool_)
        self.episode_ids = np.full(capacity, -1, dtype=np.int64)
        self.steps = np.full(capacity, -1, dtype=np.int32)
        self.track_ids = np.zeros(capacity, dtype=np.int16)
        self.geometry_seeds = np.zeros(capacity, dtype=np.uint32)
        self.boundary_stacks: dict[int, np.ndarray] = {}
        self.cursor = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)
        self._valid_rows = np.full(capacity, -1, dtype=np.int64)
        self._valid_positions = np.full(capacity, -1, dtype=np.int64)
        self._valid_count = 0

    def __len__(self) -> int:
        return self.size

    @property
    def valid_count(self) -> int:
        return self._valid_count

    def _add_valid(self, index: int) -> None:
        if self._valid_positions[index] >= 0:
            return
        position = self._valid_count
        self._valid_rows[position] = index
        self._valid_positions[index] = position
        self._valid_count += 1

    def _remove_valid(self, index: int) -> None:
        position = int(self._valid_positions[index])
        if position < 0:
            return
        last_position = self._valid_count - 1
        last_index = int(self._valid_rows[last_position])
        self._valid_rows[position] = last_index
        self._valid_positions[last_index] = position
        self._valid_rows[last_position] = -1
        self._valid_positions[index] = -1
        self._valid_count -= 1

    def _is_row(self, index: int, episode_id: int, step: int) -> bool:
        if index < 0 or index >= self.capacity:
            return False
        if self.size < self.capacity and index >= self.size:
            return False
        return int(self.episode_ids[index]) == episode_id and int(self.steps[index]) == step

    def _has_valid_stack_prefix(self, index: int) -> bool:
        episode_id, step = int(self.episode_ids[index]), int(self.steps[index])
        for back in range(1, min(step, STACK_SHAPE[0] - 1) + 1):
            previous = (index - back) % self.capacity
            if not self._is_row(previous, episode_id, step - back):
                return False
        if step < STACK_SHAPE[0] - 1:
            start = (index - step) % self.capacity
            return self._is_row(start, episode_id, 0)
        return True

    def add(self, transition: PixelTransition) -> None:
        if self.immutable:
            raise RuntimeError("offline replay is immutable after finalization")
        if not isinstance(transition, PixelTransition):
            raise TypeError("transition must be a PixelTransition")
        index = self.cursor
        overwritten_episode = int(self.episode_ids[index])
        overwritten_step = int(self.steps[index])
        self._remove_valid(index)
        if overwritten_episode >= 0:
            for forward in range(1, min(3, self.capacity - 1) + 1):
                dependent = (index + forward) % self.capacity
                if self._is_row(dependent, overwritten_episode, overwritten_step + forward):
                    self._remove_valid(dependent)
        predecessor = (index - 1) % self.capacity
        if not (self.terminated[predecessor] or self.truncated[predecessor]):
            self._remove_valid(predecessor)
        self.boundary_stacks.pop(index, None)
        self.frames[index] = transition.observation[-1]
        self.proposed_actions[index] = transition.proposed_action
        self.executed_actions[index] = transition.executed_action
        self.applied_actions[index] = transition.applied_action
        self.rewards[index] = transition.reward
        self.terminated[index] = transition.terminated
        self.truncated[index] = transition.truncated
        self.terminal[index] = transition.terminal
        self.episode_ids[index] = transition.episode_id
        self.steps[index] = transition.step
        self.track_ids[index] = transition.track_id
        self.geometry_seeds[index] = transition.geometry_seed
        if transition.terminated or transition.truncated:
            self.boundary_stacks[index] = transition.next_observation.copy()
        self.cursor = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if (
            transition.terminated or transition.truncated
        ) and self._has_valid_stack_prefix(index):
            self._add_valid(index)
        if (
            self._is_row(predecessor, transition.episode_id, transition.step - 1)
            and self._has_valid_stack_prefix(predecessor)
        ):
            self._add_valid(predecessor)

    def finalize(self) -> None:
        """Seal the fixed prior dataset against future writes."""
        self.immutable = True

    def _observation_stack(self, index: int) -> np.ndarray:
        episode_id, step = int(self.episode_ids[index]), int(self.steps[index])
        if not self._has_valid_stack_prefix(index):
            raise RuntimeError("replay row has an incomplete or overwritten frame stack")
        start = (index - step) % self.capacity if step < 3 else None
        result = np.empty(STACK_SHAPE, dtype=np.uint8)
        for channel, back in enumerate(range(3, -1, -1)):
            source_step = step - back
            if source_step < 0:
                source_index = int(start)
            else:
                source_index = (index - back) % self.capacity
                if not self._is_row(source_index, episode_id, source_step):
                    raise RuntimeError("replay stack crossed an episode or ring boundary")
            result[channel] = self.frames[source_index]
        return result

    def _next_stack(self, index: int) -> np.ndarray:
        if self.terminated[index] or self.truncated[index]:
            try:
                return self.boundary_stacks[index]
            except KeyError as exc:
                raise RuntimeError("episode boundary is missing its actual result stack") from exc
        next_index = (index + 1) % self.capacity
        if not self._is_row(
            next_index, int(self.episode_ids[index]), int(self.steps[index]) + 1
        ):
            raise RuntimeError("non-boundary replay row is missing its true successor")
        return self._observation_stack(next_index)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self._valid_count == 0:
            raise ValueError("replay has no terminal-safe transitions with full successors")
        positions = self.rng.integers(0, self._valid_count, size=batch_size)
        indices = self._valid_rows[positions].astype(np.int64, copy=False)
        observations = np.stack([self._observation_stack(int(index)) for index in indices])
        next_observations = np.stack([self._next_stack(int(index)) for index in indices])
        return {
            "observation": observations,
            "action": self.executed_actions[indices].copy(),
            "proposed_action": self.proposed_actions[indices].copy(),
            "applied_action": self.applied_actions[indices].copy(),
            "reward": self.rewards[indices].copy(),
            "next_observation": next_observations,
            "terminated": self.terminated[indices].copy(),
            "truncated": self.truncated[indices].copy(),
            "terminal": self.terminal[indices].copy(),
            "episode_id": self.episode_ids[indices].copy(),
            "step": self.steps[indices].copy(),
            "track_id": self.track_ids[indices].copy(),
            "geometry_seed": self.geometry_seeds[indices].copy(),
            "source": np.full(batch_size, self.source, dtype=object),
            "indices": indices.copy(),
        }

    def state_dict(self) -> dict[str, Any]:
        """Expose arrays by reference for bounded-memory synchronous snapshots."""
        array_names = (
            "frames", "proposed_actions", "executed_actions", "applied_actions",
            "rewards", "terminated", "truncated", "terminal", "episode_ids",
            "steps", "track_ids", "geometry_seeds",
        )
        return {
            "format": "haic-rlpd-frame-replay-v2",
            "capacity": self.capacity,
            "source": self.source,
            "immutable": self.immutable,
            "cursor": self.cursor,
            "size": self.size,
            "rng_state": self.rng.bit_generator.state,
            "valid_rows": self._valid_rows[:self._valid_count].copy(),
            "arrays": {name: getattr(self, name) for name in array_names},
            "boundary_stacks": self.boundary_stacks,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format") != "haic-rlpd-frame-replay-v2":
            raise ValueError("replay checkpoint format lacks valid-row order for exact resume")
        if (
            state.get("capacity") != self.capacity
            or state.get("source") != self.source
        ):
            raise ValueError("replay checkpoint does not match its source/capacity contract")
        size, cursor = state.get("size"), state.get("cursor")
        if type(size) is not int or not 0 <= size <= self.capacity:
            raise ValueError("invalid replay size")
        if type(cursor) is not int or not 0 <= cursor < self.capacity:
            raise ValueError("invalid replay cursor")
        arrays = state.get("arrays")
        if not isinstance(arrays, dict):
            raise ValueError("replay checkpoint is missing its arrays")
        for name, target in self.state_dict()["arrays"].items():
            value = np.asarray(arrays.get(name))
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"replay array {name} has an incompatible shape or dtype")
            target[...] = value
        boundary_stacks = state.get("boundary_stacks")
        if not isinstance(boundary_stacks, dict):
            raise ValueError("replay checkpoint boundary stacks are malformed")
        restored = {}
        for raw_index, stack in boundary_stacks.items():
            index = int(raw_index)
            stack = np.asarray(stack)
            if (
                not 0 <= index < self.capacity
                or stack.shape != STACK_SHAPE
                or stack.dtype != np.uint8
                or not self.terminated[index] and not self.truncated[index]
            ):
                raise ValueError("replay checkpoint has an invalid boundary stack")
            restored[index] = stack.copy()
        self.boundary_stacks = restored
        self.cursor = cursor
        self.size = size
        self.immutable = bool(state.get("immutable"))
        self.rng.bit_generator.state = state["rng_state"]
        self._valid_rows.fill(-1)
        self._valid_positions.fill(-1)
        self._valid_count = 0
        active = range(self.size) if self.size < self.capacity else range(self.capacity)
        for index in active:
            if not self._has_valid_stack_prefix(index):
                continue
            if self.terminated[index] or self.truncated[index]:
                if index in self.boundary_stacks:
                    self._add_valid(index)
            else:
                next_index = (index + 1) % self.capacity
                if self._is_row(next_index, int(self.episode_ids[index]), int(self.steps[index]) + 1):
                    self._add_valid(index)
        valid_rows = np.asarray(state.get("valid_rows"))
        if (
            valid_rows.ndim != 1
            or valid_rows.dtype != np.int64
            or valid_rows.size != self._valid_count
            or not np.array_equal(np.sort(valid_rows), np.sort(self._valid_rows[:self._valid_count]))
        ):
            raise ValueError("replay checkpoint valid rows do not match its reconstructable transitions")
        self._valid_rows.fill(-1)
        self._valid_positions.fill(-1)
        self._valid_rows[:self._valid_count] = valid_rows
        self._valid_positions[valid_rows] = np.arange(self._valid_count, dtype=np.int64)
