"""Training environment construction at the evaluation observation cadence."""

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from gymnasium.wrappers import TimeLimit

from core.vendor.car_racing import CarRacing
from env_wrapper import CarEnvironment
from haic_agent.observation import HUDFeatures
from training.labels import TrainingLabels, labels_with_hud


WARMUP_RAW_TICKS = 50
FRAME_SKIP_RAW_TICKS = 4
FRAME_STACK = 4


@dataclass(frozen=True)
class CollectedTransition:
    observation: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    labels: TrainingLabels
    hud_features: HUDFeatures
    label_timing: str = "next_decision"


@dataclass(frozen=True)
class TrackSeedSplit:
    train: tuple[tuple[int, int], ...]
    tune: tuple[tuple[int, int], ...]
    held_out: tuple[tuple[int, int], ...]


def split_track_seeds(
    train: Iterable[tuple[int, int]],
    tune: Iterable[tuple[int, int]],
    held_out: Iterable[tuple[int, int]],
) -> TrackSeedSplit:
    """Freeze deterministic episode lists and reject leakage between them."""
    groups = {
        "train": tuple((int(track_id), int(seed)) for track_id, seed in train),
        "tune": tuple((int(track_id), int(seed)) for track_id, seed in tune),
        "held_out": tuple((int(track_id), int(seed)) for track_id, seed in held_out),
    }
    for left, right in (("train", "tune"), ("train", "held_out"), ("tune", "held_out")):
        overlap = set(groups[left]) & set(groups[right])
        if overlap:
            raise ValueError(f"{left} and {right} episode sets overlap: {sorted(overlap)}")
    return TrackSeedSplit(**groups)


class CollectingEnvironment:
    """A CarEnvironment plus train-only state labels at each decision boundary."""

    def __init__(
        self,
        raw_environment: Any,
        *,
        default_seed: int | None = None,
        default_track_id: int | None = None,
    ):
        self.environment = CarEnvironment(
            raw_environment,
            skip_frames=FRAME_SKIP_RAW_TICKS,
            stack_frames=FRAME_STACK,
            no_operation=WARMUP_RAW_TICKS,
        )
        self._observation: np.ndarray | None = None
        self._default_seed = default_seed
        self._default_track_id = default_track_id

    @property
    def unwrapped(self) -> Any:
        return self.environment.unwrapped

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is None:
            seed = self._default_seed
        options = dict(options or {})
        if self._default_track_id is not None:
            options.setdefault("track_id", self._default_track_id)
        observation, info = self.environment.reset(seed=seed, options=options)
        self._observation = observation
        return observation, info

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.environment.step(action)
        self._observation = observation
        return observation, reward, terminated, truncated, info

    def step_transition(self, action: np.ndarray) -> CollectedTransition:
        if self._observation is None:
            raise RuntimeError("reset must be called before collecting transitions")
        observation = self._observation.copy()
        next_observation, reward, terminated, truncated, info = self.step(action)
        labels, hud_features = labels_with_hud(self.environment, info, next_observation)
        return CollectedTransition(
            observation=observation,
            action=np.asarray(action, dtype=np.float32).copy(),
            next_observation=next_observation.copy(),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            labels=labels,
            hud_features=hud_features,
        )


def make_collecting_environment(raw_environment: Any) -> CollectingEnvironment:
    return CollectingEnvironment(raw_environment)


def create_training_environment(
    *, track_id: int, seed: int, max_decisions: int, render_mode: str | None = None
) -> CollectingEnvironment:
    """Create the local simulator with the evaluation warmup and decision rate."""
    raw_ticks = WARMUP_RAW_TICKS + max_decisions * FRAME_SKIP_RAW_TICKS + 200
    raw = TimeLimit(
        CarRacing(continuous=True, render_mode=render_mode), max_episode_steps=raw_ticks
    )
    return CollectingEnvironment(
        raw, default_track_id=int(track_id), default_seed=int(seed)
    )
