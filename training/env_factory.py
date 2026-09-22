"""Training environment construction at the evaluation observation cadence."""

from dataclasses import dataclass
from typing import Any, Iterable, TypeAlias

import numpy as np
from gymnasium.wrappers import TimeLimit

from core.vendor.car_racing import CarRacing
from env_wrapper import CarEnvironment
from haic_agent.observation import HUDFeatures
from training.labels import TrainingLabels, collect_labels, labels_with_hud
from training.site_environment import SiteCustomCarRacing, attach_site_obstacles
from training.site_maps import SiteMapEpisode, SiteMapSpec, SiteMapSplit


WARMUP_RAW_TICKS = 50
FRAME_SKIP_RAW_TICKS = 4
FRAME_STACK = 4
TrainingEpisode: TypeAlias = tuple[int, int] | SiteMapEpisode


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
    observation_labels: TrainingLabels | None = None


@dataclass(frozen=True)
class TrackSeedSplit:
    train: tuple[tuple[int, int], ...]
    tune: tuple[tuple[int, int], ...]
    held_out: tuple[tuple[int, int], ...]


TrainingSplit: TypeAlias = TrackSeedSplit | SiteMapSplit


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
        site_map: SiteMapSpec | None = None,
        frame_skip: int = FRAME_SKIP_RAW_TICKS,
    ):
        self.environment = CarEnvironment(
            raw_environment,
            skip_frames=frame_skip,
            stack_frames=FRAME_STACK,
            no_operation=WARMUP_RAW_TICKS,
        )
        self._observation: np.ndarray | None = None
        self._default_seed = default_seed
        self._default_track_id = default_track_id
        self._site_map = site_map
        self._site_obstacle_count = 0

    @property
    def unwrapped(self) -> Any:
        return self.environment.unwrapped

    def close(self) -> None:
        self.environment.close()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is None:
            seed = self._default_seed
        options = dict(options or {})
        if self._site_map is not None:
            if self._site_map.map_kind == "official" and self._site_map.obstacle_mode != "custom_only":
                options.setdefault("track_id", self._site_map.track_id)
            else:
                options.pop("track_id", None)
        elif self._default_track_id is not None:
            options.setdefault("track_id", self._default_track_id)
        observation, info = self.environment.reset(seed=seed, options=options)
        if self._site_map is not None:
            self._site_obstacle_count = attach_site_obstacles(self.environment, self._site_map)
            info = dict(info)
            info.update(self._site_metadata())
        self._observation = observation
        return observation, info

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.environment.step(action)
        if self._site_map is not None:
            info = dict(info)
            info.update(self._site_metadata())
        self._observation = observation
        return observation, reward, terminated, truncated, info

    def _site_metadata(self) -> dict[str, Any]:
        assert self._site_map is not None
        return {
            "site_map_id": self._site_map.map_id,
            "site_map_kind": self._site_map.map_kind,
            "site_obstacle_count": self._site_obstacle_count,
            "obstacle_count": len(getattr(self.unwrapped, "obstacles", ())),
            "obstacle_mode": self._site_map.obstacle_mode,
        }

    def step_transition(self, action: np.ndarray) -> CollectedTransition:
        if self._observation is None:
            raise RuntimeError("reset must be called before collecting transitions")
        observation = self._observation.copy()
        observation_labels = collect_labels(self.environment, {})
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
            observation_labels=observation_labels,
        )


def make_collecting_environment(raw_environment: Any) -> CollectingEnvironment:
    return CollectingEnvironment(raw_environment)


def create_training_environment(
    *,
    track_id: int | None,
    seed: int,
    max_decisions: int,
    render_mode: str | None = None,
    site_map: SiteMapSpec | None = None,
) -> CollectingEnvironment:
    """Create the local simulator with the evaluation warmup and decision rate."""
    frame_skip = site_map.frame_skip if site_map is not None else FRAME_SKIP_RAW_TICKS
    raw_ticks = WARMUP_RAW_TICKS + max_decisions * frame_skip + 200
    if site_map is not None and site_map.map_kind == "custom":
        raw_environment = SiteCustomCarRacing(site_map, render_mode=render_mode)
    else:
        raw_environment = CarRacing(continuous=True, render_mode=render_mode)
    raw = TimeLimit(raw_environment, max_episode_steps=raw_ticks)
    site_track_id = (
        site_map.track_id
        if site_map is not None
        and site_map.map_kind == "official"
        and site_map.obstacle_mode != "custom_only"
        else None
    )
    return CollectingEnvironment(
        raw,
        default_track_id=site_track_id if site_map is not None else int(track_id) if track_id is not None else None,
        default_seed=int(seed),
        site_map=site_map,
        frame_skip=frame_skip,
    )


def create_episode_environment(
    episode: TrainingEpisode,
    *,
    max_decisions: int,
    render_mode: str | None = None,
) -> CollectingEnvironment:
    """Resolve a standard track/seed pair or a site map episode to an environment."""
    if isinstance(episode, SiteMapEpisode):
        site_map = episode.site_map
        track_id = (
            site_map.track_id
            if site_map.map_kind == "official" and site_map.obstacle_mode != "custom_only"
            else None
        )
        return create_training_environment(
            track_id=track_id,
            seed=episode.seed,
            max_decisions=max_decisions,
            render_mode=render_mode,
            site_map=site_map,
        )
    track_id, seed = episode
    return create_training_environment(
        track_id=int(track_id),
        seed=int(seed),
        max_decisions=max_decisions,
        render_mode=render_mode,
    )


def describe_episode(episode: TrainingEpisode) -> dict[str, Any]:
    """Return JSON-safe identity and obstacle metadata for a training episode."""
    if isinstance(episode, SiteMapEpisode):
        return {
            "map_id": episode.map_id,
            "map_kind": episode.site_map.map_kind,
            "seed": episode.seed,
            "obstacle_count": len(episode.site_map.obstacles),
            "source_map": episode.source_path.name,
        }
    track_id, seed = episode
    return {"track_id": int(track_id), "seed": int(seed)}


def describe_split(split: TrainingSplit) -> dict[str, list[dict[str, Any]]]:
    """Serialize a standard or site map split without dataclass/path objects."""
    return {
        name: [describe_episode(episode) for episode in getattr(split, name)]
        for name in ("train", "tune", "held_out")
    }
