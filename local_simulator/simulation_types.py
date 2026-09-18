from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .environment import TrackSnapshot
from .schema import SCHEMA_VERSION


@dataclass(frozen=True)
class RunStep:
    step: int
    sim_time_s: float
    action: tuple[float, float, float]
    position: tuple[float, float]
    angle: float
    velocity: tuple[float, float]
    progress: float
    damage: float
    collision: bool
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class RunLog:
    schema_version: int
    run: dict[str, Any]
    track: TrackSnapshot
    steps: tuple[RunStep, ...]
    summary: dict[str, Any]
    frames: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
