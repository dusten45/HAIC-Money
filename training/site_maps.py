"""Validated input types for maps exported by the HAIC Track Lab site."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


MAX_SEED = 0xFFFFFFFF
MAX_MAP_STEPS = 10_000
MAX_FRAME_SKIP = 16
MAX_CUSTOM_OBSTACLES = 64
_CUSTOM_MAP_ID = re.compile(r"^custom-track-[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_OBSTACLE_MODES = {"official", "custom_only", "official_plus_custom"}


@dataclass(frozen=True)
class SiteObstacle:
    progress: float
    lateral: float
    radius: float


@dataclass(frozen=True)
class CustomTrackGeometry:
    centerline: tuple[tuple[float, float], ...]
    width: float
    start_index: int = 0
    direction: int = 1


@dataclass(frozen=True)
class SiteMapSpec:
    map_id: str
    map_kind: str
    track_id: int
    seed: int
    obstacle_mode: str
    obstacles: tuple[SiteObstacle, ...]
    max_steps: int
    frame_skip: int
    geometry: CustomTrackGeometry | None = None
    generator_template: str | None = None
    design_seed: int | None = None


@dataclass(frozen=True)
class SiteMapEpisode:
    site_map: SiteMapSpec
    seed: int
    source_path: Path

    @property
    def map_id(self) -> str:
        return self.site_map.map_id


@dataclass(frozen=True)
class SiteMapSplit:
    train: tuple[SiteMapEpisode, ...]
    tune: tuple[SiteMapEpisode, ...]
    held_out: tuple[SiteMapEpisode, ...]


def _integer(name: str, value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return int(value)


def _number(name: str, value: object, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _obstacles(payload: Mapping[str, Any]) -> tuple[SiteObstacle, ...]:
    raw_obstacles = payload.get("obstacles", [])
    if not isinstance(raw_obstacles, list):
        raise ValueError("obstacles must be an array")
    if len(raw_obstacles) > MAX_CUSTOM_OBSTACLES:
        raise ValueError(f"at most {MAX_CUSTOM_OBSTACLES} obstacles are allowed")
    result: list[SiteObstacle] = []
    for index, item in enumerate(raw_obstacles):
        if not isinstance(item, Mapping):
            raise ValueError(f"obstacles[{index}] must be an object")
        try:
            result.append(
                SiteObstacle(
                    progress=_number(f"obstacles[{index}].progress", item["progress"], 0.0, 1.0),
                    lateral=_number(f"obstacles[{index}].lateral", item["lateral"], -1.0, 1.0),
                    radius=_number(f"obstacles[{index}].radius", item["radius"], 0.2, 4.0),
                )
            )
        except KeyError as error:
            raise ValueError(f"obstacles[{index}] is missing {error.args[0]}") from error
    return tuple(result)


def _custom_geometry(payload: Mapping[str, Any]) -> CustomTrackGeometry:
    raw_geometry = payload.get("geometry")
    if not isinstance(raw_geometry, Mapping):
        raise ValueError("geometry must be an object for custom maps")
    raw_centerline = raw_geometry.get("centerline")
    if not isinstance(raw_centerline, list) or len(raw_centerline) < 3:
        raise ValueError("geometry.centerline must contain at least three points")
    if len(raw_centerline) > 4096:
        raise ValueError("geometry.centerline supports at most 4096 points")
    centerline: list[tuple[float, float]] = []
    for index, point in enumerate(raw_centerline):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"geometry.centerline[{index}] must contain x and y")
        centerline.append(
            (
                _number(f"geometry.centerline[{index}].x", point[0], -100_000.0, 100_000.0),
                _number(f"geometry.centerline[{index}].y", point[1], -100_000.0, 100_000.0),
            )
        )
    width = _number("geometry.width", raw_geometry.get("width"), 0.5, 100.0)
    start_index = _integer(
        "geometry.start_index", raw_geometry.get("start_index", 0), 0, len(centerline) - 1
    )
    direction = _integer("geometry.direction", raw_geometry.get("direction", 1), -1, 1)
    if direction not in (-1, 1):
        raise ValueError("geometry.direction must be -1 or 1")
    return CustomTrackGeometry(tuple(centerline), width, start_index, direction)


def load_site_map_payload(payload: object) -> SiteMapSpec:
    """Validate one official or custom map JSON document from Track Lab."""
    if not isinstance(payload, Mapping):
        raise ValueError("map payload must be an object")
    version = payload.get("schema_version", 1)
    if isinstance(version, bool) or version not in (1, 2):
        raise ValueError(f"unsupported map schema_version: {version}")
    map_kind = payload.get("map_kind", "official")
    if map_kind not in ("official", "custom"):
        raise ValueError("map_kind must be official or custom")
    if map_kind == "custom" and version != 2:
        raise ValueError("custom maps require schema_version 2")

    max_steps = _integer("max_steps", payload.get("max_steps", 2000), 1, MAX_MAP_STEPS)
    frame_skip = _integer("frame_skip", payload.get("frame_skip", 4), 1, MAX_FRAME_SKIP)
    obstacles = _obstacles(payload)
    generator = payload.get("generator", {})
    if not isinstance(generator, Mapping):
        raise ValueError("generator must be an object")
    generator_template = generator.get("template")
    design_seed_value = generator.get("design_seed")
    design_seed = (
        _integer("generator.design_seed", design_seed_value, 0, MAX_SEED)
        if design_seed_value is not None
        else None
    )

    if map_kind == "custom":
        map_id = payload.get("map_id")
        if not isinstance(map_id, str) or not _CUSTOM_MAP_ID.fullmatch(map_id):
            raise ValueError("custom map_id must start with custom-track-")
        geometry = _custom_geometry(payload)
        return SiteMapSpec(
            map_id=map_id,
            map_kind="custom",
            track_id=1,
            seed=design_seed if design_seed is not None else 0,
            obstacle_mode="custom_only",
            obstacles=obstacles,
            max_steps=max_steps,
            frame_skip=frame_skip,
            geometry=geometry,
            generator_template=str(generator_template) if generator_template is not None else None,
            design_seed=design_seed,
        )

    track_id = _integer("track_id", payload.get("track_id"), 1, 2**63 - 1)
    seed = _integer("seed", payload.get("seed"), 0, MAX_SEED)
    obstacle_mode = payload.get("obstacle_mode", "official")
    if obstacle_mode not in _OBSTACLE_MODES:
        raise ValueError(f"obstacle_mode must be one of {sorted(_OBSTACLE_MODES)}")
    map_id = payload.get("map_id", f"official-track-{track_id}-seed-{seed}")
    if not isinstance(map_id, str) or not map_id:
        raise ValueError("map_id must be a non-empty string")
    return SiteMapSpec(
        map_id=map_id,
        map_kind="official",
        track_id=track_id,
        seed=seed,
        obstacle_mode=obstacle_mode,
        obstacles=obstacles,
        max_steps=max_steps,
        frame_skip=frame_skip,
        generator_template=None,
        design_seed=None,
    )


def load_site_map(path: Path | str) -> SiteMapSpec:
    """Read and validate a saved Track Lab map JSON file."""
    map_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read site map {map_path}: {error}") from error
    return load_site_map_payload(payload)


def _episodes_for_group(
    payload: object,
    *,
    group: str,
    base_directory: Path,
) -> tuple[SiteMapEpisode, ...]:
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{group} must contain at least one map entry")
    episodes: list[SiteMapEpisode] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"{group}[{index}] must be an object")
        raw_path = item.get("map")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{group}[{index}].map must be a relative file name")
        candidate = (base_directory / raw_path).resolve()
        try:
            candidate.relative_to(base_directory)
        except ValueError as error:
            raise ValueError("map paths must stay inside the manifest directory") from error
        site_map = load_site_map(candidate)
        raw_seeds = item.get("seeds")
        if not isinstance(raw_seeds, list) or not raw_seeds:
            raise ValueError(f"{group}[{index}].seeds must contain at least one seed")
        seeds = [_integer(f"{group}[{index}].seeds", value, 0, MAX_SEED) for value in raw_seeds]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{group}[{index}].seeds contains duplicate values")
        episodes.extend(
            SiteMapEpisode(site_map=site_map, seed=seed, source_path=candidate)
            for seed in seeds
        )
    return tuple(episodes)


def load_site_map_split_payload(
    payload: object,
    *,
    base_directory: Path,
) -> SiteMapSplit:
    """Load map episodes and reject map-design leakage between splits."""
    if not isinstance(payload, Mapping):
        raise ValueError("site map split must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported site map split schema_version")
    root = Path(base_directory).expanduser().resolve()
    groups = {
        name: _episodes_for_group(payload.get(name), group=name, base_directory=root)
        for name in ("train", "tune", "held_out")
    }
    for left, right in (("train", "tune"), ("train", "held_out"), ("tune", "held_out")):
        overlap = {episode.map_id for episode in groups[left]} & {
            episode.map_id for episode in groups[right]
        }
        if overlap:
            raise ValueError(f"{left} and {right} map IDs overlap: {sorted(overlap)}")
    return SiteMapSplit(**groups)


def load_site_map_split(path: Path | str) -> SiteMapSplit:
    """Read a JSON manifest whose map paths are relative to its directory."""
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read site map split {manifest_path}: {error}") from error
    return load_site_map_split_payload(payload, base_directory=manifest_path.parent)


__all__ = [
    "CustomTrackGeometry",
    "SiteMapEpisode",
    "SiteMapSpec",
    "SiteMapSplit",
    "SiteObstacle",
    "load_site_map",
    "load_site_map_payload",
    "load_site_map_split",
    "load_site_map_split_payload",
]
