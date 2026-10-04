"""Custom-only episode loading and fixed-input diagnostics for PPO screens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from haic_agent.pixel_features import FEATURE_NAMES
from training.env_factory import (
    TrainingEpisode,
    create_episode_environment,
    describe_episode,
)
from training.site_maps import SiteMapEpisode, SiteMapSplit, load_site_map
from training.train_policy import close_training_environment


def load_custom_train_tune_split(split_path: Path) -> SiteMapSplit:
    """Materialize only the registered TRAIN and TUNE custom geometries."""
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("site-map split must be a JSON object")

    def materialize(group_name: str) -> tuple[SiteMapEpisode, ...]:
        entries = payload.get(group_name)
        if not isinstance(entries, list):
            raise ValueError(f"site-map split {group_name!r} must be an array")
        episodes: list[SiteMapEpisode] = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("map"), str):
                raise ValueError(f"each {group_name} entry must name a map")
            seeds = entry.get("seeds")
            if not isinstance(seeds, list) or not seeds:
                raise ValueError(f"each {group_name} entry must contain at least one seed")
            source_path = (split_path.parent / entry["map"]).resolve()
            site_map = load_site_map(source_path)
            if site_map.map_kind != "custom":
                raise ValueError(f"non-custom geometry is not allowed in {group_name}: {source_path.name}")
            for seed in seeds:
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise ValueError(f"{group_name} seeds must be integers")
                episodes.append(
                    SiteMapEpisode(site_map=site_map, seed=seed, source_path=source_path)
                )
        return tuple(episodes)

    train = materialize("train")
    tune = materialize("tune")
    if len(train) != 4 or len(tune) != 2:
        raise ValueError(f"expected exactly 4 TRAIN and 2 TUNE episodes; got {len(train)} and {len(tune)}")
    if {episode.map_id for episode in train} & {episode.map_id for episode in tune}:
        raise ValueError("TRAIN and TUNE map families must not overlap")
    if any(episode.site_map.map_kind != "custom" for episode in (*train, *tune)):
        raise ValueError("only custom TRAIN/TUNE geometry may be materialized")
    return SiteMapSplit(train=train, tune=tune, held_out=())


def _feature_vector(model: Any) -> np.ndarray:
    visual_features = getattr(model, "last_visual_features", None)
    if visual_features is None:
        raise ValueError("trace capture requires actor visual features for each observation")
    if isinstance(visual_features, torch.Tensor):
        visual_features = visual_features.detach().cpu().numpy()
    feature_values = np.asarray(visual_features, dtype=np.float32).reshape(-1)
    if feature_values.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"actor visual features must have shape ({len(FEATURE_NAMES)},)")
    if not np.isfinite(feature_values).all():
        raise ValueError("actor visual features must be finite")
    return feature_values


def _action_summary(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    speeds = [row["speed"] for row in rows if "speed" in row]
    gases = [row["gas"] for row in rows if "gas" in row]
    brakes = [row["brake"] for row in rows if "brake" in row]
    return {
        "decisions": len(rows),
        "mean_speed": float(np.mean(speeds)) if speeds else None,
        "mean_gas": float(np.mean(gases)) if gases else None,
        "mean_brake": float(np.mean(brakes)) if brakes else None,
        "brake_active_fraction_gt_0_02": (
            float(np.mean([brake > 0.02 for brake in brakes])) if brakes else None
        ),
    }


def _pre_visible_braking(rows: list[dict[str, Any]], *, horizon: int = 3) -> dict[str, Any]:
    visible = [row["obstacle_present"] > 0.5 for row in rows]
    onsets = [
        index
        for index, is_visible in enumerate(visible)
        if is_visible and (index == 0 or not visible[index - 1])
    ]
    immediate = [rows[index - 1] for index in onsets if index > 0 and not visible[index - 1]]
    preceding_indices = {
        index
        for onset in onsets
        for index in range(max(0, onset - horizon), onset)
        if not visible[index]
    }
    preceding = [rows[index] for index in sorted(preceding_indices)]
    return {
        "definition": "actions immediately before a 0-to-visible cue and over the deduplicated preceding 3 clear decisions",
        "visible_onsets": len(onsets),
        "immediate_pre_onset": _action_summary(immediate),
        "within_3_decisions_before_onset": _action_summary(preceding),
    }


def evaluate_policy_with_traces(
    model: Any,
    episodes: Iterable[TrainingEpisode],
    *,
    max_decisions: int,
    capture_observations: bool = False,
) -> dict[str, Any]:
    """Run deterministic episodes and preserve every pre-action pixel/action/outcome row."""
    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    episode_records: list[dict[str, Any]] = []
    observation_rows: list[np.ndarray] = []
    observation_metadata: list[dict[str, Any]] = []
    was_training = getattr(model, "training", None)
    if callable(getattr(model, "eval", None)):
        model.eval()
    try:
        for episode in episodes:
            environment = create_episode_environment(episode, max_decisions=max_decisions)
            observation, _ = environment.reset()
            start_time = float(getattr(environment.unwrapped, "t", 0.0))
            decision_trace: list[dict[str, Any]] = []
            previous_collision = False
            previous_off_track = False
            final_transition = None
            try:
                for decision_index in range(max_decisions):
                    pixel_observation = np.asarray(observation, dtype=np.float32).copy()
                    with torch.no_grad():
                        policy_output = model(torch.from_numpy(pixel_observation).unsqueeze(0))
                        action_tensor = model.deterministic_actions(policy_output)
                    feature_values = _feature_vector(model)
                    action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    transition = environment.step_transition(action)
                    labels = transition.labels
                    collision = bool(labels.collision)
                    off_track = bool(labels.off_track)
                    row = {
                        "decision": decision_index + 1,
                        "progress": float(labels.tile_progress),
                        "speed": float(labels.speed),
                        "road_curve_magnitude": float(feature_values[3]),
                        "obstacle_present": float(feature_values[4]),
                        "obstacle_lateral_offset": float(feature_values[5]),
                        "obstacle_urgency": float(feature_values[6]),
                        "steer": float(action[0]),
                        "gas": float(action[1]),
                        "brake": float(action[2]),
                        "collision": collision,
                        "collision_onset": collision and not previous_collision,
                        "off_track": off_track,
                        "off_track_onset": off_track and not previous_off_track,
                        "damage": float(labels.damage),
                        "finished": bool(labels.finished),
                    }
                    decision_trace.append(row)
                    if capture_observations:
                        observation_rows.append(pixel_observation)
                        observation_metadata.append(
                            {
                                **describe_episode(episode),
                                "decision": decision_index + 1,
                                "progress": row["progress"],
                                "speed": row["speed"],
                                "road_curve_magnitude": row["road_curve_magnitude"],
                                "obstacle_present": row["obstacle_present"],
                                "obstacle_lateral_offset": row["obstacle_lateral_offset"],
                                "obstacle_urgency": row["obstacle_urgency"],
                            }
                        )
                    previous_collision = collision
                    previous_off_track = off_track
                    final_transition = transition
                    observation = transition.next_observation
                    if transition.terminated or transition.truncated:
                        break
            finally:
                close_training_environment(environment)

            labels = final_transition.labels if final_transition is not None else None
            finished = bool(labels.finished) if labels is not None else False
            finish_value = getattr(environment.unwrapped, "finish_time_s", None)
            lap_time_ms = (
                (float(finish_value) - start_time) * 1000.0
                if finished and finish_value is not None
                else None
            )
            high_curve_rows = [
                row for row in decision_trace if row["road_curve_magnitude"] >= 0.5
            ]
            episode_records.append(
                {
                    **describe_episode(episode),
                    "completed": finished,
                    "lap_time_ms": lap_time_ms,
                    "under_13": bool(lap_time_ms is not None and lap_time_ms < 13000.0),
                    "final_progress": float(labels.tile_progress) if labels is not None else 0.0,
                    "max_progress": max((row["progress"] for row in decision_trace), default=0.0),
                    "final_damage": float(labels.damage) if labels is not None else 0.0,
                    "max_damage": max((row["damage"] for row in decision_trace), default=0.0),
                    "collision_decisions": sum(row["collision"] for row in decision_trace),
                    "collision_onsets": sum(row["collision_onset"] for row in decision_trace),
                    "off_track": any(row["off_track"] for row in decision_trace),
                    "off_track_onsets": sum(row["off_track_onset"] for row in decision_trace),
                    "mean_speed": (
                        float(np.mean([row["speed"] for row in decision_trace]))
                        if decision_trace
                        else 0.0
                    ),
                    "high_curvature_actions": _action_summary(high_curve_rows),
                    "pre_visible_hazard_braking": _pre_visible_braking(decision_trace),
                    "decision_trace": decision_trace,
                }
            )
    finally:
        if was_training is True and callable(getattr(model, "train", None)):
            model.train()

    bank = (
        np.stack(observation_rows).astype(np.float32, copy=False)
        if observation_rows
        else np.empty((0, 4, 84, 84), dtype=np.float32)
    )
    return {
        "episodes": episode_records,
        "observations": bank if capture_observations else None,
        "observation_metadata": observation_metadata if capture_observations else [],
    }


def replay_actor_on_observation_bank(
    model: Any,
    observations: np.ndarray,
    observation_metadata: list[dict[str, Any]],
    *,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Evaluate deterministic actions on fixed pixels without stepping an environment."""
    pixels = np.asarray(observations, dtype=np.float32)
    if pixels.ndim != 4 or pixels.shape[1:] != (4, 84, 84):
        raise ValueError("observation bank must have shape (N, 4, 84, 84)")
    if len(pixels) != len(observation_metadata):
        raise ValueError("observation metadata count must match the pixel bank")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    was_training = getattr(model, "training", None)
    if callable(getattr(model, "eval", None)):
        model.eval()
    action_rows: list[dict[str, Any]] = []
    try:
        for start in range(0, len(pixels), batch_size):
            stop = min(len(pixels), start + batch_size)
            batch = torch.from_numpy(pixels[start:stop])
            with torch.no_grad():
                output = model(batch)
                actions = model.deterministic_actions(output).detach().cpu().numpy()
            for offset, action in enumerate(actions):
                action_rows.append(
                    {
                        **observation_metadata[start + offset],
                        "steer": float(action[0]),
                        "gas": float(action[1]),
                        "brake": float(action[2]),
                    }
                )
    finally:
        if was_training is True and callable(getattr(model, "train", None)):
            model.train()

    masks = {
        "all": [True] * len(action_rows),
        "urgent_visible": [
            row["obstacle_present"] > 0.5 and row["obstacle_urgency"] >= 0.8
            for row in action_rows
        ],
        "high_curve": [row["road_curve_magnitude"] >= 0.5 for row in action_rows],
    }
    masks["intersection"] = [
        urgent and curved
        for urgent, curved in zip(masks["urgent_visible"], masks["high_curve"])
    ]
    groups = {
        name: _action_summary(
            [row for row, include in zip(action_rows, mask) if include]
        )
        for name, mask in masks.items()
    }
    return {"rows": action_rows, "groups": groups}


__all__ = [
    "evaluate_policy_with_traces",
    "load_custom_train_tune_split",
    "replay_actor_on_observation_bank",
]
