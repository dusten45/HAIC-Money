"""Compact, train-only aggregates for visible hazard exposure during PPO rollouts."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from haic_agent.pixel_features import FEATURE_NAMES


PROGRESS_BIN_WIDTH = 0.05
PROGRESS_BIN_COUNT = 20
STEER_ALIGNMENT_NEUTRAL_PIXELS = 1.0
TARGET_PROGRESS_BINS = {
    "0.60-0.65": 12,
    "0.75-0.80": 15,
}

_BRAKE_CONDITIONS = (
    "obstacle_present",
    "urgent",
    "risk_positive",
)
_URGENCY_BAND_COUNT = 4


def _finite_number(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _empty_bin(index: int) -> dict[str, Any]:
    return {
        "bin_index": index,
        "progress_start": round(index * PROGRESS_BIN_WIDTH, 2),
        "progress_end": round((index + 1) * PROGRESS_BIN_WIDTH, 2),
        "all_decisions": 0,
        "feature_observations": 0,
        "perception_unavailable_count": 0,
        "obstacle_present_count": 0,
        "urgent_count": 0,
        "risk_positive_count": 0,
        "risk_positive_sum": 0.0,
        "risk_sum": 0.0,
        "risk_max": None,
        "gas_sum": 0.0,
        "gas_max": None,
        "gas_fraction_sum": 0.0,
        "gas_fraction_max": 0.0,
        "gas_active_count": 0,
        "brake_sum": 0.0,
        "brake_max": None,
        "brake_fraction_sum": 0.0,
        "brake_fraction_max": 0.0,
        "brake_active_count": 0,
        "next_collision_count": 0,
        "damage_delta_sum": 0.0,
        "collision_followup_count": 0,
        "collision_followup_unobserved_count": 0,
        "urgency_bands": [_empty_urgency_band(index) for index in range(_URGENCY_BAND_COUNT)],
        **{
            f"{condition}_brake_active_count": 0
            for condition in _BRAKE_CONDITIONS
        },
        **{f"{condition}_brake_sum": 0.0 for condition in _BRAKE_CONDITIONS},
        **{
            f"{condition}_brake_sample_count": 0
            for condition in _BRAKE_CONDITIONS
        },
    }


def _empty_urgency_band(index: int) -> dict[str, Any]:
    return {
        "urgency_band_index": index,
        "urgency_start": round(index / _URGENCY_BAND_COUNT, 2),
        "urgency_end": round((index + 1) / _URGENCY_BAND_COUNT, 2),
        "obstacle_observations": 0,
        "signed_obstacle_offset_sum": 0.0,
        "signed_obstacle_offset_min": None,
        "signed_obstacle_offset_max": None,
        "car_relative_obstacle_offset_px_sum": 0.0,
        "car_relative_obstacle_offset_px_min": None,
        "car_relative_obstacle_offset_px_max": None,
        "steer_away_alignment_observations": 0,
        "steer_away_alignment_sum": 0.0,
        "steer_away_alignment_min": None,
        "steer_away_alignment_max": None,
        "steer_away_aligned_count": 0,
        "steer_away_misaligned_count": 0,
        "steer_away_neutral_count": 0,
        "relative_offset_neutral_count": 0,
        "collision_next_count": 0,
    }


def _finalize_urgency_band(values: dict[str, Any]) -> dict[str, Any]:
    result = dict(values)
    count = int(values["obstacle_observations"])
    alignment_count = int(values["steer_away_alignment_observations"])
    result["signed_obstacle_offset_mean"] = (
        float(values["signed_obstacle_offset_sum"]) / count if count else None
    )
    result["car_relative_obstacle_offset_px_mean"] = (
        float(values["car_relative_obstacle_offset_px_sum"]) / count if count else None
    )
    result["steer_away_alignment_mean"] = (
        float(values["steer_away_alignment_sum"]) / alignment_count
        if alignment_count
        else None
    )
    result["steer_away_aligned_rate"] = (
        int(values["steer_away_aligned_count"]) / alignment_count
        if alignment_count
        else None
    )
    result["steer_away_misaligned_rate"] = (
        int(values["steer_away_misaligned_count"]) / alignment_count
        if alignment_count
        else None
    )
    result["collision_next_rate"] = (
        int(values["collision_next_count"]) / count if count else None
    )
    return result


def _finalize_collision_followup(values: dict[str, Any]) -> dict[str, Any]:
    result = dict(values)
    count = int(values["count"])
    for metric in (
        "followup_progress_change",
        "followup_speed_change",
        "followup_damage_delta",
        "followup_damage_change_from_collision",
        "followup_steer",
        "followup_gas",
        "followup_brake",
    ):
        result[f"{metric}_mean"] = (
            float(values[f"{metric}_sum"]) / count if count else None
        )
    return result


def _mean(total: float, count: int, *, unavailable: bool = False) -> float | None:
    if count:
        return total / count
    if unavailable:
        return None
    return 0.0


def _rate(numerator: int, denominator: int, *, no_decisions: bool) -> float | None:
    if denominator:
        return numerator / denominator
    if no_decisions:
        return 0.0
    return None


def _finalize_bin(values: dict[str, Any]) -> dict[str, Any]:
    result = dict(values)
    decisions = int(values["all_decisions"])
    feature_count = int(values["feature_observations"])
    result["risk_mean"] = _mean(
        float(values["risk_sum"]),
        feature_count,
        unavailable=decisions > 0 and feature_count == 0,
    )
    result["risk_positive_mean"] = _mean(
        float(values["risk_positive_sum"]),
        int(values["risk_positive_count"]),
        unavailable=decisions > 0 and feature_count == 0,
    )
    if values["risk_max"] is None:
        result["risk_max"] = None if decisions > 0 and feature_count == 0 else 0.0
    result["gas_mean"] = _mean(float(values["gas_sum"]), decisions)
    result["gas_fraction_mean"] = _mean(
        float(values["gas_fraction_sum"]), decisions
    )
    if values["gas_max"] is None:
        result["gas_max"] = 0.0
    result["gas_fraction_max"] = float(values["gas_fraction_max"])
    result["gas_active_rate"] = _rate(
        int(values["gas_active_count"]), decisions, no_decisions=decisions == 0
    )
    result["brake_mean"] = _mean(float(values["brake_sum"]), decisions)
    result["brake_fraction_mean"] = _mean(
        float(values["brake_fraction_sum"]), decisions
    )
    if values["brake_max"] is None:
        result["brake_max"] = 0.0
    result["brake_fraction_max"] = float(values["brake_fraction_max"])
    result["brake_active_rate"] = _rate(
        int(values["brake_active_count"]), decisions, no_decisions=decisions == 0
    )
    for condition in _BRAKE_CONDITIONS:
        count = int(values[f"{condition}_brake_sample_count"])
        active_count = int(values[f"{condition}_brake_active_count"])
        result[f"{condition}_brake_mean"] = _mean(
            float(values[f"{condition}_brake_sum"]),
            count,
            unavailable=decisions > 0 and count == 0,
        )
        result[f"{condition}_brake_active_rate"] = _rate(
            active_count, count, no_decisions=decisions == 0
        )
        result[f"{condition}_decision_count"] = count
    return result


class HazardExposureRecorder:
    """Accumulate per-episode, per-map, per-progress-bin summaries only."""

    def __init__(self) -> None:
        self._episodes: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None

    def begin_episode(
        self,
        *,
        update_number: int,
        episode_number: int,
        episode: Mapping[str, Any],
    ) -> None:
        if self._active is not None:
            raise RuntimeError("the previous hazard exposure episode is still active")
        if (
            isinstance(update_number, bool)
            or not isinstance(update_number, int)
            or update_number < 0
        ):
            raise ValueError("update_number must be a non-negative integer")
        if (
            isinstance(episode_number, bool)
            or not isinstance(episode_number, int)
            or episode_number < 0
        ):
            raise ValueError("episode_number must be a non-negative integer")
        map_id = episode.get("map_id")
        if map_id is None and episode.get("track_id") is not None:
            map_id = f"official-track-{int(episode['track_id'])}"
        if not isinstance(map_id, str) or not map_id:
            raise ValueError("episode identity must include a map_id or track_id")
        seed = episode.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("episode identity must include an integer seed")

        identity = {
            "map_id": map_id,
            "map_kind": str(episode.get("map_kind", "official")),
            "seed": seed,
        }
        if episode.get("source_map") is not None:
            identity["source_map"] = str(episode["source_map"])
        self._active = {
            **identity,
            "update_number": int(update_number),
            "episode_number": int(episode_number),
            "episode_id": f"u{int(update_number)}-e{int(episode_number)}-{map_id}-{seed}",
            "start_progress": 0.0,
            "end_progress": 0.0,
            "max_pre_action_progress": 0.0,
            "decision_count": 0,
            "collision_count": 0,
            "damage_delta_sum": 0.0,
            "pending_collision": None,
            "collision_followup_unobserved_count": 0,
            "collision_followups": {},
            "bins": [_empty_bin(index) for index in range(PROGRESS_BIN_COUNT)],
        }

    def record_decision(
        self,
        *,
        progress: float,
        damage_delta: float,
        features: np.ndarray | None,
        risk: float | None,
        action: np.ndarray,
        next_collision: bool,
        next_progress: float,
        pre_speed: float,
        next_speed: float,
        next_damage: float,
        throttle_limit: float = 1.0,
        brake_limit: float = 1.0,
    ) -> None:
        if self._active is None:
            raise RuntimeError("begin_episode must be called before record_decision")
        progress_value = _finite_number("progress", progress)
        damage_value = _finite_number("damage_delta", damage_delta)
        next_progress_value = _finite_number("next_progress", next_progress)
        pre_speed_value = _finite_number("pre_speed", pre_speed)
        next_speed_value = _finite_number("next_speed", next_speed)
        next_damage_value = _finite_number("next_damage", next_damage)
        action_values = np.asarray(action, dtype=np.float32)
        if action_values.shape != (3,) or not np.all(np.isfinite(action_values)):
            raise ValueError("action must be a finite array with shape (3,)")
        throttle = _finite_number("throttle_limit", throttle_limit)
        brake = _finite_number("brake_limit", brake_limit)
        if throttle <= 0.0 or brake <= 0.0:
            raise ValueError("pedal limits must be positive")

        feature_values: np.ndarray | None
        if features is None:
            feature_values = None
            if risk is not None:
                raise ValueError("risk must be None when visual features are unavailable")
        else:
            feature_values = np.asarray(features, dtype=np.float32)
            if feature_values.shape != (len(FEATURE_NAMES),):
                raise ValueError(
                    f"features must have shape ({len(FEATURE_NAMES)},)"
                )
            if not np.all(np.isfinite(feature_values)):
                raise ValueError("features must be finite")
            if risk is None:
                raise ValueError("risk is required when visual features are available")
            risk = _finite_number("risk", risk)
            if risk < 0.0 or risk > 1.0:
                raise ValueError("risk must be between 0 and 1")

        bounded_progress = float(np.clip(progress_value, 0.0, 1.0))
        bin_index = min(
            PROGRESS_BIN_COUNT - 1,
            int(math.floor((bounded_progress + 1e-12) / PROGRESS_BIN_WIDTH)),
        )
        row = self._active["bins"][bin_index]
        gas_value = float(action_values[1])
        brake_value = float(action_values[2])
        gas_fraction = float(np.clip(gas_value / throttle, 0.0, 1.0))
        brake_fraction = float(np.clip(brake_value / brake, 0.0, 1.0))

        row["all_decisions"] += 1
        row["gas_sum"] += gas_value
        row["gas_max"] = gas_value if row["gas_max"] is None else max(row["gas_max"], gas_value)
        row["brake_sum"] += brake_value
        row["brake_max"] = (
            brake_value
            if row["brake_max"] is None
            else max(row["brake_max"], brake_value)
        )
        row["gas_fraction_sum"] = row.get("gas_fraction_sum", 0.0) + gas_fraction
        row["gas_fraction_max"] = max(row.get("gas_fraction_max", 0.0), gas_fraction)
        row["brake_fraction_sum"] = row.get("brake_fraction_sum", 0.0) + brake_fraction
        row["brake_fraction_max"] = max(
            row.get("brake_fraction_max", 0.0), brake_fraction
        )
        if gas_value > 0.0:
            row["gas_active_count"] += 1
        if brake_value > 0.0:
            row["brake_active_count"] += 1
        if bool(next_collision):
            row["next_collision_count"] += 1
            self._active["collision_count"] += 1
        row["damage_delta_sum"] += damage_value
        self._active["damage_delta_sum"] += damage_value
        self._active["decision_count"] += 1
        self._active["max_pre_action_progress"] = max(
            self._active["max_pre_action_progress"], bounded_progress
        )

        pending_collision = self._active["pending_collision"]
        if pending_collision is not None:
            row["collision_followup_count"] += 1
            pair_key = (pending_collision["origin_bin_index"], bin_index)
            pair = self._active["collision_followups"].setdefault(
                pair_key,
                {
                    "origin_bin_index": pair_key[0],
                    "next_bin_index": pair_key[1],
                    "count": 0,
                    "next_collision_count": 0,
                    "followup_progress_change_sum": 0.0,
                    "followup_speed_change_sum": 0.0,
                    "followup_damage_delta_sum": 0.0,
                    "followup_damage_change_from_collision_sum": 0.0,
                    "followup_steer_sum": 0.0,
                    "followup_gas_sum": 0.0,
                    "followup_brake_sum": 0.0,
                },
            )
            pair["count"] += 1
            pair["next_collision_count"] += int(bool(next_collision))
            pair["followup_progress_change_sum"] += (
                bounded_progress - pending_collision["collision_state_progress"]
            )
            pair["followup_speed_change_sum"] += (
                pre_speed_value - pending_collision["collision_state_speed"]
            )
            pair["followup_damage_delta_sum"] += damage_value
            pair["followup_damage_change_from_collision_sum"] += (
                next_damage_value - pending_collision["collision_state_damage"]
            )
            pair["followup_steer_sum"] += float(action_values[0])
            pair["followup_gas_sum"] += gas_value
            pair["followup_brake_sum"] += brake_value
            self._active["pending_collision"] = None

        if feature_values is None:
            row["perception_unavailable_count"] += 1
        else:
            assert risk is not None
            obstacle_present = float(feature_values[4]) >= 0.5
            urgent = float(feature_values[6]) >= 0.5
            risk_positive = risk > 0.0
            row["feature_observations"] += 1
            row["risk_sum"] += risk
            row["risk_max"] = risk if row["risk_max"] is None else max(row["risk_max"], risk)
            if obstacle_present:
                row["obstacle_present_count"] += 1
            if urgent:
                row["urgent_count"] += 1
            if risk_positive:
                row["risk_positive_count"] += 1
                row["risk_positive_sum"] += risk
            for condition, matched in (
                ("obstacle_present", obstacle_present),
                ("urgent", urgent),
                ("risk_positive", risk_positive),
            ):
                if matched:
                    row[f"{condition}_brake_sample_count"] += 1
                    row[f"{condition}_brake_sum"] += brake_value
                    if brake_value > 0.0:
                        row[f"{condition}_brake_active_count"] += 1

            if obstacle_present:
                urgency = float(np.clip(feature_values[6], 0.0, 1.0))
                urgency_index = min(
                    _URGENCY_BAND_COUNT - 1,
                    int(math.floor(urgency * _URGENCY_BAND_COUNT + 1e-12)),
                )
                urgency_row = row["urgency_bands"][urgency_index]
                offset = float(np.clip(feature_values[5], -1.0, 1.0))
                road_center_offset = float(np.clip(feature_values[1], -1.0, 1.0))
                # Match visible_hazard_risk(): the pixel lateral feature is
                # relative to road center, so add the car's road-center offset
                # to obtain the obstacle's signed position relative to the car.
                relative_offset_px = offset * 12.0 + road_center_offset * 42.0
                urgency_row["obstacle_observations"] += 1
                urgency_row["signed_obstacle_offset_sum"] += offset
                urgency_row["signed_obstacle_offset_min"] = (
                    offset
                    if urgency_row["signed_obstacle_offset_min"] is None
                    else min(urgency_row["signed_obstacle_offset_min"], offset)
                )
                urgency_row["signed_obstacle_offset_max"] = (
                    offset
                    if urgency_row["signed_obstacle_offset_max"] is None
                    else max(urgency_row["signed_obstacle_offset_max"], offset)
                )
                urgency_row["car_relative_obstacle_offset_px_sum"] += relative_offset_px
                urgency_row["car_relative_obstacle_offset_px_min"] = (
                    relative_offset_px
                    if urgency_row["car_relative_obstacle_offset_px_min"] is None
                    else min(
                        urgency_row["car_relative_obstacle_offset_px_min"],
                        relative_offset_px,
                    )
                )
                urgency_row["car_relative_obstacle_offset_px_max"] = (
                    relative_offset_px
                    if urgency_row["car_relative_obstacle_offset_px_max"] is None
                    else max(
                        urgency_row["car_relative_obstacle_offset_px_max"],
                        relative_offset_px,
                    )
                )
                if abs(relative_offset_px) <= STEER_ALIGNMENT_NEUTRAL_PIXELS:
                    urgency_row["relative_offset_neutral_count"] += 1
                else:
                    # Screen x grows rightward; the signed policy action maps
                    # rightward in car-local coordinates after CarRacing's
                    # step() applies -action[0] to the wheel steering angle.
                    steer_away_alignment = (
                        float(action_values[0]) * -float(np.sign(relative_offset_px))
                    )
                    urgency_row["steer_away_alignment_observations"] += 1
                    urgency_row["steer_away_alignment_sum"] += steer_away_alignment
                    urgency_row["steer_away_alignment_min"] = (
                        steer_away_alignment
                        if urgency_row["steer_away_alignment_min"] is None
                        else min(
                            urgency_row["steer_away_alignment_min"],
                            steer_away_alignment,
                        )
                    )
                    urgency_row["steer_away_alignment_max"] = (
                        steer_away_alignment
                        if urgency_row["steer_away_alignment_max"] is None
                        else max(
                            urgency_row["steer_away_alignment_max"],
                            steer_away_alignment,
                        )
                    )
                    if steer_away_alignment > 0.0:
                        urgency_row["steer_away_aligned_count"] += 1
                    elif steer_away_alignment < 0.0:
                        urgency_row["steer_away_misaligned_count"] += 1
                    else:
                        urgency_row["steer_away_neutral_count"] += 1
                if bool(next_collision):
                    urgency_row["collision_next_count"] += 1

        if bool(next_collision):
            self._active["pending_collision"] = {
                "origin_bin_index": bin_index,
                "collision_state_progress": float(
                    np.clip(next_progress_value, 0.0, 1.0)
                ),
                "collision_state_speed": next_speed_value,
                "collision_state_damage": next_damage_value,
            }

    def end_episode(
        self,
        *,
        boundary: str,
        end_progress: float,
        terminated: bool,
        truncated: bool,
        finished: bool,
        off_track: bool,
    ) -> None:
        if self._active is None:
            raise RuntimeError("no hazard exposure episode is active")
        end_value = _finite_number("end_progress", end_progress)
        pending_collision = self._active["pending_collision"]
        if pending_collision is not None:
            origin = pending_collision["origin_bin_index"]
            self._active["bins"][origin]["collision_followup_unobserved_count"] += 1
            self._active["collision_followup_unobserved_count"] += 1
        bins = []
        for row in self._active["bins"]:
            finalized = _finalize_bin(row)
            finalized["urgency_bands"] = [
                _finalize_urgency_band(band) for band in row["urgency_bands"]
            ]
            bins.append(finalized)
        reached = [row for row in bins if row["all_decisions"] > 0]
        target_coverage = {
            name: {
                "bin_index": index,
                "reached": bins[index]["all_decisions"] > 0,
                "all_decisions": bins[index]["all_decisions"],
                "risk_positive_count": bins[index]["risk_positive_count"],
                "urgent_count": bins[index]["urgent_count"],
                "obstacle_present_count": bins[index]["obstacle_present_count"],
            }
            for name, index in TARGET_PROGRESS_BINS.items()
        }
        result = {
            **{
                key: value
                for key, value in self._active.items()
                if key not in {"bins", "pending_collision", "collision_followups"}
            },
            "end_progress": end_value,
            "boundary": str(boundary),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "finished": bool(finished),
            "off_track": bool(off_track),
            "coverage": {
                "reached_bin_count": len(reached),
                "unreached_bin_count": PROGRESS_BIN_COUNT - len(reached),
                "target_bins": target_coverage,
            },
            "bins": bins,
            "collision_followups": [
                _finalize_collision_followup(values)
                for _, values in sorted(
                    self._active["collision_followups"].items()
                )
            ],
        }
        self._episodes.append(result)
        self._active = None

    def to_dict(self) -> dict[str, Any]:
        if self._active is not None:
            raise RuntimeError("cannot serialize while a hazard exposure episode is active")
        return {
            "schema_version": 1,
            "progress_bin_width": PROGRESS_BIN_WIDTH,
            "progress_bin_count": PROGRESS_BIN_COUNT,
            "risk_definition": "training.train_policy.visible_hazard_risk(pre_action_pixel_features)",
            "steer_away_alignment_definition": (
                "relative_px = obstacle_lateral_offset * 12 + road_center_offset * 42; "
                "action[0] * -sign(relative_px); positive means away; abs(relative_px) "
                f"<= {STEER_ALIGNMENT_NEUTRAL_PIXELS} px is excluded"
            ),
            "urgency_bins": [
                {"index": index, "start": index / _URGENCY_BAND_COUNT,
                 "end": (index + 1) / _URGENCY_BAND_COUNT}
                for index in range(_URGENCY_BAND_COUNT)
            ],
            "action_summary_units": {
                "gas": "simulator action units",
                "brake": "simulator action units",
                "fractions": "fraction of actor pedal limit",
            },
            "episodes": [dict(episode) for episode in self._episodes],
        }


__all__ = [
    "HazardExposureRecorder",
    "PROGRESS_BIN_COUNT",
    "PROGRESS_BIN_WIDTH",
    "TARGET_PROGRESS_BINS",
]
