"""Preflight caller-supplied traces for future TRAIN-only G0 diagnosis.

This does not collect episodes, infer actor actions, verify their origin, or classify
failures. The caller must independently verify actor/source/protocol provenance,
action-output parity, and the TRAIN-cell allocation before a real G0 diagnosis.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import re
from typing import Collection, Iterable, Sequence

import numpy as np

from common_adapter import ActionAdapter, ActionSpec, ObservationSpec


SHA256_REGEX = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class G0Identity:
    actor_sha256: str
    source_sha256: str
    protocol_sha256: str
    observation_fingerprint: str
    action_fingerprint: str
    action_mode: str
    reset_contract: str

    def __post_init__(self) -> None:
        for name in (
            "actor_sha256", "source_sha256", "protocol_sha256",
            "observation_fingerprint", "action_fingerprint",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or SHA256_REGEX.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 fingerprint")
        if self.action_mode != "exported_tanh_mean":
            raise ValueError("G0 requires the deterministic exported tanh(mean) actor")
        if not isinstance(self.reset_contract, str) or not self.reset_contract.strip():
            raise ValueError("G0 requires a declared reset/history contract")


@dataclass(frozen=True)
class G0Cell:
    partition: str
    geometry_seed: int
    track_id: int


@dataclass(frozen=True)
class G0Decision:
    step: int
    observation: np.ndarray
    proposed_native_action: np.ndarray
    executed_native_action: np.ndarray
    applied_official_action: np.ndarray


def validate_g0_trace(
    *,
    identity: G0Identity,
    actor_payload: bytes,
    observation_spec: ObservationSpec,
    action_spec: ActionSpec,
    cell: G0Cell,
    allowed_train_cells: Collection[tuple[int, int]],
    decisions: Iterable[G0Decision],
) -> int:
    """Validate declared actions; allowlist cells are (track_id, geometry_seed)."""
    if not isinstance(actor_payload, bytes) or not actor_payload:
        raise ValueError("actor payload must be nonempty bytes")
    if hashlib.sha256(actor_payload).hexdigest() != identity.actor_sha256:
        raise ValueError("actor bytes do not match the declared identity")
    if (
        observation_spec.fingerprint != identity.observation_fingerprint
        or action_spec.fingerprint != identity.action_fingerprint
    ):
        raise ValueError("observation/action adapters do not match the actor identity")
    if observation_spec.channel_order != "CHW":
        raise ValueError("G0 RLPD export requires CHW observations")
    if (
        cell.partition != "TRAIN"
        or type(cell.geometry_seed) is not int
        or not 0 <= cell.geometry_seed < 2**32
        or type(cell.track_id) is not int
        or cell.track_id <= 0
        or (cell.track_id, cell.geometry_seed) not in allowed_train_cells
    ):
        raise ValueError("G0 cell must belong to the audited TRAIN-only allowlist")

    adapter = ActionAdapter(action_spec)
    count = 0
    for decision in decisions:
        if type(decision.step) is not int or decision.step != count:
            raise ValueError("G0 decision steps must start at zero and be contiguous")
        observation_spec.validate(decision.observation)
        proposed = np.asarray(decision.proposed_native_action)
        executed = np.asarray(decision.executed_native_action)
        applied = np.asarray(decision.applied_official_action)
        if any(array.shape != (3,) or array.dtype != np.float32 for array in (proposed, executed, applied)):
            raise ValueError("G0 actions must be float32 triples in declared coordinates")
        if not np.isfinite(proposed).all() or not np.isfinite(executed).all() or not np.isfinite(applied).all():
            raise ValueError("G0 actions must be finite")
        if not np.allclose(proposed, executed, rtol=0, atol=1e-6):
            raise ValueError("G0 unchanged actor proposed and executed different actions")
        mapped = adapter.to_official(executed, clip=False)
        if (
            np.any(applied < action_spec.official_low)
            or np.any(applied > action_spec.official_high)
            or not np.allclose(mapped, applied, rtol=0, atol=1e-6)
        ):
            raise ValueError("G0 applied action differs from the executed native action")
        count += 1
    if count == 0:
        raise ValueError("G0 trace must contain at least one decision")
    return count


@dataclass(frozen=True)
class G0EventRules:
    max_decisions: int
    negative_reward_limit: int
    stall_window: int
    tile_window: int
    directed_delta_epsilon: float

    def __post_init__(self) -> None:
        for field in ("max_decisions", "negative_reward_limit", "stall_window", "tile_window"):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if not np.isfinite(self.directed_delta_epsilon) or self.directed_delta_epsilon < 0:
            raise ValueError("directed_delta_epsilon must be finite and nonnegative")


@dataclass(frozen=True)
class G0Telemetry:
    step: int
    summed_reward: float
    new_tiles: int | None
    directed_delta: float | None
    centerline_far: bool | None
    contact: bool | None
    damage: float | None
    progress: float | None
    finish_qualified: bool | None
    finish_phase: str | None
    finished: bool
    terminated: bool
    truncated: bool
    out_of_bounds: bool | None
    retire_reason: str | None


@dataclass(frozen=True)
class G0EpisodeSummary:
    outcome: str
    first_observed_step: int | None
    first_observed_events: tuple[str, ...]
    first_event_status: str
    missing_steps: tuple[int, ...]
    final_negative_streak: int
    qualified_nonfinish: bool | None
    unassessed_precursors: tuple[str, ...]


def classify_g0_episode(
    records: Sequence[G0Telemetry], *, rules: G0EventRules, end_reason: str,
) -> G0EpisodeSummary:
    """Classify observed precursors; names do not assert physical or causal failure."""
    if not records or len(records) > rules.max_decisions:
        raise ValueError("G0 episode must fit its frozen decision cap")
    if end_reason not in {"finished", "retired", "out_of_bounds", "task_timeout", "collection_censored", "unknown"}:
        raise ValueError("G0 end reason must distinguish task outcome from collection censoring")

    streak = tile_streak = motion_streak = 0
    previous_progress = previous_damage = None
    missing_steps = []
    first_step = None
    first_events: tuple[str, ...] = ()
    for index, record in enumerate(records):
        if type(record.step) is not int or record.step != index:
            raise ValueError("G0 telemetry steps must start at zero and be contiguous")
        if not np.isfinite(record.summed_reward):
            raise ValueError("G0 decision reward must be finite")
        if index < len(records) - 1 and (record.finished or record.terminated or record.truncated):
            raise ValueError("G0 episode continues after a terminal decision")
        for name in ("centerline_far", "contact", "finish_qualified", "out_of_bounds",
                     "finished", "terminated", "truncated"):
            value = getattr(record, name)
            if name in ("finished", "terminated", "truncated") and type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be a boolean or missing")
        if record.new_tiles is not None and (type(record.new_tiles) is not int or record.new_tiles < 0):
            raise ValueError("new_tiles must be nonnegative or missing")
        if record.directed_delta is not None and not np.isfinite(record.directed_delta):
            raise ValueError("directed_delta must be finite or missing")
        if record.damage is not None:
            if (type(record.damage) not in (int, float, np.float32, np.float64)
                    or not np.isfinite(record.damage) or not 0 <= record.damage <= 1):
                raise ValueError("damage must be finite in [0, 1] or missing")
            prior = 0.0 if index == 0 else previous_damage
            if prior is not None:
                increment = record.damage - prior
                if increment < -1e-6 or increment > 0.2 + 1e-6:
                    raise ValueError("wrapper damage cannot decrease or increase beyond one collision")
                if record.contact is False and increment > 1e-6:
                    raise ValueError("wrapper damage increased without observed contact")
            previous_damage = float(record.damage)
        else:
            previous_damage = None
        if record.progress is not None:
            if not np.isfinite(record.progress) or not 0 <= record.progress <= 1:
                raise ValueError("visited-tile progress must be in [0, 1] or missing")
            if previous_progress is not None and record.progress + 1e-6 < previous_progress:
                raise ValueError("cumulative visited-tile progress decreased within an episode")
            previous_progress = record.progress
        if record.retire_reason not in (None, "off_track", "crash"):
            raise ValueError("unrecognized G0 retirement reason")
        if index < len(records) - 1 and record.retire_reason is not None:
            raise ValueError("G0 episode continues after wrapper retirement")

        streak = streak + 1 if record.summed_reward < 0 else 0
        if streak > rules.negative_reward_limit and (
            streak != rules.negative_reward_limit + 1
            or not record.terminated
            or record.retire_reason not in ("off_track", "crash")
        ):
            raise ValueError("G0 episode continued past the wrapper negative-reward threshold")
        tile_streak = tile_streak + 1 if record.new_tiles == 0 else 0
        motion_streak = (
            motion_streak + 1
            if record.directed_delta is not None and record.directed_delta <= rules.directed_delta_epsilon
            else 0
        )
        missing = any(
            getattr(record, field) is None
            for field in (
                "new_tiles", "centerline_far", "contact", "damage",
                "progress", "finish_qualified", "finish_phase", "out_of_bounds",
            )
        ) or (record.directed_delta is None and record.centerline_far is not True)
        if missing:
            missing_steps.append(index)
        events = []
        if record.centerline_far:
            events.append("centerline_distance_exceeds_proxy")
        if record.contact:
            events.append("observed_contact")
        if motion_streak == rules.stall_window:
            events.append("low_directed_motion")
        if tile_streak == rules.tile_window:
            events.append("no_new_tiles")
        if events and first_step is None:
            first_step, first_events = index, tuple(sorted(events))

    last = records[-1]
    if last.retire_reason == "off_track" and streak != rules.negative_reward_limit + 1:
        raise ValueError("off_track retirement precedes its negative-reward threshold")
    if last.retire_reason == "crash" and (
        last.contact is False or last.damage is not None and last.damage < 1.0 - 1e-6
    ):
        raise ValueError("crash retirement requires final contact and full damage")
    if end_reason == "finished":
        if not last.finished or not (last.terminated or last.truncated):
            raise ValueError("finished outcome requires a real terminal finish")
        outcome = "finished"
    elif end_reason == "retired":
        if not last.terminated or last.finished or last.retire_reason is None:
            raise ValueError("retired outcome requires wrapper retirement without finish")
        outcome = last.retire_reason
    elif end_reason == "out_of_bounds":
        if not last.terminated or last.finished or last.retire_reason is not None or last.out_of_bounds is not True:
            raise ValueError("out_of_bounds requires a measured playfield exit without finish")
        outcome = "out_of_bounds"
    elif end_reason == "task_timeout":
        if not last.truncated or last.finished or last.retire_reason is not None:
            raise ValueError("task timeout must be a nonfinish without wrapper retirement")
        outcome = "task_timeout"
    elif end_reason == "collection_censored":
        if last.finished or last.terminated or last.truncated or last.retire_reason is not None:
            raise ValueError("collection censoring is not a terminal failure")
        outcome = "collection_censored"
    else:
        if last.finished or last.retire_reason is not None:
            raise ValueError("known outcome cannot be marked unknown")
        outcome = "unknown"

    uncertain = bool(missing_steps and (first_step is None or missing_steps[0] <= first_step))
    status = "unknown" if uncertain else "mixed" if len(first_events) > 1 else "observed" if first_events else "none"
    qualified_nonfinish = None if last.finish_qualified is None else bool(last.finish_qualified and outcome != "finished")
    return G0EpisodeSummary(
        outcome, first_step, first_events, status, tuple(missing_steps), streak, qualified_nonfinish,
        ("finish_phase",) if qualified_nonfinish else (),
    )


def summarize_g0_pairs(
    rows: Sequence[dict], *, cells: Sequence[tuple[int, int]], actor_ids: tuple[str, str],
) -> dict:
    """Summarize complete geometry-paired TRAIN diagnosis without ranking claims."""
    if len(set(cells)) != len(cells) or len(set(actor_ids)) != 2 or len(rows) != 2 * len(cells):
        raise ValueError("G0 summary requires a complete, distinct paired cell table")
    outcomes = {actor_id: Counter() for actor_id in actor_ids}
    event_statuses = {actor_id: Counter() for actor_id in actor_ids}
    first_events = {actor_id: Counter() for actor_id in actor_ids}
    actor_hashes = {}
    missing_decisions = Counter()
    qualified_nonfinish = Counter()
    paired = []
    for index, (track_id, seed) in enumerate(cells):
        first, second = rows[2 * index:2 * index + 2]
        for row, actor_id in ((first, actor_ids[0]), (second, actor_ids[1])):
            if (
                row.get("partition") != "TRAIN"
                or row.get("track_id") != track_id or row.get("geometry_seed") != seed
                or row.get("actor_id") != actor_id
                or type(row.get("steps")) is not int or row["steps"] <= 0
                or not isinstance(row.get("summary"), dict)
                or row["summary"].get("outcome") not in (
                    "finished", "off_track", "crash", "out_of_bounds",
                    "task_timeout", "collection_censored", "unknown",
                )
                or row["summary"].get("first_event_status") not in ("observed", "mixed", "none", "unknown")
                or not isinstance(row["summary"].get("first_observed_events"), (tuple, list))
                or not isinstance(row["summary"].get("missing_steps"), (tuple, list))
                or not isinstance(row.get("actor_sha256"), str)
                or SHA256_REGEX.fullmatch(row["actor_sha256"]) is None
            ):
                raise ValueError("G0 result is not the frozen TRAIN cell/actor schedule")
            if actor_id in actor_hashes and actor_hashes[actor_id] != row["actor_sha256"]:
                raise ValueError("G0 actor bytes changed across paired cells")
            actor_hashes[actor_id] = row["actor_sha256"]
            outcomes[actor_id][row["summary"]["outcome"]] += 1
            event_statuses[actor_id][row["summary"]["first_event_status"]] += 1
            for event in row["summary"]["first_observed_events"]:
                first_events[actor_id][event] += 1
            missing_decisions[actor_id] += len(row["summary"]["missing_steps"])
            qualified_nonfinish[actor_id] += row["summary"].get("qualified_nonfinish") is True
        road_hash = first.get("road_centerline_sha256")
        if (not isinstance(road_hash, str) or SHA256_REGEX.fullmatch(road_hash) is None
                or road_hash != second.get("road_centerline_sha256")):
            raise ValueError("G0 paired actors did not receive the same road bytes")
        paired.append({
            "track_id": track_id, "geometry_seed": seed,
            "outcomes": {actor_ids[0]: first["summary"]["outcome"],
                         actor_ids[1]: second["summary"]["outcome"]},
            "road_centerline_sha256": first["road_centerline_sha256"],
        })
    return {
        "geometry_clusters": len(cells), "diagnostic_episodes": len(rows),
        "outcomes_by_actor": {key: dict(value) for key, value in outcomes.items()},
        "first_event_status_by_actor": {key: dict(value) for key, value in event_statuses.items()},
        "first_events_by_actor": {key: dict(value) for key, value in first_events.items()},
        "missing_decisions_by_actor": {key: missing_decisions[key] for key in actor_ids},
        "qualified_nonfinish_by_actor": {key: qualified_nonfinish[key] for key in actor_ids},
        "paired_cells": paired,
        "interpretation": "observational TRAIN-only proxies; no official ranking, causal effect or matched training intervention",
    }
