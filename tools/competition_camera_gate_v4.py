"""Preregistered practical gate for fresh, source-bound camera policy v4.

Three rejected earlier studies justify reducing the required finish gains and
increasing the paired finish-loss budget before any v4 geometry is bound.
Aggregate safety, runtime, exact-repeat, seed-cluster, pace and track gates
remain unchanged. Earlier studies retain their own frozen decisions.

The thresholds below are part of this module's bytes.  The comparator never
reads a protocol or caller-supplied threshold object, so changing a protocol
cannot change a decision after evaluation is bound.
"""

from __future__ import annotations

import math


PHASES = ("screen", "confirmation", "blind")
ARMS = ("control", "candidate")
_MIN_NET_GAIN = (2, 4, 2)
_MAX_LOST_FINISHES = (3, 6, 3)
_MAX_SHARED_TIME_RATIO = 1.10
_MIN_COMBINED_NET_GAIN = 12
_SEED_LOSS_CLUSTER = 2
_SEEDS_PER_PHASE = (8, 16, 8)
_REPEATS_PER_PHASE = (2, 4, 2)

# Exported for a runner to bind and validate its protocol.  The functions use
# only the immutable scalar/tuple values above, even if a caller mutates this
# export in memory.
THRESHOLDS = {
    "minimum_net_finish_gain": dict(zip(PHASES, _MIN_NET_GAIN)),
    "maximum_lost_control_finishes": dict(zip(PHASES, _MAX_LOST_FINISHES)),
    "maximum_aggregate_crash_increase": 0,
    "maximum_aggregate_contact_increase": 0,
    "maximum_aggregate_damage_increase": 0.0,
    "minimum_mean_progress_delta": 0.0,
    "maximum_aggregate_shared_finish_time_ratio": _MAX_SHARED_TIME_RATIO,
    "minimum_combined_net_finish_gain": _MIN_COMBINED_NET_GAIN,
    "minimum_seed_lost_control_finishes_for_negative_net_veto": _SEED_LOSS_CLUSTER,
    "minimum_combined_track_net_finish_gain": 0,
}

_RUNTIME_LIMITS = (
    ("initialization_ms", 10_000.0),
    ("reset_ms", 5_000.0),
    ("action_latency_max_ms", 5_000.0),
    ("peak_worker_rss_mib", 1_024.0),
)
_REPEAT_FIELDS = (
    "finished", "progress", "lap_time_ms", "retire_reason",
    "collision_count", "damage", "steps", "offtrack_samples",
    "partial_offtrack_samples", "action_trace_sha256",
)
_OUTCOME_FIELDS = (
    "finished", "progress", "lap_time_ms", "collision_count",
    "damage", "retire_reason",
)
_IDENTITY_FIELDS = (
    "protocol_sha256", "control_agent_sha256", "candidate_agent_sha256",
    "model_sha256", "harness_sha256", "runner_sha256",
    "decision_engine_sha256", "gate_sha256", "source_sha256",
    "helper_sha256", "environment_sha256",
)
_REQUIRED_IDENTITY_FIELDS = (
    "protocol_sha256", "candidate_agent_sha256", "runner_sha256",
    "decision_engine_sha256",
)
_DNF_REASONS = frozenset(("crash", "off_track", "max_steps", "terminated"))


def _is_finite_number(value: object, minimum: float, maximum: float) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and minimum <= value <= maximum)


def _metric_errors(row: dict, label: str, phase: str) -> list[str]:
    errors = []
    if row.get("partition") != phase:
        errors.append(f"{label}: wrong partition")
    if "error" not in row:
        errors.append(f"{label}: missing error field")
    elif row["error"] is not None:
        errors.append(f"{label}: operational error: {row['error']}")
        return errors
    for field, limit in _RUNTIME_LIMITS:
        if not _is_finite_number(row.get(field), 0, limit):
            errors.append(f"{label}: runtime {field} invalid or exceeded")
    if type(row.get("finished")) is not bool:
        errors.append(f"{label}: finished must be boolean")
    for field in ("lap_time_ms", "retire_reason"):
        if field not in row:
            errors.append(f"{label}: missing {field}")
    if not _is_finite_number(row.get("progress"), 0, 1.01):
        errors.append(f"{label}: invalid progress")
    if type(row.get("collision_count")) is not int or row["collision_count"] < 0:
        errors.append(f"{label}: invalid collision_count")
    if not _is_finite_number(row.get("damage"), 0, 1.0):
        errors.append(f"{label}: invalid damage")
    if type(row.get("steps")) is not int or not 0 <= row["steps"] <= 2_000:
        errors.append(f"{label}: invalid steps")
    for field in ("offtrack_samples", "partial_offtrack_samples"):
        if type(row.get(field)) is not int or row[field] < 0:
            errors.append(f"{label}: invalid {field}")
    lap_time = row.get("lap_time_ms")
    if row.get("finished") is True:
        if not _is_finite_number(lap_time, 0, float("inf")) or lap_time == 0:
            errors.append(f"{label}: finished lap needs positive lap_time_ms")
        if row.get("retire_reason") is not None:
            errors.append(f"{label}: finished lap has retire_reason")
    elif row.get("finished") is False:
        if lap_time is not None:
            errors.append(f"{label}: DNF lap_time_ms must be null")
        reason = row.get("retire_reason")
        if not isinstance(reason, str) or reason not in _DNF_REASONS:
            errors.append(f"{label}: invalid DNF retire_reason")
    trace = row.get("action_trace_sha256")
    if (not isinstance(trace, str) or len(trace) != 64
            or any(char not in "0123456789abcdef" for char in trace)):
        errors.append(f"{label}: invalid action_trace_sha256")
    return errors


def _finish_breakdowns(paired: list[dict], seeds: set[int]) -> tuple[list[dict], list[dict]]:
    """Count canonical finishes by geometry and track from validated pairs."""
    def blank() -> dict:
        return {"control_finishes": 0, "candidate_finishes": 0,
                "net_finish_gain": 0, "lost_control_finishes": 0}

    by_seed = {seed: blank() for seed in seeds}
    by_track = {track: blank() for track in (1, 2, 3, 4)}
    for pair in paired:
        if pair["status"] != "COMPLETE":
            continue
        control = int(pair["control"]["finished"])
        candidate = int(pair["candidate"]["finished"])
        lost = int(bool(control and not candidate))
        buckets = [by_seed[pair["seed"]]]
        if pair["track_id"] in by_track:
            buckets.append(by_track[pair["track_id"]])
        for bucket in buckets:
            bucket["control_finishes"] += control
            bucket["candidate_finishes"] += candidate
            bucket["net_finish_gain"] += candidate - control
            bucket["lost_control_finishes"] += lost
    seed_rows = [{"seed": seed, **by_seed[seed]} for seed in sorted(by_seed)]
    track_rows = [{"track_id": track, **by_track[track]} for track in (1, 2, 3, 4)]
    return seed_rows, track_rows


def compare_pairs(rows: list[dict], cells: list[tuple[int, int, int]], phase: str) -> dict:
    """Validate all cold receipts and apply the fixed practical v4 phase gate.

    ``cells`` is the full expected inventory including repeat-1 cells.  Only
    repeat-0 pairs enter the statistical totals.  A repeat must exactly match
    its canonical arm's action hash and outcome fields.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    expected_cells = list(cells)
    if any(not isinstance(cell, tuple) or len(cell) != 3 or
           any(type(value) is not int for value in cell) or cell[2] not in (0, 1)
           for cell in expected_cells):
        raise ValueError("cells must be (track, seed, repeat) integer tuples")
    reasons: list[str] = []
    if len(expected_cells) != len(set(expected_cells)):
        reasons.append("duplicate expected cell")
    phase_index = PHASES.index(phase)
    canonical = [(track, seed) for track, seed, repeat in expected_cells if repeat == 0]
    seed_sets = {track: {seed for cell_track, seed in canonical if cell_track == track}
                 for track in (1, 2, 3, 4)}
    if (len(canonical) != 4 * _SEEDS_PER_PHASE[phase_index]
            or any(len(seeds) != _SEEDS_PER_PHASE[phase_index]
                   for seeds in seed_sets.values())
            or any(seeds != seed_sets[1] for seeds in seed_sets.values())
            or any(track not in (1, 2, 3, 4) for track, _ in canonical)):
        reasons.append("expected four-track canonical geometry grid is incomplete or invalid")
    if sum(repeat == 1 for _, _, repeat in expected_cells) != _REPEATS_PER_PHASE[phase_index]:
        reasons.append("expected phase repeat grid is incomplete or invalid")
    expected = {(track, seed, repeat, arm)
                for track, seed, repeat in expected_cells for arm in ARMS}
    lookup: dict[tuple[int, int, int, str], dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            reasons.append("invalid receipt: row is not an object")
            continue
        track, seed, repeat, arm = (row.get(field) for field in
                                    ("track_id", "seed", "repeat", "arm"))
        if (any(type(value) is not int for value in (track, seed, repeat))
                or not isinstance(arm, str)):
            reasons.append("invalid receipt coordinates")
            continue
        key = (track, seed, repeat, arm)
        if key not in expected:
            reasons.append(f"{track}/{seed}/{repeat}/{arm}: unexpected receipt")
        elif key in lookup:
            reasons.append(f"{track}/{seed}/{repeat}/{arm}: duplicate receipt")
        else:
            lookup[key] = row

    canonical_set = set(canonical)
    missing: list[list[int | str]] = []
    valid: dict[tuple[int, int, int, str], dict] = {}
    for track, seed, repeat in expected_cells:
        if repeat == 1 and (track, seed) not in canonical_set:
            reasons.append(f"{track}/{seed}/1: repeat lacks canonical cell")
        for arm in ARMS:
            key = (track, seed, repeat, arm)
            row = lookup.get(key)
            if row is None:
                missing.append([track, seed, repeat, arm])
                continue
            row_errors = _metric_errors(row, f"{track}/{seed}/{repeat}/{arm}", phase)
            if row_errors:
                reasons.extend(row_errors)
            else:
                valid[key] = row

    repeat_pairs_checked = 0
    for track, seed, repeat in expected_cells:
        if repeat != 1:
            continue
        if all((track, seed, index, arm) in valid for index in (0, 1) for arm in ARMS):
            repeat_pairs_checked += 1
            for arm in ARMS:
                original = valid[(track, seed, 0, arm)]
                repeated = valid[(track, seed, 1, arm)]
                changed = [field for field in _REPEAT_FIELDS
                           if original[field] != repeated[field]]
                if changed:
                    reasons.append(f"{track}/{seed}/{arm}: repeat mismatch: {', '.join(changed)}")

    totals = {arm: {"finishes": 0, "progress": 0.0, "crashes": 0,
                    "contacts": 0, "damage": 0.0, "shared_time": 0.0} for arm in ARMS}
    paired: list[dict] = []
    lost_finishes = new_crashes = both_dnf_losses = shared_count = valid_count = 0
    for track, seed in canonical:
        control = valid.get((track, seed, 0, "control"))
        candidate = valid.get((track, seed, 0, "candidate"))
        if control is None or candidate is None:
            status = "MISSING_PAIR" if any(
                [track, seed, 0, arm] in missing for arm in ARMS) else "INVALID_PAIR"
            paired.append({"track_id": track, "seed": seed, "status": status})
            continue
        valid_count += 1
        flags = []
        for arm, row in (("control", control), ("candidate", candidate)):
            totals[arm]["finishes"] += int(row["finished"])
            totals[arm]["progress"] += float(row["progress"])
            totals[arm]["crashes"] += int(row["retire_reason"] == "crash")
            totals[arm]["contacts"] += row["collision_count"]
            totals[arm]["damage"] += float(row["damage"])
        if control["finished"] and not candidate["finished"]:
            lost_finishes += 1
            flags.append("control finish lost")
        if candidate["retire_reason"] == "crash" and control["retire_reason"] != "crash":
            new_crashes += 1
            flags.append("new candidate crash")
        if not control["finished"] and not candidate["finished"] and candidate["progress"] < control["progress"]:
            both_dnf_losses += 1
            flags.append("both-DNF progress loss")
        if candidate["collision_count"] > control["collision_count"]:
            flags.append("contact increase")
        if candidate["damage"] > control["damage"]:
            flags.append("damage increase")
        if control["finished"] and candidate["finished"]:
            shared_count += 1
            for arm, row in (("control", control), ("candidate", candidate)):
                totals[arm]["shared_time"] += float(row["lap_time_ms"])
            if candidate["lap_time_ms"] > control["lap_time_ms"] * _MAX_SHARED_TIME_RATIO:
                flags.append("shared-finish slowdown")
        paired.append({
            "track_id": track, "seed": seed, "status": "COMPLETE",
            "control": {field: control[field] for field in _OUTCOME_FIELDS},
            "candidate": {field: candidate[field] for field in _OUTCOME_FIELDS},
            "flags": flags,
        })

    net_gain = totals["candidate"]["finishes"] - totals["control"]["finishes"]
    control_time = totals["control"]["shared_time"]
    candidate_time = totals["candidate"]["shared_time"]
    ratio = candidate_time / control_time if shared_count else None
    seed_breakdown, track_breakdown = _finish_breakdowns(
        paired, {seed for _, seed in canonical})
    complete = not reasons and not missing and valid_count == len(canonical)
    if complete:
        if net_gain < _MIN_NET_GAIN[phase_index]:
            reasons.append("net finish gain below phase minimum")
        if lost_finishes > _MAX_LOST_FINISHES[phase_index]:
            reasons.append("lost control finishes exceed phase budget")
        if totals["candidate"]["crashes"] > totals["control"]["crashes"]:
            reasons.append("aggregate crashes increased")
        if totals["candidate"]["contacts"] > totals["control"]["contacts"]:
            reasons.append("aggregate contacts increased")
        if totals["candidate"]["damage"] > totals["control"]["damage"]:
            reasons.append("aggregate damage increased")
        if totals["candidate"]["progress"] < totals["control"]["progress"]:
            reasons.append("candidate mean progress below control")
        if shared_count and candidate_time > control_time * _MAX_SHARED_TIME_RATIO:
            reasons.append("aggregate shared-finish time regression")
        for entry in seed_breakdown:
            if (entry["lost_control_finishes"] >= _SEED_LOSS_CLUSTER and
                    entry["net_finish_gain"] < 0):
                reasons.append(f"seed {entry['seed']}: clustered control finish loss")
    decision = "REJECT" if reasons else "INCOMPLETE" if missing else "RETAIN"
    denominator = valid_count or 1
    return {
        "decision": decision, "reasons": reasons, "missing_cells": missing,
        "canonical_cells": len(canonical), "valid_canonical_cells": valid_count,
        "repeat_pairs_checked": repeat_pairs_checked, "paired_cells": paired,
        "seed_finish_breakdown": seed_breakdown,
        "track_finish_breakdown": track_breakdown,
        "control_finishes": totals["control"]["finishes"],
        "candidate_finishes": totals["candidate"]["finishes"],
        "net_finish_gain": net_gain, "lost_control_finishes": lost_finishes,
        "control_crashes": totals["control"]["crashes"],
        "candidate_crashes": totals["candidate"]["crashes"],
        "new_candidate_crashes": new_crashes,
        "both_dnf_progress_losses": both_dnf_losses,
        "control_mean_progress": totals["control"]["progress"] / denominator,
        "candidate_mean_progress": totals["candidate"]["progress"] / denominator,
        "control_contacts": totals["control"]["contacts"],
        "candidate_contacts": totals["candidate"]["contacts"],
        "control_damage": totals["control"]["damage"],
        "candidate_damage": totals["candidate"]["damage"],
        "shared_completed_cells": shared_count,
        "control_shared_completed_time_ms": control_time,
        "candidate_shared_completed_time_ms": candidate_time,
        "shared_completed_time_ratio": ratio,
    }


def combined_decision(phase_summaries: dict[str, dict]) -> dict:
    """Require retained phases, 12 net finishes and no harmed track layout."""
    if not isinstance(phase_summaries, dict):
        raise TypeError("phase_summaries must be a dictionary")
    rejected: list[str] = []
    incomplete: list[str] = []
    canonical_count = net_gain = 0
    decisions = {}
    combined_tracks = {track: {"control_finishes": 0, "candidate_finishes": 0,
                               "net_finish_gain": 0, "lost_control_finishes": 0}
                       for track in (1, 2, 3, 4)}
    expected_counts = (32, 64, 32)
    if set(phase_summaries) - set(PHASES):
        rejected.append("unexpected phase summary")
    for phase, expected in zip(PHASES, expected_counts):
        summary = phase_summaries.get(phase)
        if not isinstance(summary, dict):
            incomplete.append(f"{phase} summary missing")
            decisions[phase] = "INCOMPLETE"
            continue
        decision = summary.get("decision")
        decisions[phase] = decision
        if decision == "REJECT":
            rejected.append(f"{phase} rejected")
        elif decision == "INCOMPLETE":
            incomplete.append(f"{phase} incomplete")
        elif decision != "RETAIN":
            rejected.append(f"{phase} has invalid decision")
        count = summary.get("canonical_cells")
        gain = summary.get("net_finish_gain")
        if type(count) is int:
            canonical_count += count
        if type(count) is not int or count != expected:
            incomplete.append(f"{phase} requires {expected} canonical cells")
        if type(gain) is not int:
            incomplete.append(f"{phase} net finish gain missing")
        else:
            net_gain += gain
        control_finishes = summary.get("control_finishes")
        candidate_finishes = summary.get("candidate_finishes")
        if (type(control_finishes) is not int or type(candidate_finishes) is not int
                or not 0 <= control_finishes <= expected
                or not 0 <= candidate_finishes <= expected
                or type(gain) is not int
                or candidate_finishes - control_finishes != gain):
            rejected.append(f"{phase} finish totals and net finish gain disagree")
        if summary.get("missing_cells"):
            incomplete.append(f"{phase} has missing receipts")
        if ("valid_canonical_cells" in summary and
                summary["valid_canonical_cells"] != expected):
            incomplete.append(f"{phase} has invalid canonical pairs")
        if "partition" in summary and summary["partition"] != phase:
            rejected.append(f"{phase} partition mismatch")
        pairs = summary.get("paired_cells")
        if pairs is None:
            incomplete.append(f"{phase} paired_cells missing")
            continue
        if not isinstance(pairs, list):
            rejected.append(f"{phase} paired_cells invalid")
            continue
        if len(pairs) != expected:
            message = f"{phase} requires {expected} paired_cells"
            (incomplete if len(pairs) < expected else rejected).append(message)
            continue
        seen: set[tuple[int, int]] = set()
        malformed = False
        for pair in pairs:
            if not isinstance(pair, dict):
                rejected.append(f"{phase} has malformed paired_cells")
                malformed = True
                continue
            track, seed = pair.get("track_id"), pair.get("seed")
            if type(track) is not int or track not in (1, 2, 3, 4) or type(seed) is not int:
                rejected.append(f"{phase} has invalid paired cell coordinates")
                malformed = True
                continue
            coordinate = (track, seed)
            if coordinate in seen:
                rejected.append(f"{phase} has duplicate paired cell coordinates")
                malformed = True
                continue
            seen.add(coordinate)
            if pair.get("status") != "COMPLETE":
                incomplete.append(f"{phase} paired cell {track}/{seed} incomplete")
                malformed = True
                continue
            control, candidate = pair.get("control"), pair.get("candidate")
            if (not isinstance(control, dict) or not isinstance(candidate, dict)
                    or type(control.get("finished")) is not bool
                    or type(candidate.get("finished")) is not bool):
                rejected.append(f"{phase} paired cell {track}/{seed} lacks finish outcomes")
                malformed = True
        if malformed:
            continue
        seeds = {seed for _, seed in seen}
        if (len(seeds) != _SEEDS_PER_PHASE[PHASES.index(phase)] or
                any({seed for cell_track, seed in seen if cell_track == track} != seeds
                    for track in (1, 2, 3, 4))):
            rejected.append(f"{phase} paired_cells do not form the four-track seed grid")
            continue
        seed_rows, track_rows = _finish_breakdowns(pairs, seeds)
        if summary.get("seed_finish_breakdown") != seed_rows:
            rejected.append(f"{phase} seed finish breakdown differs from paired_cells")
        if summary.get("track_finish_breakdown") != track_rows:
            rejected.append(f"{phase} track finish breakdown differs from paired_cells")
        if (sum(row["control_finishes"] for row in track_rows) != control_finishes or
                sum(row["candidate_finishes"] for row in track_rows) != candidate_finishes or
                sum(row["lost_control_finishes"] for row in track_rows) !=
                summary.get("lost_control_finishes")):
            rejected.append(f"{phase} finish totals differ from paired_cells")
        for seed_row in seed_rows:
            if (seed_row["lost_control_finishes"] >= _SEED_LOSS_CLUSTER and
                    seed_row["net_finish_gain"] < 0):
                rejected.append(f"{phase} seed {seed_row['seed']} has clustered finish loss")
        for row in track_rows:
            bucket = combined_tracks[row["track_id"]]
            for field in ("control_finishes", "candidate_finishes", "net_finish_gain",
                          "lost_control_finishes"):
                bucket[field] += row[field]
    summaries = [phase_summaries.get(phase) for phase in PHASES]
    if all(isinstance(summary, dict) for summary in summaries):
        for field in _IDENTITY_FIELDS:
            if (field in _REQUIRED_IDENTITY_FIELDS or
                    any(field in summary for summary in summaries)):
                if any(field not in summary or summary[field] is None
                       for summary in summaries):
                    rejected.append(f"{field} missing across phases")
                elif any(summary[field] != summaries[0][field]
                         for summary in summaries[1:]):
                    rejected.append(f"{field} mismatch across phases")
    if not rejected and not incomplete and net_gain < _MIN_COMBINED_NET_GAIN:
        rejected.append(f"combined net finish gain below {_MIN_COMBINED_NET_GAIN}")
    if not rejected and not incomplete:
        for track, bucket in combined_tracks.items():
            if bucket["candidate_finishes"] < bucket["control_finishes"]:
                rejected.append(f"track {track} candidate finishes below control")
    decision = "REJECT" if rejected else "INCOMPLETE" if incomplete else "RETAIN"
    return {"decision": decision, "reasons": rejected + incomplete,
            "canonical_cells": canonical_count, "net_finish_gain": net_gain,
            "phase_decisions": decisions,
            "track_finish_breakdown": [
                {"track_id": track, **combined_tracks[track]}
                for track in (1, 2, 3, 4)],
            }
