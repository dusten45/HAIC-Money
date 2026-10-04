"""Post-cohort RLPD TRAIN coverage gate, not a G1 intervention or seed audit."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Sequence


ACTOR_HASHES = {
    "entropy-v5-author-seed50": "ba56dca388a204a902d0f198d25a909a64866511002854972c4d0dd9627e8b98",
    "long-horizon-seed11": "f5c048d9cff711705c55887a433d433b3f7eed13ed104f3f370d750a2fba17e1",
}
SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CoverageRules:
    cells: tuple[tuple[int, int], ...]
    primary_actor_id: str
    comparator_actor_id: str
    image_rubric_sha256: str
    max_decisions_per_episode: int = 2000
    max_total_decisions: int = 96000
    max_total_raw_frames: int = 386448
    max_core_hours: float | None = 4.0
    min_positive_geometries: int = 3
    min_finished_geometries: int = 3

    def __post_init__(self) -> None:
        if (
            len(self.cells) != 24
            or any(type(track) is not int or track != 1 or type(seed) is not int
                   or not 0 <= seed < 2**32 for track, seed in self.cells)
            or self.cells != tuple((1, self.cells[0][1] + offset) for offset in range(24))
        ):
            raise ValueError("G1 requires 24 consecutive, unique TRAIN track-1 geometries")
        if (
            self.primary_actor_id != "entropy-v5-author-seed50"
            or self.comparator_actor_id != "long-horizon-seed11"
            or not isinstance(self.image_rubric_sha256, str)
            or SHA256.fullmatch(self.image_rubric_sha256) is None
            or self.max_decisions_per_episode != 2000
            or self.max_total_decisions != 96000
            or self.max_total_raw_frames != 386448
            or self.max_core_hours not in (4.0, None)
            or self.min_positive_geometries != 3
            or self.min_finished_geometries != 3
        ):
            raise ValueError("G1 source actors, image rubric and hard caps must be predeclared")


def assess_coverage(
    rules: CoverageRules, rows: Sequence[dict[str, Any]], *, core_hours_used: float,
) -> dict[str, Any]:
    """Assess all scheduled roads; the caller must first pin protocol/receipt bytes."""
    if (type(core_hours_used) not in (int, float) or not 0 <= core_hours_used
            or not math.isfinite(core_hours_used)
            or rules.max_core_hours is not None and core_hours_used > rules.max_core_hours):
        raise ValueError("G1 process exceeded its frozen core-hour cap")
    if len(rows) != 2 * len(rules.cells):
        raise ValueError("G1 result must preserve every scheduled actor-road slot")

    positive = []
    finished = []
    positive_roads = []
    finished_roads = []
    uncertain = []
    unknown_image = []
    finished_with_flag = []
    total_decisions = total_driven_raw = total_reset_initial = total_reset_noop = 0
    for index, row in enumerate(rows):
        track_id, seed = rules.cells[index // 2]
        actor_id = (rules.primary_actor_id, rules.comparator_actor_id)[index % 2]
        if (
            not isinstance(row, dict)
            or row.get("partition") != "TRAIN"
            or type(row.get("track_id")) is not int
            or type(row.get("geometry_seed")) is not int
            or row.get("track_id") != track_id
            or row.get("geometry_seed") != seed
            or row.get("obstacles") is not True
            or row.get("actor_id") != actor_id
            or row.get("actor_sha256") != ACTOR_HASHES[actor_id]
            or row.get("image_rubric_sha256") != rules.image_rubric_sha256
        ):
            raise ValueError("G1 row differs from the frozen TRAIN actor-road schedule")
        status = row.get("status")
        outcome = row.get("outcome")
        decisions = row.get("decisions")
        raw = row.get("driven_raw_frames")
        initial = row.get("reset_initial_raw_frames")
        noop = row.get("reset_noop_raw_frames")
        if (
            status not in ("complete", "collection_censored", "unrun")
            or any(type(value) is not int or value < 0
                   for value in (decisions, raw, initial, noop))
            or decisions > rules.max_decisions_per_episode
            or raw > 4 * decisions
            or decisions > 0 and raw < 4 * (decisions - 1) + 1
        ):
            raise ValueError("G1 episode status, decision or raw-frame accounting is invalid")
        total_decisions += decisions
        total_driven_raw += raw
        total_reset_initial += initial
        total_reset_noop += noop
        if status == "unrun":
            if (
                any((decisions, raw, initial, noop))
                or outcome is not None
                or row.get("finished") is not None
                or row.get("terminated") is not None
                or row.get("truncated") is not None
                or row.get("image_flag") is not None
                or row.get("road_centerline_sha256") is not None
            ):
                raise ValueError("unrun G1 slot cannot masquerade as an observed episode")
            uncertain.append((seed, actor_id, "unrun"))
            continue
        road_sha = row.get("road_centerline_sha256")
        if (
            decisions < 1 or raw < decisions or initial < 1 or noop != 50
            or not isinstance(road_sha, str) or SHA256.fullmatch(road_sha) is None
            or row.get("image_flag") not in ("yes", "no", "unknown")
            or row.get("image_review_blinded") is not True
            or not isinstance(row.get("image_review_receipt_sha256"), str)
            or SHA256.fullmatch(row["image_review_receipt_sha256"]) is None
        ):
            raise ValueError("G1 attempted episode lacks complete accounting or blinded image review")
        if status == "collection_censored":
            if (
                outcome != "unknown"
                or row.get("finished") is not False
                or row.get("terminated") is not False
                or row.get("truncated") is not False
            ):
                raise ValueError("G1 collection censoring must remain unknown, not task failure")
            uncertain.append((seed, actor_id, "collection_censored"))
            if row["image_flag"] == "unknown":
                unknown_image.append((seed, actor_id))
            continue
        if type(row.get("finished")) is not bool or type(row.get("terminated")) is not bool or type(row.get("truncated")) is not bool:
            raise ValueError("G1 original finish/terminal flags must be real booleans")
        if outcome == "finished":
            if not row["finished"] or not row["truncated"]:
                raise ValueError("G1 visited-tile progress or qualification alone is not a finish")
        elif outcome == "task_timeout":
            if row["finished"] or not row["truncated"] or decisions != rules.max_decisions_per_episode:
                raise ValueError("G1 task timeout needs the original full deadline")
        elif outcome in ("off_track", "crash", "out_of_bounds"):
            if row["finished"] or not row["terminated"]:
                raise ValueError("G1 physical/retirement outcome needs an original terminal decision")
            if row["truncated"] and decisions != rules.max_decisions_per_episode:
                raise ValueError("G1 early retirement cannot claim task truncation")
            if outcome == "off_track" and decisions < 101:
                raise ValueError("G1 off_track needs the original 101-negative-decision boundary")
            if outcome == "crash" and decisions < 5:
                raise ValueError("G1 crash needs at least five damage increments")
        elif outcome == "unknown":
            if row["finished"]:
                raise ValueError("G1 real finish cannot be unknown")
            uncertain.append((seed, actor_id, "terminal_unknown"))
        else:
            raise ValueError("G1 complete episode outcome is unrecognized")
        if row["image_flag"] == "unknown":
            unknown_image.append((seed, actor_id))
        if index % 2 == 0:
            if outcome == "finished":
                finished.append(seed)
                finished_roads.append(road_sha)
                if row["image_flag"] == "yes":
                    finished_with_flag.append(seed)
            elif outcome in ("off_track", "crash", "out_of_bounds", "task_timeout") and row["image_flag"] == "yes":
                positive.append(seed)
                positive_roads.append(road_sha)

    if any(
        rows[2 * i]["status"] != "unrun"
        and rows[2 * i + 1]["status"] != "unrun"
        and rows[2 * i]["road_centerline_sha256"] != rows[2 * i + 1]["road_centerline_sha256"]
        for i in range(len(rules.cells))
    ):
        raise ValueError("G1 paired actors did not receive the same road bytes")
    if (
        total_decisions > rules.max_total_decisions
        or total_driven_raw + total_reset_initial + total_reset_noop > rules.max_total_raw_frames
    ):
        raise ValueError("G1 spent beyond its frozen decision/raw-frame budget")
    if set(positive) & set(finished):
        raise ValueError("G1 failed-positive and finished-parent geometries must be disjoint")
    complete = len(uncertain) == 0 and len(unknown_image) == 0
    observed_roads = {
        rows[2 * index]["road_centerline_sha256"]
        for index in range(len(rules.cells))
        if rows[2 * index]["status"] != "unrun"
    }
    observed_roads.discard(None)
    distinct_positive = set(positive_roads)
    distinct_finished = set(finished_roads)
    return {
        "status": "pass" if complete and len(observed_roads) == len(rules.cells)
                  and len(distinct_positive) >= rules.min_positive_geometries
                  and len(distinct_finished) >= rules.min_finished_geometries
                  and not distinct_positive.intersection(distinct_finished) else "inconclusive",
        "role": "TRAIN-only source-specific coverage, not rescue/generalization",
        "geometry_clusters": len(rules.cells), "scheduled_episodes": len(rows),
        "distinct_observed_road_geometries": len(observed_roads),
        "primary_actor_id": rules.primary_actor_id,
        "comparator_actor_id": rules.comparator_actor_id,
        "primary_positive_geometries": positive,
        "primary_finished_parent_geometries": finished,
        "distinct_positive_road_hashes": sorted(distinct_positive),
        "distinct_finished_parent_road_hashes": sorted(distinct_finished),
        "primary_finished_parent_with_image_flag": finished_with_flag,
        "uncertain_slots": uncertain,
        "unknown_image_slots": unknown_image,
        "spent_decisions": total_decisions,
        "driven_raw_frames": total_driven_raw,
        "reset_initial_raw_frames": total_reset_initial,
        "reset_noop_raw_frames": total_reset_noop,
        "core_hours_used": float(core_hours_used),
    }
