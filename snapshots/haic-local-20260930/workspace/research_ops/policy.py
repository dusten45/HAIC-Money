from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str


def _value(row: dict[str, Any], key: str, default: float) -> float:
    value = row.get(key)
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    episodes = _value(row, "episodes", 0.0)
    completion = _value(row, "completion_rate", 0.0)
    completed = _value(row, "completed_episodes", float("nan"))
    if episodes > 0 and completed == completed:
        completion = completed / episodes
    return (
        -round(completion, 6),
        _value(row, "median_finished_lap_ms", float("inf")),
        -_value(row, "mean_progress", 0.0),
        _value(row, "p90_finished_lap_ms", float("inf")),
        _value(row, "collisions", float("inf")),
        _value(row, "damage", float("inf")),
    )


def compare_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted((dict(row) for row in records), key=_rank_key)
    result: list[dict[str, Any]] = []
    previous_key: tuple[float, ...] | None = None
    rank = 0
    for index, row in enumerate(ranked, start=1):
        current_key = _rank_key(row)
        if current_key != previous_key:
            rank = index
            previous_key = current_key
        row["rank"] = rank
        result.append(row)
    return result


def promotion_gate(candidate: dict[str, Any], sota: dict[str, Any]) -> PromotionDecision:
    if candidate.get("restriction_status") != "pass":
        return PromotionDecision(False, "Restriction preflight did not pass.")
    if candidate.get("strategy") == "vision_corridor_teacher":
        return PromotionDecision(False, "Teacher lane is diagnostic-only and cannot become submission SOTA.")
    if candidate.get("package_smoke_status") != "pass":
        return PromotionDecision(False, "Submission package smoke has not passed.")
    if candidate.get("split") not in {"held_out", "official"}:
        return PromotionDecision(False, "Promotion requires held_out or official evidence.")
    if _value(candidate, "episodes", 0) < 3:
        return PromotionDecision(False, "Promotion requires at least three evaluated episodes.")
    if _rank_key(candidate) >= _rank_key(sota):
        return PromotionDecision(False, "Candidate does not strictly improve the current SOTA ordering.")
    return PromotionDecision(True, "Candidate passes restrictions and strictly improves SOTA ordering.")
