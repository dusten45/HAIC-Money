"""Matched, completion-first comparison and research search limits."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from .models import CycleSummary, ExperimentResult, GateStatus, Hypothesis, PromotionDecision


class ComparisonMismatchError(ValueError):
    """Results do not share a registered comparison protocol."""


def _protocol(result: ExperimentResult) -> tuple[object, ...]:
    return (result.comparison_id, result.split_id, frozenset(result.map_ids),
            frozenset(result.seed_ids), result.episode_count)


def _require_matched(records: Sequence[ExperimentResult]) -> None:
    if records and any(_protocol(item) != _protocol(records[0]) for item in records[1:]):
        raise ComparisonMismatchError("comparison id, split, maps, seeds and denominator must match")


def _rank_key(result: ExperimentResult) -> tuple[float, ...]:
    return (-result.completion_count / result.episode_count,
            float("inf") if result.median_finished_lap_ms is None else result.median_finished_lap_ms,
            -result.mean_incomplete_progress,
            float("inf") if result.p90_finished_lap_ms is None else result.p90_finished_lap_ms,
            result.collisions, result.damage, result.act_latency_p95_ms)


def rank_candidates(records: Sequence[ExperimentResult]) -> list[ExperimentResult]:
    """Rank eligible candidates in a single matched group; retain input ledger unchanged."""
    _require_matched(records)
    return sorted((item for item in records if item.eligibility == "candidate"), key=_rank_key)


def promotion_decision(candidate: ExperimentResult, control: ExperimentResult) -> PromotionDecision:
    _require_matched((candidate, control))
    if candidate.eligibility != "candidate" or control.eligibility != "candidate":
        return PromotionDecision(False, "candidate and control must be submission eligible")
    if candidate.rule_compliance != GateStatus.PASS or candidate.mechanism_activation != GateStatus.PASS:
        return PromotionDecision(False, "candidate rule compliance and mechanism activation must PASS")
    if control.rule_compliance != GateStatus.PASS or control.mechanism_activation != GateStatus.PASS:
        return PromotionDecision(False, "control rule compliance and mechanism activation must PASS")
    if _rank_key(candidate) < _rank_key(control):
        return PromotionDecision(True, "candidate outranks matched control")
    return PromotionDecision(False, "candidate does not outrank matched control")


def validate_batch(hypotheses: Sequence[Hypothesis]) -> None:
    """Require four independent action mechanisms within the 4/8/2 search budget."""
    directions = [item.allowed_action_or_state_change.strip() for item in hypotheses]
    identifiers = [item.hypothesis_id for item in hypotheses]
    if len(hypotheses) > 8 or len(set(directions)) < 4 or not all(directions):
        raise ValueError("batch requires at least four directions and at most eight candidates")
    if len(set(identifiers)) != len(identifiers) or not all(identifiers):
        raise ValueError("batch hypothesis identifiers must be unique and nonempty")
    if any(count > 2 for count in Counter(directions).values()):
        raise ValueError("direction exceeds two candidates")


def should_pivot(cycles: Sequence[CycleSummary], limit: int = 3) -> bool:
    """Count the trailing comparable failures; infrastructure-invalid cycles are skipped."""
    if limit < 1:
        raise ValueError("pivot limit must be positive")
    streak = 0
    for cycle in reversed(cycles):
        if cycle.infra_invalid:
            continue
        if not cycle.valid or not cycle.protocol_match or cycle.improved:
            break
        streak += 1
        if streak >= limit:
            return True
    return False
