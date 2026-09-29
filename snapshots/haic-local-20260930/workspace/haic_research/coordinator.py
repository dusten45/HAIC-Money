"""Central assignment and synthesis for bounded hypothesis batches."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Sequence

from .models import (
    AgentReportEnvelope, GateResult, GateStatus, Hypothesis, IntegrationReport,
    WorkAssignment,
)
from .policy import validate_batch


def _safe_segment(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("batch id must be a single safe path segment")
    return value


def assign_work(batch_id: str, hypotheses: Sequence[Hypothesis]) -> list[WorkAssignment]:
    """Record one isolated handoff scope and owner for each mechanism direction."""
    validate_batch(hypotheses)
    batch_id = _safe_segment(batch_id)
    groups: dict[str, list[Hypothesis]] = defaultdict(list)
    for item in hypotheses:
        groups[item.allowed_action_or_state_change.strip()].append(item)
    assignments = []
    for index, (direction, members) in enumerate(groups.items(), start=1):
        label = sha256(direction.encode("utf-8")).hexdigest()[:12]
        handoff = str(PurePosixPath("docs/handoffs") / batch_id / f"{index}-{label}.md")
        assignments.append(WorkAssignment(
            direction=direction,
            hypothesis_ids=tuple(item.hypothesis_id for item in members),
            owner=f"{batch_id}-owner-{index}",
            allowed_read_paths=tuple(dict.fromkeys(path for item in members for path in item.source_paths)),
            allowed_write_paths=(handoff,),
            handoff_path=handoff,
        ))
    return assignments


def _path(path: str) -> PurePosixPath:
    if not isinstance(path, str) or not path.strip() or "\\" in path:
        raise ValueError("scope paths must be nonempty POSIX relative paths")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or str(parsed) == ".":
        raise ValueError("scope paths must remain within the workspace")
    return parsed


def _in_scope(path: str, scope: str) -> bool:
    return _path(path).is_relative_to(_path(scope))


def integrate_reports(
    assignments: Sequence[WorkAssignment], reports: Sequence[AgentReportEnvelope]
) -> IntegrationReport:
    """Check complete isolated handoffs, then record conservative central gate states."""
    if not assignments or len(assignments) != len(reports):
        raise ValueError("each assignment requires exactly one report")
    owners: set[str] = set()
    handoffs: set[str] = set()
    directions: set[str] = set()
    hypothesis_ids: set[str] = set()
    scopes: list[str] = []
    for assignment in assignments:
        if not assignment.owner or not assignment.direction or not assignment.hypothesis_ids:
            raise ValueError("assignment owner, direction and hypotheses are required")
        if assignment.owner in owners or assignment.handoff_path in handoffs or assignment.direction in directions:
            raise ValueError("duplicate assignment ownership")
        owners.add(assignment.owner)
        directions.add(assignment.direction)
        handoffs.add(assignment.handoff_path)
        if hypothesis_ids.intersection(assignment.hypothesis_ids):
            raise ValueError("hypothesis assigned to multiple owners")
        hypothesis_ids.update(assignment.hypothesis_ids)
        if not assignment.allowed_write_paths or not any(
            _in_scope(assignment.handoff_path, scope) for scope in assignment.allowed_write_paths
        ):
            raise ValueError("handoff path must be within assignment write scope")
        for scope in assignment.allowed_write_paths:
            if any(_in_scope(scope, other) or _in_scope(other, scope) for other in scopes):
                raise ValueError("overlapping assignment write scopes")
            scopes.append(scope)

    by_handoff = {assignment.handoff_path: assignment for assignment in assignments}
    seen: set[str] = set()
    evidence: list[str] = []
    for envelope in reports:
        assignment = by_handoff.get(envelope.handoff_path)
        if assignment is None or envelope.handoff_path in seen or envelope.owner != assignment.owner:
            raise ValueError("report has missing, duplicate or mismatched assignment ownership")
        seen.add(envelope.handoff_path)
        if any(not any(_in_scope(path, scope) for scope in assignment.allowed_write_paths)
               for path in envelope.edited_paths):
            raise ValueError("report lists out-of-scope edits")
        report = envelope.report
        if not all(isinstance(getattr(report, name), str) and getattr(report, name).strip()
                   for name in ("fact", "inference", "unknown", "recommendation")):
            raise ValueError("report requires fact, inference, unknown and recommendation")
        if not report.source_paths or any(not isinstance(path, str) or not path.strip()
                                          for path in report.source_paths):
            raise ValueError("report requires source paths")
        evidence.extend(report.source_paths)
        evidence.append(envelope.handoff_path)
    if seen != handoffs:
        raise ValueError("reports do not cover every assignment")
    evidence_paths = tuple(dict.fromkeys(evidence))
    gates = tuple(GateResult(name, GateStatus.UNKNOWN,
                             "central gate decision requires independent evidence review",
                             evidence_paths)
                  for name in ("rule_compliance", "mechanism_activation",
                               "competitive_or_product_outcome"))
    return IntegrationReport(gates, evidence_paths)
