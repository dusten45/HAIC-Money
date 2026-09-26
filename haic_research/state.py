"""Explicit state changes for the HAIC research workflow."""

from __future__ import annotations

from typing import Sequence

from .models import Approval, GateResult, GateStatus, WorkflowState


class TransitionError(ValueError):
    """A requested workflow edge is not legal."""


class ApprovalError(TransitionError):
    """The required stage has no approval for this exact plan hash."""


class GateError(TransitionError):
    """A release gate is missing, duplicated, unknown, or not passing."""


_EDGES: dict[WorkflowState, frozenset[WorkflowState]] = {
    WorkflowState.STOPPED: frozenset({WorkflowState.DISCOVER}),
    WorkflowState.DISCOVER: frozenset({WorkflowState.HYPOTHESIZE}),
    WorkflowState.HYPOTHESIZE: frozenset({WorkflowState.DESIGN_PENDING_APPROVAL}),
    WorkflowState.DESIGN_PENDING_APPROVAL: frozenset({WorkflowState.IMPLEMENT_PENDING_APPROVAL}),
    WorkflowState.IMPLEMENT_PENDING_APPROVAL: frozenset({WorkflowState.EXECUTE_PENDING_APPROVAL}),
    WorkflowState.EXECUTE_PENDING_APPROVAL: frozenset({WorkflowState.EVALUATE}),
    WorkflowState.EVALUATE: frozenset({
        WorkflowState.ADVANCE, WorkflowState.REJECT, WorkflowState.REVISE, WorkflowState.PIVOT,
    }),
    WorkflowState.ADVANCE: frozenset({WorkflowState.RELEASE_IF_GATE_PASS}),
    WorkflowState.REJECT: frozenset({WorkflowState.STOPPED}),
    WorkflowState.REVISE: frozenset({WorkflowState.STOPPED}),
    WorkflowState.PIVOT: frozenset({WorkflowState.STOPPED}),
    WorkflowState.RELEASE_IF_GATE_PASS: frozenset({WorkflowState.STOPPED}),
}

_APPROVAL_FOR_EDGE = {
    (WorkflowState.DESIGN_PENDING_APPROVAL, WorkflowState.IMPLEMENT_PENDING_APPROVAL): "design",
    (WorkflowState.IMPLEMENT_PENDING_APPROVAL, WorkflowState.EXECUTE_PENDING_APPROVAL): "implementation",
    (WorkflowState.EXECUTE_PENDING_APPROVAL, WorkflowState.EVALUATE): "execution",
}

_REQUIRED_GATES = frozenset({
    "rule_compliance", "mechanism_activation", "competitive_or_product_outcome",
})


def transition(
    current: WorkflowState,
    requested: WorkflowState,
    *,
    plan_hash: str,
    approvals: Sequence[Approval],
    gates: Sequence[GateResult],
) -> WorkflowState:
    """Accept only a legal edge with its recorded approval and release gates."""
    if not isinstance(current, WorkflowState) or not isinstance(requested, WorkflowState):
        raise TransitionError("current and requested must be WorkflowState values")
    if requested not in _EDGES.get(current, frozenset()):
        raise TransitionError(f"illegal transition: {current.value} -> {requested.value}")

    required_stage = _APPROVAL_FOR_EDGE.get((current, requested))
    if required_stage and not any(
        record.stage == required_stage and record.plan_hash == plan_hash for record in approvals
    ):
        raise ApprovalError(f"{required_stage} approval for exact plan hash is required")

    if requested is WorkflowState.RELEASE_IF_GATE_PASS:
        names = [result.name for result in gates]
        if len(names) != len(_REQUIRED_GATES) or set(names) != _REQUIRED_GATES:
            raise GateError("release requires exactly the three registered gates")
        if any(result.status is not GateStatus.PASS for result in gates):
            # No validated configuration exemption is accepted by this interface.
            raise GateError("each release gate must PASS")

    return requested
