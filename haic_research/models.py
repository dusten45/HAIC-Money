"""Immutable value records for the HAIC research workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Mapping


def _freeze(value: object) -> object:
    """Detach nested record metadata from caller-owned mutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


class WorkflowState(str, Enum):
    STOPPED = "STOPPED"
    DISCOVER = "DISCOVER"
    HYPOTHESIZE = "HYPOTHESIZE"
    DESIGN_PENDING_APPROVAL = "DESIGN_PENDING_APPROVAL"
    IMPLEMENT_PENDING_APPROVAL = "IMPLEMENT_PENDING_APPROVAL"
    EXECUTE_PENDING_APPROVAL = "EXECUTE_PENDING_APPROVAL"
    EVALUATE = "EVALUATE"
    ADVANCE = "ADVANCE"
    REJECT = "REJECT"
    REVISE = "REVISE"
    PIVOT = "PIVOT"
    RELEASE_IF_GATE_PASS = "RELEASE_IF_GATE_PASS"


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class Approval:
    stage: str
    plan_hash: str
    approved_at: datetime
    source_ref: str

    def __post_init__(self) -> None:
        if self.stage not in {"design", "implementation", "execution"}:
            raise ValueError(f"unknown approval stage: {self.stage}")
        if not self.plan_hash or not self.source_ref:
            raise ValueError("approval requires a plan hash and event source reference")
        if self.approved_at.tzinfo is None or self.approved_at.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("approval timestamp must use UTC")


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    source_ref: str
    rule_or_requirement: str
    observable_information: str
    allowed_action_or_state_change: str
    expected_success_endpoint: str
    eligible_state: str
    control: str
    falsifier: str
    smallest_decisive_experiment: str
    resource_and_risk_gate: str
    source_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateResult:
    name: str
    status: GateStatus
    rationale: str = ""
    evidence_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", GateStatus(self.status))


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    purpose: str
    hypothesis_hash: str
    approval_hash: str
    candidate_revision: str
    control_revision: str
    candidate_package_hash: str
    control_package_hash: str
    tool_versions: Mapping[str, str]
    runtime_versions: Mapping[str, str]
    data_ids: tuple[str, ...]
    map_ids: tuple[str, ...]
    split_ids: tuple[str, ...]
    resource_limits: Mapping[str, object]
    permission_limits: Mapping[str, object]
    output_paths: tuple[str, ...]
    source_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in ("tool_versions", "runtime_versions", "resource_limits",
                     "permission_limits", "source_hashes"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


@dataclass(frozen=True)
class RunEvent:
    timestamp: datetime
    kind: str
    workflow_state: WorkflowState
    approval_ref: str | None = None
    execution_status: str | None = None
    error: str | None = None
    resource_usage: Mapping[str, object] = field(default_factory=dict)
    mechanism_signal: str | None = None
    endpoint: str | None = None
    correction_ref: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("event timestamp must use UTC")
        object.__setattr__(self, "resource_usage", _freeze(self.resource_usage))


@dataclass(frozen=True)
class IntegrationReport:
    gate_results: tuple[GateResult, ...]
    evidence_paths: tuple[str, ...]


@dataclass(frozen=True)
class AgentReport:
    fact: str
    inference: str
    unknown: str
    recommendation: str
    source_paths: tuple[str, ...]


@dataclass(frozen=True)
class CycleSummary:
    valid: bool
    protocol_match: bool
    improved: bool
    infra_invalid: bool


@dataclass(frozen=True)
class WorkAssignment:
    direction: str
    hypothesis_ids: tuple[str, ...]
    owner: str
    allowed_read_paths: tuple[str, ...]
    allowed_write_paths: tuple[str, ...]
    handoff_path: str
