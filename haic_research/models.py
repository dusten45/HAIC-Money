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


def _freeze_tuple_fields(record: object, names: tuple[str, ...]) -> None:
    for name in names:
        value = getattr(record, name)
        if not isinstance(value, (tuple, list)):
            raise TypeError(f"{name} must be a tuple or list")
        object.__setattr__(record, name, _freeze(value))


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
    GATE_REVIEW = "GATE_REVIEW"
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

    def __post_init__(self) -> None:
        _freeze_tuple_fields(self, ("source_paths",))


@dataclass(frozen=True)
class GateResult:
    name: str
    status: GateStatus
    rationale: str = ""
    evidence_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", GateStatus(self.status))
        _freeze_tuple_fields(self, ("evidence_paths",))


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
        _freeze_tuple_fields(self, ("data_ids", "map_ids", "split_ids", "output_paths"))


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

    def __post_init__(self) -> None:
        _freeze_tuple_fields(self, ("gate_results", "evidence_paths"))


@dataclass(frozen=True)
class AgentReport:
    fact: str
    inference: str
    unknown: str
    recommendation: str
    source_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        _freeze_tuple_fields(self, ("source_paths",))


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

    def __post_init__(self) -> None:
        _freeze_tuple_fields(self, ("hypothesis_ids", "allowed_read_paths", "allowed_write_paths"))


@dataclass(frozen=True)
class ExperimentResult:
    candidate_id: str
    comparison_id: str
    split_id: str
    map_ids: tuple[str, ...]
    seed_ids: tuple[str, ...]
    completion_count: int
    episode_count: int
    median_finished_lap_ms: float | None
    mean_incomplete_progress: float
    p90_finished_lap_ms: float | None
    collisions: float
    damage: float
    act_latency_p95_ms: float
    rule_compliance: GateStatus
    mechanism_activation: GateStatus
    official_score: float | None = None
    eligibility: str = "candidate"

    def __post_init__(self) -> None:
        _freeze_tuple_fields(self, ("map_ids", "seed_ids"))
        object.__setattr__(self, "rule_compliance", GateStatus(self.rule_compliance))
        object.__setattr__(self, "mechanism_activation", GateStatus(self.mechanism_activation))
        if not all((self.candidate_id, self.comparison_id, self.split_id, self.map_ids, self.seed_ids)):
            raise ValueError("experiment result requires candidate and comparison protocol identifiers")
        if len(set(self.map_ids)) != len(self.map_ids) or len(set(self.seed_ids)) != len(self.seed_ids):
            raise ValueError("map and seed identifiers must be unique")
        if self.episode_count <= 0 or not 0 <= self.completion_count <= self.episode_count:
            raise ValueError("completion count must fit a positive episode denominator")
        if self.eligibility not in {"candidate", "teacher", "smoke", "diagnostic"}:
            raise ValueError("unknown experiment eligibility label")


@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool
    reason: str


@dataclass(frozen=True)
class AgentReportEnvelope:
    handoff_path: str
    owner: str
    edited_paths: tuple[str, ...]
    report: AgentReport

    def __post_init__(self) -> None:
        _freeze_tuple_fields(self, ("edited_paths",))
        if not isinstance(self.report, AgentReport):
            raise TypeError("report must be an AgentReport")
