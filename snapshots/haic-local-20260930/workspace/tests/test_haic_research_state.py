"""Focused contract checks for immutable records and approval-gated states."""

import unittest
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timezone

from haic_research.models import (
    AgentReport,
    Approval,
    CheckpointRef,
    CycleSummary,
    GateResult,
    GateStatus,
    Hypothesis,
    IntegrationReport,
    RunEvent,
    RunManifest,
    WorkAssignment,
    WorkflowState,
)
from haic_research.state import ApprovalError, GateError, TransitionError, transition


PLAN_HASH = "plan-sha256"


def approval(stage, plan_hash=PLAN_HASH):
    return Approval(stage, plan_hash, datetime(2026, 9, 26, tzinfo=timezone.utc), "user:42")


def passing_gates():
    return [GateResult(name, GateStatus.PASS, "checked", ("runs/r1/report.json",)) for name in (
        "rule_compliance", "mechanism_activation", "competitive_or_product_outcome"
    )]


class SchemaTests(unittest.TestCase):
    def test_state_and_gate_vocabularies_are_complete(self):
        self.assertEqual({state.value for state in WorkflowState}, {
            "STOPPED", "DISCOVER", "HYPOTHESIZE", "DESIGN_PENDING_APPROVAL",
            "IMPLEMENT_PENDING_APPROVAL", "EXECUTE_PENDING_APPROVAL", "EVALUATE",
            "ADVANCE", "REJECT", "REVISE", "PIVOT", "GATE_REVIEW_ADVANCE", "GATE_REVIEW_REJECT", "GATE_REVIEW_REVISE", "GATE_REVIEW_PIVOT", "RELEASE_IF_GATE_PASS",
        })
        self.assertEqual({status.value for status in GateStatus}, {
            "PASS", "FAIL", "UNKNOWN", "NOT_APPLICABLE"
        })

    def test_records_expose_required_fields_and_are_frozen(self):
        expected = {
            Approval: {"stage", "plan_hash", "approved_at", "source_ref"},
            Hypothesis: {"hypothesis_id", "source_ref", "rule_or_requirement",
                         "observable_information", "allowed_action_or_state_change",
                         "expected_success_endpoint", "eligible_state", "control", "falsifier",
                         "smallest_decisive_experiment", "resource_and_risk_gate", "source_paths"},
            GateResult: {"name", "status", "rationale", "evidence_paths"},
            CheckpointRef: {"path", "sha256"},
            RunManifest: {"run_id", "purpose", "hypothesis_hash", "approval_hash",
                          "candidate_revision", "control_revision", "candidate_package_hash",
                          "control_package_hash", "tool_versions", "runtime_versions",
                          "data_ids", "map_ids", "split_ids", "resource_limits",
                          "permission_limits", "output_paths", "source_hashes", "plan_hash", "cycle_id",
                          "checkpoint_ref", "predecessor_run_id", "predecessor_decision_ref",
                          "comparison_id", "seed_ids", "comparison_episode_count"},
            RunEvent: {"timestamp", "kind", "workflow_state", "approval_ref",
                       "execution_status", "error", "resource_usage", "mechanism_signal",
                       "endpoint", "correction_ref", "event_id", "approval_stage", "approved_plan_hash",
                       "checkpoint_ref"},
            IntegrationReport: {"gate_results", "evidence_paths"},
            AgentReport: {"fact", "inference", "unknown", "recommendation", "source_paths"},
            CycleSummary: {"valid", "protocol_match", "improved", "infra_invalid"},
            WorkAssignment: {"direction", "hypothesis_ids", "owner", "allowed_read_paths",
                             "allowed_write_paths", "handoff_path"},
        }
        for cls, names in expected.items():
            with self.subTest(cls=cls.__name__):
                self.assertEqual({field.name for field in fields(cls)}, names)
                self.assertTrue(cls.__dataclass_params__.frozen)
        item = approval("design")
        with self.assertRaises(FrozenInstanceError):
            item.plan_hash = "other"

    def test_approval_timestamp_requires_utc(self):
        with self.assertRaises(ValueError):
            Approval("design", PLAN_HASH, datetime(2026, 9, 26), "user:42")

    def test_invalid_gate_status_is_rejected(self):
        with self.assertRaises(ValueError):
            GateResult("rule_compliance", "MAYBE", "", ())

    def test_record_mapping_values_cannot_be_changed_after_creation(self):
        supplied = {"cpu": {"seconds": 10}}
        event = RunEvent(datetime(2026, 9, 26, tzinfo=timezone.utc), "resource",
                         WorkflowState.EVALUATE, "resource-1", resource_usage=supplied)
        supplied["cpu"]["seconds"] = 20
        self.assertEqual(event.resource_usage["cpu"]["seconds"], 10)
        with self.assertRaises(TypeError):
            event.resource_usage["cpu"]["seconds"] = 30

    def test_event_resource_usage_requires_mapping_before_freezing(self):
        for value in ([], (), None, "not a mapping"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                RunEvent(datetime(2026, 9, 26, tzinfo=timezone.utc), "RESOURCE",
                         WorkflowState.EVALUATE, "resource-1", resource_usage=value)

    def test_tuple_fields_detach_from_caller_owned_lists(self):
        gate = GateResult("rule_compliance", GateStatus.PASS)
        samples = (
            (Hypothesis(*("entry",) * 11), ("source_paths",)),
            (gate, ("evidence_paths",)),
            (RunManifest("run", "purpose", "hypothesis", "approval", "candidate", "control",
                         "candidate-package", "control-package", {}, {}, (), (), (), {}, {}, (), {},
                         PLAN_HASH, "cycle-1"),
             ("data_ids", "map_ids", "split_ids", "output_paths")),
            (IntegrationReport((), ()), ("gate_results", "evidence_paths")),
            (AgentReport("fact", "inference", "unknown", "recommendation", ()),
             ("source_paths",)),
            (WorkAssignment("direction", (), "owner", (), (), "handoff"),
             ("hypothesis_ids", "allowed_read_paths", "allowed_write_paths")),
        )
        for prototype, names in samples:
            for name in names:
                with self.subTest(record=type(prototype).__name__, field=name):
                    first = gate if name == "gate_results" else "first"
                    second = gate if name == "gate_results" else "second"
                    supplied = [first]
                    record = replace(prototype, **{name: supplied})
                    supplied.append(second)
                    self.assertEqual(getattr(record, name), (first,))
                    self.assertIsInstance(getattr(record, name), tuple)


class WorkflowTests(unittest.TestCase):
    def test_discovery_and_hypothesis_need_no_approval(self):
        self.assertEqual(transition(WorkflowState.STOPPED, WorkflowState.DISCOVER,
                                    plan_hash=PLAN_HASH, approvals=[], gates=[]), WorkflowState.DISCOVER)
        self.assertEqual(transition(WorkflowState.DISCOVER, WorkflowState.HYPOTHESIZE,
                                    plan_hash=PLAN_HASH, approvals=[], gates=[]), WorkflowState.HYPOTHESIZE)

    def test_each_approval_edge_requires_its_own_stage_and_exact_hash(self):
        edges = (
            (WorkflowState.DESIGN_PENDING_APPROVAL, WorkflowState.IMPLEMENT_PENDING_APPROVAL, "design"),
            (WorkflowState.IMPLEMENT_PENDING_APPROVAL, WorkflowState.EXECUTE_PENDING_APPROVAL, "implementation"),
            (WorkflowState.EXECUTE_PENDING_APPROVAL, WorkflowState.EVALUATE, "execution"),
        )
        for current, requested, stage in edges:
            with self.subTest(stage=stage):
                other_stage = "execution" if stage != "execution" else "design"
                for records in ([], [approval(other_stage)], [approval(stage, "old-hash")]):
                    with self.assertRaises(ApprovalError):
                        transition(current, requested, plan_hash=PLAN_HASH,
                                   approvals=records, gates=[])
                self.assertEqual(transition(current, requested, plan_hash=PLAN_HASH,
                                            approvals=[approval(stage)], gates=[]), requested)

    def test_cannot_execute_without_matching_execution_approval(self):
        with self.assertRaises(ApprovalError):
            transition(WorkflowState.EXECUTE_PENDING_APPROVAL, WorkflowState.EVALUATE,
                       plan_hash="new", approvals=[], gates=[])

    def test_evaluation_has_four_decision_branches(self):
        for requested in (WorkflowState.ADVANCE, WorkflowState.REJECT,
                          WorkflowState.REVISE, WorkflowState.PIVOT):
            with self.subTest(requested=requested):
                self.assertEqual(transition(WorkflowState.EVALUATE, requested,
                                            plan_hash=PLAN_HASH, approvals=[], gates=[]), requested)

    def test_every_outcome_enters_gate_review(self):
        for outcome in (WorkflowState.ADVANCE, WorkflowState.REJECT,
                        WorkflowState.REVISE, WorkflowState.PIVOT):
            with self.subTest(outcome=outcome):
                self.assertEqual(transition(outcome, WorkflowState[f"GATE_REVIEW_{outcome.value}"],
                                            plan_hash=PLAN_HASH, approvals=[], gates=passing_gates()),
                                 WorkflowState[f"GATE_REVIEW_{outcome.value}"])
                for bypass in (WorkflowState.STOPPED, WorkflowState.RELEASE_IF_GATE_PASS):
                    with self.assertRaises(TransitionError):
                        transition(outcome, bypass, plan_hash=PLAN_HASH,
                                   approvals=[], gates=passing_gates())

    def test_gate_review_requires_exactly_three_registered_results(self):
        for gates in ([], passing_gates()[:2], passing_gates() + passing_gates()[:1],
                      passing_gates() + [GateResult("other", GateStatus.PASS)]):
            with self.assertRaises(GateError):
                transition(WorkflowState.REJECT, WorkflowState.GATE_REVIEW_REJECT,
                           plan_hash=PLAN_HASH, approvals=[], gates=gates)

    def test_gate_review_accepts_non_passing_results_and_can_stop(self):
        for status in GateStatus:
            gates = [GateResult(name, status) for name in (
                "rule_compliance", "mechanism_activation", "competitive_or_product_outcome"
            )]
            self.assertEqual(transition(WorkflowState.ADVANCE, WorkflowState.GATE_REVIEW_ADVANCE,
                                        plan_hash=PLAN_HASH, approvals=[], gates=gates),
                             WorkflowState.GATE_REVIEW_ADVANCE)
            self.assertEqual(transition(WorkflowState.GATE_REVIEW_ADVANCE, WorkflowState.STOPPED,
                                        plan_hash=PLAN_HASH, approvals=[], gates=gates), WorkflowState.STOPPED)

    def test_chained_non_advance_review_cannot_be_relabelled_to_release(self):
        for outcome in (WorkflowState.REJECT, WorkflowState.REVISE, WorkflowState.PIVOT):
            with self.subTest(outcome=outcome):
                reviewed = transition(outcome, WorkflowState[f"GATE_REVIEW_{outcome.value}"],
                                      plan_hash=PLAN_HASH, approvals=[], gates=passing_gates())
                with self.assertRaises(TransitionError):
                    transition(reviewed, WorkflowState.RELEASE_IF_GATE_PASS,
                               plan_hash=PLAN_HASH, approvals=[], gates=passing_gates())
                with self.assertRaises(TransitionError):
                    transition(reviewed, WorkflowState.GATE_REVIEW_ADVANCE,
                               plan_hash=PLAN_HASH, approvals=[], gates=passing_gates())
                self.assertEqual(transition(reviewed, WorkflowState.STOPPED,
                                            plan_hash=PLAN_HASH, approvals=[], gates=passing_gates()),
                                 WorkflowState.STOPPED)

    def test_outcomes_cannot_enter_another_outcomes_review(self):
        outcomes = (WorkflowState.ADVANCE, WorkflowState.REJECT, WorkflowState.REVISE, WorkflowState.PIVOT)
        for outcome in outcomes:
            for other in outcomes:
                if outcome is not other:
                    with self.assertRaises(TransitionError):
                        transition(outcome, WorkflowState[f"GATE_REVIEW_{other.value}"],
                                   plan_hash=PLAN_HASH, approvals=[], gates=passing_gates())

    def test_release_returns_to_stopped_and_new_cycle_starts_at_discover(self):
        self.assertEqual(transition(WorkflowState.RELEASE_IF_GATE_PASS, WorkflowState.STOPPED,
                                    plan_hash=PLAN_HASH, approvals=[], gates=[]), WorkflowState.STOPPED)
        self.assertEqual(transition(WorkflowState.STOPPED, WorkflowState.DISCOVER,
                                    plan_hash="new-cycle-hash", approvals=[], gates=[]), WorkflowState.DISCOVER)

    def test_release_requires_each_gate_to_pass(self):
        self.assertEqual(transition(WorkflowState.GATE_REVIEW_ADVANCE, WorkflowState.RELEASE_IF_GATE_PASS,
                                    plan_hash=PLAN_HASH, approvals=[], gates=passing_gates()),
                         WorkflowState.RELEASE_IF_GATE_PASS)
        for index in range(3):
            for bad in (GateStatus.UNKNOWN, GateStatus.FAIL, GateStatus.NOT_APPLICABLE):
                with self.subTest(gate=index, status=bad):
                    gates = passing_gates()
                    gates[index] = GateResult(gates[index].name, bad, "", ())
                    with self.assertRaises(GateError):
                        transition(WorkflowState.GATE_REVIEW_ADVANCE, WorkflowState.RELEASE_IF_GATE_PASS,
                                   plan_hash=PLAN_HASH, approvals=[], gates=gates)

    def test_cannot_release_with_unknown_gate(self):
        with self.assertRaises(GateError):
            transition(WorkflowState.GATE_REVIEW_ADVANCE, WorkflowState.RELEASE_IF_GATE_PASS,
                       plan_hash="p1", approvals=[],
                       gates=[GateResult("rule_compliance", "UNKNOWN")])

    def test_release_rejects_duplicate_or_unrecognized_gates(self):
        for gates in (passing_gates() + passing_gates()[:1],
                      passing_gates() + [GateResult("other", GateStatus.PASS)]):
            with self.assertRaises(GateError):
                transition(WorkflowState.GATE_REVIEW_ADVANCE, WorkflowState.RELEASE_IF_GATE_PASS,
                           plan_hash=PLAN_HASH, approvals=[], gates=gates)

    def test_illegal_jump_is_rejected(self):
        with self.assertRaises(TransitionError):
            transition(WorkflowState.STOPPED, WorkflowState.EVALUATE,
                       plan_hash=PLAN_HASH, approvals=[approval("execution")], gates=passing_gates())


if __name__ == "__main__":
    unittest.main()
