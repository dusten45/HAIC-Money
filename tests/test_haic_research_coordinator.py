"""Assignment ownership and report integration checks."""

import unittest
from dataclasses import FrozenInstanceError, replace

from haic_research.coordinator import assign_work, integrate_reports
from haic_research.models import AgentReport, AgentReportEnvelope, GateStatus, Hypothesis, WorkAssignment


def hypothesis(identifier, direction):
    return Hypothesis(identifier, "evidence/source.md", "rule", "observable", direction,
                      "endpoint", "eligible", "control", "falsifier", "experiment", "budget",
                      ("evidence/source.md",))


def envelope(assignment, **changes):
    values = dict(handoff_path=assignment.handoff_path, owner=assignment.owner,
                  edited_paths=assignment.allowed_write_paths,
                  report=AgentReport("fact", "inference", "unknown", "recommendation",
                                     ("evidence/source.md",)))
    values.update(changes)
    return AgentReportEnvelope(**values)


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.hypotheses = [hypothesis(str(i), f"direction-{i}") for i in range(4)]
        self.assignments = assign_work("batch-1", self.hypotheses)

    def test_one_owner_per_direction_and_nonoverlapping_scope(self):
        self.assertEqual(len(self.assignments), 4)
        self.assertEqual({a.direction for a in self.assignments},
                         {h.allowed_action_or_state_change for h in self.hypotheses})
        self.assertEqual(len({a.owner for a in self.assignments}), 4)
        self.assertEqual(len({path for a in self.assignments for path in a.allowed_write_paths}), 4)
        self.assertEqual(len({a.handoff_path for a in self.assignments}), 4)

    def test_integration_requires_one_complete_report_per_assignment(self):
        reports = [envelope(a) for a in self.assignments]
        integrated = integrate_reports(self.assignments, reports)
        self.assertEqual({g.name for g in integrated.gate_results},
                         {"rule_compliance", "mechanism_activation", "competitive_or_product_outcome"})
        self.assertTrue(all(g.status == GateStatus.UNKNOWN for g in integrated.gate_results))
        self.assertIn("evidence/source.md", integrated.evidence_paths)
        for invalid in (reports[:-1], reports + reports[:1]):
            with self.assertRaises(ValueError):
                integrate_reports(self.assignments, invalid)
        with self.assertRaises(ValueError):
            integrate_reports(self.assignments, [reports[0], reports[0]] + reports[2:])

    def test_envelope_detaches_edited_paths_and_is_frozen(self):
        edited = [self.assignments[0].handoff_path]
        item = envelope(self.assignments[0], edited_paths=edited)
        edited.append("outside.py")
        self.assertEqual(item.edited_paths, (self.assignments[0].handoff_path,))
        with self.assertRaises(FrozenInstanceError):
            item.owner = "other"
        changed = replace(item, edited_paths=[self.assignments[0].handoff_path])
        self.assertIsInstance(changed.edited_paths, tuple)

    def test_missing_report_fields_and_out_of_scope_edits_rejected(self):
        for report in (AgentReport("", "inference", "unknown", "recommendation", ("source",)),
                       AgentReport("fact", "", "unknown", "recommendation", ("source",)),
                       AgentReport("fact", "inference", "", "recommendation", ("source",)),
                       AgentReport("fact", "inference", "unknown", "", ("source",)),
                       AgentReport("fact", "inference", "unknown", "recommendation", ())):
            with self.subTest(report=report), self.assertRaises(ValueError):
                integrate_reports(self.assignments, [envelope(self.assignments[0], report=report)] +
                                  [envelope(a) for a in self.assignments[1:]])
        with self.assertRaises(ValueError):
            integrate_reports(self.assignments,
                              [envelope(self.assignments[0], edited_paths=("other/file.py",))] +
                              [envelope(a) for a in self.assignments[1:]])

    def test_duplicate_or_overlapping_assignment_ownership_rejected(self):
        first, second = self.assignments[:2]
        modified = WorkAssignment(second.direction, second.hypothesis_ids, first.owner,
                                  second.allowed_read_paths, second.allowed_write_paths,
                                  second.handoff_path)
        for assignments in (self.assignments + [first],
                            [first, modified] + self.assignments[2:]):
            with self.assertRaises(ValueError):
                integrate_reports(assignments, [envelope(a) for a in assignments])
        overlap = WorkAssignment(second.direction, second.hypothesis_ids, second.owner,
                                 second.allowed_read_paths, first.allowed_write_paths,
                                 second.handoff_path)
        with self.assertRaises(ValueError):
            integrate_reports([first, overlap] + self.assignments[2:],
                              [envelope(a) for a in [first, overlap] + self.assignments[2:]])

    def test_wrong_owner_and_handoff_path_rejected(self):
        for changed in (dict(owner="another"), dict(handoff_path="another/report.md")):
            with self.assertRaises(ValueError):
                integrate_reports(self.assignments,
                                  [envelope(self.assignments[0], **changed)] +
                                  [envelope(a) for a in self.assignments[1:]])


if __name__ == "__main__":
    unittest.main()
