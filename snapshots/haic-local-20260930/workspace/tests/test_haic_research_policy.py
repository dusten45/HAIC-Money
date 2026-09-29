"""Completion-first candidate policy and research batch checks."""

import unittest

from haic_research.models import CycleSummary, ExperimentResult, Hypothesis
from haic_research.policy import (
    ComparisonMismatchError, promotion_decision, rank_candidates,
    should_pivot, validate_batch,
)


def result(candidate_id="candidate", **changes):
    fields = dict(candidate_id=candidate_id, comparison_id="c1", split_id="held_out",
                  map_ids=("m1",), seed_ids=("s1",), completion_count=8,
                  episode_count=10, median_finished_lap_ms=24000,
                  mean_incomplete_progress=.5, p90_finished_lap_ms=30000,
                  collisions=1, damage=2, act_latency_p95_ms=10,
                  rule_compliance="PASS", mechanism_activation="PASS",
                  eligibility="candidate", official_score=None)
    fields.update(changes)
    return ExperimentResult(**fields)


def hypothesis(identifier, direction):
    return Hypothesis(identifier, "source", "rule", "observable", direction,
                      "endpoint", "eligible", "control", "falsifier", "experiment", "budget")


class CompletionPriorityTests(unittest.TestCase):
    def test_higher_completion_rate_beats_faster_lap(self):
        reliable = result("reliable")
        fast = result("fast", completion_count=6, median_finished_lap_ms=18000)
        self.assertEqual([r.candidate_id for r in rank_candidates([fast, reliable])],
                         ["reliable", "fast"])

    def test_tie_breakers_follow_declared_order_and_score_is_separate(self):
        baseline = result("baseline")
        for field, better in (("median_finished_lap_ms", 23000),
                              ("mean_incomplete_progress", .6),
                              ("p90_finished_lap_ms", 29000),
                              ("collisions", 0), ("damage", 1),
                              ("act_latency_p95_ms", 9)):
            with self.subTest(field=field):
                self.assertEqual(rank_candidates([baseline, result("better", **{field: better})])[0].candidate_id,
                                 "better")
        self.assertEqual(rank_candidates([result("a", official_score=0),
                                          result("b", official_score=999)])[0].candidate_id, "a")

    def test_mismatched_comparison_protocol_is_rejected(self):
        for changed in (dict(comparison_id="c2"), dict(split_id="tune"),
                        dict(map_ids=("m2",)), dict(seed_ids=("s2",)),
                        dict(episode_count=11)):
            with self.subTest(changed=changed), self.assertRaises(ComparisonMismatchError):
                rank_candidates([result(), result("other", **changed)])

    def test_ineligible_rows_are_excluded(self):
        records = [result("teacher", eligibility="teacher"),
                   result("smoke", eligibility="smoke"),
                   result("diagnostic", eligibility="diagnostic"), result("eligible")]
        self.assertEqual([r.candidate_id for r in rank_candidates(records)], ["eligible"])
        self.assertEqual(len(records), 4)

    def test_promotion_requires_both_gates_and_improvement(self):
        control = result("control", completion_count=7)
        self.assertTrue(promotion_decision(result(), control).eligible)
        for field in ("rule_compliance", "mechanism_activation"):
            for status in ("UNKNOWN", "FAIL", "NOT_APPLICABLE"):
                with self.subTest(field=field, status=status):
                    self.assertFalse(promotion_decision(result(**{field: status}), control).eligible)
        self.assertFalse(promotion_decision(result(completion_count=6), control).eligible)
        self.assertFalse(promotion_decision(result(eligibility="teacher"), control).eligible)
        self.assertFalse(promotion_decision(result(), result("teacher", eligibility="teacher")).eligible)

    def test_promotion_requires_matched_protocol(self):
        with self.assertRaises(ComparisonMismatchError):
            promotion_decision(result(), result("control", seed_ids=("other",)))


class ResultMetricBoundaryTests(unittest.TestCase):
    def test_counts_are_actual_integers_and_valid_denominators(self):
        for changes in ({'completion_count': True}, {'completion_count': 1.5}, {'episode_count': True},
                        {'episode_count': 10.0}, {'episode_count': 0}, {'completion_count': -1}, {'completion_count': 11}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                result(**changes)

    def test_nonfinite_nonnumeric_and_negative_metrics_refused(self):
        for name in ('median_finished_lap_ms', 'p90_finished_lap_ms', 'mean_incomplete_progress',
                     'collisions', 'damage', 'act_latency_p95_ms', 'official_score'):
            for value in (float('nan'), float('inf'), float('-inf'), True, '10', 10 ** 500):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    result(**{name: value})
            if name != 'official_score':
                with self.subTest(name=name), self.assertRaises(ValueError):
                    result(**{name: -1})

    def test_finished_lap_and_progress_consistency(self):
        for changes in ({'median_finished_lap_ms': None}, {'p90_finished_lap_ms': None},
                        {'median_finished_lap_ms': 0}, {'p90_finished_lap_ms': 20000},
                        {'mean_incomplete_progress': 1.1}, {'completion_count': 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                result(**changes)
        self.assertEqual(result(completion_count=0, median_finished_lap_ms=None, p90_finished_lap_ms=None).completion_count, 0)

    def test_identifier_types_and_empty_identifiers_refused(self):
        for changes in ({'candidate_id': 12}, {'comparison_id': ' '}, {'map_ids': (True,)}, {'seed_ids': ('',)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                result(**changes)


class BatchAndPivotTests(unittest.TestCase):
    def test_batch_limits_and_duplicate_ids(self):
        valid = [hypothesis(str(i), f"direction-{i}") for i in range(4)]
        self.assertIsNone(validate_batch(valid))
        self.assertIsNone(validate_batch([hypothesis(str(i), f"direction-{i//2}") for i in range(8)]))
        for batch in (valid[:3], valid + [hypothesis("4", "direction-0"),
                                          hypothesis("5", "direction-0")],
                      [hypothesis(str(i), f"direction-{i//2}") for i in range(9)],
                      valid + [hypothesis("0", "another")]):
            with self.assertRaises(ValueError):
                validate_batch(batch)

    def test_infra_invalid_cycles_do_not_advance_streak(self):
        bad = CycleSummary(True, True, False, False)
        invalid = CycleSummary(False, False, False, True)
        self.assertFalse(should_pivot([bad, invalid, bad]))
        self.assertTrue(should_pivot([bad, bad, bad]))
        self.assertTrue(should_pivot([bad, invalid, bad, bad]))
        self.assertFalse(should_pivot([bad, CycleSummary(True, True, True, False), bad]))
        self.assertFalse(should_pivot([bad, CycleSummary(False, True, False, False), bad]))


if __name__ == "__main__":
    unittest.main()
