"""Synthetic-only reducer tests: no real episodes or forecast execution."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import diagnose_joint_temporal_funnel as funnel


def comparison(delta=-.1, residual=.45, absolute=True, road=False, obstacle=False):
    # Large common-mode variation makes independently subtracted extrema wrong.
    baseline = [[100 * s + 7 * r for r in range(5)] for s in range(19)]
    alternative = [[value + delta for value in row] for row in baseline]
    differences = [[a - b for a, b in zip(row, base)] for row, base in zip(alternative, baseline)]
    lo, hi = min(funnel.flat(differences)), max(funnel.flat(differences))
    reasons = []
    if not absolute:
        reasons.append("unknown_trajectory_support")
    if road:
        reasons.append("road_margin")
    if obstacle:
        reasons.append("obstacle_margin")
    if lo - residual <= .05 and hi + residual >= -.05:
        reasons.append("empirical_order_uncertain")
    return dict(reference_costs=[baseline, alternative],
                reference_delta=[[[0.] * 5 for _ in range(19)], differences],
                delta_interval=[[0., 0.], [lo - residual, hi + residual]],
                common_support=True, cost_supported=[True, True], comparison_ticks=17,
                supported_ticks=[[17] * 19 for _ in range(2)],
                footprint_known_ticks=[[17] * 19 for _ in range(2)],
                absolute_supported=[[True] * 19, [absolute] * 19],
                veto=[[False] * 19, [not absolute or road or obstacle] * 19],
                abstain_reasons=[[], reasons])


def decision(step=1, comp=None, reasons=None, intervene=False):
    return dict(event="decision", cell="1/123", step=step,
                policy=dict(comparison=comp, reasons=reasons or [], eligible=comp is not None,
                            intervention=intervene, observer_invalid_reasons=[],
                            proposal_action=[.02, 0., .03], observer_state=dict(gas_state=.27)))


def reduce(rows, residual=.45):
    return funnel.reduce_decisions(rows, dict(paired_cost_residual=residual, cost_floor=.05))


class FunnelTests(unittest.TestCase):
    def test_overlapping_pre_reasons_have_fixed_order_and_overlap(self):
        rows = [decision(reasons=["action_outside_pilot_envelope", "observer_invalid", "mapping_invalid"]),
                decision(2, reasons=["mapping_invalid", "action_outside_pilot_envelope"]),
                decision(3, reasons=["feedback_capture_unavailable", "calibration_invalid"])]
        summary = reduce(rows)["summary"]
        self.assertEqual(summary["pre_exclusive"]["observer_or_forward"], 1)
        self.assertEqual(summary["pre_exclusive"]["mapping"], 1)
        self.assertEqual(summary["pre_exclusive"]["shield_recovery_feedback"], 1)
        stats = summary["literal_reasons_pre"]
        self.assertEqual(stats["nonexclusive_counts"]["mapping_invalid"], 2)
        self.assertIn(dict(reasons=["action_outside_pilot_envelope", "mapping_invalid"], count=2), stats["pairwise_overlap"])
        self.assertEqual(sum(r["count"] for r in summary["decision_group_counts"]), 3)

    def test_missing_comparison_inferred_without_invented_cost(self):
        report = reduce([decision(), decision(2, reasons=["comparison_error"])])
        self.assertEqual(report["comparisons"], [])
        self.assertEqual(report["summary"]["denominators"]["computed_comparisons"], 0)
        self.assertEqual(report["summary"]["pre_exclusive"]["comparison_missing"], 2)
        self.assertIsNone(report["summary"]["best_beneficial_margins"]["full_cost_supported"]["nominal_beneficial_margin"])
        self.assertIn("NOT_COMPUTED", report["method"]["missing_comparison"])

    def test_shared_variants_subtracted_before_extrema(self):
        record = reduce([decision(comp=comparison())])["comparisons"][0]
        self.assertAlmostEqual(record["central_midpoint_delta"], -.1)
        self.assertAlmostEqual(record["shared_19x5_interval"][0], -.1)
        self.assertAlmostEqual(record["shared_19x5_interval"][1], -.1)
        self.assertEqual(record["cost_category"], "residual_allowance_removes_robust_gain")

    def test_misaligned_saved_delta_is_rejected(self):
        comp = comparison()
        comp["reference_delta"][1][0][1] = 100
        with self.assertRaisesRegex(ValueError, "aligned reference_delta"):
            reduce([decision(comp=comp)])

    def test_variant_shape_mismatch_rejected(self):
        comp = comparison()
        comp["reference_costs"][1].pop()
        with self.assertRaisesRegex(ValueError, "shape"):
            reduce([decision(comp=comp)])

    def test_material_baseline_advantage_separate_from_uncertainty(self):
        report = reduce([decision(comp=comparison(.8)), decision(2, comp=comparison(.1)),
                         decision(3, comp=comparison(-.1))])
        self.assertEqual([r["cost_category"] for r in report["comparisons"]],
                         ["material_baseline_advantage", "central_midpoint_no_nominal_gain",
                          "residual_allowance_removes_robust_gain"])
        self.assertEqual(report["summary"]["comparison_predicates"]["nominal_no_gain"]["count"], 2)

    def test_stress_alone_removes_nominal_gain(self):
        comp = comparison(-.1)
        comp["reference_costs"][1][1][2] = comp["reference_costs"][0][1][2] + .1
        comp["reference_delta"][1][1][2] = .1
        comp["delta_interval"][1][1] = .55
        record = reduce([decision(comp=comp)])["comparisons"][0]
        self.assertEqual(record["cost_category"], "shared_stress_removes_nominal_gain")
        self.assertEqual(record["shared_extreme_variants"]["upper"], [[1, 2]])

    def test_cost_and_risk_remain_independent(self):
        comp = comparison(-1, road=True, obstacle=True, absolute=False)
        report = reduce([decision(comp=comp)])
        record = report["comparisons"][0]
        self.assertEqual(record["cost_category"], "robust_gain")
        self.assertEqual(record["performance_category"], "gain_but_risk_veto")
        self.assertEqual(record["post_first_failure"], "absolute_support_missing")
        cross = report["summary"]["risk_cost_cross_table"][0]
        self.assertEqual(cross["cost_category"], "robust_gain")
        self.assertEqual(len(cross["risk_reasons"]), 3)

    def test_full_cost_support_is_not_absolute_support(self):
        report = reduce([decision(comp=comparison(absolute=False))])
        record = report["comparisons"][0]
        self.assertTrue(record["full_cost_support"])
        self.assertFalse(record["alternative_absolute_support"])
        self.assertEqual(record["cost_category"], "residual_allowance_removes_robust_gain")

    def test_incomplete_support_keeps_unknown_not_zero(self):
        comp = comparison()
        comp["common_support"] = False
        comp["cost_supported"] = [False, False]
        comp["supported_ticks"][1][18] = 16
        comp["reference_costs"][1][18][4] = None
        comp["reference_delta"][1][18][4] = None
        comp["delta_interval"] = [[None, None], [None, None]]
        record = reduce([decision(comp=comp)])["comparisons"][0]
        self.assertEqual(record["cost_category"], "common_17_tick_cost_unsupported")
        self.assertIsNone(record["shared_19x5_interval"])
        self.assertIsNone(record["final_interval"])
        self.assertAlmostEqual(record["central_midpoint_delta"], -.1)

    def test_best_safe_margin_and_source_float32_candidate(self):
        report = reduce([decision(comp=comparison(-.2)), decision(2, comp=comparison(-2, road=True))])
        best = report["summary"]["best_beneficial_margins"]["full_cost_supported_no_risk"]
        self.assertEqual(best["denominator"], 1)
        self.assertEqual(best["nominal_beneficial_margin"]["decision"], "1/123/1")
        record = report["comparisons"][0]
        self.assertEqual(record["candidate_action"][0], 0.)
        self.assertAlmostEqual(record["candidate_action"][1], .05)
        self.assertEqual(record["candidate_action"][2], 0.)
        self.assertEqual(record["observer_gas_state"], .27)

    def test_gain_must_agree_with_intervention(self):
        with self.assertRaisesRegex(ValueError, "intervention"):
            reduce([decision(comp=comparison(-1))])
        report = reduce([decision(comp=comparison(-1), intervene=True)])
        self.assertEqual(report["summary"]["interventions"], 1)

    def test_boundaries_are_not_material_gains(self):
        self.assertEqual(funnel.cost_category(-.05, [-.05, -.05], [-.05, -.05], True),
                         "central_midpoint_no_nominal_gain")
        self.assertEqual(funnel.cost_category(-.1, [-.1, -.05], [-.2, .05], True),
                         "shared_stress_removes_nominal_gain")
        self.assertEqual(funnel.cost_category(-.1, [-.1, -.1], [-.2, -.05], True),
                         "residual_allowance_removes_robust_gain")

    def test_events_not_decision_ends_and_duplicates(self):
        report = reduce([dict(event="act"), decision()])
        self.assertEqual(report["summary"]["denominators"]["decision_ends"], 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            reduce([decision(), decision()])

    def test_input_immutable_and_json_finite(self):
        rows = [decision(comp=comparison())]
        original = deepcopy(rows)
        json.dumps(reduce(rows), allow_nan=False)
        self.assertEqual(rows, original)

    def test_existing_output_refused_before_input_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "funnel.json"
            path.touch()
            with patch.object(funnel, "load_pilot") as load:
                with self.assertRaises(FileExistsError):
                    funnel.main(["--output", str(path)])
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
