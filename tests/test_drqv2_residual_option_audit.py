import unittest

import numpy as np

from haic.algorithms.drq_v2 import ResidualOption
from scripts.diagnose.drqv2_residual_options import (
    policy_choice_summary,
    q_gap_summary,
    target_choice_summary,
)


class TestResidualOptionAudit(unittest.TestCase):
    def test_q_gap_quantiles_match_keep_relative_definition(self):
        values = np.asarray([
            [1.0, 2.0, 0.0, 0.5, -1.0],
            [3.0, 1.0, 0.0, 0.0, 0.0],
        ])
        summary = q_gap_summary(values)
        self.assertEqual(summary["states"], 2)
        self.assertAlmostEqual(summary["positive_non_keep_advantage_fraction"], 0.5)
        self.assertAlmostEqual(summary["gap_quantiles"]["0.5"], -0.5)

    def test_policy_summary_distinguishes_option_selection_from_effective_action(self):
        values = np.asarray([
            [0.0, 1.0, 4.0, 0.0, 0.0],
            [0.0, 1.0, 4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0, 0.0, 0.0],
        ])
        actions = np.asarray([
            [1.0, 0.0, 0.0],  # Positive steering residual clips to the base action.
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32)
        observed = np.asarray([2, 2, 0])
        summary = policy_choice_summary(values, actions, observed, margin=2.0)
        self.assertEqual(summary["margin_gated_choice_counts"]["STEER_PLUS"], 2)
        self.assertEqual(summary["margin_gated_option_selection_fraction"], 2 / 3)
        self.assertEqual(summary["margin_gated_effective_action_change_fraction"], 1 / 3)
        self.assertEqual(summary["current_head_greedy_vs_recorded_behavior_agreement_fraction"], 1.0)

    def test_target_selector_mismatch_excludes_terminal_rows(self):
        q_next = np.asarray([
            [0.0, 5.0, 2.0, 0.0, 0.0],
            [0.0, 5.0, 2.0, 0.0, 0.0],
            [3.0, 1.0, 0.0, 0.0, 0.0],
        ])
        summary = target_choice_summary(q_next, np.asarray([False, True, False]), margin=5.0)
        self.assertEqual(summary["bootstrap_rows"], 2)
        self.assertEqual(summary["selector_mismatch_count"], 1)
        self.assertEqual(summary["selector_mismatch_fraction"], 0.5)
        self.assertEqual(
            summary["unrestricted_next_action_counts"]["STEER_MINUS"], 1,
        )
        self.assertEqual(
            summary["margin_gated_next_action_counts"]["KEEP"], 2,
        )

    def test_target_selector_handles_all_terminal_rows(self):
        summary = target_choice_summary(
            np.zeros((2, len(ResidualOption))),
            np.asarray([True, True]),
            margin=0.0,
        )
        self.assertEqual(summary["bootstrap_rows"], 0)
        self.assertIsNone(summary["selector_mismatch_fraction"])


if __name__ == "__main__":
    unittest.main()
