import unittest

import numpy as np

from scripts.diagnose_drq_geometry_regression import classification, first_divergence


class TestGeometryRegression(unittest.TestCase):
    def test_transition_matrix(self):
        self.assertEqual(classification(True, False), "lost")
        self.assertEqual(classification(False, True), "gained")
        self.assertEqual(classification(True, True), "both_finished")
        self.assertEqual(classification(False, False), "both_failed")

    def test_sustained_action_gap_is_one_based_and_not_a_single_spike(self):
        source = np.zeros((7, 3), dtype=np.float32)
        changed = source.copy()
        changed[0, 0] = 0.2
        changed[2:5, 1] = 0.1
        divergence = first_divergence(source, changed)
        self.assertEqual(divergence["first_nontrivial_action_decision"], 1)
        self.assertEqual(divergence["first_sustained_action_decision"], 3)

    def test_short_trajectory_has_no_sustained_discrepancy(self):
        source = np.zeros((2, 3), dtype=np.float32)
        changed = np.full((4, 3), 0.4, dtype=np.float32)
        divergence = first_divergence(source, changed)
        self.assertIsNone(divergence["first_sustained_action_decision"])
        self.assertEqual(divergence["overlap_decisions"], 2)
