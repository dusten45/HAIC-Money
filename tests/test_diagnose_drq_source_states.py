import unittest

import numpy as np

from scripts.diagnose_drq_source_states import sample_decisions, _cosine


class TestSourceStatePredeclaredSampling(unittest.TestCase):
    def test_first_every_twenty_five_and_last_twenty(self):
        sample = sample_decisions(64)
        self.assertEqual(len(sample), 41)
        self.assertTrue(set(range(1, 21)) <= sample)
        self.assertTrue(set(range(45, 65)) <= sample)
        self.assertIn(25, sample)
        self.assertIn(50, sample)
        self.assertNotIn(24, sample)

    def test_minimum_episode_never_indexes_outside_trace(self):
        self.assertEqual(sample_decisions(1), {1})
        self.assertEqual(sample_decisions(20), set(range(1, 21)))
