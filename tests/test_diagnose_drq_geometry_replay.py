import unittest

from scripts.diagnose_drq_geometry_replay import progress_bin, thirds


class TestReplayProxyBins(unittest.TestCase):
    def test_frozen_geometry_progress_bins(self):
        self.assertEqual(progress_bin(None), "budget_stop_unknown")
        self.assertEqual(progress_bin(0.19999), "below_0.2")
        self.assertEqual(progress_bin(0.2), "0.2_to_below_0.9")
        self.assertEqual(progress_bin(0.9), "at_least_0.9")

    def test_episode_thirds_are_exclusive(self):
        self.assertEqual([thirds(i, 9) for i in range(9)],
                         ["early"] * 3 + ["middle"] * 3 + ["late"] * 3)
