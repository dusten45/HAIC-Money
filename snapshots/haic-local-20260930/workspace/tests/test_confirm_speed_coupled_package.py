import unittest

from training.confirm_speed_coupled_package import decide


class SpeedCoupledPackageDecisionTests(unittest.TestCase):
    def test_more_finishes_wins_despite_road_exposure_diagnostic(self):
        candidate = {"completed": 12, "denominator": 12, "median_finished_lap_ms": 23750,
                     "invalid_actions": 0, "all_wheels_off_max_streak": 49}
        control = {"completed": 11, "denominator": 12, "median_finished_lap_ms": 24300,
                   "invalid_actions": 0, "all_wheels_off_max_streak": 2}
        self.assertEqual(decide(candidate, control), "ADVANCE")

    def test_equal_finishes_with_slower_lap_is_rejected(self):
        candidate = {"completed": 12, "denominator": 12, "median_finished_lap_ms": 25000,
                     "invalid_actions": 0}
        control = {"completed": 12, "denominator": 12, "median_finished_lap_ms": 24000,
                   "invalid_actions": 0}
        self.assertEqual(decide(candidate, control), "REJECT")

    def test_invalid_actions_are_rejected(self):
        candidate = {"completed": 12, "denominator": 12, "median_finished_lap_ms": 22000,
                     "invalid_actions": 1}
        control = {"completed": 11, "denominator": 12, "median_finished_lap_ms": 24000,
                   "invalid_actions": 0}
        self.assertEqual(decide(candidate, control), "REJECT")


if __name__ == "__main__":
    unittest.main()
