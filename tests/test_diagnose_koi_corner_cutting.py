import unittest

import numpy as np

from scripts.diagnose_koi_corner_cutting import longest_run, measure, on_road


class CornerCuttingGeometryTests(unittest.TestCase):
    def test_polygon_membership_includes_edge(self):
        road = np.array([[[-2, 0], [2, 0], [2, 10], [-2, 10]]])
        self.assertEqual(on_road(np.array([[0, 5], [2, 5], [3, 5], [0, 11]]), road).tolist(),
                         [True, True, False, False])

    def test_disjoint_offroad_runs_not_summed_as_longest(self):
        self.assertAlmostEqual(longest_run([True, True, False, True], [.1, .2, .1, .2]), .3)

    def test_straight_path_has_zero_curvature_and_finite_time(self):
        track = [[0, 0, 0, 0], [0, 0, 0, 100], [0, 0, 100, 100], [0, 0, 100, 0]]
        road = np.array([[[-7, 0], [7, 0], [7, 100], [-7, 100]]])
        points = np.column_stack((np.zeros(101), np.linspace(10, 60, 101)))
        result = measure(points, np.full(101, 50.), track, road, [])
        self.assertAlmostEqual(result["path_length"], 50)
        self.assertAlmostEqual(result["estimated_duration"], 1)
        self.assertAlmostEqual(result["max_required_wheel_angle"], 0)
        self.assertEqual(result["estimated_longest_any_wheel_offroad_seconds"], 0)
        self.assertTrue(result["reentered"])

    def test_zero_shift_calibration_recovers_observed_wheel_angle(self):
        track = [[0, 0, 0, 0], [0, 0, 0, 100], [0, 0, 100, 100], [0, 0, 100, 0]]
        road = np.array([[[-7, 0], [7, 0], [7, 100], [-7, 100]]])
        points = np.column_stack((np.zeros(101), np.linspace(10, 60, 101)))
        speed = np.full(101, 50.)
        original = measure(points, speed, track, road, [])
        calibrated = measure(points, speed, track, road, [], original["kinematic_steering"], np.full(101, .2))
        self.assertAlmostEqual(calibrated["max_required_wheel_angle"], .2)
        self.assertAlmostEqual(calibrated["max_wheel_angle_rate"], 0.)


if __name__ == "__main__":
    unittest.main()
