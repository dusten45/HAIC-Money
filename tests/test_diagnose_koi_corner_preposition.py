import unittest

import numpy as np

from scripts.diagnose_koi_corner_preposition import analytic_demand, clamped_spline


class PrepositionSplineTests(unittest.TestCase):
    def test_straight_clamped_curve_preserves_positions_and_derivatives(self):
        knots = np.array([0., 10., 25., 50.])
        values = np.column_stack((np.zeros(4), knots))
        points, d1, d2, d3 = clamped_spline(knots, values, [0, 1], [0, 1], knots)
        np.testing.assert_allclose(points, values)
        np.testing.assert_allclose(d1, np.tile([0, 1], (4, 1)))
        np.testing.assert_allclose(d2, 0, atol=1e-12)
        k, angle, rate = analytic_demand(d1, d2, d3, np.full(4, 50.))
        np.testing.assert_allclose(k, 0, atol=1e-12)
        np.testing.assert_allclose(angle, 0, atol=1e-12)
        np.testing.assert_allclose(rate, 0, atol=1e-12)

    def test_c2_continuity_at_interior_knot(self):
        knots = np.array([0., 10., 25., 50.])
        values = np.array([[0, 0], [3, 10], [-4, 25], [0, 50]])
        samples = np.array([10 - 1e-7, 10 + 1e-7])
        p, d1, d2, _ = clamped_spline(knots, values, [0, 1], [0, 1], samples)
        for data in (p, d1, d2):
            np.testing.assert_allclose(data[0], data[1], atol=1e-6)

    def test_rejects_nonmonotone_control_stations(self):
        with self.assertRaises(ValueError):
            clamped_spline([0, 5, 4], [[0, 0], [1, 5], [0, 4]], [0, 1], [0, 1], [0, 1])

    def test_endpoint_second_derivatives_preserved(self):
        knots = np.array([0., 10., 25., 50.])
        values = np.array([[0, 0], [3, 10], [-4, 25], [0, 50]])
        p, d1, d2, _ = clamped_spline(knots, values, [0, 1], [0, 1], knots,
                                     [.02, 0], [-.03, 0])
        np.testing.assert_allclose(p, values, atol=1e-9)
        np.testing.assert_allclose(d1[[0, -1]], [[0, 1], [0, 1]], atol=1e-9)
        np.testing.assert_allclose(d2[[0, -1]], [[.02, 0], [-.03, 0]], atol=1e-9)

    def test_analytic_circle_has_constant_curvature_and_zero_steering_rate(self):
        theta = np.linspace(0, 1, 11)
        d1 = 20 * np.column_stack((-np.sin(theta), np.cos(theta)))
        d2 = -20 * np.column_stack((np.cos(theta), np.sin(theta)))
        d3 = -d1
        k, angle, rate = analytic_demand(d1, d2, d3, np.full(11, 40.))
        np.testing.assert_allclose(k, .05, atol=1e-12)
        np.testing.assert_allclose(angle, np.arctan(3.24 * .05), atol=1e-12)
        np.testing.assert_allclose(rate, 0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
