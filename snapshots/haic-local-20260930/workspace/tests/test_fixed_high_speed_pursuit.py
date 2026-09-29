import unittest
from unittest.mock import patch
import numpy as np

from haic_agent.fixed_high_speed_runtime import fixed_pedals
from haic_agent.fixed_high_speed_pursuit_repair import FixedHighSpeedPursuitRepair


class PursuitTests(unittest.TestCase):
    def test_saturated_base_does_not_leave_road_residual(self):
        driver = FixedHighSpeedPursuitRepair("pursuit")
        driver.steps = 10
        driver.base.act = lambda obs: np.array([.7, 0, .28], dtype=np.float32)
        centers = {54: 42., 42: 70., 30: 60.}
        with patch("haic_agent.fixed_high_speed_runtime.road_centers", return_value=centers):
            action = driver.act(np.zeros((4, 84, 84), dtype=np.float32))
        self.assertAlmostEqual(float(action[0]), .65 * (2 * 24 * 18 / (18**2 + 24**2)), places=6)
        self.assertEqual(float(action[2]), 0.)

    def test_fixed_governor_does_not_brake_below_target(self):
        for speed in (0, 30, 54, 60):
            self.assertEqual(fixed_pedals(speed)[1], 0.)
        self.assertGreater(fixed_pedals(70)[1], 0.)


if __name__ == "__main__":
    unittest.main()
