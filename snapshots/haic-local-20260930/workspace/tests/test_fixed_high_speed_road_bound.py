import unittest
from unittest.mock import patch
import numpy as np
from haic_agent.fixed_high_speed_road_bound import FixedHighSpeedRoadBound


class RoadBoundTests(unittest.TestCase):
    def evaluate(self, obstacle_side):
        driver = FixedHighSpeedRoadBound()
        driver.steps = 20
        driver.base._last_obstacle = (45., 40., 47.)
        driver.base._obstacle_side = obstacle_side

        def inherited(self, obs):
            self.last = dict(road_centers={54:47.,42:50.,30:54.}, available=True,
                             correction=.1, inherited_steer=-.1)
            return np.array([0.,.3,0.], dtype=np.float32)

        with patch('haic_agent.fixed_high_speed_runtime.FixedHighSpeedAgent.act', inherited), \
             patch('haic_agent.fixed_high_speed_road_bound.road_centers', return_value={54:45.}):
            action = driver.act(np.zeros((4,84,84), dtype=np.float32))
        return driver, action

    def test_removes_only_outward_obstacle_component_preserving_pedals(self):
        driver, action = self.evaluate(-1)
        self.assertTrue(driver.last['road_bound_active'])
        self.assertAlmostEqual(float(action[0]), .022*8+.018*3+.1, places=6)
        self.assertAlmostEqual(float(action[1]), .3, places=6)
        self.assertEqual(float(action[2]), 0.)

    def test_inward_obstacle_component_is_unchanged(self):
        driver, action = self.evaluate(1)
        self.assertFalse(driver.last['road_bound_active'])
        self.assertEqual(float(action[0]), 0.)


if __name__ == '__main__':
    unittest.main()
