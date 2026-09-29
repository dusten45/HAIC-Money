import unittest
from unittest.mock import patch

import numpy as np

from haic_agent.bend_safety_runtime import BendSafetyAgent


class _Base:
    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([0.1, 0.08, 0.0], dtype=np.float32)


class _Corridor:
    OBSTACLE_STEER = 0.34
    MAX_STEER = 0.7
    obstacle_steer_scale = 1.0

    def reset(self, observation):
        self.last_action = np.zeros(3, dtype=np.float32)

    def act(self, observation):
        self.last_action = np.asarray([0.25, 0.5, 0.04], dtype=np.float32)
        return self.last_action

    def last_step_diagnostics(self):
        return {"obstacle_y": None, "obstacle_steer_side": None,
                "obstacle_x": None, "road_center_at_obstacle": None,
                "pixel_speed": 40.0}


class BendSafetyTests(unittest.TestCase):
    def test_straight_uses_coupled_action_and_bend_keeps_trained_steer(self):
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        obs[:, 30:55, 34:51] = 0.35
        agent = BendSafetyAgent(_Base(), corridor=_Corridor())
        agent.reset(obs)
        straight = {row: 42.0 for row in (54, 50, 46, 42, 38, 34, 30)}
        bent = {54: 42.0, 50: 43.0, 46: 44.0, 42: 45.0,
                38: 47.0, 34: 49.0, 30: 51.0}
        with patch("haic_agent.bend_safety_runtime.road_centers", return_value=straight):
            self.assertAlmostEqual(float(agent.act(obs)[0]), 0.25)
        with patch("haic_agent.bend_safety_runtime.road_centers", return_value=bent):
            action = agent.act(obs)
            self.assertAlmostEqual(float(action[0]), 0.1)
            self.assertAlmostEqual(float(action[1]), 0.5)
        self.assertEqual((agent.straight_count, agent.bend_count), (1, 1))


if __name__ == "__main__":
    unittest.main()
