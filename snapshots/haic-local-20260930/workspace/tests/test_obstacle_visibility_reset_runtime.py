import unittest
from unittest.mock import patch

import numpy as np

from haic_agent.obstacle_visibility_reset_runtime import ObstacleVisibilityResetAgent


class _Base:
    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([0.0, 0.1, 0.0], dtype=np.float32)


class _Corridor:
    OBSTACLE_STEER = 0.34
    MAX_STEER = 0.7
    obstacle_steer_scale = 1.0

    def __init__(self):
        self.y = 33.0
        self.side = 1.0

    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([self.side * 0.34, 0.0, 0.05], dtype=np.float32)

    def last_step_diagnostics(self):
        return {"obstacle_y": self.y, "obstacle_steer_side": self.side,
                "obstacle_x": 42.0, "road_center_at_obstacle": 42.0,
                "pixel_speed": 40.0}


class VisibilityResetTests(unittest.TestCase):
    def test_releases_old_side_when_close_obstacle_and_road_is_occluded(self):
        corridor = _Corridor()
        agent = ObstacleVisibilityResetAgent(_Base(), corridor=corridor)
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        agent.reset(obs)
        with patch("haic_agent.obstacle_visibility_reset_runtime.road_centers",
                   return_value={54: 42.0, 50: 42.0, 46: 42.0, 42: 42.0, 38: 42.0}):
            agent.act(obs)
            corridor.y = 59.0
            agent.act(obs)
        self.assertIsNone(agent._committed_side)
        self.assertEqual(agent.visibility_reset_count, 1)


if __name__ == "__main__":
    unittest.main()
