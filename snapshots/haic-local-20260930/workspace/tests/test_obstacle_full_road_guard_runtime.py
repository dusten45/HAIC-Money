import unittest
from unittest.mock import patch

import numpy as np

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent


class _Base:
    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([0.0, 0.1, 0.0], dtype=np.float32)


class _Corridor:
    OBSTACLE_STEER = 0.34
    MAX_STEER = 0.7
    obstacle_steer_scale = 1.0

    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([0.26, 0.0, 0.05], dtype=np.float32)

    def last_step_diagnostics(self):
        return {"obstacle_y": 59.0, "obstacle_steer_side": 1.0,
                "obstacle_x": 36.0, "road_center_at_obstacle": 42.0,
                "pixel_speed": 40.0}


class FullRoadGuardTests(unittest.TestCase):
    def test_requires_all_road_rows_for_close_obstacle_steering(self):
        agent = ObstacleFullRoadGuardAgent(_Base(), corridor=_Corridor())
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        agent.reset(obs)
        with patch("haic_agent.obstacle_full_road_guard_runtime.road_centers",
                   return_value={54: 42.0, 50: 42.0, 46: 41.5,
                                 42: 40.0, 38: 38.5, 34: 38.5}):
            action = agent.act(obs)
        self.assertLess(float(action[0]), 0.0)
        self.assertEqual(agent.full_road_guard_count, 1)


if __name__ == "__main__":
    unittest.main()
