import unittest

import numpy as np

from haic_agent.obstacle_road_guard_runtime import ObstacleRoadGuardAgent


class _Base:
    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([0.05, 0.1, 0.0], dtype=np.float32)


class _Corridor:
    OBSTACLE_STEER = 0.34
    MAX_STEER = 0.7
    obstacle_steer_scale = 1.0

    def __init__(self):
        self.y = 59.0
        self.side = 1.0

    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([0.22, 0.0, 0.05], dtype=np.float32)

    def last_step_diagnostics(self):
        return {"obstacle_y": self.y, "obstacle_steer_side": self.side}


class RoadGuardTests(unittest.TestCase):
    def test_suppresses_near_field_avoidance_when_road_is_poorly_seen(self):
        agent = ObstacleRoadGuardAgent(_Base(), corridor=_Corridor())
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        agent.reset(obs)
        action = agent.act(obs)
        self.assertAlmostEqual(float(action[0]), 0.0, places=5)
        self.assertEqual(agent.road_guard_count, 1)


if __name__ == "__main__":
    unittest.main()
