import unittest

import numpy as np

from haic_agent.obstacle_guard_brake_runtime import ObstacleGuardBrakeAgent


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
        return np.asarray([0.2, 0.05, 0.0], dtype=np.float32)

    def last_step_diagnostics(self):
        return {"obstacle_y": 31.0, "obstacle_steer_side": 1.0,
                "obstacle_x": 42.0, "road_center_at_obstacle": 42.0,
                "pixel_speed": 41.0}


class GuardBrakeTests(unittest.TestCase):
    def test_brakes_for_visible_near_obstacle_at_speed(self):
        agent = ObstacleGuardBrakeAgent(_Base(), corridor=_Corridor())
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        agent.reset(obs)
        action = agent.act(obs)
        self.assertEqual(float(action[1]), 0.0)
        self.assertAlmostEqual(float(action[2]), 0.28, places=5)
        self.assertEqual(agent.guard_brake_count, 1)


if __name__ == "__main__":
    unittest.main()
