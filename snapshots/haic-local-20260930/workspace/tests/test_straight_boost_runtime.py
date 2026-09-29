import unittest
from unittest.mock import patch

import numpy as np

from haic_agent.straight_boost_runtime import StraightBoostAgent


class _Base:
    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([0.0, 0.08, 0.0], dtype=np.float32)


class _Corridor:
    OBSTACLE_STEER = 0.34
    MAX_STEER = 0.7
    obstacle_steer_scale = 1.0

    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([0.0, 0.05, 0.0], dtype=np.float32)

    def last_step_diagnostics(self):
        return {"obstacle_y": None, "obstacle_steer_side": None,
                "obstacle_x": None, "road_center_at_obstacle": None,
                "pixel_speed": 40.0}


class StraightBoostTests(unittest.TestCase):
    def test_caps_straight_acceleration_below_full_throttle(self):
        agent = StraightBoostAgent(_Base(), corridor=_Corridor())
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        agent.reset(obs)
        centers = {row: 42.0 for row in (54, 50, 46, 42, 38, 34, 30)}
        with patch("haic_agent.straight_sprint_runtime.road_centers", return_value=centers):
            actions = [agent.act(obs) for _ in range(8)]
        self.assertAlmostEqual(float(actions[-1][1]), 0.4, places=5)
        self.assertEqual(agent.sprint_count, 1)


if __name__ == "__main__":
    unittest.main()
