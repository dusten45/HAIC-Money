import unittest
from unittest.mock import patch

import numpy as np

from haic_agent.straight_sprint_runtime import StraightSprintAgent


class _Base:
    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([0.0, 0.08, 0.0], dtype=np.float32)


class _Corridor:
    OBSTACLE_STEER = 0.34
    MAX_STEER = 0.7
    obstacle_steer_scale = 1.0

    def __init__(self):
        self.y = None
        self.speed = 40.0

    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([0.0, 0.05, 0.0], dtype=np.float32)

    def last_step_diagnostics(self):
        return {"obstacle_y": self.y, "obstacle_steer_side": None,
                "obstacle_x": None, "road_center_at_obstacle": None,
                "pixel_speed": self.speed}


class StraightSprintTests(unittest.TestCase):
    def test_full_throttle_after_clear_straight_streak(self):
        corridor = _Corridor()
        agent = StraightSprintAgent(_Base(), corridor=corridor)
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        agent.reset(obs)
        centers = {row: 42.0 for row in (54, 50, 46, 42, 38, 34, 30)}
        with patch("haic_agent.straight_sprint_runtime.road_centers", return_value=centers):
            for _ in range(7):
                self.assertLess(float(agent.act(obs)[1]), 1.0)
            self.assertEqual(float(agent.act(obs)[1]), 1.0)
            corridor.speed = 60.0
            self.assertLess(float(agent.act(obs)[1]), 1.0)
            corridor.speed = 40.0
            corridor.y = 30.0
            self.assertLess(float(agent.act(obs)[1]), 1.0)
        self.assertEqual(agent.sprint_count, 1)


if __name__ == "__main__":
    unittest.main()
