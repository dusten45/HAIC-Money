import unittest
from unittest.mock import patch

import numpy as np

from haic_agent.road_margin_throttle_runtime import RoadMarginThrottleAgent
from haic_agent.road_margin_speed_cap_runtime import RoadMarginSpeedCapAgent


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


class RoadMarginThrottleTests(unittest.TestCase):
    def test_full_throttle_with_road_offset_and_small_grass_margin(self):
        corridor = _Corridor()
        agent = RoadMarginThrottleAgent(_Base(), corridor=corridor)
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        obs[:, 30:55, 37:54] = 0.35
        agent.reset(obs)
        centers = {row: 46.0 for row in (54, 50, 46, 42, 38, 34, 30)}
        with patch("haic_agent.road_margin_throttle_runtime.road_centers", return_value=centers):
            self.assertEqual(float(agent.act(obs)[1]), 1.0)
            self.assertEqual(agent.margin_throttle_count, 1)
            corridor.y = 32.0
            self.assertLess(float(agent.act(obs)[1]), 1.0)
            corridor.y = None
            corridor.speed = 70.0
            self.assertLess(float(agent.act(obs)[1]), 1.0)

    def test_no_boost_when_road_unseen(self):
        agent = RoadMarginThrottleAgent(_Base(), corridor=_Corridor())
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        agent.reset(obs)
        self.assertLess(float(agent.act(obs)[1]), 1.0)

    def test_speed_cap_stops_boost_before_high_speed(self):
        corridor = _Corridor()
        agent = RoadMarginSpeedCapAgent(_Base(), corridor=corridor)
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        obs[:, 30:55, 37:54] = 0.35
        agent.reset(obs)
        centers = {row: 46.0 for row in (54, 50, 46, 42, 38, 34, 30)}
        with patch("haic_agent.road_margin_throttle_runtime.road_centers", return_value=centers):
            self.assertEqual(float(agent.act(obs)[1]), 1.0)
            corridor.speed = 45.0
            self.assertLess(float(agent.act(obs)[1]), 1.0)


if __name__ == "__main__":
    unittest.main()
