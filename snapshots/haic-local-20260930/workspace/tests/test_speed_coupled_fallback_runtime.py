import unittest
from unittest.mock import patch

import numpy as np

from haic_agent.speed_coupled_fallback_runtime import SpeedCoupledFallbackAgent


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
        self.last_action = np.asarray([0.25, 0.5, 0.0], dtype=np.float32)
        return self.last_action

    def last_step_diagnostics(self):
        return {"obstacle_y": None, "obstacle_steer_side": None,
                "obstacle_x": None, "road_center_at_obstacle": None,
                "pixel_speed": 40.0}


class SpeedCoupledFallbackTests(unittest.TestCase):
    def test_returns_coupled_action_only_with_full_visible_road(self):
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        agent = SpeedCoupledFallbackAgent(_Base(), corridor=_Corridor())
        agent.reset(obs)
        full = {row: 42.0 for row in (54, 50, 46, 42, 38, 34, 30)}
        with patch("haic_agent.speed_coupled_fallback_runtime.road_centers", return_value=full):
            self.assertAlmostEqual(float(agent.act(obs)[1]), 0.5)
        with patch("haic_agent.speed_coupled_fallback_runtime.road_centers", return_value={}):
            self.assertAlmostEqual(float(agent.act(obs)[1]), 0.08)
        self.assertEqual((agent.coupled_count, agent.fallback_count), (1, 1))


if __name__ == "__main__":
    unittest.main()
