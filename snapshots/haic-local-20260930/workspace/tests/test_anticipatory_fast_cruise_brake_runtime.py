"""Fast cruise and advance braking obey the pixel-road safety gates."""

import unittest
from unittest.mock import patch

import numpy as np

from haic_agent.anticipatory_fast_cruise_brake_runtime import AnticipatoryFastCruiseBrakeAgent


class _Corridor:
    def __init__(self):
        self.obstacle = None

    def last_step_diagnostics(self):
        return {"obstacle_y": self.obstacle}


class _Base:
    def __init__(self):
        self.base = type("Inner", (), {"corridor": _Corridor()})()
        self.action = np.asarray([0.02, 0.08, 0.0], dtype=np.float32)

    def reset(self, observation):
        pass

    def act(self, observation):
        return self.action.copy()


class FastCruiseBrakeTests(unittest.TestCase):
    def setUp(self):
        self.base = _Base()
        self.agent = AnticipatoryFastCruiseBrakeAgent(self.base)
        self.obs = np.zeros((4, 84, 84), dtype=np.float32)
        self.centers = {row: 42.0 for row in (54, 50, 46, 42, 38, 34, 30)}
        self.road = patch("haic_agent.anticipatory_fast_cruise_brake_runtime.road_centers", return_value=self.centers)
        self.speed = patch("haic_agent.anticipatory_fast_cruise_brake_runtime.estimate_observation_speed", return_value=35.0)
        self.road.start()
        self.speed.start()
        self.addCleanup(self.road.stop)
        self.addCleanup(self.speed.stop)

    def test_boost_only_after_stable_straight(self):
        for _ in range(4):
            self.assertAlmostEqual(float(self.agent.act(self.obs)[1]), 0.08, places=5)
        np.testing.assert_allclose(self.agent.act(self.obs), [0.02, 0.22, 0], atol=1e-6)
        self.assertEqual(self.agent.boost_count, 1)

    def test_far_bend_brakes_before_near_road_moves(self):
        self.centers[30] = 48.0
        with patch("haic_agent.anticipatory_fast_cruise_brake_runtime.estimate_observation_speed", return_value=47.0):
            np.testing.assert_allclose(self.agent.act(self.obs), [0.02, 0, 0.12], atol=1e-6)
        self.assertEqual(self.agent.preview_brake_count, 1)

    def test_sharp_bend_uses_stronger_brake(self):
        self.centers[30] = 50.0
        with patch("haic_agent.anticipatory_fast_cruise_brake_runtime.estimate_observation_speed", return_value=47.0):
            self.assertAlmostEqual(float(self.agent.act(self.obs)[2]), 0.18, places=5)

    def test_off_center_near_road_cuts_gas(self):
        self.centers[54] = 46.0
        with patch("haic_agent.anticipatory_fast_cruise_brake_runtime.estimate_observation_speed", return_value=45.0):
            action = self.agent.act(self.obs)
        self.assertEqual(float(action[1]), 0.0)
        self.assertAlmostEqual(float(action[2]), 0.12, places=5)

    def test_obstacle_preserves_base_action(self):
        self.base.base.corridor.obstacle = 55.0
        self.centers[30] = 48.0
        with patch("haic_agent.anticipatory_fast_cruise_brake_runtime.estimate_observation_speed", return_value=47.0):
            np.testing.assert_allclose(self.agent.act(self.obs), self.base.action)

    def test_incomplete_road_preserves_base_action(self):
        self.centers.pop(30)
        np.testing.assert_allclose(self.agent.act(self.obs), self.base.action)


if __name__ == "__main__":
    unittest.main()
