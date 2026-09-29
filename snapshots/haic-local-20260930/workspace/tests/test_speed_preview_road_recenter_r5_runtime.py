"""Focused action-contract tests for pixel-road recentering."""

import unittest
from unittest.mock import patch

import numpy as np

from haic_agent.speed_preview_road_recenter_r5_runtime import SpeedPreviewRoadRecenterAgent


class _Corridor:
    obstacle = None
    obstacle_x = None
    road_center = None

    def last_step_diagnostics(self):
        return {"obstacle_y": self.obstacle, "obstacle_x": self.obstacle_x,
                "road_center_at_obstacle": self.road_center}


class _Base:
    def __init__(self):
        self.base = type("Preview", (), {"base": type("Guard", (), {"corridor": _Corridor()})()})()
        self.action = np.array([0.12, 0.16, 0.08], dtype=np.float32)

    def reset(self, observation):
        pass

    def act(self, observation):
        return self.action.copy()


class RecenterTests(unittest.TestCase):
    def setUp(self):
        self.obs = np.zeros((4, 84, 84), dtype=np.float32)
        self.centers = {row: 42.0 for row in (54, 50, 46, 42, 38, 34, 30)}
        road = patch("haic_agent.speed_preview_road_recenter_r5_runtime.road_centers", side_effect=lambda _: self.centers)
        road.start()
        self.addCleanup(road.stop)
        self.agent = SpeedPreviewRoadRecenterAgent(_Base())

    def test_clear_road_preserves_acceleration(self):
        action = self.agent.act(self.obs)
        self.assertAlmostEqual(float(action[1]), 0.16, places=5)
        self.assertEqual(self.agent.recenter_count, 0)

    def test_partial_left_road_reverses_wrong_steer_and_caps_pedals(self):
        self.centers = {54: 30.0, 50: 31.0, 46: 32.0}
        action = self.agent.act(self.obs)
        self.assertLess(float(action[0]), -0.25)
        self.assertLessEqual(float(action[1]), 0.06)
        self.assertLessEqual(float(action[2]), 0.02)
        self.assertEqual(self.agent.recenter_count, 1)

    def test_missing_road_holds_direction_then_expires(self):
        self.centers = {54: 36.0}
        self.agent.act(self.obs)
        self.centers = {}
        held = self.agent.act(self.obs)
        self.assertAlmostEqual(float(held[0]), -0.25, places=5)
        self.assertEqual(float(held[1]), 0.0)
        self.assertAlmostEqual(float(held[2]), 0.08, places=5)
        for _ in range(29):
            self.agent.act(self.obs)
        self.assertAlmostEqual(float(self.agent.act(self.obs)[0]), 0.12, places=5)

    def test_six_pixel_offset_starts_return(self):
        self.centers = {54: 36.0, 50: 37.0}
        self.assertLess(float(self.agent.act(self.obs)[0]), -0.10)
        self.assertEqual(self.agent.recenter_count, 1)

    def test_large_last_offset_keeps_recovery_momentum(self):
        self.centers = {54: 29.5}
        self.agent.act(self.obs)
        self.centers = {}
        held = self.agent.act(self.obs)
        self.assertAlmostEqual(float(held[1]), 0.06, places=5)
        self.assertAlmostEqual(float(held[2]), 0.02, places=5)

    def test_obstacle_preserves_base_action(self):
        self.centers = {54: 30.0}
        self.agent.base.base.base.corridor.obstacle = 50.0
        action = self.agent.act(self.obs)
        self.assertAlmostEqual(float(action[0]), 0.12, places=5)
        self.assertEqual(self.agent.recenter_count, 0)

    def test_obstacle_on_same_road_edge_keeps_car_inside_road(self):
        self.centers = {54: 33.5, 50: 33.0, 46: 32.0}
        corridor = self.agent.base.base.base.corridor
        corridor.obstacle = 55.0
        corridor.obstacle_x = 21.0
        corridor.road_center = 32.0
        action = self.agent.act(self.obs)
        self.assertLess(float(action[0]), 0.0)
        self.assertAlmostEqual(float(action[1]), 0.05, places=5)
        self.assertGreaterEqual(float(action[2]), 0.059)
        self.assertEqual(self.agent.obstacle_road_count, 1)

    def test_centered_road_clears_saved_direction(self):
        self.centers = {54: 30.0}
        self.agent.act(self.obs)
        self.centers = {row: 42.0 for row in (54, 50, 46, 42, 38, 34, 30)}
        self.agent.act(self.obs)
        self.assertEqual(self.agent.return_direction, 0.0)


if __name__ == "__main__":
    unittest.main()
