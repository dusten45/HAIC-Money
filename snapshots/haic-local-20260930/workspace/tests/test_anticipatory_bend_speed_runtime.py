"""Steering-aware speed gate preserves road and hazard overrides."""

import unittest
from unittest.mock import patch

import numpy as np

from haic_agent.anticipatory_bend_speed_runtime import AnticipatoryBendSpeedAgent


class _Corridor:
    def __init__(self, obstacle=None):
        self.obstacle = obstacle

    def last_step_diagnostics(self):
        return {"obstacle_y": self.obstacle}


class _Inner:
    def __init__(self, obstacle=None):
        self.corridor = _Corridor(obstacle)


class _Base:
    def __init__(self, steer=0.04, gas=0.08, brake=0.0, obstacle=None, preview=True):
        self.action = np.asarray([steer, gas, brake], dtype=np.float32)
        self.base = _Inner(obstacle)
        self.preview_count = 0
        self.preview = preview

    def reset(self, observation):
        self.preview_count = 0

    def act(self, observation):
        if self.preview:
            self.preview_count += 1
        return self.action.copy()


class AnticipatoryBendSpeedTests(unittest.TestCase):
    def setUp(self):
        self.observation = np.zeros((4, 84, 84), dtype=np.float32)
        self.centers = {54: 42.0, 50: 42.5, 46: 43.0, 42: 43.5, 38: 44.0, 34: 44.5, 30: 45.0}
        self.road = patch("haic_agent.anticipatory_bend_speed_runtime.road_centers", return_value=self.centers)
        self.speed = patch("haic_agent.anticipatory_bend_speed_runtime.estimate_observation_speed", return_value=30.0)
        self.road.start()
        self.speed.start()
        self.addCleanup(self.road.stop)
        self.addCleanup(self.speed.stop)

    def test_boosts_only_gas_with_safe_preview(self):
        agent = AnticipatoryBendSpeedAgent(_Base())
        action = agent.act(self.observation)
        np.testing.assert_allclose(action, [0.04, 0.16, 0.0], atol=1e-6)
        self.assertEqual((agent.boost_count, agent.boost_during_preview_count), (1, 1))

    def test_obstacle_blocks_boost(self):
        agent = AnticipatoryBendSpeedAgent(_Base(obstacle=53.0))
        self.assertAlmostEqual(float(agent.act(self.observation)[1]), 0.08)

    def test_high_speed_blocks_boost(self):
        with patch("haic_agent.anticipatory_bend_speed_runtime.estimate_observation_speed", return_value=45.0):
            agent = AnticipatoryBendSpeedAgent(_Base())
            self.assertAlmostEqual(float(agent.act(self.observation)[1]), 0.08)

    def test_large_steer_blocks_boost(self):
        agent = AnticipatoryBendSpeedAgent(_Base(steer=0.11))
        self.assertAlmostEqual(float(agent.act(self.observation)[1]), 0.08)

    def test_braking_blocks_boost(self):
        agent = AnticipatoryBendSpeedAgent(_Base(brake=0.1))
        self.assertAlmostEqual(float(agent.act(self.observation)[1]), 0.08)

    def test_off_center_road_blocks_boost(self):
        self.centers[54] = 46.0
        agent = AnticipatoryBendSpeedAgent(_Base())
        self.assertAlmostEqual(float(agent.act(self.observation)[1]), 0.08)

    def test_strong_bend_blocks_boost(self):
        self.centers[30] = 51.0
        agent = AnticipatoryBendSpeedAgent(_Base())
        self.assertAlmostEqual(float(agent.act(self.observation)[1]), 0.08)

    def test_reset_clears_counts(self):
        agent = AnticipatoryBendSpeedAgent(_Base())
        agent.act(self.observation)
        agent.reset(self.observation)
        self.assertEqual((agent.boost_count, agent.boost_during_preview_count), (0, 0))


if __name__ == "__main__":
    unittest.main()
