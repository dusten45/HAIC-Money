"""Stable-exit speed only changes gas after sustained safe observations."""

import unittest
from unittest.mock import patch

import numpy as np

from haic_agent.anticipatory_bend_speed_safe_runtime import AnticipatoryBendSafeSpeedAgent


class _Corridor:
    def __init__(self):
        self.obstacle = None

    def last_step_diagnostics(self):
        return {"obstacle_y": self.obstacle}


class _Base:
    def __init__(self):
        self.base = type("Inner", (), {"corridor": _Corridor()})()
        self.action = np.asarray([0.02, 0.07, 0.0], dtype=np.float32)

    def reset(self, observation):
        pass

    def act(self, observation):
        return self.action.copy()


class StableSpeedTests(unittest.TestCase):
    def test_requires_five_safe_decisions_and_keeps_steering(self):
        base = _Base()
        road = {54: 42.0, 50: 42.0, 46: 42.0, 42: 42.0, 38: 42.0, 34: 42.0, 30: 42.0}
        observation = np.zeros((4, 84, 84), dtype=np.float32)
        with (patch("haic_agent.anticipatory_bend_speed_safe_runtime.road_centers", return_value=road),
              patch("haic_agent.anticipatory_bend_speed_safe_runtime.estimate_observation_speed", return_value=35.0)):
            agent = AnticipatoryBendSafeSpeedAgent(base)
            for _ in range(4):
                self.assertAlmostEqual(float(agent.act(observation)[1]), 0.07, places=5)
            np.testing.assert_allclose(agent.act(observation), [0.02, 0.13, 0], atol=1e-6)
            self.assertEqual(agent.boost_count, 1)
            base.base.corridor.obstacle = 55.0
            self.assertAlmostEqual(float(agent.act(observation)[1]), 0.07, places=5)
            self.assertEqual(agent.stable_count, 0)

    def test_high_speed_and_reset_block_boost(self):
        base = _Base()
        road = {row: 42.0 for row in (54, 50, 46, 42, 38, 34, 30)}
        observation = np.zeros((4, 84, 84), dtype=np.float32)
        with (patch("haic_agent.anticipatory_bend_speed_safe_runtime.road_centers", return_value=road),
              patch("haic_agent.anticipatory_bend_speed_safe_runtime.estimate_observation_speed", return_value=50.0)):
            agent = AnticipatoryBendSafeSpeedAgent(base)
            for _ in range(6):
                self.assertAlmostEqual(float(agent.act(observation)[1]), 0.07, places=5)
            agent.reset(observation)
            self.assertEqual((agent.stable_count, agent.boost_count), (0, 0))


if __name__ == "__main__":
    unittest.main()
