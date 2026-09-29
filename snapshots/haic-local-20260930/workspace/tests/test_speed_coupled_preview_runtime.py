"""Speed-coupled preview preserves action shape and refuses opposite steering."""

import unittest
from unittest.mock import patch

import numpy as np

from haic_agent.speed_coupled_preview_runtime import SpeedCoupledPreviewAgent


class _Corridor:
    def __init__(self):
        self.obstacle = None

    def last_step_diagnostics(self):
        return {"obstacle_y": self.obstacle}


class _Base:
    def __init__(self, steer=0.0):
        self.base = type("Inner", (), {"corridor": _Corridor()})()
        self.action = np.asarray([steer, 0.07, 0.0], dtype=np.float32)

    def reset(self, observation):
        pass

    def act(self, observation):
        return self.action.copy()


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.obs = np.zeros((4, 84, 84), dtype=np.float32)
        self.current = {row: 42.0 for row in (54, 50, 46, 42, 38, 34, 30)}
        self.previous = dict(self.current)
        self.road = patch("haic_agent.speed_coupled_preview_runtime.road_centers", side_effect=lambda frame: self.current if np.shares_memory(frame, self.obs[-1]) else self.previous)
        self.speed = patch("haic_agent.speed_coupled_preview_runtime.estimate_observation_speed", return_value=40.0)
        self.road.start()
        self.speed.start()
        self.addCleanup(self.road.stop)
        self.addCleanup(self.speed.stop)

    def test_stable_straight_boosts_gas(self):
        agent = SpeedCoupledPreviewAgent(_Base())
        for _ in range(4):
            self.assertAlmostEqual(float(agent.act(self.obs)[1]), 0.07, places=5)
        self.assertAlmostEqual(float(agent.act(self.obs)[1]), 0.16, places=5)
        self.assertEqual(agent.boost_count, 1)

    def test_confirmed_far_bend_steers_earlier(self):
        self.current[30] = 47.0
        self.previous[30] = 46.0
        agent = SpeedCoupledPreviewAgent(_Base())
        action = agent.act(self.obs)
        self.assertGreater(float(action[0]), 0.05)
        self.assertEqual(agent.early_steer_count, 1)

    def test_never_reverses_strong_base_steer(self):
        self.current[30] = 47.0
        self.previous[30] = 46.0
        agent = SpeedCoupledPreviewAgent(_Base(steer=-0.12))
        self.assertAlmostEqual(float(agent.act(self.obs)[0]), -0.12, places=5)

    def test_close_obstacle_slows_without_steering_override(self):
        base = _Base(steer=0.1)
        base.base.corridor.obstacle = 52.0
        agent = SpeedCoupledPreviewAgent(base)
        with patch("haic_agent.speed_coupled_preview_runtime.estimate_observation_speed", return_value=47.0):
            action = agent.act(self.obs)
        self.assertAlmostEqual(float(action[0]), 0.1, places=5)
        self.assertEqual(float(action[1]), 0.0)
        self.assertAlmostEqual(float(action[2]), 0.08, places=5)


if __name__ == "__main__":
    unittest.main()
