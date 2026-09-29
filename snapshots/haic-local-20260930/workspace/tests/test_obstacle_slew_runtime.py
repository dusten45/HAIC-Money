import unittest

import numpy as np

from haic_agent.obstacle_slew_runtime import ObstacleSlewAgent


class _Base:
    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([0.1, 0.1, 0.0], dtype=np.float32)


class _Corridor:
    def __init__(self):
        self.y = None
        self.steer = 0.0

    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([self.steer, 0.05, 0.0], dtype=np.float32)

    def last_step_diagnostics(self):
        return {"obstacle_y": self.y}


class ObstacleSlewTests(unittest.TestCase):
    def test_clamps_abrupt_obstacle_steering_only(self):
        corridor = _Corridor()
        agent = ObstacleSlewAgent(_Base(), corridor=corridor)
        observation = np.zeros((4, 84, 84), dtype=np.float32)
        agent.reset(observation)
        agent.act(observation)
        corridor.y, corridor.steer = 35.0, -0.7
        first = agent.act(observation)
        self.assertAlmostEqual(float(first[0]), -0.1, places=6)
        second = agent.act(observation)
        self.assertAlmostEqual(float(second[0]), -0.3, places=6)
        self.assertEqual(agent.slew_count, 2)
        corridor.y = None
        last = agent.act(observation)
        self.assertAlmostEqual(float(last[0]), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
