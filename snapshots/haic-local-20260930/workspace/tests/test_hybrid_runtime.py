import unittest

import numpy as np

from haic_agent.hybrid_runtime import ObstacleHybridAgent


class _Base:
    def __init__(self):
        self.reset_count = 0

    def reset(self, observation):
        self.reset_count += 1

    def act(self, observation):
        return np.array([0.1, 0.2, 0.0], dtype=np.float32)


class _Corridor:
    def __init__(self):
        self.y = None
        self.reset_count = 0

    def reset(self, observation):
        self.reset_count += 1

    def act(self, observation):
        return np.array([-0.3, 0.0, 0.1], dtype=np.float32)

    def last_step_diagnostics(self):
        return {"obstacle_y": self.y}


class HybridRuntimeTests(unittest.TestCase):
    def test_uses_corridor_only_when_obstacle_is_near_enough(self):
        base, corridor = _Base(), _Corridor()
        agent = ObstacleHybridAgent(base, corridor=corridor)
        observation = np.zeros((4, 84, 84), dtype=np.float32)
        agent.reset(observation)
        corridor.y = 27.9
        np.testing.assert_allclose(agent.act(observation), [0.1, 0.2, 0.0], atol=1e-7)
        corridor.y = 28.0
        np.testing.assert_allclose(agent.act(observation), [-0.3, 0.0, 0.1], atol=1e-7)
        self.assertEqual(agent.override_count, 1)
        agent.reset(observation)
        self.assertEqual(agent.override_count, 0)
        self.assertEqual((base.reset_count, corridor.reset_count), (2, 2))


if __name__ == "__main__":
    unittest.main()
