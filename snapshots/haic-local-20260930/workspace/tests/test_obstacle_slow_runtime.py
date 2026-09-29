import unittest

import numpy as np

from haic_agent.obstacle_slow_runtime import ObstacleSlowAgent


class _Base:
    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([0.1, 0.1, 0.0], dtype=np.float32)


class _Corridor:
    def __init__(self):
        self.y = None
        self.speed = 0.0

    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([-0.2, 0.05, 0.0], dtype=np.float32)

    def last_step_diagnostics(self):
        return {"obstacle_y": self.y, "pixel_speed": self.speed}


class ObstacleSlowTests(unittest.TestCase):
    def test_pixel_obstacle_triggers_early_braking_at_high_speed(self):
        corridor = _Corridor()
        agent = ObstacleSlowAgent(_Base(), corridor=corridor)
        observation = np.zeros((4, 84, 84), dtype=np.float32)
        agent.reset(observation)
        corridor.y, corridor.speed = 25.0, 40.0
        np.testing.assert_allclose(agent.act(observation), [0.1, 0.0, 0.28], atol=1e-7)
        self.assertEqual(agent.brake_override_count, 1)
        corridor.y, corridor.speed = 30.0, 29.0
        np.testing.assert_allclose(agent.act(observation), [-0.2, 0.05, 0.0], atol=1e-7)
        corridor.y, corridor.speed = None, 40.0
        np.testing.assert_allclose(agent.act(observation), [0.1, 0.1, 0.0], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
