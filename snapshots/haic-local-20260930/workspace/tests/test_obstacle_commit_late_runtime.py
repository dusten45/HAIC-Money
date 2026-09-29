import unittest

import numpy as np

from haic_agent.obstacle_commit_late_runtime import ObstacleCommitLateAgent


class _Base:
    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([0.0, 0.1, 0.0], dtype=np.float32)


class _Corridor:
    OBSTACLE_STEER = 0.34
    MAX_STEER = 0.7
    obstacle_steer_scale = 1.0

    def __init__(self):
        self.y = None
        self.side = None

    def reset(self, observation):
        pass

    def act(self, observation):
        urgency = np.clip(((self.y or 22.0) - 22.0) / 18.0, 0.0, 1.0)
        return np.asarray([(self.side or 0.0) * 0.34 * urgency, 0.0, 0.1], dtype=np.float32)

    def last_step_diagnostics(self):
        return {"obstacle_y": self.y, "obstacle_steer_side": self.side}


class ObstacleCommitLateTests(unittest.TestCase):
    def test_ignores_early_side_then_commits_when_avoidance_starts(self):
        corridor = _Corridor()
        agent = ObstacleCommitLateAgent(_Base(), corridor=corridor)
        observation = np.zeros((4, 84, 84), dtype=np.float32)
        agent.reset(observation)
        corridor.y, corridor.side = 24.0, 1.0
        agent.act(observation)
        corridor.y, corridor.side = 28.0, -1.0
        first = agent.act(observation)
        self.assertLess(first[0], 0.0)
        self.assertEqual(agent.side_correction_count, 0)
        corridor.y, corridor.side = 40.0, 1.0
        second = agent.act(observation)
        self.assertLess(second[0], 0.0)
        self.assertEqual(agent.side_correction_count, 1)


if __name__ == "__main__":
    unittest.main()
