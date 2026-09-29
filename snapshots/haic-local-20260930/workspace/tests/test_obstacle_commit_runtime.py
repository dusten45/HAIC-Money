import unittest

import numpy as np

from haic_agent.obstacle_commit_runtime import ObstacleCommitAgent


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


class ObstacleCommitTests(unittest.TestCase):
    def test_obstacle_side_stays_committed_until_detection_is_lost_twice(self):
        corridor = _Corridor()
        agent = ObstacleCommitAgent(_Base(), corridor=corridor)
        observation = np.zeros((4, 84, 84), dtype=np.float32)
        agent.reset(observation)
        corridor.y, corridor.side = 28.0, 1.0
        first = agent.act(observation)
        self.assertGreater(first[0], 0.0)
        corridor.y, corridor.side = 40.0, -1.0
        second = agent.act(observation)
        self.assertGreater(second[0], 0.0)
        self.assertEqual(agent.side_correction_count, 1)
        corridor.y, corridor.side = None, None
        agent.act(observation)
        agent.act(observation)
        corridor.y, corridor.side = 40.0, -1.0
        third = agent.act(observation)
        self.assertLess(third[0], 0.0)


if __name__ == "__main__":
    unittest.main()
