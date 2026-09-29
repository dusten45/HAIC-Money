import unittest

import numpy as np

from haic_agent.obstacle_stuck_recovery_runtime import ObstacleStuckRecoveryAgent


class _Base:
    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([0.0, 0.1, 0.0], dtype=np.float32)


class _Corridor:
    OBSTACLE_STEER = 0.34
    MAX_STEER = 0.7
    obstacle_steer_scale = 1.0

    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([-0.6, 0.12, 0.0], dtype=np.float32)

    def last_step_diagnostics(self):
        return {"obstacle_y": 56.0, "obstacle_steer_side": -1.0,
                "pixel_speed": 4.0}


class StuckRecoveryTests(unittest.TestCase):
    def test_turns_away_from_committed_side_when_stopped_at_obstacle(self):
        agent = ObstacleStuckRecoveryAgent(_Base(), corridor=_Corridor())
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        agent.reset(obs)
        action = agent.act(obs)
        self.assertGreater(float(action[0]), 0.0)
        self.assertGreater(float(action[1]), 0.1)
        self.assertEqual(float(action[2]), 0.0)
        self.assertEqual(agent.recovery_count, 1)


if __name__ == "__main__":
    unittest.main()
