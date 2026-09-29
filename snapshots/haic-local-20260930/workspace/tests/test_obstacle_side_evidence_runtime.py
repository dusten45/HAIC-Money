import unittest

import numpy as np

from haic_agent.obstacle_side_evidence_runtime import ObstacleSideEvidenceAgent


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
        self.offset = None

    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray([(self.side or 0.0) * 0.34, 0.0, 0.1], dtype=np.float32)

    def last_step_diagnostics(self):
        return {"obstacle_y": self.y, "obstacle_steer_side": self.side,
                "obstacle_x": self.offset + 42.0 if self.offset is not None else None,
                "road_center_at_obstacle": 42.0,
                "pixel_speed": 40.0}


class SideEvidenceTests(unittest.TestCase):
    def test_keeps_clear_initial_side_when_later_pixel_side_flips(self):
        corridor = _Corridor()
        agent = ObstacleSideEvidenceAgent(_Base(), corridor=corridor)
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        agent.reset(obs)
        corridor.y, corridor.side, corridor.offset = 24.0, 1.0, -2.5
        agent.act(obs)
        corridor.y, corridor.side, corridor.offset = 33.0, -1.0, -0.1
        action = agent.act(obs)
        self.assertGreater(float(action[0]), 0.0)

    def test_ignores_edge_offset_before_action_range(self):
        corridor = _Corridor()
        agent = ObstacleSideEvidenceAgent(_Base(), corridor=corridor)
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        agent.reset(obs)
        corridor.y, corridor.side, corridor.offset = 24.0, 1.0, -5.0
        agent.act(obs)
        corridor.y, corridor.side, corridor.offset = 33.0, -1.0, -0.1
        action = agent.act(obs)
        self.assertLess(float(action[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
