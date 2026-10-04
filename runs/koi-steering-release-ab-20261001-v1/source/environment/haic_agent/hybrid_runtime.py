"""Pixel-only obstacle control around a trained actor."""

import numpy as np

from haic_agent.corridor_agent import VisionCorridorAgent


class ObstacleHybridAgent:
    def __init__(self, base, *, corridor=None):
        self.base = base
        self.corridor = corridor if corridor is not None else VisionCorridorAgent()
        self.override_count = 0

    def reset(self, observation):
        self.base.reset(observation)
        self.corridor.reset(observation)
        self.override_count = 0

    def act(self, observation):
        base_action = self.base.act(observation)
        corridor_action = self.corridor.act(observation)
        obstacle_y = self.corridor.last_step_diagnostics()["obstacle_y"]
        if obstacle_y is not None and obstacle_y >= 28.0:
            self.override_count += 1
            return np.asarray(corridor_action, dtype=np.float32)
        return np.asarray(base_action, dtype=np.float32)
