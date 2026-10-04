"""Pixel-triggered early obstacle braking around the fixed hybrid actor."""

import numpy as np

from haic_agent.hybrid_runtime import ObstacleHybridAgent


class ObstacleSlowAgent(ObstacleHybridAgent):
    def __init__(self, base, *, corridor=None):
        super().__init__(base, corridor=corridor)
        self.brake_override_count = 0

    def reset(self, observation):
        super().reset(observation)
        self.brake_override_count = 0

    def act(self, observation):
        action = np.asarray(super().act(observation), dtype=np.float32).copy()
        diagnostic = self.corridor.last_step_diagnostics()
        obstacle_y = diagnostic["obstacle_y"]
        if obstacle_y is not None and obstacle_y >= 25.0 and diagnostic["pixel_speed"] > 30.0:
            action[1] = 0.0
            action[2] = 0.28
            self.brake_override_count += 1
        return action
