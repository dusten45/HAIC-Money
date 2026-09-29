"""Pixel-only hybrid with bounded steering changes near visible obstacles."""

import numpy as np

from haic_agent.hybrid_runtime import ObstacleHybridAgent


class ObstacleSlewAgent(ObstacleHybridAgent):
    MAX_OBSTACLE_STEER_CHANGE = 0.2

    def __init__(self, base, *, corridor=None):
        super().__init__(base, corridor=corridor)
        self._previous_steer = None
        self.slew_count = 0

    def reset(self, observation):
        super().reset(observation)
        self._previous_steer = None
        self.slew_count = 0

    def act(self, observation):
        action = np.asarray(super().act(observation), dtype=np.float32).copy()
        obstacle_y = self.corridor.last_step_diagnostics()["obstacle_y"]
        if obstacle_y is not None and obstacle_y >= 28.0 and self._previous_steer is not None:
            adjusted = np.clip(action[0], self._previous_steer - self.MAX_OBSTACLE_STEER_CHANGE,
                               self._previous_steer + self.MAX_OBSTACLE_STEER_CHANGE)
            if adjusted != action[0]:
                action[0] = adjusted
                self.slew_count += 1
        self._previous_steer = float(action[0])
        return action
