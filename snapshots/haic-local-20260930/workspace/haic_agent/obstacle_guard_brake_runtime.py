"""Pixel-triggered high-speed obstacle braking for full-road guard."""

import numpy as np

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent


class ObstacleGuardBrakeAgent(ObstacleFullRoadGuardAgent):
    def __init__(self, base, *, corridor=None):
        super().__init__(base, corridor=corridor)
        self.guard_brake_count = 0

    def reset(self, observation):
        super().reset(observation)
        self.guard_brake_count = 0

    def act(self, observation):
        action = np.asarray(super().act(observation), dtype=np.float32).copy()
        diag = self.corridor.last_step_diagnostics()
        obstacle_y = diag["obstacle_y"]
        if obstacle_y is not None and obstacle_y >= 28.0 and diag["pixel_speed"] > 35.0:
            action[1] = 0.0
            action[2] = 0.28
            self.guard_brake_count += 1
        return action
