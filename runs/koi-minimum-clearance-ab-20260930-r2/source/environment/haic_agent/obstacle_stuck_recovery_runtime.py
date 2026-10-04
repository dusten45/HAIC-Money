"""Pixel-only escape maneuver for a car stalled beside a visible obstacle."""

import numpy as np

from haic_agent.obstacle_road_guard_runtime import ObstacleRoadGuardAgent


class ObstacleStuckRecoveryAgent(ObstacleRoadGuardAgent):
    def __init__(self, base, *, corridor=None):
        super().__init__(base, corridor=corridor)
        self.recovery_count = 0

    def reset(self, observation):
        super().reset(observation)
        self.recovery_count = 0

    def act(self, observation):
        action = super().act(observation)
        diag = self.corridor.last_step_diagnostics()
        if (diag["obstacle_y"] is None or diag["obstacle_y"] < 50.0
                or diag["pixel_speed"] >= 10.0 or self._committed_side is None):
            return action
        escaped = np.asarray(action, dtype=np.float32).copy()
        escaped[0] = -self._committed_side * self.corridor.MAX_STEER
        escaped[1] = 0.45
        escaped[2] = 0.0
        self.recovery_count += 1
        return escaped
