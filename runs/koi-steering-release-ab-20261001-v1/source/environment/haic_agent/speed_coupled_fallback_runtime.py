"""Use the coupled road controller only while all pixel road rows are visible."""

import numpy as np

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.pixel_features import current_frame, road_centers
from haic_agent.speed_coupled_corridor_runtime import SpeedCoupledCorridorAgent


class SpeedCoupledFallbackAgent(ObstacleFullRoadGuardAgent):
    def __init__(self, base, *, corridor=None):
        super().__init__(base, corridor=corridor if corridor is not None
                         else SpeedCoupledCorridorAgent())
        self.coupled_count = 0
        self.fallback_count = 0

    def reset(self, observation):
        super().reset(observation)
        self.coupled_count = 0
        self.fallback_count = 0

    def act(self, observation):
        guarded_action = np.asarray(super().act(observation), dtype=np.float32).copy()
        diag = self.corridor.last_step_diagnostics()
        centers = road_centers(current_frame(observation))
        if diag["obstacle_y"] is None and len(centers) == 7:
            self.coupled_count += 1
            return np.asarray(self.corridor.last_action, dtype=np.float32).copy()
        self.fallback_count += 1
        return guarded_action
