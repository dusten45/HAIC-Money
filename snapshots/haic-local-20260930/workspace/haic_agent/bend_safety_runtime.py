"""Accelerate across broad straights while using trained steering through bends."""

import numpy as np

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.pixel_features import ROAD_HIGH, ROAD_LOW, current_frame, road_centers
from haic_agent.speed_coupled_corridor_runtime import SpeedCoupledCorridorAgent


class BendSafetyAgent(ObstacleFullRoadGuardAgent):
    def __init__(self, base, *, corridor=None):
        super().__init__(base, corridor=corridor if corridor is not None
                         else SpeedCoupledCorridorAgent())
        self.straight_count = 0
        self.bend_count = 0
        self.fallback_count = 0

    def reset(self, observation):
        super().reset(observation)
        self.straight_count = 0
        self.bend_count = 0
        self.fallback_count = 0

    def act(self, observation):
        guarded = np.asarray(super().act(observation), dtype=np.float32).copy()
        if self.corridor.last_step_diagnostics()["obstacle_y"] is not None:
            self.fallback_count += 1
            return guarded
        frame = current_frame(observation)
        centers = road_centers(frame)
        if len(centers) != 7:
            self.fallback_count += 1
            return guarded
        corridor_action = np.asarray(self.corridor.last_action, dtype=np.float32).copy()
        spread = max(centers.values()) - min(centers.values())
        asphalt = (frame[54] >= ROAD_LOW) & (frame[54] <= ROAD_HIGH)
        near = np.flatnonzero(asphalt & (np.abs(np.arange(84) - centers[54]) <= 17.0))
        inside_margin = len(near) >= 10 and near[0] - 2 <= 42 <= near[-1] + 2
        if spread <= 4.0 and inside_margin:
            self.straight_count += 1
            return corridor_action
        guarded[1:] = corridor_action[1:]
        self.bend_count += 1
        return guarded
