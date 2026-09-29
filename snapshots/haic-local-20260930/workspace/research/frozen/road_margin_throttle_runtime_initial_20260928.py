"""Use visible road width, including a small grass margin, for pedal override."""

import numpy as np

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.pixel_features import ROAD_HIGH, ROAD_LOW, current_frame, road_centers


class RoadMarginThrottleAgent(ObstacleFullRoadGuardAgent):
    def __init__(self, base, *, corridor=None):
        super().__init__(base, corridor=corridor)
        self.margin_throttle_count = 0

    def reset(self, observation):
        super().reset(observation)
        self.margin_throttle_count = 0

    def act(self, observation):
        action = np.asarray(super().act(observation), dtype=np.float32).copy()
        diag = self.corridor.last_step_diagnostics()
        if diag["obstacle_y"] is not None or diag["pixel_speed"] >= 64.0 or action[2] > 0.01:
            return action

        frame = current_frame(observation)
        centers = road_centers(frame)
        if len(centers) != 7 or max(centers.values()) - min(centers.values()) > 10.0:
            return action

        asphalt = (frame[54] >= ROAD_LOW) & (frame[54] <= ROAD_HIGH)
        near_center = centers[54]
        near_pixels = np.flatnonzero(asphalt & (np.abs(np.arange(84) - near_center) <= 17.0))
        if len(near_pixels) < 10:
            return action
        if float(near_pixels[0]) - 2.0 <= 42.0 <= float(near_pixels[-1]) + 2.0:
            action[1] = 1.0
            action[2] = 0.0
            self.margin_throttle_count += 1
        return action
