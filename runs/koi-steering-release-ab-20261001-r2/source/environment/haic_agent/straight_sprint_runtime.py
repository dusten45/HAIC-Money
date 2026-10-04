"""Use full throttle only on a stable, clear, centered pixel road."""

import numpy as np

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.pixel_features import current_frame, road_centers


class StraightSprintAgent(ObstacleFullRoadGuardAgent):
    def __init__(self, base, *, corridor=None):
        super().__init__(base, corridor=corridor)
        self.clear_frames = 0
        self.sprint_count = 0

    def reset(self, observation):
        super().reset(observation)
        self.clear_frames = 0
        self.sprint_count = 0

    def act(self, observation):
        action = np.asarray(super().act(observation), dtype=np.float32).copy()
        diag = self.corridor.last_step_diagnostics()
        centers = road_centers(current_frame(observation))
        values = list(centers.values())
        clear_straight = (
            diag["obstacle_y"] is None
            and len(values) == 7
            and max(abs(value - 42.0) for value in values) <= 2.5
            and max(values) - min(values) <= 2.0
        )
        self.clear_frames = self.clear_frames + 1 if clear_straight else 0
        if (self.clear_frames >= 8 and diag["pixel_speed"] < 58.0
                and action[2] <= 0.01):
            action[1] = 1.0
            action[2] = 0.0
            self.sprint_count += 1
        return action
