"""Require full pixel road geometry before a last-moment obstacle turn."""

import numpy as np

from haic_agent.obstacle_side_evidence_runtime import ObstacleSideEvidenceAgent
from haic_agent.pixel_features import current_frame, road_centers


class ObstacleFullRoadGuardAgent(ObstacleSideEvidenceAgent):
    def __init__(self, base, *, corridor=None):
        super().__init__(base, corridor=corridor)
        self.full_road_guard_count = 0

    def reset(self, observation):
        super().reset(observation)
        self.full_road_guard_count = 0

    def act(self, observation):
        action = super().act(observation)
        obstacle_y = self.corridor.last_step_diagnostics()["obstacle_y"]
        if obstacle_y is None or obstacle_y < 52.0:
            return action
        centers = road_centers(current_frame(observation))
        if len(centers) >= 7:
            return action
        guarded = np.asarray(action, dtype=np.float32).copy()
        far = centers.get(42, 42.0)
        near = centers.get(54, 42.0)
        guarded[0] = np.clip(0.022 * (far - 42.0) + 0.018 * (far - near),
                             -self.corridor.MAX_STEER, self.corridor.MAX_STEER)
        self.full_road_guard_count += 1
        return guarded
