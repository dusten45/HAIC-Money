"""Forget a prior avoidance side when road pixels lose near-field coverage."""

from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.pixel_features import current_frame, road_centers


class ObstacleVisibilityResetAgent(ObstacleFullRoadGuardAgent):
    def __init__(self, base, *, corridor=None):
        super().__init__(base, corridor=corridor)
        self.visibility_reset_count = 0

    def reset(self, observation):
        super().reset(observation)
        self.visibility_reset_count = 0

    def act(self, observation):
        action = super().act(observation)
        obstacle_y = self.corridor.last_step_diagnostics()["obstacle_y"]
        if obstacle_y is None or obstacle_y < 52.0 or self._committed_side is None:
            return action
        if len(road_centers(current_frame(observation))) < 7:
            self._committed_side = None
            self.visibility_reset_count += 1
        return action
