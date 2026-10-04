"""Commit an early obstacle side only when pixel offset gives usable evidence."""

from haic_agent.obstacle_road_guard_runtime import ObstacleRoadGuardAgent


class ObstacleSideEvidenceAgent(ObstacleRoadGuardAgent):
    def __init__(self, base, *, corridor=None):
        super().__init__(base, corridor=corridor)
        self.early_evidence_count = 0

    def reset(self, observation):
        super().reset(observation)
        self.early_evidence_count = 0

    def act(self, observation):
        action = super().act(observation)
        diag = self.corridor.last_step_diagnostics()
        obstacle_y = diag["obstacle_y"]
        if obstacle_y is None or obstacle_y >= 28.0:
            return action
        obstacle_x = diag["obstacle_x"]
        road_center = diag["road_center_at_obstacle"]
        if obstacle_x is None or road_center is None:
            return action
        offset = obstacle_x - road_center
        if 1.0 <= abs(offset) <= 4.0:
            self._committed_side = 1.0 if offset < 0.0 else -1.0
            self.early_evidence_count += 1
        return action
