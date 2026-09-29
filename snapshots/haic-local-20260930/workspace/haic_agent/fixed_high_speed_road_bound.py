"""Keep the fixed-speed temporal controller's obstacle steering inside a pixel road bound."""
import numpy as np

from haic_agent.fixed_high_speed_runtime import FixedHighSpeedAgent, PREFIX_DECISIONS
from haic_agent.pixel_features import road_centers


class FixedHighSpeedRoadBound(FixedHighSpeedAgent):
    def __init__(self, mechanism="road_bound"):
        if mechanism != "road_bound":
            raise ValueError("road_bound only")
        super().__init__("damping")

    def act(self, observation):
        action = super().act(observation)
        self.last["road_bound_active"] = False
        centers = self.last["road_centers"]
        pixels = np.asarray(observation)
        previous = road_centers(pixels[-2]) if pixels.ndim == 3 else {}
        obstacle = self.base._last_obstacle
        if (self.steps <= PREFIX_DECISIONS or not self.last["available"] or
                54 not in previous or obstacle is None):
            return action
        near = centers[54]
        projected = near + 2 * (near - previous[54]) - 42
        obstacle_term = (self.base._obstacle_side * self.base.OBSTACLE_STEER *
                         self.base.obstacle_steer_scale * float(np.clip((obstacle[0] - 22) / 18, 0, 1)))
        self.last.update(projected_near_offset=projected, obstacle_term=obstacle_term)
        if abs(projected) > 6 and projected * obstacle_term < 0:
            road = .022 * (centers[42] - 42) + .018 * (centers[42] - near)
            action[0] = np.clip(road + self.last["correction"], -self.base.MAX_STEER, self.base.MAX_STEER)
            self.last.update(road_bound_active=True,
                             changed=abs(float(action[0]) - self.last["inherited_steer"]) > 1e-6)
        return action
