"""Correct pursuit road replacement before the single steering saturation."""
import numpy as np

from haic_agent.fixed_high_speed_runtime import FixedHighSpeedAgent, PREFIX_DECISIONS


class FixedHighSpeedPursuitRepair(FixedHighSpeedAgent):
    def act(self, observation):
        action = super().act(observation)
        if self.mechanism != "pursuit" or self.steps <= PREFIX_DECISIONS or not self.last["available"]:
            return action
        far = self.last["road_centers"][30]
        desired = .65 * (2 * 24 * (far - 42) / ((far - 42) ** 2 + 24 ** 2))
        obstacle_term = 0.0
        if self.base._last_obstacle is not None:
            y = self.base._last_obstacle[0]
            urgency = float(np.clip((y - 22) / 18, 0, 1))
            obstacle_term = (self.base._obstacle_side * self.base.OBSTACLE_STEER *
                             self.base.obstacle_steer_scale * urgency)
        action[0] = np.clip(desired + obstacle_term, -self.base.MAX_STEER, self.base.MAX_STEER)
        self.last.update(pursuit_road_term=desired, preserved_obstacle_term=obstacle_term,
                         correction=float(action[0]) - self.last["inherited_steer"],
                         changed=abs(float(action[0]) - self.last["inherited_steer"]) > 1e-6)
        return action
