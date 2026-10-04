"""Add far-road preview to steering when running the speed-coupled controller."""

import numpy as np

from haic_agent.pixel_features import current_frame, road_centers
from haic_agent.speed_coupled_corridor_runtime import SpeedCoupledCorridorAgent
from haic_agent.speed_coupled_fallback_runtime import SpeedCoupledFallbackAgent


class LookaheadCorridorAgent(SpeedCoupledCorridorAgent):
    def __init__(self):
        super().__init__()
        self.lookahead_count = 0

    def reset(self, observation=None):
        super().reset(observation)
        self.lookahead_count = 0

    def act(self, observation):
        action = np.asarray(super().act(observation), dtype=np.float32).copy()
        centers = road_centers(current_frame(observation))
        if len(centers) == 7:
            far_delta = centers[30] - centers[42]
            if abs(far_delta) > 1.0:
                action[0] = float(np.clip(action[0] + 0.015 * far_delta,
                                          -self.MAX_STEER, self.MAX_STEER))
                self.lookahead_count += 1
        self.last_action = action.copy()
        return action


class LookaheadFallbackAgent(SpeedCoupledFallbackAgent):
    def __init__(self, base):
        super().__init__(base, corridor=LookaheadCorridorAgent())
