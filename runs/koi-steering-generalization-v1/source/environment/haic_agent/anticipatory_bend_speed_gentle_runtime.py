"""Small gas increase after sustained pixel-road and steering stability."""

import numpy as np

from haic_agent.pixel_features import current_frame, estimate_observation_speed, road_centers


class AnticipatoryBendGentleSpeedAgent:
    def __init__(self, base):
        self.base = base
        self.stable_count = 0
        self.boost_count = 0

    def reset(self, observation):
        self.base.reset(observation)
        self.stable_count = 0
        self.boost_count = 0

    def act(self, observation):
        action = np.asarray(self.base.act(observation), dtype=np.float32).copy()
        corridor = getattr(self.base.base, "corridor", None)
        if corridor is not None and corridor.last_step_diagnostics()["obstacle_y"] is not None:
            self.stable_count = 0
            return action
        frame = current_frame(observation)
        centers = road_centers(frame)
        if (len(centers) != 7 or abs(centers[54] - 42.0) > 1.5
                or abs(centers[30] - centers[54]) > 2.5
                or abs(float(action[0])) >= 0.035 or float(action[2]) > 0.01):
            self.stable_count = 0
            return action

        self.stable_count += 1
        if (self.stable_count < 7 or
                estimate_observation_speed(observation, frame) >= 41.0 or
                float(action[1]) >= 0.12):
            return action
        action[1] = 0.12
        self.boost_count += 1
        return action
