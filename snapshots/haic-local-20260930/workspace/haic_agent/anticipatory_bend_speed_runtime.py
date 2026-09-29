"""Conservative pixel-speed increase conditioned on early steering and road shape."""

import numpy as np

from haic_agent.pixel_features import current_frame, estimate_observation_speed, road_centers


class AnticipatoryBendSpeedAgent:
    def __init__(self, base):
        self.base = base
        self.boost_count = 0
        self.boost_during_preview_count = 0

    def reset(self, observation):
        self.base.reset(observation)
        self.boost_count = 0
        self.boost_during_preview_count = 0

    def act(self, observation):
        previews_before = self.base.preview_count
        action = np.asarray(self.base.act(observation), dtype=np.float32).copy()
        corridor = getattr(self.base.base, "corridor", None)
        if corridor is not None and corridor.last_step_diagnostics()["obstacle_y"] is not None:
            return action

        frame = current_frame(observation)
        centers = road_centers(frame)
        if len(centers) != 7:
            return action
        near = centers[54]
        bend = abs(centers[30] - near)
        steer = abs(float(action[0]))
        if (abs(near - 42.0) > 2.5 or bend > 7.0 or steer > 0.10
                or float(action[2]) > 0.01):
            return action

        speed = estimate_observation_speed(observation, frame)
        target = 46.0 - 1.1 * bend - 30.0 * steer
        if speed >= target - 2.0 or float(action[1]) >= 0.16:
            return action
        action[1] = 0.16
        self.boost_count += 1
        if self.base.preview_count > previews_before:
            self.boost_during_preview_count += 1
        return action
