"""Fast aligned-road cruise with brief braking on certain, high-speed bends."""

import numpy as np

from haic_agent.pixel_features import current_frame, estimate_observation_speed, road_centers


class AnticipatoryFastCruiseBrakeR2Agent:
    def __init__(self, base):
        self.base = base
        self.stable_count = 0
        self.boost_count = 0
        self.preview_brake_count = 0
        self.alignment_brake_count = 0

    def reset(self, observation):
        self.base.reset(observation)
        self.stable_count = 0
        self.boost_count = 0
        self.preview_brake_count = 0
        self.alignment_brake_count = 0

    def act(self, observation):
        action = np.asarray(self.base.act(observation), dtype=np.float32).copy()
        corridor = getattr(self.base.base, "corridor", None)
        if corridor is not None and corridor.last_step_diagnostics()["obstacle_y"] is not None:
            self.stable_count = 0
            return action
        frame = current_frame(observation)
        centers = road_centers(frame)
        if len(centers) != 7:
            self.stable_count = 0
            return action

        near_offset = abs(centers[54] - 42.0)
        bend = abs(centers[30] - centers[54])
        speed = estimate_observation_speed(observation, frame)
        if near_offset > 5.0:
            self.stable_count = 0
            if speed > 45.0:
                action[1] = 0.0
                action[2] = max(float(action[2]), 0.10)
                self.alignment_brake_count += 1
            return action

        if bend >= 5.0:
            self.stable_count = 0
            brake = 0.12 if bend >= 9.0 and speed > 42.0 else 0.08 if speed > 44.0 else 0.0
            if brake:
                action[1] = 0.0
                action[2] = max(float(action[2]), brake)
                self.preview_brake_count += 1
            return action

        if (bend > 2.5 or near_offset > 1.5
                or abs(float(action[0])) >= 0.035 or float(action[2]) > 0.01):
            self.stable_count = 0
            return action
        self.stable_count += 1
        if self.stable_count >= 5 and speed < 48.0 and float(action[1]) < 0.20:
            action[1] = 0.20
            self.boost_count += 1
        return action
