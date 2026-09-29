"""Pixel-only earlier steering tied to speed and conservative clear-road cruise."""

import numpy as np

from haic_agent.pixel_features import current_frame, estimate_observation_speed, road_centers


class SpeedCoupledPreviewAgent:
    def __init__(self, base):
        self.base = base
        self.stable_count = 0
        self.early_steer_count = 0
        self.boost_count = 0
        self.bend_brake_count = 0
        self.hazard_brake_count = 0

    def reset(self, observation):
        self.base.reset(observation)
        self.stable_count = 0
        self.early_steer_count = 0
        self.boost_count = 0
        self.bend_brake_count = 0
        self.hazard_brake_count = 0

    def act(self, observation):
        action = np.asarray(self.base.act(observation), dtype=np.float32).copy()
        frame = current_frame(observation)
        speed = estimate_observation_speed(observation, frame)
        corridor = getattr(self.base.base, "corridor", None)
        obstacle = corridor.last_step_diagnostics()["obstacle_y"] if corridor is not None else None
        if obstacle is not None:
            self.stable_count = 0
            if obstacle >= 42.0 and speed > 42.0:
                action[1] = 0.0
                action[2] = max(float(action[2]), 0.08)
                self.hazard_brake_count += 1
            return action

        centers = road_centers(frame)
        if len(centers) != 7:
            self.stable_count = 0
            return action
        near = centers[54]
        bend = centers[30] - near
        previous = np.asarray(observation, dtype=np.float32)[-2]
        previous_centers = road_centers(previous)
        previous_bend = previous_centers.get(30, 42.0) - previous_centers.get(54, 42.0)

        if (abs(near - 42.0) <= 3.0 and abs(bend) >= 3.5
                and abs(previous_bend) >= 2.5 and bend * previous_bend > 0.0):
            self.stable_count = 0
            desired = float(np.clip(0.015 * bend * np.clip(speed / 40.0, 0.8, 1.3), -0.17, 0.17))
            if (abs(float(action[0])) < 0.05 or action[0] * desired > 0.0) and abs(desired) > abs(float(action[0])):
                action[0] = desired
                self.early_steer_count += 1
            if abs(bend) >= 7.0 and speed > 45.0:
                action[1] = 0.0
                action[2] = max(float(action[2]), 0.07)
                self.bend_brake_count += 1
            return action

        if (abs(bend) > 2.0 or abs(near - 42.0) > 1.5
                or abs(float(action[0])) >= 0.035 or float(action[2]) > 0.01):
            self.stable_count = 0
            return action
        self.stable_count += 1
        if self.stable_count >= 5 and speed < 48.0 and float(action[1]) < 0.16:
            action[1] = 0.16
            self.boost_count += 1
        return action
