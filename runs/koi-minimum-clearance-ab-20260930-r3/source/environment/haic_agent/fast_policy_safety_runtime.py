"""Select a faster pixel policy on clear road and recover with a stable policy."""

import numpy as np

from haic_agent.pixel_features import current_frame, estimate_observation_speed, road_centers


class FastPolicySafetyAgent:
    def __init__(self, fast, stable):
        self.fast = fast
        self.stable = stable
        self.protecting = False
        self.clear_decisions = 0
        self.fast_decisions = 0
        self.protected_decisions = 0
        self.protection_entries = 0

    def reset(self, observation):
        self.fast.reset(observation)
        self.stable.reset(observation)
        self.protecting = False
        self.clear_decisions = 0
        self.fast_decisions = 0
        self.protected_decisions = 0
        self.protection_entries = 0

    def act(self, observation):
        frame = current_frame(observation)
        centers = road_centers(frame)
        speed = estimate_observation_speed(observation, frame)
        near_offset = abs(centers[54] - 42.0) if 54 in centers else float("inf")
        unsafe = len(centers) != 7 or near_offset > 8.0
        if unsafe and not self.protecting:
            self.protecting = True
            self.clear_decisions = 0
            self.protection_entries += 1
        if self.protecting:
            if not unsafe and near_offset < 4.0 and speed < 49.0:
                self.clear_decisions += 1
                if self.clear_decisions >= 3:
                    self.protecting = False
                    self.clear_decisions = 0
            else:
                self.clear_decisions = 0
        fast_action = np.asarray(self.fast.act(observation), dtype=np.float32)
        stable_action = np.asarray(self.stable.act(observation), dtype=np.float32)
        if self.protecting:
            self.protected_decisions += 1
            action = stable_action.copy()
            if speed > 55.0:
                action[1] = 0.0
                action[2] = max(float(action[2]), 0.20)
            return action
        self.fast_decisions += 1
        return fast_action.copy()
