"""Increase gas from the stable actor only with visible road clearance."""

import numpy as np

from haic_agent.pixel_features import (
    ROAD_HIGH,
    ROAD_LOW,
    current_frame,
    estimate_observation_speed,
    nearest_bright_object,
    road_centers,
)


class StableRiskEnvelopeAgent:
    def __init__(self, stable):
        self.stable = stable
        self.risk_clear_decisions = 0

    def reset(self, observation):
        self.stable.reset(observation)
        self.risk_clear_decisions = 0

    @staticmethod
    def _clearance(frame, row):
        asphalt = (frame[row] >= ROAD_LOW) & (frame[row] <= ROAD_HIGH)
        if not bool(asphalt[42]):
            return 0
        left = 42
        right = 42
        while left > 0 and asphalt[left - 1]:
            left -= 1
        while right < 83 and asphalt[right + 1]:
            right += 1
        return min(42 - left, right - 42)

    def act(self, observation):
        action = np.asarray(self.stable.act(observation), dtype=np.float32).copy()
        frame = current_frame(observation)
        centers = road_centers(frame)
        if len(centers) != 7 or nearest_bright_object(frame, centers) is not None:
            return action
        if self._clearance(frame, 54) < 8 or self._clearance(frame, 50) < 7:
            return action
        if (abs(centers[54] - 42.0) > 3.0 or abs(float(action[0])) > 0.12
                or float(action[2]) > 0.01):
            return action
        pixels = np.asarray(observation, dtype=np.float32)
        prior = road_centers(pixels[-2]) if pixels.shape == (4, 84, 84) else {}
        bend = abs(centers[30] - centers[54])
        if len(prior) != 7 or bend - abs(prior[30] - prior[54]) > 1.0:
            return action
        target_speed = 54.0 - 0.6 * bend
        if estimate_observation_speed(observation, frame) < target_speed - 3.0:
            if float(action[1]) < 0.20:
                action[1] = 0.20
                self.risk_clear_decisions += 1
        return action
