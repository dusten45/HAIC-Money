"""Aim a fast pixel policy modestly toward the visible inside of a bend."""

import numpy as np

from haic_agent.fast_pedal_stable_steer_runtime import FastPedalStableSteerAgent
from haic_agent.pixel_features import (
    ROAD_HIGH,
    ROAD_LOW,
    current_frame,
    nearest_bright_object,
    road_centers,
)


class ApexLineAgent(FastPedalStableSteerAgent):
    def __init__(self, fast, stable):
        super().__init__(fast, stable)
        self.apex_decisions = 0

    def reset(self, observation):
        super().reset(observation)
        self.apex_decisions = 0

    def act(self, observation):
        previous_fast_decisions = self.fast_pedal_decisions
        action = np.asarray(super().act(observation), dtype=np.float32).copy()
        if self.fast_pedal_decisions == previous_fast_decisions:
            return action

        frame = current_frame(observation)
        centers = road_centers(frame)
        if len(centers) != 7 or nearest_bright_object(frame, centers) is not None:
            return action
        preview = centers[30] - centers[54]
        if abs(preview) < 4.0:
            return action

        near = centers[54]
        asphalt = (frame[54] >= ROAD_LOW) & (frame[54] <= ROAD_HIGH)
        near_pixels = np.flatnonzero(asphalt & (np.abs(np.arange(84) - near) <= 17.0))
        if len(near_pixels) < 10:
            return action
        half_width = 0.5 * float(near_pixels[-1] - near_pixels[0])
        inside_offset = 0.25 * half_width * np.sign(preview)
        action[0] = np.clip(float(action[0]) + 0.022 * inside_offset, -0.7, 0.7)
        self.apex_decisions += 1
        return action
