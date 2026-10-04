"""Pixel-only controller that couples steering and pedal to upcoming road shape."""

import numpy as np

from haic_agent.corridor_agent import VisionCorridorAgent


class SpeedCoupledCorridorAgent(VisionCorridorAgent):
    def __init__(self):
        super().__init__(cruise_speed=55.0, curve_speed_penalty=3.0,
                         max_gas=0.5, obstacle_far_speed=43.0,
                         obstacle_near_speed=36.0)
        self.last_action = np.zeros(3, dtype=np.float32)

    def reset(self, observation=None):
        super().reset(observation)
        self.last_action = np.zeros(3, dtype=np.float32)

    def act(self, observation):
        self.last_action = np.asarray(super().act(observation), dtype=np.float32).copy()
        return self.last_action.copy()
