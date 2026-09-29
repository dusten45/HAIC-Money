"""Pixel-only controller that couples steering and pedal to upcoming road shape."""

from haic_agent.corridor_agent import VisionCorridorAgent


class SpeedCoupledCorridorAgent(VisionCorridorAgent):
    def __init__(self):
        super().__init__(cruise_speed=55.0, curve_speed_penalty=3.0,
                         max_gas=0.5, obstacle_far_speed=43.0,
                         obstacle_near_speed=36.0)
