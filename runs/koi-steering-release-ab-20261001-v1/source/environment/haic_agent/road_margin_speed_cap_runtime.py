"""Accelerate early on visible road, then preserve the baseline's cruising speed."""

from haic_agent.road_margin_throttle_runtime import RoadMarginThrottleAgent


class RoadMarginSpeedCapAgent(RoadMarginThrottleAgent):
    MAX_PIXEL_SPEED = 45.0
