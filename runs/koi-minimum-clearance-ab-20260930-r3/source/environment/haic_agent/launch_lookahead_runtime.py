"""Keep strong launch acceleration while limiting the road-speed target."""

from haic_agent.lookahead_corridor_runtime import LookaheadCorridorAgent
from haic_agent.speed_coupled_fallback_runtime import SpeedCoupledFallbackAgent


class LaunchLookaheadCorridorAgent(LookaheadCorridorAgent):
    def __init__(self):
        super().__init__()
        self.cruise_speed = 50.0


class LaunchLookaheadFallbackAgent(SpeedCoupledFallbackAgent):
    def __init__(self, base):
        super().__init__(base, corridor=LaunchLookaheadCorridorAgent())
