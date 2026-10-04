"""Pixel-only obstacle controller that preserves one avoidance side per encounter."""

import numpy as np

from haic_agent.corridor_agent import VisionCorridorAgent


class ObstacleCommitAgent:
    def __init__(self, base, *, corridor=None):
        self.base = base
        self.corridor = corridor if corridor is not None else VisionCorridorAgent()
        self.override_count = 0
        self.side_correction_count = 0
        self._committed_side = None
        self._missing = 0

    def reset(self, observation):
        self.base.reset(observation)
        self.corridor.reset(observation)
        self.override_count = 0
        self.side_correction_count = 0
        self._committed_side = None
        self._missing = 0

    def act(self, observation):
        base_action = self.base.act(observation)
        corridor_action = np.asarray(self.corridor.act(observation), dtype=np.float32).copy()
        diagnostic = self.corridor.last_step_diagnostics()
        obstacle_y = diagnostic["obstacle_y"]
        current_side = diagnostic["obstacle_steer_side"]
        if obstacle_y is None:
            self._missing += 1
            if self._missing >= 2:
                self._committed_side = None
        else:
            self._missing = 0
            if self._committed_side is None and current_side is not None:
                self._committed_side = current_side
        if obstacle_y is not None and obstacle_y >= 28.0:
            self.override_count += 1
            if self._committed_side is not None and current_side is not None and current_side != self._committed_side:
                urgency = float(np.clip((obstacle_y - 22.0) / 18.0, 0.0, 1.0))
                corridor_action[0] += (
                    (self._committed_side - current_side)
                    * self.corridor.OBSTACLE_STEER
                    * self.corridor.obstacle_steer_scale
                    * urgency
                )
                corridor_action[0] = np.clip(corridor_action[0], -self.corridor.MAX_STEER, self.corridor.MAX_STEER)
                self.side_correction_count += 1
            return corridor_action
        return np.asarray(base_action, dtype=np.float32)
