"""Pixel-only preview steering with a short steering memory at road dropout."""

import numpy as np

from haic_agent.pixel_features import current_frame, road_centers
from haic_agent.speed_coupled_corridor_runtime import SpeedCoupledCorridorAgent


class PredictiveCorridorAgent(SpeedCoupledCorridorAgent):
    """Keep the last bend direction long enough to reacquire partially lost road."""

    def __init__(self):
        super().__init__()
        self.preview_count = 0
        self.dropout_count = 0
        self._last_visible_steer = 0.0
        self._missing_frames = 0

    def reset(self, observation=None):
        super().reset(observation)
        self.preview_count = 0
        self.dropout_count = 0
        self._last_visible_steer = 0.0
        self._missing_frames = 0

    def act(self, observation):
        action = np.asarray(super().act(observation), dtype=np.float32).copy()
        centers = road_centers(current_frame(observation))
        obstacle_present = self.last_step_diagnostics()["obstacle_y"] is not None
        if len(centers) == 7:
            self._missing_frames = 0
            if not obstacle_present:
                preview = float(centers[30] - centers[42])
                if abs(preview) >= 1.0:
                    action[0] = np.clip(action[0] + 0.02 * preview,
                                        -self.MAX_STEER, self.MAX_STEER)
                    self.preview_count += 1
            self._last_visible_steer = float(action[0])
        elif not obstacle_present:
            self._missing_frames += 1
            if self._missing_frames <= 4 and abs(self._last_visible_steer) >= 0.03:
                action[0] = float(np.clip(self._last_visible_steer * (0.85 ** self._missing_frames),
                                          -self.MAX_STEER, self.MAX_STEER))
                action[1] = 0.0
                action[2] = max(float(action[2]), 0.20)
                self.dropout_count += 1
        self.last_action = action.copy()
        return action
