"""Begin road recentering early while preserving recovery momentum."""

import numpy as np

from haic_agent.pixel_features import current_frame, road_centers


class SpeedPreviewRoadRecenterAgent:
    def __init__(self, base):
        self.base = base
        self.return_direction = 0.0
        self.unseen_count = 0
        self.recenter_count = 0
        self.held_direction_count = 0

    def reset(self, observation):
        self.base.reset(observation)
        self.return_direction = 0.0
        self.unseen_count = 0
        self.recenter_count = 0
        self.held_direction_count = 0

    def act(self, observation):
        action = np.asarray(self.base.act(observation), dtype=np.float32).copy()
        guard = getattr(getattr(self.base, "base", None), "base", None)
        corridor = getattr(guard, "corridor", None)
        if corridor is not None and corridor.last_step_diagnostics()["obstacle_y"] is not None:
            self.return_direction = 0.0
            self.unseen_count = 0
            return action

        centers = road_centers(current_frame(observation))
        near = centers.get(54)
        if len(centers) == 7 and near is not None and abs(near - 42.0) <= 3.0:
            self.return_direction = 0.0
            self.unseen_count = 0
            return action
        if len(centers) < 7 and near is not None and abs(near - 42.0) >= 5.0:
            self.return_direction = float(np.sign(near - 42.0))
            self.unseen_count = 0
            action[0] = float(np.clip(0.026 * (near - 42.0), -0.30, 0.30))
            action[1] = min(float(action[1]), 0.06)
            action[2] = min(float(action[2]), 0.02)
            self.recenter_count += 1
            return action
        if len(centers) < 7 and near is None and self.return_direction and self.unseen_count < 20:
            self.unseen_count += 1
            action[0] = 0.25 * self.return_direction
            action[1] = min(float(action[1]), 0.06)
            action[2] = min(float(action[2]), 0.02)
            self.held_direction_count += 1
            return action
        self.return_direction = 0.0
        self.unseen_count = 0
        return action
