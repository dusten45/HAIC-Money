"""Arbitrate road and obstacle through the last useful pixel-obstacle row."""

import numpy as np

from haic_agent.pixel_features import current_frame, road_centers


class SpeedPreviewRoadRecenterAgent:
    def __init__(self, base):
        self.base = base
        self.return_direction = 0.0
        self.last_near_error = 0.0
        self.unseen_count = 0
        self.recenter_count = 0
        self.held_direction_count = 0
        self.obstacle_road_count = 0

    def reset(self, observation):
        self.base.reset(observation)
        self.return_direction = 0.0
        self.last_near_error = 0.0
        self.unseen_count = 0
        self.recenter_count = 0
        self.held_direction_count = 0
        self.obstacle_road_count = 0

    def act(self, observation):
        action = np.asarray(self.base.act(observation), dtype=np.float32).copy()
        guard = getattr(getattr(self.base, "base", None), "base", None)
        corridor = getattr(guard, "corridor", None)
        centers = road_centers(current_frame(observation))
        near = centers.get(54)
        obstacle = corridor.last_step_diagnostics() if corridor is not None else {}
        if obstacle.get("obstacle_y") is not None:
            obstacle_x = obstacle.get("obstacle_x")
            obstacle_center = obstacle.get("road_center_at_obstacle")
            if (obstacle["obstacle_y"] <= 57.0
                    and len(centers) < 7 and near is not None and abs(near - 42.0) >= 5.0
                    and obstacle_x is not None and obstacle_center is not None
                    and (near - 42.0) * (obstacle_x - obstacle_center) > 0.0
                    and abs(obstacle_x - obstacle_center) >= 8.0):
                side = float(np.sign(near - 42.0))
                target = near - 3.0 * side
                action[0] = float(np.clip(0.025 * (target - 42.0), -0.30, 0.30))
                action[1] = min(float(action[1]), 0.05)
                action[2] = max(float(action[2]), 0.06)
                self.obstacle_road_count += 1
                self.return_direction = side
                self.last_near_error = float(near - 42.0)
                self.unseen_count = 0
                return action
            self.return_direction = 0.0
            self.last_near_error = 0.0
            self.unseen_count = 0
            return action
        if len(centers) == 7 and near is not None and abs(near - 42.0) <= 3.0:
            self.return_direction = 0.0
            self.last_near_error = 0.0
            self.unseen_count = 0
            return action
        if len(centers) < 7 and near is not None and abs(near - 42.0) >= 5.0:
            self.return_direction = float(np.sign(near - 42.0))
            self.last_near_error = float(near - 42.0)
            self.unseen_count = 0
            action[0] = float(np.clip(0.026 * (near - 42.0), -0.30, 0.30))
            action[1] = min(float(action[1]), 0.06)
            action[2] = min(float(action[2]), 0.02)
            self.recenter_count += 1
            return action
        hold_limit = 30 if abs(self.last_near_error) <= 8.0 else 20
        if len(centers) < 7 and near is None and self.return_direction and self.unseen_count < hold_limit:
            self.unseen_count += 1
            action[0] = 0.25 * self.return_direction
            if abs(self.last_near_error) <= 8.0:
                action[1] = 0.0
                action[2] = max(float(action[2]), 0.08)
            else:
                action[1] = min(float(action[1]), 0.06)
                action[2] = min(float(action[2]), 0.02)
            self.held_direction_count += 1
            return action
        self.return_direction = 0.0
        self.last_near_error = 0.0
        self.unseen_count = 0
        return action
