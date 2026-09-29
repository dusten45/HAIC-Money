"""Four pixel-only high-speed path and recovery interventions."""

import numpy as np

from haic_agent.pixel_features import (
    current_frame, estimate_observation_speed, nearest_bright_object, road_centers,
)


DIRECTIONS = ("road_memory", "path_exit", "obstacle_side", "speed_budget")


class HighSpeedPivotAgent:
    def __init__(self, base, direction):
        if direction not in DIRECTIONS:
            raise ValueError("unknown high-speed pivot direction")
        self.base = base
        self.direction = direction
        self.activation_count = 0
        self.last_road_direction = 0.0
        self.lost_count = 0
        self.path_direction = 0.0
        self.path_hold = 0
        self.pass_side = 0.0
        self.pass_hold = 0
        self.previous_near = None

    def reset(self, observation):
        self.base.reset(observation)
        self.activation_count = 0
        self.last_road_direction = 0.0
        self.lost_count = 0
        self.path_direction = 0.0
        self.path_hold = 0
        self.pass_side = 0.0
        self.pass_hold = 0
        self.previous_near = None

    def act(self, observation):
        base_action = np.asarray(self.base.act(observation), dtype=np.float32)
        action = base_action.copy()
        frame = current_frame(observation)
        centers = road_centers(frame)
        speed = estimate_observation_speed(observation, frame)
        if self.direction == "road_memory":
            self._road_memory(action, centers, speed)
        elif self.direction == "path_exit":
            self._path_exit(action, centers)
        elif self.direction == "obstacle_side":
            self._obstacle_side(action, frame, centers, speed)
        else:
            self._speed_budget(action, centers, speed)
        action = np.asarray((np.clip(action[0], -1.0, 1.0), np.clip(action[1], 0.0, 1.0),
                             np.clip(action[2], 0.0, 1.0)), dtype=np.float32)
        if not np.array_equal(action, base_action):
            self.activation_count += 1
        return action

    def _road_memory(self, action, centers, speed):
        if len(centers) >= 4:
            near = centers.get(54, centers[max(centers)])
            far = centers[min(centers)]
            bend = far - near
            if abs(bend) >= 2.5:
                self.last_road_direction = float(np.sign(bend))
            elif abs(near - 42.0) >= 3.0:
                self.last_road_direction = float(np.sign(near - 42.0))
            self.lost_count = 0
            return
        self.lost_count += 1
        if not self.last_road_direction or speed < 24.0:
            return
        action[0] = float(self.last_road_direction * (0.18 if self.lost_count <= 12 else 0.10))
        action[1] = 0.0
        action[2] = max(float(action[2]), 0.08 if speed >= 35.0 else 0.04)

    def _path_exit(self, action, centers):
        if len(centers) < 7:
            self.path_hold = max(0, self.path_hold - 1)
            return
        near = centers[54]
        middle = centers[42]
        far = centers[30]
        bend = 0.55 * (middle - near) + 0.45 * (far - near)
        if abs(bend) >= 3.0:
            direction = float(np.sign(bend))
            self.path_direction = direction
            self.path_hold = 4
            desired = float(np.clip(0.020 * bend, -0.20, 0.20))
            if abs(desired) > abs(float(action[0])) or desired * action[0] < 0.0:
                action[0] = desired
        elif self.path_hold and self.path_direction * float(action[0]) < -0.04:
            action[0] = 0.0
            self.path_hold -= 1
        else:
            self.path_hold = max(0, self.path_hold - 1)

    def _obstacle_side(self, action, frame, centers, speed):
        obstacle = nearest_bright_object(frame, centers)
        if obstacle is not None and 28.0 <= obstacle[0] <= 49.0:
            _, obstacle_x, road_center = obstacle
            self.pass_side = -1.0 if obstacle_x >= road_center else 1.0
            self.pass_hold = 6
        elif self.pass_hold:
            self.pass_hold -= 1
        else:
            self.pass_side = 0.0
        if not self.pass_side or len(centers) < 4:
            return
        desired = self.pass_side * 0.14
        if abs(desired) > abs(float(action[0])) or desired * action[0] < 0.0:
            action[0] = desired
        if speed > 45.0:
            action[1] = min(float(action[1]), 0.08)

    def _speed_budget(self, action, centers, speed):
        if len(centers) < 7:
            self.previous_near = None
            return
        near = centers[54]
        bend = abs(centers[30] - near)
        drift = abs(near - self.previous_near) if self.previous_near is not None else 0.0
        self.previous_near = near
        target = float(np.clip(58.0 - 1.6 * bend - 1.2 * drift, 34.0, 58.0))
        if speed > target + 3.0:
            action[1] = 0.0
            action[2] = max(float(action[2]), 0.09)
        elif speed < target - 8.0 and bend < 3.0 and drift < 2.0 and abs(near - 42.0) < 3.0:
            action[1] = max(float(action[1]), 0.28)
