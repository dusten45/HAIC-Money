"""Train-only geometry teacher for testing whether fast finished laps are feasible.

This module is never imported by submission inference. The teacher reads the
simulator track and car state only while collecting TRAIN demonstrations.
"""

import math

import numpy as np


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class FastRouteTeacherV2:
    def __init__(self, environment_getter):
        self._environment_getter = environment_getter
        self._track_index = None
        self.max_speed = 0.0
        self.brake_decisions = 0
        self.obstacle_decisions = 0

    def reset(self, observation=None):
        del observation
        self._track_index = None
        self.max_speed = 0.0
        self.brake_decisions = 0
        self.obstacle_decisions = 0

    def act(self, observation):
        del observation
        raw = self._environment_getter().unwrapped
        track = raw.track
        car = raw.car.hull
        x, y = float(car.position[0]), float(car.position[1])
        angle = float(car.angle)
        velocity = car.linearVelocity
        speed = float(np.hypot(float(velocity[0]), float(velocity[1])))
        self.max_speed = max(speed, self.max_speed)
        n = len(track)
        if self._track_index is None:
            candidates = range(n)
        else:
            candidates = ((self._track_index + step) % n for step in range(-3, 17))
        nearest = min(candidates, key=lambda i: (track[i][2] - x) ** 2 + (track[i][3] - y) ** 2)
        self._track_index = nearest

        preview_distance = float(np.clip(6.0 + 0.14 * speed, 8.0, 20.0))
        goal = nearest
        travelled = 0.0
        while travelled < preview_distance:
            following = (goal + 1) % n
            travelled += math.hypot(track[following][2] - track[goal][2],
                                    track[following][3] - track[goal][3])
            goal = following
        _, beta, target_x, target_y = track[goal]
        right_x, right_y = math.cos(beta), math.sin(beta)
        obstacle_offset = 0.0
        for body in raw.obstacles:
            ox, oy = float(body.position[0]), float(body.position[1])
            ahead = -(ox - x) * math.sin(angle) + (oy - y) * math.cos(angle)
            lateral = (ox - x) * math.cos(angle) + (oy - y) * math.sin(angle)
            if 8.0 < ahead < preview_distance + 15.0 and abs(lateral) < 9.0:
                obstacle_offset = -4.5 if lateral > 0.0 else 4.5
                self.obstacle_decisions += 1
                break
        target_x += obstacle_offset * right_x
        target_y += obstacle_offset * right_y
        dx, dy = target_x - x, target_y - y
        forward = -dx * math.sin(angle) + dy * math.cos(angle)
        lateral = dx * math.cos(angle) + dy * math.sin(angle)
        heading_error = math.atan2(lateral, forward)
        steer = float(np.clip(heading_error, -1.0, 1.0))

        bend = max(abs(_wrap(track[(nearest + step) % n][1] - track[nearest][1]))
                   for step in range(1, 17))
        target_speed = float(np.clip(80.0 - 48.0 * bend, 20.0, 80.0))
        if obstacle_offset:
            target_speed = min(target_speed, 42.0)
        if speed > target_speed + 2.0:
            self.brake_decisions += 1
            return np.array([steer, 0.0, 0.8], dtype=np.float32)
        gas = (1.0 if bend < 0.25 else 0.25) if speed < target_speed - 3.0 else 0.1
        return np.array([steer, gas, 0.0],
                        dtype=np.float32)
