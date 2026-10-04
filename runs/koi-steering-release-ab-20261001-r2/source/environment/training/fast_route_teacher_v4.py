"""Train-only state teacher with curvature speed profile and pose feedback."""

import math

import numpy as np


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class FastRouteTeacherV4:
    def __init__(self, environment_getter):
        self._environment_getter = environment_getter
        self._track_index = None
        self._track = None
        self._points = None
        self._segment_lengths = None
        self._curve_speeds = None
        self.max_speed = 0.0
        self.brake_decisions = 0
        self.obstacle_decisions = 0

    def reset(self, observation=None):
        del observation
        raw = self._environment_getter().unwrapped
        self._track = raw.track
        self._points = np.asarray([(entry[2], entry[3]) for entry in self._track], dtype=np.float64)
        following = np.roll(self._points, -1, axis=0)
        self._segment_lengths = np.linalg.norm(following - self._points, axis=1)
        n = len(self._track)
        safe_speeds = []
        for i in range(n):
            before = (i - 3) % n
            after = (i + 3) % n
            distance = sum(self._segment_lengths[(before + offset) % n] for offset in range(6))
            curvature = abs(_wrap(self._track[after][1] - self._track[before][1])) / max(distance, 1.0)
            safe_speeds.append(float(np.clip(math.sqrt(58.0 / max(curvature, 0.001)), 22.0, 92.0)))
        self._curve_speeds = safe_speeds
        self._track_index = None
        self.max_speed = 0.0
        self.brake_decisions = 0
        self.obstacle_decisions = 0

    def act(self, observation):
        del observation
        raw = self._environment_getter().unwrapped
        car = raw.car.hull
        x, y = float(car.position[0]), float(car.position[1])
        angle = float(car.angle)
        velocity = car.linearVelocity
        speed = float(np.hypot(float(velocity[0]), float(velocity[1])))
        self.max_speed = max(self.max_speed, speed)
        track = self._track
        n = len(track)
        if self._track_index is None:
            candidates = range(n)
        else:
            candidates = ((self._track_index + step) % n for step in range(-4, 22))
        nearest = min(candidates, key=lambda i: (track[i][2] - x) ** 2 + (track[i][3] - y) ** 2)
        self._track_index = nearest
        _, beta, center_x, center_y = track[nearest]
        lateral_error = (x - center_x) * math.cos(beta) + (y - center_y) * math.sin(beta)
        future_index = (nearest + max(2, min(7, round(speed / 20.0)))) % n
        future_beta = track[future_index][1]

        obstacle_offset = 0.0
        obstacle_distance = None
        for body in raw.obstacles:
            ox, oy = float(body.position[0]), float(body.position[1])
            ahead = -(ox - x) * math.sin(angle) + (oy - y) * math.cos(angle)
            side = (ox - x) * math.cos(angle) + (oy - y) * math.sin(angle)
            if 3.0 < ahead < 55.0 and abs(side) < 10.0:
                if obstacle_distance is None or ahead < obstacle_distance:
                    obstacle_distance = ahead
                    obstacle_offset = -4.5 if side > 0.0 else 4.5
        if obstacle_distance is not None:
            self.obstacle_decisions += 1
        heading_delta = _wrap(future_beta - angle)
        position_correction = math.atan2(1.4 * (lateral_error - obstacle_offset),
                                          max(8.0, speed * 0.25))
        yaw_rate = float(car.angularVelocity)
        steer = float(np.clip(-heading_delta - position_correction + 0.45 * yaw_rate,
                              -0.8, 0.8))

        target_speed = 92.0
        distance = 0.0
        for step in range(34):
            index = (nearest + step) % n
            allowed = math.sqrt(self._curve_speeds[index] ** 2 + 2.0 * 75.0 * distance)
            target_speed = min(target_speed, allowed)
            distance += float(self._segment_lengths[index])
        if obstacle_distance is not None:
            target_speed = min(target_speed, math.sqrt(35.0 ** 2 + 2.0 * 75.0 * max(0.0, obstacle_distance - 15.0)))
        error = target_speed - speed
        if error < -2.0:
            self.brake_decisions += 1
            return np.asarray((steer, 0.0, np.clip(-0.06 * error, 0.1, 0.8)), dtype=np.float32)
        gas = float(np.clip(0.18 + 0.055 * error, 0.0, 1.0))
        return np.asarray((steer, gas, 0.0), dtype=np.float32)
