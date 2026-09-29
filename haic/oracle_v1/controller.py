"""Local privileged geometry oracle; deliberately not a submission Agent."""

import math

import numpy as np


class TrackPath:
    """Closed polyline with metric arc length and signed lateral projection."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)
        self.delta = np.roll(self.points, -1, axis=0) - self.points
        self.lengths = np.linalg.norm(self.delta, axis=1)
        if np.any(self.lengths <= 0):
            raise ValueError("Track contains a zero-length segment")
        self.cumulative = np.r_[0.0, np.cumsum(self.lengths)]
        self.length = float(self.cumulative[-1])

    def project(self, position):
        relative = np.asarray(position) - self.points
        fractions = np.clip(
            np.sum(relative * self.delta, axis=1) / self.lengths**2, 0, 1
        )
        projected = self.points + fractions[:, None] * self.delta
        index = int(np.argmin(np.sum((projected - position) ** 2, axis=1)))
        tangent = self.delta[index] / self.lengths[index]
        error = np.asarray(position) - projected[index]
        lateral = tangent[0] * error[1] - tangent[1] * error[0]
        arc = self.cumulative[index] + fractions[index] * self.lengths[index]
        return float(arc), float(lateral), index

    def sample(self, arc):
        arc = float(arc) % self.length
        index = min(int(np.searchsorted(self.cumulative, arc, side="right") - 1),
                    len(self.points) - 1)
        fraction = (arc - self.cumulative[index]) / self.lengths[index]
        return self.points[index] + fraction * self.delta[index]


class OracleController:
    def __init__(self, base_env, target_speed=12.0, avoid_obstacles=False):
        if not math.isfinite(target_speed) or target_speed <= 0:
            raise ValueError("target_speed must be finite and positive")
        self.env = base_env
        self.target_speed = float(target_speed)
        self.centerline = TrackPath(np.asarray(base_env.track)[:, 2:4])
        self.path = self.centerline
        if avoid_obstacles:
            tangents = self.centerline.delta + np.roll(self.centerline.delta, 1, axis=0)
            tangents /= np.linalg.norm(tangents, axis=1)[:, None]
            normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
            offsets = np.zeros(len(self.centerline.points))
            for obstacle in base_env.track_variables.obstacles:
                arc, lateral, _ = self.centerline.project(obstacle.position)
                # Radius + vehicle half-width + tracking margin. Pass on the
                # centerline's opposite side, blending back over 25 units.
                clearance = obstacle.radius + 1.4 + 1.2
                shift = (min(0.0, lateral - clearance) if lateral >= 0
                         else max(0.0, lateral + clearance))
                distance = (self.centerline.cumulative[:-1] - arc
                            + self.centerline.length / 2) % self.centerline.length
                distance -= self.centerline.length / 2
                blend = 0.5 * (1 + np.cos(np.pi * np.clip(np.abs(distance) / 25.0, 0, 1)))
                offsets += shift * blend
            self.path = TrackPath(self.centerline.points + offsets[:, None] * normals)

    def act(self):
        hull = self.env.car.hull
        position = np.asarray(hull.position, dtype=np.float64)
        velocity = np.asarray(hull.linearVelocity, dtype=np.float64)
        forward = np.array([-math.sin(hull.angle), math.cos(hull.angle)])
        right = np.array([forward[1], -forward[0]])
        speed = float(np.linalg.norm(velocity))
        arc, lateral, index = self.centerline.project(position)
        _, path_error, _ = self.path.project(position)
        rear = position - 1.64 * forward
        rear_arc, _, _ = self.path.project(rear)
        lookahead = 6.0 + 0.25 * speed
        target = self.path.sample(rear_arc + lookahead)
        displacement = target - rear
        # The simulator action is minus the front joint angle (in radians).
        steer = math.atan2(2 * 3.24 * float(displacement @ right),
                           float(displacement @ displacement))
        speed_error = self.target_speed - speed
        gas = float(np.clip(0.12 * speed_error, 0.0, 0.4))
        brake = float(np.clip(-0.08 * speed_error, 0.0, 0.5))
        previous = self.centerline.delta[(index - 1) % len(self.centerline.points)]
        current = self.centerline.delta[index]
        curvature = math.atan2(previous[0] * current[1] - previous[1] * current[0],
                               float(previous @ current)) / self.centerline.lengths[index]
        heading_error = math.atan2(forward[0] * current[1] - forward[1] * current[0],
                                  float(forward @ current))
        action = np.array([np.clip(steer, -1, 1), gas, brake], dtype=np.float32)
        return action, {
            "arc_length": arc, "center_error": lateral, "waypoint_index": index,
            "path_error": path_error,
            "curvature": curvature, "heading_error": heading_error,
            "target": target.tolist(), "target_speed": self.target_speed,
            "lookahead": lookahead, "steer_saturated": abs(steer) >= 0.4,
            "oracle_policy_action_difference": None,
        }
