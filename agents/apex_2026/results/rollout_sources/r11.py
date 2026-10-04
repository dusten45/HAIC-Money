"""Pixel-only receding-horizon free-space controller.

The image supplies a clearance field and a navigation potential. Independent
steering/pedal trajectories compete under a small approximate vehicle model;
there are no track identifiers, stored routes, learned weights or privileged
simulator inputs.
"""
import cv2
import numpy as np


class Agent:
    # Camera geometry from the public renderer after its initial zoom settles.
    PIXELS_X = 1.3608
    PIXELS_Y = 1.701
    MAX_SPEED = 76.0

    def __init__(self):
        self.previous_steer = 0.0
        self.speed = 0.0
        self.last_diagnostics = {}
        steer, pedal, profile = np.meshgrid(
            np.linspace(-1.0, 1.0, 25), np.arange(6), np.arange(5),
            indexing='ij',
        )
        self.steers = steer.ravel()
        self.gases = np.array([1.0, 0.35, 0.1, 0.0, 0.0, 0.0])[pedal.ravel()]
        self.brakes = np.array([0.0, 0.0, 0.0, 0.0, 0.35, 0.8])[pedal.ravel()]
        self.profiles = profile.ravel()
        self.pulse_steps = np.array([0, 0, 3, 5, 8])[self.profiles]
        self.yy, self.xx = np.mgrid[:74, :84]

    def reset(self, observation=None):
        self.previous_steer = 0.0
        self.speed = 0.0
        self.last_diagnostics = {}

    def _speed(self, frame):
        # The two-stage public resize gives an affine bar-area calibration.
        # Include the baseline rows81-82: omitting them loses ~22m/s.
        mass = float(np.sum(frame[77:83, 10:13]))
        return float(np.clip((mass - 0.27) / 0.085, 0.0, 120.0))

    def _fields(self, frame):
        road = ((frame[:74] > 0.25) & (frame[:74] < 0.53)).astype(np.uint8)
        # Paint only the known car sprite footprint back into free space.
        road[59:68, 39:45] = 1
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        obstacles = (frame[:74] > 0.56).astype(np.uint8)
        obstacles[59:69, 38:46] = 0
        obstacles = cv2.dilate(obstacles, np.ones((3, 3), np.uint8))
        road[obstacles > 0] = 0
        count, labels = cv2.connectedComponents(road, connectivity=8)
        label = labels[63, 42]
        if not label:
            candidates = np.where(road[45:69])
            if not len(candidates[0]):
                return None
            nearest = np.argmin((candidates[0] + 45 - 63) ** 2 + (candidates[1] - 42) ** 2)
            label = labels[candidates[0][nearest] + 45, candidates[1][nearest]]
        road = (labels == label).astype(np.uint8)
        clearance = cv2.distanceTransform(road, cv2.DIST_L2, 5)
        valid = (clearance > 3.0) & (self.yy < 65)
        if not np.any(valid):
            valid = (road > 0) & (self.yy < 65)
        if not np.any(valid):
            return None
        # A corner may leave the visible road horizontal. Choose a distant
        # well-cleared point in front of the rear axle, rather than treating
        # the upper road edge as a destination.
        radial = np.sqrt((self.xx - 42.0) ** 2 +
                         ((self.yy - 63.0) * self.PIXELS_X / self.PIXELS_Y) ** 2)
        desirability = radial + 0.8 * clearance + 0.15 * (63.0 - self.yy)
        best_goal = float(np.max(desirability[valid]))
        goals = valid & (desirability >= best_goal - 1.0)
        potential = np.full(road.shape, 10000.0, dtype=np.float32)
        potential[goals] = 0
        toll = (1.0 + 6.0 / (clearance + 0.5) ** 2).astype(np.float32)
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(105):
            candidate = cv2.erode(potential, kernel) + toll
            potential = np.minimum(potential, candidate)
            potential[road == 0] = 10000
        # Smooth interpolation around thin road boundaries is deliberate:
        # trajectories can compare severity instead of sharing one hard cost.
        return road, clearance, potential

    @staticmethod
    def _sample(field, x, y):
        return cv2.remap(field, x.astype(np.float32)[None, :], y.astype(np.float32)[None, :],
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)[0]

    def _motion(self, previous, current):
        old = np.clip(previous * 255, 0, 255).astype(np.uint8)
        new = np.clip(current * 255, 0, 255).astype(np.uint8)
        mask = np.zeros((84, 84), np.uint8)
        mask[4:70, 4:80] = 255
        mask[56:70, 36:48] = 0
        points = cv2.goodFeaturesToTrack(old, 100, 0.015, 4, mask=mask)
        if points is None or len(points) < 6:
            return 0.0, 0.0, False
        moved, status, error = cv2.calcOpticalFlowPyrLK(old, new, points, None,
                                                     winSize=(15, 15), maxLevel=2)
        if moved is None:
            return 0.0, 0.0, False
        keep = (status.ravel() > 0) & (error.ravel() < 30)
        if np.count_nonzero(keep) < 6:
            return 0.0, 0.0, False
        scale = np.array([self.PIXELS_X, self.PIXELS_Y], np.float32)
        before = points.reshape(-1, 2)[keep] / scale
        after = moved.reshape(-1, 2)[keep] / scale
        transform, inliers = cv2.estimateAffinePartial2D(before, after,
                    method=cv2.RANSAC, ransacReprojThreshold=0.8, maxIters=500)
        if transform is None or np.count_nonzero(inliers) < 6:
            return 0.0, 0.0, False
        center = np.array([42.0, 63.0]) / scale
        delta = transform[:, :2] @ center + transform[:, 2] - center
        yaw_rate = -np.arctan2(transform[1, 0], transform[0, 0]) / 0.08
        velocity = np.array([-delta[0], delta[1]]) / 0.08
        slip = np.arctan2(velocity[0], max(0.1, velocity[1]))
        zoom = np.sqrt(np.linalg.det(transform[:, :2]))
        if not (0.96 < zoom < 1.04) or abs(yaw_rate) > 8 or np.linalg.norm(velocity) > 140:
            return 0.0, 0.0, False
        return float(yaw_rate), float(np.clip(slip, -0.8, 0.8)), True

    @staticmethod
    def _curvature(steer):
        return np.tan(np.clip(steer, -0.4, 0.4)) / 3.24

    @staticmethod
    def _acceleration(speed, gas, brake):
        # Calibrated on straight portions of the four consumed required cells.
        # Gas is near-linear over this short horizon; drag remains speed-dependent.
        return gas * (61.0 - 0.2 * speed) - 0.004 * speed ** 2 - 230.0 * brake

    def act(self, observation):
        obs = np.asarray(observation)
        if obs.shape != (4, 84, 84) or not np.isfinite(obs).all():
            return np.array([0.0, 0.0, 0.8], dtype=np.float32)
        frame = obs[-1]
        self.speed = self._speed(frame)
        measured_yaw, slip, motion_valid = self._motion(obs[-2], frame)
        fields = self._fields(frame)
        if fields is None:
            return np.array([self.previous_steer * 0.5, 0.0, 0.5], dtype=np.float32)
        road, clearance, potential = fields
        n = len(self.steers)
        x = np.full(n, 42.0)
        y = np.full(n, 63.0)
        yaw = np.zeros(n)
        yaw_rate = np.full(n, measured_yaw)
        speed = np.full(n, self.speed)
        wheel = np.full(n, np.clip(self.previous_steer, -0.4, 0.4))
        # A future steering reversal must not erase the traction cost of
        # applying power during the measured present turn.
        score = 0.02 * (self.gases * self.speed * measured_yaw) ** 2
        active = np.ones(n, dtype=bool)
        minimum_clearance = np.full(n, 100.0)
        distance = np.zeros(n)
        dt = 0.04
        start_cost = float(potential[63, 42])
        previous_cost = np.full(n, start_cost)
        for step in range(24):
            simple = np.where(self.profiles == 0, 0.45 if step >= 8 else 1.0, 1.0)
            pulse = np.where(step < self.pulse_steps, 1.0,
                             np.where(step < 2 * self.pulse_steps, -1.0, 0.0))
            steer = self.steers * np.where(self.profiles < 2, simple, pulse)
            wheel += np.clip(np.clip(steer, -0.4, 0.4) - wheel, -0.12, 0.12)
            acceleration = self._acceleration(speed, self.gases, self.brakes)
            speed = np.maximum(0.0, speed + acceleration * dt)
            curvature = self._curvature(wheel)
            yaw_rate += 0.4 * (np.clip(speed * curvature, -6.0, 6.0) - yaw_rate)
            yaw += yaw_rate * dt
            travel_yaw = yaw + slip * np.exp(-(step + 1) * dt / 0.24)
            dx = np.sin(travel_yaw) * speed * dt * self.PIXELS_X
            dy = -np.cos(travel_yaw) * speed * dt * self.PIXELS_Y
            x += dx
            y += dy
            visible = (x >= 1) & (x <= 82) & (y >= 1) & (y <= 72)
            # Beyond the forward image boundary is unknown, not a wall.
            exited_forward = (y < 1) & (np.cos(yaw) > 0.5)
            exited_forward &= active
            score -= exited_forward * (previous_cost * 0.75 + speed * 0.15)
            checked = active & ~exited_forward
            margin = self._sample(clearance, x, y)
            minimum_clearance = np.minimum(minimum_clearance, np.where(checked, margin, 100))
            road_cost = self._sample(potential, x, y)
            road_cost = np.minimum(road_cost, start_cost + 30)
            road_cost = np.where(visible, road_cost, previous_cost + 8)
            score += checked * (1.4 * np.maximum(0, 3.1 - margin) ** 2 + 0.028 * np.maximum(0, speed - self.MAX_SPEED) ** 2)
            # Full throttle while turning can destabilize the rear axle even
            # when the constant-curvature trajectory itself is road-safe.
            score += checked * 0.003 * (self.gases * speed * yaw_rate) ** 2
            score += checked * 0.06 * (speed * speed * np.abs(curvature) / 35.0) ** 2
            distance += active * speed * dt
            # Potential reduction is the progress reward; this also permits
            # sideways road-following around a corner.
            score -= checked * (previous_cost - road_cost) * 0.75
            previous_cost = np.where(checked, road_cost, previous_cost)
            collision = checked & (margin < 1.5)
            score += collision * (110.0 + 0.22 * speed ** 2)
            active &= ~(exited_forward | collision)
        score -= 0.22 * distance
        score += 0.6 * (self.steers - self.previous_steer) ** 2 + 0.4 * self.steers ** 2
        score += self.brakes * 0.25
        index = int(np.argmin(score))
        steer = float(self.steers[index])
        gas = float(self.gases[index])
        brake = float(self.brakes[index])
        self.previous_steer = steer
        self.last_diagnostics = {'speed': self.speed, 'yaw_rate': measured_yaw,
                                 'slip': slip, 'motion_valid': motion_valid, 'score': float(score[index]),
                                 'clearance': float(minimum_clearance[index])}
        return np.array([steer, gas, brake], dtype=np.float32)
