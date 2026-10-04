"""Independent metric camera racer; inference needs only NumPy.

The observation camera follows the hull with an affine overhead transform.
Road pixels therefore supply distances, heading and local curvature directly;
no episode identity, simulator import, map or learned checkpoint is used.
"""

import numpy as np


class Agent:
    PX_X = 1.3608
    PX_Y = 1.701
    CAR_X = 42.0
    CAR_Y = 63.0
    WHEELBASE = 3.24
    HALF_ROAD = 6.6666666667

    def __init__(self, cruise_speed=100.0, lateral_accel=145.0,
                 preview_time=0.19, steer_gain=1.0):
        parameters = np.asarray([cruise_speed, lateral_accel, preview_time, steer_gain])
        if not np.isfinite(parameters).all() or np.any(parameters <= 0):
            raise ValueError("controller parameters must be finite and positive")
        self.cruise_speed = float(min(cruise_speed, 130.0))
        self.lateral_accel = float(lateral_accel)
        self.preview_time = float(preview_time)
        self.steer_gain = float(steer_gain)
        self.reset()

    def reset(self, observation=None):
        self.last_steer = 0.0
        self.last_center = self.CAR_X
        self.lost_frames = 0
        self.pass_side = 0.0
        self.pass_missing = 0
        self.last_speed = 0.0
        self.last_target = self.cruise_speed

    @staticmethod
    def _frame(observation):
        pixels = np.asarray(observation)
        if pixels.ndim == 3 and pixels.shape[1:] == (84, 84):
            pixels = pixels[-1]
        if pixels.shape != (84, 84) or not np.isfinite(pixels).all():
            return None
        return pixels.astype(np.float32, copy=False)

    @staticmethod
    def _speed(frame):
        # Official white gauge: width2.1px; 0.042px height per m/s.
        # Preserve the full strip instead of saturating at 80m/s.
        gauge = frame[74:83, 10:13]
        mass = float(np.sum(gauge))
        return float(np.clip((mass - 0.20) / 0.0882, 0.0, 180.0))

    def _road(self, frame):
        asphalt = (frame >= 0.32) & (frame <= 0.51)
        rows, lefts, rights, middles = [], [], [], []
        previous = self.CAR_X
        previous_y = 69.0
        slope = 0.0
        for row in range(69, 3, -2):
            points = np.flatnonzero(asphalt[row])
            if points.size < 5:
                continue
            # Bridge a sprite/skid hole without merging separate road branches.
            cuts = np.flatnonzero(np.diff(points) > 6) + 1
            runs = [run for run in np.split(points, cuts) if run.size >= 5]
            if not runs:
                continue
            expected = previous + slope * (previous_y - row)
            run = min(runs, key=lambda p: abs(0.5 * (p[0] + p[-1]) - expected))
            left, right = float(run[0]), float(run[-1])
            middle = 0.5 * (left + right)
            if abs(middle - expected) > 22.0:
                continue
            # An edge clipped by the image is not the actual asphalt edge.
            half_width = self.HALF_ROAD * self.PX_X * np.sqrt(1.0 + (slope * self.PX_Y / self.PX_X) ** 2)
            if left == 0.0 and right < 83.0:
                middle = right - half_width
            elif right == 83.0 and left > 0.0:
                middle = left + half_width
            if rows:
                derivative = (middle - previous) / max(previous_y - row, 1.0)
                slope = 0.5 * slope + 0.5 * float(np.clip(derivative, -2.0, 2.0))
            rows.append(float(row))
            lefts.append(left)
            rights.append(right)
            middles.append(middle)
            previous, previous_y = middle, float(row)
        if len(rows) < 5:
            return None
        rows = np.asarray(rows)
        centers = np.asarray(middles)
        forward = (self.CAR_Y - rows) / self.PX_Y
        lateral = (centers - self.CAR_X) / self.PX_X
        return forward, lateral, rows, np.asarray(lefts), np.asarray(rights)

    def _obstacle(self, frame, road):
        ahead, lateral, rows, lefts, rights = road
        # Orange grey is ~0.686; green is ~0.63/0.70. Restrict to an
        # interior bounded on both sides by asphalt to reject the shoulder.
        candidates = []
        for row in range(13, 71):
            left = float(np.interp(row, rows[::-1], lefts[::-1]))
            right = float(np.interp(row, rows[::-1], rights[::-1]))
            lo, hi = max(1, int(left + 2)), min(83, int(right - 1))
            if hi - lo < 5:
                continue
            bright = np.flatnonzero((frame[row, lo:hi] >= 0.655) & (frame[row, lo:hi] <= 0.705))
            if bright.size >= 2:
                for run in np.split(bright, np.flatnonzero(np.diff(bright) > 1) + 1):
                    if 2 <= run.size <= 8:
                        x = float(lo + run.mean())
                        if x - left >= 2.0 and right - x >= 2.0:
                            candidates.append((float(row), x, left, right))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])

    def _recover(self):
        self.lost_frames += 1
        desired = np.clip((self.last_center - self.CAR_X) * 0.025, -0.35, 0.35)
        if abs(desired) < 0.04:
            desired = 0.6 * self.last_steer
        self.last_steer += float(np.clip(desired - self.last_steer, -0.24, 0.24))
        if self.lost_frames <= 3:
            return np.asarray([self.last_steer, 0.0, 0.25], dtype=np.float32)
        return np.asarray([self.last_steer, 0.09, 0.0], dtype=np.float32)

    def act(self, observation):
        frame = self._frame(observation)
        if frame is None:
            return self._recover()
        road = self._road(frame)
        if road is None:
            return self._recover()
        self.lost_frames = 0
        forward, lateral, rows, lefts, rights = road
        speed = self._speed(frame)
        self.last_speed = speed
        self.last_center = self.CAR_X + self.PX_X * float(np.interp(5.0, forward, lateral))
        preview = float(np.clip(5.5 + self.preview_time * speed, 7.0, 29.0))
        preview = min(preview, max(6.0, float(forward[-1]) - 1.0))
        x_target = float(np.interp(preview, forward, lateral))

        fit = (forward >= 0.0) & (forward <= min(32.0, forward[-1]))
        if np.count_nonzero(fit) >= 5:
            coefficients = np.polyfit(forward[fit], lateral[fit], 2)
            curvature = abs(2.0 * float(coefficients[0]))
            # Account for road heading when converting x'' into curvature.
            heading = 2.0 * coefficients[0] * min(preview, 18.0) + coefficients[1]
            curvature /= (1.0 + float(heading) ** 2) ** 1.5
        else:
            curvature = 0.03
        target_speed = min(self.cruise_speed, np.sqrt(self.lateral_accel / max(curvature, 0.001)))
        # A curve running out of view requires an approach speed compatible
        # with the remaining observed distance and available braking force.
        if forward[-1] < 22.0:
            target_speed = min(target_speed, np.sqrt(38.0 ** 2 + 2.0 * 100.0 * max(0.0, forward[-1] - 7.0)))

        obstacle = self._obstacle(frame, road)
        if obstacle is not None:
            row, x, left, right = obstacle
            self.pass_missing = 0
            if self.pass_side == 0.0:
                self.pass_side = -1.0 if x - left >= right - x else 1.0
            obstacle_distance = max(2.0, (self.CAR_Y - row) / self.PX_Y)
            pass_x = 0.5 * (left + right) + self.pass_side * 5.0
            # Vehicle halfwidth+clearance is 2.1 pixels; obstacle radius is
            # 1.63 pixels. Route only through the visible asphalt interior.
            pass_x = float(np.clip(pass_x, left + 2.8, right - 2.8))
            weight = np.clip((30.0 - obstacle_distance) / 18.0, 0.25, 1.0)
            x_target = (pass_x - self.CAR_X) / self.PX_X
            preview = min(preview, max(7.0, obstacle_distance))
            target_speed = min(target_speed, np.sqrt(58.0 ** 2 + 2.0 * 90.0 * max(0.0, obstacle_distance - 7.0)))
        else:
            self.pass_missing += 1
            if self.pass_side and self.pass_missing < 4:
                x_target += self.pass_side * 3.3
                target_speed = min(target_speed, 58.0)
            if self.pass_missing >= 4:
                self.pass_side = 0.0

        pursuit_curvature = 2.0 * x_target / (preview ** 2 + x_target ** 2)
        desired = self.steer_gain * float(np.arctan(self.WHEELBASE * pursuit_curvature))
        steer_limit = min(0.4, np.arctan(self.WHEELBASE * self.lateral_accel / max(speed * speed, 100.0)))
        desired = float(np.clip(desired, -steer_limit, steer_limit))
        self.last_steer += float(np.clip(desired - self.last_steer, -0.24, 0.24))
        target_speed = min(target_speed, np.sqrt(self.lateral_accel / max(abs(pursuit_curvature) * self.steer_gain, 0.001)))
        target_speed = min(target_speed, self.last_target + 2.5)
        self.last_target = float(target_speed)
        error = target_speed - speed
        if error < -2.0:
            gas, brake = 0.0, float(np.clip(-error * 0.018, 0.04, 0.70))
        else:
            gas = float(np.clip(0.30 + 0.055 * error, 0.0, 1.0))
            if abs(self.last_steer) > 0.04:
                gas = min(gas, 0.16)
            brake = 0.0
        return np.asarray([self.last_steer, gas, brake], dtype=np.float32)
