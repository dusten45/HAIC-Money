"""Independent metric camera racer; inference needs only NumPy.

The observation camera follows the hull with an affine overhead transform.
Road pixels therefore supply distances, heading and local curvature directly;
no episode identity, simulator import, map or learned checkpoint is used.
"""

import numpy as np


class _Camera:
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
        self.last_obstacle_y = None
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
        _ahead, _lateral, rows, lefts, rights = road
        bright = (frame >= 0.655) & (frame <= 0.705)
        allowed = np.zeros((84, 84), dtype=bool)
        for row in range(11, 71):
            left = float(np.interp(row, rows[::-1], lefts[::-1]))
            right = float(np.interp(row, rows[::-1], rights[::-1]))
            lo, hi = max(1, int(left + 2)), min(83, int(right - 1))
            allowed[row, lo:hi] = True
        pixels = bright & allowed
        seen = np.zeros((84, 84), dtype=bool)
        candidates = []
        for y0, x0 in zip(*np.nonzero(pixels)):
            if seen[y0, x0]:
                continue
            seen[y0, x0] = True
            stack, points = [(int(y0), int(x0))], []
            while stack:
                y, x = stack.pop()
                points.append((y, x))
                for ny, nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if 0 <= ny < 84 and 0 <= nx < 84 and pixels[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx] = True
                        stack.append((ny,nx))
            if not 4 <= len(points) <= 48:
                continue
            ys, xs = np.asarray(points).T
            width, height = int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)
            aspect = width * self.PX_Y / (height * self.PX_X)
            if not (2 <= width <= 8 and 2 <= height <= 9 and 0.65 <= aspect <= 1.9):
                continue
            y, x = float(ys.mean()), float(xs.mean())
            surrounding = []
            for py, px in ((y,x-4),(y,x+4),(y-5,x),(y+5,x)):
                iy, ix = int(round(py)), int(round(px))
                surrounding.append(0 <= iy < 73 and 0 <= ix < 84 and 0.32 <= frame[iy,ix] <= 0.51)
            if sum(surrounding) < 3:
                continue
            left = float(np.interp(y, rows[::-1], lefts[::-1]))
            right = float(np.interp(y, rows[::-1], rights[::-1]))
            candidates.append((y,x,left,right))
        return max(candidates,key=lambda item:item[0]) if candidates else None

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
            if self.last_obstacle_y is not None and row < self.last_obstacle_y - 14.0:
                self.pass_side = 0.0
            self.last_obstacle_y = row
            self.pass_missing = 0
            if self.pass_side == 0.0:
                self.pass_side = -1.0 if x - left >= right - x else 1.0
            obstacle_distance = max(2.0, (self.CAR_Y - row) / self.PX_Y)
            if obstacle_distance < 7.0:
                route_row = self.CAR_Y - self.PX_Y * 7.0
                left = float(np.interp(route_row, rows[::-1], lefts[::-1]))
                right = float(np.interp(route_row, rows[::-1], rights[::-1]))
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
                gas = min(gas, 0.30)
            brake = 0.0
        return np.asarray([self.last_steer, gas, brake], dtype=np.float32)


class Agent(_Camera):
    """Camera pursuit with minimum necessary obstacle displacement.

    Curve propulsion depends on tire demand, not an absolute steering cutoff.
    Inference uses only the latest image and past camera commands.
    """

    def __init__(self, cruise_speed=120., lateral_accel=200.,
                 preview_time=.19, steer_gain=1., pass_clearance=2.7,
                 braking_accel=110., gas_reserve=.85):
        values = np.asarray([pass_clearance, braking_accel, gas_reserve])
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("passing and propulsion parameters must be positive")
        self.pass_clearance = float(pass_clearance)
        self.braking_accel = float(braking_accel)
        self.gas_reserve = float(gas_reserve)
        super().__init__(cruise_speed, lateral_accel, preview_time, steer_gain)

    def reset(self, observation=None):
        super().reset(observation)
        self.pass_offset = 0.
        self.pass_distance = None

    def _pass_target(self, frame, road, speed, preview):
        forward, lateral, rows, lefts, rights = road
        obstacle = self._obstacle(frame, road)
        if obstacle is not None:
            row, column, left, right = obstacle
            distance = (self.CAR_Y - row) / self.PX_Y
            if self.pass_distance is not None and distance > self.pass_distance + 9.:
                self.pass_side = 0.
            center = .5 * (left + right)
            clearance_px = self.pass_clearance * self.PX_X
            candidates = []
            for side in (-1., 1.):
                candidate = column + side * clearance_px
                if left + 2.4 <= candidate <= right - 2.4:
                    cost = abs(candidate - center) + .15 * abs(candidate - self.CAR_X)
                    if self.pass_side == side:
                        cost -= 1.
                    candidates.append((cost, side, candidate))
            if candidates:
                _, self.pass_side, destination = min(candidates)
            else:
                self.pass_side = -1. if column - left >= right - column else 1.
                destination = float(np.clip(column + self.pass_side * clearance_px,
                                            left + 2.4, right - 2.4))
            self.pass_offset = (destination - center) / self.PX_X
            self.pass_distance = distance
            self.pass_missing = 0
        elif self.pass_distance is not None:
            self.pass_distance -= .08 * speed
            self.pass_missing += 1
            if self.pass_distance < -6.:
                self.pass_distance = None
                self.pass_offset = self.pass_side = 0.
        path = lateral.copy()
        if self.pass_distance is not None:
            # A smooth shift starts while the obstacle is still distant.
            sigma = max(9., .20 * speed)
            weight = np.exp(-.5 * ((forward - self.pass_distance) / sigma) ** 2)
            path += self.pass_offset * weight
        return path

    def act(self, observation):
        frame = self._frame(observation)
        road = None if frame is None else self._road(frame)
        if road is None:
            return self._recover()
        self.lost_frames = 0
        forward, lateral, _, _, _ = road
        speed = self._speed(frame)
        self.last_speed = speed
        self.last_center = self.CAR_X + self.PX_X * float(np.interp(5., forward, lateral))
        preview = float(np.clip(5.5 + self.preview_time * speed, 7., 29.))
        preview = min(preview, max(6., float(forward[-1]) - 1.))
        path = self._pass_target(frame, road, speed, preview)
        if self.pass_distance is not None and self.pass_distance >= 0.:
            preview = min(preview, max(5., self.pass_distance))
        # Use the same route for steering and speed demand. Duplicate fitted
        # curvature caps can disagree with the actual pursuit on pixel roads.
        x_target = float(np.interp(preview, forward, path))
        curvature = 2. * x_target / (preview ** 2 + x_target ** 2)
        desired = self.steer_gain * float(np.arctan(self.WHEELBASE * curvature))
        desired = float(np.clip(desired, -.4, .4))
        self.last_steer += float(np.clip(desired - self.last_steer, -.24, .24))
        tangent = abs(float(np.tan(self.last_steer)))
        target = min(self.cruise_speed,
                     np.sqrt(self.lateral_accel * self.WHEELBASE / max(tangent, .001)))
        if forward[-1] < 19.:
            visible = max(0., float(forward[-1]) - 4. - .08 * speed)
            target = min(target, np.sqrt(48. ** 2 + 2. * self.braking_accel * visible))
        target = min(target, self.last_target + 5.)
        self.last_target = float(target)
        error = target - speed
        if error < -2.:
            gas, brake = 0., float(np.clip(-error * .025, .04, .85))
        else:
            gas, brake = float(np.clip(.35 + .065 * error, 0., 1.)), 0.
            demand = speed ** 2 * tangent / (self.WHEELBASE * self.lateral_accel)
            # Rear-wheel propulsion shares the same friction circle as
            # cornering. Full gas at 100 m/s spins the calibrated car.
            if tangent > .008:
                force_ratio = min(1., speed ** 2 * tangent / (self.WHEELBASE * 210.))
                budget = .008 * speed * np.sqrt(max(0., 1. - force_ratio ** 2))
                budget = max(.32 if speed < 55. else .10, budget)
                gas = min(gas, float(budget * self.gas_reserve / .85))
        return np.asarray([self.last_steer, gas, brake], dtype=np.float32)
