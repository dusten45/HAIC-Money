"""Independent metric camera racer; inference needs only NumPy.

The observation camera follows the hull with an affine overhead transform.
Road pixels therefore supply distances, heading and local curvature directly;
no episode identity, simulator import, map or learned checkpoint is used.
"""

import numpy as np


class _CameraBase:
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


class Agent(_CameraBase):
    """Pixel-only short-horizon bicycle planner with visible-road constraints."""

    def __init__(self, cruise_speed=85.0, lateral_accel=140.0,
                 preview_time=0.19, steer_gain=1.0, horizon=0.4,
                 steer_samples=17, slip_margin=0.35):
        if not 0.12 <= horizon <= 0.8 or not 9 <= steer_samples <= 33:
            raise ValueError("planner horizon or steer sample count outside bounds")
        if not np.isfinite(slip_margin) or slip_margin < 0:
            raise ValueError("slip margin must be finite and nonnegative")
        self.horizon = float(horizon)
        self.steer_samples = int(steer_samples)
        self.slip_margin = float(slip_margin)
        super().__init__(cruise_speed, lateral_accel, preview_time, steer_gain)

    def reset(self, observation=None):
        super().reset(observation)
        self._planner_obstacle = None
        self._planner_missing = 0
        self.last_plan_feasible = False
        self.last_plan_violation = 0.0
        self._planner_brake_frames = 0

    @staticmethod
    def _yaw(frame):
        # The legal red HUD bar is1.68pixels wide per radian/second.
        crop = frame[76:80, 54:75]
        selected = (crop >= 0.22) & (crop <= 0.37)
        weights = np.where(selected, crop, 0.0)
        mass = float(weights.sum())
        if mass < 0.03:
            return 0.0
        mean = float((weights * np.arange(54, 75)[None, :]).sum() / mass)
        yaw = mass / (4.0 * 0.299 * 1.68)
        return float(np.clip(yaw if mean < 62.5 else -yaw, -8.0, 8.0))

    def _violations(self, x, y, heading, road, obstacle):
        forward, _centers, _rows, lefts, rights = road
        lefts = (lefts - self.CAR_X) / self.PX_X
        rights = (rights - self.CAR_X) / self.PX_X
        x, y, heading = np.asarray(x), np.asarray(y), np.asarray(heading)
        violations = np.zeros_like(x, dtype=float)
        # All four hull corners, including the nose and rear, need asphalt.
        for longitudinal in (-2.5, 2.6):
            for lateral in (-1.25, 1.25):
                px = x + np.sin(heading) * longitudinal + np.cos(heading) * lateral
                py = y + np.cos(heading) * longitudinal - np.sin(heading) * lateral
                left = np.interp(py, forward, lefts) + self.slip_margin
                right = np.interp(py, forward, rights) - self.slip_margin
                outside = np.maximum(left-px, 0.0) + np.maximum(px-right, 0.0)
                invisible = np.maximum(forward[0]-py, 0.0) + np.maximum(py-forward[-1], 0.0)
                violations = np.maximum(violations, outside + invisible)
        if obstacle is not None:
            ox, oy, radius = obstacle
            # Circle against the rotated hull rectangle, retaining margin.
            # Expanding a world-axis box falsely blocks clear front corners.
            local_x = np.cos(heading)*(ox-x)-np.sin(heading)*(oy-y)
            local_y = np.sin(heading)*(ox-x)+np.cos(heading)*(oy-y)
            dx = np.maximum(np.abs(local_x)-1.25,0.0)
            dy = np.maximum(np.maximum(local_y-2.6,-2.5-local_y),0.0)
            distance = np.sqrt(dx*dx+dy*dy)
            overlap = np.maximum(radius+self.slip_margin-distance,0.0)
            violations = np.maximum(violations, np.where(overlap>0,overlap+1.0,0.0))
        return violations

    def _trajectory_clear(self, x, y, heading, road, obstacle):
        return bool(np.all(self._violations(x, y, heading, road, obstacle) <= 0.03))

    def _rollouts(self, road, speed, proposal, previous, yaw, obstacle):
        steer_limit = min(0.4, float(np.arctan(self.WHEELBASE * self.lateral_accel / max(speed**2, 100.0))))
        commands = np.unique(np.r_[np.linspace(-steer_limit, steer_limit, self.steer_samples), proposal, previous, 0.0])
        commands = np.clip(commands, max(-steer_limit, previous-0.24), min(steer_limit, previous+0.24))
        # Evaluate maintaining speed, moderate braking and full road braking.
        # Start-up rollouts include acceleration so zero-speed planning moves.
        acceleration_modes = (35.0,6.0,-60.0,-120.0) if speed < 12.0 else (0.0,-60.0,-120.0)
        command_count = commands.size
        commands = np.tile(commands, len(acceleration_modes))
        accelerations = np.repeat(acceleration_modes, command_count)
        count = commands.size
        x = np.zeros(count)
        y = np.zeros(count)
        heading = np.zeros(count)
        velocity = np.full(count, speed,dtype=float)
        wheel = np.full(count, previous,dtype=float)
        yawrate = np.full(count, -yaw,dtype=float)
        cost = np.zeros(count)
        violation = np.zeros(count)
        dt = 0.02
        horizon = min(self.horizon, max(0.12, (road[0][-1]-3.2) / max(speed, 25.0)))
        steps = max(6, int(horizon/dt))
        side = self.pass_side
        if obstacle is not None and side == 0.0:
            side = -1.0 if obstacle[0] >= 0.0 else 1.0
        for tick in range(steps):
            if tick < 4:
                future_command = commands
            else:
                preview = np.clip(3.5+0.14*velocity,7.0,17.0)
                target_y = y+preview
                target_x = np.interp(target_y,road[0],road[1])
                if obstacle is not None:
                    passing = (y <= obstacle[1]+4.0) & (obstacle[1]-y < 34.0)
                    target_x += np.where(passing,4.0*side,0.0)
                left = (np.interp(target_y,road[0],road[3])-self.CAR_X)/self.PX_X+1.9
                right = (np.interp(target_y,road[0],road[4])-self.CAR_X)/self.PX_X-1.9
                target_x = np.clip(target_x,left,right)
                dx,dy = target_x-x,target_y-y
                cross = np.cos(heading)*dx-np.sin(heading)*dy
                future_command = np.arctan(self.WHEELBASE*2.0*cross/(dx*dx+dy*dy+1e-6))
                future_limit = np.minimum(0.4,np.arctan(self.WHEELBASE*self.lateral_accel/np.maximum(velocity*velocity,100.0)))
                future_command = np.clip(future_command,-future_limit,future_limit)
            wheel += np.clip(future_command-wheel, -3.0*dt, 3.0*dt)
            velocity = np.maximum(0.0, velocity + accelerations*dt)
            desired_yaw = velocity * np.tan(wheel) / self.WHEELBASE
            yawrate += 0.35 * (desired_yaw-yawrate)
            heading += yawrate*dt
            x += np.sin(heading) * velocity*dt
            y += np.cos(heading) * velocity*dt
            violation = np.maximum(violation, self._violations(x,y,heading,road,obstacle))
            middle = np.interp(y, road[0], road[1])
            if obstacle is not None:
                middle += np.where((y<=obstacle[1]+4.0)&(obstacle[1]-y<34.0),4.0*side,0.0)
            cost += (x-middle)**2 / steps
        far_middle = np.interp(y+2.0, road[0], road[1])
        near_middle = np.interp(y-2.0, road[0], road[1])
        road_heading = np.arctan2(far_middle-near_middle, 4.0)
        cost += 8.0*(heading-road_heading)**2 + 8.0*(commands-proposal)**2
        cost += 4.0*(commands-previous)**2 - 0.7*y
        cost += 0.025 * np.maximum(-accelerations,0.0)
        feasible = violation <= 0.03
        score = cost + np.where(feasible, 0.0, 10000.0 + 2000.0*violation)
        if self._planner_brake_frames >= 3 and speed < 12.0:
            safe_progress = feasible & (accelerations>0.0)
            if np.any(safe_progress):
                score = np.where(safe_progress,cost,np.inf)
        selected = int(np.argmin(score))
        return float(commands[selected]), float(accelerations[selected]), bool(feasible[selected]), float(violation[selected]), float(velocity[selected])

    def act(self, observation):
        previous = self.last_steer
        base = super().act(observation)
        frame = self._frame(observation)
        if frame is None:
            return base
        road = self._road(frame)
        if road is None:
            return base
        detected = self._obstacle(frame, road)
        speed = self.last_speed
        if detected is not None:
            row, x, _left, _right = detected
            self._planner_obstacle = ((x-self.CAR_X)/self.PX_X, (self.CAR_Y-row)/self.PX_Y, 1.2)
            self._planner_missing = 0
        elif self._planner_obstacle is not None:
            ox, oy, radius = self._planner_obstacle
            self._planner_missing += 1
            self._planner_obstacle = (ox, oy-speed*0.08, radius)
            if self._planner_missing > 3 or self._planner_obstacle[1] < -4.0:
                self._planner_obstacle = None
        steering, acceleration, feasible, violation, terminal_speed = self._rollouts(
            road, speed, float(base[0]), previous, self._yaw(frame), self._planner_obstacle)
        self.last_steer = steering
        self.last_plan_feasible = feasible
        self.last_plan_violation = violation
        gas, brake = float(base[1]), float(base[2])
        if not feasible:
            # Retain a path that minimizes violation while reducing motion.
            gas, brake = 0.0, 0.70 if speed > 12.0 else 0.35
            self.last_target = max(15.0,min(self.last_target,max(15.0,terminal_speed)))
        elif acceleration < 0:
            self._planner_brake_frames += 1
            gas = 0.0
            brake = max(brake, 0.35 if acceleration == -60.0 else 0.70)
            self.last_target = max(15.0,min(self.last_target,terminal_speed))
        else:
            self._planner_brake_frames = 0
            self.last_target = max(15.0,self.last_target)
            if acceleration == 6.0:
                gas,brake = 0.005,0.0
        return np.asarray([steering,gas,brake],dtype=np.float32)
