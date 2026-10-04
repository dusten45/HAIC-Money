"""Small standalone camera racer with one curve-relative navigation path.

All geometry comes from the current 84x84 grayscale camera. Obstacle routes
offset the measured road centerline, then check the rotated hull on measured
road edges. This is a research candidate, not a certified submission.
"""

import numpy as np


class Agent:
    PX_X = 1.3608
    PX_Y = 1.701
    CAR_X = 42.0
    CAR_Y = 63.0
    WHEELBASE = 3.24
    HALF_ROAD = 6.6666666667
    HULL_HALF_WIDTH = 1.25
    HULL_HALF_LENGTH = 2.60

    def __init__(self, cruise_speed=95.0, lateral_accel=125.0,
                 preview_time=0.16, braking_accel=95.0,
                 obstacle_speed=48.0, clearance=0.35):
        values = np.asarray([cruise_speed, lateral_accel, preview_time,
                             braking_accel, obstacle_speed, clearance])
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("parameters must be finite and positive")
        self.cruise_speed = float(min(cruise_speed, 130.0))
        self.lateral_accel = float(lateral_accel)
        self.preview_time = float(preview_time)
        self.braking_accel = float(braking_accel)
        self.obstacle_speed = float(obstacle_speed)
        self.clearance = float(clearance)
        self.reset()

    def reset(self, observation=None):
        self.last_steer = 0.0
        self.last_center = self.CAR_X
        self.last_speed = 0.0
        self.last_target = self.cruise_speed
        self.lost_frames = 0
        self.pass_side = 0
        self.last_obstacle = None
        self.obstacle_missing = 0
        self.last_plan = None

    @staticmethod
    def _frame(observation):
        try:
            pixels = np.asarray(observation)
            if pixels.ndim == 3 and pixels.shape[1:] == (84, 84):
                pixels = pixels[-1]
            if pixels.shape != (84, 84) or not np.isfinite(pixels).all():
                return None
            return pixels.astype(np.float32, copy=False)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _speed(frame):
        return float(np.clip((float(np.sum(frame[74:83, 10:13])) - 0.20)
                             / 0.0882, 0.0, 180.0))

    def _road(self, frame):
        asphalt = (frame >= 0.32) & (frame <= 0.51)
        rows, lefts, rights, centers = [], [], [], []
        previous, previous_y, slope = self.CAR_X, 69.0, 0.0
        for row in range(69, 3, -2):
            points = np.flatnonzero(asphalt[row])
            if points.size < 5:
                continue
            cuts = np.flatnonzero(np.diff(points) > 6) + 1
            runs = [run for run in np.split(points, cuts) if run.size >= 5]
            if not runs:
                continue
            expected = previous + slope * (previous_y - row)
            run = min(runs, key=lambda p: abs((p[0] + p[-1]) * 0.5 - expected))
            left, right = float(run[0]), float(run[-1])
            center = (left + right) * 0.5
            if abs(center - expected) > 22.0:
                continue
            half_width = self.HALF_ROAD * self.PX_X * np.sqrt(
                1.0 + (slope * self.PX_Y / self.PX_X) ** 2)
            if left == 0.0 and right < 83.0:
                center, left = right - half_width, right - 2.0 * half_width
            elif right == 83.0 and left > 0.0:
                center, right = left + half_width, left + 2.0 * half_width
            if rows:
                derivative = (center - previous) / max(previous_y - row, 1.0)
                slope = 0.5 * slope + 0.5 * float(np.clip(derivative, -2.0, 2.0))
            rows.append(float(row))
            lefts.append(left)
            rights.append(right)
            centers.append(center)
            previous, previous_y = center, float(row)
        if len(rows) < 5:
            return None
        rows = np.asarray(rows)
        return ((self.CAR_Y - rows) / self.PX_Y,
                (np.asarray(centers) - self.CAR_X) / self.PX_X,
                rows, np.asarray(lefts), np.asarray(rights))

    def _obstacles(self, frame, road):
        forward, _, rows, lefts, rights = road
        bright = (frame >= 0.655) & (frame <= 0.705)
        allowed = np.zeros((84, 84), dtype=bool)
        for row in range(max(10, int(rows[-1])), 72):
            left = float(np.interp(row, rows[::-1], lefts[::-1]))
            right = float(np.interp(row, rows[::-1], rights[::-1]))
            allowed[row, max(0, int(left + 1)):min(84, int(right))] = True
        pixels = bright & allowed
        seen = np.zeros_like(pixels)
        obstacles = []
        for y0, x0 in zip(*np.nonzero(pixels)):
            if seen[y0, x0]:
                continue
            seen[y0, x0] = True
            stack, points = [(int(y0), int(x0))], []
            while stack:
                row, col = stack.pop()
                points.append((row, col))
                for ny, nx in ((row-1, col), (row+1, col),
                               (row, col-1), (row, col+1)):
                    if (0 <= ny < 84 and 0 <= nx < 84 and pixels[ny, nx]
                            and not seen[ny, nx]):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if not 4 <= len(points) <= 48:
                continue
            ys, xs = np.asarray(points).T
            width, height = int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)
            aspect = width * self.PX_Y / (height * self.PX_X)
            if not (2 <= width <= 8 and 2 <= height <= 9 and 0.65 <= aspect <= 1.9):
                continue
            row, col = float(ys.mean()), float(xs.mean())
            surrounds = 0
            for py, px in ((row, col-4), (row, col+4), (row-5, col), (row+5, col)):
                iy, ix = int(round(py)), int(round(px))
                surrounds += (0 <= iy < 73 and 0 <= ix < 84
                              and 0.32 <= frame[iy, ix] <= 0.51)
            if surrounds < 3:
                continue
            distance = (self.CAR_Y - row) / self.PX_Y
            if forward[0] <= distance <= forward[-1]:
                # The sprite shape varies with rasterization; the physical
                # circle radius is fixed, not inferred from a 2px edge.
                obstacles.append(((col-self.CAR_X)/self.PX_X, distance, 1.2))
        return sorted(obstacles, key=lambda item: item[1])

    def _hull_margin(self, forward, lateral, road, relevant_end):
        ahead, _, _, lefts, rights = road
        lefts = (lefts-self.CAR_X)/self.PX_X
        rights = (rights-self.CAR_X)/self.PX_X
        derivative = np.gradient(lateral, forward)
        heading = np.arctan(derivative)
        relevant = (forward >= 0.0) & (forward <= relevant_end)
        margin = float("inf")
        for longitudinal in (-self.HULL_HALF_LENGTH, self.HULL_HALF_LENGTH):
            for transverse in (-self.HULL_HALF_WIDTH-self.clearance,
                               self.HULL_HALF_WIDTH+self.clearance):
                corner_y = forward + longitudinal*np.cos(heading) - transverse*np.sin(heading)
                corner_x = lateral + longitudinal*np.sin(heading) + transverse*np.cos(heading)
                observed = relevant & (corner_y >= ahead[0]) & (corner_y <= ahead[-1])
                if observed.any():
                    margin = min(margin, float(np.min(corner_x[observed]
                        - np.interp(corner_y[observed], ahead, lefts))),
                        float(np.min(np.interp(corner_y[observed], ahead, rights)
                        - corner_x[observed])))
        return margin

    def _plan_path(self, road, obstacles, speed):
        ahead, centers, _, _, _ = road
        forward = np.arange(0.0, max(3.0, float(ahead[-1])) + 0.1, 0.75)
        lateral = np.interp(forward, ahead, centers)
        for index, distance in enumerate(forward):
            weights = np.exp(-0.5*((ahead-distance)/5.0)**2)
            lateral[index] = np.polyfit(ahead-distance, centers, 2,
                                        w=weights)[2]
        # Begin at the observed ego pose and heading. A lateral road or lane
        # offset cannot teleport the current hull into a safe future lane.
        anchoring_distance = float(np.clip(0.28*speed, 12.0, 16.0))
        phase = np.clip(forward/anchoring_distance, 0.0, 1.0)
        anchor = phase**3*(10.0-15.0*phase+6.0*phase**2)
        base = lateral*anchor
        result = {"forward": forward, "lateral": base,
                  "blocked": False, "side": 0, "margin": None,
                  "obstacle_distance": None}
        if not obstacles:
            return result
        ox, oy, radius = obstacles[0]
        result["obstacle_distance"] = oy
        if oy < -self.HULL_HALF_LENGTH-radius:
            return result
        local_center = float(np.interp(oy, forward, lateral))
        slope = float(np.interp(oy, forward, np.gradient(lateral, forward)))
        required = (radius+self.HULL_HALF_WIDTH+self.clearance)*np.sqrt(1.0+slope*slope)
        approach = max(13.0, min(24.0, 0.28*speed))
        plateau = radius+self.HULL_HALF_LENGTH+0.5
        exit_distance = 9.0
        rise = np.clip((forward-(oy-plateau-approach))/approach, 0.0, 1.0)
        fall = np.clip((oy+plateau+exit_distance-forward)/exit_distance, 0.0, 1.0)
        smooth = lambda value: value*value*(3.0-2.0*value)
        weight = smooth(rise)*smooth(fall)
        candidates = []
        for side in (-1, 1):
            offset = ox-local_center+side*required
            path = (lateral+offset*weight)*anchor
            slope_path = np.gradient(path, forward)
            curvature = np.abs(np.gradient(slope_path, forward))/(1.0+slope_path*slope_path)**1.5
            relevant = forward <= min(float(ahead[-1])-1.0, max(5.0, oy+plateau+exit_distance))
            if np.any(curvature[relevant] > np.tan(0.4)/self.WHEELBASE):
                continue
            heading = np.arctan(slope_path)
            dx, dy = ox-path, oy-forward
            across = np.abs(dx*np.cos(heading)-dy*np.sin(heading))
            along = np.abs(dx*np.sin(heading)+dy*np.cos(heading))
            hull_gap = np.hypot(np.maximum(across-self.HULL_HALF_WIDTH, 0.0),
                                np.maximum(along-self.HULL_HALF_LENGTH, 0.0))
            if np.any(hull_gap[relevant] < radius+self.clearance-0.05):
                continue
            available = max(0.0, oy-radius-self.HULL_HALF_LENGTH-0.08*speed)
            pass_x = ox+side*required
            needed_curve = abs(2.0*pass_x/(available*available+pass_x*pass_x+0.01))
            if needed_curve > np.tan(0.4)/self.WHEELBASE:
                continue
            pass_limit = np.sqrt(self.lateral_accel/max(needed_curve, 0.001))
            brake_distance = max(0.0, (speed*speed-pass_limit*pass_limit)/(2.0*self.braking_accel))
            if brake_distance > available:
                continue
            margin = self._hull_margin(forward, path, road,
                min(float(ahead[-1])-1.0, max(5.0, oy+plateau+exit_distance)))
            if margin >= 0.0:
                # Stay with a feasible selected side, otherwise choose the
                # least centerline displacement with the better edge reserve.
                cost = abs(offset)-0.2*min(margin, 2.0)
                if self.pass_side == side:
                    cost -= 1.5
                candidates.append((cost, side, path, margin))
        if not candidates:
            result["blocked"] = True
            result["lateral"] = base
            return result
        _, side, path, margin = min(candidates, key=lambda item: item[0])
        result.update(lateral=path, side=side, margin=margin)
        return result

    def _curve_speed(self, forward, lateral):
        limit = self.cruise_speed
        # Local fits smooth raster staircase curvature without collapsing an
        # approaching bend and a straight into a single global parabola.
        for distance in np.arange(3.0, min(32.0, float(forward[-1]))+0.1, 3.0):
            use = (forward >= distance-5.0) & (forward <= distance+5.0)
            if np.count_nonzero(use) < 5:
                continue
            coefficients = np.polyfit(forward[use]-distance, lateral[use], 2)
            curvature = abs(2.0*coefficients[0])/(1.0+coefficients[1]**2)**1.5
            local_limit = np.sqrt(self.lateral_accel/max(curvature, 0.001))
            approach = np.sqrt(local_limit**2+2.0*self.braking_accel*max(0.0, distance-5.0))
            limit = min(limit, approach)
        return float(limit)

    def _avoidance_steer(self, desired, obstacle, side):
        """Necessary constant-curvature approach clearance for the front hull.

        Local pursuit can aim inside a bend before its later offset becomes
        the preview point. Restrict that command when it projects across the
        chosen circle side. This is a reachability bound, not a slip model.
        """
        ox, oy, radius = obstacle
        if side == 0 or oy <= radius+self.HULL_HALF_LENGTH:
            return desired
        entry = max(3.0, oy-radius-self.HULL_HALF_LENGTH)
        required = radius+self.HULL_HALF_WIDTH+self.clearance
        pass_x = ox+side*required
        curve = 2.0*pass_x/(entry*entry)
        bound = float(np.arctan(self.WHEELBASE*curve))
        return max(desired, bound) if side > 0 else min(desired, bound)

    def _recover(self):
        self.lost_frames += 1
        self.last_obstacle = None
        self.obstacle_missing = 0
        desired = float(np.clip((self.last_center-self.CAR_X)*0.025, -0.35, 0.35))
        if abs(desired) < 0.04:
            desired = self.last_steer*0.6
        self.last_steer += float(np.clip(desired-self.last_steer, -0.24, 0.24))
        gas, brake = (0.0, 0.25) if self.lost_frames <= 3 else (0.09, 0.0)
        return np.asarray([self.last_steer, gas, brake], dtype=np.float32)

    def act(self, observation):
        frame = self._frame(observation)
        road = self._road(frame) if frame is not None else None
        if road is None or road[0][-1] < 6.0:
            return self._recover()
        self.lost_frames = 0
        speed = self._speed(frame)
        self.last_speed = speed
        self.last_center = self.CAR_X+self.PX_X*float(np.interp(4.0, road[0], road[1]))
        obstacles = self._obstacles(frame, road)
        if obstacles:
            if self.last_obstacle is not None and obstacles[0][1] > self.last_obstacle[1]+10.0:
                self.pass_side = 0
            self.last_obstacle, self.obstacle_missing = obstacles[0], 0
        elif self.last_obstacle is not None and self.obstacle_missing < 3:
            self.obstacle_missing += 1
            x, distance, radius = self.last_obstacle
            self.last_obstacle = (x, distance-speed*0.08, radius)
            if self.last_obstacle[1] > -radius-self.HULL_HALF_LENGTH:
                obstacles = [self.last_obstacle]
        else:
            self.last_obstacle, self.pass_side = None, 0
        plan = self._plan_path(road, obstacles, speed)
        self.last_plan = plan
        self.pass_side = plan["side"] if obstacles else self.pass_side
        forward, lateral = plan["forward"], plan["lateral"]
        preview = float(np.clip(5.5+self.preview_time*speed, 7.0, 26.0))
        preview = min(preview, max(4.0, float(forward[-1])-1.0))
        target_x = float(np.interp(preview, forward, lateral))
        pursuit = 2.0*target_x/(preview*preview+target_x*target_x)
        desired = float(np.clip(np.arctan(self.WHEELBASE*pursuit), -0.4, 0.4))
        if obstacles and not plan["blocked"]:
            desired = float(np.clip(self._avoidance_steer(desired,
                obstacles[0], plan["side"]), -0.4, 0.4))
        self.last_steer += float(np.clip(desired-self.last_steer, -0.24, 0.24))
        target = self._curve_speed(forward, lateral)
        if forward[-1] < 22.0:
            target = min(target, np.sqrt(35.0**2+2.0*self.braking_accel*max(0.0, forward[-1]-7.0)))
        distance = plan["obstacle_distance"]
        if distance is not None:
            pass_speed = 0.0 if plan["blocked"] else self.obstacle_speed
            target = min(target, np.sqrt(pass_speed**2+2.0*self.braking_accel*max(0.0, distance-5.5)))
            if obstacles and plan["side"]:
                ox, oy, radius = obstacles[0]
                needed = radius+self.HULL_HALF_WIDTH+self.clearance+plan["side"]*ox
                if needed > 0.1:
                    entry = max(0.0, oy-radius-self.HULL_HALF_LENGTH)
                    reach_speed = np.sqrt(0.5*self.lateral_accel*entry*entry/needed)
                    target = min(target, max(10.0, reach_speed))
        tangent = abs(np.tan(self.last_steer))
        physical_ceiling = np.sqrt(self.lateral_accel*self.WHEELBASE/max(tangent, 0.001))
        target = min(target, physical_ceiling, self.last_target+3.0)
        self.last_target = float(target)
        error = target-speed
        if error < -2.0 or (plan["blocked"] and distance is not None and distance < 15.0):
            gas, brake = 0.0, float(np.clip(-error*0.02, 0.08, 0.75))
        else:
            gas, brake = float(np.clip(0.28+error*0.06, 0.0, 1.0)), 0.0
            demand = speed*speed*tangent/(self.WHEELBASE*self.lateral_accel)
            if demand > 0.75:
                gas = min(gas, 0.16)
            elif demand > 0.5:
                gas = min(gas, 0.25)
        return np.asarray([self.last_steer, gas, brake], dtype=np.float32)
