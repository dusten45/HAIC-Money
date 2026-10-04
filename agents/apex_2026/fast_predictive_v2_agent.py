"""Camera-only metric pursuit with measured yaw feedback and grip budgeting.

The HUD supplies speed and yaw.  Road and orange circles are measured from the
same grayscale camera; no simulator state, seed, map or project import is used.
The path is a steering reference, not a collision or reachability certificate.
"""
import numpy as np


class _PathReference:
    PX_X, PX_Y = 1.3608, 1.701
    CAR_X, CAR_Y = 42.0, 63.0
    WHEELBASE, HALF_ROAD = 3.24, 6.6666666667

    def __init__(self, cruise_speed=100.0, lateral_accel=190.0,
                 preview_time=.22, yaw_gain=.35, braking_accel=150.0,
                 obstacle_margin=1.20):
        values = np.asarray([cruise_speed, lateral_accel, preview_time,
                             yaw_gain, braking_accel, obstacle_margin], dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("parameters must be finite and positive")
        self.cruise_speed = min(float(cruise_speed), 110.)
        self.lateral_accel = min(float(lateral_accel), 215.)
        self.preview_time = min(float(preview_time), .6)
        self.yaw_gain = min(float(yaw_gain), 2.)
        self.braking_accel = min(float(braking_accel), 180.)
        self.obstacle_margin = min(float(obstacle_margin), 1.5)
        self.reset()

    def reset(self, observation=None):
        self.last_steer = 0.
        self.last_center = self.CAR_X
        self.last_speed = 0.
        self.last_target = self.cruise_speed
        self.last_yaw = 0.
        self.lost_frames = 0
        self.pass_side = 0
        self.pass_missing = 0
        self.pass_x = 0.
        self.pass_y = None
        self.route_distance = None
        self.spin_frames = 0

    @staticmethod
    def _frame(observation):
        try:
            frame = np.asarray(observation, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            return None
        if frame.ndim == 3 and frame.shape[1:] == (84, 84) and frame.shape[0]:
            frame = frame[-1]
        if frame.shape != (84, 84) or not np.isfinite(frame).all():
            return None
        if float(frame.min()) < 0. or float(frame.max()) > 1.:
            return None
        return frame

    @staticmethod
    def _speed(frame):
        return float(np.clip((float(frame[74:83, 10:13].sum())-.20)/.0882, 0., 180.))

    @staticmethod
    def _yaw(frame):
        # Preserve antialiased red area over the full4.2px HUD height. The
        # green wheel-angle bar ends before51, and the white gauges/text are
        # further left. A lower intensity cutoff would erase partial reds.
        crop = frame[75:81, 51:83]
        weights = np.where(crop <= .37, crop, 0.)
        mass = float(weights.sum())
        if mass < .03:
            return 0.
        middle = float((weights*np.arange(51, 83)).sum())/mass
        magnitude = mass/(4.2*.299*1.68)
        return float(np.clip(magnitude if middle >= 62.5 else -magnitude, -8., 8.))

    @staticmethod
    def _rear_speed(frame):
        # Both purple rear-wheel gauges: empirically calibrated pixel mass
        # per rad/s, then the official .54m radius. Missing bars mean no spin.
        # Row73 straddles the road/HUD boundary and is not a gauge pixel.
        crop = frame[74:83, 19:24]
        mass = float(np.where(crop <= .20, crop, 0.).sum())
        return mass/.01485*.54

    def _road(self, frame):
        asphalt = (frame >= .32) & (frame <= .51)
        rows, lefts, rights, centers = [], [], [], []
        previous, previous_y, slope = self.CAR_X, 69., 0.
        for row in range(69, 3, -2):
            points = np.flatnonzero(asphalt[row])
            if points.size < 5:
                continue
            cuts = np.flatnonzero(np.diff(points) > 6)+1
            runs = [p for p in np.split(points, cuts) if p.size >= 5]
            if not runs:
                continue
            expected = previous+slope*(previous_y-row)
            run = min(runs, key=lambda p: abs(.5*(p[0]+p[-1])-expected))
            left, right = float(run[0]), float(run[-1])
            middle = .5*(left+right)
            if abs(middle-expected) > 22.:
                continue
            half = self.HALF_ROAD*self.PX_X*np.sqrt(1.+(slope*self.PX_Y/self.PX_X)**2)
            if left == 0. and right < 83.:
                middle = right-half
            elif right == 83. and left > 0.:
                middle = left+half
            if rows:
                slope = .5*slope+.5*float(np.clip((middle-previous)/max(previous_y-row, 1.), -2., 2.))
            rows.append(float(row)); lefts.append(left); rights.append(right); centers.append(middle)
            previous, previous_y = middle, float(row)
        if len(rows) < 5:
            return None
        rows = np.asarray(rows)
        return ((self.CAR_Y-rows)/self.PX_Y,
                (np.asarray(centers)-self.CAR_X)/self.PX_X,
                rows, np.asarray(lefts), np.asarray(rights))

    def _circles(self, frame, road):
        _, _, rows, lefts, rights = road
        allowed = np.zeros((84, 84), dtype=bool)
        for row in range(10, 72):
            left = float(np.interp(row, rows[::-1], lefts[::-1]))
            right = float(np.interp(row, rows[::-1], rights[::-1]))
            allowed[row, max(0, int(left+1)):min(84, int(right))] = True
        bright = (frame >= .655) & (frame <= .705) & allowed
        seen = np.zeros_like(bright)
        circles = []
        for y0, x0 in zip(*np.nonzero(bright)):
            if seen[y0, x0]:
                continue
            seen[y0, x0] = True
            stack, pixels = [(int(y0), int(x0))], []
            while stack:
                y, x = stack.pop(); pixels.append((y, x))
                for ny, nx in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
                    if 0 <= ny < 84 and 0 <= nx < 84 and bright[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx] = True; stack.append((ny,nx))
            if not 4 <= len(pixels) <= 48:
                continue
            ys, xs = np.asarray(pixels).T
            width, height = xs.max()-xs.min()+1, ys.max()-ys.min()+1
            aspect = width*self.PX_Y/(height*self.PX_X)
            if not (2 <= width <= 8 and 2 <= height <= 9 and .65 <= aspect <= 1.9):
                continue
            x, y = float(xs.mean()), float(ys.mean())
            surrounding = 0
            for py, px in ((y,x-4), (y,x+4), (y-5,x), (y+5,x)):
                iy, ix = int(round(py)), int(round(px))
                if 0 <= iy < 73 and 0 <= ix < 84 and .32 <= frame[iy,ix] <= .51:
                    surrounding += 1
            if surrounding >= 3:
                circles.append(((self.CAR_Y-y)/self.PX_Y, (x-self.CAR_X)/self.PX_X))
        return sorted(circles)

    @staticmethod
    def _smooth(value):
        value = np.clip(value, 0., 1.)
        return value*value*(3.-2.*value)

    def _route(self, ahead, center, circles, speed, yaw):
        # First decide whether the unshifted reference actually sweeps a circle.
        # This avoids slowing for the many circles safely outside the lane.
        selected = None
        clearance = 1.2+1.2+self.obstacle_margin
        self.route_distance = None
        for distance, x in circles:
            if distance < -3.8:
                continue
            sample = max(0., distance)
            road_x = float(np.interp(sample, ahead, center))
            eps = 2.
            slope = (float(np.interp(sample+eps, ahead, center))-
                     float(np.interp(sample-eps, ahead, center)))/(2.*eps)
            normal_gap = abs(x-road_x)/np.sqrt(1.+slope*slope)
            if not self.pass_side and 0. <= distance <= max(6., 3.8+.08*speed):
                # A near circle may conflict with the road center while
                # already clearing our actual turn. Do not invent a new pass
                # across it when both current motion and preview turn away.
                curvature = yaw/max(speed, 10.)
                phase = float(np.clip(curvature*distance, -.85, .85))
                if abs(curvature) > .0001:
                    motion_x = (1.-np.sqrt(1.-phase*phase))/curvature
                else:
                    motion_x = 0.
                motion_slope = phase/np.sqrt(1.-phase*phase)
                motion_gap = abs(x-motion_x)/np.sqrt(1.+motion_slope*motion_slope)
                preview = float(np.clip(7.+self.preview_time*speed, 8., 32.))
                intent_x = float(np.interp(preview, ahead, center))
                if motion_gap >= clearance and intent_x*x <= 0.:
                    continue
            if normal_gap < clearance:
                selected = (distance, x, road_x, slope)
                break
        imminent = False
        if selected is not None:
            distance, x, road_x, slope = selected
            if self.pass_y is None or distance > self.pass_y+10.:
                self.pass_side = 0
            if not self.pass_side:
                separation = (2.4+max(self.obstacle_margin, 1.3))*np.sqrt(1.+slope*slope)
                choices = []
                for side in (-1, 1):
                    target_x = x+side*separation
                    offset = target_x-road_x
                    if abs(offset)/np.sqrt(1.+slope*slope) <= 4.95:
                        # Prefer a short move from the actual pose when both
                        # road-relative sides have room, especially in bends.
                        choices.append((abs(target_x)+.15*abs(offset), side))
                self.pass_side = min(choices)[1] if choices else (-1 if x >= 0. else 1)
            # Keep the same side while this circle moves behind the camera pose.
            self.pass_x, self.pass_y = x, distance
            self.pass_missing = 0
            imminent = abs(x) < 2.7 and distance < 3.8+.08*speed
        elif self.pass_side and self.pass_y is not None:
            self.pass_missing += 1
            angle = float(np.clip(.08*yaw, -.64, .64))
            x, y = self.pass_x, self.pass_y-.08*speed
            self.pass_x = x*np.cos(angle)-y*np.sin(angle)
            self.pass_y = x*np.sin(angle)+y*np.cos(angle)
            distance, x = self.pass_y, self.pass_x
            road_x = float(np.interp(max(0., distance), ahead, center))
            slope = 0.
            if self.pass_missing > 4 or distance < -5.:
                self.pass_side = 0; self.pass_y = None
        else:
            return center.copy(), False, None
        if not self.pass_side:
            return center.copy(), imminent, None
        distance, x = self.pass_y, self.pass_x
        self.route_distance = distance
        road_x = float(np.interp(max(0., distance), ahead, center))
        slope = (float(np.interp(distance+2., ahead, center))-
                 float(np.interp(distance-2., ahead, center)))/4.
        scale = np.sqrt(1.+slope*slope)
        separation = 2.4+max(self.obstacle_margin, 1.3)
        normal_offset = (x-road_x)/scale+self.pass_side*separation
        # A road-relative corridor rather than a straight chord across a bend.
        normal_offset = float(np.clip(normal_offset, -4.95, 4.95))
        road_slope = (np.interp(ahead+2., ahead, center)-
                      np.interp(ahead-2., ahead, center))/4.
        offsets = normal_offset*np.sqrt(1.+road_slope*road_slope)
        approach = self._smooth(np.maximum(ahead, 0.)/max(4., distance-3.))
        if distance < 4.:
            approach = np.maximum(approach, self._smooth((4.-distance)/7.))
        departure = 1.-self._smooth((ahead-distance-3.)/14.)
        return center+offsets*approach*departure, imminent, distance

    def _steering(self, ahead, path, speed, yaw):
        preview = float(np.clip(7.+self.preview_time*speed, 8., 32.))
        preview = min(preview, max(3., float(ahead[-1])-1.))
        if self.route_distance is not None:
            # Until the rear clears, the preview must remain on the pass
            # corridor rather than aim at its subsequent return to center.
            preview = min(preview, max(7., self.route_distance))
        x = float(np.interp(preview, ahead, path))
        curvature = 2.*x/(preview*preview+x*x)
        feedforward = np.arctan(self.WHEELBASE*curvature)
        feedback = self.yaw_gain*self.WHEELBASE*(speed*curvature-yaw)/max(speed, 20.)
        self.reference_curvature = curvature
        return float(np.clip(feedforward+feedback, -.4, .4))

    def _target(self, ahead, path, speed):
        target = self.cruise_speed
        for distance in (4., 10., 18., 26.):
            if distance > ahead[-1]-2.:
                continue
            selected = (ahead >= max(-2., distance-7.)) & (ahead <= distance+7.)
            if np.count_nonzero(selected) < 5:
                continue
            local = ahead[selected]-distance
            coef = np.polyfit(local, path[selected], 2,
                              w=np.exp(-.5*(local/5.)**2))
            curvature = abs(2.*coef[0])/(1.+coef[1]*coef[1])**1.5
            curve_speed = np.sqrt(self.lateral_accel/max(curvature, .0005))
            usable = max(0., distance-2.6-.08*speed)
            target = min(target, np.sqrt(curve_speed**2+2.*self.braking_accel*usable))
        visibility = max(0., float(ahead[-1])-2.6-.08*speed)
        target = min(target, np.sqrt(45.**2+2.*self.braking_accel*visibility))
        return float(target)

    def _recover(self, speed=None):
        self.lost_frames += 1
        if speed is not None:
            self.last_speed = speed
        desired = float(np.clip((self.last_center-self.CAR_X)*.025, -.3, .3))
        if abs(desired) < .025:
            desired = .6*self.last_steer
        self.last_steer += float(np.clip(desired-self.last_steer, -.24, .24))
        brake = .35 if self.last_speed > 12. or self.lost_frames <= 3 else 0.
        gas = .08 if brake == 0. and self.lost_frames <= 25 else 0.
        return np.asarray([self.last_steer, gas, brake], dtype=np.float32)


class _CorridorReference(_PathReference):
    """Research route minimizing local bending inside visible camera spans.

    Box and circle constraints describe the parsed image, not guaranteed
    vehicle motion. Infeasible observations retain the original route.
    """

    def reset(self, observation=None):
        super().reset(observation)
        self.corridor_road = None
        self.corridor_used = False

    def _road(self, frame):
        result = super()._road(frame)
        self.corridor_road = result
        return result

    def _route(self, ahead, center, circles, speed, yaw):
        original, imminent, distance = super()._route(ahead, center, circles, speed, yaw)
        self.corridor_used = False
        road = self.corridor_road
        if road is None or ahead[-1] < 12. or imminent:
            return original, imminent, distance
        y = np.linspace(0., float(ahead[-1]), 36)
        middle = np.interp(y, ahead, center)
        slope = (np.interp(y+2., ahead, center)-np.interp(y-2., ahead, center))/4.
        inset = 1.9*np.sqrt(1.+slope*slope)
        lower = (np.interp(y, ahead, road[3])-self.CAR_X)/self.PX_X+inset
        upper = (np.interp(y, ahead, road[4])-self.CAR_X)/self.PX_X-inset
        for object_y, object_x in circles:
            if object_y < -4. or object_y > y[-1]+4.:
                continue
            if distance is not None and abs(object_y-distance) < 1.:
                side = self.pass_side
            else:
                side = -1 if object_x >= np.interp(object_y, ahead, original) else 1
            extent = 3.8*np.sqrt(np.maximum(0., 1.-((y-object_y)/4.5)**2))
            active = np.abs(y-object_y) < 4.5
            if side < 0:
                upper[active] = np.minimum(upper[active], object_x-extent[active])
            else:
                lower[active] = np.maximum(lower[active], object_x+extent[active])
        if np.any(lower > upper) or np.any(lower[:2] > 0.) or np.any(upper[:2] < 0.):
            return original, imminent, distance
        # Position and heading are fixed at the actual camera ego pose. The
        # remaining convex box problem uses a weak centering preference.
        lower[:2] = upper[:2] = 0.
        count = len(y)
        identity = np.eye(count)
        d1, d2 = np.diff(identity, axis=0), np.diff(identity, n=2, axis=0)
        hessian = 30.*(d2.T@d2)+.3*(d1.T@d1)+.04*identity
        linear = .04*middle
        step = 1./float(np.max(np.sum(np.abs(hessian), axis=1)))
        x = np.clip(np.interp(y, ahead, original), lower, upper)
        extrapolated, momentum = x.copy(), 1.
        for _ in range(180):
            updated = np.clip(extrapolated-step*(hessian@extrapolated-linear), lower, upper)
            next_momentum = .5*(1.+np.sqrt(1.+4.*momentum*momentum))
            extrapolated = updated+(momentum-1.)/next_momentum*(updated-x)
            x, momentum = updated, next_momentum
        if not np.isfinite(x).all():
            return original, imminent, distance
        self.corridor_used = True
        result = np.interp(ahead, y, x)
        result[ahead < 0.] = 0.
        return result, imminent, distance

    def act(self, observation):
        frame = self._frame(observation)
        if frame is None:
            return self._recover()
        speed, yaw = self._speed(frame), self._yaw(frame)
        road = self._road(frame)
        if road is None or float(road[0][-1]) < 6.:
            return self._recover(speed)
        self.lost_frames = 0
        ahead, center = road[:2]
        self.last_center = self.CAR_X+self.PX_X*float(np.interp(5., ahead, center))
        path, imminent, obstacle_distance = self._route(ahead, center, self._circles(frame, road), speed, yaw)
        desired = self._steering(ahead, path, speed, yaw)
        self.last_steer += float(np.clip(desired-self.last_steer, -.24, .24))
        target = self._target(ahead, path, speed)
        reference_limit = np.sqrt(self.lateral_accel/max(abs(self.reference_curvature), .0005))
        target = min(target, reference_limit, self.last_target+6.)
        actual_lateral = speed*abs(yaw)
        demand = max(actual_lateral, speed*speed*abs(self.reference_curvature))
        # Powered launch slip can be productive on a straight. Tight turns
        # need rear lateral grip, so wheel-speed feedback is gated by demand.
        if self._rear_speed(frame)-speed > 15. and demand > 60.:
            self.spin_frames = 6
        else:
            self.spin_frames = max(0, self.spin_frames-1)
        if actual_lateral > self.lateral_accel:
            target = min(target, self.lateral_accel/max(abs(yaw), .05))
        if imminent:
            target = 0.
        error = target-speed
        if error < -2.:
            gas, brake = 0., float(np.clip(-error*.014, .04, .65))
        else:
            gas, brake = float(np.clip(.18+.06*error, 0., 1.)), 0.
            if demand > 120.:
                gas = min(gas, .30)
            elif demand > 60.:
                gas = min(gas, .60)
            if speed >= 95.:
                gas = min(gas, .16)
            elif speed >= 85.:
                gas = min(gas, .35)
            if self.spin_frames:
                gas = min(gas, .16 if speed > 15. else .08)
        self.last_target, self.last_speed, self.last_yaw = target, speed, yaw
        return np.asarray([self.last_steer, gas, brake], dtype=np.float32)


class _GraphReference(_CorridorReference):
    """Keep only the visible road component linked to the ego neighborhood.

    This graph connects eight neighboring asphalt pixels. It prevents a
    row scanner from jumping across grass to another visible road branch.
    It does not certify an image-center reference as dynamically feasible.
    """

    def _road(self, frame):
        asphalt = (frame >= .32) & (frame <= .51)
        asphalt[73:] = False
        candidates = []
        for row in range(53, 73):
            points = np.flatnonzero(asphalt[row])
            runs = np.split(points, np.flatnonzero(np.diff(points) > 1)+1)
            for run in runs:
                if len(run) >= 8:
                    candidates.extend((row, int(column)) for column in run)
        candidates = np.asarray(candidates, dtype=int)
        if not len(candidates):
            self.corridor_road = None
            return None
        distance = ((candidates[:, 1]-self.CAR_X)/self.PX_X)**2
        distance += ((candidates[:, 0]-self.CAR_Y)/self.PX_Y)**2
        seed = candidates[int(np.argmin(distance))]
        connected = np.zeros((84, 84), dtype=bool)
        connected[tuple(seed)] = True
        for _ in range(168):
            expanded = connected.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    y0, y1 = max(0, dy), min(84, 84+dy)
                    x0, x1 = max(0, dx), min(84, 84+dx)
                    expanded[y0:y1, x0:x1] |= connected[y0-dy:y1-dy, x0-dx:x1-dx]
            expanded &= asphalt
            if np.array_equal(expanded, connected):
                break
            connected = expanded
        road_frame = frame.copy()
        road_frame[asphalt & ~connected] = .9
        return super()._road(road_frame)


class _ConfidenceReference(_GraphReference):
    """Explicit road-support confidence before using a distant road component.

    Gray car fragments can be isolated from visible asphalt when the ego
    approaches an outer road edge. A supported remote component alone does
    not demonstrate support at the camera origin. An isolated nearest ego
    component activates the existing bounded recovery action deliberately.
    """

    def _road(self, frame):
        asphalt = (frame >= .32) & (frame <= .51)
        asphalt[73:] = False
        candidates = np.argwhere(asphalt[53:73])
        if not len(candidates):
            self.corridor_road = None
            return None
        candidates[:, 0] += 53
        distance = ((candidates[:, 1]-self.CAR_X)/self.PX_X)**2
        distance += ((candidates[:, 0]-self.CAR_Y)/self.PX_Y)**2
        seed = candidates[int(np.argmin(distance))]
        connected = np.zeros((84, 84), dtype=bool)
        connected[tuple(seed)] = True
        for _ in range(12):
            expanded = connected.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    y0, y1 = max(0, dy), min(84, 84+dy)
                    x0, x1 = max(0, dx), min(84, 84+dx)
                    expanded[y0:y1, x0:x1] |= connected[y0-dy:y1-dy, x0-dx:x1-dx]
            expanded &= asphalt
            if np.array_equal(expanded, connected):
                break
            connected = expanded
        support = np.argwhere(connected)
        # Require enough connected visible road for five sampled rows. This
        # checks ego support; super() still selects the full major road graph.
        if len(support) < 24 or np.ptp(support[:, 0]) < 8:
            self.corridor_road = None
            return None
        return super()._road(frame)


class _ClearRidgeReference(_ConfidenceReference):
    """Research clear-road ridge; hazards and unsupported ego retain confidence control."""

    @staticmethod
    def _expand(mask):
        result = mask.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                y0, y1 = max(0, dy), min(mask.shape[0], mask.shape[0]+dy)
                x0, x1 = max(0, dx), min(mask.shape[1], mask.shape[1]+dx)
                result[y0:y1, x0:x1] |= mask[y0-dy:y1-dy, x0-dx:x1-dx]
        return result


    def _distance_field(self, frame):
        road = (frame[:73] >= .32) & (frame[:73] <= .51)
        background = ~road
        exterior = np.zeros_like(road)
        exterior[[0, -1], :] = background[[0, -1], :]
        exterior[:, [0, -1]] = background[:, [0, -1]]
        for _ in range(168):
            updated = self._expand(exterior) & background
            if np.array_equal(updated, exterior):
                break
            exterior = updated
        holes = background & ~exterior
        visited = np.zeros_like(road)
        for y, x in np.argwhere(holes):
            if visited[y, x]:
                continue
            stack, component = [(int(y), int(x))], []
            visited[y, x] = True
            while stack:
                py, px = stack.pop()
                component.append((py, px))
                for ny, nx in ((py-1, px), (py+1, px), (py, px-1), (py, px+1)):
                    if 0 <= ny < 73 and 0 <= nx < 84 and holes[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(component) <= 110:
                ys, xs = np.asarray(component).T
                road[ys, xs] = True
        distance = np.where(road, 1000., 0.)
        neighbors = [(dy, dx, np.hypot(dx/self.PX_X, dy/self.PX_Y))
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
        for _ in range(100):
            updated = distance.copy()
            for dy, dx, weight in neighbors:
                y0, y1 = max(0, dy), min(73, 73+dy)
                x0, x1 = max(0, dx), min(84, 84+dx)
                np.minimum(updated[y0:y1, x0:x1],
                           distance[y0-dy:y1-dy, x0-dx:x1-dx]+weight,
                           out=updated[y0:y1, x0:x1])
            if np.max(abs(updated-distance)) < 1e-8:
                break
            distance = updated
        return distance


    def _sample_distance(self, distance, points):
        x = self.CAR_X+points[:, 0]*self.PX_X
        y = self.CAR_Y-points[:, 1]*self.PX_Y
        valid = (x >= 0.) & (x <= 83.) & (y >= 0.) & (y <= 72.)
        x, y = np.clip(x, 0., 83.), np.clip(y, 0., 72.)
        ix, iy = x.astype(int), y.astype(int)
        nx, ny = np.minimum(ix+1, 83), np.minimum(iy+1, 72)
        wx, wy = x-ix, y-iy
        value = ((1.-wx)*(1.-wy)*distance[iy, ix]+wx*(1.-wy)*distance[iy, nx]
                 +(1.-wx)*wy*distance[ny, ix]+wx*wy*distance[ny, nx])
        return np.where(valid, value, 0.)


    def _ridge(self, frame):
        distance = self._distance_field(frame)
        starts = np.column_stack((np.linspace(-8., 8., 65), np.full(65, 3.)))
        initial = self._sample_distance(distance, starts)
        current = starts[int(np.argmax(initial))]
        if float(initial.max()) < 2.:
            return None
        direction = np.asarray([0., 1.])
        points = [current.copy()]
        for _ in range(30):
            normal = np.asarray([direction[1], -direction[0]])
            offsets = np.linspace(-4., 4., 41)
            candidates = current+2.*direction+offsets[:, None]*normal
            score = self._sample_distance(distance, candidates)-.035*offsets**2
            chosen = candidates[int(np.argmax(score))]
            if (self._sample_distance(distance, chosen[None])[0] < 2.
                    or not (-30. < chosen[0] < 30. and -5. < chosen[1] < 37.)):
                break
            heading = chosen-current
            heading /= max(np.linalg.norm(heading), .01)
            direction = .25*direction+.75*heading
            direction /= max(np.linalg.norm(direction), .01)
            current = chosen
            points.append(current.copy())
        path = np.asarray(points)
        return path if len(path) >= 5 else None


    @staticmethod
    def _arc(path):
        return np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]


    def _ridge_target(self, path, speed):
        arc = self._arc(path)
        target = self.cruise_speed
        for at in (4., 10., 18., 26.):
            if at > arc[-1]-2.:
                continue
            selected = (arc >= max(0., at-7.)) & (arc <= at+7.)
            if np.count_nonzero(selected) < 5:
                continue
            s = arc[selected]-at
            weights = np.exp(-.5*(s/5.)**2)
            cx = np.polyfit(s, path[selected, 0], 2, w=weights)
            cy = np.polyfit(s, path[selected, 1], 2, w=weights)
            curve = abs(cx[1]*2.*cy[0]-cy[1]*2.*cx[0])/max((cx[1]**2+cy[1]**2)**1.5, .001)
            cap = np.sqrt(self.lateral_accel/max(curve, .0005))
            usable = max(0., at-2.6-.08*speed)
            target = min(target, np.sqrt(cap**2+2.*self.braking_accel*usable))
        visibility = max(0., arc[-1]-2.6-.08*speed)
        return float(min(target, np.sqrt(45.**2+2.*self.braking_accel*visibility)))

    def reset(self, observation=None):
        super().reset(observation)
        self.mode = "confidence"
        self.circle_cooldown = 0

    def act(self, observation):
        previous_steer, previous_target = self.last_steer, self.last_target
        previous_spin = self.spin_frames
        original = super().act(observation)
        self.mode = "confidence"
        if self.lost_frames:
            return original
        frame, road = self._frame(observation), self.corridor_road
        if frame is None or road is None:
            return original
        if self.pass_side or self._circles(frame, road):
            self.circle_cooldown = 12
            return original
        self.circle_cooldown = max(0, self.circle_cooldown-1)
        if self.circle_cooldown:
            return original
        path = self._ridge(frame)
        if path is None:
            return original
        # Check interpolated chords too: endpoint clearance alone can jump
        # a grass gap between visible road branches. This remains a model.
        dense = []
        for a, b in zip(path[:-1], path[1:]):
            length = float(np.linalg.norm(b-a))
            phase = np.linspace(0., 1., max(2, int(np.ceil(length/.5))+1))
            dense.append(a[None]+phase[:, None]*(b-a)[None])
        field = self._distance_field(frame)
        if min(self._sample_distance(field, np.concatenate(dense))) < 1.9:
            return original
        speed, yaw = self.last_speed, self.last_yaw
        arc = self._arc(path)
        preview = min(float(np.clip(7.+self.preview_time*speed, 8., 32.)),
                      max(3., arc[-1]-1.))
        x = float(np.interp(preview, arc, path[:, 0]))
        y = float(np.interp(preview, arc, path[:, 1]))
        curvature = 2.*x/max(x*x+y*y, 1.)
        desired = np.arctan(self.WHEELBASE*curvature)
        desired += self.yaw_gain*self.WHEELBASE*(speed*curvature-yaw)/max(speed, 20.)
        steer = previous_steer+float(np.clip(np.clip(desired, -.4, .4)-previous_steer, -.24, .24))
        target = min(self._ridge_target(path, speed),
                     np.sqrt(self.lateral_accel/max(abs(curvature), .0005)),
                     previous_target+6.)
        actual_lateral = speed*abs(yaw)
        demand = max(actual_lateral, speed*speed*abs(curvature))
        if self._rear_speed(frame)-speed > 15. and demand > 60.:
            self.spin_frames = 6
        else:
            self.spin_frames = max(0, previous_spin-1)
        if actual_lateral > self.lateral_accel:
            target = min(target, self.lateral_accel/max(abs(yaw), .05))
        error = target-speed
        if error < -2.:
            gas, brake = 0., float(np.clip(-error*.014, .04, .65))
        else:
            gas, brake = float(np.clip(.18+.06*error, 0., 1.)), 0.
            if demand > 120.:
                gas = min(gas, .30)
            elif demand > 60.:
                gas = min(gas, .60)
            if speed >= 95.:
                gas = min(gas, .16)
            elif speed >= 85.:
                gas = min(gas, .35)
            if self.spin_frames:
                gas = min(gas, .16 if speed > 15. else .08)
            projected_speed = min(100., max(self.cruise_speed, speed))
            if speed >= 45. and projected_speed**2*abs(np.tan(steer))/self.WHEELBASE >= 130.:
                gas = min(gas, .30)
        action = np.asarray([steer, gas, brake], dtype=np.float32)
        self.last_steer, self.last_target = float(action[0]), float(target)
        self.reference_curvature = curvature
        self.last_center = self.CAR_X+self.PX_X*float(path[0, 0])
        self.mode = "ridge"
        return action


class _ClearSupportedReference(_ClearRidgeReference):
    """Use only the ridge prefix with camera boundary-distance support.

    A low depth can reflect occlusion, clipped image edges or a wrong ridge,
    rather than a change in actual road width. Rejecting that reference keeps
    the exact inherited confidence action. This is a camera heuristic and
    does not certify tyre sweep or dynamic feasibility.
    """

    def _ridge(self, frame):
        path = super()._ridge(frame)
        if path is None:
            return None
        depth = self._sample_distance(self._distance_field(frame), path)
        unsupported = np.flatnonzero(depth < 5.8)
        if len(unsupported):
            path = path[:int(unsupported[0])]
        return path if len(path) >= 5 else None


class _EarlyCircleReference(_ClearSupportedReference):
    """Include antialiased inner circle pixels with unchanged shape/context gates."""

    def _circles(self, frame, road):
        _, _, rows, lefts, rights = road
        allowed = np.zeros((84, 84), dtype=bool)
        for row in range(10, 72):
            left = float(np.interp(row, rows[::-1], lefts[::-1]))
            right = float(np.interp(row, rows[::-1], rights[::-1]))
            allowed[row, max(0, int(left+1)):min(84, int(right))] = True
        bright = (frame >= .60) & (frame <= .705) & allowed
        seen = np.zeros_like(bright)
        circles = []
        for y0, x0 in zip(*np.nonzero(bright)):
            if seen[y0, x0]:
                continue
            seen[y0, x0] = True
            stack, pixels = [(int(y0), int(x0))], []
            while stack:
                y, x = stack.pop(); pixels.append((y, x))
                for ny, nx in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
                    if 0 <= ny < 84 and 0 <= nx < 84 and bright[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx] = True; stack.append((ny,nx))
            if not 4 <= len(pixels) <= 48:
                continue
            ys, xs = np.asarray(pixels).T
            width, height = xs.max()-xs.min()+1, ys.max()-ys.min()+1
            aspect = width*self.PX_Y/(height*self.PX_X)
            if not (2 <= width <= 8 and 2 <= height <= 9 and .65 <= aspect <= 1.9):
                continue
            x, y = float(xs.mean()), float(ys.mean())
            surrounding = 0
            for py, px in ((y,x-4), (y,x+4), (y-5,x), (y+5,x)):
                iy, ix = int(round(py)), int(round(px))
                if 0 <= iy < 73 and 0 <= ix < 84 and .32 <= frame[iy,ix] <= .51:
                    surrounding += 1
            if surrounding >= 3:
                circles.append(((self.CAR_Y-y)/self.PX_Y, (x-self.CAR_X)/self.PX_X))
        return sorted(circles)


class _ArcGuardReference(_EarlyCircleReference):
    """Guard the base emitted turn using visible near vehicle support.

    A distant row reference can weaken a turn while the nearby road bends
    out of its trajectory. Only an unsupported base turn tries one shorter
    preview. Sampled camera clearance is a heuristic, not a dynamic or tyre
    sweep certificate. Existing ridge selection and recovery remain active.
    """

    def reset(self, observation=None):
        super().reset(observation)
        self._arc_guard_frame = None
        self.arc_guard_used = False
        self.arc_guard_curve = 0.

    def _road(self, frame):
        self._arc_guard_frame = frame
        return super()._road(frame)

    def _arc_guard_depth(self, field, steer):
        curvature = np.tan(steer)/self.WHEELBASE
        distance = np.linspace(0., 8., 33)
        angle = curvature*distance
        if abs(curvature) < 1e-8:
            center = np.column_stack((np.zeros_like(distance), distance))
        else:
            center = np.column_stack(((1.-np.cos(angle))/curvature,
                                      np.sin(angle)/curvature))
        corners = []
        for lateral, longitudinal in ((-1.6, -2.4), (-1.6, 2.6),
                                      (1.6, -2.4), (1.6, 2.6)):
            offsets = np.column_stack((lateral*np.cos(angle)+longitudinal*np.sin(angle),
                                       -lateral*np.sin(angle)+longitudinal*np.cos(angle)))
            corners.append(center+offsets)
        return float(np.min(self._sample_distance(field, np.concatenate(corners))))

    def _steering(self, ahead, path, speed, yaw):
        desired = super()._steering(ahead, path, speed, yaw)
        emitted = self.last_steer+float(np.clip(desired-self.last_steer, -.24, .24))
        frame = self._arc_guard_frame
        if frame is None:
            return desired
        field = self._distance_field(frame)
        original_depth = self._arc_guard_depth(field, emitted)
        if original_depth >= 1.9:
            return desired
        preview = min(8., max(3., float(ahead[-1])-1.))
        x = float(np.interp(preview, ahead, path))
        curvature = 2.*x/(preview*preview+x*x)
        short = np.arctan(self.WHEELBASE*curvature)
        short += self.yaw_gain*self.WHEELBASE*(speed*curvature-yaw)/max(speed, 20.)
        short = float(np.clip(short, -.4, .4))
        short_emitted = self.last_steer+float(np.clip(short-self.last_steer, -.24, .24))
        short_depth = self._arc_guard_depth(field, short_emitted)
        if short_depth < 1.9 or short_depth <= original_depth:
            return desired
        self.arc_guard_used = True
        self.arc_guard_curve = abs(np.tan(short_emitted)/self.WHEELBASE)
        self.reference_curvature = curvature
        return short

    def _target(self, ahead, path, speed):
        target = super()._target(ahead, path, speed)
        if self.arc_guard_used:
            target = min(target, np.sqrt(self.lateral_accel/max(self.arc_guard_curve, .0005)))
        return float(target)

    def act(self, observation):
        self.arc_guard_used = False
        self.arc_guard_curve = 0.
        action = super().act(observation)
        if self.mode != "confidence" or self.lost_frames:
            self.arc_guard_used = False
            return action
        if self.arc_guard_used and action[1] > 0.:
            demand = max(self.last_speed*abs(self.last_yaw),
                         self.last_speed**2*self.arc_guard_curve)
            if demand > 120.:
                action[1] = min(float(action[1]), .30)
            elif demand > 60.:
                action[1] = min(float(action[1]), .60)
        return action


class _ArcClearReference(_ArcGuardReference):
    """Keep obstacle plans and the current emitted turn direction.

    Near support can prefer the opposite side of an S bend after a pass.
    Shortening that reverses the selected turn is outside this heuristic's
    scope. This gate preserves the original action in those observations;
    it does not establish global road or obstacle safety.
    """

    def _steering(self, ahead, path, speed, yaw):
        original = _EarlyCircleReference._steering(self, ahead, path, speed, yaw)
        original_curvature = self.reference_curvature
        frame, road = self._arc_guard_frame, self.corridor_road
        if self.pass_side or (frame is not None and road is not None
                              and self._circles(frame, road)):
            return original
        selected = super()._steering(ahead, path, speed, yaw)
        if self.arc_guard_used:
            original_emitted = self.last_steer+float(np.clip(original-self.last_steer, -.24, .24))
            selected_emitted = self.last_steer+float(np.clip(selected-self.last_steer, -.24, .24))
            if original_emitted*selected_emitted < 0.:
                self.arc_guard_used = False
                self.arc_guard_curve = 0.
                self.reference_curvature = original_curvature
                return original
        return selected


class Agent(_ArcClearReference):
    """Release the post-pass phase on confirmed camera rear clearance.

    Missing-object expiry remains conservative. Existing circle detection,
    ridge support, near-body checks and force bounds decide the next action;
    this transition alone is not a collision or dynamics certificate.
    """

    def _route(self, ahead, center, circles, speed, yaw):
        previous_side, previous_x, previous_y = self.pass_side, self.pass_x, self.pass_y
        result = super()._route(ahead, center, circles, speed, yaw)
        if (previous_side and previous_y is not None and not self.pass_side
                and self.pass_missing <= 4 and not circles and self.lost_frames == 0
                and np.isfinite([previous_x, previous_y, speed, yaw]).all()):
            angle = float(np.clip(.08*yaw, -.64, .64))
            rear_y = previous_x*np.sin(angle)+(previous_y-.08*speed)*np.cos(angle)
            if rear_y < -5.:
                self.circle_cooldown = 0
        return result



_BaseRearClearAgent = Agent

import math
import copy
import time

DT = .02


def rotate(vector, angle):
    c, s = (math.cos(angle), math.sin(angle))
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])

def cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])

def tire_inputs(state):
    front, side = ([], [])
    for i, offset in enumerate(state['wheel_offsets']):
        if state.get('wheel_velocity_override') is not None:
            world_velocity = state['wheel_velocity_override'][i]
        else:
            arm = rotate(offset, state['angle'])
            world_velocity = state['velocity'] + state['yaw'] * np.array([-arm[1], arm[0]])
            world_velocity *= min(1.0, 100.0 / max(100.0, float(np.linalg.norm(world_velocity))))
        local = rotate(world_velocity, -(state['angle'] + state['joint'][i]))
        side.append(float(local[0]))
        front.append(float(local[1]))
    return (np.asarray(front), np.asarray(side))

def tire_forces(omega, forward, sideways):
    requested = 82.0 * np.column_stack((-sideways, 0.54 * omega - forward))
    ratios = np.linalg.norm(requested, axis=1) / 400.0
    applied = requested / np.maximum(1.0, ratios)[:, None]
    return (applied, ratios)

def predict_step(state, command):
    joint_target, requested_gas, brake = command
    state = copy.deepcopy(state)
    state['gas'][2:] += np.minimum(requested_gas - state['gas'][2:], 0.1)
    omega = state['omega'] + DT * 40000.0 * state['gas'] / 1.6 / (abs(state['omega']) + 5.0)
    if brake >= 0.9:
        omega[:] = 0.0
    elif brake > 0.0:
        omega -= np.sign(omega) * np.minimum(15.0 * brake, abs(omega))
    forward, sideways = tire_inputs(state)
    forces, ratios = tire_forces(omega, forward, sideways)
    state['omega'] = omega - DT * forces[:, 1] * 0.54 / 1.6
    body_forces = np.array([rotate(force, state['joint'][i]) for i, force in enumerate(forces)])
    world_force = rotate(body_forces.sum(axis=0), state['angle'])
    torque = sum((cross(offset, force) for offset, force in zip(state['wheel_offsets'], body_forces)))
    state['velocity'] += DT * world_force / state['mass']
    state['velocity'] *= min(1.0, 100.0 / max(100.0, float(np.linalg.norm(state['velocity']))))
    state['yaw'] += DT * torque / state['inertia']
    state['angle'] += DT * state['yaw']
    state['joint'][:2] += DT * np.clip(50.0 * (joint_target - state['joint'][:2]), -3.0, 3.0)
    state['joint'] = np.clip(state['joint'], -0.4, 0.4)
    state['wheel_velocity_override'] = None
    diagnostics = {'wheel_forward_mps': forward.tolist(), 'wheel_lateral_mps': sideways.tolist(), 'tire_force_local_N': forces.tolist(), 'unsaturated_tire_demand_ratio': ratios.tolist()}
    return (state, diagnostics)

def joint_measurement(frame):
    columns = np.asarray(frame[75:81, 32:51].sum(axis=0), float)
    mass = float(columns.sum())
    centroid = float(columns @ np.arange(32, 51) / mass) if mass else 42.0
    return (mass, 1 if centroid < 42.0 else -1, centroid)

def body_measurement(frame):
    return float(frame[74:83, 10:13].sum())

def decode(frame, calibration):
    mass = body_measurement(frame)
    body = max(0.0, (mass - calibration['body_intercept']) / calibration['body_slope']) if mass else 0.0
    mass, sign, _ = joint_measurement(frame)
    coeff = calibration['joint_positive' if sign > 0 else 'joint_negative']
    joint = sign * max(0.0, (mass - coeff[1]) / coeff[0]) if mass else 0.0
    return (body, joint)

PUBLIC_GEOMETRY = {'mass': 7.3019199669361115, 'inertia': 19.170645977936854, 'local_center': [0.0, -0.0804591735470409], 'wheel_anchors': [[-1.1, 1.6], [1.1, 1.6], [-1.1, -1.64], [1.1, -1.64]]}

SENSOR_BOUNDS = {'body_speed_mps': 1.6, 'joint_rad': 0.006, 'rear_individual_rolling_mps': 1.94, 'yaw_radps_provisional': 0.15}

class CameraObserver:
    """Camera-only interface: observe(frame), advance(legal_action), reset()."""

    def __init__(self, calibration):
        self.calibration = copy.deepcopy(calibration)
        self.reset()

    def reset(self):
        center = np.array(PUBLIC_GEOMETRY['local_center'], float)
        self.state = {'angle': 0.0, 'velocity': np.zeros(2), 'yaw': 0.0, 'mass': PUBLIC_GEOMETRY['mass'], 'inertia': PUBLIC_GEOMETRY['inertia'], 'local_center': center, 'wheel_offsets': np.array(PUBLIC_GEOMETRY['wheel_anchors'], float) - center, 'joint': np.zeros(4), 'omega': np.zeros(4), 'gas': np.zeros(4), 'wheel_velocity_override': None}
        self.actions_seen = 0
        self.last_sensor = None

    def observe(self, frame):
        frame = np.asarray(frame, float)
        if frame.shape != (84, 84) or not np.isfinite(frame).all():
            raise ValueError('expected finite84x84 camera')
        speed, joint = decode(frame, self.calibration['body_joint'])
        speed = float(np.clip(speed, 0.0, 100.0))
        rear_mass = frame[74:83, [20, 22]].sum(axis=0)
        rear_rolling = np.maximum(0.0, (rear_mass - self.calibration['rear_intercept']) @ self.calibration['rear_inverse']) * 0.54
        red = frame[75:81, 51:83]
        red = np.where((red > 0.0) & (red < 0.37), red, 0.0)
        mass = float(red.sum())
        centroid = float(red.sum(axis=0) @ np.arange(51, 83) / mass) if mass else 62.5
        magnitude = max(0.0, (mass - self.calibration['yaw_intercept']) / self.calibration['yaw_slope'])
        camera_yaw = magnitude if centroid >= 62.5 else -magnitude
        body_yaw = float(np.clip(-camera_yaw, -8.0, 8.0))
        old = self.state
        local_velocity = rotate(old['velocity'], -old['angle'])
        prior_speed = float(np.linalg.norm(local_velocity))
        prior_rear = 0.54 * old['omega'][2:]
        prior_yaw = old['yaw']
        side = float(np.clip(local_velocity[0], -0.8 * speed, 0.8 * speed))
        old['velocity'] = np.array([side, math.sqrt(max(0.0, speed * speed - side * side))])
        old['angle'] = 0.0
        old['yaw'] = body_yaw
        old['joint'][:2] = np.clip(old['joint'][:2], joint - SENSOR_BOUNDS['joint_rad'], joint + SENSOR_BOUNDS['joint_rad'])
        radius = SENSOR_BOUNDS['rear_individual_rolling_mps'] / 0.54
        old['omega'][2:] = np.maximum(0.0, np.clip(old['omega'][2:], rear_rolling / 0.54 - radius, rear_rolling / 0.54 + radius))
        old['wheel_velocity_override'] = None
        self.last_sensor = {'speed_mps': speed, 'body_yaw_radps': body_yaw, 'joint_rad': joint, 'rear_rolling_mps': rear_rolling.tolist(), 'speed_innovation_mps': speed - prior_speed, 'yaw_innovation_radps': body_yaw - prior_yaw, 'rear_innovation_max_mps': float(np.max(abs(rear_rolling - prior_rear))), 'rear_bar_top_nonzero': bool(np.any(frame[74, [20, 22]] > 0.0)), 'yaw_beyond_calibration': abs(body_yaw) > 4.5, 'dynamics_innovation_suspect': self.actions_seen > 0 and (abs(speed - prior_speed) > 3.2 or abs(body_yaw - prior_yaw) > 0.3 or float(np.max(abs(rear_rolling - prior_rear))) > 6.0), 'lateral_uncertainty_certified': False}
        return (copy.deepcopy(old), copy.deepcopy(self.last_sensor))

    def advance(self, legal_action):
        action = np.asarray(legal_action, float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError('expected finite3-vector legal action')
        action = np.clip(action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        diagnostics = []
        for _ in range(4):
            self.state, row = predict_step(self.state, [-action[0], action[1], action[2]])
            diagnostics.append(row)
        self.actions_seen += 1
        return (copy.deepcopy(self.state), diagnostics)

def arclength(path):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]

def project_one(path, point, lower, upper):
    """Nearest projection restricted to an ordered reachable arc window."""
    arc = arclength(path)
    links = np.diff(path, axis=0)
    length = np.diff(arc)
    active = (arc[:-1] <= upper) & (arc[1:] >= lower) & (length > 1e-08)
    indices = np.flatnonzero(active)
    if not len(indices):
        raise ValueError('Projection interval misses supported reference')
    lo = np.clip((lower - arc[indices]) / length[indices], 0.0, 1.0)
    hi = np.clip((upper - arc[indices]) / length[indices], 0.0, 1.0)
    fraction = np.sum((point - path[indices]) * links[indices], axis=1) / length[indices] ** 2
    fraction = np.clip(fraction, lo, hi)
    projected = path[indices] + fraction[:, None] * links[indices]
    chosen = int(np.argmin(np.sum((projected - point) ** 2, axis=1)))
    index = indices[chosen]
    tangent = links[index] / length[index]
    lateral = float(np.dot(point - projected[chosen], [tangent[1], -tangent[0]]))
    return (float(arc[index] + fraction[chosen] * length[index]), lateral, tangent)

def project_progress(path, positions, velocity, dt, projection_uncertainty_m=0.0):
    """Progress along the reference, without a road-center rejoin penalty.

    A backward first-tangent extension establishes ego arc origin only; it
    does not invent supported asphalt. Future projection is monotone and
    cannot advance more than max(prior,post speed)*dt plus supplied
    uncertainty, respecting semi-implicit post-velocity pose integration.
    """
    path = np.asarray(path, float)
    positions = np.asarray(positions, float)
    velocity = np.asarray(velocity, float)
    direction = path[1] - path[0]
    direction /= np.linalg.norm(direction)
    extended = np.vstack((path[0] - 10.0 * direction, path))
    origin, _, _ = project_one(extended, positions[0], 0.0, 10.0)
    current = origin
    rows = []
    extent = arclength(extended)[-1]
    for i, point in enumerate(positions):
        advance = 0.0 if i == 0 else float(max(np.linalg.norm(velocity[i - 1]), np.linalg.norm(velocity[i])) * dt + projection_uncertainty_m)
        current, lateral, tangent = project_one(extended, point, current, min(extent, current + advance))
        rows.append((current - origin, lateral, tangent[0], tangent[1]))
    return dict(projections=np.asarray(rows), supported_remaining_m=float(extent - current), reference_origin_arc_m=float(origin - 10.0), supported_extent_from_ego_m=float(extent - origin))

def body_points(positions, angles):
    """Full25-point rectangle stencil, including edges/interior and corners."""
    positions = np.asarray(positions, float)
    angles = np.asarray(angles, float)
    lateral, longitudinal = np.meshgrid(np.linspace(-1.6, 1.6, 5), np.linspace(-2.4, 2.6, 5))
    lat = lateral.ravel()
    long = longitudinal.ravel()
    c = np.cos(angles)[:, None]
    s = np.sin(angles)[:, None]
    offsets = np.stack((c * lat - s * long, s * lat + c * long), axis=2)
    return positions[:, None, :] + offsets

def sample_field(field, points):
    """Bilinear camera field sampling; outside road image rows0..72 is0."""
    shape = points.shape[:-1]
    points = np.asarray(points).reshape(-1, 2)
    x = 42.0 + 1.3608 * points[:, 0]
    y = 63.0 - 1.701 * points[:, 1]
    valid = (x >= 0.0) & (x <= 83.0) & (y >= 0.0) & (y <= 72.0)
    x = np.clip(x, 0, 83)
    y = np.clip(y, 0, 72)
    ix = x.astype(int)
    iy = y.astype(int)
    nx = np.minimum(ix + 1, 83)
    ny = np.minimum(iy + 1, 72)
    wx = x - ix
    wy = y - iy
    values = (1 - wx) * (1 - wy) * field[iy, ix] + wx * (1 - wy) * field[iy, nx] + (1 - wx) * wy * field[ny, ix] + wx * wy * field[ny, nx]
    return np.where(valid, values, 0.0).reshape(shape)

def road_slack(field, positions, angles, minimum_depth_m=1.9, grass_mask=None):
    points = body_points(positions, angles)
    depth = sample_field(field, points)
    if grass_mask is not None:
        grass = sample_field(np.asarray(grass_mask, float), points)
        depth = np.where(grass > 0.5, 0.0, depth)
    return np.min(depth, axis=1) - minimum_depth_m

def obstacle_slack(positions, angles, circles, uncertainty_m=0.0):
    """Detected/transported camera centers are(x,y), unlike detector(y,x)."""
    positions = np.asarray(positions)
    angles = np.asarray(angles)
    if not len(circles):
        return np.full(len(positions), np.inf)
    centers = np.asarray(circles, float)
    delta = centers[None] - positions[:, None, :]
    center = np.linalg.norm(delta, axis=2) - (3.7 + uncertainty_m)
    c = np.cos(angles)[:, None]
    s = np.sin(angles)[:, None]
    x = c * delta[:, :, 0] + s * delta[:, :, 1]
    y = -s * delta[:, :, 0] + c * delta[:, :, 1]
    dx = x - np.clip(x, -1.6, 1.6)
    dy = y - np.clip(y, -2.4, 2.6)
    rectangle = np.hypot(dx, dy) - (1.2 + 0.75 + uncertainty_m)
    return np.min(np.minimum(center, rectangle), axis=1)

def terminal_speed_limit(reference, angle, tangent, speed, braking_accel, braking_delay_s, uncertainty_m=0.0):
    """Stop before supported reference end; supplied braking model is assumed."""
    if braking_accel <= 0.0 or braking_delay_s < 0.0:
        raise ValueError('Invalid braking assumption')
    offsets = body_points(np.zeros((1, 2)), np.asarray([angle]))[0]
    front = max(0.0, float(np.max(offsets @ np.asarray(tangent))))
    usable = max(0.0, reference['supported_remaining_m'] - front - speed * braking_delay_s - uncertainty_m)
    return float(np.sqrt(2.0 * braking_accel * usable))

class CameraGeometry:
    """Frozen reference and camera support for one MPC planning call.

    evaluate inputs include the initial state before raw integration ticks.
    Re-evaluate each full beam prefix so branch progress survives .08 knots.
    circles are detected/remembered camera(x,y), never detector(y,x).
    """

    def __init__(self, path, field, circles=(), grass_mask=None, minimum_depth_m=1.9, obstacle_uncertainty_m=0.0, projection_uncertainty_m=0.0):
        self.path = np.asarray(path, float).copy()
        self.field = np.asarray(field, float).copy()
        self.circles = np.asarray(circles, float).reshape(-1, 2).copy()
        self.grass_mask = None if grass_mask is None else np.asarray(grass_mask, bool).copy()
        self.minimum_depth_m = minimum_depth_m
        self.obstacle_uncertainty_m = obstacle_uncertainty_m
        self.projection_uncertainty_m = projection_uncertainty_m
        if len(self.path) < 2 or not np.isfinite(self.path).all() or (not np.isfinite(self.field).all()):
            raise ValueError('Unsupported camera reference')

    def evaluate(self, positions, angles, velocities, dt=0.02):
        positions = np.asarray(positions, float)
        angles = np.asarray(angles, float)
        velocities = np.asarray(velocities, float)
        if positions.shape != velocities.shape or positions.shape != (len(angles), 2):
            raise ValueError('Pose/velocity arrays must align')
        if not np.isfinite(positions).all() or not np.isfinite(angles).all() or (not np.isfinite(velocities).all()):
            raise ValueError('Nonfinite predicted state')
        projection = project_progress(self.path, positions, velocities, dt, self.projection_uncertainty_m)
        road = road_slack(self.field, positions, angles, self.minimum_depth_m, self.grass_mask)
        obstacle = obstacle_slack(positions, angles, self.circles, self.obstacle_uncertainty_m)
        tangents = projection['projections'][:, 2:4]
        offsets = body_points(positions, angles) - positions[:, None, :]
        front = np.maximum(0.0, np.max(np.sum(offsets * tangents[:, None, :], axis=2), axis=1))
        progress = projection['projections'][:, 0]
        front_slack = projection['supported_extent_from_ego_m'] - progress - front
        return dict(progress=progress, progress_m=progress, lateral_offset=projection['projections'][:, 1], path_tangents=tangents, road_slack=road, road_slack_m=road, obstacle_slack=obstacle, obstacle_slack_m=obstacle, reference_front_slack_m=front_slack, violation=(road < 0.0) | (obstacle < 0.0) | (front_slack < 0.0), supported_remaining_m=projection['supported_remaining_m'], supported_extent_from_ego_m=projection['supported_extent_from_ego_m'])

class FixedControlPlanner:
    DT, STEPS, EXECUTED_STEPS, BACKUP_STEPS = (0.02, 16, 4, 32)
    STEER_OFFSETS = np.asarray([-0.24, -0.12, -0.06, -0.03, 0.0, 0.03, 0.06, 0.12, 0.24])
    PEDALS = ((0.0, 0.0), (0.16, 0.0), (0.3, 0.0), (0.6, 0.0), (1.0, 0.0), (0.0, 0.25), (0.0, 0.55), (0.0, 1.0))
    ARRAY_SHAPES = {'velocity': (2,), 'local_center': (2,), 'wheel_offsets': (4, 2), 'joint': (4,), 'omega': (4,), 'gas': (4,)}
    SCALARS = ('angle', 'yaw', 'mass', 'inertia')

    def __init__(self, predict_step, geometry):
        self.predict_step = predict_step
        self.geometry = geometry
        self.model_steps = 0

    @classmethod
    def _copy_state(cls, state):
        expected = set(cls.ARRAY_SHAPES) | set(cls.SCALARS) | {'wheel_velocity_override'}
        if set(state) != expected or state['wheel_velocity_override'] is not None:
            raise ValueError('expected supplied camera model state without wheel velocity override')
        result = {'wheel_velocity_override': None}
        for name in cls.SCALARS:
            result[name] = float(state[name])
            if not np.isfinite(result[name]):
                raise ValueError('nonfinite model scalar')
        if result['mass'] <= 0.0 or result['inertia'] <= 0.0:
            raise ValueError('invalid mechanical constants')
        for name, shape in cls.ARRAY_SHAPES.items():
            value = np.asarray(state[name], dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('invalid model array')
            result[name] = value.copy()
        if np.any(abs(result['joint']) > 0.400001) or np.any(result['gas'] < 0.0) or np.any(result['gas'] > 1.0):
            raise ValueError('model joint or gas outside mechanical bounds')
        return result

    @staticmethod
    def _rotate(vector, angle):
        c, s = (np.cos(angle), np.sin(angle))
        return np.asarray([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])

    def candidate_actions(self, previous_action):
        previous = np.asarray(previous_action, dtype=float)
        if previous.shape != (3,) or not np.isfinite(previous).all():
            raise ValueError('expected finite previous legal action')
        if np.any(previous < [-1.0, 0.0, 0.0]) or np.any(previous > 1.0):
            raise ValueError('previous action outside legal bounds')
        commands = np.unique(np.clip(previous[0] + self.STEER_OFFSETS, -0.4, 0.4))
        commands = commands[abs(commands - previous[0]) <= 0.2400001]
        return np.asarray([[steer, gas, brake] for steer in commands for gas, brake in self.PEDALS], dtype=np.float32).reshape(-1, 3)

    def _advance(self, state, cm, action):
        self.model_steps += 1
        state, diagnostics = self.predict_step(state, [-float(action[0]), float(action[1]), float(action[2])])
        state = self._copy_state(state)
        cm = cm + self.DT * state['velocity']
        hull = cm - self._rotate(state['local_center'], state['angle'])
        forces = np.asarray(diagnostics['tire_force_local_N'], dtype=float)
        ratios = np.asarray(diagnostics['unsaturated_tire_demand_ratio'], dtype=float)
        forward = np.asarray(diagnostics['wheel_forward_mps'], dtype=float)
        if forces.shape != (4, 2) or ratios.shape != (4,) or forward.shape != (4,) or (not np.isfinite(forces).all()) or (not np.isfinite(ratios).all()) or (not np.isfinite(forward).all()) or (np.max(np.linalg.norm(forces, axis=1)) > 400.00001):
            raise ValueError('invalid bounded four-tire diagnostics')
        return (state, cm, hull, diagnostics)

    def rollout(self, supplied_state, action):
        state = self._copy_state(supplied_state)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.isfinite(action).all() or abs(action[0]) > 0.400001 or np.any(action[1:] < 0.0) or np.any(action[1:] > 1.0) or (action[1] * action[2] != 0.0):
            raise ValueError('invalid separate-pedal constant action')
        cm = self._rotate(state['local_center'], state['angle'])
        states, centers, positions = ([state], [cm.copy()], [np.zeros(2)])
        angles, velocities, diagnostics = ([state['angle']], [state['velocity'].copy()], [])
        for _ in range(self.STEPS):
            state, cm, hull, row = self._advance(state, cm, action)
            states.append(state)
            centers.append(cm.copy())
            positions.append(hull)
            angles.append(state['angle'])
            velocities.append(state['velocity'].copy())
            diagnostics.append(row)
        return {'states': states, 'cm_positions': np.asarray(centers), 'positions': np.asarray(positions), 'angles': np.asarray(angles), 'velocities': np.asarray(velocities), 'diagnostics': diagnostics}

    def _geometry(self, positions, angles, velocities):
        result = self.geometry.evaluate(positions, angles, velocities, self.DT)
        count = len(positions)
        progress = np.asarray(result['progress_m'], dtype=float)
        road = np.asarray(result['road_slack_m'], dtype=float)
        obstacle = np.asarray(result['obstacle_slack_m'], dtype=float)
        front = np.asarray(result['reference_front_slack_m'], dtype=float)
        if any((value.shape != (count,) for value in (progress, road, obstacle, front))) or not np.isfinite(progress).all() or (not np.isfinite(road).all()) or (not np.isfinite(front).all()) or np.isnan(obstacle).any():
            raise ValueError('invalid camera geometry result')
        violation = float(np.max(np.maximum(0.0, -np.minimum(np.minimum(road, obstacle), front))))
        return (progress, violation)

    @staticmethod
    def _stopped(state):
        return float(np.linalg.norm(state['velocity'])) <= 0.5 and abs(state['yaw']) <= 0.1

    def _backup(self, rollout, action):
        index = self.EXECUTED_STEPS
        state = self._copy_state(rollout['states'][index])
        cm = rollout['cm_positions'][index].copy()
        positions = list(rollout['positions'][:index + 1].copy())
        angles = list(rollout['angles'][:index + 1].copy())
        velocities = list(rollout['velocities'][:index + 1].copy())
        steer = float(np.clip(-np.mean(state['joint'][:2]), -0.4, 0.4))
        steer = float(np.clip(steer, float(action[0]) - 0.24, float(action[0]) + 0.24))
        command = np.asarray([steer, 0.0, 1.0], dtype=np.float32)
        ticks = 0
        while not self._stopped(state) and ticks < self.BACKUP_STEPS:
            state, cm, hull, _ = self._advance(state, cm, command)
            positions.append(hull)
            angles.append(state['angle'])
            velocities.append(state['velocity'].copy())
            ticks += 1
        progress, violation = self._geometry(np.asarray(positions), np.asarray(angles), np.asarray(velocities))
        return {'stopped': self._stopped(state), 'violation': violation, 'progress_m': float(progress[-1]), 'duration_s': ticks * self.DT}

    def plan(self, supplied_states, previous_action):
        """Return a finite feasible first action, or None for caller fallback.

        Hard constraints hold for every supplied scenario; they do not cover
        unprovided observation errors. Emergency feasibility starts after
        the only executed .08s block, not after the unexecuted .32s horizon.
        """
        self.model_steps = 0
        try:
            states = [self._copy_state(state) for state in supplied_states]
            if not 1 <= len(states) <= 7:
                raise ValueError('expected one to seven supplied camera state scenarios')
            actions = self.candidate_actions(previous_action)
            best, feasible_count, worst_violation = (None, 0, 0.0)
            for action in actions:
                summaries = []
                for state in states:
                    rollout = self.rollout(state, action)
                    progress, violation = self._geometry(rollout['positions'], rollout['angles'], rollout['velocities'])
                    worst_violation = max(worst_violation, violation)
                    if violation > 0.0:
                        break
                    backup = self._backup(rollout, action)
                    worst_violation = max(worst_violation, backup['violation'])
                    if not backup['stopped'] or backup['violation'] > 0.0:
                        break
                    ratio_cost, slip_cost = ([], [])
                    for predicted, row in zip(rollout['states'][1:], rollout['diagnostics']):
                        ratio = np.asarray(row['unsaturated_tire_demand_ratio'])
                        slip = 0.54 * predicted['omega'][2:] - np.asarray(row['wheel_forward_mps'])[2:]
                        ratio_cost.append(float(np.mean(np.maximum(0.0, ratio - 1.0) ** 2)))
                        slip_cost.append(float(np.mean(slip ** 2)))
                    summaries.append({'progress': float(progress[-1]), 'model_cost': 0.1 * float(np.mean(ratio_cost)) + 0.002 * float(np.mean(slip_cost)), 'terminal_speed': float(np.linalg.norm(rollout['velocities'][-1])), 'backup_progress': backup['progress_m'], 'backup_duration': backup['duration_s']})
                if len(summaries) != len(states):
                    continue
                feasible_count += 1
                progress = min((row['progress'] for row in summaries))
                score = -progress + max((row['model_cost'] for row in summaries))
                score += 0.5 * (float(action[0]) - float(previous_action[0])) ** 2 + 0.02 * float(action[2])
                if best is None or score < best['score']:
                    best = {'feasible': True, 'action': action.copy(), 'reason': 'feasible', 'score': float(score), 'worst_progress_m': progress, 'predicted_terminal_speed_mps': min((row['terminal_speed'] for row in summaries)), 'backup_start_s': self.EXECUTED_STEPS * self.DT, 'backup_end_progress_m': max((row['backup_progress'] for row in summaries)), 'backup_duration_s': max((row['backup_duration'] for row in summaries))}
            if best is None:
                best = {'feasible': False, 'action': None, 'reason': 'no_feasible_candidate'}
            best.update(candidate_count=len(actions), feasible_count=feasible_count, scenario_count=len(states), model_steps=self.model_steps, rejected_max_violation_m=worst_violation)
            return best
        except Exception as error:
            return {'feasible': False, 'action': None, 'reason': 'planning_error', 'error': type(error).__name__ + ': ' + str(error)[:160], 'model_steps': self.model_steps}

def _fixed_camera_calibration():
    return {
        'body_joint': {
            'body_slope': .08573317526988666,
            'body_intercept': .3065159749445597,
            'joint_positive': [53.32976668292053, .3237707931261848],
            'joint_negative': [53.33307758282946, .11603323404605852],
        },
        'rear_intercept': np.array([.020195027723669583, .00796898243261536]),
        'rear_inverse': np.array([[282.49406468720383, -6.042857175259085e-14],
                                 [2.3915441163734697e-13, 309.2412452796282]]),
        'yaw_slope': 2.1639217019081123,
        'yaw_intercept': .151470661163329,
    }


class _HullCameraGeometry(CameraGeometry):
    def evaluate(self, positions, angles, velocities, dt=.02):
        positions = np.asarray(positions, float)
        hull_velocity = np.asarray(velocities, float).copy()
        if len(positions) > 1:
            # Projection positions are hull origins, not compound centers.
            hull_velocity[1:] = np.diff(positions, axis=0)/dt
        return super().evaluate(positions, angles, hull_velocity, dt)


class _PlanningBudgetExceeded(RuntimeError):
    pass


class Agent(_BaseRearClearAgent):
    """Coupled camera/model control with a real-action fallback and stop tail.

    Three representative state scenarios are not a complete uncertainty box
    or a safety certificate. Unknown camera support retains the parent action.
    """

    def __init__(self, *args, planning_budget_s=3., **kwargs):
        if not np.isfinite(planning_budget_s) or planning_budget_s < 0.:
            raise ValueError('planning budget must be finite and nonnegative')
        self._planning_budget_s = min(float(planning_budget_s), 3.)
        self._observer = CameraObserver(_fixed_camera_calibration())
        super().__init__(*args, **kwargs)

    def reset(self, observation=None):
        super().reset(observation)
        self._observer.reset()
        self._predictive_previous_action = np.zeros(3, np.float32)
        self._predictive_deadline = 0.
        self.predictive_status = 'reset'
        self.predictive_result = {}
        self.predictive_sensor = {}
        self.predictive_latency_s = 0.
        self.predictive_model_calls = 0

    def _predictive_field(self, frame, circles):
        image = frame[:73]
        road = (image >= .32) & (image <= .51)
        x = (np.arange(84)-self.CAR_X)/self.PX_X
        y = (self.CAR_Y-np.arange(73))/self.PX_Y
        # Fill only rendered dark car pixels inside the known vehicle bounds;
        # a bright enclosed grass component is never converted to asphalt.
        car = ((abs(x)[None, :] <= 1.6) & (y[:, None] >= -2.4) &
               (y[:, None] <= 2.6) & (image < .32))
        road |= car
        for object_x, object_y in circles:
            core = ((x[None, :]-object_x)**2+(y[:, None]-object_y)**2 <= 1.2**2)
            road |= core & (image >= .655) & (image <= .705)
        # The image boundary is a visibility boundary; no exterior road is
        # inferred from an asphalt run touching it.
        road[[0, -1], :] = False
        road[:, [0, -1]] = False
        unknown = ~road
        distance = np.where(road, 1000., 0.)
        neighbors = [(dy, dx, np.hypot(dx/self.PX_X, dy/self.PX_Y))
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
        for _ in range(100):
            updated = distance.copy()
            for dy, dx, weight in neighbors:
                y0, y1 = max(0, dy), min(73, 73+dy)
                x0, x1 = max(0, dx), min(84, 84+dx)
                np.minimum(updated[y0:y1, x0:x1],
                           distance[y0-dy:y1-dy, x0-dx:x1-dx]+weight,
                           out=updated[y0:y1, x0:x1])
            if np.max(abs(updated-distance)) < 1e-8:
                break
            distance = updated
        return distance, unknown

    def _predictive_scenarios(self, state):
        scenarios = [copy.deepcopy(state)]
        speed = float(np.linalg.norm(state['velocity']))
        # Two fixed correlated alternatives, not all independent sensor-error
        # combinations. Axle rolling speed covaries with body-speed magnitude.
        for sign in (-1., 1.):
            other = copy.deepcopy(state)
            other_speed = float(np.clip(speed+sign*1.6, 0., 100.))
            side = float(np.clip(state['velocity'][0]+sign*.35, -.8*other_speed, .8*other_speed))
            other['velocity'] = np.array([side, np.sqrt(max(0., other_speed**2-side**2))])
            other['yaw'] = float(state['yaw']+sign*.15)
            other['joint'][:2] = np.clip(state['joint'][:2]+sign*.006, -.4, .4)
            other['omega'] = np.maximum(0., state['omega']+(other_speed-speed)/.54)
            scenarios.append(other)
        return scenarios

    def _predictive_step(self, state, command):
        if time.perf_counter() >= self._predictive_deadline:
            raise _PlanningBudgetExceeded('camera planning time budget reached')
        self.predictive_model_calls += 1
        return predict_step(state, command)

    def _predictive_supported_prefix(self, path, field):
        # Keep original nodes, including their minimum-five-node semantics.
        # Dense samples only locate the first unknown/unsupported boundary;
        # a later supported road branch can never rejoin after a grass gap.
        if self._sample_distance(field, path[:1])[0] < 1.9:
            return None
        prefix = [path[0].copy()]
        for left, right in zip(path[:-1], path[1:]):
            count = max(2, int(np.ceil(np.linalg.norm(right-left)/.25))+1)
            dense = np.linspace(left, right, count)
            depth = self._sample_distance(field, dense)
            unsafe = np.flatnonzero(depth < 1.9)
            if len(unsafe):
                if len(prefix) < 5:
                    return None
                index = int(unsafe[0])
                if index > 0 and np.linalg.norm(dense[index-1]-prefix[-1]) > 1e-8:
                    prefix.append(dense[index-1].copy())
                return np.asarray(prefix)
            prefix.append(right.copy())
        return np.asarray(prefix) if len(prefix) >= 5 else None

    def _predictive_geometry(self, frame, state):
        road = self.corridor_road
        if road is None or self.lost_frames:
            return None
        circles = [(float(x), float(y)) for y, x in self._circles(frame, road)]
        # Parent act already transported the active pass once. Keep that
        # remembered actual circle even on a current detection miss.
        if (self.pass_side and self.pass_y is not None and self.pass_missing <= 4 and
                np.isfinite([self.pass_x, self.pass_y]).all()):
            remembered = np.array([self.pass_x, self.pass_y], float)
            if all(np.linalg.norm(remembered-np.asarray(circle)) > .5 for circle in circles):
                circles.append(tuple(remembered))
        field, unknown = self._predictive_field(frame, circles)
        path = _ClearRidgeReference._ridge(self, frame)
        if path is not None:
            path = self._predictive_supported_prefix(path, field)
        if path is None or len(path) < 5:
            return None
        dense = np.concatenate([np.linspace(left, right, 10) for left, right in zip(path[:-1], path[1:])])
        if np.min(self._sample_distance(field, dense)) < 1.9:
            return None
        geometry = _HullCameraGeometry(path, field, circles, unknown)
        initial = geometry.evaluate(np.zeros((1, 2)), np.zeros(1),
                                    np.asarray(state['velocity'])[None], .02)
        return geometry if not initial['violation'].any() else None

    def act(self, observation):
        started = time.perf_counter()
        self._predictive_deadline = started+self._planning_budget_s
        previous = self._predictive_previous_action.copy()
        # Exactly one parent call updates all inherited perception/pass memory.
        fallback = np.asarray(super().act(observation), np.float32)
        emitted = fallback
        self.predictive_status = 'fallback_invalid_camera'
        self.predictive_result = {}
        self.predictive_sensor = {}
        self.predictive_model_calls = 0
        frame = self._frame(observation)
        if frame is not None:
            state, sensor = self._observer.observe(frame)
            self.predictive_sensor = sensor
            if sensor['speed_mps'] < 20.:
                self.predictive_status = 'fallback_low_speed'
            elif sensor['dynamics_innovation_suspect'] or sensor['rear_bar_top_nonzero']:
                self.predictive_status = 'fallback_innovation'
            elif sensor['yaw_beyond_calibration']:
                self.predictive_status = 'fallback_yaw'
            else:
                geometry = self._predictive_geometry(frame, state)
                self.predictive_status = 'fallback_unsupported'
                if geometry is not None:
                    planner = FixedControlPlanner(self._predictive_step, geometry)
                    result = planner.plan(self._predictive_scenarios(state), previous)
                    self.predictive_result = result
                    if result.get('feasible') and result.get('action') is not None:
                        action = np.asarray(result['action'], np.float32)
                        if (action.shape == (3,) and np.isfinite(action).all() and
                                np.all(action >= [-1., 0., 0.]) and np.all(action <= 1.)):
                            emitted = action
                            self.predictive_status = 'predictive'
                    else:
                        self.predictive_status = 'fallback_timeout' if '_PlanningBudgetExceeded' in result.get('error', '') else 'fallback_no_feasible'
        emitted = np.asarray(emitted, np.float32)
        # Every actual action, including fallback, advances history exactly once.
        self._observer.advance(emitted)
        self._predictive_previous_action = emitted.copy()
        self.last_steer = float(emitted[0])
        self.predictive_latency_s = time.perf_counter()-started
        return emitted
