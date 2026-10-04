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


class Agent(_GraphReference):
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
