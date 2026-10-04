"""Pixel-only free-space racing-line optimization, independent of other drivers.

A second-order lattice search selects an entire visible path, including obstacle
clearance. Pure pursuit follows that path and a backwards braking envelope sets
speed. All state is derived from observations and previous actions.
"""
from __future__ import annotations
import cv2
import numpy as np


DEFAULTS = dict(max_speed=72., min_speed=24., lateral_accel=48., braking_accel=65.,
                lookahead=12., speed_lookahead=.16, pursuit_gain=6.,
                steer_smoothing=.35, max_steer=.85, clearance=3.,
                obstacle_margin=3, bend_cost=.8, slope_cost=.04,
                center_cost=2., previous_cost=.015, brake_gain=.025,
                road_low=.24, road_high=.52, motion_preview=0.)


class Agent:
    PIXELS_X, PIXELS_Y = 1.3608, 1.701
    def __init__(self, config=None, *, project_root=None):
        del project_root
        self.config = dict(DEFAULTS)
        if config:
            unknown = set(config) - set(self.config)
            if unknown:
                raise ValueError(f'unknown configuration keys: {sorted(unknown)}')
            self.config.update(config)
        self.reset()

    def reset(self, observation=None):
        del observation
        self.last_steer = 0.
        self.previous_path = None
        self.diagnostics = {}

    def _free_space(self, frame):
        c = self.config
        road = ((frame >= c['road_low']) & (frame <= c['road_high'])).astype(np.uint8)
        # Small image holes arise from aliasing/paint, not only obstacles. Recover
        # the road surface first, then subtract separately identified hazards.
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        bright = (frame >= .54).astype(np.uint8)
        bright[:5] = 0
        bright[61:] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
        obstacles = np.zeros_like(road)
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if 4 <= area <= 90 and 2 <= w <= 11 and 2 <= h <= 12:
                # Component must be surrounded by road, excluding grass islands.
                region = road[max(0,y-2):min(74,y+h+2), max(0,x-2):min(84,x+w+2)]
                if region.mean() > .2:
                    obstacles[labels == i] = 1
        radius = int(c['obstacle_margin'])
        obstacles = cv2.dilate(obstacles, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,2*radius+1)))
        road[74:] = 0
        free = road.copy()
        free[obstacles > 0] = 0
        clearance = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        return free, obstacles, clearance

    def _path(self, frame):
        free, obstacles, clearance = self._free_space(frame)
        c = self.config
        rows = np.arange(58, 7, -2)
        xs = np.arange(84)
        slopes = np.arange(-4, 5)
        costs = np.full((84, 9), np.inf)
        # Starting pose is known from the fixed renderer, not simulator state.
        costs[:, 4] = .35*(xs-42.)**2
        costs[np.abs(xs-42) > 12] = np.inf
        parents = []
        accepted = []
        last_costs = costs
        for row in rows:
            d = clearance[row]
            node = c['center_cost']/(d+.5) + 3.*np.maximum(c['clearance']-d, 0.)**2
            node += 70.*(free[row] == 0) + 300.*obstacles[row]
            if self.previous_path is not None:
                px = float(np.interp(row, self.previous_path[::-1,1], self.previous_path[::-1,0]))
                node += c['previous_cost']*(xs-px)**2
            next_costs = np.full_like(costs, np.inf)
            parent = np.zeros((84,9), np.int16)
            for j, delta in enumerate(slopes):
                prevx = xs-delta
                valid = (prevx >= 0) & (prevx < 84)
                indices = np.flatnonzero(valid)
                options = costs[prevx[valid]] + c['bend_cost']*(slopes-delta)**2
                which = np.argmin(options, axis=1)
                next_costs[indices,j] = options[np.arange(len(indices)),which] + node[valid] + c['slope_cost']*delta**2
                parent[indices,j] = which
            # Stop at the camera's useful road horizon rather than hallucinate
            # a route over grass beyond an almost-horizontal visible bend.
            best = np.unravel_index(np.argmin(next_costs), next_costs.shape)
            if accepted and (not free[row,best[0]] or np.min(next_costs) > np.min(costs)+100.):
                break
            accepted.append(row)
            parents.append(parent)
            costs = next_costs
            last_costs = costs
        x, j = np.unravel_index(np.argmin(last_costs), last_costs.shape)
        result = []
        for k in range(len(accepted)-1, -1, -1):
            result.append((float(x),float(accepted[k])))
            oldj = int(parents[k][x,j])
            x -= int(slopes[j])
            j = oldj
        path = np.asarray(result[::-1], np.float32).reshape(-1,2)
        return path, free, obstacles

    def _tracking_curvature(self, path, lookahead, free, obstacles):
        """Choose a pursuit target whose implied arc respects nearby free space.

        A distant point alone can shortcut through a planned obstacle bypass.
        Evaluate the arcs to all nearer path points over the same forward horizon;
        obstacle and road occupancy are costs, independent of track identity.
        """
        distances = 63.-path[:, 1]
        target_x = float(np.interp(lookahead, distances, path[:, 0]))
        lateral_m = (target_x-42.)/1.3608
        forward_m = lookahead/1.701
        preferred = 2.*lateral_m/(forward_m**2+lateral_m**2)
        if not obstacles.any():
            return preferred
        eligible = (distances >= 9.) & (distances <= lookahead)
        offsets = (path[eligible, 0]-42.)/1.3608
        forwards = distances[eligible]/1.701
        candidates = np.r_[preferred, 2.*offsets/(forwards*forwards+offsets*offsets)]
        sample_pixels = np.arange(6., lookahead+1, 1.)
        sample_m = sample_pixels/1.701
        turns = candidates[:, None]*sample_m[None]
        predicted = np.divide(1.-np.sqrt(np.maximum(1.-turns*turns, 0.)),
                              candidates[:, None], out=np.zeros_like(turns),
                              where=np.abs(candidates[:, None]) > 1e-7)
        x = 42.+predicted*1.3608
        xi = np.clip(np.rint(x), 0, 83).astype(int)
        yi = np.rint(63.-sample_pixels).astype(int)
        outside = (x < 0) | (x > 83) | (np.abs(turns) > .97)
        collision = obstacles[yi[None], xi] > 0
        offroad = (free[yi[None], xi] == 0) | outside
        reference = np.interp(sample_pixels, distances, path[:, 0])
        cost = (1000.*collision.sum(axis=1) + 10.*offroad.sum(axis=1)
                + .03*np.sum((x-reference[None])**2, axis=1)
                + 20.*(candidates-preferred)**2)
        return float(candidates[np.argmin(cost)])

    def _motion(self, previous, current):
        """Shared measurement concept independently validated by the rollout lane."""
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


    def _preview_curvature(self, curvature, speed, yaw_rate, slip, lookahead):
        distance = max(lookahead/1.701, 4.)
        dt = float(self.config['motion_preview'])
        return float(curvature + 2.*dt*(speed*curvature-yaw_rate)/distance
                     - 2.*speed*np.sin(slip)*dt/(distance*distance))

    def act(self, observation):
        obs = np.asarray(observation)
        if obs.shape != (4,84,84) or not np.isfinite(obs).all() or obs.min() < 0 or obs.max() > 1:
            self.diagnostics = {'valid': False}
            return np.array([0.,0.,.3], np.float32)
        frame = obs[-1].astype(np.float32)
        path, free, obstacles = self._path(frame)
        if len(path) < 3:
            self.diagnostics = {'valid': False, 'path': path.tolist()}
            return np.array([self.last_steer*.8,0.,.3], np.float32)
        c = self.config
        # Fixed HUD image calibration, independent of track and episode identity.
        mass = obs[-2:,77:83,10:13].sum(axis=(1,2)).mean()
        speed = float(np.clip((mass-.27)/.085, 0., 100.))
        distances = 63.-path[:,1]
        lookahead = float(np.clip(c['lookahead']+c['speed_lookahead']*speed, 9., distances[-1]))
        curvature = self._tracking_curvature(path, lookahead, free, obstacles)
        yaw_rate, slip, motion_valid = 0., 0., False
        if c['motion_preview'] > 0:
            yaw_rate, slip, motion_valid = self._motion(obs[-2], obs[-1])
            if motion_valid:
                curvature = self._preview_curvature(curvature, speed, yaw_rate, slip, lookahead)
        steer = float(np.clip(c['pursuit_gain']*curvature, -c['max_steer'], c['max_steer']))
        steer = (1.-c['steer_smoothing'])*steer + c['steer_smoothing']*self.last_steer
        # Fit overlapping local quadratics to suppress lattice quantization.
        kappas = np.zeros(len(path))
        for i in range(len(path)):
            lo, hi = max(0,i-3), min(len(path),i+4)
            if hi-lo >= 4:
                polynomial = np.polyfit(distances[lo:hi]/1.701, path[lo:hi,0]/1.3608, 2)
                slope = 2*polynomial[0]*distances[i]/1.701+polynomial[1]
                kappas[i] = abs(2*polynomial[0])/(1+slope*slope)**1.5
        caps = np.sqrt(c['lateral_accel']/np.maximum(kappas,.001))
        caps = np.clip(caps,c['min_speed'],c['max_speed'])
        # Pure pursuit also encodes urgently needed lateral correction.
        correction_cap = np.sqrt(c['lateral_accel']/max(abs(curvature),.001))
        target_speed = min(float(np.min(np.sqrt(caps*caps + 2*c['braking_accel']*distances/1.701))),float(correction_cap),c['max_speed'])
        if distances[-1] < 18:
            target_speed = min(target_speed,32.)
        target_speed = max(c['min_speed'],target_speed)
        excess = speed-target_speed
        gas = float(np.clip((target_speed-speed)*.10+.25,0.,1.)) if excess <= 1 else 0.
        brake = float(np.clip(excess*c['brake_gain'],.03,.7)) if excess > 1 else 0.
        self.last_steer = steer
        self.previous_path = path.copy()
        self.diagnostics = dict(valid=True,path=path.tolist(),speed=speed,target_speed=target_speed,
                                lookahead=lookahead,curvature=float(curvature),steer=steer,
                                obstacle_pixels=int(obstacles.sum()), yaw_rate=yaw_rate, slip=slip, motion_valid=motion_valid)
        return np.array([steer,gas,brake],np.float32)

    def last_step_diagnostics(self):
        return dict(self.diagnostics)
