"""Pixel-only road-bounded passing paths and positive traction acceleration."""
import numpy as np

from haic_agent.fast_completion_coordination import FastCompletionCoordination
from haic_agent.pixel_features import current_frame

MODES = ('control', 'pedal_max', 'pursuit', 'tangent', 'rollout', 'launch_guarded')
PX_X, PX_Y, WHEELBASE = 1.3608, 1.701, 3.24


class GeometricPassingAgent(FastCompletionCoordination):
    def __init__(self, mechanism='rollout'):
        if mechanism not in MODES:
            raise ValueError('unknown high-acceleration direction')
        self.path_mode = mechanism
        super().__init__('preview_row_repair')

    def reset(self, observation=None):
        super().reset(observation)
        self.predicted_wheel = 0.
        self.passing_side = 0.
        self.passing_missing = 0
        self.launch_done = False

    def _corridor(self, frame):
        result = []
        anchor, slope = 42., 0.
        for row in range(54, 9, -4):
            expected = anchor + slope
            columns = np.flatnonzero((frame[row] >= .24) & (frame[row] <= .52)
                                     & (np.abs(np.arange(84)-expected) <= 17))
            if len(columns) < 4:
                continue
            center = float(columns.mean())
            slope = float(np.clip(center-anchor, -8, 8))
            anchor = center
            result.append(((63-row)/PX_Y, (center-42)/PX_X,
                           (float(columns[0])-42)/PX_X, (float(columns[-1])-42)/PX_X))
        return np.asarray(result, dtype=float)

    def _passing_path(self, corridor, obstacle):
        self.passing_status = 'no_obstacle' if obstacle is None else 'insufficient_geometry'
        if obstacle is None:
            self.passing_missing += 1
            if self.passing_missing >= 2:
                self.passing_side = 0.
            return corridor, None
        self.passing_missing = 0
        if len(corridor) < 2:
            return corridor, None
        forward, center, left, right = corridor.T
        ox, oy = (obstacle[1]-42)/PX_X, (63-obstacle[0])/PX_Y
        lo, hi = float(np.interp(oy, forward, left)+1.2), float(np.interp(oy, forward, right)-1.2)
        if lo > hi:
            return corridor, None
        if not self.passing_side:
            choices = [(side, float(np.clip(ox+3.*side, lo, hi))) for side in (-1., 1.)]
            self.passing_side = min(choices, key=lambda pair: (abs(pair[1]-ox)<2.9, abs(pair[1])))[0]
        passing_x = float(np.clip(ox+3.*self.passing_side, lo, hi))
        if abs(passing_x-ox) < 2.9:
            self.passing_status = 'insufficient_clearance'
            return corridor, None
        self.passing_status = 'proposed'
        offset = passing_x-float(np.interp(oy, forward, center))
        result = corridor.copy()
        lower, upper = left+1.2, right-1.2
        middle = (lower+upper)/2
        result[:, 1] = np.clip(center+offset*np.exp(-.5*((forward-oy)/8.)**2),
                               np.minimum(lower,middle), np.maximum(upper,middle))
        return result, (passing_x, oy, abs(passing_x-ox))

    def act(self, observation):
        action = super().act(observation)
        baseline = action.copy()
        frame = current_frame(observation)
        # Same public speed bar calibration, extended upwards; no artificial 80 cap.
        speed = max(0., (float(frame[74:83, 10:13].sum())-.27)/.085)
        if self.path_mode == 'launch_guarded':
            self.launch_done = self.launch_done or speed >= 40.
            centers = self.last['road_centers']
            straight = (len(centers) == 7 and max(centers.values())-min(centers.values()) < 3.
                        and abs(centers[54]-42) < 2. and self.base._last_obstacle is None and speed < 70.)
            if not self.launch_done:
                action[1], action[2] = 1., 0.
            elif straight:
                action[1], action[2] = max(float(action[1]), .55), 0.
            self.brake_history[-1] = float(action[2])
            self.last.update(path_mode=self.path_mode, launch_done=self.launch_done,
                             straight_acceleration=bool(straight), unbounded_hud_speed=speed,
                             completion_changed=bool(np.max(np.abs(action-baseline))>1e-6),
                             final_gas=float(action[1]), final_brake=float(action[2]))
            return action
        corridor = self._corridor(frame)
        passing = None
        passing_applied = False
        self.passing_status = 'control'
        if self.path_mode != 'control':
            corridor, passing = self._passing_path(corridor, self.base._last_obstacle)
        path_points, selected_cost = [], None
        valid = len(corridor) >= 5 and corridor[0, 0] < 9.
        if self.path_mode != 'control':
            action[1], action[2] = 1., 0.
            if self.path_mode == 'pedal_max' and passing is not None:
                centers = self.last['road_centers']
                if 42 in centers and 54 in centers:
                    passing_applied = True
                    px, py, clearance = passing
                    ox = (self.base._last_obstacle[2]-42)/PX_X
                    passing_bearing = np.arctan(WHEELBASE*2*px/(py*py+px*px))
                    road_bearing = np.arctan(WHEELBASE*2*ox/(py*py+ox*ox))
                    road_steer = .022*(centers[42]-42)+.018*(centers[42]-centers[54])+self.last['correction']
                    action[0] = np.clip(road_steer+passing_bearing-road_bearing, -.7, .7)
            if self.path_mode != 'pedal_max' and valid:
                passing_applied = passing is not None
                forward, center, left, right = corridor.T
                coefficients = np.polyfit(forward, center, 2)
                near_error = float(np.polyval(coefficients, 0.))
                lookahead = float(np.clip(.22*speed+5., 9., forward[-1]))
                obstacle = self.base._last_obstacle
                avoid = 0.
                if self.path_mode == 'pursuit':
                    lateral = float(np.interp(lookahead, forward, center))
                    steer = np.arctan(WHEELBASE*2*lateral/(lookahead**2+lateral**2)) + avoid
                elif self.path_mode == 'tangent':
                    slope = 2*coefficients[0]*lookahead+coefficients[1]
                    curvature = 2*coefficients[0]/(1+slope*slope)**1.5
                    steer = np.arctan(WHEELBASE*curvature) + .55*np.arctan(slope) + np.arctan(.9*near_error/max(speed, 15.)) + avoid
                else:
                    # Smooth a path within the observed road width, anchored at the car.
                    distances = np.linspace(0., forward[-1], 17)
                    target = np.interp(distances, forward, center)
                    lower = np.interp(distances, forward, left)+1.2
                    upper = np.interp(distances, forward, right)-1.2
                    middle = (lower+upper)/2
                    lower, upper = np.minimum(lower, middle), np.maximum(upper, middle)
                    path = target.copy()
                    for _ in range(20):
                        second = path[:-2]-2*path[1:-1]+path[2:]
                        gradient = np.zeros_like(path)
                        gradient[:-2] += second
                        gradient[1:-1] -= 2*second
                        gradient[2:] += second
                        path = np.clip(path-.05*gradient-.003*(path-target), lower, upper)
                        path[0] = 0.
                    commands = np.linspace(-.4, .4, 41)
                    x, y, heading = np.zeros(41), np.zeros(41), np.zeros(41)
                    wheel = np.full(41, self.predicted_wheel)
                    cost = .03*(commands-self.predicted_wheel)**2
                    effective_speed = max(speed, 8.)
                    dt = min(.08, forward[-1]/(6*effective_speed))
                    for _ in range(6):
                        wheel += np.clip(commands-wheel, -3*dt, 3*dt)
                        yaw_rate = np.clip(effective_speed*np.tan(wheel)/WHEELBASE,
                                           -150/effective_speed, 150/effective_speed)
                        heading += yaw_rate*dt
                        x += effective_speed*np.sin(heading)*dt
                        y += effective_speed*np.cos(heading)*dt
                        reference = np.interp(y, distances, path)
                        lo, hi = np.interp(y, distances, lower), np.interp(y, distances, upper)
                        cost += (x-reference)**2 + 40*(np.maximum(lo-x, 0)**2+np.maximum(x-hi, 0)**2)
                        if obstacle is not None:
                            ox, oy = (obstacle[1]-42)/PX_X, (63-obstacle[0])/PX_Y
                            clearance = np.hypot(x-ox, y-oy)
                            cost += 100*np.maximum(3.-clearance, 0)**2
                    cost -= .3*y
                    best = int(np.argmin(cost))
                    steer, selected_cost = commands[best], float(cost[best])
                    path_points = [[float(s), float(x)] for s, x in zip(distances, path)]
                action[0] = np.clip(steer, -.7, .7)
        if self.path_mode != 'control':
            lateral_fraction = min(.995, speed*speed*abs(np.tan(np.clip(action[0], -.4, .4)))/WHEELBASE/150.)
            grip_fraction = np.sqrt(max(.01, 1.-lateral_fraction*lateral_fraction))
            action[1], action[2] = np.clip(.0054*(speed/.54+5.)*grip_fraction, .03, 1.), 0.
        self.predicted_wheel += float(np.clip(np.clip(action[0], -.4, .4)-self.predicted_wheel, -.24, .24))
        self.brake_history[-1] = float(action[2])
        self.last.update(path_mode=self.path_mode, unbounded_hud_speed=speed,
                         passing_point=passing, passing_side=self.passing_side,
                         passing_status=self.passing_status, passing_applied=passing_applied,
                         steering_delta_from_baseline=float(action[0]-baseline[0]),
                         corridor_valid=bool(valid), metric_corridor=corridor.tolist(),
                         planned_path=path_points, rollout_cost=selected_cost,
                         predicted_wheel=self.predicted_wheel,
                         completion_changed=bool(np.max(np.abs(action-baseline)) > 1e-6),
                         final_gas=float(action[1]), final_brake=float(action[2]))
        return action
