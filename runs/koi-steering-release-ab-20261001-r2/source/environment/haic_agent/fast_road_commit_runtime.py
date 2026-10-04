"""Keep fast acceleration while separating avoidance and collision-recovery state."""
import numpy as np
from haic_agent.fast_completion_coordination import FastCompletionCoordination, observed_center
from haic_agent.pixel_features import current_frame
from haic_agent.geometric_passing_runtime import GeometricPassingAgent, PX_X, PX_Y, WHEELBASE

MODES = ('control', 'fast_control', 'commit_side', 'passing_bearing', 'shock_recovery', 'continuous_avoidance')


class FastRoadCommitAgent(FastCompletionCoordination):
    def __init__(self, mechanism='shock_recovery'):
        if mechanism not in MODES:
            raise ValueError('unknown mechanism')
        self.fast_mode = mechanism
        super().__init__('preview_row_repair')

    def reset(self, observation=None):
        super().reset(observation)
        self.launch_done = False
        self.committed_side = 0.
        self.missing_obstacle = 0
        self.avoid_component = 0.
        self.previous_hud_speed = None
        self.previous_issued_brake = 0.
        self.shock_left = 0

    def act(self, observation):
        impact_before = self.impact_left
        action = super().act(observation)
        baseline = action.copy()
        frame = current_frame(observation)
        speed = max(0., (float(frame[74:83, 10:13].sum())-.27)/.085)
        centers = self.last['road_centers']
        obstacle = self.base._last_obstacle
        obstacle_term = self.base._obstacle_side*.34*float(np.clip((obstacle[0]-22)/18, 0, 1)) if obstacle is not None else 0.
        desired_term = obstacle_term
        middle = observed_center(centers,42) if 54 in centers and len(centers)>=2 else None
        road = (.022*(middle-42)+.018*(middle-centers[54])) if middle is not None else None
        clearing = (impact_before>0 or self.last['impact_proxy_trigger']) and not self.last['braking_proxy_veto'] and 42 in centers and 54 in centers
        def steer_with_term(term):
            if self.last['geometry_repaired'] or clearing:
                return float(np.clip(road+term+self.last['correction'],-.7,.7))
            return float(np.clip(np.clip(road+term,-.7,.7)+self.last['correction'],-.7,.7))
        applied = False
        passing = None
        shock = False
        if obstacle is None:
            self.missing_obstacle += 1
            if self.missing_obstacle >= 2:
                self.committed_side = 0.
        else:
            self.missing_obstacle = 0
            if self.committed_side == 0.:
                self.committed_side = float(self.base._obstacle_side)
        if self.fast_mode in ('commit_side', 'passing_bearing') and obstacle is not None:
            desired_term = self.committed_side*.34*float(np.clip((obstacle[0]-22)/18,0,1))
            if self.fast_mode == 'passing_bearing':
                corridor = GeometricPassingAgent._corridor(self, frame)
                if len(corridor)>=2:
                    forward, center, left, right = corridor.T
                    ox, oy = (obstacle[1]-42)/PX_X, (63-obstacle[0])/PX_Y
                    lo, hi = np.interp(oy,forward,left)+1.2, np.interp(oy,forward,right)-1.2
                    if lo<=hi:
                        px = float(np.clip(ox+3*self.committed_side,lo,hi))
                        if abs(px-ox)>=2.9:
                            cx = float(np.interp(oy,forward,center))
                            desired_term = float(np.arctan(WHEELBASE*2*px/(oy*oy+px*px))-np.arctan(WHEELBASE*2*cx/(oy*oy+cx*cx)))
                            passing = (px,oy)
            if road is not None:
                action[0] = steer_with_term(0. if clearing else desired_term)
                applied = True
        elif self.fast_mode == 'continuous_avoidance':
            desired_term = 0. if clearing else obstacle_term
            self.avoid_component += float(np.clip(desired_term-self.avoid_component,-.08,.08))
            if road is not None:
                action[0] = steer_with_term(self.avoid_component)
                applied = True
        elif self.fast_mode == 'shock_recovery':
            braking_allowance = 240*self.previous_issued_brake*.08
            shock = self.previous_hud_speed is not None and self.previous_hud_speed-speed > 10.+braking_allowance
            if shock:
                self.shock_left = 12
            if self.shock_left>0:
                if road is not None:
                    action[0] = np.clip(road+self.last['correction'],-.7,.7)
                    applied = True
                self.shock_left -= 1
        if self.fast_mode != 'control':
            self.launch_done = self.launch_done or speed>=40.
            straight = (len(centers)==7 and max(centers.values())-min(centers.values())<3.
                        and abs(centers[54]-42)<2. and obstacle is None and speed<70.)
            if not self.launch_done:
                action[1],action[2] = 1.,0.
            elif straight:
                action[1],action[2] = max(float(action[1]),.55),0.
        else:
            straight = False
        self.brake_history[-1] = float(action[2])
        self.previous_issued_brake = float(action[2])
        self.previous_hud_speed = speed
        self.last.update(baseline_action=baseline.tolist(),fast_mode=self.fast_mode, unbounded_hud_speed=speed,
                         straight_acceleration=bool(straight),launch_done=self.launch_done,
                         original_obstacle_term=obstacle_term,requested_obstacle_term=desired_term,
                         filtered_obstacle_term=self.avoid_component,committed_side=self.committed_side,
                         observed_obstacle=obstacle, road_relative_side=float(self.base._obstacle_side),
                         passing_point=passing,steering_repair_applied=applied,
                         shock_trigger=bool(shock),shock_remaining=self.shock_left,
                         steering_delta_from_baseline=float(action[0]-baseline[0]),
                         completion_changed=bool(np.max(np.abs(action-baseline))>1e-6),
                         final_gas=float(action[1]),final_brake=float(action[2]))
        return action
