"""Pixel racing-line phases with measured-width offsets and exit acceleration."""
import numpy as np
from haic_agent.fast_completion_coordination import FastCompletionCoordination, observed_center
from haic_agent.pixel_features import current_frame,road_centers

MODES=('control','line_only','line_fast','unwind','radius_speed','exit_boost')


class OutInOutAgent(FastCompletionCoordination):
    def __init__(self,mechanism='line_fast'):
        if mechanism not in MODES:raise ValueError('unknown out-in-out mechanism')
        self.line_mode=mechanism
        super().__init__('preview_row_repair')

    def reset(self,observation=None):
        super().reset(observation)
        self.phase='straight'
        self.phase_age=0
        self.bend_direction=0.
        self.line_offset=0.
        self.committed_side=0.
        self.obstacle_missing=0
        self.straight_frames=0.

    def act(self,observation):
        prior_impact=self.impact_left
        action=super().act(observation)
        original=action.copy()
        centers=self.last['road_centers']
        frame=current_frame(observation)
        extended=dict(centers)
        if 30 in centers:
            anchor=centers[30]
            for y in (26,22):
                locations=np.flatnonzero((frame[y]>=.24)&(frame[y]<=.52)&(np.abs(np.arange(84)-anchor)<=17))
                if len(locations)>=4:
                    anchor=float(locations.mean());extended[y]=anchor
        valid=all(y in centers for y in (30,42,54))
        curve=0.;near_heading=0.;far_heading=0.;width=0.
        if valid:
            n,m,f=centers[54],centers[42],centers[30]
            near_heading=m-n;far_heading=f-m
            curve=f-2*m+n
            preview_curve=extended.get(22,f)-2*centers.get(38,m)+n if 22 in extended and 38 in centers else curve
            locations=np.flatnonzero((frame[54]>=.24)&(frame[54]<=.52)&(np.abs(np.arange(84)-n)<=17))
            width=float(locations[-1]-locations[0])/2 if len(locations)>=8 else 0.
            if self.steps>10:
                self.phase_age+=1
                self.straight_frames=self.straight_frames+1 if abs(curve)<1 and abs(near_heading)<3 else 0
                if self.phase=='straight' and abs(preview_curve)>3:
                    self.bend_direction=float(np.sign(preview_curve));self.phase='entry';self.phase_age=0
                elif self.phase=='entry' and self.phase_age>=2 and (self.bend_direction*near_heading>3 or self.phase_age>=8):
                    self.phase='apex';self.phase_age=0
                elif self.phase=='apex' and self.phase_age>=2 and (self.bend_direction*(far_heading-near_heading)<-1 or self.straight_frames>=2 or self.phase_age>=18):
                    self.phase='exit';self.phase_age=0
                elif self.phase=='exit' and (abs(f-n)<4 or self.phase_age>=10):
                    self.phase='straight';self.phase_age=0;self.bend_direction=0.
        else:
            self.straight_frames=0
        detected=self.base._last_obstacle
        obstacle=detected is not None
        replaced=False
        obstacle_term=0.
        if obstacle:
            self.obstacle_missing=0
            if self.committed_side==0.:
                self.committed_side=1. if detected[1]<42 else -1.
            obstacle_term=self.committed_side*.34*float(np.clip((detected[0]-22)/18,0,1))
        else:
            self.obstacle_missing+=1
            if self.obstacle_missing>=2:self.committed_side=0.
        clearing=(prior_impact>0 or self.last['impact_proxy_trigger']) and not self.last['braking_proxy_veto']
        if self.steps>10 and self.line_mode in ('line_only','line_fast') and obstacle and not clearing and self.committed_side!=self.base._obstacle_side:
            n=centers.get(54,42.)
            m=observed_center(centers,42) if self.last['geometry_repaired'] else centers.get(42,42.)
            if m is not None:
                road=.022*(m-42)+.018*(m-n)
                raw=road+obstacle_term
                if not self.last['geometry_repaired']:raw=float(np.clip(raw,-.7,.7))
                action[0]=np.clip(raw+self.last['correction'],-.7,.7)
                replaced=True
        desired=0.
        amplitude=max(0.,min(.35*width,width-3.,5.))
        if valid and not obstacle and self.phase!='straight':
            desired=self.bend_direction*amplitude*(1 if self.phase=='apex' else -1)
        prior_offset=self.line_offset
        self.line_offset=float(np.clip(self.line_offset+float(np.clip(desired-self.line_offset,-1.2,1.2)),-amplitude,amplitude))
        target=self.last['target_speed']
        if self.steps>10:
            if self.line_mode in ('line_only','line_fast') and valid and not obstacle:
                action[0]=np.clip(float(action[0])+.022*self.line_offset+.025*(self.line_offset-prior_offset),-.7,.7)
            elif self.line_mode=='unwind' and valid:
                stack=np.asarray(observation)
                previous=road_centers(stack[-2]) if stack.ndim==3 else centers
                motion=centers[54]-previous.get(54,centers[54])
                if (centers[54]-42)*motion<0 and float(action[0])*motion<0:
                    action[0]*=.65
            if self.line_mode=='line_fast' and valid and not obstacle:
                target=min(66.,self.last['target_speed']+6.)
            elif self.line_mode=='radius_speed' and valid:
                target=float(np.clip(66.-2.*abs(curve),48.,66.))
                if obstacle:target=min(target,48.)
            elif self.line_mode=='exit_boost' and valid and not obstacle:
                if self.phase=='exit' or (abs(curve)<1 and abs(near_heading)<3):target=66.
            if self.line_mode in ('line_fast','radius_speed','exit_boost') and target!=self.last['target_speed']:
                speed=self.last['pixel_speed']
                action[1]=np.clip(.12+.05*(target-speed),0,.8)
                action[2]=np.clip(.02*(speed-target-2),0,.28)
        # The inherited impact veto must see actual issued brakes, not overwritten ones.
        self.brake_history[-1]=float(action[2])
        self.last.update(obstacle_x=detected[1] if obstacle else None,obstacle_y=detected[0] if obstacle else None,
                         committed_side=self.committed_side,avoidance_replaced=replaced,committed_obstacle_term=obstacle_term,
                         line_mode=self.line_mode,line_phase=self.phase,bend_direction=self.bend_direction,
                         desired_line_offset=desired,line_offset=self.line_offset,
                         observed_lateral_offset=42-centers[54] if 54 in centers else None,
                         visible_half_width=width,near_heading=near_heading,far_heading=far_heading,
                         road_curvature=curve,extended_road_centers=extended,target_speed=target,
                         completion_changed=bool(np.max(np.abs(action-original))>1e-6),
                         final_gas=float(action[1]),final_brake=float(action[2]))
        return action
