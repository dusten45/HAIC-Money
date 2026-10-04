"""Pixel-only propulsion timing and projected free-space acceleration."""
import numpy as np
from haic_agent.fast_completion_coordination import FastCompletionCoordination
from haic_agent.pixel_features import current_frame
MODES=('control','launch_full','exit_burst','swept_sprint','horizon_brake')

def projected_free_distance(frame, speed, steer):
    x=y=heading=distance=0.
    last_clear=0.
    samples=0
    for _ in range(24):
        heading += max(speed,10.)*.04*np.tan(np.clip(steer,-.4,.4))/3.24
        x += max(speed,10.)*.04*np.sin(heading)
        y += max(speed,10.)*.04*np.cos(heading)
        distance += max(speed,10.)*.04
        px,py=int(round(42.+1.3608*x)),int(round(63.-1.701*y))
        if py>55: continue
        if py<6 or px<4 or px>79: break
        footprint=frame[py-1:py+2,px-3:px+4]
        if float(np.mean((footprint>=.24)&(footprint<=.52)))<.85: break
        samples+=1
        last_clear=distance
    return last_clear,samples

class AccelerationEnvelopeAgent(FastCompletionCoordination):
    def __init__(self,mechanism='control'):
        if mechanism not in MODES: raise ValueError('unknown mechanism')
        self.accel_mode=mechanism
        super().__init__('preview_row_repair')
    def reset(self,observation=None):
        super().reset(observation)
        self.launch_open=True
        self.sweeps=[]
        self.burst_left=0
    def act(self,observation):
        action=super().act(observation)
        parent=action.copy()
        centers=self.last['road_centers']; speed=self.last['pixel_speed']
        obstacle=self.base._last_obstacle
        sweep=max(centers.values())-min(centers.values()) if len(centers)>1 else 99.
        safe=(obstacle is None and 42 in centers and 54 in centers and abs(centers[54]-42.)<3.)
        active=False; distance=0.;samples=0;target=None
        if self.accel_mode=='launch_full':
            if not (self.steps<=20 and speed<40. and abs(float(action[0]))<.08 and obstacle is None):
                self.launch_open=False
            if self.launch_open:
                action[1],action[2]=1.,0.;active=True;target=40.
        elif self.steps>10:
            if self.accel_mode=='exit_burst':
                exit_phase=(len(self.sweeps)>=2 and all(v is not None for v in self.sweeps[-2:]) and len(centers)==7 and self.sweeps[-2]>self.sweeps[-1]>sweep and self.sweeps[-2]>3.)
                if safe and abs(float(action[0]))<.12 and exit_phase: self.burst_left=6
                if not safe or abs(float(action[0]))>=.12 or speed>=60.: self.burst_left=0
                if self.burst_left>0:
                    action[1],action[2]=1.,0.;active=True;target=60.;self.burst_left-=1
            elif self.accel_mode in ('swept_sprint','horizon_brake'):
                proposal=max(speed,72.) if self.accel_mode=='swept_sprint' else speed
                distance,samples=projected_free_distance(current_frame(observation),proposal,float(action[0]))
                if self.accel_mode=='swept_sprint':
                    if safe and samples>=4 and abs(float(action[0]))<.18 and distance>=max(18.,proposal*.4) and speed<72.:
                        action[1],action[2]=1.,0.;active=True;target=72.
                elif safe and samples>=4:
                    target=float(np.clip(np.sqrt(42.**2+110.*max(distance-7.,0.)),42.,80.))
                    action[2]=float(np.clip((speed-target)*.06,0.,.8))
                    action[1]=0. if action[2]>0 else (1. if speed<target-1. else .15)
                    active=True
        self.sweeps=(self.sweeps+[sweep if len(centers)==7 else None])[-3:]
        self.brake_history[-1]=float(action[2])
        self.last.update(accel_mode=self.accel_mode,mechanism_active=bool(active),
                         mechanism_changed=bool(np.max(np.abs(action-parent))>1e-6),
                         completion_changed=bool(np.max(np.abs(action-parent))>1e-6),
                         parent_action=parent.tolist(),free_distance=distance,free_samples=samples,
                         acceleration_target=target,launch_open=self.launch_open,burst_left=self.burst_left,
                         final_gas=float(action[1]),final_brake=float(action[2]))
        return action
