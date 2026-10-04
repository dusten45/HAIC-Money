"""Coordinate high-speed pedals and steering using only rendered road cues."""
import numpy as np
from haic_agent.fixed_high_speed_recovery_v2 import FixedHighSpeedRecoveryV2
from haic_agent.pixel_features import road_centers

MODES=('control','preview_budget','actuator_rate','impact_traction','recovery_budget',
       'preview_action_aware','recovery_geometry_aware','preview_row_repair')


def observed_center(centers,row):
    if row in centers: return centers[row]
    known=sorted(centers,key=lambda y:abs(y-row))
    if len(known)<2: return None
    a,b=known[:2]
    return float(np.clip(centers[a]+(row-a)*(centers[b]-centers[a])/(b-a),0,83))


class FastCompletionCoordination(FixedHighSpeedRecoveryV2):
    def __init__(self,mechanism='control'):
        if mechanism not in MODES: raise ValueError('unknown mechanism')
        self.coordination_mode=mechanism
        super().__init__('impact_clear')

    def reset(self,observation=None):
        super().reset(observation)
        self.last_command=0.
        self.recovering=False
        self.stable=0
        self.brake_history=[]
        self.visible_steer=0.
        self.lost_rows=0

    def act(self,observation):
        impact_before=self.impact_left
        action=super().act(observation)
        original=action.copy()
        centers=self.last['road_centers']
        speed=self.last['pixel_speed']
        target=60.
        veto=False
        geometry_repaired=False
        if self.steps>10:
            if self.coordination_mode in ('preview_action_aware','recovery_geometry_aware','preview_row_repair'):
                veto=self.last['impact_proxy_trigger'] and any(b>.01 for b in self.brake_history)
                if veto:
                    action[0]=self.last['damping_steer']
                    self.impact_left=0
            if self.coordination_mode=='preview_row_repair' and 42 not in centers and 54 in centers and len(centers)>=2:
                m=observed_center(centers,42)
                obstacle=self.base._last_obstacle
                term=self.base._obstacle_side*.34*float(np.clip((obstacle[0]-22)/18,0,1)) if obstacle is not None else 0.
                action[0]=np.clip(.022*(m-42)+.018*(m-centers[54])+term+self.last['correction'],-.7,.7)
                geometry_repaired=True
            if self.coordination_mode=='recovery_geometry_aware':
                if 42 not in centers or 54 not in centers:
                    n,m=observed_center(centers,54),observed_center(centers,42)
                    if n is not None and m is not None:
                        stack=np.asarray(observation)
                        prev=road_centers(stack[-2]) if stack.ndim==3 else centers
                        pm=observed_center(prev,42)
                        obstacle=self.base._last_obstacle
                        clearing=(impact_before>0 or self.last['impact_proxy_trigger']) and not veto
                        term=self.base._obstacle_side*.34*float(np.clip((obstacle[0]-22)/18,0,1)) if obstacle is not None and not clearing else 0.
                        action[0]=np.clip(.022*(m-42)+.018*(m-n)+term+(.025*(m-pm) if pm is not None else 0.),-.7,.7)
                        self.visible_steer=float(action[0]);self.lost_rows=0
                        geometry_repaired=True
                    else:
                        self.lost_rows+=1
                        if self.lost_rows<=12:
                            action[0]=self.visible_steer
                            geometry_repaired=True
                else:
                    self.visible_steer=float(action[0]);self.lost_rows=0
            if self.coordination_mode in ('preview_budget','preview_action_aware','preview_row_repair'):
                sweep=max(centers.values())-min(centers.values()) if len(centers)>1 else 28.
                target=max(38.,60.-.8*sweep)
                if self.base._last_obstacle is not None: target=min(target,44.)
            elif self.coordination_mode=='actuator_rate':
                action[0]=np.clip(action[0],self.last_command-.24,self.last_command+.24)
            elif self.coordination_mode=='impact_traction':
                if impact_before>0 or self.last['impact_proxy_trigger']:
                    action[1]=min(action[1],.18)
            elif self.coordination_mode in ('recovery_budget','recovery_geometry_aware'):
                error=abs(centers.get(54,0)-42)
                if error>6 or 42 not in centers: self.recovering=True
                self.stable=self.stable+1 if len(centers)==7 and error<4 else 0
                if self.stable>=3: self.recovering=False
                if self.recovering: target=40.
            if target<60:
                action[1]=np.clip(.12+.04*(target-speed),0,.6)
                action[2]=np.clip(.02*(speed-target-2),0,.28)
        self.last_command=float(action[0])
        self.brake_history=(self.brake_history+[float(action[2])])[-4:]
        self.last.update(coordination_mode=self.coordination_mode,target_speed=target,
                         braking_proxy_veto=bool(veto),geometry_repaired=geometry_repaired,
                         completion_changed=bool(np.max(np.abs(action-original))>1e-6),
                         final_gas=float(action[1]),final_brake=float(action[2]))
        return action
