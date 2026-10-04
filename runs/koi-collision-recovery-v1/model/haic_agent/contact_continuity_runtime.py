"""Independent pixel-only mechanisms for repeated-contact failure."""
import numpy as np
from haic_agent.far_hazard_runtime import FarHazardAgent
from haic_agent.fast_completion_coordination import observed_center

MODES = ('control', 'arrival_reference', 'impact_override', 'contact_coast',
         'side_hysteresis', 'crossing_projection')

class ContactContinuityAgent(FarHazardAgent):
    def __init__(self, mechanism='control'):
        if mechanism not in MODES:
            raise ValueError('unknown mechanism')
        self.contact_mode = mechanism
        super().__init__('control' if mechanism=='control' else 'arrival_speed')

    def reset(self, observation=None):
        super().reset(observation)
        self.contact_speeds = []
        self.contact_left = 0
        self.pass_side = None
        self.absent = 0
        self.previous_object = None

    def act(self, observation):
        impact_before = self.impact_left
        action = super().act(observation)
        parent = action.copy()
        speed = self.last['pixel_speed']
        obj = self.base._last_obstacle
        centers = self.last['road_centers']
        near, middle = observed_center(centers,54), observed_center(centers,42)
        road = None if near is None or middle is None else .022*(middle-42)+.018*(middle-near)+self.last['correction']
        def avoid_steer(term):
            if self.last['geometry_repaired']:
                return np.clip(road+term,-.7,.7)
            return np.clip(np.clip(road-self.last['correction']+term,-.7,.7)+self.last['correction'],-.7,.7)
        proxy = bool(self.contact_speeds and max(self.contact_speeds)-speed>12
                     and speed<35 and obj is not None and obj[0]>=48)
        if proxy and self.contact_left==0:
            self.contact_left = 12 if self.contact_mode=='impact_override' else 8
        self.contact_speeds = (self.contact_speeds+[speed])[-3:]
        if obj is None:
            self.absent += 1
            if self.absent >= 3:
                self.pass_side = None
        else:
            self.absent = 0
            if obj[0]>=61:
                self.pass_side = None
            elif obj[0]>=30 and self.pass_side is None:
                self.pass_side = float(self.base._obstacle_side)
        active = False
        projected = None
        genuine_clear = (impact_before>0 or self.last['impact_proxy_trigger']) and not self.last['braking_proxy_veto']
        if self.steps>10:
            if self.contact_mode=='impact_override' and self.contact_left>0 and road is not None:
                action[0] = np.clip(road,-.7,.7)
                active = True
            elif self.contact_mode=='contact_coast' and self.contact_left>0 and obj is not None and obj[0]>=48 and speed<35:
                action[1:] = 0.
                active = True
            elif self.contact_mode=='side_hysteresis' and obj is not None and self.pass_side is not None and 42 in centers and 54 in centers and not genuine_clear:
                urgency = float(np.clip((obj[0]-22)/18,0,1))
                action[0] = avoid_steer(self.pass_side*.34*urgency)
                active = True
            elif self.contact_mode=='crossing_projection' and obj is not None and self.previous_object is not None and road is not None and not genuine_clear:
                y,x,_ = obj
                py,px,_ = self.previous_object
                if 30<=y<=55 and y-py>1 and np.hypot(y-py,x-px)<=15:
                    projected = float(x+(x-px)*(60-y)/(y-py))
                    if abs(projected-42)<6:
                        urgency = float(np.clip((y-22)/18,0,1))
                        action[0] = avoid_steer(self.base._obstacle_side*.55*urgency)
                        active = True
        self.previous_object = obj
        if self.contact_left:
            self.contact_left -= 1
        self.brake_history[-1] = float(action[2])
        self.last.update(contact_mode=self.contact_mode, contact_proxy=proxy,
                         contact_remaining=self.contact_left, pass_side=self.pass_side,
                         projected_obstacle_x=projected, contact_active=active,
                         contact_changed=bool(np.max(np.abs(action-parent))>1e-6),
                         mechanism_changed=bool(np.max(np.abs(action-np.asarray(self.last['parent_action'])))>1e-6),
                         completion_changed=bool(np.max(np.abs(action-np.asarray(self.last['parent_action'])))>1e-6),
                         final_gas=float(action[1]),final_brake=float(action[2]))
        return action
