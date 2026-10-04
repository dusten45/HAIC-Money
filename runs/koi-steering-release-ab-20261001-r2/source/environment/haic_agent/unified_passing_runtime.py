"""Independent image-only passing, motion observation and distance-speed candidates."""
import numpy as np
from haic_agent.fast_completion_coordination import FastCompletionCoordination, observed_center
MODES = ('control','unified_gap','predicted_gap','lateral_observer','distance_speed')

class UnifiedPassingAgent(FastCompletionCoordination):
    def __init__(self, mechanism='control'):
        if mechanism not in MODES: raise ValueError('unknown mechanism')
        self.unified_mode = mechanism
        super().__init__('preview_row_repair')
    def reset(self, observation=None):
        super().reset(observation)
        self.passing_side = 0.
        self.previous_object = None
        self.object_missing = 0
        self.previous_near = None
        self.lateral_motion = 0.
    def act(self, observation):
        impact_before = self.impact_left
        action = super().act(observation)
        original = action.copy()
        centers = self.last['road_centers']
        obj = self.base._last_obstacle
        matching = (obj is not None and self.previous_object is not None and self.object_missing == 0
                    and abs(obj[0]-self.previous_object[0]) < 20 and abs(obj[1]-self.previous_object[1]) < 25)
        predicted_offset = None
        if obj is not None:
            offset = obj[1]-obj[2]
            velocity = offset-(self.previous_object[1]-self.previous_object[2]) if matching else 0.
            predicted_offset = offset+float(np.clip(2.*velocity,-8.,8.)) if self.unified_mode == 'predicted_gap' else offset
            if self.passing_side == 0. or (self.previous_object is not None and not matching and self.object_missing == 0):
                self.passing_side = -1. if predicted_offset > 0. else 1.
            self.previous_object = obj
            self.object_missing = 0
        else:
            self.object_missing += 1
            if self.object_missing >= 4:
                self.passing_side = 0.
                self.previous_object = None
        near = centers.get(54)
        if near is not None and self.previous_near is not None:
            self.lateral_motion = .6*self.lateral_motion+.4*(near-self.previous_near)
        else:
            self.lateral_motion = 0.
        self.previous_near = near
        active = False
        target = None
        replacement_road = replacement_avoid = nominal_inherited_avoid = None
        clearing = (impact_before > 0 or self.last['impact_proxy_trigger']) and not self.last['braking_proxy_veto']
        if self.steps > 10:
            if self.unified_mode in ('unified_gap','predicted_gap') and obj is not None and not clearing and near is not None and len(centers) >= 2:
                middle = observed_center(centers,42)
                road = .022*(middle-42.)+.018*(middle-near)
                urgency = float(np.clip((obj[0]-22.)/18.,0.,1.))
                avoid = float(np.clip(.04*(predicted_offset+self.passing_side*9.),-.34,.34))*urgency
                replacement_road, replacement_avoid = road, avoid
                nominal_inherited_avoid = float(self.base._obstacle_side)*.34*urgency
                action[0] = np.clip(road+avoid+self.last['correction'],-.7,.7)
                active = True
            elif self.unified_mode == 'lateral_observer' and obj is None and 42 in centers and near is not None and not clearing:
                action[0] = np.clip(action[0]+np.clip(.044*self.lateral_motion,-.12,.12),-.7,.7)
                active = True
            elif self.unified_mode == 'distance_speed' and len(centers) >= 3:
                pts = np.asarray([[(centers[y]-42.)/1.3608,(63.-y)/1.701] for y in sorted(centers,reverse=True)])
                segments = np.diff(pts,axis=0)
                lengths = np.linalg.norm(segments,axis=1)
                angles = np.arctan2(segments[:,0],segments[:,1])
                changes = np.abs((np.diff(angles)+np.pi)%(2*np.pi)-np.pi)
                curvature = changes/np.maximum((lengths[:-1]+lengths[1:])*.5,1.)
                targets = [72.]
                for i,k in enumerate(curvature):
                    bend = float(np.clip(np.sqrt(160./max(float(k),1e-5)),35.,72.))
                    distance = max(float(pts[i+1,1])-5.,0.)
                    targets.append(float(np.sqrt(bend*bend+90.*distance)))
                if obj is not None:
                    distance = max((63.-obj[0])/1.701-7.,0.)
                    targets.append(float(np.sqrt(42.*42.+90.*distance)))
                target = min(targets)
                speed = self.last['pixel_speed']
                action[2] = np.clip(.025*(speed-target-1.),0.,.5)
                action[1] = 0. if action[2] > 0. else np.clip(.12+.04*(target-speed),0.,.8)
                active = True
        self.brake_history[-1] = float(action[2])
        self.last.update(unified_mode=self.unified_mode,mechanism_active=bool(active),
                         mechanism_changed=bool(np.max(np.abs(action-original)) > 1e-6),
                         completion_changed=bool(np.max(np.abs(action-original)) > 1e-6),
                         parent_action=original.tolist(),passing_side=self.passing_side,
                         inherited_side=float(self.base._obstacle_side),predicted_offset=predicted_offset,
                         observed_obstacle=None if obj is None else list(obj),object_matching=bool(matching),
                         lateral_motion=self.lateral_motion,distance_speed_target=target,
                         replacement_road=replacement_road,replacement_avoid=replacement_avoid,
                         nominal_inherited_avoid=nominal_inherited_avoid,
                         final_gas=float(action[1]),final_brake=float(action[2]))
        return action
