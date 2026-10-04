"""Earlier image-only obstacle intervention over the frozen sprint actor."""
import cv2
import numpy as np
from haic_agent.acceleration_envelope_runtime import AccelerationEnvelopeAgent
from haic_agent.pixel_features import current_frame, road_centers, center_at
from haic_agent.sprint72_relief_runtime import apply_sprint72_relief

MODES = ('control', 'sprint_reference', 'far_veto', 'temporal_hazard',
         'clearance_preview', 'arrival_speed')

def far_objects(frame):
    centers = road_centers(frame)
    if 30 not in centers:
        return [], centers
    previous = centers[30]
    asphalt = (frame >= .24) & (frame <= .52)
    for row in (26, 22, 18, 14, 10, 6):
        xs = np.flatnonzero(asphalt[row] & (np.abs(np.arange(84)-previous) <= 17))
        if len(xs) < 4:
            break
        previous = centers[row] = float(xs.mean())
    mask = (frame >= .54).astype(np.uint8)
    mask[:8] = 0
    mask[62:] = 0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    objects = []
    for i in range(1, count):
        _, _, width, height, area = stats[i]
        x, y = map(float, centroids[i])
        if (4 <= area <= 80 and 2 <= width <= 9 and 2 <= height <= 10
                and min(centers) <= y < 22 and abs(x-center_at(y, centers)) <= 12):
            objects.append((y, x, center_at(y, centers)))
    return sorted(objects, reverse=True), centers

class FarHazardAgent(AccelerationEnvelopeAgent):
    def __init__(self, mechanism='control'):
        if mechanism not in MODES:
            raise ValueError('unknown mechanism')
        self.far_mode = mechanism
        super().__init__('control' if mechanism == 'control' else 'swept_sprint')

    def reset(self, observation=None):
        super().reset(observation)
        self.track = None
        self.velocity = np.zeros(2)
        self.misses = 0
        self.matches = 0
        self.last_detection = None

    def act(self, observation):
        impact_before = self.impact_left
        action = super().act(observation)
        sprint = action.copy()
        frame = current_frame(observation)
        objects, centers = far_objects(frame)
        far = objects[0] if objects else None
        if far is not None:
            position = np.array(far[:2])
            if self.track is not None and np.linalg.norm(position-self.track) <= 12:
                self.velocity = (position-self.last_detection)/(self.misses+1)
                self.matches += 1
            else:
                self.velocity = np.zeros(2)
                self.matches = 0
            self.track, self.misses = position, 0
            self.last_detection = position.copy()
        elif self.track is not None:
            self.misses += 1
            predicted = self.track+self.velocity
            if self.misses <= 3 and self.matches > 0 and 8 <= predicted[0] < 62 and 0 <= predicted[1] < 84:
                self.track = predicted
            else:
                self.track = None
                self.matches = 0
        near = self.base._last_obstacle
        active = False
        side = clearance = cap = None
        apply_sprint72_relief(self, action, near, far, impact_before)
        if self.steps > 10:
            if self.far_mode in ('far_veto', 'temporal_hazard'):
                active = far is not None or (self.far_mode == 'temporal_hazard' and self.track is not None)
                if active:
                    action[1:] = self.last['parent_action'][1:]
            elif self.far_mode == 'clearance_preview' and far is not None and near is None:
                clearing = impact_before > 0 or self.last.get('impact_proxy_trigger', False)
                if not clearing:
                    y, x, center = far
                    row = int(round(y))
                    xs = np.flatnonzero((frame[row] >= .24) & (frame[row] <= .52) & (np.abs(np.arange(84)-center) <= 17))
                    if len(xs) >= 4:
                        left, right = float(xs.min()), float(xs.max())
                        options = [(min(x+s*9-left, right-(x+s*9)), s) for s in (-1, 1)]
                        clearance, side = max(options)
                        if clearance >= 2:
                            action[0] = np.clip(action[0]+np.clip(.018*(x+side*9-center), -.12, .12), -1, 1)
                            active = True
            elif self.far_mode == 'arrival_speed':
                obj = near if near is not None else far
                if obj is not None:
                    distance = max((63-obj[0])/1.701-7, 0)
                    cap = min(72., float(np.sqrt(44**2+110*distance)))
                    excess = self.last['pixel_speed']-cap
                    if excess > 0:
                        action[1], action[2] = 0., max(float(action[2]), float(np.clip(.04*excess, 0, .6)))
                        active = True
        self.brake_history[-1] = float(action[2])
        changed = bool(np.max(np.abs(action-np.asarray(self.last['parent_action']))) > 1e-6)
        self.last.update(far_mode=self.far_mode, far_objects=objects, near_object=near,
                         tracked_object=None if self.track is None else self.track.tolist(),
                         track_matches=self.matches, track_misses=self.misses,
                         far_active=bool(active), far_changed=bool(np.max(np.abs(action-sprint)) > 1e-6),
                         far_side=side, far_clearance=clearance, arrival_cap=cap,
                         completion_changed=changed, mechanism_changed=changed,
                         final_gas=float(action[1]), final_brake=float(action[2]))
        return action
