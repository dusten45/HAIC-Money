"""Connected pixel road paths for preview steering; no simulator state input."""
import cv2
import numpy as np

from haic_agent.fast_completion_coordination import FastCompletionCoordination, observed_center
from haic_agent.pixel_features import current_frame

MODES = ('control', 'connected_pursuit', 'dual_preview', 'motion_memory', 'shared_path')
SCALE = np.asarray([1.3608, 1.701], dtype=np.float32)
ORIGIN = np.asarray([42., 63.], dtype=np.float32)


def connected_path(frame):
    mask = ((frame >= .24) & (frame <= .52)).astype(np.uint8)
    mask[61:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask[[0, -1], :] = 0
    mask[:, [0, -1]] = 0
    _, labels = cv2.connectedComponents(mask, connectivity=8)
    choices = np.flatnonzero(mask[54])
    choices = choices[np.abs(choices-42) <= 20]
    if not len(choices):
        return np.empty((0, 2)), mask
    seed_x = int(choices[np.argmin(np.abs(choices-42))])
    component = (labels == labels[54, seed_x]).astype(np.uint8)
    distance = cv2.distanceTransform(component, cv2.DIST_L2, 3)
    # Start at a central point of the anchored road, not a different component.
    xs = np.flatnonzero(component[54] & (np.abs(np.arange(84)-seed_x) <= 12))
    seed_x = int(xs[np.argmax(distance[54, xs]-.08*np.abs(xs-42))])
    seed = np.asarray([seed_x, 54.], dtype=float)
    beam = [(0., -np.pi/2, [seed])]
    best = beam[0]
    for _ in range(22):
        candidates = []
        for score, angle, path in beam:
            for change in np.linspace(-.65, .65, 9):
                heading = angle+change
                point = path[-1]+3.*np.asarray([np.cos(heading), np.sin(heading)])
                x, y = np.rint(point).astype(int)
                if not (1 <= x < 83 and 1 <= y < 60) or distance[y, x] < 1.5:
                    continue
                if len(path) > 5 and min(np.linalg.norm(point-p) for p in path[:-4]) < 3.:
                    continue
                value = score+min(float(distance[y, x]), 8.)-.9*change*change
                candidates.append((value, heading, path+[point]))
        if not candidates:
            break
        candidates.sort(key=lambda r: r[0], reverse=True)
        beam, used = [], set()
        for item in candidates:
            x, y = item[2][-1]
            key = (int(x/3), int(y/3), int(item[1]/.3))
            if key in used:
                continue
            used.add(key)
            beam.append(item)
            if len(beam) == 5:
                break
        best = beam[0]
    return np.asarray(best[2]), component


def metric_path(pixels):
    points = (pixels-ORIGIN)/SCALE
    points[:, 1] *= -1
    return points


def path_target(points, distance):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[np.linalg.norm(points[0]), np.linalg.norm(points[0])+np.cumsum(lengths)]
    return np.asarray([np.interp(distance, arc, points[:, i]) for i in (0, 1)]), float(arc[-1])


def bearing(point):
    return float(np.arctan2(2*3.24*point[0], max(float(point @ point), 1.)))


class ConnectedGuardedPreviewAgent(FastCompletionCoordination):
    def __init__(self, mechanism='connected_pursuit'):
        if mechanism not in MODES:
            raise ValueError('unknown connected preview mode')
        self.preview_mode = mechanism
        super().__init__('preview_row_repair')

    def reset(self, observation=None):
        super().reset(observation)
        self.previous_frame = None
        self.previous_path = np.empty((0, 2))
        self.path_age = 0
        self.obstacle_hold = 0

    def _transport(self, frame, mask):
        if self.previous_frame is None or len(self.previous_path) < 5:
            return np.empty((0, 2)), None
        old = (self.previous_frame*255).astype(np.uint8)
        new = (frame*255).astype(np.uint8)
        feature_mask = np.full((84, 84), 255, np.uint8)
        feature_mask[61:] = 0
        feature_mask[49:61, 32:52] = 0
        p = cv2.goodFeaturesToTrack(old, 70, .03, 4, mask=feature_mask)
        if p is None or len(p) < 8:
            return np.empty((0, 2)), None
        q, valid, _ = cv2.calcOpticalFlowPyrLK(old, new, p, None, winSize=(15, 15), maxLevel=2)
        if q is None or valid is None:
            return np.empty((0, 2)), None
        ok = valid.ravel().astype(bool)
        if np.count_nonzero(ok) < 8:
            return np.empty((0, 2)), None
        a, b = (p.reshape(-1, 2)[ok]-ORIGIN)/SCALE, (q.reshape(-1, 2)[ok]-ORIGIN)/SCALE
        transform, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                      ransacReprojThreshold=.8, maxIters=100)
        if transform is None or inliers is None or int(inliers.sum()) < 8:
            return np.empty((0, 2)), None
        rotation = float(np.arctan2(transform[1, 0], transform[0, 0]))
        scale = float(np.linalg.norm(transform[:, 0]))
        if not (.9 < scale < 1.1) or abs(rotation) > .6 or np.linalg.norm(transform[:, 2]) > 15.:
            return np.empty((0, 2)), None
        local = (self.previous_path-ORIGIN)/SCALE
        moved = (local @ transform[:, :2].T+transform[:, 2])*SCALE+ORIGIN
        inside = (moved[:, 0] >= 1) & (moved[:, 0] < 83) & (moved[:, 1] >= 1) & (moved[:, 1] < 58)
        moved = moved[inside]
        if len(moved) < 5:
            return np.empty((0, 2)), None
        index = np.rint(moved).astype(int)
        support = float(np.mean(mask[index[:, 1], index[:, 0]] > 0))
        quality = dict(inliers=int(inliers.sum()), rotation=rotation, scale=scale, support=support)
        if support < .7:
            return np.empty((0, 2)), quality
        return moved, quality

    def act(self, observation):
        impact_before = self.impact_left
        action = super().act(observation)
        original = action.copy()
        frame = current_frame(observation)
        pixels, mask = connected_path(frame)
        memory_used, memory_fused, quality = False, False, None
        if self.preview_mode == 'motion_memory':
            transported, quality = self._transport(frame, mask)
            if len(pixels) < 8 and len(transported) >= 8 and self.path_age < 3:
                pixels = transported
                memory_used = True
            elif len(pixels) >= 8 and len(transported) >= 8:
                distances = np.linalg.norm(pixels[:, None, :]-transported[None, :, :], axis=2)
                indices = distances.argmin(axis=1)
                good = distances[np.arange(len(pixels)), indices] < 4.
                if float(np.mean(good)) >= .7:
                    pixels[good] = .75*pixels[good]+.25*transported[indices[good]]
                    memory_fused = True
        self.path_age = self.path_age+1 if memory_used else 0
        self.previous_path, self.previous_frame = pixels.copy(), frame.copy()
        points = metric_path(pixels) if len(pixels) else np.empty((0, 2))
        speed = float(self.last['pixel_speed'])
        near = far = None
        reach = 0.
        applied, accelerated = False, False
        centers = self.last['road_centers']
        middle = observed_center(centers, 42) if 54 in centers and len(centers) >= 2 else None
        if len(points) >= 8 and middle is not None:
            near, _ = path_target(points, 12.)
            far, reach = path_target(points, max(18., speed*.32))
            usable = far[1] > 5. and reach >= 18.
            if self.steps > 10 and self.preview_mode != 'control' and usable:
                road = .022*(middle-42)+.018*(middle-centers[54])
                obstacle = self.base._last_obstacle
                clearing = ((impact_before > 0 or self.last['impact_proxy_trigger'])
                            and not self.last['braking_proxy_veto'] and 42 in centers and 54 in centers)
                avoid = self.base._obstacle_side*.34*float(np.clip((obstacle[0]-22)/18, 0., 1.)) if obstacle is not None and not clearing else 0.
                desired = bearing(far)
                if self.preview_mode == 'dual_preview':
                    desired = road+.4*(bearing(far)-bearing(near))
                action[0] = np.clip(desired+avoid+self.last['correction'], -.7, .7)
                applied = True
                if self.preview_mode == 'shared_path':
                    # Only accelerate when the same planned path is long and nearly straight.
                    straight = (reach > 28. and abs(bearing(far)) < .08 and abs(far[0]-near[0]) < 2.
                                and abs(centers[54]-42) < 3. and obstacle is None and speed < 75.)
                    if straight:
                        action[1], action[2] = max(float(action[1]), .65), 0.
                        accelerated = True
        if self.base._last_obstacle is not None:
            self.obstacle_hold = 3
        preview_authorized = applied and self.obstacle_hold == 0 and self.base._last_obstacle is None
        if self.base._last_obstacle is None:
            self.obstacle_hold = max(0, self.obstacle_hold-1)
        if self.preview_mode != 'dual_preview':
            preview_authorized = preview_authorized and abs(centers.get(54, 100.)-42.) < 3.
        if applied:
            if preview_authorized:
                action[0] = np.clip(original[0]+np.clip(action[0]-original[0], -.12, .12), -.7, .7)
            else:
                action = original.copy()
                accelerated = False
                applied = False
        self.brake_history[-1] = float(action[2])
        self.last.update(connected_mode=self.preview_mode, connected_pixels=pixels.tolist(),
                         connected_target=None if far is None else far.tolist(),
                         connected_reach=reach, connected_time_ahead=None if far is None else float(far[1]/max(speed, 1.)),
                         connected_steering_applied=applied, connected_acceleration=accelerated,
                         transported_memory_used=memory_used, transported_memory_age=self.path_age,
                         transported_memory_fused=memory_fused, preview_authorized=bool(preview_authorized),
                         obstacle_preview_hold=self.obstacle_hold,
                         optical_flow_quality=quality, baseline_action=original.tolist(),
                         completion_changed=bool(np.max(np.abs(action-original)) > 1e-6),
                         final_gas=float(action[1]), final_brake=float(action[2]))
        return action
