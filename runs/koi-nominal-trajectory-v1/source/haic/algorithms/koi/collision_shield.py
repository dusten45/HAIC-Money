"""Short, pixel-only collision shield over the unchanged crossing controller.

Projection is a nominal actuator/bicycle model, NOT a physical safety guarantee.
Only the first .08s is the candidate command; the remaining lookahead uses the
baseline command. There is no recovery controller, road veto, or pedal change.
"""

import math
from typing import Any

import cv2
import numpy as np


ACTION_SECONDS = .08
HORIZON_ACTIONS = 4
MAX_INTERVENTION_ACTIONS = 6
REARM_CLEAR_ACTIONS = 3
MAX_BOX_AGE = 2
X_SCALE, Y_SCALE, WHEELBASE = 1.3608, 1.701, 3.24
HALF_WIDTH = 1.1 + .28 * math.cos(.4) + .54 * math.sin(.4) + .01
FRONT, REAR = 2.61, 2.41
ROAD_COST_WEIGHT = .01


def detect_boxes(frame):
    """Full components, not the baseline's cropped/selected obstacle centroid."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        (frame >= .54).astype(np.uint8), connectivity=8)
    boxes = []
    for x, y, w, h, area in stats[1:]:
        if (4 <= area <= 80 and 2 <= w <= 9 and 2 <= h <= 10
                and 0 < x and x + w < 84 and 0 < y and y + h < 73):
            # Pixel-edge/downsampling uncertainty, independent of road geometry.
            boxes.append(((x - 1 - 42) / X_SCALE, (63 - y - h) / Y_SCALE,
                          (x + w - 42) / X_SCALE, (64 - y) / Y_SCALE))
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def project_paths(steers, baseline_steer, speed, wheel_angle, actions=HORIZON_ACTIONS):
    """Positive steering turns image-right; estimate the rate-limited actuator.

    Sample at <=.5 vertical raster pixel of nominal travel. Unknown slip/damage
    are not observed and are not claimed to be bounded by this model.
    """
    steers = np.asarray(steers, dtype=np.float64)
    # Include rotation of the furthest body corner in the sample-spacing bound.
    corner_factor = 1 + math.hypot(FRONT, HALF_WIDTH) * math.tan(.4) / WHEELBASE
    per_action = max(8, int(math.ceil(speed * corner_factor * ACTION_SECONDS * Y_SCALE / .5)))
    dt = ACTION_SECONDS / per_action
    paths = np.zeros((len(steers), per_action * actions + 1, 3))
    wheels = np.full(len(steers), wheel_angle, dtype=np.float64)
    first_wheels = wheels.copy()
    for step in range(1, paths.shape[1]):
        target = steers if step <= per_action else baseline_steer
        wheels += np.clip(50 * (target - wheels), -3., 3.) * dt
        wheels = np.clip(wheels, -.4, .4)
        yaw_step = speed * np.tan(wheels) / WHEELBASE * dt
        heading = paths[:, step - 1, 2] + .5 * yaw_step
        paths[:, step, 0] = paths[:, step - 1, 0] + speed * np.sin(heading) * dt
        paths[:, step, 1] = paths[:, step - 1, 1] + speed * np.cos(heading) * dt
        paths[:, step, 2] = paths[:, step - 1, 2] + yaw_step
        if step == per_action:
            first_wheels = wheels.copy()
    return paths, first_wheels, per_action


def footprint_gaps(paths, boxes):
    """Four-axis SAT separation of full wheel/hull envelope and all object boxes.

    Negative means projected overlap, positive a separating-axis gap (not an
    exact Euclidean clearance). Half-pixel sweep padding covers raster spacing.
    Physical fixture clearance is measured separately by the passive evaluator.
    """
    clearance = np.full(paths.shape[0], np.inf)
    sine, cosine = np.sin(paths[:, :, 2]), np.cos(paths[:, :, 2])
    half_length = .5 * (FRONT + REAR) + .5 / Y_SCALE
    half_width = HALF_WIDTH + .5 / X_SCALE
    center_shift = .5 * (FRONT - REAR)
    px = paths[:, :, 0] + center_shift * sine
    py = paths[:, :, 1] + center_shift * cosine
    for xlo, ylo, xhi, yhi in boxes:
        dx, dy = .5 * (xlo + xhi) - px, .5 * (ylo + yhi) - py
        hx, hy = .5 * (xhi - xlo), .5 * (yhi - ylo)
        gap = np.maximum.reduce((
            np.abs(dx) - hx - half_width * np.abs(cosine) - half_length * np.abs(sine),
            np.abs(dy) - hy - half_width * np.abs(sine) - half_length * np.abs(cosine),
            np.abs(dx * cosine - dy * sine) - half_width - hx * np.abs(cosine) - hy * np.abs(sine),
            np.abs(dx * sine + dy * cosine) - half_length - hx * np.abs(sine) - hy * np.abs(cosine)))
        clearance = np.minimum(clearance, gap.min(axis=1))
    return clearance


def road_cost(paths, frame, centers):
    """Soft observed lateral footprint departure, never a feasibility constraint."""
    cost = np.zeros(paths.shape[:2])
    known = sorted(int(row) for row in centers)
    if not known:
        return cost.mean(axis=1)
    xs = 42 + X_SCALE * paths[:, :, 0]
    rows = np.rint(63 - Y_SCALE * paths[:, :, 1]).astype(int)
    width = X_SCALE * (HALF_WIDTH * np.abs(np.cos(paths[:, :, 2]))
                       + FRONT * np.abs(np.sin(paths[:, :, 2])))
    for row in np.unique(rows):
        if not max(23, known[0]) <= row <= min(61, known[-1]):
            continue
        center = np.interp(row, known, [centers[key] for key in known])
        asphalt = np.flatnonzero((frame[row] >= .24) & (frame[row] <= .52)
                                & (np.abs(np.arange(84) - center) <= 17))
        if len(asphalt) < 4:
            continue
        penalty = np.maximum(0, np.maximum(asphalt[0] - xs + width,
                                           xs + width - asphalt[-1])) / X_SCALE
        cost = np.where(rows == row, penalty, cost)
    return cost.mean(axis=1)


def advance_boxes(boxes, pose):
    """Propagate only a short occlusion memory, conservatively enclosing rotation."""
    if not len(boxes):
        return boxes.copy()
    corners = boxes[:, [0, 0, 2, 2]], boxes[:, [1, 3, 1, 3]]
    dx, dy = corners[0] - pose[0], corners[1] - pose[1]
    sine, cosine = math.sin(pose[2]), math.cos(pose[2])
    x, y = dx * cosine - dy * sine, dx * sine + dy * cosine
    return np.column_stack((x.min(axis=1), y.min(axis=1), x.max(axis=1), y.max(axis=1)))


class CollisionShieldAgent:
    def __init__(self, driver: Any):
        self.driver: Any = getattr(driver, 'driver', driver)
        if getattr(self.driver, 'contact_mode', None) != 'crossing_projection':
            raise ValueError('requires unchanged crossing_projection, not a recovery/release wrapper')
        self._clear()

    def _clear(self):
        self.wheel_angle = 0.
        self.boxes = np.empty((0, 4))
        self.box_ages = np.empty(0, dtype=int)
        self.box_threatened = np.empty(0, dtype=bool)
        self.rearm_blocked = False
        self.encounter_id = self.encounter_actions = self.threat_free_decisions = 0
        self.encounter_open = False
        self.last: dict[str, Any] = {}
        self.last_shield: dict[str, Any] = {}

    def reset(self, observation=None):
        self.driver.reset(observation)
        self._clear()

    def act(self, observation):
        original = self.driver.act(observation)
        action = original.copy()
        info = self.driver.last_step_diagnostics()
        frame = self.driver.base._frame(observation)
        speed = float(info.get('pixel_speed', 0.))
        valid = (frame is not None and frame.shape == (84, 84)
                 and np.isfinite(frame).all() and math.isfinite(speed) and 0 < speed < 80)
        if valid:
            frames = np.asarray(observation)
            frames = frames[-2:] if frames.ndim == 3 else frames[None]
            readings = [speed, *(float(self.driver.base._estimate_speed(item)) for item in frames)]
            valid = all(math.isfinite(value) and 0 <= value < 80 for value in readings)
            speed = max(readings) if valid else float('nan')
        baseline_gap = chosen_gap = None
        candidate_count = 0
        threat = False
        clear_observed = False
        reason = 'invalid_projection'
        paths = wheels = None
        index = 0
        if valid:
            fresh = detect_boxes(frame)
            retained, ages, threatened = [], [], []
            fresh_threatened = np.zeros(len(fresh), dtype=bool)
            distances = np.linalg.norm(.5 * (self.boxes[:, None, :2] + self.boxes[:, None, 2:])
                                       - .5 * (fresh[None, :, :2] + fresh[None, :, 2:]), axis=2)
            associations = (distances < 3.) & np.all(
                fresh[None, :, 2:] - fresh[None, :, :2]
                >= .8 * (self.boxes[:, None, 2:] - self.boxes[:, None, :2]), axis=2)
            for box, age, was_threat, possible in zip(
                    self.boxes, self.box_ages, self.box_threatened, associations):
                matches = np.flatnonzero(possible)
                if len(matches) == 1 and np.count_nonzero(associations[:, matches[0]]) == 1:
                    fresh_threatened[matches[0]] |= was_threat
                elif age < MAX_BOX_AGE and box[3] >= -REAR:
                    retained.append(box)
                    ages.append(age + 1)
                    threatened.append(was_threat)
                elif was_threat and self.encounter_open:
                    # This only inhibits later rearming; it NEVER retains control.
                    self.rearm_blocked = True
            self.boxes = np.concatenate((fresh, np.asarray(retained).reshape(-1, 4)))
            self.box_ages = np.asarray([0] * len(fresh) + ages)
            self.box_threatened = np.concatenate((fresh_threatened, np.asarray(threatened, dtype=bool)))
            steers = np.concatenate(([float(original[0])], np.linspace(-.4, .4, 33))).astype(
                action.dtype).astype(np.float64)
            paths, wheels, hold = project_paths(steers, float(original[0]), speed, self.wheel_angle)
            gaps = footprint_gaps(paths, self.boxes)
            baseline_gap = float(gaps[0]) if len(self.boxes) else None
            threat = bool(gaps[0] <= 0)
            clear_observed = bool(len(fresh) and not threat and not self.rearm_blocked
                                  and (not self.encounter_open or fresh_threatened.any()))
            reason = 'baseline_clear' if len(self.boxes) else 'no_visible_obstacle'
            if threat:
                self.threat_free_decisions = 0
                if not self.encounter_open:
                    self.encounter_id += 1
                    self.encounter_open = True
                self.box_threatened |= np.asarray([
                    footprint_gaps(paths[:1], box[None])[0] <= 0 for box in self.boxes])
                safe = np.flatnonzero(gaps > 0)
                candidate_count = len(safe)
                reason = 'no_collision_free_candidate'
                if self.driver.steps <= 10:
                    reason = 'launch_prefix'
                elif self.encounter_actions >= MAX_INTERVENTION_ACTIONS:
                    reason = 'encounter_budget'
                elif len(safe):
                    costs = np.abs(steers - float(original[0])) + ROAD_COST_WEIGHT * road_cost(
                        paths, frame, info.get('road_centers', {}))
                    index = int(safe[np.argmin(costs[safe])])
                    action[0] = steers[index]
                    self.encounter_actions += 1
                    reason = 'collision_shield'
            else:
                # Clear projections hand back NOW, not after a stable recovery hold.
                # Missing pixels alone cannot reset an exhausted encounter budget.
                if clear_observed:
                    self.threat_free_decisions += 1
                    if self.threat_free_decisions >= REARM_CLEAR_ACTIONS:
                        self.encounter_actions = 0
                        self.encounter_open = False
                        self.box_threatened[:] = False
                else:
                    self.threat_free_decisions = 0
            chosen_gap = float(gaps[index]) if len(self.boxes) else None
            self.wheel_angle = float(wheels[index])
            self.boxes = advance_boxes(self.boxes, paths[index, hold])
        else:
            # No hidden road-search or braking fallback, and no stale box revival.
            if self.encounter_open and self.box_threatened.any():
                self.rearm_blocked = True
            self.boxes = np.empty((0, 4))
            self.box_ages = np.empty(0, dtype=int)
            self.box_threatened = np.empty(0, dtype=bool)
            self.threat_free_decisions = 0
            _, wheels, _ = project_paths([float(original[0])], float(original[0]),
                                         0., self.wheel_angle, actions=1)
            self.wheel_angle = float(wheels[0])
        active = not np.array_equal(action, original)
        self.last_shield = dict(
            active=active, reason=reason, baseline_action=original.tolist(),
            baseline_threat=threat, baseline_clearance=baseline_gap,
            baseline_clear_observed=clear_observed,
            chosen_clearance=chosen_gap, candidate_count=candidate_count,
            encounter_id=self.encounter_id, encounter_actions=self.encounter_actions,
            threat_free_decisions=self.threat_free_decisions,
            rearm_blocked=self.rearm_blocked,
            tracked_boxes=len(self.boxes), wheel_angle_estimate=self.wheel_angle,
            projection_speed=speed if math.isfinite(speed) else None,
            horizon_seconds=HORIZON_ACTIONS * ACTION_SECONDS,
            command_hold_seconds=ACTION_SECONDS,
            steering_delta=float(action[0]) - float(original[0]))
        self.last = dict(info, **{'shield_' + key: value for key, value in self.last_shield.items()})
        return action

    def last_step_diagnostics(self):
        return dict(self.last)
