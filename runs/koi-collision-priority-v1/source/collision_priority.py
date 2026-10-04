"""Narrow pixel-only collision priority over an unchanged steering-release v2.

Road boundaries are a preference, never a rejection of an emergency trajectory.
The official streak is reward-based and unavailable to act(observation). Timers
below bound this intervention, not the unobservable environment reward streak.
"""

import math
from typing import Any

import cv2
import numpy as np


# Participants env_wrapper.py: off_track_counter > 100, once per skip-4 action.
NEGATIVE_REWARD_LIMIT = 100
HOLD = 4 / 50
X_SCALE, Y_SCALE = 1.3608, 1.701
HALF_WIDTH = 1.1 + .28 * math.cos(.4) + .54 * math.sin(.4) + .01
FRONT, REAR, WHEELBASE = 2.61, 2.41, 3.24
LOOKAHEAD_ACTIONS = 8


def obstacle_box(frame, obj):
    mask = np.zeros((84, 84), np.uint8)
    mask[22:62] = frame[22:62] >= .54
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for stat, center in zip(stats[1:], centers[1:]):
        x, y, w, h, area = map(int, stat)
        if (4 <= area <= 80 and 2 <= w <= 9 and 2 <= h <= 10
                and abs(center[0] - obj[1]) < 1e-6 and abs(center[1] - obj[0]) < 1e-6):
            return np.array([[(a - 42) / X_SCALE, (63 - b) / Y_SCALE]
                             for a in (x - 1, x + w) for b in (y - 1, y + h)])
    return None


def relative_box(box, steer, distance):
    """Constant-command ego-motion proxy, used only through brief dropout."""
    k = math.tan(float(np.clip(steer, -.4, .4))) / WHEELBASE
    angle = k * distance
    c, s = math.cos(angle), math.sin(angle)
    px = (1 - c) / k if abs(k) > 1e-10 else 0.
    py = s / k if abs(k) > 1e-10 else distance
    dx, dy = box[:, 0] - px, box[:, 1] - py
    return np.column_stack((dx * c - dy * s, dx * s + dy * c))


def arc_clearance(box, steer, distance):
    """Signed conservative rectangle gap along a short arc, not a safety proof."""
    distances = np.linspace(0, max(distance, .01), max(2, int(math.ceil(distance * Y_SCALE / .5)) + 1))
    k = math.tan(float(np.clip(steer, -.4, .4))) / WHEELBASE
    angle = k * distances
    c, s = np.cos(angle), np.sin(angle)
    px = (1 - c) / k if abs(k) > 1e-10 else np.zeros_like(distances)
    py = s / k if abs(k) > 1e-10 else distances
    dx, dy = box[:, 0, None] - px, box[:, 1, None] - py
    u, v = dx * c - dy * s, dx * s + dy * c
    gaps = np.maximum.reduce((u.min(axis=0) - HALF_WIDTH - .5 / X_SCALE,
                              -u.max(axis=0) - HALF_WIDTH - .5 / X_SCALE,
                              v.min(axis=0) - FRONT - .5 / Y_SCALE,
                              -v.max(axis=0) - REAR - .5 / Y_SCALE))
    return float(gaps.min())


class CollisionPriorityAgent:
    def __init__(self, driver):
        # The caller supplies the immutable ZIP's SteeringReleaseAgent.
        self.driver = driver if hasattr(driver, 'stabilize_ambiguous_flank') else driver.driver
        self._clear()

    def _clear(self):
        self.age = 0
        self.side = 0.
        self.box = None
        self.missing = 0
        self.rejoined = 0
        self.previous_steer = 0.
        self.previous_speed = 0.
        self.previous_object = None
        self.last: dict[str, Any] = {}

    def reset(self, observation):
        self.driver.reset(observation)
        self._clear()

    def act(self, observation):
        action = np.asarray(self.driver.act(observation), dtype=np.float32).copy()
        original = action.copy()
        info = self.driver.last_step_diagnostics()
        base = self.driver.driver
        frame = base.base._frame(observation)
        terms = info.get('steering_terms', {})
        raw = terms.get('raw_terms', {})
        centers = {int(k): float(v) for k, v in info.get('road_centers', {}).items()}
        obj = info.get('near_object')
        speed = float(info.get('pixel_speed', 0.))
        side = float(base.base._obstacle_side)
        motion_risk = False
        if obj is not None and self.previous_object is not None:
            dy, dx = obj[0] - self.previous_object[0], obj[1] - self.previous_object[1]
            if dy > 1 and math.hypot(dy, dx) <= 15 and 30 <= obj[0] <= 55:
                remaining = (60 - obj[0]) / dy
                projected = obj[1] + dx * remaining
                motion_risk = remaining <= LOOKAHEAD_ACTIONS and abs(projected - 42) < 6
        mode = 'v2'
        road = nominal_gap = baseline_gap = chosen_gap = None
        fresh_box = obstacle_box(frame, obj) if frame is not None and obj is not None else None
        if self.box is not None:
            self.box = relative_box(self.box, self.previous_steer, self.previous_speed * HOLD)
        if fresh_box is not None:
            self.box, self.missing = fresh_box, 0
        else:
            self.missing += 1
        valid = (frame is not None and terms.get('reconstruction_valid')
                 and not terms.get('unidentifiable') and base.steps > 10
                 and all(raw.get(k) is not None for k in ('selected_road_position', 'selected_lookahead', 'damping'))
                 and math.isfinite(speed) and 0 < speed < 80)
        if valid:
            road = float(np.clip(raw['selected_road_position'] + raw['selected_lookahead'] + raw['damping'], -.7, .7))
            bend = (42 in centers and 54 in centers and
                    (abs(centers[42] - centers[54]) >= 4 or
                     (30 in centers and abs(centers[30] - 2 * centers[42] + centers[54]) >= 2)))
            horizon = speed * HOLD * LOOKAHEAD_ACTIONS
            if self.box is not None:
                nominal_gap = arc_clearance(self.box, road, horizon)
                baseline_gap = arc_clearance(self.box, float(action[0]), horizon)
            nominal_overlap = bool(obj is not None and centers and abs(obj[1] - np.interp(
                obj[0], sorted(centers), [centers[y] for y in sorted(centers)])) < 6)
            arc_risk = (nominal_gap is not None and nominal_gap < 0
                        and baseline_gap is not None and baseline_gap < 0)
            trigger = (fresh_box is not None and bend and side in (-1., 1.)
                       and road * side < 0 and ((nominal_overlap and motion_risk) or arc_risk))
            if trigger and self.age == 0:
                self.age, self.side = 1, side
            elif self.age:
                self.age += 1
            if trigger:
                self.side = side
            if self.age:
                # Spend at most one quarter of the verified limit avoiding;
                # reserve the remainder for recovery. Neither is a reward counter.
                budget_recovery = self.age >= NEGATIVE_REWARD_LIMIT // 4
                danger = (self.box is not None and (motion_risk or (nominal_gap is not None and nominal_gap < 0))
                          and self.missing <= 4)
                if danger and not budget_recovery:
                    urgency = float(np.clip((obj[0] - 22) / 18, 0, 1)) if obj is not None else 1.
                    magnitude = abs(raw.get('effective_avoidance') or .34 * urgency)
                    escape = self.side * magnitude
                    candidates = [float(action[0] + weight * (escape - action[0])) for weight in (.25, .5, .75, 1.)]
                    gaps = [arc_clearance(self.box, steer, horizon) for steer in candidates]
                    safe = [i for i, gap in enumerate(gaps) if gap >= 0]
                    index = safe[0] if safe else int(np.argmax(gaps))
                    if motion_risk:
                        # Optical approach includes motion absent from the ideal arc.
                        index = len(candidates) - 1
                    if motion_risk or (baseline_gap is not None and gaps[index] > baseline_gap):
                        action[0], chosen_gap, mode = candidates[index], gaps[index], 'avoid'
                else:
                    action[0], mode = road, 'budget_recovery' if budget_recovery else 'recover'
                    # Heading + road-position agreement, not a reward-streak reset.
                    aligned = (54 in centers and 42 in centers and abs(centers[54] - 42) < 3
                               and abs(centers[42] - centers[54]) < 3)
                    self.rejoined = self.rejoined + 1 if aligned else 0
                    if self.rejoined >= 2:
                        self.age, self.side, self.box = 0, 0., None
            else:
                self.box = None
        elif self.age:
            self.age += 1
        changed = not np.array_equal(action, original)
        self.previous_steer, self.previous_speed = float(action[0]), speed if math.isfinite(speed) else 0.
        self.previous_object = tuple(obj) if obj is not None else None
        if changed:
            self.driver._last_steer = float(action[0])
            base.last_command = tuple(float(value) for value in action)
        self.last = dict(info, priority_mode=mode, priority_changed=changed,
                         priority_episode_age=self.age, priority_road_steer=road,
                         priority_base_steer=float(original[0]), priority_nominal_clearance=nominal_gap,
                         priority_base_clearance=baseline_gap, priority_clearance=chosen_gap,
                         priority_motion_risk=motion_risk,
                         priority_reward_counter_observable=False,
                         priority_negative_reward_limit=NEGATIVE_REWARD_LIMIT,
                         priority_kinematic_proxy_not_guarantee=True)
        # Frozen source terms explain v2, not this wrapper's final changed command.
        self.last['v2_steering_terms'] = self.last.pop('steering_terms', {})
        return action

    def last_step_diagnostics(self):
        return dict(self.last)
