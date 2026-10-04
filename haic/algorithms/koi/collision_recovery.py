"""Collision priority and heading-first re-entry over frozen crossing_projection.

No steering-release/near_release dependency. All inputs are pixels and controller
history. The reward-streak counter is unavailable to the competition Agent API.
"""

import math
from typing import Any

import cv2
import numpy as np


NEGATIVE_REWARD_LIMIT = 100  # Official wrapper retires when counter > 100.
DT = 4 / 50
XS, YS, WB = 1.3608, 1.701, 3.24
WIDTH = 1.1 + .28 * math.cos(.4) + .54 * math.sin(.4) + .01
FRONT, REAR = 2.61, 2.41


def observed_road(frame, centers) -> dict[str, Any] | None:
    """Local road tangent and nearest safe edge, never a centerline target."""
    centers = {int(k): float(v) for k, v in centers.items()}
    if len(centers) < 2:
        # The inherited +/-17px search can lose the road after a lateral excursion.
        previous = 42.
        for row in (54, 50, 46, 42, 38, 34, 30):
            locations = np.flatnonzero((frame[row] >= .24) & (frame[row] <= .52))
            groups = [g for g in np.split(locations, np.flatnonzero(np.diff(locations) > 8) + 1) if len(g) >= 4]
            if groups:
                group = min(groups, key=lambda g: abs(float(g.mean()) - previous))
                previous = centers[row] = float(group.mean())
    if 54 not in centers or len(centers) < 2:
        return None
    if 42 in centers:
        middle = centers[42]
    else:
        rows = sorted(centers, key=lambda row: abs(row - 42))[:2]
        a, b = rows
        middle = centers[a] + (42 - a) * (centers[b] - centers[a]) / (b - a)
    near = centers[54]
    heading_row = max((row for row in centers if row < 54), default=42)
    heading_x = centers.get(heading_row, middle)
    heading = math.atan2((heading_x - near) / XS, (54 - heading_row) / YS)
    locations = np.flatnonzero((frame[54] >= .24) & (frame[54] <= .52)
                               & (abs(np.arange(84) - near) <= 17))
    if len(locations) < 4:
        return None
    # Extrapolate only nine pixels from the near row to the car; no map knowledge.
    shift = .75 * (near - middle)
    left, right = float(locations[0] + shift), float(locations[-1] + shift)
    pad = WIDTH * XS + .5
    supported = left + pad <= 42 <= right - pad
    entry_x = float(np.clip(42, left + pad, right - pad)) if right - left >= 2 * pad else .5 * (left + right)
    edge_error = (entry_x - 42) / XS
    return dict(heading=heading, edge_error=edge_error, supported=supported,
                middle=middle, near=near, centers=centers)


def detected_box(frame, obj):
    mask = np.zeros((84, 84), np.uint8)
    mask[22:62] = frame[22:62] >= .54
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for stat, center in zip(stats[1:], centers[1:]):
        x, y, w, h, area = map(int, stat)
        if (4 <= area <= 80 and 2 <= w <= 9 and 2 <= h <= 10
                and abs(center[0] - obj[1]) < 1e-6 and abs(center[1] - obj[0]) < 1e-6):
            return np.array([[(a - 42) / XS, (63 - b) / YS]
                             for a in (x - 1, x + w) for b in (y - 1, y + h)])
    return None


def sweep(box, steer, distance):
    """Short body-envelope sweep and end-relative box; actuator/slip proxy only."""
    distances = np.linspace(0, max(distance, .01), max(2, int(math.ceil(distance * YS / .5)) + 1))
    k = math.tan(float(np.clip(steer, -.4, .4))) / WB
    c, s = np.cos(k * distances), np.sin(k * distances)
    px = (1-c)/k if abs(k) > 1e-10 else np.zeros_like(distances)
    py = s/k if abs(k) > 1e-10 else distances
    dx, dy = box[:, 0, None]-px, box[:, 1, None]-py
    u, v = dx*c-dy*s, dx*s+dy*c
    gap = np.maximum.reduce((u.min(axis=0)-WIDTH-.5/XS, -u.max(axis=0)-WIDTH-.5/XS,
                             v.min(axis=0)-FRONT-.5/YS, -v.max(axis=0)-REAR-.5/YS))
    return float(gap.min()), np.column_stack((u[:, -1], v[:, -1]))


class CollisionRecoveryAgent:
    def __init__(self, driver):
        if getattr(driver, 'contact_mode', None) != 'crossing_projection':
            raise ValueError('requires the frozen crossing_projection driver, not steering-release')
        self.driver = driver
        self._clear()

    def _clear(self):
        self.mode = 'baseline'
        self.age = self.stable = self.absent = self.armed = self.road_missing = 0
        self.previous_object = self.box = None
        self.previous_speed = self.previous_steer = self.last_heading = 0.
        self.side = 0.
        self.last = {}

    def reset(self, observation):
        self.driver.reset(observation)
        self._clear()

    def act(self, observation):
        action = np.asarray(self.driver.act(observation), dtype=np.float32).copy()
        original = action.copy()
        info = self.driver.last_step_diagnostics()
        frame = self.driver.base._frame(observation)
        obj = info.get('near_object')
        speed = float(info.get('pixel_speed', 0.))
        road = observed_road(frame, info.get('road_centers', {})) if frame is not None else None
        self.road_missing = self.road_missing + 1 if road is None else 0
        side = float(self.driver.base._obstacle_side)
        if self.box is not None:
            _, self.box = sweep(self.box, self.previous_steer, self.previous_speed * DT)
        fresh = detected_box(frame, obj) if frame is not None and obj is not None else None
        if fresh is not None:
            self.box, self.absent = fresh, 0
        else:
            self.absent += 1
        if self.box is not None and self.box[:, 1].max() < -REAR:
            self.box = None
        threat = False
        baseline_gap = recovery_gap = None
        road_steer = heading = edge_error = None
        used_mode = 'baseline'
        if self.driver.steps > 10 and math.isfinite(speed) and 0 < speed < 80:
            motion_risk = False
            if obj is not None and self.previous_object is not None:
                dy, dx = obj[0]-self.previous_object[0], obj[1]-self.previous_object[1]
                if dy > 1 and math.hypot(dy, dx) <= 15 and 30 <= obj[0] <= 55:
                    remaining = (60-obj[0])/dy
                    motion_risk = remaining <= 8 and abs(obj[1]+dx*remaining-42) < 6
            if road is not None:
                heading, edge_error = road['heading'], road['edge_error']
                self.last_heading = heading
                road_steer = .022*(road['middle']-42) + .018*(road['middle']-road['near']) + float(info.get('correction', 0.))
                corner = abs(heading) >= .2 or abs(road['middle'] - road['near']) >= 4
                if obj is not None and obj[0] >= 40 and corner:
                    self.armed = 12
                else:
                    self.armed = max(0, self.armed-1)
                if self.box is not None:
                    baseline_gap, _ = sweep(self.box, float(action[0]), speed*DT*4)
                opposed = side in (-1., 1.) and road_steer*side < 0 and float(action[0])*side <= .05
                emergency = bool(fresh is not None and corner and opposed
                                 and (motion_risk or (baseline_gap is not None and baseline_gap < 0)))
                if emergency and self.mode == 'baseline':
                    self.mode, self.age, self.side = 'avoid', 0, side
                if self.mode != 'baseline':
                    self.age += 1
                # Align heading first. Only once aligned aim for the closest safe
                # road edge, not its centerline, using a bounded merge angle.
                merge_angle = float(np.clip(math.atan2(edge_error, max(8., speed*.4)), -.2, .2)) if abs(heading) < .25 else 0.
                recovery_steer = float(np.clip(.8*heading + merge_angle, -.4, .4))
                if self.box is not None:
                    recovery_gap, _ = sweep(self.box, recovery_steer, speed*DT*4)
                # A risky road-only proposal must not prolong escape after the
                # issued crossing command is already clear (the v1 regression).
                threat = bool(self.box is not None and ((motion_risk and opposed)
                              or (baseline_gap is not None and baseline_gap < 0
                                  and recovery_gap is not None and recovery_gap < 0)))
                separated = bool(obj is not None and obj[0] >= 48 and side*(obj[1]-42) < -6)
                behind = bool(self.box is not None and self.box[:, 1].max() < -REAR)
                post_obstacle = separated or behind or (obj is None and self.absent >= 2)
                unstable = abs(heading) > .25 or not road['supported']
                if self.mode == 'baseline' and self.armed and post_obstacle and unstable and not threat:
                    self.mode, self.age, self.side = 'align', 1, side
                if self.mode != 'baseline':
                    budget_recovery = self.age >= NEGATIVE_REWARD_LIMIT // 4
                    if threat and not budget_recovery:
                        used_mode = self.mode = 'avoid'
                        urgency = float(np.clip((obj[0]-22)/18, 0, 1)) if obj is not None else 1.
                        escape = (side if side in (-1., 1.) else self.side) * (.55 if info.get('contact_active') else .34) * urgency
                        gap = sweep(self.box, escape, speed*DT*4)[0]
                        if motion_risk or (baseline_gap is not None and gap > baseline_gap):
                            action[0] = escape
                        self.stable = 0
                    else:
                        used_mode = self.mode = 'align' if abs(heading) >= .25 else 'reenter'
                        action[0] = recovery_steer
                        if recovery_gap is not None and recovery_gap < 0 and baseline_gap is not None and baseline_gap >= 0:
                            action[0] = original[0]
                            used_mode = 'reentry_guard'
                        # Recovery restores feasible heading before high-speed travel.
                        # These are the inherited 38/44 curve/obstacle speed levels.
                        target = 38. if abs(heading) >= .25 or not road['supported'] else 44.
                        if speed > target:
                            action[1] = 0.
                            action[2] = max(float(action[2]), min(.6, .04*(speed-target)))
                        stable = road['supported'] and abs(heading) < .12 and not threat
                        self.stable = self.stable+1 if stable else 0
                        if self.stable >= 3:
                            self.mode, self.age, self.armed = 'baseline', 0, 0
                            action, used_mode = original.copy(), 'handoff'
            elif self.mode != 'baseline':
                self.age += 1
                used_mode = self.mode = 'road_search'
                action[0] = float(np.clip(.8*self.last_heading, -.25, .25)) if self.road_missing <= 4 else 0.
                action[1] = 0.
                action[2] = max(float(action[2]), min(.6, .04*max(0., speed-20.)))
        changed = not np.array_equal(action, original)
        if changed:
            self.driver.last_command = tuple(float(v) for v in action)
            if self.driver.brake_history:
                self.driver.brake_history[-1] = float(action[2])
        self.previous_object = tuple(obj) if obj is not None else None
        self.previous_speed = speed if math.isfinite(speed) else 0.
        self.previous_steer = float(action[0])
        self.last = dict(info, priority_changed=changed, priority_mode=used_mode,
                         recovery_mode_next=self.mode, recovery_age=self.age,
                         recovery_heading=heading, recovery_edge_error=edge_error,
                         recovery_supported=None if road is None else road['supported'],
                         recovery_stable=self.stable, recovery_threat=threat,
                         recovery_nominal_steer=road_steer, recovery_baseline_gap=baseline_gap,
                         recovery_path_gap=recovery_gap, recovery_base_action=original.tolist(),
                         recovery_reward_counter_observable=False,
                         recovery_negative_reward_limit=NEGATIVE_REWARD_LIMIT)
        return action

    def last_step_diagnostics(self):
        return dict(self.last)
