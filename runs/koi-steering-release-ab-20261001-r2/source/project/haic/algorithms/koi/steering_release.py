"""Near-avoidance release after observed projection clearance, pixels/history only."""

from importlib import import_module
import math
from typing import Any

import cv2
import numpy as np

from .steering_terms import decompose, near_scaled_steer, near_scaled_terms


X_SCALE, Y_SCALE = 1.3608, 1.701
HALF_WIDTH = 1.1 + .28 * math.cos(.4) + .54 * math.sin(.4) + .01
FRONT, REAR, WHEELBASE = 2.61, 2.41, 3.24
PROJECTION_BAND = 6.
EDGE_PAD, SAMPLE_PAD = 1., .5
HOLD = .08


def detected_bounds(frame, obj):
    mask = np.zeros((84, 84), dtype=np.uint8)
    mask[22:62] = frame[22:62] >= .54
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    matches = []
    for stat, center in zip(stats[1:], centers[1:]):
        x, y, width, height, area = map(int, stat)
        if (4 <= area <= 80 and 2 <= width <= 9 and 2 <= height <= 10
                and abs(center[0] - obj[1]) < 1e-6 and abs(center[1] - obj[0]) < 1e-6):
            matches.append((x, y, width, height))
    return matches[0] if len(matches) == 1 else None


def current_reentry(box, obj, side):
    x, y, width, height = box
    front = (63. - (y - EDGE_PAD)) / Y_SCALE
    rear = (63. - (y + height - 1 + EDGE_PAD)) / Y_SCALE
    if rear > FRONT + SAMPLE_PAD / Y_SCALE or front < -REAR - SAMPLE_PAD / Y_SCALE:
        return False
    edge = ((x + width - 1 + EDGE_PAD - 42.) / X_SCALE if side > 0 else
            -(x - EDGE_PAD - 42.) / X_SCALE)
    return (edge >= -HALF_WIDTH - SAMPLE_PAD / X_SCALE
            or side * (obj[1] - 42.) >= -PROJECTION_BAND)


def guarded_arc(box, centroid, side, steer, distance, *, require_rear_clear):
    """Conservative constant-command body-frame reentry check, not dynamics proof."""
    if not np.isfinite([*centroid, side, steer, distance]).all() or distance <= 0 or side not in (-1., 1.):
        return False
    curvature = math.tan(float(np.clip(steer, -.4, .4))) / WHEELBASE
    if abs(curvature * distance) > 1.25:
        return False
    x, y, width, height = box
    corners = np.array([[(a - 42.) / X_SCALE, (63. - b) / Y_SCALE]
                        for a in (x - EDGE_PAD, x + width - 1 + EDGE_PAD)
                        for b in (y - EDGE_PAD, y + height - 1 + EDGE_PAD)])
    center = np.array([(centroid[1] - 42.) / X_SCALE, (63. - centroid[0]) / Y_SCALE])
    distances = np.linspace(0., distance, max(2, int(math.ceil(distance * Y_SCALE / SAMPLE_PAD)) + 1))
    angle = curvature * distances
    c, s = np.cos(angle), np.sin(angle)
    px = (1. - c) / curvature if abs(curvature) > 1e-10 else np.zeros_like(distances)
    py = s / curvature if abs(curvature) > 1e-10 else distances
    dx, dy = corners[:, 0, None] - px, corners[:, 1, None] - py
    u, v = dx * c - dy * s, dx * s + dy * c
    involved = (v.min(axis=0) <= FRONT + SAMPLE_PAD / Y_SCALE) & (v.max(axis=0) >= -REAR - SAMPLE_PAD / Y_SCALE)
    edge = u.max(axis=0) if side > 0 else -u.min(axis=0)
    center_u = (center[0] - px) * c - (center[1] - py) * s
    separated = ((edge < -HALF_WIDTH - SAMPLE_PAD / X_SCALE)
                 & (side * center_u < -PROJECTION_BAND / X_SCALE))
    if not np.all(~involved | separated):
        return False
    # Passage is across the original forward plane, not a rotating body axis.
    rear_on_forward_axis = py[-1] - HALF_WIDTH * abs(s[-1]) - REAR * c[-1]
    return not require_rear_clear or rear_on_forward_axis > corners[:, 1].max() + SAMPLE_PAD / Y_SCALE


def release_safe(box, obj, side, steer, previous_steer, speed):
    # Previous-command hold is a lag proxy, not an actuator/slip envelope.
    distance = (63. - box[1] + EDGE_PAD) / Y_SCALE + FRONT + REAR
    return (guarded_arc(box, obj, side, steer, distance, require_rear_clear=True)
            and guarded_arc(box, obj, side, steer, speed * HOLD, require_rear_clear=False)
            and guarded_arc(box, obj, side, previous_steer, speed * HOLD, require_rear_clear=False))


class SteeringReleaseAgent:
    def __init__(self, driver=None, stabilize_ambiguous_flank=False):
        if driver is None:
            driver = import_module('haic_agent.contact_continuity_runtime').ContactContinuityAgent('crossing_projection')
        self.driver = driver
        self.stabilize_ambiguous_flank = stabilize_ambiguous_flank
        self._previous = self._previous_side = self._last_steer = None
        self._projection_cleared = False
        self.last: dict[str, Any] = {}

    def reset(self, observation):
        self.driver.reset(observation)
        self._previous = self._previous_side = self._last_steer = None
        self._projection_cleared = False
        self.last = {}

    def act(self, observation):
        motion_ignored = False
        previous_offset = current_offset = offset_motion = None
        if self.stabilize_ambiguous_flank and self.driver.steps >= 10 and self._previous is not None:
            frame = self.driver.base._frame(observation)
            if frame is not None:
                centers = self.driver.base._road_centers(frame)
                current = self.driver.base._nearest_bright_object(frame, centers)
                before_side = float(self.driver.base._obstacle_side)
                continuous = bool(current is not None and before_side == self._previous_side
                                  and -.5 <= current[0] - self._previous[0] < 16
                                  and abs(current[1] - self._previous[1]) < 15)
                previous_offset = self.driver.base._last_obstacle_side_offset
                if current is not None and continuous and before_side in (-1., 1.) and current[0] < 52 and previous_offset is not None:
                    current_offset = float(current[1] - current[2])
                    offset_motion = current_offset - previous_offset
                    offset_side = 1. if current_offset < 0 else -1.
                    would_flip = ((before_side > 0 and current_offset > -3 and offset_motion > 2)
                                  or (before_side < 0 and current_offset < 3 and offset_motion < -2))
                    if abs(current_offset) <= 1 and offset_side == before_side and would_flip:
                        # Ignore this noisy derivative, not the current detection.
                        self.driver.base._last_obstacle_side_offset = None
                        motion_ignored = True
        pre_impact = int(self.driver.impact_left)
        baseline = np.asarray(self.driver.act(observation), dtype=np.float32).copy()
        info = dict(self.driver.last_step_diagnostics())
        info['impact_left_before_act'] = pre_impact
        terms = decompose(self.driver, observation, baseline, info)
        action = baseline.copy()
        obj, side = info.get('near_object'), float(self.driver.base._obstacle_side)
        same = bool(obj is not None and self._previous is not None and side == self._previous_side
                    and -.5 <= obj[0] - self._previous[0] < 16 and abs(obj[1] - self._previous[1]) < 15)
        if not same:
            self._projection_cleared = False
        projected = info.get('projected_obstacle_x')
        valid_off = bool(projected is not None and math.isfinite(projected)
                         and side * (projected - 42.) < -PROJECTION_BAND)
        if info.get('contact_active') or (projected is not None and not valid_off):
            self._projection_cleared = False
        elif same and valid_off:
            self._projection_cleared = True
        reason, alpha, box, speed = 'no_near_object', 1., None, None
        reentered = bounds_unknown = False
        if self._projection_cleared and obj is not None:
            frame = self.driver.base._frame(observation)
            box = detected_bounds(frame, obj) if frame is not None else None
            bounds_unknown = box is None
            reentered = box is not None and current_reentry(box, obj, side)
            if bounds_unknown or reentered:
                self._projection_cleared = False
        if self.driver.steps <= 10:
            reason = 'launch_prefix'
            self._projection_cleared = False
        elif obj is not None:
            if not terms.get('reconstruction_valid') or terms.get('unidentifiable'):
                reason = 'unidentified_terms'
            elif terms.get('active_path') not in ('inherited_damping', 'geometry_repair'):
                reason = 'protected_replacement'
            elif pre_impact or self.driver.impact_left or info.get('impact_proxy_trigger') or info.get('contact_proxy'):
                reason = 'impact_guard'
            elif reentered:
                reason = 'current_reentry'
            elif bounds_unknown:
                reason = 'ambiguous_bounds'
            elif not same or not self._projection_cleared:
                reason = 'projection_not_cleared'
            elif abs(terms.get('avoidance_component') or 0.) <= .01:
                reason = 'no_effective_near_term'
            else:
                frame = self.driver.base._frame(observation)
                images = np.asarray(observation)
                images = images[-4:] if images.ndim == 3 else images[None]
                if frame is None or images.shape[1:] != (84, 84) or not np.isfinite(images).all():
                    reason = 'invalid_pixels'
                else:
                    speeds = [self.driver.base._estimate_speed(image) for image in images]
                    speeds.extend(self.driver.speed_history[-4:])
                    if not all(math.isfinite(value) and 0 <= value < 80 for value in speeds) or max(speeds) <= 0:
                        reason = 'hud_uncertainty'
                    else:
                        speed = float(max(speeds))
                        box = detected_bounds(frame, obj)
                        if box is None:
                            reason = 'ambiguous_bounds'
                            self._projection_cleared = False
                        else:
                            reason = 'reentry_guard'
                            for scale in (0., .25, .5, .75):
                                steer = near_scaled_steer(terms, scale)
                                if steer is not None and release_safe(box, obj, side, steer,
                                                                      self._last_steer if self._last_steer is not None else baseline[0], speed):
                                    alpha, action[0], reason = scale, steer, 'near_release'
                                    break
        changed = bool(action[0] != baseline[0])
        actual_terms = near_scaled_terms(terms, alpha, final_steer=float(action[0])) if changed else terms
        if changed and (actual_terms is None or not actual_terms['reconstruction_valid'] or not actual_terms['residual_valid']):
            action, actual_terms, alpha, changed, reason = baseline.copy(), terms, 1., False, 'accounting_guard'
        self.last = dict(info, baseline_steer=float(baseline[0]), baseline_pedals=baseline[1:].tolist(),
                         baseline_steering_terms=terms, steering_terms=actual_terms,
                         steering_release_changed=changed, steering_release_reason=reason,
                         steering_release_alpha=alpha, steering_release_safe=reason == 'near_release',
                         steering_release_gate=self._projection_cleared,
                         steering_release_box=list(box) if box is not None else None,
                         steering_release_same_object=same,
                         steering_release_identity_contract='heuristic selected pixel-component continuity; not physical object ID',
                         steering_release_projection_cleared=self._projection_cleared,
                         steering_release_projection_available=projected is not None and math.isfinite(projected),
                         steering_release_speed_bound=speed, steering_release_hold_s=HOLD,
                         steering_release_projection_band_px=PROJECTION_BAND,
                         steering_generation_stabilization_enabled=self.stabilize_ambiguous_flank,
                         steering_generation_ambiguous_motion_ignored=motion_ignored,
                         steering_generation_previous_observed_offset=previous_offset,
                         steering_generation_current_observed_offset=current_offset,
                         steering_generation_observed_offset_motion=offset_motion,
                         steering_release_edge_pad_px=EDGE_PAD, steering_release_sample_pad_px=SAMPLE_PAD,
                         steering_release_kinematic_guard_not_certification=True)
        self._previous = tuple(obj) if obj is not None else None
        self._previous_side, self._last_steer = side, float(action[0])
        self.driver.last_command = tuple(float(value) for value in action)
        return action

    def last_step_diagnostics(self):
        return dict(self.last)
