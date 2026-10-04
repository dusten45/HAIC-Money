"""Stateless clearance-demand steering before the unchanged collision-shield v1.

The current selected flank, pedals and steering pipeline are inherited. Geometry
and crossing motion are pixel proxies, not a collision-free driving guarantee.
"""

import math
from typing import Any

import numpy as np

from haic.algorithms.koi.collision_shield import (
    ACTION_SECONDS, FRONT, HALF_WIDTH, HORIZON_ACTIONS, WHEELBASE, X_SCALE, Y_SCALE,
)
from haic.algorithms.koi.minimum_clearance import component_bounds
from haic.algorithms.koi.steering_terms import reconstruct_steering


def bounded_magnitude(box, side, speed, legacy_magnitude, projected_shift_m=0.) -> dict[str, Any] | None:
    """Current-frame lateral demand plus imminent projected overlap, never a plan.

    `box` already includes the same one-pixel object pad as v1. Its half-pixel
    footprint allowance is retained too; only the nominal steering magnitude
    changes. The arc demand is a geometric control proxy, not a held trajectory.
    """
    if (len(box) != 4 or not all(math.isfinite(v) for v in (*box, speed,
            legacy_magnitude, projected_shift_m)) or side not in (-1., 1.)
            or not 0 < speed < 80 or not 0 < legacy_magnitude <= .55
            or box[2] <= box[0] or box[3] <= box[1]):
        return None
    xlo, ylo, xhi, _ = box
    width = HALF_WIDTH + .5 / X_SCALE
    required = max(0., width + side * (xhi if side > 0 else xlo))
    approach = max(0., ylo - FRONT - .5 / Y_SCALE)
    ttc = approach / speed
    distance = max(approach, speed * ACTION_SECONDS)
    demand = math.atan2(2. * WHEELBASE * required, distance**2)

    # Both endpoints can be clear while observed crossing motion sweeps through
    # the car. Enclose the transverse sweep, not only the y=60 projected endpoint.
    swept_gap = max(min(xlo, xlo + projected_shift_m) - width,
                    -width - max(xhi, xhi + projected_shift_m))
    penetration = float(np.clip(-swept_gap / (.5 / X_SCALE), 0., 1.))
    urgency = float(np.clip((HORIZON_ACTIONS * ACTION_SECONDS - ttc)
                           / ((HORIZON_ACTIONS - 1) * ACTION_SECONDS), 0., 1.))
    risk = penetration * urgency
    magnitude = min(legacy_magnitude, max(demand, legacy_magnitude * risk))
    return dict(required_clearance_m=required, approach_m=approach, ttc_s=ttc,
                geometric_demand=demand, swept_gap_m=swept_gap,
                collision_risk=risk, legacy_magnitude=legacy_magnitude,
                magnitude=magnitude, side=side, current_bbox_m=list(box),
                projected_shift_m=projected_shift_m)


def replacement_steer(terms, magnitude):
    """Replace only the identified avoidance term through its original clips."""
    raw = terms['raw_terms']
    road = raw['selected_road_position'] + raw['selected_lookahead']
    correction = raw['damping']
    avoidance = terms['selected_flank'] * magnitude
    path = terms['active_path']
    if path == 'geometry_repair' or (path == 'crossing_replacement'
                                    and terms['active_geometry_repaired']):
        steering = np.clip(road + avoidance + correction, -.7, .7)
    else:
        inner = np.clip(road + avoidance, -.7, .7)
        if path == 'inherited_damping':
            inner = float(np.float32(inner))
        steering = np.clip(inner + correction, -.7, .7)
    return float(np.float32(steering))


class AvoidanceMagnitudeAgent:
    """No new controller state: only the original nominal and last diagnostics."""

    def __init__(self, nominal):
        if getattr(nominal, 'contact_mode', None) != 'crossing_projection':
            raise ValueError('requires frozen crossing_projection')
        self.nominal = nominal
        self.last: dict[str, Any] = {}

    def __getattr__(self, name):
        # v1 unwraps `.driver`; keep this nominal adapter ahead of its safety layer.
        if name == 'driver':
            raise AttributeError(name)
        return getattr(self.nominal, name)

    def reset(self, observation=None):
        self.nominal.reset(observation)
        self.last = {}

    def act(self, observation):
        impact_before = self.nominal.impact_left
        original = self.nominal.act(observation)
        action = original.copy()
        info = self.nominal.last_step_diagnostics()
        obj = self.nominal.base._last_obstacle
        proposal = None
        reason = 'no_obstacle'
        if self.nominal.steps <= 10:
            reason = 'launch_prefix'
        elif obj is not None:
            terms = reconstruct_steering(info, float(original[0]), state=dict(
                obstacle_side=float(self.nominal.base._obstacle_side),
                steps=int(self.nominal.steps), impact_left_before_act=impact_before))
            reason = 'unidentified_or_protected_steering'
            if (terms['reconstruction_valid'] and terms['residual_valid']
                    and not terms['unidentifiable']
                    and terms['active_path'] in ('inherited_damping', 'geometry_repair',
                                                 'crossing_replacement')
                    and terms['effective_avoidance_raw'] is not None
                    and abs(terms['effective_avoidance_raw']) > 0):
                frame = self.nominal.base._frame(observation)
                images = np.asarray(observation)
                images = images[-2:] if images.ndim == 3 else images[None]
                readings = [float(info.get('pixel_speed', 0.))]
                valid = (frame is not None and frame.shape == (84, 84)
                         and np.isfinite(frame).all() and images.shape[1:] == (84, 84)
                         and np.isfinite(images).all())
                if valid:
                    readings.extend(float(self.nominal.base._estimate_speed(image))
                                    for image in images)
                valid = (valid and 0 < readings[0] < 80
                         and all(math.isfinite(v) and 0 <= v < 80 for v in readings))
                reason = 'unknown_geometry_or_speed'
                box = component_bounds(frame, obj) if valid else None
                projected = info.get('projected_obstacle_x')
                if box is not None and (projected is None or math.isfinite(projected)):
                    left, top, width, height = box
                    expanded = ((left - 1 - 42) / X_SCALE, (63 - top - height) / Y_SCALE,
                                (left + width - 42) / X_SCALE, (64 - top) / Y_SCALE)
                    shift = 0. if projected is None else (projected - obj[1]) / X_SCALE
                    proposal = bounded_magnitude(expanded, terms['selected_flank'],
                                                 max(readings), abs(terms['effective_avoidance_raw']), shift)
                    if proposal is not None:
                        terms['active_geometry_repaired'] = bool(info['geometry_repaired'])
                        if proposal['magnitude'] < proposal['legacy_magnitude']:
                            action[0] = replacement_steer(terms, proposal['magnitude'])
                        reason = 'bounded_magnitude'
        self.last = dict(info, magnitude_baseline_action=original.tolist(),
                         magnitude_changed=bool(action[0] != original[0]),
                         magnitude_reason=reason, magnitude_proposal=proposal,
                         final_steer=float(action[0]))
        return action

    def last_step_diagnostics(self):
        return dict(self.last)
