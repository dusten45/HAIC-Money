"""Bilateral approach/pass/rejoin selection before the unchanged v1 shield.

Only reference generation and its nominal steering command change. Clearance
geometry is unchanged; pixel paths and the lag model are not safety guarantees.
"""

import numpy as np

from haic.algorithms.koi.minimum_clearance import (
    HALF_LENGTH_M, HALF_WIDTH_M, OBSTACLE_MARGIN_PX, PATH_MARGIN_PX,
    X_SCALE, Y_SCALE, component_bounds, footprint_clearance,
    supported_boundaries,
)
from haic.algorithms.koi.collision_shield import (
    detect_boxes, footprint_gaps, project_paths,
)
from haic.algorithms.koi.trajectory_rollout import rollout_trajectories


def checked_rollouts(references, frame, centers, box, speed, wheel_angle, preview, remaining_actions=12):
    actual, commands, variation = rollout_trajectories(references, speed, wheel_angle, preview)
    boundaries = supported_boundaries(frame, centers, box)
    boxes = detect_boxes(frame)
    gaps = footprint_gaps(actual, boxes)
    holds, _, _ = project_paths(commands, 0., speed, wheel_angle, actions=1)
    gaps = np.minimum(gaps, footprint_gaps(holds, boxes))
    terminal = actual[:, :, 1] >= references[:, -1, 1, None] - .01
    terminal_actions = np.ceil(np.argmax(terminal, axis=1) / (holds.shape[1] - 1))
    feasible = ((gaps > 0) & np.all(np.abs(actual[:, :, 2]) <= .8, axis=1)
                & terminal.any(axis=1) & (terminal_actions <= remaining_actions)
                & (actual[:, -1, 1] >= references[:, -1, 1] - .01)
                & (np.abs(actual[:, -1, 0] - references[:, -1, 0]) <= 1.)
                & (np.abs(actual[:, -1, 2] - references[:, -1, 2]) <= .2))
    margins = []
    for index, path in enumerate(actual):
        valid, road_margin, obstacle_margin = footprint_clearance(
            (42. + X_SCALE * path[:, 0])[None], path[:, 2][None],
            63. - Y_SCALE * path[:, 1], box, boundaries)
        feasible[index] &= bool(valid[0])
        hold = holds[index]
        hold_valid, _, _ = footprint_clearance(
            (42. + X_SCALE * hold[:, 0])[None], hold[:, 2][None],
            63. - Y_SCALE * hold[:, 1], box, boundaries)
        feasible[index] &= bool(hold_valid[0])
        margins.append((float(road_margin[0]), float(obstacle_margin[0])))
    return actual, commands, variation, feasible, gaps, margins


def select_trajectory(frame, centers, box, speed, wheel_angle, previous_steer):
    """Rank complete footprint-safe paths, not a first feasible inherited flank."""
    left, top, width, height = box
    envelope = Y_SCALE * np.hypot(HALF_LENGTH_M, HALF_WIDTH_M) + PATH_MARGIN_PX
    approach = 63. - (top + height - 1 + OBSTACLE_MARGIN_PX + envelope)
    passed = 63. - (top - OBSTACLE_MARGIN_PX - envelope)
    rejoin = 16.  # Longitudinal pixels, not a reduced safety margin.
    end = passed + rejoin
    if approach < 3. or 63. - end - envelope < 8:
        return None
    rows = np.linspace(63., 63. - end, int(np.ceil(end / .5)) + 1)
    distance = (63. - rows) / Y_SCALE
    known = sorted(centers)
    road = np.interp(rows, known, [centers[row] for row in known])
    # Match the car's initial heading and rejoin the observed road with no
    # residual lateral offset or lateral-profile derivative at the terminal.
    origin_phase = np.clip((63. - rows) / 9., 0., 1.)
    smooth = lambda phase: phase**3 * (10. - 15. * phase + 6. * phase**2)
    road = 42. + (road - 42.) * smooth(origin_phase)
    entry = smooth(np.clip((63. - rows) / approach, 0., 1.))
    exit_phase = smooth(np.clip((63. - rows - passed) / rejoin, 0., 1.))
    profile = entry * (1. - exit_phase)
    offsets = np.arange(-12., 12.01, .5)
    x = road[None, :] + offsets[:, None] * profile
    dx = np.gradient((x - 42.) / X_SCALE, distance, axis=1)
    angle = np.arctan(dx)
    boundaries = supported_boundaries(frame, centers, box)
    boxes = detect_boxes(frame)
    feasible, _, _ = footprint_clearance(
        x, angle, rows, box, boundaries)
    paths = np.stack(((x - 42.) / X_SCALE,
                      np.broadcast_to(distance, x.shape), angle), axis=2)
    # All components, including a different next obstacle, must be clear.
    gaps = footprint_gaps(paths, boxes)
    feasible &= (gaps > 0) & np.all(np.abs(angle) <= .8, axis=1)
    preview = min(max(3., speed * .12), approach / Y_SCALE)
    indices = np.flatnonzero(feasible)
    if not len(indices):
        return None
    references = paths[indices]
    actual, commands, variation, valid, actual_gaps, margins = checked_rollouts(
        references, frame, centers, box, speed, wheel_angle, preview)
    eligible = np.flatnonzero(valid)
    if not len(eligible):
        return None
    length = np.linalg.norm(np.diff(actual[:, :, :2], axis=1), axis=2).sum(axis=1)
    actual_road = np.interp(63. - Y_SCALE * actual[:, :, 1], known,
                            [centers[row] for row in known])
    lateral = np.max(np.abs(actual[:, :, 0] - (actual_road - 42.) / X_SCALE), axis=1)
    variation += np.abs(commands.astype(float) - previous_steer)
    cost = length - distance[-1] + .25 * lateral + .5 * variation
    index = int(eligible[np.argmin(cost[eligible])])
    reference_index = int(indices[index])
    return dict(steer=float(commands[index]), side=float(np.sign(offsets[reference_index])),
                displacement_px=float(offsets[reference_index]), max_lateral_m=float(lateral[index]),
                path_length_m=float(length[index]), steering_variation=float(variation[index]),
                cost=float(cost[index]), candidate_count=int(len(eligible)),
                road_margin_px=margins[index][0], obstacle_margin_px=margins[index][1],
                all_object_gap_m=float(actual_gaps[index]), terminal_offset_px=0.,
                terminal_tracking_error_m=float(actual[index, -1, 0] - references[index, -1, 0]),
                terminal_heading_error_rad=float(actual[index, -1, 2] - references[index, -1, 2]),
                _reference=references[index], _preview=preview)


class NominalTrajectoryAgent:
    """Transparent nominal adapter without `.driver`, which v1 would unwrap."""

    def __init__(self, nominal):
        if getattr(nominal, 'contact_mode', None) != 'crossing_projection':
            raise ValueError('requires frozen crossing_projection')
        self.nominal = nominal
        self.wheel_angle = self.previous_steer = 0.
        self.last = {}
        self.last_nominal_action = None
        self.reference: np.ndarray | None = None
        self.preview = self.speed = 0.
        self.plan_actions = 0

    def __getattr__(self, name):
        if name == 'driver':
            raise AttributeError(name)
        return getattr(self.nominal, name)

    def reset(self, observation=None):
        self.nominal.reset(observation)
        self.wheel_angle = self.previous_steer = 0.
        self.last = {}
        self.last_nominal_action = None
        self.reference = None
        self.preview = self.speed = 0.
        self.plan_actions = 0

    def observe_executed_action(self, action):
        """Receive final post-shield action, never simulator actuator state."""
        paths, wheels, hold = project_paths([float(action[0])], float(action[0]), self.speed,
                                           self.wheel_angle, actions=1)
        self.wheel_angle = float(wheels[0])
        self.previous_steer = float(action[0])
        if self.reference is not None:
            pose = paths[0, hold]
            dx, dy = self.reference[:, 0] - pose[0], self.reference[:, 1] - pose[1]
            sine, cosine = np.sin(pose[2]), np.cos(pose[2])
            transformed = np.column_stack((dx * cosine - dy * sine,
                                           dx * sine + dy * cosine,
                                           self.reference[:, 2] - pose[2]))
            future = transformed[transformed[:, 1] > 0]
            if (len(future) < 2 or future[-1, 1] < max(1., self.speed * .08)
                    or np.any(np.diff(future[:, 1]) <= 0)):
                self.reference = None
            else:
                self.reference = np.vstack((np.zeros(3), future))

    def act(self, observation):
        original = self.nominal.act(observation)
        self.last_nominal_action = original.tolist()
        action = original.copy()
        info = self.nominal.last_step_diagnostics()
        obj = self.nominal.base._last_obstacle
        frame = self.nominal.base._frame(observation)
        centers = info.get('road_centers', {})
        proposal = None
        reason = 'no_obstacle'
        images = np.asarray(observation)
        images = images[-2:] if images.ndim == 3 else images[None]
        readings = [float(info.get('pixel_speed', 0.))]
        if frame is not None and frame.shape == (84, 84) and np.isfinite(frame).all():
            readings.extend(float(self.nominal.base._estimate_speed(image)) for image in images)
        speed_valid = (len(readings) > 1 and 0 < readings[0] < 80
                       and all(np.isfinite(value) and 0 <= value < 80 for value in readings))
        valid = (speed_valid and len(centers) >= 7 and not info.get('geometry_repaired')
                 and not info.get('impact_proxy_trigger') and self.nominal.impact_left == 0
                 and not info.get('contact_proxy'))
        self.speed = max(readings) if speed_valid else 0.
        if not valid or self.plan_actions >= 12:
            self.reference = None
        if self.nominal.steps <= 10:
            reason = 'launch_prefix'
        elif self.reference is not None:
            box = component_bounds(frame, obj) if obj is not None else None
            box = box if box is not None else (0, 0, 0, 0)
            actual, commands, variation, feasible, gaps, margins = checked_rollouts(
                self.reference[None], frame, centers, box, self.speed, self.wheel_angle, self.preview,
                remaining_actions=12 - self.plan_actions)
            reason = 'remaining_trajectory_invalid'
            if feasible[0]:
                action[0] = commands[0]
                self.plan_actions += 1
                reason = 'nominal_trajectory_continue'
            else:
                self.reference = None
        elif obj is not None:
            reason = 'geometry_or_impact_guard'
            if valid and 26 <= obj[0] <= 50:
                reason = 'unknown_speed_or_component'
                box = component_bounds(frame, obj)
                if box is not None:
                    proposal = select_trajectory(frame, centers, box, self.speed,
                                                 self.wheel_angle, self.previous_steer)
                    reason = 'no_feasible_complete_trajectory'
                    if proposal is not None:
                        action[0] = proposal['steer']
                        self.reference = np.asarray(proposal.pop('_reference'), dtype=np.float64)
                        self.preview = float(proposal.pop('_preview'))
                        self.plan_actions = 1
                        reason = 'nominal_trajectory'
        changed = bool(action[0] != original[0])
        self.last = dict(info, nominal_reason=reason, nominal_changed=changed,
                         nominal_baseline_action=original.tolist(),
                         nominal_proposal=proposal, nominal_plan_actions=self.plan_actions,
                         nominal_plan_remaining=self.reference is not None,
                         final_steer=float(action[0]))
        return action

    def last_step_diagnostics(self):
        return dict(self.last)
