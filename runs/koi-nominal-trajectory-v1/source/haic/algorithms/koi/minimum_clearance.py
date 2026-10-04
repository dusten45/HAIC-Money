"""Pixel-only minimum supported-clearance steering over frozen crossing KOI.

This is a supported image-space reference path, not a physical safety proof.
The baseline still computes every speed target and pedal, including fallback.
"""

from importlib import import_module

import cv2
import numpy as np


X_SCALE = 1.3608  # Steady zoom 2.7*6, window1000 -> observation84.
Y_SCALE = 1.701  # window800 -> observation84.
WHEELBASE = 3.24  # (80 - (-82))*0.02.
POLYGON_SKIN_M = .01
NOMINAL_HALF_WIDTH_M = 1.1 + .28 * np.cos(.4) + .54 * np.sin(.4)
HALF_WIDTH_M = NOMINAL_HALF_WIDTH_M + POLYGON_SKIN_M
HALF_LENGTH_M = 2.6 + POLYGON_SKIN_M  # Includes front130/rear120*0.02 and skin.
OBSTACLE_MARGIN_PX = 1.0  # One threshold/downsampling edge pixel.
PATH_MARGIN_PX = .5  # Half-pixel center quantization and <=.5px sweep spacing.
DISPLACEMENT_RESOLUTION_PX = .05


def car_occlusion_mask():
    """Fixed car silhouette in the car-follow camera, with one raster pixel pad.

    Only this known current-car silhouette may interrupt near-field asphalt;
    visible approach/wheel-support pixels outside it must remain asphalt.
    """
    mask = np.zeros((84, 84), np.uint8)
    hulls = [[(-60, 130), (60, 130), (60, 110), (-60, 110)],
             [(-15, 120), (15, 120), (20, 20), (-20, 20)],
             [(25, 20), (50, -10), (50, -40), (20, -90), (-20, -90), (-50, -40), (-50, -10), (-25, 20)],
             [(-50, -120), (50, -120), (50, -90), (-50, -90)]]
    polygons = [np.asarray(poly, np.float64) * .02 for poly in hulls]
    wheel = np.asarray([(-.28, .54), (.28, .54), (.28, -.54), (-.28, -.54)])
    for wx, wy in ((-1.1, 1.6), (1.1, 1.6), (-1.1, -1.64), (1.1, -1.64)):
        for angle in np.linspace(-.4, .4, 9):
            rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            polygons.append(wheel @ rotation.T + [wx, wy])
    for poly in polygons:
        pixels = np.rint(poly * [X_SCALE, -Y_SCALE] + [42., 63.]).astype(np.int32)
        cv2.fillPoly(mask, [pixels], (1,))
    return cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(bool)


def component_bounds(frame, obj):
    """Match the baseline centroid to an untruncated full bright component."""
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        (frame >= .54).astype(np.uint8), connectivity=8
    )
    matches = [i for i in range(1, count)
               if np.hypot(centroids[i, 0] - obj[1], centroids[i, 1] - obj[0]) < .01]
    if len(matches) != 1:
        return None
    left, top, width, height, area = map(int, stats[matches[0]])
    if not (4 <= area <= 80 and 2 <= width <= 9 and 2 <= height <= 10
            and 0 < left and left + width < 84 and 22 < top and top + height < 62):
        return None
    return left, top, width, height


def supported_boundaries(frame, centers, box):
    """Observed contiguous asphalt edges, excluding only the matched obstacle.

    Do not mistake an image/search limit or a dark/bright interior hole for a
    road edge. No inferred geometry or simulator state is used.
    """
    left, top, width, height = box
    bounds = {}
    rows = sorted(centers)
    occlusion = car_occlusion_mask()
    for row in range(8, 70):
        center = float(np.interp(row, rows, [centers[y] for y in rows]))
        xs = np.flatnonzero((frame[row] >= .24) & (frame[row] <= .52)
                            & (np.abs(np.arange(84) - center) <= 17))
        if len(xs) < 4:
            continue
        lo, hi = int(xs[0]), int(xs[-1])
        if lo <= 0 or hi >= 83 or abs(lo - 1 - center) > 17 or abs(hi + 1 - center) > 17:
            continue
        if .24 <= frame[row, lo - 1] <= .52 or .24 <= frame[row, hi + 1] <= .52:
            continue
        supported = (frame[row, lo:hi + 1] >= .24) & (frame[row, lo:hi + 1] <= .52)
        supported |= occlusion[row, lo:hi + 1]
        if top <= row < top + height:
            indices = np.arange(lo, hi + 1)
            supported |= (indices >= left) & (indices < left + width)
        if np.all(supported):
            # These are pixel centers, already conservative by half a pixel.
            bounds[row] = (float(lo), float(hi))
    return bounds


def sweep_clearance(displacements, centers, box, boundaries):
    """Check the full hull/wheel envelope along smooth lateral reference paths.

    A1px obstacle-edge pad plus .5px footprint pad gives1.5px total separation.
    The body rectangle bounds every hull and every front-wheel joint angle in
    [-.4,.4], including the wider wheel corners, not just the red hull.
    """
    displacements = np.asarray(displacements, dtype=np.float64).reshape(-1)
    left, top, width, height = box
    # Continue until even the rear has passed the object's far extent.
    end = top - OBSTACLE_MARGIN_PX - Y_SCALE * (HALF_LENGTH_M + HALF_WIDTH_M) - 1
    rows = np.arange(63., end, -.5)
    known = sorted(centers) + [63]
    road_x = np.interp(rows, known, [centers[y] for y in sorted(centers)] + [42.])
    contact_row = top + height - 1 + OBSTACLE_MARGIN_PX + Y_SCALE * (HALF_LENGTH_M + HALF_WIDTH_M)
    ramp_length = max(63. - contact_row, .5)
    phase = np.clip((63. - rows) / ramp_length, 0., 1.)
    profile = .5 - .5 * np.cos(np.pi * phase)
    x = road_x + displacements[:, None] * profile
    # Reference yaw includes the lane transition, not only asphalt heading.
    slope = -np.gradient(x, -.5, axis=1) * Y_SCALE / X_SCALE
    angle = np.arctan(slope)
    valid, road_margin, obstacle_margin = footprint_clearance(x, angle, rows, box, boundaries)
    valid &= np.all(np.abs(angle) <= .6, axis=1)
    return valid, road_margin, obstacle_margin


def footprint_clearance(x, angle, rows, box, boundaries):
    left, top, width, height = box
    sine, cosine = np.sin(angle), np.cos(angle)
    valid = np.ones(x.shape[0], bool)
    half_x = X_SCALE * (HALF_WIDTH_M * cosine + HALF_LENGTH_M * np.abs(sine)) + PATH_MARGIN_PX
    half_y = Y_SCALE * (HALF_LENGTH_M * cosine + HALF_WIDTH_M * np.abs(sine)) + PATH_MARGIN_PX
    xlo, xhi = x - half_x, x + half_x
    ylo, yhi = rows - half_y, rows + half_y
    overlap = ((yhi >= top - OBSTACLE_MARGIN_PX)
               & (ylo <= top + height - 1 + OBSTACLE_MARGIN_PX)
               & (xhi >= left - OBSTACLE_MARGIN_PX)
               & (xlo <= left + width - 1 + OBSTACLE_MARGIN_PX))
    valid &= ~np.any(overlap, axis=1)
    # Approach56..57 and near-wheel road support are checked too. Only the
    # explicitly rendered current-car silhouette can mask asphalt pixels.
    valid &= np.all(ylo >= 8, axis=1)
    road_margin = np.full(x.shape[0], np.inf)
    valid &= np.all(yhi <= 69, axis=1)
    for row in range(8, 70):
        occupied = (ylo <= row + .5) & (yhi >= row - .5)
        if row not in boundaries:
            valid &= ~np.any(occupied, axis=1)
            continue
        lo, hi = boundaries[row]
        margin = np.minimum(xlo - lo, hi - xhi)
        valid &= ~np.any(occupied & (margin < 0), axis=1)
        road_margin = np.minimum(road_margin, np.min(np.where(occupied, margin, np.inf), axis=1))
    vertical = (yhi >= top - OBSTACLE_MARGIN_PX) & (ylo <= top + height - 1 + OBSTACLE_MARGIN_PX)
    obstacle_margin = np.min(np.where(vertical,
        np.maximum(left - OBSTACLE_MARGIN_PX - xhi,
                   xlo - (left + width - 1 + OBSTACLE_MARGIN_PX)), np.inf), axis=1)
    return valid, road_margin, obstacle_margin


def commanded_path_clear(frame, centers, box, steer, forward_limit=None):
    """Additional same-command arc guard, including the applied correction.

    By default the first candidate requires full rear-clear on this command.
    The separate hold variant checks only the distance before the next action;
    its full desired-lane footprint is still checked by minimum_command.
    """
    end = box[1] - OBSTACLE_MARGIN_PX - Y_SCALE * (HALF_LENGTH_M + HALF_WIDTH_M) - 1
    rows = np.arange(63., end, -.5)
    if forward_limit is not None:
        if not np.isfinite(forward_limit) or forward_limit <= 0:
            return False
        distance = min((63. - end) / Y_SCALE, forward_limit)
        rows = np.linspace(63., 63. - Y_SCALE * distance,
                           max(2, int(np.ceil(Y_SCALE * distance / .5)) + 1))
    forward = (63. - rows) / Y_SCALE
    curvature = np.tan(steer) / WHEELBASE
    sine = curvature * forward
    if np.any(np.abs(sine) >= .95):
        return False
    cosine = np.sqrt(1. - sine**2)
    x = (42. + X_SCALE * curvature * forward**2 / (1. + cosine))[None, :]
    angle = np.arcsin(sine)[None, :]
    valid, _, _ = footprint_clearance(x, angle, rows, box, supported_boundaries(frame, centers, box))
    return bool(valid[0])


def minimum_command(frame, centers, box, road, side):
    """Minimum lateral lane offset on the baseline-selected flank.

    Convert the geometric target with the original corridor's two gains. This
    changes the target, never multiplies the baseline obstacle steering term.
    """
    boundaries = supported_boundaries(frame, centers, box)
    displacements = side * np.arange(0., 12. + DISPLACEMENT_RESOLUTION_PX / 2, DISPLACEMENT_RESOLUTION_PX)
    valid, road_margin, obstacle_margin = sweep_clearance(displacements, centers, box, boundaries)
    contact_row = box[1] + box[3] - 1 + OBSTACLE_MARGIN_PX + Y_SCALE * (HALF_LENGTH_M + HALF_WIDTH_M)
    phase = np.clip((63. - np.asarray([42., 54.])) / max(63. - contact_row, .5), 0., 1.)
    profile = .5 - .5 * np.cos(np.pi * phase)
    commands = road + displacements * (.022 * profile[0] + .018 * (profile[0] - profile[1]))
    valid &= np.abs(commands) <= .4
    indices = np.flatnonzero(valid)
    if not len(indices):
        return None
    index = int(indices[0])
    return dict(steer=float(commands[index]), road_margin=float(road_margin[index]),
                obstacle_margin=float(obstacle_margin[index]),
                displacement=float(displacements[index]),
                target_x=float(np.interp(box[1] + (box[3] - 1) / 2, sorted(centers),
                                        [centers[y] for y in sorted(centers)]) + displacements[index]),
                road_clear=bool(valid[0]), supported_rows=len(boundaries))


class MinimumClearanceAgent:
    def __init__(self, driver=None, command_hold_seconds=None, require_projection_clearance=False):
        if command_hold_seconds not in (None, .08):
            raise ValueError('command hold must be full-passage or the actual .08s action')
        if require_projection_clearance and command_hold_seconds is None:
            raise ValueError('projection clearance is the separate actual-hold correction')
        if driver is None:
            driver = import_module("haic_agent.contact_continuity_runtime").ContactContinuityAgent("crossing_projection")
        self.driver = driver
        self.command_hold_seconds = command_hold_seconds
        self.require_projection_clearance = require_projection_clearance
        self.last = {}

    def reset(self, observation=None):
        self.driver.reset(observation)
        self.last = {}

    def act(self, observation):
        impact_before = self.driver.impact_left
        prior_speeds = tuple(self.driver.speed_history)
        action = self.driver.act(observation).copy()
        original = action.copy()
        info = self.driver.last
        obj = self.driver.base._last_obstacle
        centers = info["road_centers"]
        reason = "no_near_obstacle"
        box = proposal = road = None
        command_distance = command_speed = None
        required_projection_gap = None
        if self.command_hold_seconds is not None:
            # Averaging can hide a newly saturated HUD, so inspect raw frames too.
            images = np.asarray(observation)
            images = images[-4:] if images.ndim == 3 else images[None]
            speeds = [info['pixel_speed'], *prior_speeds[-4:]]
            valid_images = (images.shape[1:] == (84, 84) and np.isfinite(images).all())
            if valid_images:
                speeds.extend(self.driver.base._estimate_speed(image) for image in images)
            if valid_images and all(np.isfinite(speed) and 0 <= speed < 80 for speed in speeds):
                command_speed = float(max(speeds))
            command_distance = (command_speed * self.command_hold_seconds
                                if command_speed is not None and command_speed > 0 else 0.)
        side = float(self.driver.base._obstacle_side)
        projection = bool(info["contact_active"])
        projected = info["projected_obstacle_x"]
        clearing = (impact_before > 0 or info["impact_proxy_trigger"]) and not info.get("braking_proxy_veto", False)
        baseline_avoidance = bool(obj is not None and self.driver.steps > 10 and not clearing)
        if self.driver.steps <= 10:
            reason = "launch_prefix"
        elif obj is not None:
            if (impact_before > 0 or self.driver.impact_left > 0 or info["impact_proxy_trigger"]
                    or info["contact_proxy"] or (prior_speeds and max(prior_speeds) - info["pixel_speed"] > 10)):
                reason = "impact_guard"
            elif set(centers) != {30, 34, 38, 42, 46, 50, 54} or info["geometry_repaired"]:
                reason = "geometry_guard"
            elif side not in (-1., 1.) or not 30 <= obj[0] <= 55:
                reason = "object_range_guard"
            elif abs(centers[54] - 42.) > 6 or max(centers.values()) - min(centers.values()) > 12:
                reason = "alignment_guard"
            else:
                road = float(np.clip(.022 * (centers[42] - 42.) + .018 * (centers[42] - centers[54]), -.7, .7))
                road = float(np.clip(road + info["correction"], -.7, .7))
                reason = "road_steer_guard"
                if abs(road) <= .25:
                    frame = self.driver.base._frame(observation)
                    box = component_bounds(frame, obj)
                    reason = "component_guard"
                    if box is not None:
                        proposal = minimum_command(frame, centers, box, road, side)
                        reason = "unsupported_sweep"
                        if proposal is not None:
                            radius = max(obj[1] - box[0], box[0] + box[2] - 1 - obj[1])
                            required_projection_gap = (X_SCALE * HALF_WIDTH_M + radius
                                                       + OBSTACLE_MARGIN_PX + PATH_MARGIN_PX)
                            # Never return on a contradictory imminent temporal crossing.
                            conflict = (proposal["road_clear"] and projected is not None
                                        and abs(projected - 42.) < X_SCALE * HALF_WIDTH_M + box[2] / 2 + 1.5)
                            if self.require_projection_clearance and (
                                    projected is None or not np.isfinite(projected)
                                    or side * (projected - 42.) >= -required_projection_gap):
                                reason = 'projection_footprint_guard'
                            elif conflict:
                                reason = "return_projection_guard"
                            elif not commanded_path_clear(frame, centers, box, proposal["steer"], command_distance):
                                reason = "applied_command_guard"
                            else:
                                action[0] = proposal["steer"]
                                reason = "return_centerline" if proposal["road_clear"] else "minimum_clearance"
        changed = bool(action[0] != original[0])
        returning = reason == "return_centerline"
        avoidance = bool(baseline_avoidance and not returning)
        self.driver.last_command = float(action[0])
        self.driver.last.update(
            minimum_clearance_reason=reason, minimum_clearance_changed=changed,
            minimum_clearance_box=None if box is None else list(box),
            minimum_clearance_road_steer=road,
            minimum_clearance_steer=None if proposal is None else proposal["steer"],
            minimum_clearance_displacement_px=None if proposal is None else proposal["displacement"],
            minimum_clearance_target_x=None if proposal is None else proposal["target_x"],
            minimum_clearance_road_margin_px=None if proposal is None else proposal["road_margin"],
            minimum_clearance_obstacle_margin_px=None if proposal is None else proposal["obstacle_margin"],
            minimum_clearance_supported_rows=None if proposal is None else proposal["supported_rows"],
            minimum_clearance_half_width_m=float(HALF_WIDTH_M),
            minimum_clearance_nominal_half_width_m=float(NOMINAL_HALF_WIDTH_M),
            minimum_clearance_polygon_skin_m=POLYGON_SKIN_M,
            minimum_clearance_half_length_m=HALF_LENGTH_M,
            minimum_clearance_quantization_margin_px=1.5,
            minimum_clearance_displacement_resolution_px=DISPLACEMENT_RESOLUTION_PX,
            minimum_clearance_command_hold_seconds=self.command_hold_seconds,
            minimum_clearance_command_distance_m=command_distance,
            minimum_clearance_command_speed_bound_mps=command_speed,
            minimum_clearance_require_projection_clearance=self.require_projection_clearance,
            minimum_clearance_required_projection_gap_px=required_projection_gap,
            baseline_steer=float(original[0]), baseline_pedals=original[1:].tolist(),
            baseline_projection_active=projection,
            baseline_projection_available=projected is not None,
            baseline_projection_clear=bool(projected is not None and abs(projected - 42.) >= 6.),
            baseline_avoidance_active=baseline_avoidance,
            baseline_avoidance_component=None if road is None else float(original[0]) - road,
            avoidance_active=avoidance,
            avoidance_component=None if road is None else float(action[0]) - road,
            return_centerline_active=returning,
            recovery_active=returning,
            final_steer=float(action[0]), final_gas=float(action[1]), final_brake=float(action[2]),
        )
        self.last = dict(self.driver.last)
        return action

    def last_step_diagnostics(self):
        return dict(self.last)
