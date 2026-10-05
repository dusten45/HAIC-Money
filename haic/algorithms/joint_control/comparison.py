"""Pure paired local comparisons, not an action selector or a safety guarantee.

Only grayscale images/causal image motion supply road and obstacle geometry.
Physical yaw is CCW, x right/y forward, and pixels have unequal metric scales.
Costs are a frozen short-horizon proxy, NOT official progress, lap utility, or
directed road heading. Increasing image-forward road arc is the local direction.
The same initial-state hypotheses and current nominal continuation serve every
action. Finite basis scenarios are not a certified nonlinear uncertainty bound.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType

import cv2
import numpy as np

from .motion import estimate_body_motion
from .physics import PhysicsParameters, PhysicsState, RAW_DT, predict


X_SCALE, Y_SCALE = 1.3608, 1.701
FRONT, REAR = 2.61, 2.41
HALF_WIDTH = 1.1 + .28 * np.cos(.4) + .54 * np.sin(.4) + .01
CORNER_RADIUS = float(np.hypot(FRONT, HALF_WIDTH))
COST_FLOOR = .05
COST_WEIGHTS = MappingProxyType(dict(
    progress=-1., lateral_squared=.20, heading_squared=2., road_risk=.50,
    obstacle_risk=1., road_clearance_loss=.25, obstacle_clearance_loss=.50,
    effort=.02,
))
_COL, _ROW = np.meshgrid(np.arange(84), np.arange(84))
_X, _Y = (_COL - 42.) / X_SCALE, (63. - _ROW) / Y_SCALE
_VISIBLE = (_ROW >= 8) & (_ROW <= 72) & (_COL >= 1) & (_COL <= 82)
_EGO = (np.abs(_X) <= HALF_WIDTH + 1.) & (_Y >= -REAR - .75) & (_Y <= FRONT + .75)
_CELL_RADIUS = float(.5 * np.hypot(1 / X_SCALE, 1 / Y_SCALE))


@dataclass(frozen=True)
class ObservedScene:
    """Image evidence in the CURRENT hull frame; missing centers remain NaN.

    road/known are84x84 masks, boxes are metric[xlo,ylo,xhi,yhi]. center_x holds
    one genuinely observed bounded road interval per raster row, not a filled
    nominal42. History is used only with a valid chain of causal registrations.
    Construct with extract_scene, not privileged road/contact/catalogue labels.
    """

    road: np.ndarray
    known: np.ndarray
    boxes: np.ndarray
    center_x: np.ndarray
    history_used: int
    motion_provenance: tuple[str, ...]


def _pixels(frame):
    frame = np.asarray(frame)
    if frame.shape != (84, 84):
        raise ValueError("expected84x84 grayscale frame")
    if frame.dtype == np.uint8:
        return frame.astype(float) / 255.
    if (not np.issubdtype(frame.dtype, np.floating)
            or not np.isfinite(frame).all() or np.any((frame < 0) | (frame > 1))):
        raise ValueError("frames must be uint8 or finite normalized floats")
    return frame.astype(float)


def _evidence(frame):
    road = (frame >= .24) & (frame <= .52) & _VISIBLE & ~_EGO
    bright = (frame >= .54) & _VISIBLE & ~_EGO
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8), connectivity=8)
    boxes, objects = [], np.zeros((84, 84), bool)
    # Asphalt and broad grass components are classified. Dark/ambiguous marks,
    # clipped small bright objects and ego pixels remain unknown, not clear.
    known = road.copy()
    for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if 4 <= area <= 80 and 2 <= w <= 9 and 2 <= h <= 10:
            if x > 1 and x + w < 83 and y > 8 and y + h < 73:
                boxes.append(((x - 1 - 42) / X_SCALE, (63 - y - h) / Y_SCALE,
                              (x + w - 42) / X_SCALE, (64 - y) / Y_SCALE))
                objects[labels == label] = True
                known[labels == label] = True
        elif area > 80:
            known[labels == label] = True
    return road, known, np.asarray(boxes, dtype=float).reshape(-1, 4), objects


def extract_scene(frames, *, motions: Sequence[Mapping] | None = None):
    """Extract current evidence from1..4 causal frames, oldest first.

    Optional motions are corresponding observer.mapping_motion or accepted
    estimate_body_motion results (not telemetry). A bounded, explicitly inferred
    observer mapping does not require successful flow on every frame. Its
    position_uncertainty/yaw_uncertainty are interval envelopes, not sigmas to
    refit here. Supplying cached motions avoids repeating registration.
    An invalid link blocks all earlier history. Historical free asphalt repairs
    ONLY current ego occlusion, after erosion for registration uncertainty;
    visible current unknown/offroad pixels cannot be overwritten by old asphalt.
    All old observed boxes are retained, conservatively enlarged, even on dropout.
    No extrapolation past image/road support or default center is performed.
    """
    frames = np.asarray(frames)
    if frames.ndim == 2:
        frames = frames[None]
    if frames.ndim != 3 or not 1 <= len(frames) <= 4:
        raise ValueError("expected1..4 causal grayscale frames")
    images = [_pixels(frame) for frame in frames]
    if motions is None:
        motions = [estimate_body_motion(a, b) for a, b in zip(frames[:-1], frames[1:])]
    if len(motions) != len(images) - 1:
        raise ValueError("one causal motion is required per image interval")
    road, known, current_boxes, objects = _evidence(images[-1])
    boxes = list(current_boxes)
    rotation, translation = np.eye(2), np.zeros(2)
    position_error, yaw_error, history_used = 0., 0., 0
    provenance = []
    grid = np.stack((_X, _Y), axis=-1)
    for index in range(len(images) - 2, -1, -1):
        motion = motions[index]
        if (not motion.get("valid", False)
                or not motion.get("accepted_for_correction", True)):
            break
        values = np.array([motion[key] for key in ("right", "forward", "yaw_delta", "residual_p90")])
        if values.shape != (4,) or not np.isfinite(values).all() or values[3] < 0:
            raise ValueError("valid causal motion must be finite with nonnegative residual")
        right, forward, yaw, residual = values
        c, s = np.cos(yaw), np.sin(yaw)
        step_rotation = np.array([[c, -s], [s, c]])
        translation = step_rotation @ translation + (right, forward)
        rotation = step_rotation @ rotation
        interval_error = np.array([motion.get("position_uncertainty", max(.08, residual)),
                                   motion.get("yaw_uncertainty", .05)], dtype=float)
        if interval_error.shape != (2,) or not np.isfinite(interval_error).all() or np.any(interval_error < 0):
            raise ValueError("mapping uncertainty must be finite and nonnegative")
        # Propagate interval errors separately from raster resampling; these are
        # engineering envelopes, NOT calibrated flow covariance/coverage.
        position_error += interval_error[0] + np.hypot(right, forward) * yaw_error
        yaw_error += interval_error[1]
        previous_grid = grid @ rotation.T + translation
        map_x = (42 + X_SCALE * previous_grid[..., 0]).astype(np.float32)
        map_y = (63 - Y_SCALE * previous_grid[..., 1]).astype(np.float32)
        old_road, _, old_boxes, _ = _evidence(images[index])
        distance = cv2.distanceTransform(old_road.astype(np.uint8), cv2.DIST_L2, 0) / Y_SCALE
        required = position_error + yaw_error * np.linalg.norm(previous_grid, axis=-1) + _CELL_RADIUS
        distance = cv2.remap(distance, map_x, map_y, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0.,))
        repair = _EGO & _VISIBLE & (distance > required)
        road |= repair
        known |= repair
        for box in old_boxes:
            corners = np.array([[box[0], box[1]], [box[0], box[3]],
                                [box[2], box[1]], [box[2], box[3]]])
            transformed = (corners - translation) @ rotation
            padding = position_error + yaw_error * np.linalg.norm(corners, axis=1).max()
            boxes.append(np.r_[transformed.min(axis=0) - padding,
                               transformed.max(axis=0) + padding])
        history_used += 1
        provenance.append(str(motion.get("provenance", "image_registration")))

    centers = np.full(84, np.nan)
    previous = None
    # Object holes may join two OBSERVED asphalt edges for the center reference
    # only. This never fills road/free-space support underneath an object.
    for row in range(72, 7, -1):
        mask = road[row] | objects[row]
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        runs = [(a, b) for a, b in zip(edges[::2], edges[1::2])
                if a > 1 and b < 83 and road[row, a:b].sum() >= 4
                and known[row, a - 1] and known[row, b]]
        if not runs:
            previous = None
            continue
        candidates = np.array([.5 * (a + b - 1) for a, b in runs])
        if len(candidates) > 1:
            if previous is None:
                continue
            distances = np.abs(candidates - previous)
            order = np.argsort(distances)
            if distances[order[1]] - distances[order[0]] <= 1:
                previous = None
                continue
            center = candidates[order[0]]
        else:
            center = candidates[0]
        if previous is not None and abs(center - previous) > 4:
            previous = None
            continue
        centers[row] = (center - 42) / X_SCALE
        previous = center
    return ObservedScene(road, known, np.asarray(boxes).reshape(-1, 4), centers,
                         history_used, tuple(provenance))


def paired_cost_summary(costs):
    """Subtract SAME-hypothesis baseline first; never subtract marginal extrema.

    Rows are baseline plus1/2 alternatives, columns shared hypotheses. Nonfinite
    support is retained and invalidates that row's interval, not silently dropped.
    The closed[-.05,.05] band is a tie; neither a tie nor unknown is a sign win.
    """
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 2 or costs.shape[0] not in (2, 3) or costs.shape[1] < 1:
        raise ValueError("costs must have shape(2or3, hypotheses)")
    with np.errstate(invalid="ignore"):
        delta = costs - costs[:1]
    valid = np.isfinite(delta).all(axis=1)
    interval = np.full((len(costs), 2), np.nan)
    interval[valid, 0] = delta[valid].min(axis=1)
    interval[valid, 1] = delta[valid].max(axis=1)
    sign = np.zeros(len(costs), dtype=np.int8)
    sign[valid & (interval[:, 0] > COST_FLOOR)] = 1
    sign[valid & (interval[:, 1] < -COST_FLOOR)] = -1
    return dict(delta=delta, delta_interval=interval, robust_sign=sign,
                cost_supported=valid, baseline_cost=costs[0].copy())


def _road_reference(scene, poses):
    """Row-contiguous visible arc; no bridging missing support, no GT heading."""
    y = (63 - np.arange(83, -1, -1)) / Y_SCALE
    centers = scene.center_x[::-1]
    valid = np.isfinite(centers)
    segment = np.cumsum(valid & ~np.r_[False, valid[:-1]])
    length = np.zeros(84)
    length[1:] = np.where(valid[1:] & valid[:-1],
                          np.hypot(np.diff(centers), np.diff(y)), 0.)
    arc = np.cumsum(length)
    fractional = np.clip(poses[..., 1] * Y_SCALE + 20, 0, 83)
    lo = np.clip(np.floor(fractional).astype(int), 0, 82)
    hi = lo + 1
    f = fractional - lo
    inside = (poses[..., 1] >= y[0]) & (poses[..., 1] <= y[-1])
    supported = inside & valid[lo] & valid[hi]
    supported &= segment[lo] == segment[lo[..., :1]]
    center = centers[lo] * (1 - f) + centers[hi] * f
    slope = (centers[hi] - centers[lo]) * Y_SCALE
    heading = -np.arctan(slope)
    lateral = (poses[..., 0] - center) / np.sqrt(1 + slope * slope)
    angle = (poses[..., 2] - heading + np.pi) % (2 * np.pi) - np.pi
    coordinate = arc[lo] * (1 - f) + arc[hi] * f
    complete = supported.all(axis=-1)
    progress = np.where(complete, coordinate[..., -1] - coordinate[..., 0], np.nan)
    return complete, progress, lateral, angle


def _clearances(scene, poses, padding):
    """Conservative raster footprint containment and metric rectangle SAT gaps.

    Cell-radius padding includes every raster cell touched by the oriented hull/
    wheel envelope. Per-interval sweep padding covers linear inter-tick poses;
    this does not bound unmodeled physical trajectories or perception failures.
    Positive SAT gaps are separating-axis gaps, not exact Euclidean distances.
    """
    batch, steps, _ = poses.shape
    known = np.ones((batch, steps), bool)
    road_gap = np.full((batch, steps), np.inf)
    obstacle_gap = np.full((batch, steps), np.inf)
    distance = cv2.distanceTransform(scene.road.astype(np.uint8), cv2.DIST_L2, 0) / Y_SCALE
    outside = cv2.distanceTransform((~scene.road).astype(np.uint8), cv2.DIST_L2, 0) / X_SCALE
    signed = np.where(scene.road, distance - _CELL_RADIUS, -outside)
    for tick in range(steps):
        pose = poses[:, tick]
        c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
        width = HALF_WIDTH + padding[:, tick, 0]
        length = .5 * (FRONT + REAR) + padding[:, tick, 1]
        px = pose[:, 0] - .5 * (FRONT - REAR) * s
        py = pose[:, 1] + .5 * (FRONT - REAR) * c
        extent_x = width * np.abs(c) + length * np.abs(s)
        extent_y = width * np.abs(s) + length * np.abs(c)
        in_view = ((px - extent_x >= (1 - 42) / X_SCALE)
                   & (px + extent_x <= (82 - 42) / X_SCALE)
                   & (py - extent_y >= (63 - 72) / Y_SCALE)
                   & (py + extent_y <= (63 - 8) / Y_SCALE))
        # Crop only the evaluation domain, not the footprint. Include the exact
        # cell-radius-expanded rectangle and an extra raster pixel for bounding
        # arithmetic roundoff; slice the original grids to retain their bits.
        raster_x = (width + _CELL_RADIUS) * np.abs(c) + (length + _CELL_RADIUS) * np.abs(s)
        raster_y = (width + _CELL_RADIUS) * np.abs(s) + (length + _CELL_RADIUS) * np.abs(c)
        col0 = np.clip(np.floor(42 + X_SCALE * (px - raster_x)) - 1, 0, 84).astype(int)
        col1 = np.clip(np.ceil(42 + X_SCALE * (px + raster_x)) + 2, 0, 84).astype(int)
        row0 = np.clip(np.floor(63 - Y_SCALE * (py + raster_y)) - 1, 0, 84).astype(int)
        row1 = np.clip(np.ceil(63 - Y_SCALE * (py - raster_y)) + 2, 0, 84).astype(int)
        known[:, tick] = in_view
        for index in range(batch):
            crop = np.s_[row0[index]:row1[index], col0[index]:col1[index]]
            dx, dy = _X[crop] - px[index], _Y[crop] - py[index]
            u = dx * c[index] + dy * s[index]
            v = -dx * s[index] + dy * c[index]
            occupied = ((np.abs(u) <= width[index] + _CELL_RADIUS)
                        & (np.abs(v) <= length[index] + _CELL_RADIUS))
            known[index, tick] &= not np.any(occupied & ~scene.known[crop])
            if occupied.size:
                road_gap[index, tick] = np.min(np.where(occupied, signed[crop], np.inf))
        road_gap[~in_view, tick] = -np.inf
        for xlo, ylo, xhi, yhi in scene.boxes:
            bx, by = .5 * (xlo + xhi) - px, .5 * (ylo + yhi) - py
            hx, hy = .5 * (xhi - xlo), .5 * (yhi - ylo)
            gap = np.maximum.reduce((
                np.abs(bx) - hx - width * np.abs(c) - length * np.abs(s),
                np.abs(by) - hy - width * np.abs(s) - length * np.abs(c),
                np.abs(bx * c + by * s) - width - hx * np.abs(c) - hy * np.abs(s),
                np.abs(-bx * s + by * c) - length - hx * np.abs(s) - hy * np.abs(c)))
            obstacle_gap[:, tick] = np.minimum(obstacle_gap[:, tick], gap)
    return known, road_gap, obstacle_gap


def score_trajectories(scene: ObservedScene, poses, controls):
    """Apply the identical frozen cost to a batch of supplied raw-tick paths.

    poses[B,17,3] includes initial zero and16 .02s endpoints (right,forward,
    unwrapped physical CCW yaw); controls[B,16,3] are official action triples. Runtime uses
    predicted paths. An offline evaluator may score recorded paths against the
    SAME pre-action image scene, without substituting GT road/progress/heading.
    No input is modified. Unknown footprint/reference evidence yields NaN costs,
    while individual available progress/lateral/heading diagnostics are retained.
    """
    poses, controls = np.asarray(poses, dtype=float), np.asarray(controls, dtype=float)
    if (poses.ndim != 3 or poses.shape[1:] != (17, 3) or len(poses) < 1
            or controls.shape != (len(poses), 16, 3) or not np.isfinite(poses).all()
            or not np.isfinite(controls).all() or np.any(poses[:, 0] != 0)):
        raise ValueError("expected finite zero-origin poses(B,17,3) and controls(B,16,3)")
    if np.any(np.abs(controls[..., 0]) > 1) or np.any((controls[..., 1:] < 0) | (controls[..., 1:] > 1)):
        raise ValueError("controls outside official action bounds")
    reference_supported, progress, lateral, heading = _road_reference(scene, poses)
    # Enclose every linear inter-tick sweep in its midpoint BODY axes, rather
    # than treating forward travel as an equally large lateral uncertainty.
    swept = np.concatenate((poses[:, :1], .5 * (poses[:, 1:] + poses[:, :-1])), axis=1)
    displacement = np.diff(poses[..., :2], axis=1)
    yaw_change = np.abs(np.diff(poses[..., 2], axis=1))
    c, s = np.cos(swept[:, 1:, 2]), np.sin(swept[:, 1:, 2])
    local_displacement = np.stack((displacement[..., 0] * c + displacement[..., 1] * s,
                                  -displacement[..., 0] * s + displacement[..., 1] * c), axis=-1)
    sweep_margin = np.concatenate((np.zeros((len(poses), 1, 2)),
        .5 * (np.abs(local_displacement) + CORNER_RADIUS * yaw_change[..., None])), axis=1)
    known, road_gap, obstacle_gap = _clearances(scene, swept, sweep_margin)
    capped_road, capped_obstacle = np.clip(road_gap, -8., 8.), np.clip(obstacle_gap, -8., 8.)
    components = dict(
        progress=progress,
        lateral_squared=np.mean(lateral[:, 1:] ** 2, axis=1),
        heading_squared=np.mean(heading[:, 1:] ** 2, axis=1),
        road_risk=np.mean(np.maximum(0, 2 - capped_road[:, 1:]) ** 2, axis=1),
        obstacle_risk=np.mean(np.maximum(0, 2 - capped_obstacle[:, 1:]) ** 2, axis=1),
        road_clearance_loss=np.maximum(0, capped_road[:, 0] - capped_road[:, 1:].min(axis=1)),
        obstacle_clearance_loss=np.maximum(0, capped_obstacle[:, 0] - capped_obstacle[:, 1:].min(axis=1)),
        effort=RAW_DT * np.sum(controls ** 2, axis=(1, 2)),
    )
    for name in ("road_risk", "obstacle_risk", "road_clearance_loss", "obstacle_clearance_loss"):
        components[name][~known.all(axis=1)] = np.nan
    costs = np.sum([COST_WEIGHTS[key] * value for key, value in components.items()], axis=0)
    costs[~reference_supported] = np.nan
    return dict(costs=costs, components=components, cost_supported=np.isfinite(costs),
                reference_supported=reference_supported,
                known=known, road_clearance=np.where(known, road_gap, np.nan),
                obstacle_clearance=np.where(known, obstacle_gap, np.nan),
                swept=swept, sweep_margin=sweep_margin)


def compare_candidates(scene: ObservedScene, hypotheses: PhysicsState, actions, *,
                       observer_valid: bool, absolute_position_uncertainty=None,
                       absolute_yaw_uncertainty=None):
    """Compare baseline + one/two preregistered local triples, without selecting.

    hypotheses is observer.shared_hypotheses(): scenario0 is the central state.
    It is tiled verbatim across actions, never independently sampled. All16 raw
    ticks use source coefficients1/1/1. First4 ticks apply the row action; the
    final12 apply SAME CURRENT actions[0], not a queried future nominal policy.

    Explicit absolute residual envelopes cover errors NOT already represented by
    the hypotheses, including unrepresented joint-state combinations, common-mode
    model and image-registration errors over.32s. Do not count a sigma twice.
    Missing envelopes force a veto. Finite basis extrema alone are not a bound.
    Supplied envelopes have engineering floors.5world/.05rad, in addition to
    raster/sweep padding. These floors are NOT calibrated safety bounds; the
    caller must validate absolute coverage separately from pairwise cost signs.
    confidence is a structural support indicator, NOT a probability of safety.
    """
    actions = np.asarray(actions, dtype=float)
    if (actions.ndim != 2 or actions.shape not in ((2, 3), (3, 3))
            or not np.isfinite(actions).all()):
        raise ValueError("actions must be baseline plus1/2 finite local triples")
    if (np.any(np.abs(actions[:, 0]) > 1) or np.any((actions[:, 1:] < 0) | (actions[:, 1:] > 1))
            or np.any(np.abs(actions[1:] - actions[0]) > .1 + 1e-12)):
        raise ValueError("actions exceed bounds or local .1/channel neighborhood")
    if not isinstance(hypotheses, PhysicsState):
        raise TypeError("hypotheses must be the shared observer PhysicsState batch")
    vf = np.atleast_1d(np.asarray(hypotheses.forward_speed, dtype=float))
    if vf.ndim != 1 or not 1 <= len(vf) <= 19:
        raise ValueError("expected1..19 deterministic shared hypotheses")
    count, scenarios = len(actions), len(vf)
    tiled = {}
    for field in fields(hypotheses):
        value = getattr(hypotheses, field.name)
        if value is None:
            raise ValueError("comparison requires explicit observed/inferred wheel omega")
        shape = (scenarios, 4) if field.name in ("wheel_omega", "friction") else (scenarios,)
        value = np.broadcast_to(np.asarray(value, dtype=float), shape)
        tiled[field.name] = np.tile(value, (count, 1)) if len(shape) == 2 else np.tile(value, count)
    controls = np.broadcast_to(actions[0], (count, scenarios, 16, 3)).copy()
    controls[:, :, :4] = actions[:, None, None, :]
    flat_controls = controls.reshape(-1, 16, 3)
    prediction = predict(PhysicsState(**tiled), flat_controls, RAW_DT,
                         parameters=PhysicsParameters(1., 1., 1.))
    poses = np.stack([prediction[key] for key in ("relative_x", "relative_y", "yaw_delta")], axis=-1)
    poses = np.concatenate((np.zeros((count * scenarios, 1, 3)), poses), axis=1)
    scored = score_trajectories(scene, poses, flat_controls)
    raw_known = scored["known"]
    reference_supported, components = scored["reference_supported"], scored["components"]
    raw_supported = raw_known.all(axis=1)
    costs = scored["costs"]
    absolute_supplied = absolute_position_uncertainty is not None and absolute_yaw_uncertainty is not None
    absolute_margin = 0.
    if absolute_supplied:
        uncertainty = np.asarray([absolute_position_uncertainty, absolute_yaw_uncertainty], dtype=float)
        if uncertainty.shape != (2,) or not np.isfinite(uncertainty).all() or np.any(uncertainty < 0):
            raise ValueError("absolute envelopes must be finite nonnegative scalars")
        absolute_margin = max(.5, uncertainty[0]) + CORNER_RADIUS * min(2., max(.05, uncertainty[1]))
    # A one-pixel square in anisotropic metric space needs its DIAGONAL bound.
    absolute_known, absolute_road, absolute_obstacle = _clearances(
        scene, scored["swept"], scored["sweep_margin"] + absolute_margin + 2 * _CELL_RADIUS)
    shape = (count, scenarios)
    absolute_supported = absolute_known.all(axis=1).reshape(shape) & absolute_supplied & bool(observer_valid)
    road_veto = (absolute_road.min(axis=1) < .25).reshape(shape)
    obstacle_veto = (absolute_obstacle.min(axis=1) < .5).reshape(shape)
    veto = ~absolute_supported | road_veto | obstacle_veto
    costs = costs.reshape(shape)
    summary = paired_cost_summary(costs)
    supported = summary["cost_supported"] & absolute_supported.all(axis=1)
    reasons = []
    for action in range(count):
        flags = []
        if not observer_valid:
            flags.append("observer_invalid")
        if not summary["cost_supported"][action]:
            if not reference_supported.reshape(shape)[action].all():
                flags.append("unknown_road_reference")
            if not raw_supported.reshape(shape)[action].all():
                flags.append("unknown_cost_clearance")
        if not absolute_supplied:
            flags.append("absolute_uncertainty_missing")
        if not absolute_known.reshape(count, scenarios, 17)[action].all():
            flags.append("unknown_swept_support")
        if road_veto[action].any():
            flags.append("road_margin")
        if obstacle_veto[action].any():
            flags.append("obstacle_margin")
        if summary["robust_sign"][action] == 0:
            flags.append("tie_or_uncertain_sign")
        reasons.append(tuple(flags))
    return dict(
        **summary, costs=costs, components={key: value.reshape(shape) for key, value in components.items()},
        progress=components["progress"].reshape(shape), poses=poses.reshape(count, scenarios, 17, 3),
        road_clearance=scored["road_clearance"].reshape(count, scenarios, 17),
        obstacle_clearance=scored["obstacle_clearance"].reshape(count, scenarios, 17),
        absolute_road_clearance=np.where(absolute_known & absolute_supplied, absolute_road, np.nan).reshape(count, scenarios, 17),
        absolute_obstacle_clearance=np.where(absolute_known & absolute_supplied, absolute_obstacle, np.nan).reshape(count, scenarios, 17),
        absolute_supported=absolute_supported, supported=supported, veto=veto,
        confidence=np.where(supported, raw_known.reshape(count, scenarios, 17).mean(axis=(1, 2)), 0.),
        abstain=(~supported | veto.any(axis=1) | (summary["robust_sign"] == 0)),
        abstain_reasons=tuple(reasons),
    )
