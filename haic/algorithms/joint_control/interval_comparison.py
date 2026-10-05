"""Bounded road references, without turning unknown pixels into free space.

Cost weights, dynamics and H4 continuation are inherited unchanged. Boundary
brackets are geometric intervals; the five shared reference realizations are a
finite stress set, NOT an exhaustive nonlinear interval or safety certificate.
"""

from dataclasses import dataclass, replace

import numpy as np

from . import comparison as old


@dataclass(frozen=True)
class IntervalScene(old.ObservedScene):
    center_bounds: np.ndarray
    left_bounds: np.ndarray
    right_bounds: np.ndarray
    uncertain_boundary: np.ndarray
    unobserved: np.ndarray


def extract_scene(frames, *, motions=None):
    original = old.extract_scene(frames, motions=motions)
    current = np.asarray(frames)
    if current.ndim == 3:
        current = current[-1]
    _, _, _, objects = old._evidence(old._pixels(current))
    centers = np.full((84, 2), np.nan)
    left, right = centers.copy(), centers.copy()
    boundary = np.zeros((84, 84), bool)
    previous = None
    for row in range(72, 7, -1):
        mask = original.road[row] | objects[row]
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        spans = []
        for a, b in zip(edges[::2], edges[1::2]):
            if a <= 1 or b >= 83 or original.road[row, a:b].sum() < 4:
                continue
            # Search only through ambiguous pixels to an OBSERVED non-road
            # bracket. Another road run or the image edge is not a boundary.
            lo, hi = a - 1, b
            while lo > 1 and not original.known[row, lo]:
                lo -= 1
            while hi < 82 and not original.known[row, hi]:
                hi += 1
            if (not original.known[row, lo] or not original.known[row, hi]
                    or mask[lo] or mask[hi]):
                continue
            lb, rb = np.array([lo + .5, a - .5]), np.array([b - .5, hi - .5])
            cb = .5 * (lb + rb)
            spans.append((a, b, lo, hi, lb, rb, cb))
        if not spans:
            previous = None
            continue
        midpoints = np.array([item[-1].mean() for item in spans])
        if len(spans) > 1:
            if previous is None:
                continue
            order = np.argsort(np.abs(midpoints - previous))
            if abs(midpoints[order[1]] - previous) - abs(midpoints[order[0]] - previous) <= 1:
                previous = None
                continue
            chosen = int(order[0])
        else:
            chosen = 0
        a, b, lo, hi, lb, rb, cb = spans[chosen]
        midpoint = midpoints[chosen]
        if previous is not None and abs(midpoint - previous) > 4:
            previous = None
            continue
        left[row], right[row], centers[row] = [(v - 42) / old.X_SCALE for v in (lb, rb, cb)]
        boundary[row, lo + 1:a] = True
        boundary[row, b:hi] = True
        previous = midpoint
    boundary &= ~original.known
    return IntervalScene(original.road, original.known, original.boxes,
                         centers.mean(axis=1), original.history_used,
                         original.motion_provenance, centers, left, right,
                         boundary, ~original.known & ~boundary)


def reference_realizations(scene):
    """Same five reference shapes for every candidate and initial-state scenario."""
    lower, upper = scene.center_bounds.T
    alternate = np.arange(84) % 2 == 0
    return (scene.center_x, lower, upper,
            np.where(alternate, lower, upper), np.where(alternate, upper, lower))


def score_trajectories(scene, poses, controls):
    """Full 17 endpoints for ALL paths; never normalize over surviving support.

    Clearance uses only confirmed road as its conservative lower estimate.
    Boundary ambiguity can support a center reference, never a footprint claim.
    Road-reference stresses reuse the identical footprint calculation.
    """
    poses, controls = np.asarray(poses, dtype=float), np.asarray(controls, dtype=float)
    scored = old.score_trajectories(scene, poses, controls)
    costs, reference_support = [], []
    for centers in reference_realizations(scene):
        valid, progress, lateral, heading = old._road_reference(replace(scene, center_x=centers), poses)
        components = dict(scored["components"], progress=progress,
                          lateral_squared=np.mean(lateral[:, 1:] ** 2, axis=1),
                          heading_squared=np.mean(heading[:, 1:] ** 2, axis=1))
        costs.append(sum(old.COST_WEIGHTS[key] * value for key, value in components.items()))
        reference_support.append(valid)
    costs = np.stack(costs, axis=-1)
    common = bool(np.isfinite(costs).all() and scored["known"].all())
    # Match the frozen reference's row/segment support, but retain per-tick
    # evidence for before/after coverage rather than counting only footprints.
    valid = np.isfinite(scene.center_x[::-1])
    segment = np.cumsum(valid & ~np.r_[False, valid[:-1]])
    row = np.clip(poses[..., 1] * old.Y_SCALE + 20, 0, 83)
    lo = np.clip(np.floor(row).astype(int), 0, 82)
    reference_known = ((poses[..., 1] >= -20 / old.Y_SCALE)
                       & (poses[..., 1] <= 63 / old.Y_SCALE)
                       & valid[lo] & valid[lo + 1]
                       & (segment[lo] == segment[lo[:, :1]]))
    scored.update(reference_costs=costs, common_support=common,
                  reference_realization_support=np.stack(reference_support, axis=-1),
                  comparison_ticks=17,
                  footprint_known_ticks=scored["known"].sum(axis=1),
                  reference_known=reference_known,
                  supported_ticks=(scored["known"] & reference_known).sum(axis=1))
    return scored


def compare_candidates(scene, hypotheses, actions, *, observer_valid,
                       position_residual=None, yaw_residual=None,
                       paired_cost_residual=0.):
    """Pure comparison plus empirical trajectory/error guards; never commits.

    Residual arrays include dynamics and unresolved initialization errors. They
    have 17 entries (initial zero plus 16 raw ticks). No omitted residual becomes
    a zero safety allowance. The finite initial-state/reference scenarios and
    fitted maximum cost-difference residual are EMPIRICAL, not guarantees.
    """
    result = old.compare_candidates(scene, hypotheses, actions, observer_valid=observer_valid)
    poses = result["poses"]
    count, scenarios = poses.shape[:2]
    controls = np.broadcast_to(np.asarray(actions)[0], (count, scenarios, 16, 3)).copy()
    controls[:, :, :4] = np.asarray(actions)[:, None, None]
    scored = score_trajectories(scene, poses.reshape(-1, 17, 3), controls.reshape(-1, 16, 3))
    reference_costs = scored["reference_costs"].reshape(count, scenarios, 5)
    delta = reference_costs - reference_costs[:1]
    supported = np.isfinite(delta).all(axis=(1, 2)) & scored["common_support"]
    interval = np.full((count, 2), np.nan)
    interval[supported, 0] = delta[supported].min(axis=(1, 2))
    interval[supported, 1] = delta[supported].max(axis=(1, 2))
    residual = float(paired_cost_residual)
    if not np.isfinite(residual) or residual < 0:
        raise ValueError("paired cost residual must be finite and nonnegative")
    interval[1:] += np.array([-residual, residual])
    sign = np.zeros(count, dtype=np.int8)
    sign[supported & (interval[:, 0] > old.COST_FLOOR)] = 1
    sign[supported & (interval[:, 1] < -old.COST_FLOOR)] = -1
    supplied = position_residual is not None and yaw_residual is not None
    margin = np.zeros(17)
    if supplied:
        position, yaw = np.asarray(position_residual), np.asarray(yaw_residual)
        if (position.shape != (17,) or yaw.shape != (17,)
                or not np.isfinite(position).all() or not np.isfinite(yaw).all()
                or np.any(position < 0) or np.any(yaw < 0)):
            raise ValueError("trajectory residuals require 17 finite nonnegative entries")
        # A swept interval must use both endpoint errors, not only the later one.
        radius = position + old.CORNER_RADIUS * np.minimum(2., yaw)
        margin = np.r_[radius[0], np.maximum(radius[1:], radius[:-1])]
    known, road, obstacle = old._clearances(
        scene, scored["swept"], scored["sweep_margin"] + margin[None, :, None] + 2 * old._CELL_RADIUS)
    shape = (count, scenarios)
    absolute = known.all(axis=1).reshape(shape) & supplied & bool(observer_valid)
    veto = (~absolute | (road.min(axis=1).reshape(shape) < .25)
            | (obstacle.min(axis=1).reshape(shape) < .5))
    reasons = []
    for i in range(count):
        flags = []
        if not observer_valid:
            flags.append("observer_invalid")
        if not supported[i]:
            flags.append("common_full_horizon_unsupported")
        if not supplied:
            flags.append("trajectory_calibration_missing")
        if not absolute[i].all():
            flags.append("unknown_trajectory_support")
        if (road.reshape(count, scenarios, 17)[i] < .25).any():
            flags.append("road_margin")
        if (obstacle.reshape(count, scenarios, 17)[i] < .5).any():
            flags.append("obstacle_margin")
        if sign[i] == 0:
            flags.append("empirical_order_uncertain")
        reasons.append(tuple(flags))
    result.update(reference_costs=reference_costs, reference_delta=delta,
                  delta_interval=interval, robust_sign=sign, cost_supported=supported,
                  common_support=scored["common_support"], comparison_ticks=17,
                  supported_ticks=scored["supported_ticks"].reshape(shape),
                  footprint_known_ticks=scored["footprint_known_ticks"].reshape(shape),
                  absolute_supported=absolute, veto=veto,
                  supported=supported & absolute.all(axis=1),
                  confidence=np.where(supported & absolute.all(axis=1), 1., 0.),
                  abstain=~supported | veto.any(axis=1) | (sign == 0),
                  abstain_reasons=tuple(reasons),
                  absolute_road_clearance=np.where(known & supplied, road, np.nan).reshape(count, scenarios, 17),
                  absolute_obstacle_clearance=np.where(known & supplied, obstacle, np.nan).reshape(count, scenarios, 17))
    return result
