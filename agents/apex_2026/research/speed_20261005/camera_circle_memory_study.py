"""Research-only causal circle transport and exact instantaneous body geometry.

No Agent, episode, map, simulator state or control selection is implemented.
Caller-supplied uncertainty bounds are assumptions, not certified from pixels.
"""
import numpy as np


DT = .08
PX_X, PX_Y = 1.3608, 1.701
RASTER_UNCERTAINTY = .5*np.hypot(1./PX_X, 1./PX_Y)


def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def motion_samples(previous, current, mode):
    """Return speed/rightward yaw with explicit interval alignment.

    endpoint_mean transports previous-frame memory to the current frame.
    past_slope_forecast forecasts current-to-next using past/current samples.
    The two modes therefore use different endpoints in a replay evaluator.
    """
    previous, current = np.asarray(previous, float), np.asarray(current, float)
    if previous.shape != (2,) or current.shape != (2,) or not np.isfinite([previous, current]).all():
        raise ValueError("expected finite [speed, rightward yaw] samples")
    if mode == "previous":
        result = previous
    elif mode == "current":
        result = current
    elif mode == "endpoint_mean":
        result = .5*(previous+current)
    elif mode == "past_slope_forecast":
        result = current+.5*(current-previous)
    else:
        raise ValueError("unknown timing mode")
    return float(max(0., result[0])), float(np.clip(result[1], -8., 8.))


def interval_pose(speed, yaw):
    """SE(2) scalar-speed approximation; lateral slip is not observed here."""
    angle = DT*yaw
    if abs(yaw) < 1e-8:
        translation = np.array([0., DT*speed])
    else:
        translation = speed/yaw*np.array([1.-np.cos(angle), np.sin(angle)])
    return angle, translation


def transform_points(points, rightward_turn, hull_translation):
    points = np.asarray(points, float).reshape(-1, 2)
    translation = np.asarray(hull_translation, float)
    if translation.shape != (2,) or not np.isfinite(points).all() or not np.isfinite(translation).all() or not np.isfinite(rightward_turn):
        raise ValueError("expected finite point transport")
    return (rotation(rightward_turn) @ (points-translation).T).T


def rectangle_clearance(circle, pose, rightward_heading, uncertainty=0., margin=0.):
    """Exact circle-to-rotated rectangle gap, including body interior and edges.

    Vehicle envelope is x +/-1.6 and y[-2.4,2.6] in body coordinates.
    A rightward heading rotates the body by negative mathematical angle.
    """
    circle, pose = np.asarray(circle, float), np.asarray(pose, float)
    if circle.shape != (2,) or pose.shape != (2,) or not np.isfinite([*circle, *pose, rightward_heading, uncertainty, margin]).all() or min(uncertainty, margin) < 0.:
        raise ValueError("expected finite rectangle/circle geometry")
    local = rotation(rightward_heading) @ (circle-pose)
    dx = max(abs(local[0])-1.6, 0.)
    dy = max(-2.4-local[1], 0., local[1]-2.6)
    return float(np.hypot(dx, dy)-1.2-uncertainty-margin)


def path_projections(path, point, tie_tolerance=.1):
    """Keep competing closest branches with their true ordered arc distance."""
    path, point = np.asarray(path, float), np.asarray(point, float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or point.shape != (2,) or not np.isfinite(path).all() or not np.isfinite(point).all():
        raise ValueError("expected finite ordered path and point")
    delta = np.diff(path, axis=0)
    length = np.linalg.norm(delta, axis=1)
    arc = np.r_[0., np.cumsum(length)]
    fraction = np.clip(np.sum((point-path[:-1])*delta, axis=1)/np.maximum(length*length, 1e-12), 0., 1.)
    closest = path[:-1]+fraction[:, None]*delta
    distance = np.linalg.norm(closest-point, axis=1)
    candidates = []
    for index in np.flatnonzero(distance <= float(distance.min())+tie_tolerance):
        if length[index] < 1e-8:
            continue
        along = float(arc[index]+fraction[index]*length[index])
        if any(abs(along-item["arc"]) < .5 for item in candidates):
            continue
        candidates.append({"arc": along, "distance": float(distance[index]),
                           "point": closest[index].tolist(), "segment": int(index)})
    return candidates


class CircleMemory:
    """Union of current/missed circles; ambiguity never erases an old hazard.

    No age-based deletion or control policy is implemented. Overflow explicitly
    invalidates a bounded planner rather than silently dropping occupied space.
    """

    def __init__(self, max_slots=12):
        self.max_slots = int(max_slots)
        if self.max_slots < 1:
            raise ValueError("max_slots must be positive")
        self.reset()

    def reset(self):
        self.slots = []
        self.next_id = 0
        self.ambiguous = self.overflow = False

    def update(self, observed, rightward_turn, translation, motion_uncertainty):
        observed = np.asarray(observed, float).reshape(-1, 2)
        if not np.isfinite(observed).all() or not np.isfinite(motion_uncertainty) or motion_uncertainty < 0.:
            raise ValueError("expected finite observations and nonnegative uncertainty")
        for slot in self.slots:
            slot["center"] = transform_points(slot["center"][None], rightward_turn, translation)[0]
            slot["uncertainty"] += motion_uncertainty
            slot["missed"] += 1
        predicted = np.array([s["center"] for s in self.slots]).reshape(-1, 2)
        if len(predicted) and len(observed):
            distance = np.linalg.norm(predicted[:, None]-observed[None], axis=2)
            gates = np.array([s["uncertainty"] for s in self.slots])[:, None]+RASTER_UNCERTAINTY
            eligible = distance <= gates
        else:
            eligible = np.zeros((len(predicted), len(observed)), dtype=bool)
        self.ambiguous = bool(np.any(eligible.sum(axis=0) > 1) or np.any(eligible.sum(axis=1) > 1))
        matched = set()
        for old, new in zip(*np.nonzero(eligible)):
            if eligible[old].sum() != 1 or eligible[:, new].sum() != 1:
                continue
            self.slots[old]["center"] = observed[new].copy()
            self.slots[old]["uncertainty"] = float(RASTER_UNCERTAINTY)
            self.slots[old]["missed"] = 0
            matched.add(int(new))
        for index, center in enumerate(observed):
            if index not in matched:
                self.slots.append({"id": self.next_id, "center": center.copy(),
                                   "uncertainty": float(RASTER_UNCERTAINTY), "missed": 0})
                self.next_id += 1
        self.overflow = len(self.slots) > self.max_slots
        return self.slots
