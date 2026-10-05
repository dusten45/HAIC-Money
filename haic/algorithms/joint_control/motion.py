"""Causal body-relative motion from two permitted grayscale observations."""

import math

import cv2
import numpy as np


PIXEL_SCALE = np.array([1.3608, -1.701], dtype=np.float64)
ORIGIN = np.array([42.0, 63.0], dtype=np.float64)


def _image(frame):
    frame = np.asarray(frame)
    if frame.shape != (84, 84) or not np.isfinite(frame).all():
        raise ValueError('expected a finite84x84 grayscale frame')
    if frame.dtype == np.uint8:
        return frame
    if frame.min() < 0 or frame.max() > 1:
        raise ValueError('floating frames must be normalized to[0,1]')
    return np.rint(frame * 255).astype(np.uint8)


def estimate_body_motion(previous, current, dt=0.08):
    """Estimate the preceding interval, never the next interval.

    Displacement is expressed in the PREVIOUS hull frame (right, forward).
    Velocity is the interval displacement rotated to the CURRENT hull frame.
    Positive yaw is physical counterclockwise rotation, not action-positive steer.
    Invalid observations return zero placeholders with valid=False, not known rest.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    old, new = _image(previous), _image(current)
    result = dict(valid=False, right=0.0, forward=0.0, yaw_delta=0.0,
                  lateral_speed=0.0, forward_speed=0.0, yaw_rate=0.0,
                  tracks=0, inliers=0, inlier_fraction=0.0, residual_p90=0.0)
    mask = np.zeros((84, 84), np.uint8)
    mask[6:72, 3:81] = 255
    mask[55:72, 33:51] = 0
    points = cv2.goodFeaturesToTrack(old, 100, 0.01, 3, mask=mask, blockSize=3)
    if points is None or len(points) < 6:
        return result
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        old, new, points, points.copy(), winSize=(15, 15), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if moved is None:
        return result
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
        new, old, moved, moved.copy(), winSize=(15, 15), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if backward is None:
        return result
    a, b = points[:, 0].astype(float), moved[:, 0].astype(float)
    rounded = np.rint(np.nan_to_num(b, nan=-100.0)).astype(int)
    inside = ((rounded[:, 0] >= 0) & (rounded[:, 0] < 84)
              & (rounded[:, 1] >= 0) & (rounded[:, 1] < 84))
    supported = np.zeros(len(a), dtype=bool)
    supported[inside] = mask[rounded[inside, 1], rounded[inside, 0]] != 0
    keep = ((status[:, 0] != 0) & (back_status[:, 0] != 0) & supported
            & np.isfinite(b).all(axis=1)
            & (np.linalg.norm(np.asarray(backward[:, 0], dtype=float)
                              - np.asarray(points[:, 0], dtype=float), axis=1) <= 0.8))
    a, b = (a[keep] - ORIGIN) / PIXEL_SCALE, (b[keep] - ORIGIN) / PIXEL_SCALE
    result['tracks'] = len(a)
    if len(a) < 6:
        return result
    transform, inliers = cv2.estimateAffinePartial2D(
        a, b, method=cv2.RANSAC, ransacReprojThreshold=0.6,
        maxIters=500, confidence=0.99, refineIters=10)
    if transform is None or inliers is None or not np.isfinite(transform).all():
        return result
    selected = inliers[:, 0].astype(bool)
    result.update(inliers=int(selected.sum()), inlier_fraction=float(selected.mean()))
    scale = float(np.linalg.norm(transform[:, 0]))
    if selected.sum() < 6 or selected.mean() < 0.6 or not 0.92 <= scale <= 1.08:
        return result
    # Camera has fixed zoom. Refit a rigid transform rather than interpreting
    # feature-fit scale noise as longitudinal acceleration.
    aa, bb = a[selected], b[selected]
    ac, bc = aa.mean(axis=0), bb.mean(axis=0)
    u, _, vt = np.linalg.svd((aa - ac).T @ (bb - bc))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) <= 0:
        return result
    translation = bc - rotation @ ac
    residual = np.linalg.norm(aa @ rotation.T + translation - bb, axis=1)
    result['residual_p90'] = float(np.quantile(residual, 0.9))
    yaw = -math.atan2(rotation[1, 0], rotation[0, 0])
    if result['residual_p90'] > 0.6 or abs(yaw) > 0.8:
        return result
    displacement = -rotation.T @ translation
    current_velocity = -translation / dt
    result.update(valid=True, right=float(displacement[0]), forward=float(displacement[1]),
                  yaw_delta=yaw, lateral_speed=float(current_velocity[0]),
                  forward_speed=float(current_velocity[1]), yaw_rate=yaw / dt)
    return result


def estimate_motion_sequence(frames, dt=0.08):
    """Return frame-aligned causal estimates; the first frame has no history."""
    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.shape[1:] != (84, 84) or len(frames) < 1:
        raise ValueError('expected nonempty(N,84,84) frames')
    first = estimate_body_motion(np.zeros((84, 84), np.uint8), frames[0], dt)
    rows = [first] + [estimate_body_motion(a, b, dt) for a, b in zip(frames[:-1], frames[1:])]
    return {key: np.asarray([row[key] for row in rows]) for key in first}
