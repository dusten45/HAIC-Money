"""Pure reference-tracking actuator/bicycle rollout, not a physical safety proof."""

import math

import numpy as np

from .collision_shield import ACTION_SECONDS, FRONT, HALF_WIDTH, WHEELBASE, Y_SCALE


MAX_ACTIONS = 128


def rollout_trajectories(references, speed, wheel_angle, preview_distance):
    """Return ``(paths[N,K,3], first_commands[N], command_variation[N])``.

    References are finite metric (lateral x, forward y, yaw) paths, starting at
    x=y=0 with strictly increasing y. Actual poses always start at (0,0,0).
    Pure pursuit interpolates a target at actual forward y + preview distance;
    float32 commands are held for .08s, including actuator lag and wheel limits.
    Completed lanes repeat their terminal pose. Work ends after reference arc
    length + two preview distances, or 128 holds: callers must reject incomplete
    forward passage and terminal lateral/yaw mismatch, as well as collisions.
    Variation sums jumps between issued commands, excluding the unknown command
    before this rollout. The caller can add its first-command boundary jump.
    """
    try:
        reference = np.asarray(references, dtype=np.float64)
        speed, wheel_angle, preview_distance = map(
            float, (speed, wheel_angle, preview_distance))
    except (TypeError, ValueError) as error:
        raise ValueError('invalid rollout inputs') from error
    if (reference.ndim != 3 or reference.shape[0] < 1
            or reference.shape[1] < 2 or reference.shape[2] != 3
            or not np.isfinite(reference).all()
            or not np.isfinite([speed, wheel_angle, preview_distance]).all()
            or not 0 < speed < 80 or abs(wheel_angle) > .4
            or preview_distance <= 0):
        raise ValueError('finite forward references and valid speed/wheel/preview required')
    if (not np.allclose(reference[:, 0, :2], 0., atol=1e-8, rtol=0.)
            or np.any(np.diff(reference[:, :, 1], axis=1) <= 0)):
        raise ValueError('references must start at x=y=0 and increase in forward y')

    corner_factor = 1 + math.hypot(FRONT, HALF_WIDTH) * math.tan(.4) / WHEELBASE
    per_action = max(8, math.ceil(speed * corner_factor * ACTION_SECONDS * Y_SCALE / .5))
    dt = ACTION_SECONDS / per_action
    length = np.linalg.norm(np.diff(reference[:, :, :2], axis=1), axis=2).sum(axis=1)
    distance_bound = float(length.max()) + 2 * preview_distance
    if not math.isfinite(distance_bound):
        raise ValueError('reference/preview extent exceeds finite rollout bounds')
    actions = max(1, math.ceil(min(
        MAX_ACTIONS, distance_bound / speed / ACTION_SECONDS)))
    count = len(reference)
    paths = np.zeros((count, actions * per_action + 1, 3), dtype=np.float64)
    wheels = np.full(count, wheel_angle)
    completed = np.zeros(count, dtype=bool)
    variation = np.zeros(count)
    first_commands = np.empty(count, dtype=np.float32)
    previous = np.zeros(count)
    terminal_y = reference[:, -1, 1]

    for action in range(actions):
        start = action * per_action
        pose = paths[:, start]
        active = ~completed
        target_y = np.minimum(pose[:, 1] + preview_distance, terminal_y)
        target_x = np.asarray([np.interp(y, lane[:, 1], lane[:, 0])
                               for y, lane in zip(target_y, reference)])
        dx, dy = target_x - pose[:, 0], target_y - pose[:, 1]
        lateral = dx * np.cos(pose[:, 2]) - dy * np.sin(pose[:, 2])
        commands = np.clip(np.arctan(
            2 * WHEELBASE * lateral / np.maximum(dx * dx + dy * dy, 1e-12)),
            -.4, .4).astype(np.float32)
        if action == 0:
            first_commands = commands.copy()
        else:
            variation += np.where(active, np.abs(
                commands.astype(np.float64) - previous), 0.)
        previous = commands.astype(np.float64)
        wheel_samples = np.empty((count, per_action))
        for sample in range(per_action):
            wheels = np.clip(wheels + np.clip(
                50 * (commands - wheels), -3., 3.) * dt, -.4, .4)
            wheel_samples[:, sample] = wheels
        yaw_steps = speed * np.tan(wheel_samples) / WHEELBASE * dt
        yaw = pose[:, 2, None] + np.cumsum(yaw_steps, axis=1)
        heading = np.column_stack((pose[:, 2], yaw[:, :-1])) + .5 * yaw_steps
        advanced = np.stack((
            pose[:, 0, None] + np.cumsum(speed * np.sin(heading) * dt, axis=1),
            pose[:, 1, None] + np.cumsum(speed * np.cos(heading) * dt, axis=1), yaw), axis=2)
        beyond = (advanced[:, :, 1] >= terminal_y[:, None]) & active[:, None]
        crossed = beyond.any(axis=1)
        # Stop each lane at its first terminal crossing, then repeat that pose.
        lanes = np.flatnonzero(crossed)
        if len(lanes):
            samples = np.argmax(beyond[lanes], axis=1)
            before = np.concatenate((pose[:, None, :], advanced), axis=1)[lanes, samples]
            delta = advanced[lanes, samples] - before
            fraction = (terminal_y[lanes] - before[:, 1]) / delta[:, 1]
            stopped = before + fraction[:, None] * delta
            advanced[lanes] = np.where(
                np.arange(per_action)[None, :, None] >= samples[:, None, None],
                stopped[:, None, :], advanced[lanes])
        paths[:, start + 1:start + per_action + 1] = np.where(
            active[:, None, None], advanced, pose[:, None, :])
        completed |= crossed
        if completed.all():
            paths = paths[:, :start + int(np.argmax(beyond, axis=1).max()) + 2]
            break
    return paths, first_commands, variation
