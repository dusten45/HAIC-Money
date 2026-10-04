"""NumPy-only batch equivalent of the frozen uniform-asphalt tire predictor.

Axis zero contains independent supplied states. Commands are physical joint
targets, gas and brake; the legal steering sign conversion belongs to callers.
The legal observer always supplies wheel_velocity_override=None. Non-None
overrides are deliberately unsupported. No map, labels or simulator are read.
"""
import numpy as np


BATCH_DT = .02
BATCH_ARRAY_SHAPES = {'velocity': (2,), 'local_center': (2,), 'wheel_offsets': (4, 2),
                      'joint': (4,), 'omega': (4,), 'gas': (4,)}
BATCH_SCALARS = ('angle', 'yaw', 'mass', 'inertia')


def pack_states(states):
    """Copy finite scalar predictor states into row-aligned float64 arrays."""
    states = list(states)
    expected = set(BATCH_ARRAY_SHAPES) | set(BATCH_SCALARS) | {'wheel_velocity_override'}
    if not states or any(set(state) != expected or state['wheel_velocity_override'] is not None
                         for state in states):
        raise ValueError('expected nonempty supplied states without wheel velocity overrides')
    result = {'wheel_velocity_override': None}
    for name in BATCH_SCALARS:
        result[name] = np.asarray([float(state[name]) for state in states], dtype=float)
    for name, shape in BATCH_ARRAY_SHAPES.items():
        values = [np.asarray(state[name], dtype=float) for state in states]
        if any(value.shape != shape for value in values):
            raise ValueError('misaligned scalar model arrays')
        result[name] = np.stack(values)
    return _copy_batch(result)


def _copy_batch(state):
    expected = set(BATCH_ARRAY_SHAPES) | set(BATCH_SCALARS) | {'wheel_velocity_override'}
    if set(state) != expected or state['wheel_velocity_override'] is not None:
        raise ValueError('invalid supported batch state keys or wheel override')
    angle = np.asarray(state['angle'], dtype=float)
    if angle.ndim != 1 or not len(angle):
        raise ValueError('expected nonempty batch axis')
    count = len(angle)
    result = {'wheel_velocity_override': None}
    for name in BATCH_SCALARS:
        value = np.asarray(state[name], dtype=float)
        if value.shape != (count,) or not np.isfinite(value).all():
            raise ValueError('invalid batch scalar array')
        result[name] = value.copy()
    if np.any(result['mass'] <= 0.) or np.any(result['inertia'] <= 0.):
        raise ValueError('invalid mechanical constants')
    for name, shape in BATCH_ARRAY_SHAPES.items():
        value = np.asarray(state[name], dtype=float)
        if value.shape != (count, *shape) or not np.isfinite(value).all():
            raise ValueError('invalid batch model array')
        result[name] = value.copy()
    return result


def unpack_state(state, index):
    """Copy one row back to the scalar callback's ordinary state contract."""
    index = int(index)
    count = len(state['angle'])
    if not 0 <= index < count:
        raise ValueError('batch row out of bounds')
    result = {'wheel_velocity_override': None}
    for name in BATCH_SCALARS:
        result[name] = float(state[name][index])
    for name in BATCH_ARRAY_SHAPES:
        result[name] = np.asarray(state[name][index], dtype=float).copy()
    return result


def _rotate_batch(vector, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack((c*vector[..., 0]-s*vector[..., 1],
                     s*vector[..., 0]+c*vector[..., 1]), axis=-1)


def predict_batch(supplied_state, commands):
    """Predict one .02s tick without changing scalar force/update semantics.

    No internal command clipping is introduced. Finite raw inputs preserve the
    scalar predictor's immediate gas decrease and negative-brake branch too;
    legal callers separately enforce their action contract.
    """
    state = _copy_batch(supplied_state)
    # Float64 preserves each normalized float32 action's exact value. Keeping
    # a float32 comparison array would accidentally round the .9 threshold
    # down and treat float32(.9) as locking, unlike the scalar Python branch.
    commands = np.asarray(commands, dtype=float)
    count = len(state['angle'])
    if commands.shape != (count, 3) or not np.isfinite(commands).all():
        raise ValueError('expected one finite physical command per batch row')
    target, requested_gas, brake = commands.T
    state['gas'][:, 2:] += np.minimum(requested_gas[:, None]-state['gas'][:, 2:], .1)
    omega = state['omega']+BATCH_DT*40000.*state['gas']/1.6/(abs(state['omega'])+5.)
    partial = omega-np.sign(omega)*np.minimum(15.*brake[:, None], abs(omega))
    omega = np.where((brake >= .9)[:, None], 0.,
                     np.where((brake > 0.)[:, None], partial, omega))

    # Tire velocities and force frames still use the OLD joint/body state.
    arm = _rotate_batch(state['wheel_offsets'], state['angle'][:, None])
    angular_velocity = state['yaw'][:, None, None]*np.stack((-arm[..., 1], arm[..., 0]), axis=-1)
    wheel_velocity = state['velocity'][:, None, :]+angular_velocity
    wheel_velocity *= (100./np.maximum(100., np.linalg.norm(wheel_velocity, axis=-1)))[..., None]
    local_velocity = _rotate_batch(wheel_velocity, -(state['angle'][:, None]+state['joint']))
    sideways, forward = local_velocity[..., 0], local_velocity[..., 1]
    requested = 82.*np.stack((-sideways, .54*omega-forward), axis=-1)
    ratios = np.linalg.norm(requested, axis=-1)/400.
    forces = requested/np.maximum(1., ratios)[..., None]
    state['omega'] = omega-BATCH_DT*forces[..., 1]*.54/1.6

    body_forces = _rotate_batch(forces, state['joint'])
    world_force = _rotate_batch(body_forces.sum(axis=1), state['angle'])
    offsets = state['wheel_offsets']
    torque = np.sum(offsets[..., 0]*body_forces[..., 1]-offsets[..., 1]*body_forces[..., 0], axis=1)
    state['velocity'] += BATCH_DT*world_force/state['mass'][:, None]
    state['velocity'] *= (100./np.maximum(100., np.linalg.norm(state['velocity'], axis=1)))[:, None]
    state['yaw'] += BATCH_DT*torque/state['inertia']
    state['angle'] += BATCH_DT*state['yaw']
    state['joint'][:, :2] += BATCH_DT*np.clip(50.*(target[:, None]-state['joint'][:, :2]), -3., 3.)
    state['joint'] = np.clip(state['joint'], -.4, .4)
    diagnostics = {'wheel_forward_mps': forward, 'wheel_lateral_mps': sideways,
                   'tire_force_local_N': forces, 'unsaturated_tire_demand_ratio': ratios}
    return state, diagnostics
