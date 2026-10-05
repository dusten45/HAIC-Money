"""Small, uncalibrated NumPy predictor, not a Box2D implementation or controller.

Source: Participants dfb7a2de2178825ca5c5ce20bab01ba67052ba31,
``core/vendor/car_dynamics.py`` and ``car_racing.py``. Actions are reapplied every
0.02 s, including the asymmetric gas ramp and the *per-tick*, not per-second,
brake decrement. Tire forces use the old wheel angle; the motor advances the
joint during the ensuing body update. Forces therefore do not turn the velocity
instantaneously to a kinematic bicycle heading.

The approximation fixes four wheel centers to one planar body, shares the two
front joint angles and rear gas states, and ignores suspension/joint constraint
impulses, motor reaction torque, contacts, and changes in terrain or damage.
Body mass and yaw inertia come from the source fixture areas/second moments,
including the wheels. Inertia is about the hull origin; the small center-of-mass
offset is ignored, including in the interpretation of hull linearVelocity.
Future friction and damage multipliers remain at their initial values. This is
not an official simulator, collision predictor, safety bound, or ranking proof.

All distances/speeds use source world units, angles radians, and time seconds.
Physical yaw and front joint angle are positive counterclockwise. Initial-frame
x is right and y is forward: world velocity is
``(-vf*sin(yaw) + vl*cos(yaw), vf*cos(yaw) + vl*sin(yaw))``.
Positive action steering turns right, hence targets a *negative* joint angle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike


RAW_DT = 0.02
WHEEL_RADIUS = 27.0 * 0.02
WHEEL_INERTIA = 4000.0 * 0.02**2
ENGINE_POWER = 100000000.0 * 0.02**2
TIRE_STIFFNESS = 205000.0 * 0.02**2
FRICTION_LIMIT = 1000000.0 * 0.02**2
WHEEL_X = np.array([-55.0, 55.0, -55.0, 55.0]) * 0.02
WHEEL_Y = np.array([80.0, 80.0, -82.0, -82.0]) * 0.02
# Hull fixtures: mass 7.06, polar moment 18.2603666667 about the origin.
# Each wheel: mass .06048, own polar moment m*(.56**2 + 1.08**2)/12.
BODY_MASS = 7.06 + 4.0 * 0.06048
BODY_YAW_INERTIA = (
    18.260366666666666
    + 0.06048 * (4.0 * 1.1**2 + 2.0 * 1.6**2 + 2.0 * 1.64**2)
    + 4.0 * 0.06048 * (0.56**2 + 1.08**2) / 12.0
)


@dataclass(frozen=True)
class PhysicsState:
    """Initial body state, scalar or batched, without automatic label access.

    ``forward_speed`` is signed body-forward velocity, NOT the HUD speed norm.
    ``lateral_speed`` is body-right velocity; ``yaw_rate`` is physical CCW rate.
    Missing lateral/yaw/joint/gas values assume zero, not measurements. The two
    front joints are represented by one ``wheel_angle`` and the two rear gas
    actuators by one ``gas_state``. A finite initial angle outside [-.4,.4] is
    preserved for the first tire-force calculation, not silently clipped as an
    observation. Thereafter the model projects each motor update to [-.4,.4].
    This accepts measured/estimated limit overshoot but does not model Box2D
    joint slop or certify an out-of-range observation; callers should flag it.

    Wheel order is front-left, front-right, rear-left, rear-right. Missing
    ``wheel_omega`` assumes free rolling at each wheel's initial projected point
    velocity, including yaw and steering. This cannot recover wheel spin/lock.
    Supplied omega includes the preceding tick's tire-reaction update, as in
    the source. Even a hard-braked wheel need not have zero *post-tick* omega.

    ``friction`` is the per-wheel effective terrain factor before damage grip
    (source road default 1, grass default .6). It accepts scalar, (4,), (B,1),
    or (B,4). Other fields accept scalars or (B,), except wheel omega which
    accepts (4,) or (B,4). Multipliers default to 1 (undamaged), stay constant,
    and may be zero for synthetic disabled-effect tests. No future terrain,
    privileged labels, or observation decoding is queried here. The caller must
    keep oracle-initialized validation and runtime-estimated inputs separate.
    """

    forward_speed: ArrayLike
    lateral_speed: ArrayLike = 0.0
    yaw_rate: ArrayLike = 0.0
    wheel_angle: ArrayLike = 0.0
    wheel_omega: ArrayLike | None = None
    gas_state: ArrayLike = 0.0
    friction: ArrayLike = 1.0
    grip_multiplier: ArrayLike = 1.0
    engine_multiplier: ArrayLike = 1.0
    steering_multiplier: ArrayLike = 1.0


@dataclass(frozen=True)
class PhysicsParameters:
    """Three optional positive calibration scalars; defaults are source-derived.

    Nothing is fitted in this module. Changing these requires an explicitly
    declared calibration split; held-out labels must not be used for tuning.
    """

    mass_scale: float = 1.0
    yaw_inertia_scale: float = 1.0
    tire_stiffness_scale: float = 1.0


class PhysicsPrediction(TypedDict):
    speed: np.ndarray
    relative_x: np.ndarray
    relative_y: np.ndarray
    yaw_delta: np.ndarray
    forward_speed: np.ndarray
    lateral_speed: np.ndarray
    yaw_rate: np.ndarray
    wheel_angle: np.ndarray
    gas_state: np.ndarray
    wheel_omega: np.ndarray
    final_state: PhysicsState


def _array(value: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.broadcast_to(np.asarray(value, dtype=np.float64), shape).copy()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must broadcast to {shape}") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def predict(
    state: PhysicsState | Mapping[str, ArrayLike],
    actions: ArrayLike,
    decision_dt: float = 0.08,
    *,
    parameters: PhysicsParameters | None = None,
) -> PhysicsPrediction:
    """Predict end-of-hold states for independent action sequences (B,H,3).

    Actions are [steering, gas, brake], constrained to [-1,1], [0,1], [0,1].
    ``decision_dt`` must be a positive integer multiple of .02; shortened
    terminal holds must be represented explicitly rather than relabeled .08.
    Returned scalar trajectories have shape (B,H), wheel omega (B,H,4); they
    exclude the initial state. ``speed`` is the velocity norm. ``relative_x``,
    ``relative_y`` and unwrapped ``yaw_delta`` are relative to the initial body
    pose, NOT road coordinates. ``final_state`` contains the last propagated
    body/actuator state and constant conditions, suitable for a new call (whose
    displacement frame starts afresh). All calculations are float64, deterministic
    and vectorized across B; neither inputs nor any simulator are mutated.
    """
    if isinstance(state, Mapping):
        state = PhysicsState(**state)
    if not isinstance(state, PhysicsState):
        raise TypeError("state must be PhysicsState or a mapping of its fields")
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 3 or actions.shape[2] != 3 or min(actions.shape[:2]) < 1:
        raise ValueError("actions must have nonempty shape (B,H,3)")
    if not np.isfinite(actions).all():
        raise ValueError("actions must be finite")
    if (np.abs(actions[..., 0]) > 1.0).any() or (
        (actions[..., 1:] < 0.0) | (actions[..., 1:] > 1.0)
    ).any():
        raise ValueError("actions outside steering [-1,1], gas/brake [0,1]")
    if not np.isfinite(decision_dt) or decision_dt <= 0:
        raise ValueError("decision_dt must be a positive multiple of .02")
    raw_steps = int(round(decision_dt / RAW_DT))
    if raw_steps < 1 or not np.isclose(
        decision_dt, raw_steps * RAW_DT, rtol=0.0, atol=1e-12
    ):
        raise ValueError("decision_dt must be a positive multiple of .02")

    parameters = parameters or PhysicsParameters()
    scales = np.asarray(
        [parameters.mass_scale, parameters.yaw_inertia_scale, parameters.tire_stiffness_scale],
        dtype=np.float64,
    )
    if not np.isfinite(scales).all() or (scales <= 0.0).any():
        raise ValueError("calibration scales must be finite and positive")
    mass = BODY_MASS * parameters.mass_scale
    inertia = BODY_YAW_INERTIA * parameters.yaw_inertia_scale
    stiffness = TIRE_STIFFNESS * parameters.tire_stiffness_scale
    batch, horizon, _ = actions.shape
    vf = _array(state.forward_speed, (batch,), "forward_speed")
    vl = _array(state.lateral_speed, (batch,), "lateral_speed")
    rate = _array(state.yaw_rate, (batch,), "yaw_rate")
    angle = _array(state.wheel_angle, (batch,), "wheel_angle")
    gas = _array(state.gas_state, (batch,), "gas_state")
    friction = _array(state.friction, (batch, 4), "friction")
    grip = _array(state.grip_multiplier, (batch,), "grip_multiplier")
    engine = _array(state.engine_multiplier, (batch,), "engine_multiplier")
    steering = _array(state.steering_multiplier, (batch,), "steering_multiplier")
    if ((gas < 0.0) | (gas > 1.0)).any():
        raise ValueError("gas_state must lie within [0,1]")
    if any((value < 0.0).any() for value in (friction, grip, engine, steering)):
        raise ValueError("friction and effect multipliers must be nonnegative")
    force_limit = FRICTION_LIMIT * friction * grip[:, None]
    wheel_angles = np.zeros((batch, 4))
    wheel_angles[:, :2] = angle[:, None]
    if state.wheel_omega is None:
        omega = (
            -(vl[:, None] - rate[:, None] * WHEEL_Y) * np.sin(wheel_angles)
            + (vf[:, None] + rate[:, None] * WHEEL_X) * np.cos(wheel_angles)
        ) / WHEEL_RADIUS
    else:
        omega = _array(state.wheel_omega, (batch, 4), "wheel_omega")

    names = (
        "speed", "relative_x", "relative_y", "yaw_delta", "forward_speed",
        "lateral_speed", "yaw_rate", "wheel_angle", "gas_state",
    )
    result: dict[str, Any] = {name: np.empty((batch, horizon)) for name in names}
    result["wheel_omega"] = np.empty((batch, horizon, 4))
    x, y, yaw = np.zeros((3, batch))
    vx, vy = vl.copy(), vf.copy()
    for hold in range(horizon):
        target = -actions[:, hold, 0]
        gas_command = actions[:, hold, 1]
        brake = actions[:, hold, 2, None]
        for _ in range(raw_steps):
            gas += np.minimum(gas_command - gas, 0.1)
            motor_rate = np.clip(50.0 * (target - angle), -3.0, 3.0) * steering
            wheel_angles[:, :2] = angle[:, None]
            ws, wc = np.sin(wheel_angles), np.cos(wheel_angles)
            point_side = vl[:, None] - rate[:, None] * WHEEL_Y
            point_forward = vf[:, None] + rate[:, None] * WHEEL_X
            wheel_forward = -ws * point_side + wc * point_forward
            wheel_side = wc * point_side + ws * point_forward

            omega[:, 2:] += (
                RAW_DT * ENGINE_POWER / WHEEL_INERTIA
                * engine[:, None] * gas[:, None] / (np.abs(omega[:, 2:]) + 5.0)
            )
            omega -= np.sign(omega) * np.minimum(np.abs(omega), 15.0 * brake)
            omega = np.where(brake >= 0.9, 0.0, omega)
            f_forward = stiffness * (omega * WHEEL_RADIUS - wheel_forward)
            f_side = -stiffness * wheel_side
            force_norm = np.hypot(f_forward, f_side)
            saturation = np.minimum(1.0, force_limit / np.maximum(force_norm, 1e-30))
            f_forward *= saturation
            f_side *= saturation
            omega -= RAW_DT * f_forward * WHEEL_RADIUS / WHEEL_INERTIA

            fx = f_side * wc - f_forward * ws
            fy = f_side * ws + f_forward * wc
            torque = np.sum(WHEEL_X * fy - WHEEL_Y * fx, axis=1)
            fx_total, fy_total = fx.sum(axis=1), fy.sum(axis=1)
            sy, cy = np.sin(yaw), np.cos(yaw)
            # Integrating world velocity preserves slip and avoids spurious
            # speed gain from Euler integration in a rotating body frame.
            vx += RAW_DT * (cy * fx_total - sy * fy_total) / mass
            vy += RAW_DT * (sy * fx_total + cy * fy_total) / mass
            rate += RAW_DT * torque / inertia
            yaw += RAW_DT * rate
            x += RAW_DT * vx
            y += RAW_DT * vy
            sy, cy = np.sin(yaw), np.cos(yaw)
            vf, vl = -sy * vx + cy * vy, cy * vx + sy * vy
            angle = np.clip(angle + RAW_DT * motor_rate, -0.4, 0.4)

        for name, value in zip(
            names, (np.hypot(vf, vl), x, y, yaw, vf, vl, rate, angle, gas)
        ):
            result[name][:, hold] = value
        result["wheel_omega"][:, hold] = omega

    result["final_state"] = PhysicsState(
        forward_speed=vf.copy(), lateral_speed=vl.copy(), yaw_rate=rate.copy(),
        wheel_angle=angle.copy(), wheel_omega=omega.copy(), gas_state=gas.copy(),
        friction=friction, grip_multiplier=grip, engine_multiplier=engine,
        steering_multiplier=steering,
    )
    return cast(PhysicsPrediction, result)
