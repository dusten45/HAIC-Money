"""Causal, action-committed vehicle observer; not a controller or safety bound.

Only the newest permitted grayscale frame and previously *issued* controls are
used. Source-default physics (1/1/1) predicts one actual hold before correction.
HUD speed is a norm, never evidence of forward travel. Image registration measures
the preceding interval, so its innovation is against the predicted interval, not
against instantaneous endpoint velocity. No future image, action or label is read.

State/interval units and signs are those of PhysicsState: forward/right velocity,
physical CCW yaw rate and joint angle, FL/FR/RL/RR wheel omega, rear gas actuator.
Intervals are heuristic uncertainty envelopes, NOT calibrated confidence regions.
They omit unknown terrain, damage, contact impulses and model correlations. The
finite shared basis is not every corner of that box and cannot certify safety.
"""

from copy import deepcopy
from numbers import Real

import numpy as np

from .hud import HudCalibration, decode_hud
from .motion import estimate_body_motion
from .physics import RAW_DT, PhysicsParameters, PhysicsState, predict


# Retained original FIT speed correction; no new observer-label fit.
_CALIBRATION = HudCalibration(
    gain=(1.0095651493835058, 1, 1, 1, 1, 1, 1),
    bias=(0.014490804482105879, 0, 0, 0, 0, 0, 0),
)
_NAMES = ("forward_speed", "lateral_speed", "yaw_rate", "wheel_angle",
          "wheel_omega_0", "wheel_omega_1", "wheel_omega_2", "wheel_omega_3",
          "gas_state")
_NOISE = np.array([1.3, 1.3, .06, .006, 12., 12., 12., 12., 0.])
_LIMITS = np.array([3., 3., .25, .04, 24., 24., 24., 24., .1])


def _state(vector):
    """Pack scalar or batched coordinates, never sharing mutable output arrays."""
    values = np.asarray(vector, dtype=float)
    field = lambda i: float(values[i]) if values.ndim == 1 else values[:, i].copy()
    return PhysicsState(forward_speed=field(0), lateral_speed=field(1),
                        yaw_rate=field(2), wheel_angle=field(3),
                        wheel_omega=values[..., 4:8].copy(), gas_state=field(8))


def _frame(observation):
    image = np.asarray(observation)
    if image.shape == (4, 84, 84):
        image = image[-1]
    if image.shape != (84, 84):
        raise ValueError("observation must be (84,84) or (4,84,84)")
    if image.dtype != np.uint8 and (
        not np.issubdtype(image.dtype, np.floating)
        or not np.isfinite(image).all() or np.any((image < 0) | (image > 1))
    ):
        raise ValueError("observation must be uint8 or finite normalized float")
    return image.copy()


def _duration(value):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError("duration must be a positive multiple of .02 seconds")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("duration must be a positive multiple of .02 seconds")
    ticks = int(round(float(value) / RAW_DT))
    if ticks < 1 or not np.isclose(value, ticks * RAW_DT, rtol=0, atol=1e-12):
        raise ValueError("duration must be a positive multiple of .02 seconds")
    return ticks * RAW_DT


class TemporalObserver:
    """One reset, then alternating commit_action/observe, with pure snapshots.

    ``reset(obs)`` consumes the initial observation (decision index 0); do NOT
    immediately call observe on it again. Select using its snapshot, commit the
    actual issued [steer,gas,brake] exactly once, then observe the next endpoint.
    Commit stores a copy; it does not predict or correct. No speculative candidate
    belongs in this history. Repeated snapshot calls cannot advance the observer.

    ``observe(obs, elapsed_dt=...)`` explicitly represents a shortened final hold;
    omitted duration is the constructor's fixed cadence. Longer/missing holds are
    refused. The caller owns time/issuance truth: pixel equality cannot distinguish
    a legitimately unchanged frame from an erroneously replayed observation.

    Frozen validity rules: >=2 holds; HUD speed 5..80; speed/joint age <=.24s,
    yaw age <=.16s; ground-motion or near-rest witness age <=.8s; half widths no
    greater than [3,3,.25,.04,24,24,24,24,.1]; |joint|<=.48; three clean holds
    after a shock. Near-rest HUD <=2 bounds both velocity components but does
    not prove exact zero or directed travel. No GT contact/heading/damage gate.
    A collision without an observable innovation can remain undetected.

    ``mapping_motion`` is separate from the raw measured ``motion``. It may use
    a source prediction plus a linearly distributed endpoint correction when
    registration abstains, but only between two valid causal states. Its explicit
    provenance and heuristic pose uncertainty must survive downstream mapping;
    it must not be called a measured transform or a calibrated safety margin.
    """

    def __init__(self, decision_dt=.08):
        self.decision_dt = _duration(decision_dt)
        self._value = None
        self._pending = None

    def reset(self, observation) -> dict:
        frame = _frame(observation)
        hud = decode_hud(frame, _CALIBRATION)
        valid = np.asarray(hud["valid"], dtype=bool)
        speed = float(hud["speed"]) if valid[0] else 80.
        value = np.zeros(9)
        value[2:4] = [hud["yaw_rate"] if valid[2] else 0.,
                      hud["wheel_angle"] if valid[1] else 0.]
        value[4:8] = np.where(valid[3:], hud["wheel_omega"], 0.)
        uncertainty = np.array([speed + 1.5, speed + 1.5,
                                .06 if valid[2] else 2.,
                                .006 if valid[1] else .5,
                                *np.where(valid[3:], 12., 60.), 1.])
        self._value, self._uncertainty = value, uncertainty
        self._frame, self._hud = frame, hud
        self._ages = np.where(valid, 0., np.inf)
        self._ground_age = 0. if valid[0] and speed <= 2. else np.inf
        self._pending = None
        self._index = 0
        self._time = self._dt = 0.
        self._cooldown = 0
        self._innovations = {}
        self._motion = {"valid": False, "accepted_for_correction": False,
                        "reason": "no_preceding_interval"}
        self._mapping_motion = {"valid": False, "provenance": "unsupported",
                                "reason": "no_preceding_interval"}
        return self.snapshot()

    def commit_action(self, action):
        """Record only the selected, actually issued action, without advancing."""
        if self._value is None:
            raise RuntimeError("reset must precede commit_action")
        if self._pending is not None:
            raise RuntimeError("action already committed; observe its endpoint first")
        action = np.asarray(action, dtype=float)
        if (action.shape != (3,) or not np.isfinite(action).all()
                or abs(action[0]) > 1 or np.any((action[1:] < 0) | (action[1:] > 1))):
            raise ValueError("action must be [steer(-1..1), gas(0..1), brake(0..1)]")
        self._pending = action.copy()

    def observe(self, observation, *, elapsed_dt=None) -> dict:
        if self._value is None or self._pending is None:
            raise RuntimeError("observe requires exactly one preceding commit_action")
        dt = self.decision_dt if elapsed_dt is None else _duration(elapsed_dt)
        if dt > self.decision_dt + 1e-12:
            raise ValueError("elapsed_dt cannot exceed one committed decision hold")
        frame = _frame(observation)
        hud = decode_hud(frame, _CALIBRATION)
        valid = np.asarray(hud["valid"], dtype=bool)
        motion = estimate_body_motion(self._frame, frame, dt)
        motion["accepted_for_correction"] = False
        previous_valid = self.snapshot()["valid"]
        previous_uncertainty = self._uncertainty.copy()
        prediction = predict(_state(self._value), self._pending[None, None], dt,
                             parameters=PhysicsParameters(1., 1., 1.))
        value = np.array([prediction[name][0, 0] for name in _NAMES[:4]]
                         + prediction["wheel_omega"][0, 0].tolist()
                         + [prediction["gas_state"][0, 0]])
        predicted = value.copy()
        turn = float(prediction["yaw_delta"][0, 0])
        c, s = np.cos(turn), np.sin(turn)
        # Heuristic process allowance, not a propagated probabilistic covariance.
        process = np.array([.6, .5, .2, .008, 5., 5., 8., 8., 0.]) * (dt / .08)
        process[:2] += abs(self._value[2]) * dt * self._uncertainty[1::-1]
        process[1] += abs(self._value[0]) * self._uncertainty[2] * dt
        uncertainty = np.hypot(self._uncertainty, process)
        uncertainty[:2] = np.sqrt(
            c*c * self._uncertainty[:2]**2 + s*s * self._uncertainty[1::-1]**2
            + process[:2]**2)
        gas_bounds = np.clip(self._value[8] + np.array([-1, 1]) * self._uncertainty[8], 0, 1)
        for _ in range(int(round(dt / RAW_DT))):
            gas_bounds += np.minimum(self._pending[1] - gas_bounds, .1)
        uncertainty[8] = float(np.max(np.abs(gas_bounds - value[8])))
        ages = self._ages + dt
        ages[valid] = 0.
        ground_age = self._ground_age + dt
        innovations = {}
        shocks = []

        def correct(index, measurement, noise):
            error = measurement - value[index]
            gain = uncertainty[index]**2 / (uncertainty[index]**2 + noise**2)
            value[index] += gain * error
            # Do not average away fixed raster bias by repeated observations.
            uncertainty[index] = max(noise, np.sqrt(1 - gain) * uncertainty[index])

        for channel, index, limit in ((1, 3, .08), (2, 2, .75),
                                      (3, 4, 35.), (4, 5, 35.),
                                      (5, 6, 35.), (6, 7, 35.)):
            if not valid[channel]:
                continue
            measurement = (float(hud["wheel_angle"]) if channel == 1 else
                           float(hud["yaw_rate"]) if channel == 2 else
                           float(hud["wheel_omega"][channel - 3]))
            innovation = measurement - value[index]
            innovations[_NAMES[index]] = innovation
            if self._index >= 1 and abs(innovation) > limit:
                shocks.append(_NAMES[index])
            correct(index, measurement, _NOISE[index])

        if motion["valid"]:
            # Registration reports a CURRENT-frame interval mean. The model's
            # endpoint-minus-mean offset keeps acceleration causal and explicit.
            right, forward = (float(prediction[key][0, 0])
                              for key in ("relative_x", "relative_y"))
            interval = np.array([(-s * right + c * forward) / dt,
                                 (c * right + s * forward) / dt])
            observed_interval = np.array([motion["forward_speed"], motion["lateral_speed"]])
            candidate = observed_interval + predicted[:2] - interval
            motion_noise = max(.75, float(motion["residual_p90"]) / dt) + .25
            yaw_reference = turn
            if self._hud["valid"][2] and valid[2]:
                yaw_reference = .5 * (float(self._hud["yaw_rate"]) + float(hud["yaw_rate"])) * dt
            yaw_error = float(motion["yaw_delta"] - yaw_reference)
            speed_error = (float(np.linalg.norm(candidate) - hud["speed"])
                           if valid[0] else 0.)
            innovations.update(motion_velocity=(observed_interval - interval).copy(),
                               motion_yaw=yaw_error, motion_speed=speed_error)
            consistent = (np.isfinite(candidate).all()
                          and abs(yaw_error) <= .04 + .25 * dt
                          and (not valid[0] or abs(speed_error) <= max(5., .15 * float(hud["speed"])) + motion_noise))
            if consistent:
                if (np.isfinite(self._ground_age) and self._index >= 1
                        and np.linalg.norm(observed_interval - interval) > max(8., .25 * np.linalg.norm(predicted[:2]))):
                    shocks.append("ground_motion")
                for index in (0, 1):
                    correct(index, candidate[index], motion_noise)
                ground_age = 0.
                motion["accepted_for_correction"] = True
                if not valid[2]:
                    correct(2, float(motion["yaw_rate"]) + predicted[2] - turn / dt,
                            max(.12, abs(yaw_error) / dt))
            else:
                uncertainty[:2] = np.hypot(uncertainty[:2], 1.)

        if valid[0]:
            speed = float(hud["speed"])
            innovation = speed - float(np.linalg.norm(predicted[:2]))
            innovations["speed"] = innovation
            if self._index >= 1 and np.isfinite(self._ground_age) and abs(innovation) > max(4., .12 * speed):
                shocks.append("speed")
            norm = float(np.linalg.norm(value[:2]))
            if np.isfinite(ground_age) and norm > 1e-8:
                direction = value[:2] / norm
                radial = float(np.linalg.norm(direction * uncertainty[:2]))
                gain = radial**2 / (radial**2 + _NOISE[0]**2)
                value[:2] *= (norm + gain * (speed - norm)) / norm
                # Radial evidence cannot shrink transverse/sign uncertainty.
                uncertainty[:2] = np.maximum(_NOISE[:2], np.sqrt(
                    uncertainty[:2]**2 * (1 - gain * direction**2)))
            elif not np.isfinite(ground_age):
                # Preserve both signs until some ground-motion/rest evidence.
                uncertainty[:2] = np.abs(value[:2]) + speed + 1.5
            if speed <= 2.:
                ground_age = 0.
                uncertainty[:2] = np.minimum(uncertainty[:2], np.abs(value[:2]) + speed + 1.5)

        cooldown = max(0, self._cooldown - 1)
        if shocks:
            cooldown = 3
            uncertainty[:4] = np.maximum(uncertainty[:4], [5., 5., .3, .02])
            uncertainty[4:8] = np.maximum(uncertainty[4:8], 24.)
        innovations["shock_channels"] = tuple(shocks)
        if not np.isfinite(value).all() or not np.isfinite(uncertainty).all():
            raise ValueError("nonfinite observer prediction; reset required")
        self._value, self._uncertainty = value, uncertainty
        self._ages, self._ground_age = ages, ground_age
        self._hud, self._frame = hud, frame
        self._motion, self._innovations = motion, innovations
        self._cooldown = cooldown
        self._index += 1
        self._time += dt
        self._dt = dt
        self._pending = None
        result = self.snapshot()
        measured = bool(motion["accepted_for_correction"])
        if measured:
            right, forward, yaw = (float(motion[key]) for key in ("right", "forward", "yaw_delta"))
            position_error = max(.08, float(motion["residual_p90"]))
            yaw_error = .015
            state_error = dict(right=0., forward=0., yaw_delta=0.)
            model_allowance = dict(position=0., yaw=0.)
        else:
            # Distribute the observed endpoint innovation linearly over the
            # *preceding* hold. This is inferred motion, not new image evidence.
            dvf, dvl = value[:2] - predicted[:2]
            right = float(prediction["relative_x"][0, 0] + .5 * dt * (c * dvl - s * dvf))
            forward = float(prediction["relative_y"][0, 0] + .5 * dt * (s * dvl + c * dvf))
            yaw = turn + .5 * dt * (value[2] - predicted[2])
            width = .5 * dt * (previous_uncertainty[:3] + uncertainty[:3])
            rotation_error = np.hypot(right, forward) * width[2]
            state_error = dict(right=float(width[1] + rotation_error),
                               forward=float(width[0] + rotation_error), yaw_delta=float(width[2]))
            model_allowance = dict(position=.05, yaw=.005)
            position_error = max(state_error["right"], state_error["forward"]) + .05
            yaw_error = state_error["yaw_delta"] + .005
        self._mapping_motion = dict(
            valid=bool(not cooldown and (measured or (previous_valid and result["valid"]))),
            right=right, forward=forward, yaw_delta=yaw,
            residual_p90=float(motion["residual_p90"]) if measured else 0.,
            position_uncertainty=position_error, yaw_uncertainty=yaw_error,
            state_uncertainty=state_error, model_allowance=model_allowance,
            provenance="measured_image" if measured else "predicted_corrected_issued_action",
        )
        result["mapping_motion"] = deepcopy(self._mapping_motion)
        return result

    def snapshot(self) -> dict:
        """Return detached endpoint state, intervals, evidence ages and validity."""
        if self._value is None:
            raise RuntimeError("reset must precede snapshot")
        reasons = []
        if self._index < 2:
            reasons.append("startup_history")
        if self._ground_age > .8 + 1e-12:
            reasons.append("ground_motion_stale_or_unknown")
        for channel, limit in ((0, .24), (1, .24), (2, .16)):
            if self._ages[channel] > limit + 1e-12:
                reasons.append(("speed", "wheel_angle", "yaw_rate")[channel] + "_stale")
        speed = float(self._hud["speed"])
        if not self._hud["valid"][0]:
            speed = float(np.linalg.norm(self._value[:2]))
        if not 5. <= speed <= 80.:
            reasons.append("speed_outside_support")
        if abs(self._value[3]) > .48:
            reasons.append("joint_outside_support")
        reasons.extend("uncertain_" + _NAMES[index]
                       for index in np.flatnonzero(self._uncertainty > _LIMITS + 1e-12))
        if self._cooldown:
            reasons.append("recent_innovation_shock")
        lower, upper = self._value - self._uncertainty, self._value + self._uncertainty
        lower[8], upper[8] = max(0., lower[8]), min(1., upper[8])
        uncertainty: dict = {name: float(self._uncertainty[i]) for i, name in enumerate(_NAMES[:4])}
        uncertainty.update(wheel_omega=self._uncertainty[4:8].copy(), gas_state=float(self._uncertainty[8]))
        age = lambda value: float(value) if np.isfinite(value) else None
        return dict(state=_state(self._value), lower=_state(lower), upper=_state(upper),
                    uncertainty=uncertainty, valid=not reasons, invalid_reasons=tuple(reasons),
                    decision_index=self._index, time_seconds=self._time, interval_seconds=self._dt,
                    awaiting_observation=self._pending is not None,
                    measurement_age=dict(speed=age(self._ages[0]), wheel_angle=age(self._ages[1]),
                                         yaw_rate=age(self._ages[2]),
                                         wheel_omega=tuple(age(x) for x in self._ages[3:]),
                                         ground_motion=age(self._ground_age)),
                    hud=deepcopy(self._hud), motion=deepcopy(self._motion),
                    mapping_motion=deepcopy(self._mapping_motion),
                    innovations=deepcopy(self._innovations), shock_cooldown=self._cooldown)

    def shared_hypotheses(self):
        """Return center then -/+ vf,vl,r,joint,omega0..3,gas: (19,) / (19,4).

        Reuse this EXACT initial-state batch for every compared action. This pure,
        deterministic finite coordinate basis does not cover all joint extremes.
        It is available even for invalid diagnostics, but callers must check valid.
        """
        if self._value is None or self._pending is not None:
            raise RuntimeError("hypotheses require an observed, uncommitted decision")
        batch = np.tile(self._value, (19, 1))
        for index in range(9):
            batch[1 + 2 * index, index] -= self._uncertainty[index]
            batch[2 + 2 * index, index] += self._uncertainty[index]
        batch[:, 8] = np.clip(batch[:, 8], 0, 1)
        return _state(batch)
