"""P1 pedal-predictor ablation; no path, steering, target, or limit changes.

Independent public mechanics, initialized solely from public HUD/flow and action
memory. Road grip, no damage, and unsigned RPM are assumptions, not guarantees.
"""
import math
import numpy as np
from agents.apex_2026.candidate.agent import Agent as FrozenP1
from agents.apex_2026.v2.shadow_physics import ShadowCar, decode_wheel_omega


class Agent(FrozenP1):
    def __init__(self, config=None, *, project_root=None):
        super().__init__(config, project_root=project_root)
        self.shadow = ShadowCar()

    def reset(self, observation=None):
        super().reset(observation)
        self._sensor = (0., 0., None)
        self._initial_slip = self._propagated_slip = 0.
        self._flow_valid = False
        self._prediction_calls = 0

    def _motion(self, previous, current):
        result = super()._motion(previous, current)
        self._flow_valid = bool(result[2])
        if self._flow_valid:
            self._initial_slip = float(result[1])
        return result

    def act(self, observation):
        obs = np.asarray(observation)
        self._flow_valid = False
        self._initial_slip = self._propagated_slip
        if obs.shape == (4,84,84) and np.isfinite(obs).all():
            yaw, wheel = self._hud_dynamics(obs[-1])
            self._sensor = (yaw, wheel, decode_wheel_omega(obs[-1]))
        return super().act(observation)

    def _predict_exact(self, speed, steer, gas, brake, throttle):
        yaw, wheel, omega = self._sensor
        slip = self._initial_slip
        self.shadow.reset(speed*math.cos(slip), -yaw, -wheel,
                          lateral_speed=speed*math.sin(slip),
                          throttle=throttle, omegas=omega)
        pred = self.shadow.step((steer,gas,brake),4)
        self._prediction_calls += 1
        return pred[3]

    def _allocate_pedals(self, speed, target, steer, gas_cap, slip):
        initial = self.internal_throttle
        self._prediction_calls = 0
        monotonic = None
        mode = 'exact'
        if speed < 20. or speed > 90. or abs(slip) > math.radians(5):
            gas, brake = super()._allocate_pedals(speed,target,steer,gas_cap,slip)
            mode = 'legacy_outside_rolling'
        else:
            coast = self._predict_exact(speed,steer,0.,0.,initial)
            gas = brake = 0.
            if target-1. <= coast <= target+1.:
                mode = 'coast_deadband'
            else:
                desired = max(0., speed+.08*float(np.clip((target-speed)/.16,
                                    -self.config['braking_accel'],45.)))
                braking = coast > desired
                limit = .4 if braking else float(gas_cap)
                grid = np.linspace(0.,limit,9)
                def predict(p):
                    return self._predict_exact(speed,steer,0. if braking else p,
                                               p if braking else 0.,initial)
                values = np.array([predict(p) for p in grid])
                monotonic = bool(np.all(np.diff(values) <= 1e-4) if braking else
                                 np.all(np.diff(values) >= -1e-4))
                if monotonic:
                    low, high = 0., limit
                    for _ in range(14):
                        middle = (low+high)*.5
                        predicted = predict(middle)
                        if (predicted > desired) if braking else (predicted < desired):
                            low = middle
                        else:
                            high = middle
                    chosen = high
                else:
                    # Tire saturation can invalidate scalar bisection. Match the
                    # same desired next speed on the finite measured pedal grid.
                    chosen = float(grid[np.argmin(np.abs(values-desired))])
                    mode = 'exact_nonmonotonic_grid'
                gas, brake = (0.,chosen) if braking else (chosen,0.)
            for _ in range(4):
                self.internal_throttle += min(gas-self.internal_throttle,.1)
        predicted = self._predict_exact(speed,steer,gas,brake,initial)
        angle = self.shadow.hull.angle
        velocity = self.shadow.hull.linearVelocity
        lateral = math.cos(angle)*velocity.x+math.sin(angle)*velocity.y
        forward = -math.sin(angle)*velocity.x+math.cos(angle)*velocity.y
        self._propagated_slip = math.atan2(lateral,max(forward,.1))
        self.pedal_diagnostics = dict(allocator_mode=mode,
            internal_throttle=float(self.internal_throttle),predicted_next_speed=predicted,
            exact_monotonic=monotonic,exact_prediction_calls=self._prediction_calls,
            exact_slip_source='flow' if self._flow_valid else 'propagated_unknown',
            exact_initial_slip=self._initial_slip,exact_next_slip=self._propagated_slip,
            exact_yaw=float(self._sensor[0]),exact_wheel=float(self._sensor[1]),
            exact_omega=None if self._sensor[2] is None else self._sensor[2].tolist())
        return float(gas),float(brake)
