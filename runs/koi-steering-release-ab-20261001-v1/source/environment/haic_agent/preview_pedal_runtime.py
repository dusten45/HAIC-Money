"""Pixel-only preview acceleration and braking; preserve stable steering."""
import numpy as np

from haic_agent.fast_completion_coordination import FastCompletionCoordination
from haic_agent.geometric_passing_runtime import GeometricPassingAgent
from haic_agent.pixel_features import current_frame

MODES = ('control', 'distance', 'time_budget', 'steering_budget', 'response')


class PreviewPedalAgent(FastCompletionCoordination):
    def __init__(self, mechanism='distance'):
        if mechanism not in MODES:
            raise ValueError('unknown preview pedal mechanism')
        self.pedal_mode = mechanism
        super().__init__('preview_row_repair')

    def reset(self, observation=None):
        super().reset(observation)
        self.previous_speed = None
        self.previous_pedal = (0., 0.)
        self.previous_steer = 0.
        self.accel_gain = 45.
        self.brake_gain = 200.

    def act(self, observation):
        action = super().act(observation)
        original = action.copy()
        frame = current_frame(observation)
        speed = max(0., (float(frame[74:83, 10:13].sum())-.27)/.085)
        corridor = GeometricPassingAgent._corridor(self, frame)
        observed_accel = None
        if self.previous_speed is not None:
            observed_accel = (speed-self.previous_speed)/.08
            pg, pb = self.previous_pedal
            # Update only plausible non-impact responses, before today's command.
            if abs(self.previous_steer) < .15 and abs(observed_accel) < 100.:
                if pg > .2 and pb < .001 and observed_accel > 0.:
                    self.accel_gain = .9*self.accel_gain+.1*float(np.clip(observed_accel/pg, 15., 90.))
                elif pb > .08 and pg < .001 and observed_accel < 0.:
                    self.brake_gain = .9*self.brake_gain+.1*float(np.clip(-observed_accel/pb, 80., 350.))
        valid = len(corridor) >= 5 and 42 in self.last['road_centers'] and 54 in self.last['road_centers']
        target = float(self.last['target_speed'])
        phase, limiting, constraints = 'fallback', None, []
        if valid and self.steps > 10:
            y, x = corridor[:, 0], corridor[:, 1]
            # Five-point metric fits avoid differentiating individual pixel steps.
            for i in range(2, len(y)-2):
                z = y[i-2:i+3]-y[i]
                a, b, _ = np.polyfit(z, x[i-2:i+3], 2)
                curvature = abs(2*a)/(1+b*b)**1.5
                cap = float(np.clip(np.sqrt(100./max(curvature, .001)), 32., 85.))
                constraints.append((float(y[i]), cap, 'curve'))
            constraints.append((float(y[-1]), 60., 'horizon'))
            obstacle = self.base._last_obstacle
            if obstacle is not None:
                constraints.append((max(0., (63-obstacle[0])/1.701-3.), 40., 'obstacle'))
            brake_capacity = 40.
            if self.pedal_mode == 'response':
                brake_capacity = float(np.clip(self.brake_gain*.25, 25., 60.))
            delay = .16
            reach = [np.sqrt(v*v+2*brake_capacity*max(0., d-speed*delay-3.)) for d,v,_ in constraints]
            j = int(np.argmin(reach))
            target = min(85., float(reach[j]))
            limiting = constraints[j]
            if self.pedal_mode == 'time_budget':
                # Reserve actual time to decelerate before each visible restriction.
                target = 85.
                limiting = None
                for d, v, kind in constraints:
                    arrival = max(0., d-3.)/max(speed, 5.)
                    braking = max(0., speed-v)/brake_capacity+delay
                    if arrival <= braking+.08 and v < target:
                        target, limiting = v, (d, v, kind)
            elif self.pedal_mode == 'steering_budget':
                turn = abs(np.tan(np.clip(float(action[0]), -.4, .4)))/3.24
                target = min(target, float(np.clip(np.sqrt(100./max(turn, .001)), 32., 85.)))
            error = target-speed
            if self.pedal_mode == 'response':
                desired = float(np.clip(error/.24, -60., 60.))
                if desired > 2.:
                    action[1], action[2] = np.clip(.08+desired/self.accel_gain, 0., 1.), 0.
                    phase = 'accelerate'
                elif desired < -2.:
                    action[1], action[2] = 0., np.clip(-desired/self.brake_gain, 0., .5)
                    phase = 'brake'
                else:
                    action[1], action[2], phase = .08, 0., 'hold'
            elif error < -2.:
                action[1], action[2], phase = 0., min(.5, .04*(-error)), 'brake'
            elif error > 3.:
                action[1], action[2], phase = 1., 0., 'accelerate'
            else:
                action[1], action[2], phase = max(0., .12+.04*error), 0., 'hold'
        if self.pedal_mode == 'control':
            action = original
            phase = 'control'
        # Recovery's braking veto must see the pedals actually issued.
        self.brake_history[-1] = float(action[2])
        self.previous_speed = speed
        self.previous_pedal = (float(action[1]), float(action[2]))
        self.previous_steer = float(action[0])
        self.last.update(pedal_mode=self.pedal_mode, preview_valid=bool(valid),
                         preview_target=target, preview_phase=phase,
                         preview_constraint=limiting, preview_constraints=constraints,
                         hud_speed_unbounded=speed, observed_acceleration=observed_accel,
                         estimated_accel_gain=self.accel_gain, estimated_brake_gain=self.brake_gain,
                         baseline_action=original.tolist(),
                         completion_changed=bool(np.max(np.abs(action-original))>1e-6),
                         final_gas=float(action[1]), final_brake=float(action[2]))
        return action
