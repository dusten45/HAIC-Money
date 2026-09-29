"""Pixel-only steering with positive engine torque allocated to available grip."""
import numpy as np
from haic_agent.high_acceleration_path_runtime import HighAccelerationPathAgent, WHEELBASE
from haic_agent.pixel_features import current_frame


class TractionPathAgent(HighAccelerationPathAgent):
    def act(self, observation):
        action = super().act(observation)
        frame = current_frame(observation)
        speed = self.last['unbounded_hud_speed']
        # Decoder is diagnostic only until compared against pre-action evaluator truth.
        rear_omega = float(frame[72:83, 19:24].sum()) / ((44/255)*4.2*.021)
        wheel_right = float(frame[75:81, 42:53].sum()-frame[75:81, 31:42].sum())/(.587*4.2*21)
        lateral_fraction = min(.995, speed*speed*abs(np.tan(np.clip(action[0], -.4, .4)))/WHEELBASE/150.)
        grip_fraction = float(np.sqrt(max(.01, 1.-lateral_fraction*lateral_fraction)))
        if self.path_mode != 'control':
            action[1] = np.clip(.0054*(speed/.54+5.)*grip_fraction, .03, 1.)
            action[2] = 0.
        self.brake_history[-1] = float(action[2])
        self.last.update(traction_gas=float(action[1]), final_gas=float(action[1]), final_brake=float(action[2]),
                         grip_fraction=grip_fraction, hud_rear_omega=rear_omega,
                         hud_wheel_right=wheel_right, completion_changed=self.path_mode != 'control')
        return action
