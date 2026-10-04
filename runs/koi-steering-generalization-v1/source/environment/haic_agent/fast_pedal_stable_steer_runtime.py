"""Use a faster learned pedal command while retaining stable learned steering."""

import numpy as np

from haic_agent.pixel_features import current_frame, estimate_observation_speed, road_centers


class FastPedalStableSteerAgent:
    def __init__(self, fast, stable):
        self.fast = fast
        self.stable = stable
        self.fast_pedal_decisions = 0
        self.stable_decisions = 0

    def reset(self, observation):
        self.fast.reset(observation)
        self.stable.reset(observation)
        self.fast_pedal_decisions = 0
        self.stable_decisions = 0

    def act(self, observation):
        frame = current_frame(observation)
        centers = road_centers(frame)
        speed = estimate_observation_speed(observation, frame)
        fast_action = np.asarray(self.fast.act(observation), dtype=np.float32)
        stable_action = np.asarray(self.stable.act(observation), dtype=np.float32)
        near_offset = abs(centers[54] - 42.0) if 54 in centers else float("inf")
        unsafe = (len(centers) != 7 or near_offset > 8.0
                  or abs(float(stable_action[0])) > 0.25 or speed > 55.0
                  or float(stable_action[2]) > 0.0)
        if unsafe:
            self.stable_decisions += 1
            return stable_action.copy()
        self.fast_pedal_decisions += 1
        action = stable_action.copy()
        action[1:] = fast_action[1:]
        return action
