"""One-shot pixel-gated launch acceleration over the selected safe driver."""

import numpy as np

from haic_agent.pixel_features import current_frame, road_centers


class LaunchSurgeAgent:
    """Accelerate on the opening straight, then permanently return to base control."""

    TARGET_GAS = 0.45
    EXIT_SPEED = 34.0
    MAX_LAUNCH_DECISIONS = 18

    def __init__(self, base):
        self.base = base
        self.launch_active = True
        self.decision_count = 0
        self.launch_count = 0
        self.exit_reason = None

    def reset(self, observation):
        self.base.reset(observation)
        self.launch_active = True
        self.decision_count = 0
        self.launch_count = 0
        self.exit_reason = None

    def act(self, observation):
        action = np.asarray(self.base.act(observation), dtype=np.float32).copy()
        if not self.launch_active:
            return action

        self.decision_count += 1
        if self.decision_count > self.MAX_LAUNCH_DECISIONS:
            self._exit("decision_limit")
            return action

        diagnostics = self.base.corridor.last_step_diagnostics()
        if diagnostics["obstacle_y"] is not None:
            self._exit("obstacle")
            return action
        if float(action[2]) > 0.01:
            self._exit("base_brake")
            return action
        if abs(float(action[0])) > 0.08:
            self._exit("base_steer")
            return action
        if float(diagnostics["pixel_speed"]) >= self.EXIT_SPEED:
            self._exit("speed_limit")
            return action

        centers = road_centers(current_frame(observation))
        if len(centers) != 7:
            self._exit("road_visibility")
            return action
        if (abs(centers[30] - centers[54]) > 2.0
                or max(abs(center - 42.0) for center in centers.values()) > 2.5):
            self._exit("road_bend")
            return action

        action[1] = max(float(action[1]), self.TARGET_GAS)
        self.launch_count += 1
        return action

    def _exit(self, reason):
        self.launch_active = False
        self.exit_reason = reason
