"""Preserve opening acceleration while limiting early steering at high pixel speed."""

import numpy as np

from haic_agent.anticipatory_bend_runtime import AnticipatoryBendAgent
from haic_agent.launch_surge_runtime import LaunchSurgeAgent
from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.pixel_features import current_frame, estimate_observation_speed, road_centers


class SpeedGatedAnticipatoryBendAgent(AnticipatoryBendAgent):
    MAX_PREVIEW_SPEED = 45.0

    def __init__(self, base):
        super().__init__(base)
        self.suppressed_count = 0

    def reset(self, observation):
        super().reset(observation)
        self.suppressed_count = 0

    def act(self, observation):
        frame = current_frame(observation)
        if estimate_observation_speed(observation, frame) <= self.MAX_PREVIEW_SPEED:
            return super().act(observation)

        action = np.asarray(self.base.act(observation), dtype=np.float32).copy()
        corridor = getattr(self.base, "corridor", None)
        if corridor is not None and corridor.last_step_diagnostics()["obstacle_y"] is not None:
            return action
        centers = road_centers(frame)
        if len(centers) != 7:
            return action
        near = centers[54]
        middle = centers[42]
        far = centers[30]
        bend = far - near
        if (abs(near - 42.0) <= 3.0 and abs(bend) >= 5.0
                and (middle - near) * (far - middle) >= 0.0
                and abs(float(action[0])) < 0.05):
            self.suppressed_count += 1
        return action


class LaunchPreviewSpeedGateAgent:
    def __init__(self, base):
        self.launch = LaunchSurgeAgent(ObstacleFullRoadGuardAgent(base))
        self.launch.corridor = self.launch.base.corridor
        self.preview = SpeedGatedAnticipatoryBendAgent(self.launch)

    def reset(self, observation):
        self.preview.reset(observation)

    def act(self, observation):
        return self.preview.act(observation)

    @property
    def launch_count(self):
        return self.launch.launch_count

    @property
    def exit_reason(self):
        return self.launch.exit_reason

    @property
    def preview_count(self):
        return self.preview.preview_count

    @property
    def suppressed_count(self):
        return self.preview.suppressed_count
