"""Compose frozen opening acceleration with pixel-only early bend steering."""

from haic_agent.anticipatory_bend_runtime import AnticipatoryBendAgent
from haic_agent.launch_surge_runtime import LaunchSurgeAgent
from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent


class LaunchPreviewAgent:
    def __init__(self, base):
        self.launch = LaunchSurgeAgent(ObstacleFullRoadGuardAgent(base))
        self.launch.corridor = self.launch.base.corridor
        self.preview = AnticipatoryBendAgent(self.launch)

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
    def bend_brake_count(self):
        return 0

    @property
    def boost_count(self):
        return 0
