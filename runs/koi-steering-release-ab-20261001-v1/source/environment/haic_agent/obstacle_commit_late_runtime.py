"""Commit obstacle avoidance side only once the image cue reaches action range."""

from haic_agent.obstacle_commit_runtime import ObstacleCommitAgent


class ObstacleCommitLateAgent(ObstacleCommitAgent):
    def act(self, observation):
        action = super().act(observation)
        obstacle_y = self.corridor.last_step_diagnostics()["obstacle_y"]
        if obstacle_y is not None and obstacle_y < 28.0:
            self._committed_side = None
        return action
