"""Moderate straight-line pedal boost using the sprint safety gate."""

import numpy as np

from haic_agent.straight_sprint_runtime import StraightSprintAgent


class StraightBoostAgent(StraightSprintAgent):
    def act(self, observation):
        prior_sprints = self.sprint_count
        action = np.asarray(super().act(observation), dtype=np.float32).copy()
        if self.sprint_count > prior_sprints:
            action[1] = 0.4
        return action
