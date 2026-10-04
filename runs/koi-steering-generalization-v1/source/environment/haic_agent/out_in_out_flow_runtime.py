"""Preserve racing-line control while anticipating obstacle image motion."""
import numpy as np
from haic_agent.out_in_out_runtime import OutInOutAgent
from haic_agent.pixel_features import current_frame, road_centers, nearest_bright_object


class OutInOutFlowAgent(OutInOutAgent):
    def reset(self, observation=None):
        super().reset(observation)
        self.flow_previous = None
        self.flow_locked = False
        self.flow_missing = 0
        self.projected_obstacle_x = None

    def act(self, observation):
        frame = current_frame(observation)
        centers = road_centers(frame)
        obstacle = nearest_bright_object(frame, centers)
        if obstacle is None:
            self.flow_missing += 1
            if self.flow_missing >= 2:
                self.flow_previous = None
                self.flow_locked = False
                self.projected_obstacle_x = None
        else:
            y, x, center = obstacle
            self.flow_missing = 0
            if not self.flow_locked:
                self.committed_side = 1. if x < center else -1.
                if self.flow_previous is not None:
                    previous_y, previous_x = self.flow_previous
                    dy = y - previous_y
                    if dy > 0:
                        projected = float(np.clip(x + (x-previous_x)*(63.-y)/dy, 0, 83))
                        self.projected_obstacle_x = projected
                        self.committed_side = 1. if projected < 42 else -1.
                        self.flow_locked = True
            self.flow_previous = (y, x)
        action = super().act(observation)
        self.last.update(flow_locked=self.flow_locked, projected_obstacle_x=self.projected_obstacle_x)
        return action
