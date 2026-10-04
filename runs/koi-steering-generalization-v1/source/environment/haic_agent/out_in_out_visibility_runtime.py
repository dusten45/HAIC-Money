"""Preserve racing-line control while anticipating obstacle image motion."""
import numpy as np
from haic_agent.out_in_out_runtime import OutInOutAgent
from haic_agent.pixel_features import current_frame, road_centers, nearest_bright_object


class OutInOutVisibilityAgent(OutInOutAgent):
    def reset(self, observation=None):
        super().reset(observation)
        self.initial_camera_side = 0.
        self.projection_valid = None
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
                self.projection_valid = None
        else:
            y, x, center = obstacle
            self.flow_missing = 0
            if self.flow_previous is None:
                self.initial_camera_side = 1. if x < 42 else -1.
            if not self.flow_locked:
                self.committed_side = 1. if x < center else -1.
                if self.flow_previous is not None:
                    previous_y, previous_x = self.flow_previous
                    dy = y - previous_y
                    if dy > 0:
                        projected = float(x + (x-previous_x)*(63.-y)/dy)
                        self.projection_valid = 0 <= projected <= 83
                        self.projected_obstacle_x = projected
                        self.committed_side = (1. if projected < 42 else -1.) if self.projection_valid else self.initial_camera_side
                        self.flow_locked = True
            self.flow_previous = (y, x)
        action = super().act(observation)
        extra_acceleration = False
        if self.steps > 10 and self.line_mode == 'line_fast':
            clear = all(y in centers for y in (30,42,54)) and obstacle is None and abs(self.last['road_curvature'])<1 and abs(self.last['near_heading'])<3
            extra_acceleration = bool(clear)
            if not clear:
                sweep=max(centers.values())-min(centers.values()) if len(centers)>1 else 28.
                target=max(38.,60.-.8*sweep)
                if obstacle is not None: target=min(target,44.)
                speed=self.last['pixel_speed']
                action[1]=np.clip(.12+.04*(target-speed),0,.6)
                action[2]=np.clip(.02*(speed-target-2),0,.28)
                self.brake_history[-1]=float(action[2])
                self.last.update(target_speed=target,final_gas=float(action[1]),final_brake=float(action[2]))
        self.last.update(projection_valid=self.projection_valid,extra_acceleration=extra_acceleration,flow_locked=self.flow_locked, projected_obstacle_x=self.projected_obstacle_x)
        return action
