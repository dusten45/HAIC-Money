"""Pixel-only recovery mechanisms; all retain the original high-speed governor."""
import numpy as np

from haic_agent.fixed_high_speed_runtime import FixedHighSpeedAgent
from haic_agent.pixel_features import current_frame, ROAD_LOW, ROAD_HIGH


MODES = ("partial_rows", "road_memory", "wide_search", "impact_slew")


def extrapolate(centers, target):
    if target in centers:
        return centers[target]
    rows = sorted(centers, key=lambda row: abs(row - target))[:2]
    if len(rows) < 2:
        return None
    a, b = rows
    return float(centers[a] + (target - a) * (centers[b] - centers[a]) / (b - a))


def wide_center(frame, row, anchor):
    indices = np.flatnonzero((frame[row] >= ROAD_LOW) & (frame[row] <= ROAD_HIGH))
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    groups = [g for g in groups if 4 <= len(g) <= 45]
    return float(min(groups, key=lambda g: abs(float(g.mean()) - anchor)).mean()) if groups else None


class FixedHighSpeedRecovery(FixedHighSpeedAgent):
    def __init__(self, mechanism="partial_rows"):
        if mechanism not in MODES:
            raise ValueError("unknown recovery mode")
        self.recovery_mode = mechanism
        super().__init__("damping")

    def reset(self, observation=None):
        super().reset(observation)
        self.remembered_road = 0.0
        self.last_centers = {42: 42.0, 54: 42.0}
        self.missing_steps = 0
        self.speed_history = []
        self.impact_left = 0
        self.previous_steer = 0.0

    def obstacle_term(self):
        obstacle = self.base._last_obstacle
        if obstacle is None:
            return 0.0
        return (self.base._obstacle_side * self.base.OBSTACLE_STEER * self.base.obstacle_steer_scale
                * float(np.clip((obstacle[0] - 22) / 18, 0, 1)))

    def act(self, observation):
        action = super().act(observation)
        original_steer = float(action[0])
        centers = self.last["road_centers"]
        speed = self.last["pixel_speed"]
        complete = 42 in centers and 54 in centers
        self.missing_steps = 0 if complete else self.missing_steps + 1
        trigger = False
        if (self.steps > 10 and self.impact_left == 0 and self.speed_history and
                speed < 50 and max(self.speed_history) - speed > 10):
            self.impact_left = 12
            trigger = True
        if self.steps > 10:
            if self.recovery_mode == "partial_rows" and not complete and len(centers) >= 2:
                near, far = extrapolate(centers, 54), extrapolate(centers, 42)
                action[0] = np.clip(.022 * (far - 42) + .018 * (far - near) + self.obstacle_term(), -.7, .7)
            elif (self.recovery_mode == "road_memory" and 42 not in centers and
                  self.missing_steps <= 30 and abs(self.remembered_road) > .01):
                action[0] = np.sign(self.remembered_road) * np.clip(abs(self.remembered_road), .18, .55)
            elif self.recovery_mode == "wide_search" and not complete:
                frame = current_frame(observation)
                near = wide_center(frame, 54, self.last_centers[54])
                far = wide_center(frame, 42, self.last_centers[42])
                if near is not None and far is not None:
                    action[0] = np.clip(.022 * (far - 42) + .018 * (far - near) + self.obstacle_term(), -.7, .7)
            elif self.recovery_mode == "impact_slew" and self.impact_left > 0:
                action[0] = np.clip(action[0], self.previous_steer - .18, self.previous_steer + .18)
        if complete:
            self.remembered_road = .022 * (centers[42] - 42) + .018 * (centers[42] - centers[54])
            self.last_centers = {42: centers[42], 54: centers[54]}
        if self.impact_left:
            self.impact_left -= 1
        self.speed_history = (self.speed_history + [speed])[-4:]
        self.previous_steer = float(action[0])
        self.last.update(recovery_mode=self.recovery_mode, damping_steer=original_steer,
                         recovery_changed=abs(float(action[0]) - original_steer) > 1e-6,
                         missing_steps=self.missing_steps, remembered_road=self.remembered_road,
                         impact_proxy_trigger=trigger, impact_steps_remaining=self.impact_left)
        return action
