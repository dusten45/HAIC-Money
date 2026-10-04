"""Optional pixel-only bend preview around a stable driving policy."""

import numpy as np

from haic_agent.pixel_features import current_frame, road_centers


class AnticipatoryBendAgent:
    def __init__(self, base):
        self.base = base
        self.preview_count = 0

    def reset(self, observation):
        self.base.reset(observation)
        self.preview_count = 0

    def act(self, observation):
        action = np.asarray(self.base.act(observation), dtype=np.float32).copy()
        corridor = getattr(self.base, "corridor", None)
        if corridor is not None and corridor.last_step_diagnostics()["obstacle_y"] is not None:
            return action
        centers = road_centers(current_frame(observation))
        if len(centers) != 7:
            return action

        near = centers[54]
        middle = centers[42]
        far = centers[30]
        bend = far - near
        if (abs(near - 42.0) > 3.0 or abs(bend) < 5.0
                or (middle - near) * (far - middle) < 0.0):
            return action

        if abs(float(action[0])) >= 0.05:
            return action
        action[0] = float(np.clip(0.012 * bend, -0.12, 0.12))
        self.preview_count += 1
        return action
