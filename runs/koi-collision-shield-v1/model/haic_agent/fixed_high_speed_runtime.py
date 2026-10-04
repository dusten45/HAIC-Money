"""Pixel-only fixed-speed steering mechanisms for a TRAIN feasibility study."""
import numpy as np

from haic_agent.corridor_agent import VisionCorridorAgent
from haic_agent.pixel_features import current_frame, road_centers, estimate_observation_speed


ARMS = ("control", "preview", "curvature", "damping", "pursuit")
TARGET_SPEED = 60.0
PREFIX_DECISIONS = 10


def fixed_pedals(speed):
    return (float(np.clip(0.12 + 0.04 * (TARGET_SPEED - speed), 0, 0.6)),
            float(np.clip(0.02 * (speed - TARGET_SPEED - 2), 0, 0.15)))


class FixedHighSpeedAgent:
    def __init__(self, mechanism="control"):
        if mechanism not in ARMS:
            raise ValueError("unknown steering mechanism")
        self.mechanism = mechanism
        self.base = VisionCorridorAgent(cruise_speed=55, curve_speed_penalty=3, max_gas=0.5,
                                        obstacle_far_speed=43, obstacle_near_speed=36)
        self.reset()

    def reset(self, observation=None):
        self.base.reset(observation)
        self.steps = 0
        self.last = {}

    def act(self, observation):
        frame = current_frame(observation)
        centers = road_centers(frame)
        inherited = self.base.act(observation)
        near, middle = centers.get(54, 42.0), centers.get(42, 42.0)
        baseline_road = 0.022 * (middle - 42) + 0.018 * (middle - near)
        correction = 0.0
        available = all(row in centers for row in (30, 42, 54))
        if self.steps >= PREFIX_DECISIONS and available:
            far = centers[30]
            if self.mechanism == "preview":
                correction = 0.018 * (far - middle)
            elif self.mechanism == "curvature":
                correction = 0.018 * (far - 2 * middle + near)
            elif self.mechanism == "damping":
                pixels = np.asarray(observation)
                previous = road_centers(pixels[-2]) if pixels.ndim == 3 else {}
                if 42 in previous:
                    correction = 0.025 * (middle - previous[42])
                else:
                    available = False
            elif self.mechanism == "pursuit":
                # Image-space bearing; no claim of metric camera calibration.
                desired = 0.65 * (2 * 24 * (far - 42) / ((far - 42) ** 2 + 24 ** 2))
                correction = desired - baseline_road
        steer = float(np.clip(float(inherited[0]) + correction, -self.base.MAX_STEER,
                              self.base.MAX_STEER))
        speed = estimate_observation_speed(observation, frame)
        gas, brake = fixed_pedals(speed)
        self.steps += 1
        self.last = dict(target_speed=TARGET_SPEED, pixel_speed=float(speed),
                         inherited_steer=float(inherited[0]), correction=correction,
                         changed=abs(steer - float(inherited[0])) > 1e-6,
                         available=available, road_centers=centers,
                         final_gas=gas, final_brake=brake)
        return np.asarray([steer, gas, brake], dtype=np.float32)

    def last_step_diagnostics(self):
        return dict(self.last)
