"""Clearance-dependent obstacle speed over the frozen crossing controller.

The speed budget is an experimental image-space heuristic, not a certified
collision-free velocity. Uncertain passages retain the original action.
"""

from importlib import import_module

import cv2
import numpy as np


def commanded_passage_clear(frame, steer, box, side):
    """Reject relaxation when the current-command footprint cannot pass the box."""
    left, top, width, height = box
    if top > 55:
        return False
    curvature = float(np.tan(steer) / 3.24)
    edge = left + width if side > 0 else left - 1
    # Include the entire near extent of tall components, not just rows <= 55.
    for row in range(max(55, top + height - 1), top - 1, -1):
        forward = (63.0 - row) / 1.701
        turn = curvature * forward
        if abs(turn) >= 1.0:
            return False
        lateral = 0.0 if abs(curvature) < 1e-8 else (1.0 - np.sqrt(1.0 - turn**2)) / curvature
        x = int(round(42.0 + 1.3608 * lateral))
        if not 4 <= x <= 79:
            return False
        if row <= top + height and side * (x - edge) < 4:
            return False
        footprint = frame[row - 1:row + 2, x - 3:x + 4]
        if not np.all((footprint >= 0.24) & (footprint <= 0.52)):
            return False
    return True


class AdaptiveAvoidanceAgent:
    def __init__(self, driver=None):
        if driver is None:
            module = import_module("haic_agent.contact_continuity_runtime")
            driver = module.ContactContinuityAgent("crossing_projection")
        self.driver = driver
        self.previous_pass_speed = 44.0
        self.previous_speed_target = None
        self.last = {}

    def reset(self, observation=None):
        self.driver.reset(observation)
        self.previous_pass_speed = 44.0
        self.previous_speed_target = None
        self.last = {}

    def act(self, observation):
        impact_before = self.driver.impact_left
        prior_speeds = tuple(self.driver.speed_history)
        action = self.driver.act(observation).copy()
        original = action.copy()
        info = self.driver.last
        centers = info["road_centers"]
        speed = info["pixel_speed"]
        obj = info["near_object"]
        reason = "no_near_obstacle"
        pass_speed = arrival_cap = target = road_target = passage_width = None

        if self.driver.steps <= 10:
            reason = "launch_prefix"
        elif obj is not None:
            if set(centers) != {30, 34, 38, 42, 46, 50, 54}:
                reason = "missing_road_rows"
            elif (
                impact_before > 0
                or self.driver.impact_left > 0
                or info["impact_proxy_trigger"]
                or info["contact_proxy"]
                or (prior_speeds and max(prior_speeds) - speed > 10)
            ):
                reason = "impact_guard"
            else:
                side = self.driver.base._obstacle_side
                steer = float(action[0])
                road_target = max(38.0, 60.0 - 0.8 * (max(centers.values()) - min(centers.values())))
                if side not in (-1.0, 1.0) or steer * side <= 0 or abs(steer) >= 0.5:
                    reason = "steering_guard"
                elif abs(centers[54] - 42.0) >= 4.0:
                    reason = "alignment_guard"
                elif road_target <= 44.0:
                    reason = "curve_limited"
                else:
                    frame = np.asarray(observation, dtype=np.float32)
                    if frame.ndim == 3:
                        frame = frame[-1]
                    mask = (frame >= 0.54).astype(np.uint8)
                    mask[:22] = 0
                    mask[62:] = 0
                    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                    matching = [
                        i for i in range(1, count)
                        if np.hypot(centroids[i, 0] - obj[1], centroids[i, 1] - obj[0]) < 0.01
                    ]
                    reason = "passage_unobserved"
                    if len(matching) == 1:
                        left, top, width, height, area = map(int, stats[matching[0]])
                        if 4 <= area <= 80 and 2 <= width <= 9 and 2 <= height <= 10 and top > 22 and top + height < 62:
                            edge = left + width if side > 0 else left - 1
                            widths = []
                            for row in range(top - 1, top + height + 1):
                                x = edge
                                free = 0
                                while 0 <= x < 84 and abs(x - obj[2]) <= 17:
                                    if not 0.24 <= frame[row, x] <= 0.52:
                                        break
                                    free += 1
                                    x += int(side)
                                # A clipped image/search edge is not an observed road boundary.
                                if not 0 <= x < 84 or abs(x - obj[2]) > 17:
                                    widths = []
                                    break
                                widths.append(free)
                            if widths:
                                passage_width = min(widths)
                                clearance = float(np.clip((passage_width - 9.0) / 6.0, 0.0, 1.0))
                                steering = float(np.clip((0.5 - abs(steer)) / 0.3, 0.0, 1.0))
                                alignment = float(np.clip(1.0 - abs(centers[54] - 42.0) / 4.0, 0.0, 1.0))
                                pass_speed = 44.0 + 16.0 * clearance * steering * alignment
                                reason = "narrow_passage"
                                route_clear = commanded_passage_clear(frame, steer, (left, top, width, height), side)
                                if pass_speed > 44.0 and not route_clear:
                                    reason = "swept_path_guard"
                                elif pass_speed > 44.0:
                                    pass_speed = min(pass_speed, self.previous_pass_speed + 1.0)
                                    distance = max((63.0 - obj[0]) / 1.701 - 7.0, 0.0)
                                    arrival_cap = min(72.0, float(np.sqrt(pass_speed**2 + 110.0 * distance)))
                                    target = min(road_target, arrival_cap)
                                    previous_target = (
                                        info["target_speed"] if self.previous_speed_target is None
                                        else self.previous_speed_target
                                    )
                                    target = min(target, previous_target + 1.0)
                                    # Replace both obstacle-44 constraints, not just the final brake.
                                    gas = float(np.clip(0.12 + 0.04 * (target - speed), 0.0, 0.6))
                                    brake = float(np.clip(0.02 * (speed - target - 2.0), 0.0, 0.28))
                                    if speed > arrival_cap:
                                        brake = max(brake, float(np.clip(0.04 * (speed - arrival_cap), 0.0, 0.6)))
                                        gas = 0.0
                                    if brake > 0.0:
                                        gas = 0.0
                                    action[1], action[2] = gas, brake
                                    self.previous_pass_speed = pass_speed
                                    self.previous_speed_target = target
                                    reason = "adaptive"

        if reason != "adaptive":
            self.previous_pass_speed = 44.0
            self.previous_speed_target = (
                info["target_speed"] if obj is not None and self.driver.steps > 10 else None
            )
        self.driver.brake_history[-1] = float(action[2])
        self.driver.last.update(
            adaptive_speed_reason=reason,
            adaptive_speed_changed=bool(np.max(np.abs(action - original)) > 1e-6),
            adaptive_pass_speed=pass_speed,
            adaptive_arrival_cap=arrival_cap,
            adaptive_speed_target=target,
            adaptive_road_target=road_target,
            adaptive_passage_width_px=passage_width,
            baseline_pedals=original[1:].tolist(),
            final_gas=float(action[1]),
            final_brake=float(action[2]),
        )
        self.last = dict(self.driver.last)
        return action

    def last_step_diagnostics(self):
        return dict(self.last)
