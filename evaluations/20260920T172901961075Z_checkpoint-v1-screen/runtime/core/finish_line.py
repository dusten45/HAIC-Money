from __future__ import annotations

import math
from dataclasses import dataclass


LAP_QUALIFICATION_RATIO = 0.95
MIN_FORWARD_CROSSING_SPEED = 1e-4
MAX_LATERAL_TO_FORWARD_RATIO = 4.0


@dataclass
class FinishLineTracker:
    center: tuple[float, float]
    forward: tuple[float, float]
    half_width: float
    half_depth: float
    qualification_ratio: float = LAP_QUALIFICATION_RATIO
    departed_start_area: bool = False
    crossing_from_back: bool = False
    previous_longitudinal: float = 0.0
    previous_lateral: float = 0.0
    qualified_time_s: float | None = None
    candidate_crossing_time_s: float | None = None
    finish_time_s: float | None = None

    def update(
        self,
        position: tuple[float, float],
        velocity: tuple[float, float],
        progress: float,
        physics_time_s: float,
    ) -> bool:
        forward_x, forward_y = self.forward
        lateral_x, lateral_y = forward_y, -forward_x
        relative_x = float(position[0]) - self.center[0]
        relative_y = float(position[1]) - self.center[1]
        longitudinal = relative_x * forward_x + relative_y * forward_y
        lateral = relative_x * lateral_x + relative_y * lateral_y

        qualified = progress >= self.qualification_ratio
        if qualified and self.qualified_time_s is None:
            self.qualified_time_s = float(physics_time_s)

        if self.finish_time_s is not None:
            self.previous_longitudinal = longitudinal
            self.previous_lateral = lateral
            return False

        inside_width = abs(lateral) <= self.half_width
        previous_inside_width = abs(self.previous_lateral) <= self.half_width
        if not self.departed_start_area:
            if abs(longitudinal) > self.half_depth or not inside_width:
                self.departed_start_area = True
            self.previous_longitudinal = longitudinal
            self.previous_lateral = lateral
            return False

        entered_from_back = (
            qualified
            and previous_inside_width
            and inside_width
            and self.previous_longitudinal <= -self.half_depth
            and longitudinal > -self.half_depth
        )
        if not self.crossing_from_back and entered_from_back:
            self.crossing_from_back = True

        if self.crossing_from_back:
            if not inside_width or longitudinal <= -self.half_depth:
                self.crossing_from_back = False
                self.candidate_crossing_time_s = None
            else:
                forward_speed = float(velocity[0]) * forward_x + float(velocity[1]) * forward_y
                lateral_speed = float(velocity[0]) * lateral_x + float(velocity[1]) * lateral_y
                direction_is_valid = (
                    math.isfinite(forward_speed)
                    and math.isfinite(lateral_speed)
                    and forward_speed > MIN_FORWARD_CROSSING_SPEED
                    and abs(lateral_speed) <= forward_speed * MAX_LATERAL_TO_FORWARD_RATIO
                )
                if longitudinal < 0 and forward_speed <= 0:
                    self.candidate_crossing_time_s = None
                elif longitudinal >= 0 and direction_is_valid and self.candidate_crossing_time_s is None:
                    self.candidate_crossing_time_s = float(physics_time_s)

            if self.crossing_from_back and longitudinal >= self.half_depth:
                crossing_time_s = self.candidate_crossing_time_s
                self.crossing_from_back = False
                self.candidate_crossing_time_s = None
                if qualified and crossing_time_s is not None:
                    self.finish_time_s = crossing_time_s
                    self.previous_longitudinal = longitudinal
                    self.previous_lateral = lateral
                    return True

        self.previous_longitudinal = longitudinal
        self.previous_lateral = lateral
        return False
