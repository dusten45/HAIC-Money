"""Pixel-only racing controller using road geometry, speed HUD, and obstacle cues."""

from __future__ import annotations

import numpy as np


class VisionCorridorAgent:
    """Follow the road corridor and regulate speed from rendered pixels only."""

    MAX_STEER = 0.7
    ROAD_LOW = 0.24
    ROAD_HIGH = 0.52
    OBSTACLE_LOW = 0.54
    OBSTACLE_STEER = 0.34
    MAX_TEACHER_GAS = 0.12
    MAX_TEACHER_BRAKE = 0.28
    SPEED_ROI = (77, 83, 10, 13)
    SPEED_BASELINE = 0.27
    SPEED_PER_UNIT = 0.085

    def __init__(
        self,
        *,
        cruise_speed: float = 62.0,
        curve_speed_penalty: float = 2.0,
        max_gas: float = MAX_TEACHER_GAS,
        obstacle_steer_scale: float = 1.0,
        obstacle_far_speed: float = 47.0,
        obstacle_near_speed: float = 40.0,
    ) -> None:
        if not np.isfinite(cruise_speed) or not 36.0 <= cruise_speed <= 80.0:
            raise ValueError("cruise_speed must be finite and between 36 and 80")
        if not np.isfinite(curve_speed_penalty) or not 0.0 <= curve_speed_penalty <= 4.0:
            raise ValueError("curve_speed_penalty must be finite and between 0 and 4")
        if not np.isfinite(max_gas) or not 0.0 < max_gas <= 1.0:
            raise ValueError("max_gas must be finite and in (0, 1]")
        if not np.isfinite(obstacle_steer_scale) or not 0.0 <= obstacle_steer_scale <= 2.0:
            raise ValueError("obstacle_steer_scale must be finite and between 0 and 2")
        for name, value in (
            ("obstacle_far_speed", obstacle_far_speed),
            ("obstacle_near_speed", obstacle_near_speed),
        ):
            if not np.isfinite(value) or not 20.0 <= value <= 80.0:
                raise ValueError(f"{name} must be finite and between 20 and 80")
        self.cruise_speed = float(cruise_speed)
        self.curve_speed_penalty = float(curve_speed_penalty)
        self.max_gas = float(max_gas)
        self.obstacle_steer_scale = float(obstacle_steer_scale)
        self.obstacle_far_speed = float(obstacle_far_speed)
        self.obstacle_near_speed = float(obstacle_near_speed)
        self._obstacle_side = 0.0
        self._obstacle_missing = 0
        self._last_obstacle_side_offset: float | None = None
        self._target_speed: float | None = None
        self._obstacle_encounters = 0
        self._obstacle_detection_frames = 0
        self._speed_samples = 0
        self._speed_sum = 0.0
        self._speed_max = 0.0
        self._gas_frames = 0
        self._brake_frames = 0
        self._last_obstacle: tuple[float, float, float] | None = None
        self._last_speed_estimate = 0.0
        self._last_target_speed = 0.0
        self._last_gas = 0.0
        self._last_brake = 0.0

    def reset(self, observation=None) -> None:
        """Clear route-dependent state between episodes."""
        del observation
        self._obstacle_side = 0.0
        self._obstacle_missing = 0
        self._last_obstacle_side_offset = None
        self._target_speed = None
        self._obstacle_encounters = 0
        self._obstacle_detection_frames = 0
        self._speed_samples = 0
        self._speed_sum = 0.0
        self._speed_max = 0.0
        self._gas_frames = 0
        self._brake_frames = 0
        self._last_obstacle = None
        self._last_speed_estimate = 0.0
        self._last_target_speed = 0.0
        self._last_gas = 0.0
        self._last_brake = 0.0

    @staticmethod
    def _frame(observation) -> np.ndarray | None:
        try:
            pixels = np.asarray(observation, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if pixels.shape == (4, 84, 84):
            frame = pixels[-1]
        elif pixels.shape == (84, 84):
            frame = pixels
        else:
            return None
        if (
            not np.all(np.isfinite(frame))
            or float(frame.min()) < 0.0
            or float(frame.max()) > 1.0
        ):
            return None
        return frame

    def _road_centers(self, frame: np.ndarray) -> dict[int, float]:
        asphalt = (frame >= self.ROAD_LOW) & (frame <= self.ROAD_HIGH)
        horizontal = np.arange(frame.shape[1], dtype=np.float32)
        centers: dict[int, float] = {}
        previous = 42.0
        for row in (54, 50, 46, 42, 38, 34, 30):
            selected = asphalt[row] & (np.abs(horizontal - previous) <= 17.0)
            locations = np.flatnonzero(selected)
            if len(locations) >= 4:
                previous = float(locations.mean())
                centers[row] = previous
        return centers

    @staticmethod
    def _center_at(row: float, centers: dict[int, float]) -> float:
        known_rows = sorted(centers)
        return float(
            np.interp(
                row,
                np.asarray(known_rows, dtype=np.float32),
                np.asarray([centers[y] for y in known_rows], dtype=np.float32),
            )
        )

    def _nearest_bright_object(
        self, frame: np.ndarray, centers: dict[int, float]
    ) -> tuple[float, float, float] | None:
        """Find the nearest small bright component inside the tracked road band."""
        if len(centers) < 3:
            return None
        bright = frame >= self.OBSTACLE_LOW
        bright[:22, :] = False
        bright[62:, :] = False
        visited = np.zeros(bright.shape, dtype=np.bool_)
        candidates: list[tuple[float, float, float]] = []
        height, width = bright.shape
        horizontal = np.arange(width, dtype=np.float32)

        for start_y in range(22, 62):
            for start_x in range(width):
                if not bright[start_y, start_x] or visited[start_y, start_x]:
                    continue
                stack = [(start_x, start_y)]
                visited[start_y, start_x] = True
                min_x = max_x = start_x
                min_y = max_y = start_y
                sum_x = sum_y = area = 0
                while stack:
                    x, y = stack.pop()
                    area += 1
                    sum_x += x
                    sum_y += y
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                    for neighbor_y in range(max(22, y - 1), min(62, y + 2)):
                        for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                            if bright[neighbor_y, neighbor_x] and not visited[neighbor_y, neighbor_x]:
                                visited[neighbor_y, neighbor_x] = True
                                stack.append((neighbor_x, neighbor_y))

                box_width = max_x - min_x + 1
                box_height = max_y - min_y + 1
                if not (4 <= area <= 80 and 2 <= box_width <= 9 and 2 <= box_height <= 10):
                    continue
                center_x = sum_x / area
                center_y = sum_y / area
                road_center = self._center_at(center_y, centers)
                road_row = int(np.clip(round(center_y), 22, 61))
                row_asphalt = (frame[road_row] >= self.ROAD_LOW) & (
                    frame[road_row] <= self.ROAD_HIGH
                )
                road_pixels = np.flatnonzero(
                    row_asphalt
                    & (np.abs(horizontal - road_center) <= 24.0)
                )
                if len(road_pixels) >= 6:
                    inside_visible_road = (
                        float(road_pixels[0]) - 2.0
                        <= center_x
                        <= float(road_pixels[-1]) + 2.0
                    )
                else:
                    inside_visible_road = abs(center_x - road_center) <= 12.0
                if inside_visible_road:
                    candidates.append((center_y, center_x, road_center))

        return max(candidates, key=lambda item: item[0]) if candidates else None

    @classmethod
    def _estimate_speed(cls, frame: np.ndarray) -> float:
        """Decode speed from the lower HUD bar at its calibrated pixel scale."""
        top, bottom, left, right = cls.SPEED_ROI
        mass = float(np.asarray(frame, dtype=np.float32)[top:bottom, left:right].sum())
        return float(np.clip((mass - cls.SPEED_BASELINE) / cls.SPEED_PER_UNIT, 0.0, 80.0))

    def _observation_speed(self, observation, current_frame: np.ndarray) -> float:
        try:
            pixels = np.asarray(observation, dtype=np.float32)
        except (TypeError, ValueError):
            return self._estimate_speed(current_frame)
        frames = pixels[-2:] if pixels.shape == (4, 84, 84) else pixels[np.newaxis, ...]
        estimates = [self._estimate_speed(frame) for frame in frames]
        return float(np.mean(estimates))

    @staticmethod
    def _road_sweep(centers: dict[int, float]) -> float:
        """Estimate upcoming bend strength from the pixel centerline spread."""
        return max(
            (
                abs(centers.get(row, 42.0) - centers.get(row + 24, 42.0))
                for row in (30, 34, 38, 42)
            ),
            default=0.0,
        )

    def _pedals(self, speed: float, target_speed: float) -> tuple[float, float]:
        if speed > target_speed + 1.0:
            brake = float(np.clip((speed - target_speed) * 0.012, 0.04, self.MAX_TEACHER_BRAKE))
            return 0.0, brake
        if speed < target_speed - 8.0:
            return self.max_gas, 0.0
        if speed < target_speed - 3.0:
            return self.max_gas * (2.0 / 3.0), 0.0
        return self.max_gas * (5.0 / 12.0), 0.0

    def act(self, observation) -> np.ndarray:
        frame = self._frame(observation)
        if frame is None:
            return np.zeros(3, dtype=np.float32)

        centers = self._road_centers(frame)
        self._last_obstacle = None
        far = centers.get(42, 42.0)
        near = centers.get(54, 42.0)
        steering = 0.022 * (far - 42.0) + 0.018 * (far - near)
        target_speed = float(
            np.clip(
                self.cruise_speed - self.curve_speed_penalty * self._road_sweep(centers),
                36.0,
                self.cruise_speed,
            )
        )

        obstacle = self._nearest_bright_object(frame, centers)
        if obstacle is not None:
            self._last_obstacle = obstacle
            obstacle_y, obstacle_x, road_center = obstacle
            # The image side of an obstacle can shift with perspective and bends.
            # Re-evaluate to the near field, ignoring sub-pixel side noise.
            if self._obstacle_side == 0.0:
                self._obstacle_encounters += 1
            side_offset = obstacle_x - road_center
            candidate_side = 1.0 if side_offset < 0.0 else -1.0
            if self._obstacle_side == 0.0:
                self._obstacle_side = candidate_side
            elif obstacle_y < 52.0:
                if abs(side_offset) > 1.0:
                    self._obstacle_side = candidate_side
                elif self._last_obstacle_side_offset is not None:
                    offset_motion = side_offset - self._last_obstacle_side_offset
                    if self._obstacle_side > 0.0 and side_offset > -3.0 and offset_motion > 2.0:
                        self._obstacle_side = -1.0
                    elif self._obstacle_side < 0.0 and side_offset < 3.0 and offset_motion < -2.0:
                        self._obstacle_side = 1.0
            self._last_obstacle_side_offset = side_offset
            self._obstacle_detection_frames += 1
            self._obstacle_missing = 0
            urgency = float(np.clip((obstacle_y - 22.0) / 18.0, 0.0, 1.0))
            steering += (
                self._obstacle_side
                * self.OBSTACLE_STEER
                * self.obstacle_steer_scale
                * urgency
            )
            target_speed = min(
                target_speed,
                self.obstacle_far_speed
                if obstacle_y < 44.0
                else self.obstacle_near_speed,
            )
        elif self._obstacle_side != 0.0:
            self._obstacle_missing += 1
            if self._obstacle_missing <= 1:
                target_speed = min(target_speed, 43.0)
            else:
                self._obstacle_side = 0.0
                self._obstacle_missing = 0
                self._last_obstacle_side_offset = None

        if self._target_speed is not None:
            target_speed = 0.65 * self._target_speed + 0.35 * target_speed
        self._target_speed = target_speed

        speed = self._observation_speed(observation, frame)
        self._last_speed_estimate = speed
        self._speed_samples += 1
        self._speed_sum += speed
        self._speed_max = max(self._speed_max, speed)
        gas, brake = self._pedals(speed, target_speed)
        self._last_target_speed = target_speed
        self._last_gas = gas
        self._last_brake = brake
        self._gas_frames += int(gas > 0.0)
        self._brake_frames += int(brake > 0.0)
        return np.asarray(
            [
                np.clip(steering, -self.MAX_STEER, self.MAX_STEER),
                gas,
                brake,
            ],
            dtype=np.float32,
        )

    def diagnostics(self) -> dict[str, float | int]:
        """Expose pixel-controller counters for evaluation without simulator state."""
        return {
            "obstacle_encounters": self._obstacle_encounters,
            "obstacle_detection_frames": self._obstacle_detection_frames,
            "pixel_speed_mean": (
                self._speed_sum / self._speed_samples if self._speed_samples else 0.0
            ),
            "pixel_speed_max": self._speed_max,
            "gas_frames": self._gas_frames,
            "brake_frames": self._brake_frames,
        }

    def last_step_diagnostics(self) -> dict[str, float | None]:
        """Expose the last pixel-based control decision for offline trace analysis."""
        obstacle = self._last_obstacle
        return {
            "target_speed": self._last_target_speed,
            "pixel_speed": self._last_speed_estimate,
            "gas": self._last_gas,
            "brake": self._last_brake,
            "obstacle_y": obstacle[0] if obstacle is not None else None,
            "obstacle_x": obstacle[1] if obstacle is not None else None,
            "road_center_at_obstacle": obstacle[2] if obstacle is not None else None,
            "obstacle_steer_side": self._obstacle_side or None,
        }


__all__ = ["VisionCorridorAgent"]
