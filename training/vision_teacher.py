"""Training-only pixel corridor teacher used to seed visual policy learning."""

from __future__ import annotations

import numpy as np


class VisionCorridorAgent:
    """Estimate road center and bright obstacles from grayscale pixels."""

    GAS = 0.005
    MAX_STEER = 0.7
    ROAD_LOW = 0.24
    ROAD_HIGH = 0.52
    OBSTACLE_LOW = 0.54
    OBSTACLE_STEER = 0.18

    def reset(self, observation=None) -> None:
        """Keep the agent lifecycle; this teacher has no episode state."""
        del observation

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
                            if bright[neighbor_y, neighbor_x] and not visited[
                                neighbor_y, neighbor_x
                            ]:
                                visited[neighbor_y, neighbor_x] = True
                                stack.append((neighbor_x, neighbor_y))

                box_width = max_x - min_x + 1
                box_height = max_y - min_y + 1
                if not (4 <= area <= 80 and 2 <= box_width <= 9 and 2 <= box_height <= 10):
                    continue
                center_x = sum_x / area
                center_y = sum_y / area
                road_center = self._center_at(center_y, centers)
                if abs(center_x - road_center) <= 12.0:
                    candidates.append((center_y, center_x, road_center))

        return max(candidates, key=lambda item: item[0]) if candidates else None

    def act(self, observation) -> np.ndarray:
        frame = self._frame(observation)
        if frame is None:
            return np.zeros(3, dtype=np.float32)

        centers = self._road_centers(frame)
        far = centers.get(42, 42.0)
        near = centers.get(54, 42.0)
        steering = 0.018 * (far - 42.0) + 0.014 * (far - near)

        obstacle = self._nearest_bright_object(frame, centers)
        if obstacle is not None:
            obstacle_y, obstacle_x, road_center = obstacle
            side = 1.0 if obstacle_x < road_center else -1.0
            urgency = float(np.clip((obstacle_y - 26.0) / 16.0, 0.0, 1.0))
            steering += side * self.OBSTACLE_STEER * urgency

        return np.asarray(
            [np.clip(steering, -self.MAX_STEER, self.MAX_STEER), self.GAS, 0.0],
            dtype=np.float32,
        )


__all__ = ["VisionCorridorAgent"]
