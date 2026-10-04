"""Standalone camera-only experiment with distance-aware curve braking.

The controller dependency definitions below were extracted at build time from
root agent.py SHA256 2b2248f0d8fc2d35eddf20a6b9cca0302b95a428c04a3459b564ec6e2181c988.
Only the selected _ClearRoadRow42DropoutController dependency closure is kept.
Inference imports NumPy and Python standard-library modules, never the project,
environment, model checkpoints, or telemetry. The Agent extension accelerates
clear-road underspeed while retaining inherited steering/obstacle/recovery.
Overlapping metric road windows constrain bend approach speed; this research
candidate has not established the requested pace or development profile.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from collections.abc import Mapping, Sequence
import math
import numpy as np

@dataclass(frozen=True)
class _TemporalObstacleTrack:
    """One obstacle in camera pixels; missed frames carry a predicted box."""

    bbox: tuple[float, float, float, float]
    last_seen_bbox: tuple[float, float, float, float]
    dx_samples: tuple[float, ...]
    dy_samples: tuple[float, ...]
    misses: int
    observations: int
    identity: int

    @property
    def vx(self) -> float:
        return float(np.median(self.dx_samples)) if self.dx_samples else 0.0

    @property
    def vy(self) -> float:
        return float(np.median(self.dy_samples)) if self.dy_samples else 0.0

    @property
    def closing_rate_upper(self) -> float:
        return max((rate for rate in self.dy_samples if rate > 0.0), default=0.0)

    @property
    def uncertainty_px(self) -> float:
        return 1.0 + 2.0 * self.misses

@dataclass(frozen=True)
class _TemporalPassAssessment:
    """Local camera evidence and a necessary steering-slew reachability check."""

    selected_side: int
    left_width_px: float | None
    right_width_px: float | None
    ttc_decisions: float | None
    slew_decisions: int | None
    road_rows: tuple[int, ...]
    brake_required: bool
    selected_target_x: float | None
    selected_path_margin_px: float | None
    left_shift_per_row: float | None
    right_shift_per_row: float | None
    left_approach_clearance_px: float | None
    right_approach_clearance_px: float | None

def _temporal_path_steer(target_x: float, near_y: float) -> float:
    """Match the controller's geometric position and heading command."""
    control_y = max(42.0, near_y)
    near_control_y = min(60.0, max(54.0, control_y + 4.0))

    def approach_x(row: float) -> float:
        return 41.5 + (target_x - 41.5) * (63.0 - row) / (63.0 - near_y)

    control_x = approach_x(control_y)
    near_control_x = approach_x(near_control_y)
    return 0.016 * (control_x - 41.5) + 0.012 * (control_x - near_control_x)

def _track_obstacle_stack(
    previous: _TemporalObstacleTrack | None,
    stack_bboxes: Sequence[tuple[float, float, float, float] | None],
) -> _TemporalObstacleTrack | None:
    """Associate four oldest-to-newest camera boxes without external state.

    A persistent caller supplies its preceding result. Then only the newest
    box advances time; the other three overlap the previous observation.
    On first sighting all four stack frames can establish approach velocity.
    """
    if len(stack_bboxes) != 4:
        raise ValueError("expected four camera-frame boxes")

    def advance(
        track: _TemporalObstacleTrack | None,
        detected: tuple[float, float, float, float] | None,
    ) -> _TemporalObstacleTrack | None:
        if detected is None:
            if track is None or track.misses >= 2:
                return None
            left, top, right, bottom = track.bbox
            return _TemporalObstacleTrack(
                (left + track.vx, top + track.vy,
                 right + track.vx, bottom + track.vy),
                track.last_seen_bbox, track.dx_samples, track.dy_samples,
                track.misses + 1, track.observations, track.identity,
            )

        box = tuple(float(value) for value in detected)
        if len(box) != 4 or not all(np.isfinite(box)) or box[0] > box[2] or box[1] > box[3]:
            raise ValueError("invalid camera obstacle box")
        observed_x = 0.5 * (box[0] + box[2])
        observed_y = 0.5 * (box[1] + box[3])
        if track is not None:
            predicted_x = 0.5 * (track.bbox[0] + track.bbox[2]) + track.vx
            predicted_y = 0.5 * (track.bbox[1] + track.bbox[3]) + track.vy
            residual_x = observed_x - predicted_x
            residual_y = observed_y - predicted_y
            same_obstacle = (
                abs(residual_x) <= 8.0 + 2.0 * track.misses
                and -3.0 - 2.0 * track.misses <= residual_y
                <= 12.0 + 2.0 * track.misses
            )
            if same_obstacle:
                gap = track.misses + 1
                last = track.last_seen_bbox
                last_x = 0.5 * (last[0] + last[2])
                last_y = 0.5 * (last[1] + last[3])
                return _TemporalObstacleTrack(
                    box, box,
                    (track.dx_samples + ((observed_x - last_x) / gap,))[-4:],
                    (track.dy_samples + ((observed_y - last_y) / gap,))[-4:],
                    0, track.observations + 1, track.identity,
                )
        return _TemporalObstacleTrack(
            box, box, (), (), 0, 1,
            track.identity + 1 if track is not None else 1,
        )

    if previous is not None:
        return advance(previous, stack_bboxes[-1])
    track = None
    prior_bbox = None
    for bbox in stack_bboxes:
        if bbox is None:
            track = advance(track, None)
        elif bbox != prior_bbox:
            track = advance(track, bbox)
        prior_bbox = bbox
    return track

def _assess_temporal_pass(
    track: _TemporalObstacleTrack,
    road_edges: Mapping[int, tuple[float, float]],
    *,
    committed_side: int,
    last_steer: float,
) -> _TemporalPassAssessment:
    """Choose only a locally observed pass that has time to slew its steer.

    Camera-row time-to-contact and command slew are necessary conditions,
    not a vehicle-dynamics safety proof. Missing edge rows cannot certify a
    passage, and the caller must retain ordinary road-loss recovery.
    """
    left, top, right, bottom = track.bbox
    center_y = 0.5 * (top + bottom)
    nearest = sorted(
        (row for row in road_edges if abs(row - center_y) <= 13.0),
        key=lambda row: (abs(row - center_y), row),
    )[:3]
    rows = tuple(sorted(nearest)) if (
        len(nearest) == 3 and max(nearest) - min(nearest) == 2
    ) else ()
    rate = track.closing_rate_upper
    ttc = (
        max(0.0, (54.0 - bottom - track.uncertainty_px) / rate)
        if rate > 0.0 else None
    )
    if not rows:
        return _TemporalPassAssessment(
            committed_side, None, None, ttc, None, (), True,
            None, None, None, None, None, None,
        )

    road_left = max(road_edges[row][0] + 2.63 for row in rows)
    road_right = min(road_edges[row][1] - 2.63 for row in rows)
    left_obstacle_bound = left - 3.3
    right_obstacle_bound = right + 3.3
    left_width = left_obstacle_bound - road_left
    right_width = road_right - right_obstacle_bound

    near_y = bottom + 2.0
    approach_rows = range(ceil(near_y), 59)
    obstacle_rows = range(max(22, math.floor(top) - 2), min(58, ceil(near_y)) + 1)
    approach_is_observed = (
        near_y < 63.0
        and len(approach_rows) > 0
        and all(row in road_edges for row in approach_rows)
        and len(obstacle_rows) > 0
        and all(row in road_edges for row in obstacle_rows)
    )

    def candidate(side: int) -> tuple[bool, int, float, float, float | None]:
        width = left_width if side < 0 else right_width
        target = (
            0.5 * (road_left + left_obstacle_bound)
            if side < 0 else 0.5 * (right_obstacle_bound + road_right)
        )
        shift_per_row = (
            abs(target - 41.5) / (63.0 - near_y)
            if near_y < 63.0 else float("inf")
        )
        if near_y >= 63.0:
            return False, 0, target, shift_per_row, None
        path_margin = None
        if approach_is_observed:
            shift_reserve = 0.4 * (63.0 - near_y) - abs(target - 41.5)
            path_margin = shift_reserve
            for row in obstacle_rows:
                edge_left, edge_right = road_edges[row]
                path_margin = min(
                    path_margin,
                    target - edge_left - 2.63,
                    edge_right - 2.63 - target,
                )
            for row in approach_rows:
                x = 41.5 + (target - 41.5) * (63.0 - row) / (63.0 - near_y)
                edge_left, edge_right = road_edges[row]
                path_margin = min(
                    path_margin,
                    x - edge_left - 2.63,
                    edge_right - 2.63 - x,
                )
        desired = float(np.clip(_temporal_path_steer(target, near_y), -0.32, 0.32))
        # The runtime scales urgency from 0.5 to 1.0. Check both endpoints
        # and keep the larger command change as the necessary slew budget.
        slew = ceil(max(
            abs(desired - last_steer), abs(0.5 * desired - last_steer),
        ) / 0.07)
        reachable = (
            width >= 0.75 + 2.0 * track.uncertainty_px
            and path_margin is not None
            and path_margin >= 0.0
            and ttc is not None
            and ttc > slew + 2
        )
        return reachable, slew, target, shift_per_row, path_margin

    left_reachable, left_slew, left_target, left_shift, left_path = candidate(-1)
    right_reachable, right_slew, right_target, right_shift, right_path = candidate(1)
    current_reachable = (
        left_reachable if committed_side == -1 else right_reachable
        if committed_side == 1 else False
    )
    if current_reachable:
        selected = committed_side
    elif left_reachable and right_reachable:
        selected = -1 if left_width >= right_width else 1
    elif left_reachable:
        selected = -1
    elif right_reachable:
        selected = 1
    else:
        selected = committed_side
    selected_reachable = (
        left_reachable if selected == -1 else right_reachable
        if selected == 1 else False
    )
    slew = left_slew if selected == -1 else right_slew if selected == 1 else None
    return _TemporalPassAssessment(
        selected, left_width, right_width, ttc, slew, rows,
        not selected_reachable or track.misses > 0,
        (left_target if selected == -1 else right_target)
        if selected_reachable else None,
        (left_path if selected == -1 else right_path)
        if selected_reachable else None,
        left_shift, right_shift, left_path, right_path,
    )

class _StableCompletionController:
    """Completion-first bare-checkpoint runtime with frozen source lineage.

    The broader racing-line controller below accumulated several interacting
    grayscale hazard heuristics without a matched multi-track evaluation.  The
    smaller controller at source commit ``52976fe`` is the only local runtime
    in this lineage with a reproducible clean finish.  Keep that policy
    explicit here instead of silently treating the neural ``model.pt`` state
    dict as the effective policy.

    This class intentionally reproduces the historical camera-only action
    contract.  DrQ actors, explicit baseline exports, and the HAIC policy path
    do not use it.
    """

    IMAGE_CENTER = 41.5
    ROAD_LOW = 0.24
    ROAD_HIGH = 0.52
    OBSTACLE_LOW = 0.54
    MAX_STEER = 0.48
    STRAIGHT_MAX_STEER = 0.16
    STRAIGHT_CENTER_DEADBAND = 1.25
    STRAIGHT_SWEEP_DEADBAND = 1.5
    MAX_STEER_STEP = 0.07
    MAX_GAS = 0.08
    MAX_BRAKE = 0.28
    OBSTACLE_MISS_LIMIT = 4
    SIDE_SWITCH_ROW = 52.0
    SPEED_ROI = (77, 83, 10, 13)
    SPEED_BASELINE = 0.27
    SPEED_PER_UNIT = 0.085

    def __init__(self, *, cruise_speed: float = 48.0) -> None:
        self.cruise_speed = float(cruise_speed)
        self._obstacle_side = 0.0
        self._obstacle_missing = 0
        self._last_obstacle_side_offset = None
        self._last_steer = 0.0
        self._target_speed = None
        self.road_visible = False
        self.has_seen_road = False

    def reset(self, observation=None) -> None:
        del observation
        self._obstacle_side = 0.0
        self._obstacle_missing = 0
        self._last_obstacle_side_offset = None
        self._last_steer = 0.0
        self._target_speed = None
        self.road_visible = False
        self.has_seen_road = False

    @staticmethod
    def _frame(observation) -> np.ndarray | None:
        try:
            pixels = np.asarray(observation, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            return None
        if pixels.shape != (4, 84, 84):
            return None
        frame = pixels[-1]
        if not np.all(np.isfinite(frame)):
            return None
        if float(frame.min()) < 0.0 or float(frame.max()) > 1.0:
            return None
        return frame

    def _road_centers(self, frame: np.ndarray) -> dict[int, float]:
        asphalt = (frame >= self.ROAD_LOW) & (frame <= self.ROAD_HIGH)
        horizontal = np.arange(frame.shape[1], dtype=np.float32)
        centers: dict[int, float] = {}
        previous = self.IMAGE_CENTER
        for row in (54, 50, 46, 42, 38, 34, 30):
            selected = asphalt[row] & (np.abs(horizontal - previous) <= 17.0)
            locations = np.flatnonzero(selected)
            if len(locations) >= 4:
                previous = float(locations.mean())
                centers[row] = previous
        return centers

    @classmethod
    def _center_at(cls, row: float, centers: dict[int, float]) -> float:
        if not centers:
            return cls.IMAGE_CENTER
        known_rows = sorted(centers)
        return float(
            np.interp(
                row,
                np.asarray(known_rows, dtype=np.float32),
                np.asarray([centers[y] for y in known_rows], dtype=np.float32),
            )
        )

    @classmethod
    def _road_sweep(cls, centers: dict[int, float]) -> float:
        return max(
            (
                abs(
                    centers.get(row, cls.IMAGE_CENTER)
                    - centers.get(row + 24, cls.IMAGE_CENTER)
                )
                for row in (30, 34, 38, 42)
            ),
            default=0.0,
        )

    def _nearest_obstacle(
        self,
        frame: np.ndarray,
        centers: dict[int, float],
        spans: dict[int, tuple[float, float]] | None = None,
    ) -> tuple[float, float, float] | None:
        """Reproduce the compact-component gate from the frozen controller."""
        del spans
        if len(centers) < 3:
            return None
        bright = frame >= self.OBSTACLE_LOW
        bright[:22, :] = False
        bright[62:, :] = False
        visited = np.zeros(bright.shape, dtype=np.bool_)
        candidates: list[tuple[float, float, float]] = []
        width = bright.shape[1]
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
                            if (
                                bright[neighbor_y, neighbor_x]
                                and not visited[neighbor_y, neighbor_x]
                            ):
                                visited[neighbor_y, neighbor_x] = True
                                stack.append((neighbor_x, neighbor_y))
                box_width = max_x - min_x + 1
                box_height = max_y - min_y + 1
                if not (
                    4 <= area <= 80
                    and 2 <= box_width <= 9
                    and 2 <= box_height <= 10
                ):
                    continue
                center_x = sum_x / area
                center_y = sum_y / area
                road_center = self._center_at(center_y, centers)
                if abs(center_x - road_center) <= 24.0:
                    candidates.append((center_y, center_x, road_center))
        return max(candidates, key=lambda item: item[0]) if candidates else None

    @classmethod
    def _estimate_speed(cls, frame: np.ndarray) -> float:
        top, bottom, left, right = cls.SPEED_ROI
        mass = float(frame[top:bottom, left:right].sum())
        return float(
            np.clip(
                (mass - cls.SPEED_BASELINE) / cls.SPEED_PER_UNIT,
                0.0,
                80.0,
            )
        )

    def _pedals(self, speed: float, target_speed: float) -> tuple[float, float]:
        if speed > target_speed + 1.0:
            brake = float(
                np.clip((speed - target_speed) * 0.012, 0.04, self.MAX_BRAKE)
            )
            return 0.0, brake
        if speed < target_speed - 8.0:
            return self.MAX_GAS, 0.0
        if speed < target_speed - 3.0:
            return self.MAX_GAS * (2.0 / 3.0), 0.0
        return self.MAX_GAS * (5.0 / 12.0), 0.0

    def _lost_road_action(self) -> np.ndarray:
        self._last_steer = float(
            self._last_steer
            + np.clip(-self._last_steer, -self.MAX_STEER_STEP, self.MAX_STEER_STEP)
        )
        return np.asarray(
            [self._last_steer, 0.0, min(self.MAX_BRAKE, 0.06)],
            dtype=np.float32,
        )

    def _adjust_obstacle_steering(
        self,
        *,
        base_steering: float,
        obstacle_bias: float,
        straight: bool,
    ) -> float:
        """Extension point whose default preserves the frozen controller."""
        del straight
        return base_steering + obstacle_bias

    def _adjust_obstacle_urgency(
        self,
        *,
        urgency: float,
        straight: bool,
    ) -> float:
        """Extension point whose default preserves the frozen controller."""
        del straight
        return urgency

    def _allow_obstacle_side_switch(
        self,
        *,
        obstacle_y: float,
        obstacle_x: float,
        candidate_side: float,
    ) -> bool:
        """Extension point whose default preserves the existing side latch."""
        del obstacle_x, candidate_side
        return obstacle_y < self.SIDE_SWITCH_ROW

    def _adjust_road_steering(
        self,
        *,
        steering: float,
        straight: bool,
        centers: dict[int, float],
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        """Extension point whose default preserves the frozen controller."""
        del straight, centers, obstacle
        return steering

    def _adjust_target_speed(
        self,
        *,
        target_speed: float,
        curve_target_speed: float,
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        """Extension point whose default preserves the frozen controller."""
        del curve_target_speed, obstacle
        return target_speed

    def _adjust_target_speed_for_steering(
        self,
        *,
        target_speed: float,
        steering: float,
        straight: bool,
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        """Extension point after the final obstacle steering request exists."""
        del steering, straight, obstacle
        return target_speed

    def _adjust_pedals(
        self,
        *,
        gas: float,
        brake: float,
        straight: bool,
        obstacle: tuple[float, float, float] | None,
    ) -> tuple[float, float]:
        """Extension point whose default preserves the frozen controller."""
        del straight, obstacle
        return gas, brake

    def act(self, observation) -> np.ndarray:
        frame = self._frame(observation)
        if frame is None:
            self.road_visible = False
            if self.has_seen_road:
                return self._lost_road_action()
            return np.zeros(3, dtype=np.float32)

        centers = self._road_centers(frame)
        self.road_visible = len(centers) >= 3
        if not self.road_visible:
            if self.has_seen_road:
                return self._lost_road_action()
            return np.zeros(3, dtype=np.float32)
        self.has_seen_road = True

        far = centers.get(42, self.IMAGE_CENTER)
        near = centers.get(54, self.IMAGE_CENTER)
        road_sweep = self._road_sweep(centers)
        center_offset = far - self.IMAGE_CENTER
        straight = (
            abs(center_offset) <= self.STRAIGHT_CENTER_DEADBAND
            and abs(far - near) <= self.STRAIGHT_SWEEP_DEADBAND
            and road_sweep <= self.STRAIGHT_SWEEP_DEADBAND
        )
        if straight:
            steering = 0.0
            steer_limit = self.STRAIGHT_MAX_STEER
        else:
            steering = 0.016 * center_offset + 0.012 * (far - near)
            steer_limit = self.MAX_STEER
        curve_target_speed = float(
            np.clip(self.cruise_speed - 2.0 * road_sweep, 36.0, self.cruise_speed)
        )
        target_speed = curve_target_speed

        obstacle = self._nearest_obstacle(frame, centers)
        steering = self._adjust_road_steering(
            steering=steering,
            straight=straight,
            centers=centers,
            obstacle=obstacle,
        )
        if obstacle is not None:
            obstacle_y, obstacle_x, road_center = obstacle
            steer_limit = min(self.MAX_STEER, 0.32)
            side_offset = obstacle_x - road_center
            candidate_side = 1.0 if side_offset < 0.0 else -1.0
            if self._obstacle_side == 0.0:
                self._obstacle_side = candidate_side
            elif self._allow_obstacle_side_switch(
                obstacle_y=obstacle_y,
                obstacle_x=obstacle_x,
                candidate_side=candidate_side,
            ):
                self._obstacle_side = candidate_side
            self._last_obstacle_side_offset = side_offset
            urgency = float(np.clip((obstacle_y - 22.0) / 18.0, 0.0, 1.0))
            urgency = self._adjust_obstacle_urgency(
                urgency=urgency,
                straight=straight,
            )
            steering = self._adjust_obstacle_steering(
                base_steering=steering,
                obstacle_bias=self._obstacle_side * 0.24 * urgency,
                straight=straight,
            )
            target_speed = min(target_speed, 47.0 if obstacle_y < 44.0 else 40.0)
            self._obstacle_missing = 0
        elif self._obstacle_side != 0.0:
            self._obstacle_missing += 1
            if self._obstacle_missing > self.OBSTACLE_MISS_LIMIT:
                self._obstacle_side = 0.0
                self._obstacle_missing = 0
                self._last_obstacle_side_offset = None

        if self._target_speed is not None:
            target_speed = 0.65 * self._target_speed + 0.35 * target_speed
        self._target_speed = target_speed
        target_speed = self._adjust_target_speed(
            target_speed=target_speed,
            curve_target_speed=curve_target_speed,
            obstacle=obstacle,
        )
        target_speed = self._adjust_target_speed_for_steering(
            target_speed=target_speed,
            steering=steering,
            straight=straight,
            obstacle=obstacle,
        )
        if not straight and abs(steering) > 0.28:
            target_speed = min(target_speed, 44.0)
        speed = self._estimate_speed(frame)
        gas, brake = self._pedals(speed, target_speed)

        if not straight and abs(steering) > 0.28:
            gas = min(gas, self.MAX_GAS * 0.5)
        gas, brake = self._adjust_pedals(
            gas=gas,
            brake=brake,
            straight=straight,
            obstacle=obstacle,
        )
        steering = float(np.clip(steering, -steer_limit, steer_limit))
        if (
            self._last_steer != 0.0
            and steering * self._last_steer < 0.0
            and abs(self._last_steer) > self.MAX_STEER_STEP
        ):
            steering = 0.0
        steering = self._last_steer + float(
            np.clip(
                steering - self._last_steer,
                -self.MAX_STEER_STEP,
                self.MAX_STEER_STEP,
            )
        )
        steering = float(np.clip(steering, -steer_limit, steer_limit))
        self._last_steer = steering
        return np.asarray([steering, gas, brake], dtype=np.float32)

class _CompoundHazardController(_StableCompletionController):
    """Add immediate longitudinal margin for obstacle-adjacent sharp bends."""

    CURVE_TARGET_FLOOR = 36.0
    COMPOUND_TARGET_SPEED = 30.0

    def _adjust_target_speed(
        self,
        *,
        target_speed: float,
        curve_target_speed: float,
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        if obstacle is not None and curve_target_speed <= self.CURVE_TARGET_FLOOR:
            return min(target_speed, self.COMPOUND_TARGET_SPEED)
        return target_speed

class _CurveAwareObstacleController(_CompoundHazardController):
    """Keep obstacle avoidance from cancelling the visible bend command."""

    MIN_CURVE_RETENTION = 0.5

    def _adjust_obstacle_steering(
        self,
        *,
        base_steering: float,
        obstacle_bias: float,
        straight: bool,
    ) -> float:
        proposed = base_steering + obstacle_bias
        if straight or base_steering * obstacle_bias >= 0.0:
            return proposed
        direction = 1.0 if base_steering > 0.0 else -1.0
        minimum = self.MIN_CURVE_RETENTION * abs(base_steering)
        return direction * max(direction * proposed, minimum)

class _GuardedCompletionController(_CurveAwareObstacleController):
    """Final bare-checkpoint policy with conservative clear-straight pace."""

    CLEAR_STRAIGHT_MAX_GAS = 0.10

    def _adjust_pedals(
        self,
        *,
        gas: float,
        brake: float,
        straight: bool,
        obstacle: tuple[float, float, float] | None,
    ) -> tuple[float, float]:
        if (
            straight
            and obstacle is None
            and gas > 0.0
            and brake == 0.0
            and abs(self._last_steer) <= self.MAX_STEER_STEP
        ):
            gas = min(
                gas * (self.CLEAR_STRAIGHT_MAX_GAS / self.MAX_GAS),
                self.CLEAR_STRAIGHT_MAX_GAS,
            )
        return gas, brake

class _DistantObstacleController(_GuardedCompletionController):
    """Begin a bounded lateral offset on the first straight obstacle read."""

    STRAIGHT_OBSTACLE_MIN_URGENCY = 0.50

    def _adjust_obstacle_urgency(
        self,
        *,
        urgency: float,
        straight: bool,
    ) -> float:
        if straight:
            return max(urgency, self.STRAIGHT_OBSTACLE_MIN_URGENCY)
        return urgency

class _AnticipatoryCompletionController(_DistantObstacleController):
    """Align turn-in with the distant road rows already used for speed."""

    DISTANT_PREVIEW_ROWS = (30, 34)
    DISTANT_PREVIEW_STEER = 0.05
    DISTANT_PREVIEW_BASE_LIMIT = 0.5 * _StableCompletionController.MAX_STEER_STEP

    def _adjust_road_steering(
        self,
        *,
        steering: float,
        straight: bool,
        centers: dict[int, float],
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        if (
            obstacle is not None
            or straight
            or abs(steering) > self.DISTANT_PREVIEW_BASE_LIMIT
            or 42 not in centers
            or 54 not in centers
            or any(row not in centers for row in self.DISTANT_PREVIEW_ROWS)
        ):
            return steering

        far = centers[42]
        near = centers[54]
        if (
            abs(far - self.IMAGE_CENTER) > self.STRAIGHT_CENTER_DEADBAND
            or abs(far - near) > self.STRAIGHT_SWEEP_DEADBAND
        ):
            return steering

        deltas = tuple(centers[row] - far for row in self.DISTANT_PREVIEW_ROWS)
        if (
            any(abs(delta) <= self.STRAIGHT_CENTER_DEADBAND for delta in deltas)
            or deltas[0] * deltas[1] <= 0.0
        ):
            return steering

        direction = 1.0 if deltas[0] > 0.0 else -1.0
        if steering * direction < 0.0:
            return steering
        return direction * self.DISTANT_PREVIEW_STEER

class _ObstaclePriorityController(_AnticipatoryCompletionController):
    """Prevent a prior bend preview from weakening new obstacle avoidance."""

    def __init__(self, *, cruise_speed: float = 48.0) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._previous_action_was_preview = False
        self._preview_transition_pending = False
        self._preview_requested = 0.0

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._previous_action_was_preview = False
        self._preview_transition_pending = False
        self._preview_requested = 0.0

    def act(self, observation) -> np.ndarray:
        self._preview_transition_pending = self._previous_action_was_preview
        self._preview_requested = 0.0
        action = super().act(observation)
        self._previous_action_was_preview = (
            self._preview_requested != 0.0
            and self._last_steer == self._preview_requested
        )
        self._preview_transition_pending = False
        return action

    def _adjust_road_steering(
        self,
        *,
        steering: float,
        straight: bool,
        centers: dict[int, float],
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        adjusted = super()._adjust_road_steering(
            steering=steering,
            straight=straight,
            centers=centers,
            obstacle=obstacle,
        )
        if adjusted != steering:
            self._preview_requested = adjusted
        return adjusted

    def _adjust_obstacle_steering(
        self,
        *,
        base_steering: float,
        obstacle_bias: float,
        straight: bool,
    ) -> float:
        adjusted = super()._adjust_obstacle_steering(
            base_steering=base_steering,
            obstacle_bias=obstacle_bias,
            straight=straight,
        )
        if self._preview_transition_pending and adjusted * self._last_steer < 0.0:
            self._last_steer = 0.0
        return adjusted

class _LaunchThrottleController(_ObstaclePriorityController):
    """Add a small launch-only gain without moving the braking boundary."""

    CLEAR_STRAIGHT_LAUNCH_GAS = 0.11

    def _adjust_pedals(
        self,
        *,
        gas: float,
        brake: float,
        straight: bool,
        obstacle: tuple[float, float, float] | None,
    ) -> tuple[float, float]:
        adjusted_gas, adjusted_brake = super()._adjust_pedals(
            gas=gas,
            brake=brake,
            straight=straight,
            obstacle=obstacle,
        )
        if (
            gas == self.MAX_GAS
            and adjusted_gas == self.CLEAR_STRAIGHT_MAX_GAS
            and adjusted_brake == 0.0
        ):
            adjusted_gas = self.CLEAR_STRAIGHT_LAUNCH_GAS
        return adjusted_gas, adjusted_brake

class _LatchedClearStraightSustainController(_LaunchThrottleController):
    """Sustain launch gas only after obstacle avoidance has fully cleared."""

    def _adjust_pedals(
        self,
        *,
        gas: float,
        brake: float,
        straight: bool,
        obstacle: tuple[float, float, float] | None,
    ) -> tuple[float, float]:
        adjusted_gas, adjusted_brake = super()._adjust_pedals(
            gas=gas,
            brake=brake,
            straight=straight,
            obstacle=obstacle,
        )
        if (
            self._obstacle_side == 0.0
            and straight
            and obstacle is None
            and gas > 0.0
            and brake == 0.0
            and abs(self._last_steer) <= self.MAX_STEER_STEP
        ):
            sustained_gas = min(
                gas * (self.CLEAR_STRAIGHT_LAUNCH_GAS / self.MAX_GAS),
                self.CLEAR_STRAIGHT_LAUNCH_GAS,
            )
            adjusted_gas = max(adjusted_gas, sustained_gas)
        return adjusted_gas, adjusted_brake

class _HighSpeedPreviewBrakeController(_LatchedClearStraightSustainController):
    """Accelerate hard only on clear straights and brake from curve preview.

    The lateral controller and its obstacle state machine remain inherited.
    This layer owns only the emitted longitudinal speed envelope: a high clear
    straight target, an immediate sweep-based corner target, and stronger
    overspeed feedback while a curve remains visible.
    """

    STRAIGHT_TARGET_SPEED = 68.0
    STRAIGHT_GAS_FULL = 0.18
    STRAIGHT_GAS_MID = 0.135
    STRAIGHT_GAS_HOLD = 0.09
    CURVE_SWEEP_THRESHOLD = 1.5
    CURVE_TARGET_INTERCEPT = 52.0
    CURVE_TARGET_SLOPE = 3.0
    CURVE_TARGET_FLOOR = 30.0
    CURVE_TARGET_CEILING = 48.0
    CURVE_BRAKE_TRIGGER_DELTA = 0.5
    CURVE_BRAKE_BASE = 0.06
    CURVE_BRAKE_GAIN = 0.018
    COMPOUND_SWEEP_THRESHOLD = 6.0

    def __init__(self, *, cruise_speed: float = STRAIGHT_TARGET_SPEED) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._pace_straight = False
        self._pace_sweep = 0.0
        self._pace_speed = 0.0
        self._pace_command_target = None
        self._pace_latched_target = None

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._pace_straight = False
        self._pace_sweep = 0.0
        self._pace_speed = 0.0
        self._pace_command_target = None
        self._pace_latched_target = None

    def _adjust_road_steering(
        self,
        *,
        steering: float,
        straight: bool,
        centers: dict[int, float],
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        self._pace_straight = bool(straight)
        return super()._adjust_road_steering(
            steering=steering,
            straight=straight,
            centers=centers,
            obstacle=obstacle,
        )

    def _adjust_target_speed(
        self,
        *,
        target_speed: float,
        curve_target_speed: float,
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        inherited = super()._adjust_target_speed(
            target_speed=target_speed,
            curve_target_speed=curve_target_speed,
            obstacle=obstacle,
        )
        # The raw inherited curve target is cruise-2*sweep before its EMA.
        # Recover that generic image-space sweep so falling targets can bypass
        # stale straight-speed memory without changing road perception.
        sweep = max(0.0, 0.5 * (self.cruise_speed - curve_target_speed))
        self._pace_sweep = sweep

        if obstacle is not None:
            obstacle_y = float(obstacle[0])
            obstacle_target = 47.0 if obstacle_y < 44.0 else 40.0
            if sweep >= self.COMPOUND_SWEEP_THRESHOLD:
                obstacle_target = min(
                    obstacle_target, self.COMPOUND_TARGET_SPEED
                )
            self._pace_latched_target = obstacle_target
            command_target = min(inherited, obstacle_target)
        elif self._obstacle_side != 0.0:
            latch_target = (
                self._pace_latched_target
                if self._pace_latched_target is not None
                else self.COMPOUND_TARGET_SPEED
            )
            command_target = min(inherited, latch_target)
        else:
            self._pace_latched_target = None
            if self._pace_straight and sweep <= self.CURVE_SWEEP_THRESHOLD:
                command_target = self.STRAIGHT_TARGET_SPEED
            else:
                command_target = float(
                    np.clip(
                        self.CURVE_TARGET_INTERCEPT
                        - self.CURVE_TARGET_SLOPE * sweep,
                        self.CURVE_TARGET_FLOOR,
                        self.CURVE_TARGET_CEILING,
                    )
                )
        self._pace_command_target = float(command_target)
        return float(command_target)

    def _pedals(self, speed: float, target_speed: float) -> tuple[float, float]:
        self._pace_speed = float(speed)
        return super()._pedals(speed, target_speed)

    def _curve_brake_envelope(
        self,
        *,
        excess: float,
        base_brake: float,
    ) -> float:
        """Return the curve overspeed brake without weakening the base brake."""
        curve_brake = float(
            np.clip(
                self.CURVE_BRAKE_BASE + self.CURVE_BRAKE_GAIN * excess,
                self.CURVE_BRAKE_BASE,
                self.MAX_BRAKE,
            )
        )
        return max(base_brake, curve_brake)

    def _adjust_pedals(
        self,
        *,
        gas: float,
        brake: float,
        straight: bool,
        obstacle: tuple[float, float, float] | None,
    ) -> tuple[float, float]:
        adjusted_gas, adjusted_brake = super()._adjust_pedals(
            gas=gas,
            brake=brake,
            straight=straight,
            obstacle=obstacle,
        )
        fast_straight = (
            self._obstacle_side == 0.0
            and self._pace_straight
            and self._pace_sweep <= self.CURVE_SWEEP_THRESHOLD
            and obstacle is None
            and gas > 0.0
            and brake == 0.0
            and adjusted_brake == 0.0
            and abs(self._last_steer) <= self.MAX_STEER_STEP
        )
        if fast_straight:
            if gas >= self.MAX_GAS - 1e-12:
                adjusted_gas = self.STRAIGHT_GAS_FULL
            elif gas >= self.MAX_GAS * (2.0 / 3.0) - 1e-12:
                adjusted_gas = self.STRAIGHT_GAS_MID
            else:
                adjusted_gas = self.STRAIGHT_GAS_HOLD
            return adjusted_gas, 0.0

        curve_overspeed = (
            not self._pace_straight
            and self._pace_command_target is not None
            and self._pace_speed
            > self._pace_command_target + self.CURVE_BRAKE_TRIGGER_DELTA
        )
        if curve_overspeed:
            excess = (
                self._pace_speed
                - self._pace_command_target
                - self.CURVE_BRAKE_TRIGGER_DELTA
            )
            requested_brake = self._curve_brake_envelope(
                excess=excess,
                base_brake=adjusted_brake,
            )
            bounded_brake = min(
                requested_brake,
                float(
                    np.nextafter(
                        np.float32(self.MAX_BRAKE), np.float32(0.0)
                    )
                ),
            )
            return 0.0, bounded_brake
        adjusted_brake = min(
            adjusted_brake,
            float(
                np.nextafter(np.float32(self.MAX_BRAKE), np.float32(0.0))
            ),
        )
        return adjusted_gas, adjusted_brake

class _FastCornerCarryController(_HighSpeedPreviewBrakeController):
    """Carry speed through clear curves without weakening hazard handling.

    The parent remains the exact safety controller whenever an obstacle is
    visible or its miss latch is active.  On obstacle-free curves only, this
    layer replaces the abrupt linear speed drop with a bounded inverse-square-
    root envelope and reserves some of the shared tire-force budget for the
    requested steering angle.
    """

    CARRY_CURVATURE_GAIN = 0.10
    CARRY_TARGET_FLOOR = 40.0
    CARRY_TARGET_CEILING = 64.0
    CARRY_GAS_FULL = 0.14
    CARRY_GAS_MID = 0.11
    CARRY_GAS_HOLD = 0.07
    CARRY_BRAKE_TRIGGER_DELTA = 2.0
    CARRY_BRAKE_BASE = 0.04
    CARRY_BRAKE_GAIN = 0.012
    CARRY_BRAKE_MAX = 0.18

    def __init__(self, *, cruise_speed: float = 68.0) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._carry_steer_request = 0.0
        self._carry_sweep = 0.0
        self._carry_latched_at_frame_start = False
        self._pace_effective_target = None

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._carry_steer_request = 0.0
        self._carry_sweep = 0.0
        self._carry_latched_at_frame_start = False
        self._pace_effective_target = None

    def _adjust_road_steering(
        self,
        *,
        steering: float,
        straight: bool,
        centers: dict[int, float],
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        adjusted = super()._adjust_road_steering(
            steering=steering,
            straight=straight,
            centers=centers,
            obstacle=obstacle,
        )
        self._carry_steer_request = float(adjusted)
        self._carry_sweep = float(self._road_sweep(centers))
        # act() updates the miss counter after this hook.  Remember the entry
        # state so the frame that clears a latch still uses the parent policy.
        self._carry_latched_at_frame_start = self._obstacle_side != 0.0
        return adjusted

    def _adjust_target_speed(
        self,
        *,
        target_speed: float,
        curve_target_speed: float,
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        inherited = super()._adjust_target_speed(
            target_speed=target_speed,
            curve_target_speed=curve_target_speed,
            obstacle=obstacle,
        )
        if (
            obstacle is not None
            or self._obstacle_side != 0.0
            or self._carry_latched_at_frame_start
            or self._pace_straight
        ):
            return inherited

        command_target = float(
            np.clip(
                self.STRAIGHT_TARGET_SPEED
                / np.sqrt(
                    1.0 + self.CARRY_CURVATURE_GAIN * self._carry_sweep
                ),
                self.CARRY_TARGET_FLOOR,
                self.CARRY_TARGET_CEILING,
            )
        )
        self._pace_command_target = command_target
        return command_target

    def _pedals(self, speed: float, target_speed: float) -> tuple[float, float]:
        # This receives the strong-steering cap applied by act() after the
        # preview target hook, so longitudinal control cannot bypass target44.
        self._pace_effective_target = float(target_speed)
        return super()._pedals(speed, target_speed)

    def _adjust_pedals(
        self,
        *,
        gas: float,
        brake: float,
        straight: bool,
        obstacle: tuple[float, float, float] | None,
    ) -> tuple[float, float]:
        inherited_gas, inherited_brake = super()._adjust_pedals(
            gas=gas,
            brake=brake,
            straight=straight,
            obstacle=obstacle,
        )
        if (
            obstacle is not None
            or self._obstacle_side != 0.0
            or self._carry_latched_at_frame_start
            or self._pace_straight
            or self._pace_effective_target is None
        ):
            return inherited_gas, inherited_brake

        target = self._pace_effective_target
        if self._pace_speed > target + self.CARRY_BRAKE_TRIGGER_DELTA:
            excess = (
                self._pace_speed
                - target
                - self.CARRY_BRAKE_TRIGGER_DELTA
            )
            curve_brake = float(
                np.clip(
                    self.CARRY_BRAKE_BASE
                    + self.CARRY_BRAKE_GAIN * excess,
                    self.CARRY_BRAKE_BASE,
                    self.CARRY_BRAKE_MAX,
                )
            )
            curve_brake = min(
                curve_brake,
                float(
                    np.nextafter(
                        np.float32(self.CARRY_BRAKE_MAX), np.float32(0.0)
                    )
                ),
            )
            return 0.0, curve_brake
        if self._pace_speed >= target:
            return 0.0, 0.0

        steering_ratio = abs(self._carry_steer_request) / self.MAX_STEER
        steering_factor = float(
            np.sqrt(max(0.25, 1.0 - steering_ratio * steering_ratio))
        )
        if self._pace_speed < target - 8.0:
            curve_gas = self.CARRY_GAS_FULL
        elif self._pace_speed < target - 3.0:
            curve_gas = self.CARRY_GAS_MID
        else:
            curve_gas = self.CARRY_GAS_HOLD
        return curve_gas * steering_factor, 0.0

class _PostObstacleCurveRetentionController(_FastCornerCarryController):
    """Retain a coherent turn only through the bounded obstacle-miss latch."""

    RETENTION_GAIN = 1.20
    RETENTION_MAX_INCREMENT = 0.06
    RETENTION_DELTA_MIN = 1.5
    RETENTION_ROWS = (30, 34, 38, 42, 46, 50, 54)

    def __init__(self, *, cruise_speed: float = 68.0) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._curve_retention_active = False
        self._curve_retention_direction = 0.0
        self._curve_retention_control_request = 0.0

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._curve_retention_active = False
        self._curve_retention_direction = 0.0
        self._curve_retention_control_request = 0.0

    def _lost_road_action(self) -> np.ndarray:
        self._curve_retention_active = False
        self._curve_retention_direction = 0.0
        self._curve_retention_control_request = 0.0
        return super()._lost_road_action()

    def _adjust_road_steering(
        self,
        *,
        steering: float,
        straight: bool,
        centers: dict[int, float],
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        inherited = super()._adjust_road_steering(
            steering=steering,
            straight=straight,
            centers=centers,
            obstacle=obstacle,
        )
        self._curve_retention_active = False
        self._curve_retention_direction = 0.0
        self._curve_retention_control_request = float(inherited)

        if (
            straight
            or obstacle is not None
            or not self._carry_latched_at_frame_start
            or self._obstacle_side == 0.0
            or self._obstacle_missing >= self.OBSTACLE_MISS_LIMIT
            or any(row not in centers for row in self.RETENTION_ROWS)
        ):
            return inherited

        near = float(centers[54])
        delta30 = float(centers[30]) - near
        delta34 = float(centers[34]) - near
        if (
            abs(delta30) <= self.RETENTION_DELTA_MIN
            or abs(delta34) <= self.RETENTION_DELTA_MIN
            or delta30 * delta34 <= 0.0
        ):
            return inherited

        direction = 1.0 if delta30 > 0.0 else -1.0
        if (
            direction != self._obstacle_side
            or inherited * direction <= 0.0
            or any(
                (float(centers[row]) - near) * direction < 0.0
                for row in self.RETENTION_ROWS[:-1]
            )
        ):
            return inherited

        magnitude = min(
            self.MAX_STEER,
            self.RETENTION_GAIN * abs(inherited),
            abs(inherited) + self.RETENTION_MAX_INCREMENT,
        )
        if abs(inherited) <= self.MAX_STEER_STEP <= magnitude:
            return inherited

        candidate = direction * magnitude
        self._curve_retention_active = True
        self._curve_retention_direction = direction
        self._carry_steer_request = candidate
        if self._preview_requested != 0.0 and self._preview_requested == inherited:
            self._preview_requested = candidate
        return candidate

class _DoubleClearStraightThrottleController(
    _PostObstacleCurveRetentionController
):
    """Double propulsion only inside the settled clear-straight gate.

    Speed targets and every brake, curve, obstacle, latch and steering branch
    remain inherited.  This changes acceleration toward the existing target68;
    it does not ask the camera speed estimator to regulate above its range.
    """

    CLEAR_STRAIGHT_THROTTLE_GAIN = 2.0
    CLEAR_STRAIGHT_GAS_CAP = 0.36

    def _adjust_pedals(
        self,
        *,
        gas: float,
        brake: float,
        straight: bool,
        obstacle: tuple[float, float, float] | None,
    ) -> tuple[float, float]:
        inherited_gas, inherited_brake = super()._adjust_pedals(
            gas=gas,
            brake=brake,
            straight=straight,
            obstacle=obstacle,
        )
        settled_clear_straight = (
            inherited_gas > 0.0
            and inherited_brake == 0.0
            and straight
            and self._pace_straight
            and self._pace_sweep <= self.CURVE_SWEEP_THRESHOLD
            and obstacle is None
            and self._obstacle_side == 0.0
            and not self._carry_latched_at_frame_start
            and abs(self._last_steer) <= self.MAX_STEER_STEP
        )
        if settled_clear_straight:
            return (
                min(
                    self.CLEAR_STRAIGHT_GAS_CAP,
                    self.CLEAR_STRAIGHT_THROTTLE_GAIN * inherited_gas,
                ),
                0.0,
            )
        return inherited_gas, inherited_brake

class _CompoundObstacleBrakeCarryController(
    _DoubleClearStraightThrottleController
):
    """Reduce only redundant curve braking in a latched compound hazard.

    The target30 speed envelope, obstacle detector, four-frame miss latch,
    steering, gas, and inherited distance/speed brake all remain unchanged.
    At moderate overspeed the supplemental curve brake uses the already-tested
    clear-corner slope; at high overspeed the inherited base brake is the floor.
    """

    COMPOUND_CARRY_BRAKE_BASE = 0.04
    COMPOUND_CARRY_BRAKE_GAIN = 0.012
    COMPOUND_CARRY_BRAKE_MAX = 0.18

    def _compound_carry_active(self) -> bool:
        return bool(
            self._pace_latched_target == self.COMPOUND_TARGET_SPEED
            and self._pace_command_target == self.COMPOUND_TARGET_SPEED
            and (
                self._obstacle_side != 0.0
                or self._carry_latched_at_frame_start
            )
        )

    def _curve_brake_envelope(
        self,
        *,
        excess: float,
        base_brake: float,
    ) -> float:
        if not self._compound_carry_active():
            return super()._curve_brake_envelope(
                excess=excess,
                base_brake=base_brake,
            )

        carry_brake = float(
            np.clip(
                self.COMPOUND_CARRY_BRAKE_BASE
                + self.COMPOUND_CARRY_BRAKE_GAIN * excess,
                self.COMPOUND_CARRY_BRAKE_BASE,
                self.COMPOUND_CARRY_BRAKE_MAX,
            )
        )
        return max(base_brake, carry_brake)

class _CompoundObstacleLatchReleaseController(
    _CompoundObstacleBrakeCarryController
):
    """Drop only redundant curve braking during compound detector misses.

    The obstacle-side latch, target30, steering and inherited base speed brake
    remain active.  A currently visible obstacle always keeps the parent brake
    envelope; only miss frames1--4 use the base brake without its supplement.
    """

    def _curve_brake_envelope(
        self,
        *,
        excess: float,
        base_brake: float,
    ) -> float:
        compound_detector_miss = (
            self._pace_latched_target == self.COMPOUND_TARGET_SPEED
            and self._pace_command_target == self.COMPOUND_TARGET_SPEED
            and self._carry_latched_at_frame_start
            and self._obstacle_side != 0.0
            and 0 < self._obstacle_missing <= self.OBSTACLE_MISS_LIMIT
        )
        if compound_detector_miss:
            return base_brake
        return super()._curve_brake_envelope(
            excess=excess,
            base_brake=base_brake,
        )

class _AdaptiveCompoundTargetController(
    _CompoundObstacleLatchReleaseController
):
    """Raise only far, moderate compound targets with continuous safeguards.

    Near obstacles, extreme hairpins and every detector-miss latch retain the
    exact target30 policy.  The current command can rise to36, but the stored
    latch remains30 so a single missed detection restores the safety target.
    """

    ADAPTIVE_COMPOUND_MAX_TARGET = 36.0
    ADAPTIVE_COMPOUND_FAR_ROW = 32.0
    ADAPTIVE_COMPOUND_NEAR_ROW = 44.0
    ADAPTIVE_COMPOUND_EXTREME_SWEEP = 12.0
    ADAPTIVE_COMPOUND_MAX_STEER_REQUEST = 0.28

    def _adjust_target_speed(
        self,
        *,
        target_speed: float,
        curve_target_speed: float,
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        inherited = super()._adjust_target_speed(
            target_speed=target_speed,
            curve_target_speed=curve_target_speed,
            obstacle=obstacle,
        )
        if (
            obstacle is None
            or self._pace_latched_target != self.COMPOUND_TARGET_SPEED
            or self._pace_sweep < self.COMPOUND_SWEEP_THRESHOLD
            or self._pace_sweep >= self.ADAPTIVE_COMPOUND_EXTREME_SWEEP
        ):
            return inherited

        obstacle_y = float(obstacle[0])
        if obstacle_y >= self.ADAPTIVE_COMPOUND_NEAR_ROW:
            return inherited

        distance_factor = float(
            np.clip(
                (self.ADAPTIVE_COMPOUND_NEAR_ROW - obstacle_y)
                / (
                    self.ADAPTIVE_COMPOUND_NEAR_ROW
                    - self.ADAPTIVE_COMPOUND_FAR_ROW
                ),
                0.0,
                1.0,
            )
        )
        severity_factor = float(
            np.clip(
                (
                    self.ADAPTIVE_COMPOUND_EXTREME_SWEEP
                    - self._pace_sweep
                )
                / (
                    self.ADAPTIVE_COMPOUND_EXTREME_SWEEP
                    - self.COMPOUND_SWEEP_THRESHOLD
                ),
                0.0,
                1.0,
            )
        )
        command_target = float(
            self.COMPOUND_TARGET_SPEED
            + (
                self.ADAPTIVE_COMPOUND_MAX_TARGET
                - self.COMPOUND_TARGET_SPEED
            )
            * distance_factor
            * severity_factor
        )
        self._pace_command_target = command_target
        return command_target

    def _adjust_target_speed_for_steering(
        self,
        *,
        target_speed: float,
        steering: float,
        straight: bool,
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        inherited = super()._adjust_target_speed_for_steering(
            target_speed=target_speed,
            steering=steering,
            straight=straight,
            obstacle=obstacle,
        )
        if (
            obstacle is not None
            and not straight
            and abs(steering) > self.ADAPTIVE_COMPOUND_MAX_STEER_REQUEST
            and self._pace_latched_target == self.COMPOUND_TARGET_SPEED
            and self._pace_command_target is not None
            and self._pace_command_target > self.COMPOUND_TARGET_SPEED
        ):
            self._pace_command_target = self.COMPOUND_TARGET_SPEED
            return self.COMPOUND_TARGET_SPEED
        return inherited

    def _compound_carry_active(self) -> bool:
        if super()._compound_carry_active():
            return True
        return bool(
            self._pace_latched_target == self.COMPOUND_TARGET_SPEED
            and self._pace_command_target is not None
            and self.COMPOUND_TARGET_SPEED
            < self._pace_command_target
            <= self.ADAPTIVE_COMPOUND_MAX_TARGET
            and self._obstacle_side != 0.0
            and self._obstacle_missing == 0
            and self.COMPOUND_SWEEP_THRESHOLD
            <= self._pace_sweep
            < self.ADAPTIVE_COMPOUND_EXTREME_SWEEP
        )

class _AggressiveCompoundPaceController(_AdaptiveCompoundTargetController):
    """Carry two more speed units only inside the adaptive safety gate."""

    ADAPTIVE_COMPOUND_MAX_TARGET = 38.0

class _CompoundClearingBrakeCarryController(_AggressiveCompoundPaceController):
    """Avoid a one-frame brake spike when a safe compound latch clears.

    A far/moderate adaptive obstacle arms this transition only after its final
    steering request passes the existing safety veto.  Miss frames1--4 remain
    parent-exact.  On the fifth miss, when the obstacle latch clears, the target,
    gas and steering stay unchanged while the brake moves by at most0.04 toward
    the already-qualified compound envelope for that frame only.
    """

    CLEARING_BRAKE_RELIEF_MAX = 0.04

    def __init__(self, *, cruise_speed: float = 68.0) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._compound_clearing_armed = False
        self._compound_clearing_active = False
        self._compound_clearing_direction = 0.0

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._compound_clearing_armed = False
        self._compound_clearing_active = False
        self._compound_clearing_direction = 0.0

    def _lost_road_action(self) -> np.ndarray:
        self._compound_clearing_armed = False
        self._compound_clearing_active = False
        self._compound_clearing_direction = 0.0
        return super()._lost_road_action()

    def _adjust_target_speed_for_steering(
        self,
        *,
        target_speed: float,
        steering: float,
        straight: bool,
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        inherited = super()._adjust_target_speed_for_steering(
            target_speed=target_speed,
            steering=steering,
            straight=straight,
            obstacle=obstacle,
        )
        self._compound_clearing_active = False

        if obstacle is not None:
            self._compound_clearing_armed = bool(
                self._pace_latched_target == self.COMPOUND_TARGET_SPEED
                and self._pace_command_target is not None
                and self._pace_command_target > self.COMPOUND_TARGET_SPEED
                and self.COMPOUND_SWEEP_THRESHOLD <= self._pace_sweep
                < self.ADAPTIVE_COMPOUND_EXTREME_SWEEP
                and steering != 0.0
                and abs(steering) <= self.ADAPTIVE_COMPOUND_MAX_STEER_REQUEST
            )
            self._compound_clearing_direction = (
                float(np.sign(steering))
                if self._compound_clearing_armed
                else 0.0
            )
            return inherited

        steering_reversed = bool(
            self._compound_clearing_direction != 0.0
            and steering != 0.0
            and steering * self._compound_clearing_direction < 0.0
        )
        safe_compound_geometry = bool(
            self.COMPOUND_SWEEP_THRESHOLD <= self._pace_sweep
            < self.ADAPTIVE_COMPOUND_EXTREME_SWEEP
            and abs(steering) <= self.ADAPTIVE_COMPOUND_MAX_STEER_REQUEST
            and not steering_reversed
        )
        if self._carry_latched_at_frame_start and self._obstacle_side != 0.0:
            if not safe_compound_geometry:
                self._compound_clearing_armed = False
                self._compound_clearing_direction = 0.0
            return inherited

        clearing_frame = bool(
            self._compound_clearing_armed
            and self._carry_latched_at_frame_start
            and self._obstacle_side == 0.0
            and self._obstacle_missing == 0
            and self._pace_latched_target is None
            and self._pace_command_target is not None
            and self._pace_command_target > self.COMPOUND_TARGET_SPEED
            and safe_compound_geometry
        )
        if clearing_frame:
            self._compound_clearing_active = True
            self._compound_clearing_armed = False
            self._compound_clearing_direction = 0.0
        elif not self._carry_latched_at_frame_start:
            self._compound_clearing_armed = False
            self._compound_clearing_direction = 0.0
        return inherited

    def _curve_brake_envelope(
        self,
        *,
        excess: float,
        base_brake: float,
    ) -> float:
        control_brake = super()._curve_brake_envelope(
            excess=excess,
            base_brake=base_brake,
        )
        if not self._compound_clearing_active:
            return control_brake
        carry_brake = float(
            np.clip(
                self.COMPOUND_CARRY_BRAKE_BASE
                + self.COMPOUND_CARRY_BRAKE_GAIN * excess,
                self.COMPOUND_CARRY_BRAKE_BASE,
                self.COMPOUND_CARRY_BRAKE_MAX,
            )
        )
        return max(
            base_brake,
            carry_brake,
            control_brake - self.CLEARING_BRAKE_RELIEF_MAX,
        )

class _FeasibleCorridorObstacleController(_CompoundClearingBrakeCarryController):
    """Diagnostic-only obstacle pass constrained by camera-visible free space.

    The existing Agent route does not select this class. Clear-road decisions
    follow the parent exactly. An uncertain obstacle or road edge also falls
    back to the parent's steering; only a fully checked passage replaces the
    sign-of-centroid obstacle bias.
    """

    CAR_ROW = 63.0
    ROAD_EDGE_MARGIN = 2.63
    OBSTACLE_EDGE_MARGIN = 3.3
    MIN_FREE_WIDTH = 0.75
    MAX_SHIFT_PER_ROW = 0.4
    CORRIDOR_POSITION_GAIN = 0.016
    CORRIDOR_HEADING_GAIN = 0.012
    MIN_SWITCH_CLEARANCE_GAIN = 2.0
    SWITCH_APPROACH_RESERVE_ROWS = 4.0

    def __init__(self, *, cruise_speed: float = 68.0) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._corridor_bbox = None
        self._corridor_edges = {}
        self._corridor_centers = {}
        self._corridor_detection = None
        self._corridor_previous = None
        self._corridor_misses = 0
        self._corridor_plan = None

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._corridor_bbox = None
        self._corridor_edges = {}
        self._corridor_centers = {}
        self._corridor_detection = None
        self._corridor_previous = None
        self._corridor_misses = 0
        self._corridor_plan = None

    def act(self, observation) -> np.ndarray:
        self._corridor_plan = None
        self._corridor_detection = None
        self._corridor_bbox = None
        self._corridor_edges = {}
        self._corridor_centers = {}
        return super().act(observation)

    def _lost_road_action(self) -> np.ndarray:
        self._corridor_previous = None
        self._corridor_misses = 0
        return super()._lost_road_action()

    @classmethod
    def _component_bbox(
        cls, frame: np.ndarray, x: float, y: float,
    ) -> tuple[int, int, int, int] | None:
        """Recover the component bounds omitted from the legacy detector."""
        bright = frame >= cls.OBSTACLE_LOW
        cx, cy = int(round(x)), int(round(y))
        seeds = [
            (px, py)
            for py in range(max(22, cy - 3), min(62, cy + 4))
            for px in range(max(0, cx - 3), min(84, cx + 4))
            if bright[py, px]
        ]
        if not seeds:
            return None
        seed = min(seeds, key=lambda point: (point[0] - x) ** 2 + (point[1] - y) ** 2)
        visited = {seed}
        stack = [seed]
        pixels = []
        while stack:
            px, py = stack.pop()
            pixels.append((px, py))
            if len(pixels) > 80:
                return None
            for ny in range(max(22, py - 1), min(62, py + 2)):
                for nx in range(max(0, px - 1), min(84, px + 2)):
                    point = (nx, ny)
                    if bright[ny, nx] and point not in visited:
                        visited.add(point)
                        stack.append(point)
        xs, ys = zip(*pixels)
        bbox = min(xs), min(ys), max(xs), max(ys)
        if (
            not 4 <= len(pixels) <= 80
            or not 2 <= bbox[2] - bbox[0] + 1 <= 9
            or not 2 <= bbox[3] - bbox[1] + 1 <= 10
            or abs(float(np.mean(xs)) - x) > 1.5
            or abs(float(np.mean(ys)) - y) > 1.5
        ):
            return None
        return bbox

    @classmethod
    def _visible_edges(
        cls,
        frame: np.ndarray,
        centers: dict[int, float],
        bbox: tuple[int, int, int, int],
    ) -> dict[int, tuple[float, float]]:
        asphalt = (frame >= cls.ROAD_LOW) & (frame <= cls.ROAD_HIGH)
        edges = {}
        obscured = {}
        for row in range(22, 59):
            center = cls._center_at(float(row), centers)
            lo = max(0, int(np.floor(center - 18.0)))
            hi = min(83, int(np.ceil(center + 18.0)))
            xs = np.flatnonzero(asphalt[row, lo : hi + 1]) + lo
            if len(xs) < 6 or xs[0] <= lo or xs[-1] >= hi:
                continue
            gaps = np.flatnonzero(np.diff(xs) > 1)
            if any(
                not (bbox[1] <= row <= bbox[3]
                     and xs[index] < bbox[0] <= bbox[2] < xs[index + 1])
                for index in gaps
            ):
                continue
            span = float(xs[0]), float(xs[-1])
            if bbox[1] <= row <= bbox[3]:
                obscured[row] = span
            else:
                edges[row] = span
        clean_edges = dict(edges)
        for row, visible in obscured.items():
            below = max((known for known in clean_edges if known < row), default=None)
            above = min((known for known in clean_edges if known > row), default=None)
            if below is not None and above is not None and above - below <= 12:
                weight = (row - below) / (above - below)
                interpolated = tuple(
                    (1.0 - weight) * clean_edges[below][side]
                    + weight * clean_edges[above][side]
                    for side in (0, 1)
                )
                edges[row] = (
                    max(visible[0], interpolated[0]),
                    min(visible[1], interpolated[1]),
                )
            else:
                neighbor = below if below is not None else above
                if neighbor is not None and abs(neighbor - row) <= 4:
                    edges[row] = (
                        max(visible[0], clean_edges[neighbor][0]),
                        min(visible[1], clean_edges[neighbor][1]),
                    )
        # A compact obstacle can hide a complete row. Interpolate only when
        # both nearby rows independently reveal both outer asphalt edges.
        for row in range(22, 59):
            if row in edges:
                continue
            below = max((known for known in edges if known < row), default=None)
            above = min((known for known in edges if known > row), default=None)
            if below is None or above is None or above - below > 8:
                continue
            weight = (row - below) / (above - below)
            edges[row] = tuple(
                (1.0 - weight) * edges[below][side] + weight * edges[above][side]
                for side in (0, 1)
            )
        return edges

    def _nearest_obstacle(
        self,
        frame: np.ndarray,
        centers: dict[int, float],
        spans: dict[int, tuple[float, float]] | None = None,
    ) -> tuple[float, float, float] | None:
        obstacle = super()._nearest_obstacle(frame, centers, spans)
        if obstacle is None:
            self._corridor_misses += 1
            if self._corridor_misses > self.OBSTACLE_MISS_LIMIT:
                self._corridor_previous = None
            return None
        self._corridor_misses = 0
        self._corridor_detection = obstacle
        self._corridor_centers = dict(centers)
        self._corridor_bbox = self._component_bbox(frame, obstacle[1], obstacle[0])
        if self._corridor_bbox is not None:
            self._corridor_edges = self._visible_edges(
                frame, centers, self._corridor_bbox
            )
        return obstacle

    @classmethod
    def _corridor_approach_x(
        cls, row: float, target: float, near_y: float,
    ) -> float:
        """Return the checked straight segment from ego to the pass target."""
        return cls.IMAGE_CENTER + (
            (target - cls.IMAGE_CENTER)
            * (cls.CAR_ROW - row) / (cls.CAR_ROW - near_y)
        )

    def _corridor_candidate(self, side: float):
        bbox = self._corridor_bbox
        obstacle = self._corridor_detection
        edges = self._corridor_edges
        if bbox is None or obstacle is None:
            return None
        # The checked road path ends at row 58. Do not clip the obstacle's
        # near extent into that range: doing so would leave its closest rows
        # unverified while still accepting the pass.
        if bbox[3] + 2 > 58:
            return None
        near_y = bbox[3] + 2
        hold_y = max(22, bbox[1] - 2)
        far_y = max(22, bbox[1] - 10)
        if near_y >= self.CAR_ROW:
            return None
        at_obstacle = edges.get(int(round(obstacle[0])))
        if at_obstacle is None:
            return None
        left, right = at_obstacle
        if side < 0.0:
            free_left = left + self.ROAD_EDGE_MARGIN
            free_right = bbox[0] - self.OBSTACLE_EDGE_MARGIN
        else:
            free_left = bbox[2] + self.OBSTACLE_EDGE_MARGIN
            free_right = right - self.ROAD_EDGE_MARGIN
        if free_right - free_left < self.MIN_FREE_WIDTH:
            return None
        target = 0.5 * (free_left + free_right)
        if abs(target - self.IMAGE_CENTER) > self.MAX_SHIFT_PER_ROW * (
            self.CAR_ROW - near_y
        ):
            return None
        far_center = self._center_at(float(far_y), self._corridor_centers)
        clearance = float("inf")
        for row in range(22, 59):
            span = edges.get(row)
            if span is None:
                return None
            if row > near_y:
                x = self._corridor_approach_x(row, target, near_y)
            elif row >= hold_y:
                x = target
            elif row >= far_y:
                x = target + (far_center - target) * (
                    (hold_y - row) / max(1, hold_y - far_y)
                )
            else:
                x = self._center_at(float(row), self._corridor_centers)
            road_clearance = min(
                x - (span[0] + self.ROAD_EDGE_MARGIN),
                (span[1] - self.ROAD_EDGE_MARGIN) - x,
            )
            if road_clearance < 0.0:
                return None
            clearance = min(clearance, road_clearance)
            if bbox[1] - 2 <= row <= bbox[3] + 2:
                obstacle_clearance = (
                    bbox[0] - self.OBSTACLE_EDGE_MARGIN - x
                    if side < 0.0
                    else x - bbox[2] - self.OBSTACLE_EDGE_MARGIN
                )
                if obstacle_clearance < 0.0:
                    return None
                clearance = min(clearance, obstacle_clearance)
        return target, clearance, near_y

    def _adjust_obstacle_steering(
        self, *, base_steering: float, obstacle_bias: float, straight: bool,
    ) -> float:
        if self._corridor_bbox is None:
            return super()._adjust_obstacle_steering(
                base_steering=base_steering,
                obstacle_bias=obstacle_bias,
                straight=straight,
            )
        previous_side = None
        if self._corridor_previous is not None and self._corridor_detection is not None:
            px, py, previous_side = self._corridor_previous
            y, x, _road_center = self._corridor_detection
            if abs(x - px) > 8.0 or not py - 3.0 <= y <= py + 12.0:
                previous_side = None
        choices = []
        for side in (-1.0, 1.0):
            candidate = self._corridor_candidate(side)
            if candidate is None:
                continue
            target, clearance, near_y = candidate
            score = clearance - 0.08 * abs(target - self.IMAGE_CENTER)
            choices.append((score, side, target, clearance, near_y))
        if not choices:
            self._corridor_previous = None
            return super()._adjust_obstacle_steering(
                base_steering=base_steering,
                obstacle_bias=obstacle_bias,
                straight=straight,
            )
        chosen = max(choices)
        previous_choice = next(
            (choice for choice in choices if choice[1] == previous_side), None
        )
        if previous_choice is not None:
            alternate = next(
                (choice for choice in choices if choice[1] != previous_side), None
            )
            if alternate is None:
                chosen = previous_choice
            else:
                _alt_score, _alt_side, alt_target, alt_clearance, alt_near_y = alternate
                approach_rows = (
                    self.CAR_ROW - alt_near_y
                    - self.SWITCH_APPROACH_RESERVE_ROWS
                )
                materially_safer = (
                    alt_clearance - previous_choice[3]
                    >= self.MIN_SWITCH_CLEARANCE_GAIN
                )
                enough_approach = (
                    approach_rows > 0.0
                    and abs(alt_target - self.IMAGE_CENTER)
                    <= self.MAX_SHIFT_PER_ROW * approach_rows
                )
                chosen = (
                    alternate if materially_safer and enough_approach
                    else previous_choice
                )
        _score, side, target, clearance, near_y = chosen
        y, x, _road_center = self._corridor_detection
        self._obstacle_side = side
        self._corridor_previous = (x, y, side)
        control_y = max(42, near_y)
        near_control_y = min(60, max(54, control_y + 4))
        control_x = self._corridor_approach_x(control_y, target, near_y)
        near_control_x = self._corridor_approach_x(
            near_control_y, target, near_y
        )
        self._corridor_plan = {
            "side": side,
            "target_x": target,
            "clearance_px": clearance,
            "bbox": self._corridor_bbox,
            "edges": dict(self._corridor_edges),
            "control_waypoints": (
                (near_control_y, near_control_x),
                (control_y, control_x),
            ),
        }
        urgency = min(1.0, max(0.5, abs(obstacle_bias) / 0.24))
        desired = (
            self.CORRIDOR_POSITION_GAIN * (control_x - self.IMAGE_CENTER)
            + self.CORRIDOR_HEADING_GAIN * (control_x - near_control_x)
        ) * urgency
        if self._preview_transition_pending and desired * self._last_steer < 0.0:
            self._last_steer = 0.0
        return float(desired)

class _FeasibleCorridorFallbackSpeedController(_FeasibleCorridorObstacleController):
    """Brake for a visible obstacle when no camera-verified pass is available."""

    NO_CORRIDOR_TARGET_SPEED = 18.0

    def _adjust_target_speed_for_steering(
        self,
        *,
        target_speed: float,
        steering: float,
        straight: bool,
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        inherited = super()._adjust_target_speed_for_steering(
            target_speed=target_speed,
            steering=steering,
            straight=straight,
            obstacle=obstacle,
        )
        if obstacle is None or self._corridor_plan is not None:
            return inherited
        return min(inherited, self.NO_CORRIDOR_TARGET_SPEED)

class _FeasibleCorridorRoadDropoutController(_FeasibleCorridorFallbackSpeedController):
    """Keep an observed bend when an obstacle hides the far steering row.

    This diagnostic changes only no-corridor obstacle steering. Clear road and
    camera-verified passages retain the preceding controller's decisions.
    """

    def _adjust_road_steering(
        self,
        *,
        steering: float,
        straight: bool,
        centers: dict[int, float],
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        if (
            obstacle is not None
            and not straight
            and 42 not in centers
            and 54 in centers
            and 50 in centers
            and 46 in centers
            and self._corridor_candidate(-1.0) is None
            and self._corridor_candidate(1.0) is None
        ):
            far = self._center_at(42.0, centers)
            near = float(centers[54])
            steering = 0.016 * (far - self.IMAGE_CENTER) + 0.012 * (far - near)
        return super()._adjust_road_steering(
            steering=steering,
            straight=straight,
            centers=centers,
            obstacle=obstacle,
        )

class _FeasibleCorridorTemporalSideController(_FeasibleCorridorRoadDropoutController):
    """Ignore a near-center side flip for the same unplanned obstacle."""

    CENTERLINE_SWITCH_DEADBAND = 1.0
    TRACKED_OBSTACLE_MAX_DX = 8.0
    TRACKED_OBSTACLE_MIN_DY = -3.0
    TRACKED_OBSTACLE_MAX_DY = 12.0

    def __init__(self, *, cruise_speed: float = 68.0) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._temporal_obstacle = None

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._temporal_obstacle = None

    def act(self, observation) -> np.ndarray:
        action = super().act(observation)
        self._temporal_obstacle = None
        if self.road_visible and self._corridor_detection is not None:
            y, x, road_center = self._corridor_detection
            self._temporal_obstacle = (x, y, x - road_center)
        return action

    def _lost_road_action(self) -> np.ndarray:
        self._temporal_obstacle = None
        return super()._lost_road_action()

    def _allow_obstacle_side_switch(
        self,
        *,
        obstacle_y: float,
        obstacle_x: float,
        candidate_side: float,
    ) -> bool:
        inherited = super()._allow_obstacle_side_switch(
            obstacle_y=obstacle_y,
            obstacle_x=obstacle_x,
            candidate_side=candidate_side,
        )
        previous = self._temporal_obstacle
        current = self._corridor_detection
        if (
            not inherited
            or candidate_side == self._obstacle_side
            or previous is None
            or current is None
            or self._corridor_candidate(-1.0) is not None
            or self._corridor_candidate(1.0) is not None
        ):
            return inherited
        previous_x, previous_y, previous_offset = previous
        current_y, current_x, road_center = current
        current_offset = current_x - road_center
        same_obstacle = (
            abs(current_x - previous_x) <= self.TRACKED_OBSTACLE_MAX_DX
            and previous_y + self.TRACKED_OBSTACLE_MIN_DY <= current_y
            <= previous_y + self.TRACKED_OBSTACLE_MAX_DY
        )
        near_center_sign_flip = (
            abs(current_offset) <= self.CENTERLINE_SWITCH_DEADBAND
            and previous_offset * current_offset < 0.0
        )
        return inherited and not (same_obstacle and near_center_sign_flip)

class _FeasibleCorridorSideWidthController(_FeasibleCorridorTemporalSideController):
    """Veto a same-obstacle switch into a visibly pinched road side."""

    SIDE_WIDTH_ROAD_ROWS = 3
    SIDE_WIDTH_ROW_RADIUS = 13.0

    def _allow_obstacle_side_switch(
        self,
        *,
        obstacle_y: float,
        obstacle_x: float,
        candidate_side: float,
    ) -> bool:
        inherited = super()._allow_obstacle_side_switch(
            obstacle_y=obstacle_y,
            obstacle_x=obstacle_x,
            candidate_side=candidate_side,
        )
        current_side = self._obstacle_side
        previous = self._temporal_obstacle
        current = self._corridor_detection
        bbox = self._corridor_bbox
        if (
            not inherited
            or current_side == 0.0
            or candidate_side == current_side
            or previous is None
            or current is None
            or bbox is None
        ):
            return inherited
        previous_x, previous_y, _ = previous
        current_y, current_x, _ = current
        same_obstacle = (
            abs(current_x - previous_x) <= self.TRACKED_OBSTACLE_MAX_DX
            and previous_y + self.TRACKED_OBSTACLE_MIN_DY <= current_y
            <= previous_y + self.TRACKED_OBSTACLE_MAX_DY
        )
        if not same_obstacle:
            return inherited
        if (
            self._corridor_candidate(-1.0) is not None
            or self._corridor_candidate(1.0) is not None
        ):
            return inherited
        nearest_rows = sorted(
            (
                row for row in self._corridor_edges
                if abs(row - obstacle_y) <= self.SIDE_WIDTH_ROW_RADIUS
            ),
            key=lambda row: (abs(row - obstacle_y), row),
        )[:self.SIDE_WIDTH_ROAD_ROWS]
        if (
            len(nearest_rows) < self.SIDE_WIDTH_ROAD_ROWS
            or max(nearest_rows) - min(nearest_rows)
            > self.SIDE_WIDTH_ROAD_ROWS - 1
        ):
            return inherited

        def usable_width(side: float, row: int) -> float:
            left, right = self._corridor_edges[row]
            if side < 0.0:
                return bbox[0] - self.OBSTACLE_EDGE_MARGIN - (
                    left + self.ROAD_EDGE_MARGIN
                )
            return (right - self.ROAD_EDGE_MARGIN) - (
                bbox[2] + self.OBSTACLE_EDGE_MARGIN
            )

        blocked_switch = all(
            usable_width(candidate_side, row) < self.MIN_FREE_WIDTH
            and usable_width(current_side, row) >= self.MIN_FREE_WIDTH
            for row in nearest_rows
        )
        return inherited and not blocked_switch

class _TemporalReachabilityController(_FeasibleCorridorSideWidthController):
    """Use stacked camera evidence to veto late passes and steer a checked path.

    The active Agent route remains the frozen controller below. This candidate
    only supplements the prior diagnostic when an obstacle is detected; a
    complete corridor still follows the existing full-path planner.
    """

    VERIFIED_PASS_TARGET_SPEED = 30.0
    VERIFIED_PASS_MIN_MARGIN = 0.5
    OCCLUDED_OBSTACLE_TARGET_SPEED = 24.0
    UNCERTAIN_NEAR_BRAKE_FRAMES = 2
    UNCERTAIN_NEAR_CRAWL_GAS = 0.035

    def __init__(self, *, cruise_speed: float = 68.0) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._temporal_track = None
        self._temporal_assessment = None
        self._temporal_stack = None
        self._temporal_brake_frames = 0
        self._temporal_brake_identity = None

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._temporal_track = None
        self._temporal_assessment = None
        self._temporal_stack = None
        self._temporal_brake_frames = 0
        self._temporal_brake_identity = None

    def act(self, observation) -> np.ndarray:
        self._temporal_assessment = None
        frame = self._frame(observation)
        self._temporal_stack = (
            np.asarray(observation, dtype=np.float32)
            if frame is not None else None
        )
        try:
            return super().act(observation)
        finally:
            self._temporal_stack = None

    def _lost_road_action(self) -> np.ndarray:
        self._temporal_track = None
        self._temporal_assessment = None
        self._temporal_brake_frames = 0
        self._temporal_brake_identity = None
        return super()._lost_road_action()

    def _nearest_obstacle(
        self,
        frame: np.ndarray,
        centers: dict[int, float],
        spans: dict[int, tuple[float, float]] | None = None,
    ) -> tuple[float, float, float] | None:
        obstacle = super()._nearest_obstacle(frame, centers, spans)
        current_bbox = self._corridor_bbox
        stack_bboxes = [None, None, None, current_bbox]
        if self._temporal_track is None and self._temporal_stack is not None:
            for index, past_frame in enumerate(self._temporal_stack[:3]):
                past_centers = self._road_centers(past_frame)
                past_obstacle = _StableCompletionController._nearest_obstacle(
                    self, past_frame, past_centers,
                )
                if past_obstacle is not None:
                    stack_bboxes[index] = self._component_bbox(
                        past_frame, past_obstacle[1], past_obstacle[0],
                    )
        self._temporal_track = _track_obstacle_stack(
            self._temporal_track, stack_bboxes,
        )
        self._temporal_assessment = None
        if obstacle is not None and self._temporal_track is not None and current_bbox is not None:
            self._temporal_assessment = _assess_temporal_pass(
                self._temporal_track,
                self._corridor_edges,
                committed_side=int(self._obstacle_side),
                last_steer=self._last_steer,
            )
        return obstacle

    def _allow_obstacle_side_switch(
        self, *, obstacle_y: float, obstacle_x: float, candidate_side: float,
    ) -> bool:
        inherited = super()._allow_obstacle_side_switch(
            obstacle_y=obstacle_y, obstacle_x=obstacle_x,
            candidate_side=candidate_side,
        )
        assessment = self._temporal_assessment
        track = self._temporal_track
        bbox = self._corridor_bbox
        if (
            not inherited or assessment is None or track is None or bbox is None
            or self._obstacle_side == 0.0
            or candidate_side == self._obstacle_side
            or self._corridor_candidate(-1.0) is not None
            or self._corridor_candidate(1.0) is not None
        ):
            return inherited
        if not getattr(assessment, "road_rows", ()):
            # No contiguous nearby road evidence exists to contradict the
            # established side choice; keep the parent's escape transition.
            return inherited
        candidate_width = (
            getattr(assessment, "left_width_px", None) if candidate_side < 0
            else getattr(assessment, "right_width_px", None)
        )
        current_width = (
            getattr(assessment, "left_width_px", None) if self._obstacle_side < 0
            else getattr(assessment, "right_width_px", None)
        )
        if (
            candidate_width is not None and current_width is not None
            and candidate_width < self.MIN_FREE_WIDTH
            and current_width >= self.MIN_FREE_WIDTH
        ):
            return False
        if (
            candidate_width is not None and current_width is not None
            and current_width < 0.0 and candidate_width >= 3.0
        ):
            # The tracked obstacle has closed the committed side. Preserve
            # the parent controller's switch to the only visibly open side,
            # even when the full temporal path cannot yet be certified.
            return inherited
        if assessment.selected_side == int(self._obstacle_side):
            if assessment.selected_target_x is not None:
                return False
            if track.observations >= 2 and bbox[3] >= 40:
                return False
            return inherited
        if assessment.selected_side == int(candidate_side) and not assessment.brake_required:
            return True
        # A close tracked object leaves no time to discover a new side by
        # oscillating the legacy centroid rule on a single apparent opening.
        if track.observations >= 2 and bbox[3] >= 40:
            return False
        return inherited

    def _adjust_obstacle_steering(
        self, *, base_steering: float, obstacle_bias: float, straight: bool,
    ) -> float:
        assessment = self._temporal_assessment
        bbox = self._corridor_bbox
        if (
            assessment is None or bbox is None
            or self._corridor_candidate(-1.0) is not None
            or self._corridor_candidate(1.0) is not None
        ):
            return super()._adjust_obstacle_steering(
                base_steering=base_steering,
                obstacle_bias=obstacle_bias,
                straight=straight,
            )
        target = assessment.selected_target_x
        if target is None:
            target = self._partial_near_target(assessment, bbox)
        if target is None:
            return super()._adjust_obstacle_steering(
                base_steering=base_steering,
                obstacle_bias=obstacle_bias,
                straight=straight,
            )
        near_y = bbox[3] + 2.0
        urgency = min(1.0, max(0.5, abs(obstacle_bias) / 0.24))
        desired = _temporal_path_steer(target, near_y) * urgency
        self._obstacle_side = float(assessment.selected_side)
        if self._preview_transition_pending and desired * self._last_steer < 0.0:
            self._last_steer = 0.0
        if self._corridor_detection is not None:
            y, x, _road_center = self._corridor_detection
            self._corridor_previous = (x, y, float(assessment.selected_side))
        return float(desired)

    def _partial_near_target(
        self, assessment: _TemporalPassAssessment,
        bbox: tuple[int, int, int, int],
    ) -> float | None:
        """Guide an already chosen side using the visible near-car path.

        This does not certify a full pass or release the uncertain speed cap.
        It only avoids steering toward a visibly blocked opposite side while
        additional obstacle rows come into view.
        """
        track = self._temporal_track
        side = assessment.selected_side
        if track is None or track.observations < 2 or side not in (-1, 1):
            return None
        own_width = assessment.left_width_px if side < 0 else assessment.right_width_px
        other_width = assessment.right_width_px if side < 0 else assessment.left_width_px
        own_shift = (
            assessment.left_shift_per_row if side < 0
            else assessment.right_shift_per_row
        )
        other_shift = (
            assessment.right_shift_per_row if side < 0
            else assessment.left_shift_per_row
        )
        if (
            own_width is None or other_width is None
            or own_shift is None or other_shift is None
            or own_width < 3.0 or own_shift > self.MAX_SHIFT_PER_ROW
            or not (other_width < self.MIN_FREE_WIDTH
                    or other_shift > self.MAX_SHIFT_PER_ROW)
            or len(assessment.road_rows) != 3
        ):
            return None
        near_y = bbox[3] + 2.0
        if near_y >= self.CAR_ROW:
            return None
        if side < 0:
            road_bound = max(
                self._corridor_edges[row][0] + self.ROAD_EDGE_MARGIN
                for row in assessment.road_rows
            )
            obstacle_bound = bbox[0] - self.OBSTACLE_EDGE_MARGIN
        else:
            road_bound = min(
                self._corridor_edges[row][1] - self.ROAD_EDGE_MARGIN
                for row in assessment.road_rows
            )
            obstacle_bound = bbox[2] + self.OBSTACLE_EDGE_MARGIN
        target = 0.5 * (road_bound + obstacle_bound)
        if abs(target - self.IMAGE_CENTER) > self.MAX_SHIFT_PER_ROW * (self.CAR_ROW - near_y):
            return None
        visible_approach = [
            row for row in range(math.ceil(near_y), 59)
            if row in self._corridor_edges
        ]
        required_near_rows = range(max(math.ceil(near_y), 53), 59)
        if (
            len(required_near_rows) < 2
            or any(row not in self._corridor_edges for row in required_near_rows)
        ):
            return None
        for row in range(max(22, bbox[1] - 2), min(58, math.ceil(near_y)) + 1):
            span = self._corridor_edges.get(row)
            if span is not None and not (
                span[0] + self.ROAD_EDGE_MARGIN
                <= target <= span[1] - self.ROAD_EDGE_MARGIN
            ):
                return None
        for row in visible_approach:
            x = self._corridor_approach_x(row, target, near_y)
            left, right = self._corridor_edges[row]
            if (
                x < left + self.ROAD_EDGE_MARGIN
                or x > right - self.ROAD_EDGE_MARGIN
            ):
                return None
        return target

    def _adjust_target_speed_for_steering(
        self, *, target_speed: float, steering: float, straight: bool,
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        inherited = super()._adjust_target_speed_for_steering(
            target_speed=target_speed, steering=steering,
            straight=straight, obstacle=obstacle,
        )
        assessment = self._temporal_assessment
        track = self._temporal_track
        if (
            obstacle is None and track is not None
            and 0 < track.misses <= 2
            and track.last_seen_bbox[3] < 54.0
        ):
            return min(inherited, self.OCCLUDED_OBSTACLE_TARGET_SPEED)
        if (
            obstacle is None or self._corridor_plan is not None
            or assessment is None or track is None
            or assessment.brake_required
            or assessment.selected_target_x is None
            or assessment.selected_path_margin_px is None
            or assessment.selected_path_margin_px < self.VERIFIED_PASS_MIN_MARGIN
            or track.observations < 2
        ):
            return inherited
        return min(target_speed, self.VERIFIED_PASS_TARGET_SPEED)

    def _adjust_pedals(
        self, *, gas: float, brake: float, straight: bool,
        obstacle: tuple[float, float, float] | None,
    ) -> tuple[float, float]:
        gas, brake = super()._adjust_pedals(
            gas=gas, brake=brake, straight=straight, obstacle=obstacle,
        )
        bbox = self._corridor_bbox
        assessment = self._temporal_assessment
        track = self._temporal_track
        track_identity = getattr(track, "identity", None)
        imminent_uncertain = (
            obstacle is not None and bbox is not None
            and 44 <= bbox[3] < 58
            and assessment is not None and assessment.brake_required
        )
        if not imminent_uncertain:
            tracked_occlusion = (
                obstacle is None and track is not None
                and 0 < track.misses <= 2
                and track_identity == self._temporal_brake_identity
            )
            if not tracked_occlusion:
                self._temporal_brake_frames = 0
                self._temporal_brake_identity = None
            return gas, brake
        if track_identity != self._temporal_brake_identity:
            self._temporal_brake_frames = 0
            self._temporal_brake_identity = track_identity
        self._temporal_brake_frames += 1
        if self._temporal_brake_frames <= self.UNCERTAIN_NEAR_BRAKE_FRAMES:
            return 0.0, max(brake, 0.12)
        if brake > 0.0 and gas <= 0.0:
            return gas, brake
        return min(max(gas, self.UNCERTAIN_NEAR_CRAWL_GAS), 0.04), 0.0

class _SparseRoadBendPreviewController(_TemporalReachabilityController):
    """Anticipate an observed bend only while the nearby road is occluded.

    When local edge rows are visible, the temporal pass planner's measured
    corridor takes priority. A far obstacle on a sharp bend can hide those
    rows before a full corridor exists; in that case, hold a bounded early
    turn from the visible far/near centerline until a checked path appears.
    """

    def __init__(self, *, cruise_speed: float = 68.0) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._preview_track_identity = None
        self._preview_armed = False

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._preview_track_identity = None
        self._preview_armed = False

    def act(self, observation) -> np.ndarray:
        action = super().act(observation)
        if self._temporal_track is None:
            self._preview_track_identity = None
            self._preview_armed = False
        return action

    def _adjust_obstacle_steering(
        self, *, base_steering: float, obstacle_bias: float, straight: bool,
    ) -> float:
        desired = super()._adjust_obstacle_steering(
            base_steering=base_steering, obstacle_bias=obstacle_bias,
            straight=straight,
        )
        centers = self._corridor_centers
        bbox = self._corridor_bbox
        track = self._temporal_track
        if bbox is None or track is None:
            return desired
        if track.identity != self._preview_track_identity:
            self._preview_track_identity = track.identity
            self._preview_armed = False
        near_road_span = self._corridor_edges.get(54)
        near_halfwidth = (
            0.5 * (near_road_span[1] - near_road_span[0])
            if near_road_span is not None else None
        )
        if (
            not self._preview_armed and bbox[3] <= 34
            and self._temporal_assessment is not None
            and not self._temporal_assessment.road_rows
            and 30 in centers and 54 in centers
            and near_halfwidth is not None and near_halfwidth > 1.0
            and abs(0.5 * (bbox[0] + bbox[2]) - centers[30]) <= near_halfwidth - 1.0
            and abs(centers[30] - centers[54]) >= 8.0
        ):
            self._preview_armed = True
        if (
            self._preview_armed and self._corridor_plan is None
            and bbox[3] < 55 and 30 in centers and 54 in centers
        ):
            # Hold the early turn for this tracked object as local edge rows
            # reappear. They alone do not certify a complete passing path.
            bend = centers[30] - centers[54]
            if bend <= -8.0:
                return min(desired, -0.28)
            if bend >= 8.0:
                return max(desired, 0.28)
        return desired

class _ImpactAwareSparseRoadController(_SparseRoadBendPreviewController):
    """Make a short, slew-limited escape after a likely obstacle contact.

    The HUD speed estimate is noisy. Require an unexpected speed drop while
    the same nearby obstacle and chosen side remain visible, then change only
    the steering command. The inherited pedals and sparse-road preview remain
    responsible for speed and early corner entry.
    """

    def __init__(self, *, cruise_speed: float = 68.0) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._reset_impact()

    def _reset_impact(self) -> None:
        self._impact_previous_hud = None
        self._impact_previous_brake = 0.0
        self._impact_previous_box = None
        self._impact_previous_track_identity = None
        self._impact_previous_side = 0.0
        self._impact_escape_frames = 0
        self._impact_escape_identity = None
        self._impact_escape_side = 0.0
        self._impact_recovered_identity = None

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._reset_impact()

    def act(self, observation) -> np.ndarray:
        frame = self._frame(observation)
        hud_speed = self._estimate_speed(frame) if frame is not None else None
        previous_speed = self._impact_previous_hud
        previous_brake = self._impact_previous_brake
        previous_box = self._impact_previous_box
        previous_identity = self._impact_previous_track_identity
        previous_side = self._impact_previous_side
        previous_steer = self._last_steer
        action = super().act(observation)
        bbox = self._corridor_bbox
        track = self._temporal_track
        assessment = self._temporal_assessment
        side = self._obstacle_side
        if track is None:
            self._impact_escape_frames = 0
            self._impact_recovered_identity = None
        elif (
            self._impact_escape_frames == 0
            and track.identity != self._impact_recovered_identity
            and track.identity == previous_identity
            and side == previous_side
            and side in (-1.0, 1.0)
            and assessment is not None
            and assessment.selected_side == side
            and len(assessment.road_rows) == 3
            and self._corridor_plan is None
            and track.observations >= 2
            and previous_speed is not None and hud_speed is not None
            and previous_speed - hud_speed >= 4.5
            and previous_brake < 0.079
            and float(action[2]) < 0.079
            and previous_box is not None and previous_box[3] >= 54
            and bbox is not None and bbox[3] >= 55
        ):
            self._impact_escape_frames = 2
            self._impact_escape_identity = track.identity
            self._impact_escape_side = side
            self._impact_recovered_identity = track.identity
        if self._impact_escape_frames > 0:
            self._impact_escape_frames -= 1
            if (
                bbox is not None and track is not None
                and track.identity == self._impact_escape_identity
                and assessment is not None and len(assessment.road_rows) == 3
                and side == self._impact_escape_side
                and assessment.selected_side == side
                and self._corridor_plan is None
                and float(action[2]) < 0.079
            ):
                own_width = (
                    assessment.left_width_px if side < 0
                    else assessment.right_width_px
                )
                if (
                    own_width is not None and own_width >= 2.0
                    and float(action[0]) * side <= 0.0
                ):
                    proposed = float(np.clip(
                        0.28 * side,
                        previous_steer - self.MAX_STEER_STEP,
                        previous_steer + self.MAX_STEER_STEP,
                    ))
                    if abs(proposed) <= min(self.MAX_STEER, 0.32):
                        action = action.copy()
                        action[0] = proposed
                        self._last_steer = proposed
        self._impact_previous_hud = hud_speed
        self._impact_previous_brake = float(action[2])
        self._impact_previous_box = bbox
        self._impact_previous_track_identity = track.identity if track is not None else None
        self._impact_previous_side = side if track is not None else 0.0
        return action

class _HighSpeedBendPriorityController(_ImpactAwareSparseRoadController):
    """Preserve a visible bend when an uncertified high-speed pass opposes it.

    This is a diagnostic candidate. The live ``Agent`` route remains on the
    established controller until a new frozen evaluation supports promotion.
    """

    BEND_PRIORITY_MIN_HUD_SPEED = 45.0
    BEND_PRIORITY_MIN_SWEEP = 5.0
    BEND_PRIORITY_MIN_STEER = 0.05
    BEND_PRIORITY_MIN_BBOX_BOTTOM = 36

    def _adjust_obstacle_steering(
        self, *, base_steering: float, obstacle_bias: float, straight: bool,
    ) -> float:
        proposed = super()._adjust_obstacle_steering(
            base_steering=base_steering,
            obstacle_bias=obstacle_bias,
            straight=straight,
        )
        assessment = self._temporal_assessment
        bbox = self._corridor_bbox
        centers = self._corridor_centers
        far = next((centers[row] for row in (30, 34, 38) if row in centers), None)
        near = next((centers[row] for row in (54, 50) if row in centers), None)
        if (
            self._corridor_plan is None
            and assessment is not None and assessment.brake_required
            and bbox is not None and bbox[3] >= self.BEND_PRIORITY_MIN_BBOX_BOTTOM
            and (self._impact_previous_hud or 0.0)
            >= self.BEND_PRIORITY_MIN_HUD_SPEED
            and far is not None and near is not None
            and abs(far - near) >= self.BEND_PRIORITY_MIN_SWEEP
            and abs(base_steering) >= self.BEND_PRIORITY_MIN_STEER
            and proposed * base_steering <= 0.0
            and obstacle_bias * base_steering < 0.0
            and (far - near) * base_steering > 0.0
        ):
            return float(base_steering)
        return proposed

class _BoundedSideHoldController(_HighSpeedBendPriorityController):
    """Keep an early, visibly wider pass side for at most three decisions.

    The short latch belongs to the same controller that emits the action. It
    only applies after a full spatial path was seen and that path temporarily
    disappears; it does not invent a newly verified passage. This remains a
    diagnostic candidate until a frozen independent evaluation succeeds.
    """

    SIDE_HOLD_MARGIN_MIN = 1.0
    SIDE_HOLD_MARGIN_MAX = 1.6
    SIDE_HOLD_MIN_WIDTH = 2.0
    SIDE_HOLD_MIN_WIDTH_GAIN = 0.8
    SIDE_HOLD_STEER_FLOOR = 0.07
    SIDE_HOLD_MAX_DECISIONS = 3

    def __init__(self, *, cruise_speed: float = 68.0) -> None:
        self._clear_side_hold()
        super().__init__(cruise_speed=cruise_speed)

    def _clear_side_hold(self) -> None:
        self._side_hold_identity = None
        self._side_hold_side = 0.0
        self._side_hold_decisions = 0

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._clear_side_hold()

    def _lost_road_action(self) -> np.ndarray:
        self._clear_side_hold()
        return super()._lost_road_action()

    def _adjust_obstacle_steering(
        self, *, base_steering: float, obstacle_bias: float, straight: bool,
    ) -> float:
        proposed = float(super()._adjust_obstacle_steering(
            base_steering=base_steering,
            obstacle_bias=obstacle_bias,
            straight=straight,
        ))
        plan = self._corridor_plan
        assessment = self._temporal_assessment
        track = self._temporal_track
        bbox = self._corridor_bbox
        if bbox is None or assessment is None or track is None:
            self._clear_side_hold()
            return proposed

        if (
            plan is not None and 24 <= bbox[3] <= 30
            and self.SIDE_HOLD_MARGIN_MIN <= plan["clearance_px"]
            <= self.SIDE_HOLD_MARGIN_MAX
            and track.observations == 1
            and assessment.brake_required
            and assessment.selected_target_x is None
            and len(assessment.road_rows) == 3
        ):
            side = float(plan["side"])
            own = (assessment.left_width_px if side < 0
                   else assessment.right_width_px)
            other = (assessment.right_width_px if side < 0
                     else assessment.left_width_px)
            if (
                own is not None and other is not None
                and own >= self.SIDE_HOLD_MIN_WIDTH
                and own - other >= self.SIDE_HOLD_MIN_WIDTH_GAIN
            ):
                self._side_hold_identity = track.identity
                self._side_hold_side = side
                self._side_hold_decisions = 0

        if (
            plan is None
            and self._side_hold_identity == track.identity
            and self._side_hold_side in (-1.0, 1.0)
            and self._side_hold_decisions < self.SIDE_HOLD_MAX_DECISIONS
            and 30 <= bbox[3] <= 52
            and len(assessment.road_rows) == 3
        ):
            side = self._side_hold_side
            own = (assessment.left_width_px if side < 0
                   else assessment.right_width_px)
            other = (assessment.right_width_px if side < 0
                     else assessment.left_width_px)
            if (
                own is not None and other is not None
                and own >= self.SIDE_HOLD_MIN_WIDTH
                and own - other >= self.SIDE_HOLD_MIN_WIDTH_GAIN
                and proposed * side < self.SIDE_HOLD_STEER_FLOOR
            ):
                self._side_hold_decisions += 1
                return float(side * self.SIDE_HOLD_STEER_FLOOR)

        if bbox[3] > 52 or track.identity != self._side_hold_identity:
            self._clear_side_hold()
        return proposed

class _ClearRoadRow42DropoutController(_BoundedSideHoldController):
    """Recover an observed clear-road bend when its reference row disappears."""

    def __init__(self, *, cruise_speed: float = 68.0) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._row42_recent_obstacle_valid = False
        self._row42_current_hud_speed = None

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._row42_recent_obstacle_valid = False
        self._row42_current_hud_speed = None

    def _lost_road_action(self) -> np.ndarray:
        # The inherited obstacle latch does not age on lost-road decisions.
        # Invalidate only this correction's evidence, preserving parent state.
        self._row42_recent_obstacle_valid = False
        return super()._lost_road_action()

    def act(self, observation) -> np.ndarray:
        frame = self._frame(observation)
        self._row42_current_hud_speed = (
            None if frame is None else self._estimate_speed(frame)
        )
        return super().act(observation)

    def _adjust_road_steering(
        self,
        *,
        steering: float,
        straight: bool,
        centers: dict[int, float],
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        # Parent act calls this hook only after accepting the visible road.
        if obstacle is not None:
            self._row42_recent_obstacle_valid = True
        if (
            obstacle is None
            and not straight
            and 42 not in centers
            and all(row in centers for row in (54, 50, 46))
            and (
                (
                    self._row42_recent_obstacle_valid
                    and self._obstacle_side != 0.0
                    # Evaluated before the parent's miss increment, so this
                    # retains the first three clear visible-road decisions.
                    and self._obstacle_missing <= 2
                )
                or (
                    self._row42_current_hud_speed is not None
                    and self._row42_current_hud_speed >= 30.0
                )
            )
        ):
            far = self._center_at(42.0, centers)
            near = float(centers[54])
            steering = 0.016 * (far - self.IMAGE_CENTER) + 0.012 * (far - near)
        return super()._adjust_road_steering(
            steering=steering, straight=straight, centers=centers,
            obstacle=obstacle,
        )


class _RobustAgent(_ClearRoadRow42DropoutController):
    """Observe the camera, preserve avoidance, and close clear-road speed error."""

    def __init__(self, model_path=None, *, cruise_speed=80.0, propulsion=0.85,
                 corner_boost=0.45):
        del model_path
        self.STRAIGHT_TARGET_SPEED = float(np.clip(cruise_speed, 40.0, 150.0))
        self.propulsion = float(np.clip(propulsion, 0.0, 1.0))
        self.corner_boost = float(np.clip(corner_boost, 0.0, 1.0))
        self._boost_clear = False
        super().__init__(cruise_speed=self.STRAIGHT_TARGET_SPEED)

    def reset(self, observation=None):
        super().reset(observation)
        self._boost_clear = False

    @classmethod
    def _estimate_speed(cls, frame):
        # The official white HUD bar starts at row 74. Its full height extends
        # above the historical row-77 crop once real speed exceeds ~110.
        # Separate sums preserve the original low-speed floating-point trace
        # exactly when the upper extension is empty.
        mass = float(frame[77:83, 10:13].sum()) + float(frame[74:77, 10:13].sum())
        return float(np.clip((mass - 0.27) / 0.085, 0.0, 300.0))

    def _adjust_road_steering(self, *, steering, straight, centers, obstacle):
        self._boost_clear = bool(obstacle is None and self._obstacle_side == 0.0)
        return super()._adjust_road_steering(steering=steering, straight=straight,
            centers=centers, obstacle=obstacle)

    def _adjust_pedals(self, *, gas, brake, straight, obstacle):
        inherited_gas, inherited_brake = super()._adjust_pedals(
            gas=gas, brake=brake, straight=straight, obstacle=obstacle)
        if not self._boost_clear or obstacle is not None or self._obstacle_side != 0.0:
            return inherited_gas, inherited_brake
        target = self._pace_effective_target
        if target is None or self._pace_speed >= target - 0.5:
            return inherited_gas, inherited_brake
        if inherited_brake > 0.0:
            return inherited_gas, inherited_brake
        clear_exit = bool(getattr(self, "_clear_exit", False))
        requested = self.propulsion if straight or clear_exit else self.corner_boost
        speed_error = max(0.0, target - self._pace_speed)
        gain = min(1.0, max(0.20, speed_error / 12.0))
        if not straight and not clear_exit:
            steering_fraction = abs(self._carry_steer_request) / max(self.MAX_STEER, 1e-6)
            gain *= float(np.sqrt(max(0.25, 1.0 - steering_fraction**2)))
        return max(inherited_gas, requested * gain), 0.0

    def act(self, observation):
        action = super().act(observation)
        return np.clip(action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]).astype(np.float32)

"""Independent metric camera racer; inference needs only NumPy.

The observation camera follows the hull with an affine overhead transform.
Road pixels therefore supply distances, heading and local curvature directly;
no episode identity, simulator import, map or learned checkpoint is used.
"""

import numpy as np


class _MetricAgent:
    PX_X = 1.3608
    PX_Y = 1.701
    CAR_X = 42.0
    CAR_Y = 63.0
    WHEELBASE = 3.24
    HALF_ROAD = 6.6666666667

    def __init__(self, cruise_speed=100.0, lateral_accel=190.0,
                 preview_time=0.12, steer_gain=1.0,
                 low_gas_gate=0.90, high_gas_gate=0.98,
                 braking_accel=100.0, curve_window=12.0, curve_step=4.0):
        parameters = np.asarray([cruise_speed, lateral_accel, preview_time,
                                 steer_gain, low_gas_gate, high_gas_gate,
                                 braking_accel, curve_window, curve_step])
        if not np.isfinite(parameters).all() or np.any(parameters <= 0):
            raise ValueError("controller parameters must be finite and positive")
        if low_gas_gate >= high_gas_gate or high_gas_gate > 1.0:
            raise ValueError("gas demand gates must satisfy 0 < low < high <= 1")
        if not 6.0 <= curve_window <= 32.0 or not 1.0 <= curve_step <= 0.5 * curve_window:
            raise ValueError("curve windows need 6..32 m and overlapping steps of at least 1 m")
        self.cruise_speed = float(min(cruise_speed, 130.0))
        self.lateral_accel = float(lateral_accel)
        self.preview_time = float(preview_time)
        self.steer_gain = float(steer_gain)
        self.low_gas_gate = float(low_gas_gate)
        self.high_gas_gate = float(high_gas_gate)
        self.braking_accel = float(braking_accel)
        self.curve_window = float(curve_window)
        self.curve_step = float(curve_step)
        self.reset()

    def reset(self, observation=None):
        self.last_steer = 0.0
        self.last_center = self.CAR_X
        self.lost_frames = 0
        self.pass_side = 0.0
        self.pass_missing = 0
        self.last_speed = 0.0
        self.last_target = self.cruise_speed

    @staticmethod
    def _frame(observation):
        pixels = np.asarray(observation)
        if pixels.ndim == 3 and pixels.shape[1:] == (84, 84):
            pixels = pixels[-1]
        if pixels.shape != (84, 84) or not np.isfinite(pixels).all():
            return None
        return pixels.astype(np.float32, copy=False)

    @staticmethod
    def _speed(frame):
        # Official white gauge: width2.1px; 0.042px height per m/s.
        # Preserve the full strip instead of saturating at 80m/s.
        gauge = frame[74:83, 10:13]
        mass = float(np.sum(gauge))
        return float(np.clip((mass - 0.20) / 0.0882, 0.0, 180.0))

    def _road(self, frame):
        asphalt = (frame >= 0.32) & (frame <= 0.51)
        rows, lefts, rights, middles = [], [], [], []
        previous = self.CAR_X
        previous_y = 69.0
        slope = 0.0
        for row in range(69, 3, -2):
            points = np.flatnonzero(asphalt[row])
            if points.size < 5:
                continue
            # Bridge a sprite/skid hole without merging separate road branches.
            cuts = np.flatnonzero(np.diff(points) > 6) + 1
            runs = [run for run in np.split(points, cuts) if run.size >= 5]
            if not runs:
                continue
            expected = previous + slope * (previous_y - row)
            run = min(runs, key=lambda p: abs(0.5 * (p[0] + p[-1]) - expected))
            left, right = float(run[0]), float(run[-1])
            middle = 0.5 * (left + right)
            if abs(middle - expected) > 22.0:
                continue
            # An edge clipped by the image is not the actual asphalt edge.
            half_width = self.HALF_ROAD * self.PX_X * np.sqrt(1.0 + (slope * self.PX_Y / self.PX_X) ** 2)
            if left == 0.0 and right < 83.0:
                middle = right - half_width
            elif right == 83.0 and left > 0.0:
                middle = left + half_width
            if rows:
                derivative = (middle - previous) / max(previous_y - row, 1.0)
                slope = 0.5 * slope + 0.5 * float(np.clip(derivative, -2.0, 2.0))
            rows.append(float(row))
            lefts.append(left)
            rights.append(right)
            middles.append(middle)
            previous, previous_y = middle, float(row)
        if len(rows) < 5:
            return None
        rows = np.asarray(rows)
        centers = np.asarray(middles)
        forward = (self.CAR_Y - rows) / self.PX_Y
        lateral = (centers - self.CAR_X) / self.PX_X
        return forward, lateral, rows, np.asarray(lefts), np.asarray(rights)

    def _obstacle(self, frame, road):
        ahead, lateral, rows, lefts, rights = road
        # Orange grey is ~0.686; green is ~0.63/0.70. Restrict to an
        # interior bounded on both sides by asphalt to reject the shoulder.
        candidates = []
        for row in range(13, 71):
            left = float(np.interp(row, rows[::-1], lefts[::-1]))
            right = float(np.interp(row, rows[::-1], rights[::-1]))
            lo, hi = max(1, int(left + 2)), min(83, int(right - 1))
            if hi - lo < 5:
                continue
            bright = np.flatnonzero((frame[row, lo:hi] >= 0.655) & (frame[row, lo:hi] <= 0.705))
            if bright.size >= 2:
                for run in np.split(bright, np.flatnonzero(np.diff(bright) > 1) + 1):
                    if 2 <= run.size <= 8:
                        x = float(lo + run.mean())
                        if x - left >= 2.0 and right - x >= 2.0:
                            candidates.append((float(row), x, left, right))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])

    def _recover(self):
        self.lost_frames += 1
        desired = np.clip((self.last_center - self.CAR_X) * 0.025, -0.35, 0.35)
        if abs(desired) < 0.04:
            desired = 0.6 * self.last_steer
        self.last_steer += float(np.clip(desired - self.last_steer, -0.24, 0.24))
        if self.lost_frames <= 3:
            return np.asarray([self.last_steer, 0.0, 0.25], dtype=np.float32)
        return np.asarray([self.last_steer, 0.09, 0.0], dtype=np.float32)

    def _curve_speed(self, forward, lateral, speed=0.0):
        """Let present pursuit geometry own clear-road corner speed.

        A five-point quadratic on rasterized asphalt can invent an immediate
        sharp bend. This experimental lane removes that separate minimum;
        act still limits steering and speed by its observed pursuit curvature,
        and preserves the visible-horizon and obstacle braking budgets.
        It provides no braking certificate for curves beyond the target.
        """
        return self.cruise_speed

    def act(self, observation):
        frame = self._frame(observation)
        if frame is None:
            return self._recover()
        road = self._road(frame)
        if road is None:
            return self._recover()
        self.lost_frames = 0
        forward, lateral, rows, lefts, rights = road
        speed = self._speed(frame)
        self.last_speed = speed
        self.last_center = self.CAR_X + self.PX_X * float(np.interp(5.0, forward, lateral))
        preview = float(np.clip(5.5 + self.preview_time * speed, 7.0, 29.0))
        preview = min(preview, max(6.0, float(forward[-1]) - 1.0))
        x_target = float(np.interp(preview, forward, lateral))

        target_speed = self._curve_speed(forward, lateral, speed=speed)
        # A curve running out of view requires an approach speed compatible
        # with the remaining observed distance and available braking force.
        if forward[-1] < 22.0:
            target_speed = min(target_speed, np.sqrt(38.0 ** 2 + 2.0 * 100.0 * max(0.0, forward[-1] - 7.0)))

        obstacle = self._obstacle(frame, road)
        if obstacle is not None:
            row, x, left, right = obstacle
            self.pass_missing = 0
            if self.pass_side == 0.0:
                self.pass_side = -1.0 if x - left >= right - x else 1.0
            obstacle_distance = max(2.0, (self.CAR_Y - row) / self.PX_Y)
            pass_x = 0.5 * (left + right) + self.pass_side * 5.0
            # Vehicle halfwidth+clearance is 2.1 pixels; obstacle radius is
            # 1.63 pixels. Route only through the visible asphalt interior.
            pass_x = float(np.clip(pass_x, left + 2.8, right - 2.8))
            weight = np.clip((30.0 - obstacle_distance) / 18.0, 0.25, 1.0)
            x_target = (pass_x - self.CAR_X) / self.PX_X
            preview = min(preview, max(7.0, obstacle_distance))
            target_speed = min(target_speed, np.sqrt(58.0 ** 2 + 2.0 * 90.0 * max(0.0, obstacle_distance - 7.0)))
        else:
            self.pass_missing += 1
            if self.pass_side and self.pass_missing < 4:
                x_target += self.pass_side * 3.3
                target_speed = min(target_speed, 58.0)
            if self.pass_missing >= 4:
                self.pass_side = 0.0

        pursuit_curvature = 2.0 * x_target / (preview ** 2 + x_target ** 2)
        desired = self.steer_gain * float(np.arctan(self.WHEELBASE * pursuit_curvature))
        steer_limit = min(0.4, np.arctan(self.WHEELBASE * self.lateral_accel / max(speed * speed, 100.0)))
        desired = float(np.clip(desired, -steer_limit, steer_limit))
        self.last_steer += float(np.clip(desired - self.last_steer, -0.24, 0.24))
        target_speed = min(target_speed, np.sqrt(self.lateral_accel / max(abs(pursuit_curvature) * self.steer_gain, 0.001)))
        target_speed = min(target_speed, self.last_target + 2.5)
        self.last_target = float(target_speed)
        error = target_speed - speed
        if error < -2.0:
            gas, brake = 0.0, float(np.clip(-error * 0.018, 0.04, 0.70))
        else:
            gas = float(np.clip(0.30 + 0.055 * error, 0.0, 1.0))
            lateral_demand = speed * speed * abs(float(np.tan(self.last_steer))) / (
                self.WHEELBASE * self.lateral_accel
            )
            if lateral_demand > self.high_gas_gate:
                gas = min(gas, 0.16)
            elif lateral_demand > self.low_gas_gate:
                gas = min(gas, 0.25)
            brake = 0.0
        return np.asarray([self.last_steer, gas, brake], dtype=np.float32)


class _MetricClearAgent(_MetricAgent):
    """Road-only metric control; robust detection owns all obstacle memory."""

    def _obstacle(self, frame, road):
        return None


class _BaseAgent:
    """Age both visual controllers, then route clear and uncertain frames."""

    def __init__(self, cruise_speed=100.0, lateral_accel=190.0,
                 preview_time=.12, steer_gain=1.0, robust_cruise_speed=100.0,
                 propulsion=1.0, corner_boost=.50, unstable_sweep=40.0,
                 low_gas_gate=.90, high_gas_gate=.98,
                 braking_accel=100.0, curve_window=12.0, curve_step=4.0):
        self._metric = _MetricClearAgent(cruise_speed=cruise_speed,
            lateral_accel=lateral_accel, preview_time=preview_time,
            steer_gain=steer_gain, low_gas_gate=low_gas_gate,
            high_gas_gate=high_gas_gate, braking_accel=braking_accel,
            curve_window=curve_window, curve_step=curve_step)
        self._robust = _RobustAgent(cruise_speed=robust_cruise_speed,
            propulsion=propulsion, corner_boost=corner_boost)
        self.unstable_sweep = float(unstable_sweep)
        self.reset()

    def reset(self, observation=None):
        self._metric.reset(observation)
        self._robust.reset(observation)
        self.mode = 'robust'
        self._lost_frames = 0
        self.last_speed = self.last_target = self.last_steer = 0.0

    def act(self, observation):
        metric_action = self._metric.act(observation)
        robust_action = self._robust.act(observation)
        frame = self._metric._frame(observation)
        road = None if frame is None else self._metric._road(frame)
        robust = self._robust
        hazard = (robust._temporal_track is not None
                  or robust._corridor_bbox is not None
                  or robust._obstacle_side != 0.0)
        unstable = not robust.road_visible or road is None
        if not unstable:
            forward, lateral, _, _, _ = road
            centers = robust._road_centers(frame)
            unstable = (forward[-1] < 22.0 or 42 not in centers
                        or abs(float(np.interp(5.0, forward, lateral))) > 4.5
                        or robust._road_sweep(centers) > self.unstable_sweep)
        self.mode = 'robust' if hazard or unstable else 'metric'
        action = robust_action if self.mode == 'robust' else metric_action
        assessment = robust._temporal_assessment
        uncertain = hazard and (robust._corridor_plan is None
                                 or (assessment is not None and assessment.brake_required))
        target = robust._pace_effective_target if self.mode == 'robust' else self._metric.last_target
        if self.mode == 'robust' and not uncertain and robust.road_visible and road is not None:
            self.mode = 'guided'
            speed = max(self._metric._speed(frame), float(robust._pace_speed))
            action = self._steering_pedals(metric_action, steer=float(robust_action[0]),
                speed=speed, lateral_accel=self._metric.lateral_accel,
                low_gas_gate=self._metric.low_gas_gate,
                high_gas_gate=self._metric.high_gas_gate)
            target = self._metric.last_target
            if hazard:
                robust_target = robust._pace_effective_target
                if robust_target is not None and np.isfinite(robust_target):
                    target = min(float(target), max(0.0, float(robust_target)))
                action = self._hazard_pedals(action, robust_action, speed=speed, target=target)
        if frame is not None and not robust.road_visible:
            self._lost_frames += 1
            if self._lost_frames > 3:
                self.mode = 'recovery'
                action = metric_action.copy()
                action[1] = min(float(action[1]), .09)
                action[2] = 0.0
        else:
            self._lost_frames = 0
        action = np.clip(action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]).astype(np.float32)
        speed = max(float(self._metric.last_speed), float(robust._pace_speed))
        action = self._traction_pedals(action, speed=speed)
        # Each controller advanced, but only this steering reaches the car.
        emitted = float(action[0])
        self._metric.last_steer = emitted
        self._robust._last_steer = emitted
        self.last_steer = emitted
        self.last_speed = float(self._metric.last_speed)
        if self.mode == 'recovery':
            target = self._metric.last_target
        self.last_target = 0.0 if target is None else float(target)
        return action

    @staticmethod
    def _traction_pedals(action, *, speed):
        """Reserve rear-wheel lateral grip while a visible turn accelerates.

        Official engine force is approximately 40000*gas/speed per driven
        wheel, with a 400 N tire circle. A 0.8 propulsion reserve and 210 m/s²
        lateral budget are below the measured flat-asphalt 219 upper bound.
        This quasistatic camera-speed bound is experimental, not a sideslip
        observer or a dynamic tire certificate.
        """
        action = np.asarray(action, dtype=np.float32).copy()
        if action[2] > 0.0:
            action[1] = 0.0
        elif speed >= 25.0 and abs(float(action[0])) > .005:
            lateral = speed * speed * abs(float(np.tan(action[0]))) / 3.24
            remaining = np.sqrt(max(0.0, 1.0 - (lateral / 210.0) ** 2))
            action[1] = min(float(action[1]), float(np.clip(.008 * speed * remaining, 0.0, 1.0)))
            # The instantaneous force budget alone spun while accelerating
            # into a .06-rad bend. Gas .3 was stable from70 and100m/s in the
            # official-Car characterization; .5+ was not. Anticipate that
            # measured high-speed regime instead of waiting for speed to rise.
            projected_lateral = 100.0 ** 2 * abs(float(np.tan(action[0]))) / 3.24
            if projected_lateral >= 130.0:
                action[1] = min(float(action[1]), .30)
        return action

    @staticmethod
    def _hazard_pedals(action, robust_action, *, speed, target):
        action = np.asarray(action, dtype=np.float32).copy()
        action[1] = min(float(action[1]), float(robust_action[1]))
        action[2] = max(float(action[2]), float(robust_action[2]))
        if target is not None and np.isfinite(target):
            error = float(target) - float(speed)
            if error < -2.0:
                action[2] = max(float(action[2]), float(np.clip(-error * .018, .04, .7)))
            else:
                action[1] = min(float(action[1]), float(np.clip(.3 + .055 * error, 0.0, 1.0)))
        if action[2] > 0.0:
            action[1] = 0.0
        return action

    @staticmethod
    def _steering_pedals(metric_action, *, steer, speed, lateral_accel,
                         low_gas_gate=.50, high_gas_gate=.75):
        action = np.asarray(metric_action, dtype=np.float32).copy()
        action[0] = steer
        tangent = abs(float(np.tan(steer)))
        speed_ceiling = float(np.sqrt(lateral_accel * 3.24 / max(tangent, .003)))
        if speed > speed_ceiling:
            action[1] = 0.0
            action[2] = max(float(action[2]), float(np.clip(.018 * (speed - speed_ceiling), .04, .7)))
        else:
            demand = speed * speed * tangent / (3.24 * lateral_accel)
            if demand > high_gas_gate:
                action[1] = min(float(action[1]), .16)
            elif demand > low_gas_gate:
                action[1] = min(float(action[1]), .25)
        if action[2] > 0.0:
            action[1] = 0.0
        return action


class _PassAgent(_BaseAgent):
    """Try an ego-anchored metric pass only inside the currently visible road.

    The geometric checks use an oriented conservative hull and the official
    obstacle radius. They establish clearance in an interpolated camera model,
    not complete tire, visibility or steering-lag safety on the real track.
    Unchecked frames retain the inherited obstacle behavior.
    """

    def __init__(self, *args, pass_separation=4.0, pass_cruise=85.0, **kwargs):
        if (not np.isfinite([pass_separation, pass_cruise]).all()
                or not 3.5 <= pass_separation <= 5.0
                or not 30.0 <= pass_cruise <= 100.0):
            raise ValueError("pass separation needs3.5..5m and cruise30..100m/s")
        self.pass_separation = float(pass_separation)
        self.pass_cruise = float(pass_cruise)
        super().__init__(*args, **kwargs)

    def reset(self, observation=None):
        super().reset(observation)
        self.last_pass_plan = None
        self._pass_side = 0.0
        self._pass_misses = 0

    @staticmethod
    def _hull_road_margin(y, x, heading, ahead, lefts, rights):
        """Check polygon extrema at all piecewise-linear road breakpoints."""
        sine, cosine = np.sin(heading), np.cos(heading)
        longitudinal = np.asarray([-2.4, -2.4, 2.6, 2.6])
        # Front wheel center1.1m and its rotated .28x.54m body reach1.568m
        # at the official .4-rad steering limit. This full-vehicle envelope
        # covers that footprint in addition to the narrower painted hull.
        lateral = np.asarray([-1.6, 1.6, 1.6, -1.6])
        vx = x + longitudinal*sine + lateral*cosine
        vy = y + longitudinal*cosine - lateral*sine
        if vy.min() < ahead[0] or vy.max() > ahead[-1]:
            return None
        # A polygon side and an observed-road boundary are both linear between
        # their combined breakpoints, so minima occur at these checked heights.
        heights = np.unique(np.r_[vy, ahead[(ahead >= vy.min()) & (ahead <= vy.max())]])
        intersections = np.full((4, len(heights)), np.nan)
        for i in range(4):
            j = (i+1) % 4
            dy = vy[j]-vy[i]
            if abs(dy) < 1e-9:
                continue
            valid = (heights >= min(vy[i],vy[j])-1e-9) & (heights <= max(vy[i],vy[j])+1e-9)
            intersections[i,valid] = vx[i]+(heights[valid]-vy[i])*(vx[j]-vx[i])/dy
        if np.any(~np.isfinite(intersections).any(axis=0)):
            return None
        low = np.nanmin(intersections,axis=0)
        high = np.nanmax(intersections,axis=0)
        margin = min(float(np.min(low-np.interp(heights,ahead,lefts))),
                     float(np.min(np.interp(heights,ahead,rights)-high)))
        return margin

    def _curved_pass(self, road, bbox):
        ahead, side, _, pixel_lefts, pixel_rights = road
        obstacle_x = (.5*(bbox[0]+bbox[2])-42.0)/1.3608
        obstacle_y = (63.0-.5*(bbox[1]+bbox[3]))/1.701
        if not 4.0 <= obstacle_y <= 27.0:
            return None
        end = obstacle_y+3.0
        if ahead[0] > -2.4 or ahead[-1] < end+2.7:
            return None
        fit = (ahead >= -2.0) & (ahead <= end+3.0)
        if np.count_nonzero(fit) < 10 or np.max(np.diff(ahead[fit])) > 2.5:
            return None
        coefficients = np.polyfit(ahead[fit], side[fit], 3)
        residual = abs(np.polyval(coefficients,ahead[fit])-side[fit])
        # A branch jump or sparse hairpin is not a single certified road model.
        if np.max(residual) > 1.5:
            return None
        derivative = np.polyder(coefficients)
        second = np.polyder(derivative)
        initial = -float(np.polyval(coefficients,0.0))
        initial_derivative = -float(np.polyval(derivative,0.0))
        bend_slope = float(np.polyval(derivative,obstacle_y))
        switch = max(3.0, obstacle_y-4.0)
        ys = np.unique(np.r_[np.arange(0.0,end,.35),end])
        t = np.minimum(1.0,ys/switch)
        h00,h10,h01 = 2*t**3-3*t**2+1, t**3-2*t**2+t, -2*t**3+3*t**2
        h00d,h10d,h01d = 6*t*t-6*t, 3*t*t-4*t+1, -6*t*t+6*t
        h00dd,h10dd,h01dd = 12*t-6, 6*t-4, -12*t+6
        lefts = (pixel_lefts-42.0)/1.3608
        rights = (pixel_rights-42.0)/1.3608
        choices = []
        for passing_side in (-1.0,1.0):
            destination = obstacle_x+passing_side*self.pass_separation*np.sqrt(1+bend_slope*bend_slope)
            goal = destination-float(np.polyval(coefficients,obstacle_y))
            offset = h00*initial+h10*switch*initial_derivative+h01*goal
            xs = np.polyval(coefficients,ys)+offset
            slopes = np.polyval(derivative,ys)+(h00d*initial+h10d*switch*initial_derivative+h01d*goal)/switch
            curves = np.polyval(second,ys)+(h00dd*initial+h10dd*switch*initial_derivative+h01dd*goal)/switch**2
            slopes[ys >= switch] = np.polyval(derivative,ys[ys >= switch])
            curves[ys >= switch] = np.polyval(second,ys[ys >= switch])
            xs[0],slopes[0] = 0.0,0.0
            headings = np.arctan(slopes)
            curvature = abs(curves)/(1+slopes*slopes)**1.5
            peak = float(curvature.max())
            if peak > np.tan(.4)/3.24:
                continue
            # Closest separation on the complete sampled polyline, including
            # between pose samples, is more conservative than checking only
            # the one image row containing the circle center.
            delta_x,delta_y = np.diff(xs),np.diff(ys)
            fraction = np.clip(((obstacle_x-xs[:-1])*delta_x+(obstacle_y-ys[:-1])*delta_y)
                               /(delta_x*delta_x+delta_y*delta_y),0.0,1.0)
            separation = float(np.min(np.hypot(xs[:-1]+fraction*delta_x-obstacle_x,
                                              ys[:-1]+fraction*delta_y-obstacle_y)))
            if separation < 3.5:
                continue
            dx,dy = obstacle_x-xs,obstacle_y-ys
            lat = dx*np.cos(headings)-dy*np.sin(headings)
            long = dx*np.sin(headings)+dy*np.cos(headings)
            outside_lat = np.maximum(abs(lat)-1.6,0.0)
            outside_long = np.maximum(np.maximum(long-2.6,-long-2.4),0.0)
            obstacle_margin = float(np.min(np.hypot(outside_lat,outside_long)-1.2))
            if obstacle_margin < .75:
                continue
            margins = [self._hull_road_margin(y,x,heading,ahead,lefts,rights)
                       for y,x,heading in zip(ys,xs,headings)]
            if any(margin is None for margin in margins):
                continue
            road_margin = min(margins)
            if road_margin < .30:
                continue
            score = road_margin-4*peak+(.40 if passing_side == self._pass_side else 0.0)
            choices.append((score, {"side":passing_side,"x":xs,"y":ys,"heading":headings,
                                   "road_margin":float(road_margin),"obstacle_margin":obstacle_margin,
                                   "min_separation":separation,"max_curvature":peak,
                                   "obstacle_distance":obstacle_y}))
        return max(choices,key=lambda choice:choice[0])[1] if choices else None

    def act(self, observation):
        previous_emitted = self.last_steer
        inherited = super().act(observation)
        self.last_pass_plan = None
        frame = self._metric._frame(observation)
        road = None if frame is None else self._metric._road(frame)
        bbox = self._robust._corridor_bbox
        if road is None or bbox is None:
            self._pass_misses += 1
            if self._pass_misses > 4:
                self._pass_side = 0.0
            return inherited
        self._pass_misses = 0
        plan = self._curved_pass(road,bbox)
        if plan is None:
            return inherited
        speed = max(float(self._metric.last_speed),float(self._robust._pace_speed))
        preview = min(float(np.clip(5.5+self._metric.preview_time*speed,7.0,22.0)),
                      max(4.0,plan["obstacle_distance"]),float(plan["y"][-1]))
        x = float(np.interp(preview,plan["y"],plan["x"]))
        pursuit = 2*x/(preview*preview+x*x)
        limit = min(.4,float(np.arctan(3.24*self._metric.lateral_accel/max(speed*speed,100.0))))
        desired = float(np.clip(np.arctan(3.24*pursuit),-limit,limit))
        steer = float(previous_emitted+np.clip(desired-previous_emitted,-.24,.24))
        target = min(self.pass_cruise,self._metric.cruise_speed,
                     float(np.sqrt(self._metric.lateral_accel/max(plan["max_curvature"],.001))),
                     float(np.sqrt(self._metric.lateral_accel/max(abs(pursuit),.001))),
                     float(np.sqrt(3.24*self._metric.lateral_accel/max(abs(np.tan(steer)),.003))))
        error = target-speed
        if error < -2:
            gas,brake = 0.0,float(np.clip(-error*.018,.04,.7))
        else:
            gas,brake = float(np.clip(.3+.055*error,0.0,1.0)),0.0
        action = self._traction_pedals([steer,gas,brake],speed=speed)
        self.mode = "curved_pass"
        self.last_pass_plan = plan
        self._pass_side = plan["side"]
        self.last_target = float(target)
        self.last_steer = float(action[0])
        self._metric.last_steer = self._robust._last_steer = self.last_steer
        self._robust._obstacle_side = self._pass_side
        return action


class _CommittedHazardController(_PassAgent):
    """Track an accepted camera-local pass instead of resetting its curve.

    A current, consistently associated circle supplies translation; the legal
    red yaw HUD supplies rotation. Reference geometry is rechecked each frame.
    These camera checks do not certify tire dynamics, steering lag or unseen
    branches. Inconsistent observations retain the frozen V2 fallback.
    """

    def reset(self, observation=None):
        super().reset(observation)
        self._committed = None
        self._commit_circle = None
        self._camera_previous_speed = None
        self._camera_previous_yaw = None
        self.last_yaw = 0.0
        self.last_commit_rejection = None

    @staticmethod
    def _camera_yaw(frame):
        # Official red bar: pivot63, height4.2px, width1.68px per rightward
        # rad/s. Use all antialiased area, excluding the boundary at row73.
        crop = frame[75:81,51:83]
        weights = np.where(crop <= .37,crop,0.0)
        mass = float(weights.sum())
        if mass < .03:
            return 0.0
        center = float((weights*np.arange(51,83)).sum()/mass)
        magnitude = mass/(4.2*.299*1.68)
        return float(np.clip(magnitude if center >= 62.5 else -magnitude,-8.0,8.0))

    @staticmethod
    def _camera_transform(x,y,turn,translation_x,translation_y):
        cosine,sine = np.cos(turn),np.sin(turn)
        dx,dy = np.asarray(x)-translation_x,np.asarray(y)-translation_y
        return cosine*dx-sine*dy,sine*dx+cosine*dy

    @staticmethod
    def _circle_center(bbox):
        return np.asarray([(.5*(bbox[0]+bbox[2])-42.0)/1.3608,
                           (63.0-.5*(bbox[1]+bbox[3]))/1.701])

    @staticmethod
    def _with_curvature(plan):
        plan = dict(plan)
        arc = np.r_[0.0,np.cumsum(np.hypot(np.diff(plan["x"]),np.diff(plan["y"])))]
        plan["curvature"] = np.gradient(np.unwrap(plan["heading"]),arc)
        return plan

    def _transport_reference(self,road,bbox,speed,yaw):
        self.last_commit_rejection = None
        old = self._committed
        if (old is None or self._commit_circle is None
                or self._camera_previous_speed is None
                or self._camera_previous_yaw is None):
            return None
        turn = .04*(yaw+self._camera_previous_yaw)
        distance = .04*(speed+self._camera_previous_speed)
        if abs(turn) > .5 or abs(speed-self._camera_previous_speed) > 35.0:
            self.last_commit_rejection = "inconsistent HUD motion"
            return None
        if abs(turn) < 1e-6:
            tx,ty = 0.0,distance
        else:
            radius = distance/turn
            tx,ty = radius*(1-np.cos(turn)),radius*np.sin(turn)
        predicted = np.asarray(self._camera_transform(
            self._commit_circle[0],self._commit_circle[1],turn,tx,ty))
        circle = self._circle_center(bbox)
        if np.linalg.norm(circle-predicted) > 2.0:
            self.last_commit_rejection = "circle association disagrees with HUD motion"
            return None
        # Correct the translation from the observed same circle. This admits
        # lateral motion without assuming that velocity equals body heading.
        ox,oy = self._camera_transform(old["x"]-self._commit_circle[0],
                                     old["y"]-self._commit_circle[1],turn,0.0,0.0)
        xs,ys = ox+circle[0],oy+circle[1]
        headings = old["heading"]-turn
        if np.any(np.diff(ys) <= .01):
            self.last_commit_rejection = "reference folds in current forward coordinates"
            return None
        if ys[0] > 1.0 or ys[-1] < 4.0:
            self.last_commit_rejection = "reference does not cover current ego and preview"
            return None
        x_error = float(np.interp(0.0,ys,xs))
        heading_error = float(np.interp(0.0,ys,headings))
        if abs(x_error) > 1.5 or abs(heading_error) > .45:
            self.last_commit_rejection = "actual ego departed the committed reference"
            return None
        # The passed prefix is no longer part of the current future path.
        # Interpolate its exact y0 crossing without reanchoring x or heading.
        future = ys > 0.0
        xs = np.r_[x_error,xs[future]]
        curves = np.r_[np.interp(0.0,ys,old["curvature"]),old["curvature"][future]]
        headings = np.r_[heading_error,headings[future]]
        ys = np.r_[0.0,ys[future]]
        ahead,side,_,pixel_lefts,pixel_rights = road
        if (len(ahead) < 10 or np.max(np.diff(ahead)) > 2.5
                or ahead[0] > -2.4 or ahead[-1] < ys[-1]+2.7):
            self.last_commit_rejection = "insufficient current road support"
            return None
        fit = (ahead >= -2.0) & (ahead <= circle[1]+6.0)
        if np.count_nonzero(fit) < 10 or np.max(np.diff(ahead[fit])) > 2.5:
            self.last_commit_rejection = "insufficient current road support"
            return None
        coefficients = np.polyfit(ahead[fit],side[fit],3)
        if np.max(abs(np.polyval(coefficients,ahead[fit])-side[fit])) > 1.5:
            self.last_commit_rejection = "ambiguous current road model"
            return None
        lefts,rights = (pixel_lefts-42.0)/1.3608,(pixel_rights-42.0)/1.3608
        margins = [self._hull_road_margin(0.0,0.0,0.0,ahead,lefts,rights)]
        margins.extend(self._hull_road_margin(y,x,h,ahead,lefts,rights)
                       for y,x,h in zip(ys,xs,headings))
        if any(m is None for m in margins) or min(margins) < .30:
            self.last_commit_rejection = "current full vehicle road margin"
            return None
        dx,dy = np.diff(xs),np.diff(ys)
        fractions = np.clip(((circle[0]-xs[:-1])*dx+(circle[1]-ys[:-1])*dy)
                            /(dx*dx+dy*dy),0.0,1.0)
        separation = float(np.min(np.hypot(xs[:-1]+fractions*dx-circle[0],
                                          ys[:-1]+fractions*dy-circle[1])))
        # Include the actual ego footprint independently of reference poses.
        cx,cy = circle[0]-np.r_[0.0,xs],circle[1]-np.r_[0.0,ys]
        hs = np.r_[0.0,headings]
        lateral = cx*np.cos(hs)-cy*np.sin(hs)
        longitudinal = cx*np.sin(hs)+cy*np.cos(hs)
        outside_lat = np.maximum(abs(lateral)-1.6,0.0)
        outside_long = np.maximum(np.maximum(longitudinal-2.6,-longitudinal-2.4),0.0)
        obstacle_margin = float(np.min(np.hypot(outside_lat,outside_long)-1.2))
        if separation < 3.5 or obstacle_margin < .75:
            self.last_commit_rejection = "current circle clearance"
            return None
        plan = dict(old,x=xs,y=ys,heading=headings,curvature=curves,
                    road_margin=float(min(margins)),obstacle_margin=obstacle_margin,
                    min_separation=separation,obstacle_distance=float(circle[1]))
        return plan

    def _track_reference(self,plan,speed,yaw,previous_emitted):
        ys,xs = plan["y"],plan["x"]
        preview = min(float(np.clip(5.5+self._metric.preview_time*speed,7.0,22.0)),
                      max(4.0,plan["obstacle_distance"]),float(ys[-1]))
        x = float(np.interp(preview,ys,xs))
        pursuit = 2*x/(preview*preview+x*x)
        error_x = float(np.interp(0.0,ys,xs))
        error_heading = float(np.interp(0.0,ys,plan["heading"]))
        curvature = float(np.interp(min(1.5,ys[-1]),ys,plan["curvature"]))
        # Signed feedforward plus bounded heading/cross-track/yaw feedback.
        # HUD yaw is body heading rate, not a privileged trajectory/slip field.
        desired = (np.arctan(3.24*curvature)+.65*error_heading
                   +np.arctan(2.0*error_x/max(speed,20.0))
                   +.20*3.24*(speed*curvature-yaw)/max(speed,20.0))
        limit = min(.4,float(np.arctan(3.24*self._metric.lateral_accel/max(speed*speed,100.0))))
        desired = float(np.clip(desired,-limit,limit))
        steer = float(previous_emitted+np.clip(desired-previous_emitted,-.24,.24))
        target = min(self.pass_cruise,self._metric.cruise_speed,
                     float(np.sqrt(self._metric.lateral_accel/max(plan["max_curvature"],.001))),
                     float(np.sqrt(self._metric.lateral_accel/max(abs(pursuit),.001))),
                     float(np.sqrt(3.24*self._metric.lateral_accel/max(abs(np.tan(steer)),.003))))
        error = target-speed
        if error < -2.0:
            gas,brake = 0.0,float(np.clip(-error*.018,.04,.7))
        else:
            gas,brake = float(np.clip(.3+.055*error,0.0,1.0)),0.0
        return self._traction_pedals([steer,gas,brake],speed=speed),float(target)

    def act(self,observation):
        previous_emitted = self.last_steer
        inherited = _BaseAgent.act(self,observation)
        self.last_pass_plan = None
        frame = self._metric._frame(observation)
        road = None if frame is None else self._metric._road(frame)
        bbox = self._robust._corridor_bbox
        speed = max(float(self._metric.last_speed),float(self._robust._pace_speed))
        yaw = self._camera_yaw(frame) if frame is not None else 0.0
        self.last_yaw = yaw
        plan = None
        committed = False
        if road is not None and bbox is not None:
            plan = self._transport_reference(road,bbox,speed,yaw)
            committed = plan is not None
            if plan is None:
                plan = self._curved_pass(road,bbox)
                if plan is not None:
                    plan = self._with_curvature(plan)
        self._camera_previous_speed,self._camera_previous_yaw = speed,yaw
        if plan is None:
            self._committed = self._commit_circle = None
            self._pass_misses += 1
            if self._pass_misses > 4:
                self._pass_side = 0.0
            return inherited
        self._pass_misses = 0
        action,target = self._track_reference(plan,speed,yaw,previous_emitted)
        self.mode = "committed_pass" if committed else "curved_pass"
        self.last_pass_plan = plan
        self._committed = plan
        self._commit_circle = self._circle_center(bbox)
        self._pass_side = plan["side"]
        self.last_target = target
        self.last_steer = float(action[0])
        self._metric.last_steer = self._robust._last_steer = self.last_steer
        self._robust._obstacle_side = self._pass_side
        return action


"""Camera-only metric pursuit with measured yaw feedback and grip budgeting.

The HUD supplies speed and yaw.  Road and orange circles are measured from the
same grayscale camera; no simulator state, seed, map or project import is used.
The path is a steering reference, not a collision or reachability certificate.
"""
import numpy as np


class _PathReference:
    PX_X, PX_Y = 1.3608, 1.701
    CAR_X, CAR_Y = 42.0, 63.0
    WHEELBASE, HALF_ROAD = 3.24, 6.6666666667

    def __init__(self, cruise_speed=100.0, lateral_accel=190.0,
                 preview_time=.22, yaw_gain=.35, braking_accel=150.0,
                 obstacle_margin=1.20):
        values = np.asarray([cruise_speed, lateral_accel, preview_time,
                             yaw_gain, braking_accel, obstacle_margin], dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("parameters must be finite and positive")
        self.cruise_speed = min(float(cruise_speed), 110.)
        self.lateral_accel = min(float(lateral_accel), 215.)
        self.preview_time = min(float(preview_time), .6)
        self.yaw_gain = min(float(yaw_gain), 2.)
        self.braking_accel = min(float(braking_accel), 180.)
        self.obstacle_margin = min(float(obstacle_margin), 1.5)
        self.reset()

    def reset(self, observation=None):
        self.last_steer = 0.
        self.last_center = self.CAR_X
        self.last_speed = 0.
        self.last_target = self.cruise_speed
        self.last_yaw = 0.
        self.lost_frames = 0
        self.pass_side = 0
        self.pass_missing = 0
        self.pass_x = 0.
        self.pass_y = None
        self.route_distance = None
        self.spin_frames = 0

    @staticmethod
    def _frame(observation):
        try:
            frame = np.asarray(observation, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            return None
        if frame.ndim == 3 and frame.shape[1:] == (84, 84) and frame.shape[0]:
            frame = frame[-1]
        if frame.shape != (84, 84) or not np.isfinite(frame).all():
            return None
        if float(frame.min()) < 0. or float(frame.max()) > 1.:
            return None
        return frame

    @staticmethod
    def _speed(frame):
        return float(np.clip((float(frame[74:83, 10:13].sum())-.20)/.0882, 0., 180.))

    @staticmethod
    def _yaw(frame):
        # Preserve antialiased red area over the full4.2px HUD height. The
        # green wheel-angle bar ends before51, and the white gauges/text are
        # further left. A lower intensity cutoff would erase partial reds.
        crop = frame[75:81, 51:83]
        weights = np.where(crop <= .37, crop, 0.)
        mass = float(weights.sum())
        if mass < .03:
            return 0.
        middle = float((weights*np.arange(51, 83)).sum())/mass
        magnitude = mass/(4.2*.299*1.68)
        return float(np.clip(magnitude if middle >= 62.5 else -magnitude, -8., 8.))

    @staticmethod
    def _rear_speed(frame):
        # Both purple rear-wheel gauges: empirically calibrated pixel mass
        # per rad/s, then the official .54m radius. Missing bars mean no spin.
        # Row73 straddles the road/HUD boundary and is not a gauge pixel.
        crop = frame[74:83, 19:24]
        mass = float(np.where(crop <= .20, crop, 0.).sum())
        return mass/.01485*.54

    def _road(self, frame):
        asphalt = (frame >= .32) & (frame <= .51)
        rows, lefts, rights, centers = [], [], [], []
        previous, previous_y, slope = self.CAR_X, 69., 0.
        for row in range(69, 3, -2):
            points = np.flatnonzero(asphalt[row])
            if points.size < 5:
                continue
            cuts = np.flatnonzero(np.diff(points) > 6)+1
            runs = [p for p in np.split(points, cuts) if p.size >= 5]
            if not runs:
                continue
            expected = previous+slope*(previous_y-row)
            run = min(runs, key=lambda p: abs(.5*(p[0]+p[-1])-expected))
            left, right = float(run[0]), float(run[-1])
            middle = .5*(left+right)
            if abs(middle-expected) > 22.:
                continue
            half = self.HALF_ROAD*self.PX_X*np.sqrt(1.+(slope*self.PX_Y/self.PX_X)**2)
            if left == 0. and right < 83.:
                middle = right-half
            elif right == 83. and left > 0.:
                middle = left+half
            if rows:
                slope = .5*slope+.5*float(np.clip((middle-previous)/max(previous_y-row, 1.), -2., 2.))
            rows.append(float(row)); lefts.append(left); rights.append(right); centers.append(middle)
            previous, previous_y = middle, float(row)
        if len(rows) < 5:
            return None
        rows = np.asarray(rows)
        return ((self.CAR_Y-rows)/self.PX_Y,
                (np.asarray(centers)-self.CAR_X)/self.PX_X,
                rows, np.asarray(lefts), np.asarray(rights))

    def _circles(self, frame, road):
        _, _, rows, lefts, rights = road
        allowed = np.zeros((84, 84), dtype=bool)
        for row in range(10, 72):
            left = float(np.interp(row, rows[::-1], lefts[::-1]))
            right = float(np.interp(row, rows[::-1], rights[::-1]))
            allowed[row, max(0, int(left+1)):min(84, int(right))] = True
        bright = (frame >= .655) & (frame <= .705) & allowed
        seen = np.zeros_like(bright)
        circles = []
        for y0, x0 in zip(*np.nonzero(bright)):
            if seen[y0, x0]:
                continue
            seen[y0, x0] = True
            stack, pixels = [(int(y0), int(x0))], []
            while stack:
                y, x = stack.pop(); pixels.append((y, x))
                for ny, nx in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
                    if 0 <= ny < 84 and 0 <= nx < 84 and bright[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx] = True; stack.append((ny,nx))
            if not 4 <= len(pixels) <= 48:
                continue
            ys, xs = np.asarray(pixels).T
            width, height = xs.max()-xs.min()+1, ys.max()-ys.min()+1
            aspect = width*self.PX_Y/(height*self.PX_X)
            if not (2 <= width <= 8 and 2 <= height <= 9 and .65 <= aspect <= 1.9):
                continue
            x, y = float(xs.mean()), float(ys.mean())
            surrounding = 0
            for py, px in ((y,x-4), (y,x+4), (y-5,x), (y+5,x)):
                iy, ix = int(round(py)), int(round(px))
                if 0 <= iy < 73 and 0 <= ix < 84 and .32 <= frame[iy,ix] <= .51:
                    surrounding += 1
            if surrounding >= 3:
                circles.append(((self.CAR_Y-y)/self.PX_Y, (x-self.CAR_X)/self.PX_X))
        return sorted(circles)

    @staticmethod
    def _smooth(value):
        value = np.clip(value, 0., 1.)
        return value*value*(3.-2.*value)

    def _route(self, ahead, center, circles, speed, yaw):
        # First decide whether the unshifted reference actually sweeps a circle.
        # This avoids slowing for the many circles safely outside the lane.
        selected = None
        clearance = 1.2+1.2+self.obstacle_margin
        self.route_distance = None
        for distance, x in circles:
            if distance < -3.8:
                continue
            sample = max(0., distance)
            road_x = float(np.interp(sample, ahead, center))
            eps = 2.
            slope = (float(np.interp(sample+eps, ahead, center))-
                     float(np.interp(sample-eps, ahead, center)))/(2.*eps)
            normal_gap = abs(x-road_x)/np.sqrt(1.+slope*slope)
            if not self.pass_side and 0. <= distance <= max(6., 3.8+.08*speed):
                # A near circle may conflict with the road center while
                # already clearing our actual turn. Do not invent a new pass
                # across it when both current motion and preview turn away.
                curvature = yaw/max(speed, 10.)
                phase = float(np.clip(curvature*distance, -.85, .85))
                if abs(curvature) > .0001:
                    motion_x = (1.-np.sqrt(1.-phase*phase))/curvature
                else:
                    motion_x = 0.
                motion_slope = phase/np.sqrt(1.-phase*phase)
                motion_gap = abs(x-motion_x)/np.sqrt(1.+motion_slope*motion_slope)
                preview = float(np.clip(7.+self.preview_time*speed, 8., 32.))
                intent_x = float(np.interp(preview, ahead, center))
                if motion_gap >= clearance and intent_x*x <= 0.:
                    continue
            if normal_gap < clearance:
                selected = (distance, x, road_x, slope)
                break
        imminent = False
        if selected is not None:
            distance, x, road_x, slope = selected
            if self.pass_y is None or distance > self.pass_y+10.:
                self.pass_side = 0
            if not self.pass_side:
                separation = (2.4+max(self.obstacle_margin, 1.3))*np.sqrt(1.+slope*slope)
                choices = []
                for side in (-1, 1):
                    target_x = x+side*separation
                    offset = target_x-road_x
                    if abs(offset)/np.sqrt(1.+slope*slope) <= 4.95:
                        # Prefer a short move from the actual pose when both
                        # road-relative sides have room, especially in bends.
                        choices.append((abs(target_x)+.15*abs(offset), side))
                self.pass_side = min(choices)[1] if choices else (-1 if x >= 0. else 1)
            # Keep the same side while this circle moves behind the camera pose.
            self.pass_x, self.pass_y = x, distance
            self.pass_missing = 0
            imminent = abs(x) < 2.7 and distance < 3.8+.08*speed
        elif self.pass_side and self.pass_y is not None:
            self.pass_missing += 1
            angle = float(np.clip(.08*yaw, -.64, .64))
            x, y = self.pass_x, self.pass_y-.08*speed
            self.pass_x = x*np.cos(angle)-y*np.sin(angle)
            self.pass_y = x*np.sin(angle)+y*np.cos(angle)
            distance, x = self.pass_y, self.pass_x
            road_x = float(np.interp(max(0., distance), ahead, center))
            slope = 0.
            if self.pass_missing > 4 or distance < -5.:
                self.pass_side = 0; self.pass_y = None
        else:
            return center.copy(), False, None
        if not self.pass_side:
            return center.copy(), imminent, None
        distance, x = self.pass_y, self.pass_x
        self.route_distance = distance
        road_x = float(np.interp(max(0., distance), ahead, center))
        slope = (float(np.interp(distance+2., ahead, center))-
                 float(np.interp(distance-2., ahead, center)))/4.
        scale = np.sqrt(1.+slope*slope)
        separation = 2.4+max(self.obstacle_margin, 1.3)
        normal_offset = (x-road_x)/scale+self.pass_side*separation
        # A road-relative corridor rather than a straight chord across a bend.
        normal_offset = float(np.clip(normal_offset, -4.95, 4.95))
        road_slope = (np.interp(ahead+2., ahead, center)-
                      np.interp(ahead-2., ahead, center))/4.
        offsets = normal_offset*np.sqrt(1.+road_slope*road_slope)
        approach = self._smooth(np.maximum(ahead, 0.)/max(4., distance-3.))
        if distance < 4.:
            approach = np.maximum(approach, self._smooth((4.-distance)/7.))
        departure = 1.-self._smooth((ahead-distance-3.)/14.)
        return center+offsets*approach*departure, imminent, distance

    def _steering(self, ahead, path, speed, yaw):
        preview = float(np.clip(7.+self.preview_time*speed, 8., 32.))
        preview = min(preview, max(3., float(ahead[-1])-1.))
        if self.route_distance is not None:
            # Until the rear clears, the preview must remain on the pass
            # corridor rather than aim at its subsequent return to center.
            preview = min(preview, max(7., self.route_distance))
        x = float(np.interp(preview, ahead, path))
        curvature = 2.*x/(preview*preview+x*x)
        feedforward = np.arctan(self.WHEELBASE*curvature)
        feedback = self.yaw_gain*self.WHEELBASE*(speed*curvature-yaw)/max(speed, 20.)
        self.reference_curvature = curvature
        return float(np.clip(feedforward+feedback, -.4, .4))

    def _target(self, ahead, path, speed):
        target = self.cruise_speed
        for distance in (4., 10., 18., 26.):
            if distance > ahead[-1]-2.:
                continue
            selected = (ahead >= max(-2., distance-7.)) & (ahead <= distance+7.)
            if np.count_nonzero(selected) < 5:
                continue
            local = ahead[selected]-distance
            coef = np.polyfit(local, path[selected], 2,
                              w=np.exp(-.5*(local/5.)**2))
            curvature = abs(2.*coef[0])/(1.+coef[1]*coef[1])**1.5
            curve_speed = np.sqrt(self.lateral_accel/max(curvature, .0005))
            usable = max(0., distance-2.6-.08*speed)
            target = min(target, np.sqrt(curve_speed**2+2.*self.braking_accel*usable))
        visibility = max(0., float(ahead[-1])-2.6-.08*speed)
        target = min(target, np.sqrt(45.**2+2.*self.braking_accel*visibility))
        return float(target)

    def _recover(self, speed=None):
        self.lost_frames += 1
        if speed is not None:
            self.last_speed = speed
        desired = float(np.clip((self.last_center-self.CAR_X)*.025, -.3, .3))
        if abs(desired) < .025:
            desired = .6*self.last_steer
        self.last_steer += float(np.clip(desired-self.last_steer, -.24, .24))
        brake = .35 if self.last_speed > 12. or self.lost_frames <= 3 else 0.
        gas = .08 if brake == 0. and self.lost_frames <= 25 else 0.
        return np.asarray([self.last_steer, gas, brake], dtype=np.float32)


class _CorridorReference(_PathReference):
    """Research route minimizing local bending inside visible camera spans.

    Box and circle constraints describe the parsed image, not guaranteed
    vehicle motion. Infeasible observations retain the original route.
    """

    def reset(self, observation=None):
        super().reset(observation)
        self.corridor_road = None
        self.corridor_used = False

    def _road(self, frame):
        result = super()._road(frame)
        self.corridor_road = result
        return result

    def _route(self, ahead, center, circles, speed, yaw):
        original, imminent, distance = super()._route(ahead, center, circles, speed, yaw)
        self.corridor_used = False
        road = self.corridor_road
        if road is None or ahead[-1] < 12. or imminent:
            return original, imminent, distance
        y = np.linspace(0., float(ahead[-1]), 36)
        middle = np.interp(y, ahead, center)
        slope = (np.interp(y+2., ahead, center)-np.interp(y-2., ahead, center))/4.
        inset = 1.9*np.sqrt(1.+slope*slope)
        lower = (np.interp(y, ahead, road[3])-self.CAR_X)/self.PX_X+inset
        upper = (np.interp(y, ahead, road[4])-self.CAR_X)/self.PX_X-inset
        for object_y, object_x in circles:
            if object_y < -4. or object_y > y[-1]+4.:
                continue
            if distance is not None and abs(object_y-distance) < 1.:
                side = self.pass_side
            else:
                side = -1 if object_x >= np.interp(object_y, ahead, original) else 1
            extent = 3.8*np.sqrt(np.maximum(0., 1.-((y-object_y)/4.5)**2))
            active = np.abs(y-object_y) < 4.5
            if side < 0:
                upper[active] = np.minimum(upper[active], object_x-extent[active])
            else:
                lower[active] = np.maximum(lower[active], object_x+extent[active])
        if np.any(lower > upper) or np.any(lower[:2] > 0.) or np.any(upper[:2] < 0.):
            return original, imminent, distance
        # Position and heading are fixed at the actual camera ego pose. The
        # remaining convex box problem uses a weak centering preference.
        lower[:2] = upper[:2] = 0.
        count = len(y)
        identity = np.eye(count)
        d1, d2 = np.diff(identity, axis=0), np.diff(identity, n=2, axis=0)
        hessian = 30.*(d2.T@d2)+.3*(d1.T@d1)+.04*identity
        linear = .04*middle
        step = 1./float(np.max(np.sum(np.abs(hessian), axis=1)))
        x = np.clip(np.interp(y, ahead, original), lower, upper)
        extrapolated, momentum = x.copy(), 1.
        for _ in range(180):
            updated = np.clip(extrapolated-step*(hessian@extrapolated-linear), lower, upper)
            next_momentum = .5*(1.+np.sqrt(1.+4.*momentum*momentum))
            extrapolated = updated+(momentum-1.)/next_momentum*(updated-x)
            x, momentum = updated, next_momentum
        if not np.isfinite(x).all():
            return original, imminent, distance
        self.corridor_used = True
        result = np.interp(ahead, y, x)
        result[ahead < 0.] = 0.
        return result, imminent, distance

    def act(self, observation):
        frame = self._frame(observation)
        if frame is None:
            return self._recover()
        speed, yaw = self._speed(frame), self._yaw(frame)
        road = self._road(frame)
        if road is None or float(road[0][-1]) < 6.:
            return self._recover(speed)
        self.lost_frames = 0
        ahead, center = road[:2]
        self.last_center = self.CAR_X+self.PX_X*float(np.interp(5., ahead, center))
        path, imminent, obstacle_distance = self._route(ahead, center, self._circles(frame, road), speed, yaw)
        desired = self._steering(ahead, path, speed, yaw)
        self.last_steer += float(np.clip(desired-self.last_steer, -.24, .24))
        target = self._target(ahead, path, speed)
        reference_limit = np.sqrt(self.lateral_accel/max(abs(self.reference_curvature), .0005))
        target = min(target, reference_limit, self.last_target+6.)
        actual_lateral = speed*abs(yaw)
        demand = max(actual_lateral, speed*speed*abs(self.reference_curvature))
        # Powered launch slip can be productive on a straight. Tight turns
        # need rear lateral grip, so wheel-speed feedback is gated by demand.
        if self._rear_speed(frame)-speed > 15. and demand > 60.:
            self.spin_frames = 6
        else:
            self.spin_frames = max(0, self.spin_frames-1)
        if actual_lateral > self.lateral_accel:
            target = min(target, self.lateral_accel/max(abs(yaw), .05))
        if imminent:
            target = 0.
        error = target-speed
        if error < -2.:
            gas, brake = 0., float(np.clip(-error*.014, .04, .65))
        else:
            gas, brake = float(np.clip(.18+.06*error, 0., 1.)), 0.
            if demand > 120.:
                gas = min(gas, .30)
            elif demand > 60.:
                gas = min(gas, .60)
            if speed >= 95.:
                gas = min(gas, .16)
            elif speed >= 85.:
                gas = min(gas, .35)
            if self.spin_frames:
                gas = min(gas, .16 if speed > 15. else .08)
        self.last_target, self.last_speed, self.last_yaw = target, speed, yaw
        return np.asarray([self.last_steer, gas, brake], dtype=np.float32)


class _GraphReference(_CorridorReference):
    """Keep only the visible road component linked to the ego neighborhood.

    This graph connects eight neighboring asphalt pixels. It prevents a
    row scanner from jumping across grass to another visible road branch.
    It does not certify an image-center reference as dynamically feasible.
    """

    def _road(self, frame):
        asphalt = (frame >= .32) & (frame <= .51)
        asphalt[73:] = False
        candidates = []
        for row in range(53, 73):
            points = np.flatnonzero(asphalt[row])
            runs = np.split(points, np.flatnonzero(np.diff(points) > 1)+1)
            for run in runs:
                if len(run) >= 8:
                    candidates.extend((row, int(column)) for column in run)
        candidates = np.asarray(candidates, dtype=int)
        if not len(candidates):
            self.corridor_road = None
            return None
        distance = ((candidates[:, 1]-self.CAR_X)/self.PX_X)**2
        distance += ((candidates[:, 0]-self.CAR_Y)/self.PX_Y)**2
        seed = candidates[int(np.argmin(distance))]
        connected = np.zeros((84, 84), dtype=bool)
        connected[tuple(seed)] = True
        for _ in range(168):
            expanded = connected.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    y0, y1 = max(0, dy), min(84, 84+dy)
                    x0, x1 = max(0, dx), min(84, 84+dx)
                    expanded[y0:y1, x0:x1] |= connected[y0-dy:y1-dy, x0-dx:x1-dx]
            expanded &= asphalt
            if np.array_equal(expanded, connected):
                break
            connected = expanded
        road_frame = frame.copy()
        road_frame[asphalt & ~connected] = .9
        return super()._road(road_frame)


class _ConfidenceReference(_GraphReference):
    """Explicit road-support confidence before using a distant road component.

    Gray car fragments can be isolated from visible asphalt when the ego
    approaches an outer road edge. A supported remote component alone does
    not demonstrate support at the camera origin. An isolated nearest ego
    component activates the existing bounded recovery action deliberately.
    """

    def _road(self, frame):
        asphalt = (frame >= .32) & (frame <= .51)
        asphalt[73:] = False
        candidates = np.argwhere(asphalt[53:73])
        if not len(candidates):
            self.corridor_road = None
            return None
        candidates[:, 0] += 53
        distance = ((candidates[:, 1]-self.CAR_X)/self.PX_X)**2
        distance += ((candidates[:, 0]-self.CAR_Y)/self.PX_Y)**2
        seed = candidates[int(np.argmin(distance))]
        connected = np.zeros((84, 84), dtype=bool)
        connected[tuple(seed)] = True
        for _ in range(12):
            expanded = connected.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    y0, y1 = max(0, dy), min(84, 84+dy)
                    x0, x1 = max(0, dx), min(84, 84+dx)
                    expanded[y0:y1, x0:x1] |= connected[y0-dy:y1-dy, x0-dx:x1-dx]
            expanded &= asphalt
            if np.array_equal(expanded, connected):
                break
            connected = expanded
        support = np.argwhere(connected)
        # Require enough connected visible road for five sampled rows. This
        # checks ego support; super() still selects the full major road graph.
        if len(support) < 24 or np.ptp(support[:, 0]) < 8:
            self.corridor_road = None
            return None
        return super()._road(frame)


class _ClearRidgeReference(_ConfidenceReference):
    """Research clear-road ridge; hazards and unsupported ego retain confidence control."""

    @staticmethod
    def _expand(mask):
        result = mask.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                y0, y1 = max(0, dy), min(mask.shape[0], mask.shape[0]+dy)
                x0, x1 = max(0, dx), min(mask.shape[1], mask.shape[1]+dx)
                result[y0:y1, x0:x1] |= mask[y0-dy:y1-dy, x0-dx:x1-dx]
        return result


    def _distance_field(self, frame):
        road = (frame[:73] >= .32) & (frame[:73] <= .51)
        background = ~road
        exterior = np.zeros_like(road)
        exterior[[0, -1], :] = background[[0, -1], :]
        exterior[:, [0, -1]] = background[:, [0, -1]]
        for _ in range(168):
            updated = self._expand(exterior) & background
            if np.array_equal(updated, exterior):
                break
            exterior = updated
        holes = background & ~exterior
        visited = np.zeros_like(road)
        for y, x in np.argwhere(holes):
            if visited[y, x]:
                continue
            stack, component = [(int(y), int(x))], []
            visited[y, x] = True
            while stack:
                py, px = stack.pop()
                component.append((py, px))
                for ny, nx in ((py-1, px), (py+1, px), (py, px-1), (py, px+1)):
                    if 0 <= ny < 73 and 0 <= nx < 84 and holes[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(component) <= 110:
                ys, xs = np.asarray(component).T
                road[ys, xs] = True
        distance = np.where(road, 1000., 0.)
        neighbors = [(dy, dx, np.hypot(dx/self.PX_X, dy/self.PX_Y))
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
        for _ in range(100):
            updated = distance.copy()
            for dy, dx, weight in neighbors:
                y0, y1 = max(0, dy), min(73, 73+dy)
                x0, x1 = max(0, dx), min(84, 84+dx)
                np.minimum(updated[y0:y1, x0:x1],
                           distance[y0-dy:y1-dy, x0-dx:x1-dx]+weight,
                           out=updated[y0:y1, x0:x1])
            if np.max(abs(updated-distance)) < 1e-8:
                break
            distance = updated
        return distance


    def _sample_distance(self, distance, points):
        x = self.CAR_X+points[:, 0]*self.PX_X
        y = self.CAR_Y-points[:, 1]*self.PX_Y
        valid = (x >= 0.) & (x <= 83.) & (y >= 0.) & (y <= 72.)
        x, y = np.clip(x, 0., 83.), np.clip(y, 0., 72.)
        ix, iy = x.astype(int), y.astype(int)
        nx, ny = np.minimum(ix+1, 83), np.minimum(iy+1, 72)
        wx, wy = x-ix, y-iy
        value = ((1.-wx)*(1.-wy)*distance[iy, ix]+wx*(1.-wy)*distance[iy, nx]
                 +(1.-wx)*wy*distance[ny, ix]+wx*wy*distance[ny, nx])
        return np.where(valid, value, 0.)


    def _ridge(self, frame):
        distance = self._distance_field(frame)
        starts = np.column_stack((np.linspace(-8., 8., 65), np.full(65, 3.)))
        initial = self._sample_distance(distance, starts)
        current = starts[int(np.argmax(initial))]
        if float(initial.max()) < 2.:
            return None
        direction = np.asarray([0., 1.])
        points = [current.copy()]
        for _ in range(30):
            normal = np.asarray([direction[1], -direction[0]])
            offsets = np.linspace(-4., 4., 41)
            candidates = current+2.*direction+offsets[:, None]*normal
            score = self._sample_distance(distance, candidates)-.035*offsets**2
            chosen = candidates[int(np.argmax(score))]
            if (self._sample_distance(distance, chosen[None])[0] < 2.
                    or not (-30. < chosen[0] < 30. and -5. < chosen[1] < 37.)):
                break
            heading = chosen-current
            heading /= max(np.linalg.norm(heading), .01)
            direction = .25*direction+.75*heading
            direction /= max(np.linalg.norm(direction), .01)
            current = chosen
            points.append(current.copy())
        path = np.asarray(points)
        return path if len(path) >= 5 else None


    @staticmethod
    def _arc(path):
        return np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]


    def _ridge_target(self, path, speed):
        arc = self._arc(path)
        target = self.cruise_speed
        for at in (4., 10., 18., 26.):
            if at > arc[-1]-2.:
                continue
            selected = (arc >= max(0., at-7.)) & (arc <= at+7.)
            if np.count_nonzero(selected) < 5:
                continue
            s = arc[selected]-at
            weights = np.exp(-.5*(s/5.)**2)
            cx = np.polyfit(s, path[selected, 0], 2, w=weights)
            cy = np.polyfit(s, path[selected, 1], 2, w=weights)
            curve = abs(cx[1]*2.*cy[0]-cy[1]*2.*cx[0])/max((cx[1]**2+cy[1]**2)**1.5, .001)
            cap = np.sqrt(self.lateral_accel/max(curve, .0005))
            usable = max(0., at-2.6-.08*speed)
            target = min(target, np.sqrt(cap**2+2.*self.braking_accel*usable))
        visibility = max(0., arc[-1]-2.6-.08*speed)
        return float(min(target, np.sqrt(45.**2+2.*self.braking_accel*visibility)))

    def reset(self, observation=None):
        super().reset(observation)
        self.mode = "confidence"
        self.circle_cooldown = 0

    def act(self, observation):
        previous_steer, previous_target = self.last_steer, self.last_target
        previous_spin = self.spin_frames
        original = super().act(observation)
        self.mode = "confidence"
        if self.lost_frames:
            return original
        frame, road = self._frame(observation), self.corridor_road
        if frame is None or road is None:
            return original
        if self.pass_side or self._circles(frame, road):
            self.circle_cooldown = 12
            return original
        self.circle_cooldown = max(0, self.circle_cooldown-1)
        if self.circle_cooldown:
            return original
        path = self._ridge(frame)
        if path is None:
            return original
        # Check interpolated chords too: endpoint clearance alone can jump
        # a grass gap between visible road branches. This remains a model.
        dense = []
        for a, b in zip(path[:-1], path[1:]):
            length = float(np.linalg.norm(b-a))
            phase = np.linspace(0., 1., max(2, int(np.ceil(length/.5))+1))
            dense.append(a[None]+phase[:, None]*(b-a)[None])
        field = self._distance_field(frame)
        if min(self._sample_distance(field, np.concatenate(dense))) < 1.9:
            return original
        speed, yaw = self.last_speed, self.last_yaw
        arc = self._arc(path)
        preview = min(float(np.clip(7.+self.preview_time*speed, 8., 32.)),
                      max(3., arc[-1]-1.))
        x = float(np.interp(preview, arc, path[:, 0]))
        y = float(np.interp(preview, arc, path[:, 1]))
        curvature = 2.*x/max(x*x+y*y, 1.)
        desired = np.arctan(self.WHEELBASE*curvature)
        desired += self.yaw_gain*self.WHEELBASE*(speed*curvature-yaw)/max(speed, 20.)
        steer = previous_steer+float(np.clip(np.clip(desired, -.4, .4)-previous_steer, -.24, .24))
        target = min(self._ridge_target(path, speed),
                     np.sqrt(self.lateral_accel/max(abs(curvature), .0005)),
                     previous_target+6.)
        actual_lateral = speed*abs(yaw)
        demand = max(actual_lateral, speed*speed*abs(curvature))
        if self._rear_speed(frame)-speed > 15. and demand > 60.:
            self.spin_frames = 6
        else:
            self.spin_frames = max(0, previous_spin-1)
        if actual_lateral > self.lateral_accel:
            target = min(target, self.lateral_accel/max(abs(yaw), .05))
        error = target-speed
        if error < -2.:
            gas, brake = 0., float(np.clip(-error*.014, .04, .65))
        else:
            gas, brake = float(np.clip(.18+.06*error, 0., 1.)), 0.
            if demand > 120.:
                gas = min(gas, .30)
            elif demand > 60.:
                gas = min(gas, .60)
            if speed >= 95.:
                gas = min(gas, .16)
            elif speed >= 85.:
                gas = min(gas, .35)
            if self.spin_frames:
                gas = min(gas, .16 if speed > 15. else .08)
            projected_speed = min(100., max(self.cruise_speed, speed))
            if speed >= 45. and projected_speed**2*abs(np.tan(steer))/self.WHEELBASE >= 130.:
                gas = min(gas, .30)
        action = np.asarray([steer, gas, brake], dtype=np.float32)
        self.last_steer, self.last_target = float(action[0]), float(target)
        self.reference_curvature = curvature
        self.last_center = self.CAR_X+self.PX_X*float(path[0, 0])
        self.mode = "ridge"
        return action


class _ClearSupportedReference(_ClearRidgeReference):
    """Use only the ridge prefix with camera boundary-distance support.

    A low depth can reflect occlusion, clipped image edges or a wrong ridge,
    rather than a change in actual road width. Rejecting that reference keeps
    the exact inherited confidence action. This is a camera heuristic and
    does not certify tyre sweep or dynamic feasibility.
    """

    def _ridge(self, frame):
        path = super()._ridge(frame)
        if path is None:
            return None
        depth = self._sample_distance(self._distance_field(frame), path)
        unsupported = np.flatnonzero(depth < 5.8)
        if len(unsupported):
            path = path[:int(unsupported[0])]
        return path if len(path) >= 5 else None


class _FastController(_ClearSupportedReference):
    """Include antialiased inner circle pixels with unchanged shape/context gates."""

    def _circles(self, frame, road):
        _, _, rows, lefts, rights = road
        allowed = np.zeros((84, 84), dtype=bool)
        for row in range(10, 72):
            left = float(np.interp(row, rows[::-1], lefts[::-1]))
            right = float(np.interp(row, rows[::-1], rights[::-1]))
            allowed[row, max(0, int(left+1)):min(84, int(right))] = True
        bright = (frame >= .60) & (frame <= .705) & allowed
        seen = np.zeros_like(bright)
        circles = []
        for y0, x0 in zip(*np.nonzero(bright)):
            if seen[y0, x0]:
                continue
            seen[y0, x0] = True
            stack, pixels = [(int(y0), int(x0))], []
            while stack:
                y, x = stack.pop(); pixels.append((y, x))
                for ny, nx in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
                    if 0 <= ny < 84 and 0 <= nx < 84 and bright[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx] = True; stack.append((ny,nx))
            if not 4 <= len(pixels) <= 48:
                continue
            ys, xs = np.asarray(pixels).T
            width, height = xs.max()-xs.min()+1, ys.max()-ys.min()+1
            aspect = width*self.PX_Y/(height*self.PX_X)
            if not (2 <= width <= 8 and 2 <= height <= 9 and .65 <= aspect <= 1.9):
                continue
            x, y = float(xs.mean()), float(ys.mean())
            surrounding = 0
            for py, px in ((y,x-4), (y,x+4), (y-5,x), (y+5,x)):
                iy, ix = int(round(py)), int(round(px))
                if 0 <= iy < 73 and 0 <= ix < 84 and .32 <= frame[iy,ix] <= .51:
                    surrounding += 1
            if surrounding >= 3:
                circles.append(((self.CAR_Y-y)/self.PX_Y, (x-self.CAR_X)/self.PX_X))
        return sorted(circles)



class Agent:
    """Research selection of current committed pass or fast camera control."""

    def __init__(self):
        self.fast = _FastController()
        self.hazard = _CommittedHazardController()
        self.reset()

    def reset(self, observation=None):
        self.fast.reset(observation)
        self.hazard.reset(observation)
        self.last_steer = 0.
        self.last_target = 100.
        self.mode = "fast"

    def act(self, observation):
        fast_action = self.fast.act(observation)
        hazard_action = self.hazard.act(observation)
        selected = self.hazard if self.hazard.last_pass_plan is not None else self.fast
        action = hazard_action if selected is self.hazard else fast_action
        self.mode = "hazard" if selected is self.hazard else "fast"
        self.last_target = float(selected.last_target)
        self.last_steer = float(action[0])
        self.fast.last_steer = self.hazard.last_steer = self.last_steer
        self.hazard._metric.last_steer = self.last_steer
        self.hazard._robust._last_steer = self.last_steer
        return action
