import math
import pickle
import time

import numpy as np
import torch
from torch import nn

MODEL_FILENAME = "model.pt"
POLICY_MODEL_FILENAME = "policy.pt"
DYNAMICS_MODEL_FILENAME = "dynamics.pt"
DRQ_ACTOR_FORMAT = "haic-drq-v2-actor-v1"

# Small single-observation CNN inference is faster and more predictable without
# the default large CPU thread pool.
torch.set_num_threads(1)


class NatureFeatures(nn.Module):
    def __init__(self, input_channels=4, action_dimensions=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

    def forward(self, observation):
        return self.linear(self.cnn(observation.float()))


class Baseline1Actor(nn.Module):
    """Deterministic actor equivalent to the Baseline 1 SB3 CnnPolicy."""

    def __init__(self, input_channels=4, action_dimensions=3):
        super().__init__()
        self.input_channels = int(input_channels)
        self.features_extractor = NatureFeatures(self.input_channels)
        self.policy_net = nn.Sequential()
        self.action_dimensions = int(action_dimensions)
        self.action_net = nn.Linear(512, self.action_dimensions)

    def forward(self, observation):
        features = self.features_extractor(observation)
        return self.action_net(self.policy_net(features))

    def predict_action(self, observation):
        mean = self.forward(observation)
        return torch.stack(
            (
                mean[..., 0].clamp(-1.0, 1.0),
                mean[..., 1].clamp(0.0, 1.0),
                mean[..., 2].clamp(0.0, 1.0),
            ),
            dim=-1,
        )


class DrQFeatures(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Flatten(), nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim), nn.Tanh(),
        )

    def forward(self, observation):
        return self.linear(self.convolution(observation.float() - 0.5))


class DrQActor(nn.Module):
    """Inference-only architecture matching the native DrQ actor state keys."""

    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.encoder = DrQFeatures(feature_dim)
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, 3)

    def forward(self, observation):
        return torch.tanh(self.policy(self.trunk(self.encoder(observation))))


class _ForwardCorridorController:
    """Small pixel-only controller used by the legacy baseline checkpoint.

    The old baseline actor was trained without an explicit forward-direction
    constraint.  In practice a large steering command could turn the car
    around, after which positive throttle became reverse progress.  This
    controller keeps the baseline runtime on the visible road corridor,
    limits steering changes, and emits mutually exclusive forward throttle or
    brake.  It intentionally uses only the public four-frame camera input so
    it remains valid in the submission container.
    """

    IMAGE_CENTER = 41.5
    ROAD_LOW = 0.24
    ROAD_HIGH = 0.52
    OBSTACLE_LOW = 0.54
    # Conservative limits are intentional: this phase prioritizes staying on
    # the road over lap time.  Curves may use more steering than straights,
    # but neither mode can jump directly to a large drift-like command.
    MAX_STEER = 0.48
    STRAIGHT_MAX_STEER = 0.16
    STRAIGHT_CENTER_DEADBAND = 1.25
    # Pixel quantization can make a genuinely straight approach look like a
    # small multi-pixel sweep.  Keep that weak apparent bend in the straight
    # steering regime so it cannot seed a counter-steer; the preview speed
    # planner still sees it and can begin braking early.
    STRAIGHT_SWEEP_DEADBAND = 3.25
    MAX_STEER_STEP = 0.07
    # F1-style pace comes from carrying speed only on a confirmed straight;
    # the curve, hazard, and speed watchdog branches below still reduce gas
    # independently.  Keep this cap modest because the camera controller has
    # no wheel-speed or tyre-temperature telemetry.
    MAX_GAS = 0.16
    MAX_BRAKE = 0.28
    OBSTACLE_MISS_LIMIT = 4
    # A road rendered by the official environment is dark gray (~0.40 after
    # preprocessing); grass/background and orange obstacles are both bright
    # (~0.63--0.70).  Keep a minimum visible road corridor and treat any
    # bright intrusion into that corridor as a hazard.  This deliberately
    # does not try to distinguish grass from an obstacle after RGB->gray
    # conversion: both are unsafe for the car.
    MIN_ROAD_WIDTH = 14.0
    # Treat the camera center as the vehicle envelope, not a point.  A road
    # can still contain the center pixel while the body is already touching
    # green; require a larger near-edge margin before allowing throttle.
    MIN_SAFE_WIDTH = 20.0
    MIN_EDGE_CLEARANCE = 7.0
    HAZARD_TARGET_SPEED = 24.0
    HAZARD_BRAKE = 0.16
    RECOVERY_BRAKE = 0.12
    # Safety braking is a transient speed-management state, not a terminal
    # action.  Once a few frames have been spent reducing speed, the car
    # resumes a small forward crawl so a noisy visual hazard cannot leave it
    # parked indefinitely before the next steering correction.
    HAZARD_BRAKE_FRAMES = 3
    GREEN_BRAKE_FRAMES = 5
    HAZARD_CRAWL_GAS = 0.035
    RECOVERY_BRAKE_FRAMES = 2
    RECOVERY_GAS = 0.028
    SPEED_BRAKE_FRAMES = 8
    SPEED_WATCHDOG_GAS = 0.03
    # Preview-based curve management.  The long look-ahead rows give the
    # speed controller time to settle before the near road starts turning;
    # the gas scale then reduces longitudinal force continuously with the
    # observed bend instead of waiting for a large steering command.
    CURVE_TRIGGER_SWEEP = 3.0
    CURVE_ENTRY_SPEED_PENALTY = 0.80
    CURVE_SPEED_PENALTY = 2.80
    CURVE_SPEED_FLOOR = 28.0
    CURVE_GAS_REDUCTION = 0.035
    CURVE_MIN_GAS_SCALE = 0.40
    CURVE_MAX_STEER = 0.40
    CURVE_STEER_STEP = 0.055
    CURVE_STEER_GAIN_SCALE = 0.68
    CURVE_HIGH_SPEED_STEER_FLOOR = 0.58
    CURVE_TARGET_FALL_BLEND = 0.35
    RECOVERY_MAX_STEER = 0.30
    RECOVERY_STEER_GAIN = 0.020
    RECOVERY_HEADING_GAIN = 0.014
    RECOVERY_SEARCH_STEER = 0.16
    RECOVERY_SEARCH_PERIOD = 4
    HAZARD_ROWS = (54, 50, 46, 42, 38)
    ROAD_ROWS = (54, 50, 46, 42, 38, 34, 30, 26, 22)
    SPEED_ROI = (77, 83, 10, 13)
    SPEED_BASELINE = 0.27
    SPEED_PER_UNIT = 0.085
    STRAIGHT_CRUISE_SPEED = 68.0
    # A compact bright component is treated as a dynamic occupancy rather
    # than as a binary "steer now" cue.  These speed limits are a cheap
    # camera-only approximation of a time-to-collision envelope: as the
    # component moves down the image, leave less longitudinal speed for the
    # lateral escape manoeuvre.  The limits stay non-zero so the controller
    # keeps making forward progress while it searches for a safe side.
    OBSTACLE_FAR_SPEED = 47.0
    OBSTACLE_MID_SPEED = 30.0
    OBSTACLE_CLOSE_SPEED = 18.0

    def __init__(self, *, cruise_speed: float = STRAIGHT_CRUISE_SPEED) -> None:
        self.cruise_speed = float(cruise_speed)
        self._obstacle_side = 0.0
        self._obstacle_missing = 0
        self._last_obstacle_side_offset = None
        self._last_steer = 0.0
        self._target_speed = None
        self._hazard_frames = 0
        self._recovery_frames = 0
        self._brake_frames = 0
        self._last_road_center = self.IMAGE_CENTER
        self._last_road_heading = 0.0
        self._last_recovery_hint = 0.0
        self._search_sign = 1.0
        self.road_visible = False
        self.has_seen_road = False

    def reset(self, observation=None) -> None:
        del observation
        self._obstacle_side = 0.0
        self._obstacle_missing = 0
        self._last_obstacle_side_offset = None
        self._last_steer = 0.0
        self._target_speed = None
        self._hazard_frames = 0
        self._recovery_frames = 0
        self._brake_frames = 0
        self._last_road_center = self.IMAGE_CENTER
        self._last_road_heading = 0.0
        self._last_recovery_hint = 0.0
        self._search_sign = 1.0
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
        centers, _spans = self._road_geometry(frame)
        return centers

    def _road_geometry(
        self, frame: np.ndarray
    ) -> tuple[dict[int, float], dict[int, tuple[float, float]]]:
        """Return centerline samples and their visible asphalt spans.

        The span is intentionally derived from the same local search used for
        the centerline.  It gives the safety layer a direct measure of how
        much road remains under the car instead of assuming that a center
        estimate alone means the whole corridor is safe.
        """
        asphalt = (frame >= self.ROAD_LOW) & (frame <= self.ROAD_HIGH)
        horizontal = np.arange(frame.shape[1], dtype=np.float32)
        centers: dict[int, float] = {}
        spans: dict[int, tuple[float, float]] = {}
        previous = self.IMAGE_CENTER
        for row in self.ROAD_ROWS:
            selected = asphalt[row] & (np.abs(horizontal - previous) <= 17.0)
            locations = np.flatnonzero(selected)
            if len(locations) >= 4:
                previous = float(locations.mean())
                centers[row] = previous
                spans[row] = (float(locations[0]), float(locations[-1]))
        return centers, spans

    def _wide_road_geometry(
        self, frame: np.ndarray
    ) -> tuple[dict[int, float], dict[int, tuple[float, float]]]:
        """Search the complete image for a road after the local view is lost.

        Normal tracking intentionally searches near the previous centerline to
        reject HUD/background texture.  During recovery that restriction can
        hide the road precisely when the car has moved far to one side, so use
        contiguous asphalt runs across the full width and follow the run
        closest to the last remembered center.
        """
        asphalt = (frame >= self.ROAD_LOW) & (frame <= self.ROAD_HIGH)
        centers: dict[int, float] = {}
        spans: dict[int, tuple[float, float]] = {}
        previous = float(self._last_road_center)
        for row in self.ROAD_ROWS:
            locations = np.flatnonzero(asphalt[row])
            if len(locations) < 4:
                continue
            breaks = np.flatnonzero(np.diff(locations) > 1) + 1
            runs = np.split(locations, breaks)
            runs = [run for run in runs if len(run) >= 4]
            if not runs:
                continue
            run = min(
                runs,
                key=lambda candidate: (
                    abs(float(candidate.mean()) - previous),
                    -len(candidate),
                ),
            )
            previous = float(run.mean())
            centers[row] = previous
            spans[row] = (float(run[0]), float(run[-1]))
        return centers, spans

    @classmethod
    def _safe_center_at(
        cls,
        row: float,
        centers: dict[int, float],
        spans: dict[int, tuple[float, float]] | None,
    ) -> float:
        """Return a preview center clipped to the vehicle-safe road envelope."""
        center = cls._center_at(row, centers)
        if not spans:
            return center
        span = cls._span_at(row, spans)
        if span is None:
            return center
        left = span[0] + cls.MIN_EDGE_CLEARANCE
        right = span[1] - cls.MIN_EDGE_CLEARANCE
        if left > right:
            return (span[0] + span[1]) * 0.5
        return float(np.clip(center, left, right))

    @classmethod
    def _track_bound_steering(
        cls,
        steering: float,
        spans: dict[int, tuple[float, float]],
    ) -> float:
        """Prevent a command from pointing outside the near-car envelope.

        The camera centre is the projected vehicle centre.  If the visible
        asphalt envelope is already to one side, an outward steering command
        cannot be a valid racing line: it would move the body farther over the
        track edge.  Bias toward the widest safe interval and suppress only
        the outward component, leaving the normal rate limiter to smooth the
        transition.
        """
        if not spans:
            return float(steering)
        near_span = spans.get(54) or spans.get(50)
        if near_span is None:
            return float(steering)
        left = near_span[0] + cls.MIN_EDGE_CLEARANCE
        right = near_span[1] - cls.MIN_EDGE_CLEARANCE
        if left > right:
            target = (near_span[0] + near_span[1]) * 0.5
        else:
            target = float(np.clip(cls.IMAGE_CENTER, left, right))
        envelope_error = target - cls.IMAGE_CENTER
        if abs(envelope_error) <= 0.25:
            return float(steering)
        correction = float(np.clip(0.035 * envelope_error, -0.24, 0.24))
        if steering * envelope_error < 0.0:
            steering = 0.0
        return float(0.55 * steering + 0.45 * correction)

    def _remember_road(
        self,
        centers: dict[int, float],
        spans: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        """Store a short look-ahead road estimate for recovery steering."""
        if not centers:
            return
        near = self._safe_center_at(54.0, centers, spans)
        far = self._safe_center_at(34.0, centers, spans)
        self._last_road_center = float(0.35 * near + 0.65 * far)
        self._last_road_heading = float(far - near)
        error = self._last_road_center - self.IMAGE_CENTER
        if abs(error) > self.STRAIGHT_CENTER_DEADBAND:
            self._last_recovery_hint = float(np.sign(error))
        elif abs(self._last_steer) > self.MAX_STEER_STEP:
            self._last_recovery_hint = float(np.sign(self._last_steer))
        elif abs(self._last_road_heading) <= self.STRAIGHT_SWEEP_DEADBAND:
            self._last_recovery_hint = 0.0

    def _recovery_steer(self, centers: dict[int, float] | None) -> float:
        """Produce a bounded look-ahead correction while road visibility heals."""
        if centers:
            near = self._center_at(54.0, centers)
            far = self._center_at(34.0, centers)
            projected = 0.35 * near + 0.65 * far
            desired = (
                self.RECOVERY_STEER_GAIN * (projected - self.IMAGE_CENTER)
                + self.RECOVERY_HEADING_GAIN * (far - near)
            )
            if abs(desired) < 0.04 and self._last_recovery_hint != 0.0:
                desired = 0.12 * self._last_recovery_hint
        else:
            if (
                self._recovery_frames > self.RECOVERY_BRAKE_FRAMES
                and (self._recovery_frames - self.RECOVERY_BRAKE_FRAMES)
                % self.RECOVERY_SEARCH_PERIOD
                == 0
            ):
                self._search_sign *= -1.0
            heading_term = float(
                np.clip(
                    self.RECOVERY_HEADING_GAIN * self._last_road_heading,
                    -0.08,
                    0.08,
                )
            )
            desired = self._search_sign * self.RECOVERY_SEARCH_STEER + heading_term
            if self._last_recovery_hint != 0.0:
                desired = 0.65 * self._last_recovery_hint + 0.35 * desired
        return float(np.clip(desired, -self.RECOVERY_MAX_STEER, self.RECOVERY_MAX_STEER))

    @classmethod
    def _span_at(
        cls, row: float, spans: dict[int, tuple[float, float]]
    ) -> tuple[float, float] | None:
        if not spans:
            return None
        known_rows = sorted(spans)
        left = float(
            np.interp(
                row,
                np.asarray(known_rows, dtype=np.float32),
                np.asarray([spans[y][0] for y in known_rows], dtype=np.float32),
            )
        )
        right = float(
            np.interp(
                row,
                np.asarray(known_rows, dtype=np.float32),
                np.asarray([spans[y][1] for y in known_rows], dtype=np.float32),
            )
        )
        return left, right

    @classmethod
    def _corridor_hazard(
        cls,
        frame: np.ndarray,
        centers: dict[int, float],
        spans: dict[int, tuple[float, float]],
    ) -> tuple[bool, bool, float]:
        """Assess green/off-road intrusion without relying on RGB colors.

        Returns ``(hazard, blocked, steer_hint)``.  ``steer_hint`` is positive
        when the safer side is to the right, negative for the left, and zero
        when both sides are equally unsafe.  A very narrow or discontinuous
        visible corridor is considered blocked: braking is safer than adding
        steering that could put the car onto grass or into an obstacle.
        """
        if len(centers) < 3 or not spans:
            return True, True, 0.0

        # Prefer an actually observed near row.  Interpolating over a missing
        # near sample would make a disappearing road look safe.
        near_span = spans.get(54) or spans.get(50)
        far_span = spans.get(34) or spans.get(38) or spans.get(42)
        near_width = (
            near_span[1] - near_span[0] + 1.0 if near_span is not None else 0.0
        )
        far_width = (
            far_span[1] - far_span[0] + 1.0 if far_span is not None else 0.0
        )
        blocked = near_span is None or near_width < cls.MIN_ROAD_WIDTH
        # Near the camera the projected road should not be narrower than its
        # farther samples.  A sudden collapse is the visual signature of the
        # car reaching a green shoulder or of a large bright obstruction.
        if far_width >= cls.MIN_SAFE_WIDTH and near_width + 3.0 < 0.72 * far_width:
            blocked = True
        near_center = cls._center_at(54.0, centers)
        if abs(near_center - cls.IMAGE_CENTER) > 15.0:
            blocked = True
        if near_span is not None and not (
            near_span[0] + cls.MIN_EDGE_CLEARANCE <= cls.IMAGE_CENTER <=
            near_span[1] - cls.MIN_EDGE_CLEARANCE
        ):
            blocked = True

        bright = frame >= cls.OBSTACLE_LOW
        danger_x: list[float] = []
        safe_left = safe_right = 0.0
        for row in cls.HAZARD_ROWS:
            span = cls._span_at(float(row), spans)
            if span is None:
                continue
            left, right = span
            center = (left + right) * 0.5
            row_index = int(np.clip(row, 0, frame.shape[0] - 1))
            locations = np.flatnonzero(bright[row_index])
            inside = locations[(locations >= left) & (locations <= right)]
            if len(inside):
                danger_x.extend(float(value) for value in inside)
            # Penalize a side whose visible asphalt clearance is already too
            # small.  Dark/bright grass outside the span is not itself a
            # detection; the shrinking asphalt span is the evidence that it
            # has entered the drivable corridor.
            safe_left += max(0.0, center - left)
            safe_right += max(0.0, right - center)

        intrusion = len(danger_x) >= 2
        hazard = blocked or intrusion or near_width < cls.MIN_SAFE_WIDTH
        if not hazard:
            return False, False, 0.0
        if danger_x:
            danger_center = float(np.mean(danger_x))
            reference = cls._center_at(46.0, centers)
            steer_hint = 1.0 if danger_center < reference else -1.0
        elif near_center > cls.IMAGE_CENTER + 1.0:
            steer_hint = 1.0
        elif near_center < cls.IMAGE_CENTER - 1.0:
            steer_hint = -1.0
        elif safe_right > safe_left + 1.0:
            steer_hint = 1.0
        elif safe_left > safe_right + 1.0:
            steer_hint = -1.0
        else:
            steer_hint = 0.0
        return True, blocked, steer_hint

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

    def _nearest_obstacle(
        self,
        frame: np.ndarray,
        centers: dict[int, float],
        spans: dict[int, tuple[float, float]] | None = None,
    ) -> tuple[float, float, float] | None:
        """Find a compact bright object inside the visible road band."""
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
                            if bright[neighbor_y, neighbor_x] and not visited[neighbor_y, neighbor_x]:
                                visited[neighbor_y, neighbor_x] = True
                                stack.append((neighbor_x, neighbor_y))
                box_width = max_x - min_x + 1
                box_height = max_y - min_y + 1
                # The official obstacle generator is deterministic per
                # (track_id, seed), but its rendered projection changes with
                # distance and curvature.  Do not assume one tiny sprite
                # size: retain compact-to-wide components while rejecting the
                # connected bright background/shoulder itself.
                if not (
                    4 <= area <= 360
                    and 2 <= box_width <= 24
                    and 2 <= box_height <= 28
                ):
                    continue
                center_x = sum_x / area
                center_y = sum_y / area
                road_center = self._center_at(center_y, centers)
                visible_span = self._span_at(center_y, spans)
                in_visible_corridor = visible_span is None or (
                    visible_span[0] - 3.0
                    <= center_x
                    <= visible_span[1] + 3.0
                )
                if in_visible_corridor and abs(center_x - road_center) <= 28.0:
                    candidates.append((center_y, center_x, road_center))
        return max(candidates, key=lambda item: item[0]) if candidates else None

    @classmethod
    def _obstacle_speed_limit(cls, obstacle_y: float) -> float:
        """Map image-space obstacle distance to a conservative speed target.

        The forward road projection is monotonic: a larger ``y`` means the
        object is closer.  Piecewise interpolation is less brittle than a
        hard near/far switch and gives the longitudinal controller time to
        brake before the object reaches the vehicle envelope.
        """
        y = float(np.clip(obstacle_y, 22.0, 58.0))
        if y <= 38.0:
            return cls.OBSTACLE_FAR_SPEED
        if y <= 44.0:
            return float(
                np.interp(
                    y,
                    (38.0, 44.0),
                    (cls.OBSTACLE_FAR_SPEED, cls.OBSTACLE_MID_SPEED),
                )
            )
        if y <= 50.0:
            return float(
                np.interp(
                    y,
                    (44.0, 50.0),
                    (cls.OBSTACLE_MID_SPEED, cls.OBSTACLE_CLOSE_SPEED),
                )
            )
        return cls.OBSTACLE_CLOSE_SPEED

    @classmethod
    def _estimate_speed(cls, frame: np.ndarray) -> float:
        top, bottom, left, right = cls.SPEED_ROI
        mass = float(frame[top:bottom, left:right].sum())
        return float(np.clip((mass - cls.SPEED_BASELINE) / cls.SPEED_PER_UNIT, 0.0, 80.0))

    @classmethod
    def _road_sweep(cls, centers: dict[int, float]) -> float:
        """Estimate the largest centerline displacement over a preview span."""
        if len(centers) < 2:
            return 0.0
        near = cls._center_at(54.0, centers)
        return max(
            (
                abs(cls._center_at(float(row), centers) - near)
                for row in (22, 26, 30, 34, 38, 42)
            ),
            default=0.0,
        )

    def _pedals(self, speed: float, target_speed: float) -> tuple[float, float]:
        if speed > target_speed + 1.0:
            brake = float(np.clip((speed - target_speed) * 0.012, 0.04, self.MAX_BRAKE))
            return 0.0, brake
        if speed < target_speed - 8.0:
            return self.MAX_GAS, 0.0
        if speed < target_speed - 3.0:
            return self.MAX_GAS * (2.0 / 3.0), 0.0
        return self.MAX_GAS * (5.0 / 12.0), 0.0

    def _lost_road_action(
        self, centers: dict[int, float] | None = None
    ) -> np.ndarray:
        """Brake briefly, then steer toward the remembered road and crawl."""
        self._recovery_frames += 1
        self._brake_frames = 0
        desired = self._recovery_steer(centers)
        if self._last_steer != 0.0 and desired * self._last_steer < 0.0:
            # Recovery is allowed to change sides, but only through a neutral
            # command.  This keeps the bounded search from injecting its own
            # counter-steer when the remembered road direction changes.
            desired = 0.0
        self._last_steer = float(
            self._last_steer
            + np.clip(
                desired - self._last_steer,
                -self.MAX_STEER_STEP,
                self.MAX_STEER_STEP,
            )
        )
        self._last_steer = float(
            np.clip(self._last_steer, -self.RECOVERY_MAX_STEER, self.RECOVERY_MAX_STEER)
        )
        if self._recovery_frames <= self.RECOVERY_BRAKE_FRAMES:
            gas, brake = 0.0, self.RECOVERY_BRAKE
        else:
            gas, brake = self.RECOVERY_GAS, 0.0
        return np.asarray([self._last_steer, gas, brake], dtype=np.float32)

    def act(self, observation) -> np.ndarray:
        frame = self._frame(observation)
        if frame is None:
            self.road_visible = False
            if self.has_seen_road:
                return self._lost_road_action()
            return np.zeros(3, dtype=np.float32)

        centers, spans = self._road_geometry(frame)
        self.road_visible = len(centers) >= 3
        if not self.road_visible:
            if self.has_seen_road:
                # First search the full image: the car may have moved far
                # enough that the local centerline window no longer contains
                # the road.  If even that fails, use the remembered
                # look-ahead direction and a bounded alternating search.
                recovery_centers, _recovery_spans = self._wide_road_geometry(frame)
                if len(recovery_centers) >= 2:
                    self._remember_road(recovery_centers, _recovery_spans)
                    return self._lost_road_action(recovery_centers)
                return self._lost_road_action()
            return np.zeros(3, dtype=np.float32)
        self.has_seen_road = True
        self._recovery_frames = 0
        self._remember_road(centers, spans)
        # Use the far preview for turn anticipation, but clip both preview
        # points to the visible vehicle-safe envelope.  This prevents a noisy
        # centre estimate from asking the car's centre to cross the track edge.
        far = self._safe_center_at(42.0, centers, spans)
        near = self._safe_center_at(54.0, centers, spans)
        road_sweep = self._road_sweep(centers)
        # Compare the farthest visible preview with the mid-preview as well
        # as the near edge.  This catches a bend while it is still several
        # camera rows ahead, before a large steering command is necessary.
        preview_far = self._safe_center_at(22.0, centers, spans)
        preview_mid = self._safe_center_at(42.0, centers, spans)
        curve_entry_sweep = abs(preview_far - preview_mid)
        curve_strength = max(road_sweep, curve_entry_sweep)
        curve_mode = curve_strength >= self.CURVE_TRIGGER_SWEEP
        center_offset = far - self.IMAGE_CENTER
        corridor_hazard, corridor_blocked, corridor_hint = self._corridor_hazard(
            frame, centers, spans
        )
        if corridor_hazard:
            self._hazard_frames += 1
        else:
            self._hazard_frames = 0
        straight = (
            abs(center_offset) <= self.STRAIGHT_CENTER_DEADBAND
            and abs(far - near) <= self.STRAIGHT_SWEEP_DEADBAND
            and road_sweep <= self.STRAIGHT_SWEEP_DEADBAND
        )
        if straight:
            # Quantization noise on a straight road should not create a
            # persistent weave.  A small residual correction remains only
            # outside the deadband.
            steering = 0.0
            steer_limit = self.STRAIGHT_MAX_STEER
        else:
            # The look-ahead center is the primary turn cue.  A raw heading
            # derivative can briefly point the other way when the near edge
            # is noisy (or while the car is still recovering from the prior
            # bend); adding it without a sign check creates the observed
            # counter-steer and lets the car drift before the real turn.
            heading_error = far - near
            heading_term = 0.012 * heading_error
            if (
                abs(center_offset) > self.STRAIGHT_CENTER_DEADBAND
                and center_offset * heading_term < 0.0
            ):
                heading_term = 0.0
            steering = 0.016 * center_offset + heading_term
            steer_limit = self.MAX_STEER
        target_speed = float(
            np.clip(
                self.cruise_speed
                - self.CURVE_SPEED_PENALTY * road_sweep
                - self.CURVE_ENTRY_SPEED_PENALTY * curve_entry_sweep,
                self.CURVE_SPEED_FLOOR,
                self.cruise_speed,
            )
        )

        obstacle = self._nearest_obstacle(frame, centers, spans)
        if obstacle is not None:
            obstacle_y, obstacle_x, road_center = obstacle
            # A straight road normally has a tight steering limit; a detected
            # obstacle is the explicit exception, but still stays well below
            # a drift-sized command.
            steer_limit = min(self.MAX_STEER, 0.32)
            side_offset = obstacle_x - road_center
            # Select the side with the larger visible clearance, rather than
            # blindly mirroring the obstacle around the estimated centerline.
            # This is the local-planning equivalent of choosing the wider
            # free-space corridor in a real autonomous stack.
            obstacle_span = self._span_at(float(obstacle_y), spans)
            if obstacle_span is not None:
                left_clearance = max(0.0, obstacle_x - obstacle_span[0])
                right_clearance = max(0.0, obstacle_span[1] - obstacle_x)
                candidate_side = 1.0 if right_clearance >= left_clearance else -1.0
            else:
                candidate_side = 1.0 if side_offset < 0.0 else -1.0
            if self._obstacle_side == 0.0:
                self._obstacle_side = candidate_side
            elif obstacle_y < 52.0:
                self._obstacle_side = candidate_side
            self._last_obstacle_side_offset = side_offset
            urgency = float(np.clip((obstacle_y - 22.0) / 18.0, 0.0, 1.0))
            steering += self._obstacle_side * 0.24 * urgency
            target_speed = min(target_speed, self._obstacle_speed_limit(obstacle_y))
            self._obstacle_missing = 0
        elif self._obstacle_side != 0.0:
            self._obstacle_missing += 1
            if self._obstacle_missing > self.OBSTACLE_MISS_LIMIT:
                self._obstacle_side = 0.0
                self._obstacle_missing = 0
                self._last_obstacle_side_offset = None

        if corridor_hazard:
            # Grass and orange obstacles occupy the same bright range after
            # RGB->gray preprocessing.  Apply one conservative response to
            # both: bias only toward the visible road, brake briefly, then
            # continue at a crawl while the bounded avoidance steer works.
            steering += 0.18 * corridor_hint
            target_speed = min(target_speed, self.HAZARD_TARGET_SPEED)
            if corridor_blocked:
                # Green/off-road contact has priority over curve following or
                # obstacle avoidance.  Discard competing steering and use
                # only the side that returns toward the observed asphalt.
                steering = 0.24 * corridor_hint
                steer_limit = min(steer_limit, 0.28)

        # F1 track-limit discipline: the near vehicle envelope is a hard
        # geometric constraint.  Apply it after obstacle/curve proposals so
        # a fast racing-line cue can never point farther toward the green
        # shoulder or an unseen edge.
        steering = self._track_bound_steering(steering, spans)

        if self._target_speed is not None:
            # Retain a slower target while entering a bend, but let a clear
            # road recover speed promptly after a hazard or a completed turn.
            # The asymmetric blend avoids the sluggish multi-second return to
            # cruise caused by the old all-purpose 0.65 memory.
            if target_speed < self._target_speed:
                blend = self.CURVE_TARGET_FALL_BLEND if curve_mode else 0.65
            else:
                blend = 0.45
            target_speed = blend * self._target_speed + (1.0 - blend) * target_speed
        if not straight and abs(steering) > 0.28:
            target_speed = min(target_speed, 44.0)
        # Store the post-curve-limit target so the next frame does not briefly
        # recover toward an obsolete, faster target during the same bend.
        self._target_speed = target_speed
        speed = self._estimate_speed(frame)
        gas, brake = self._pedals(speed, target_speed)
        if corridor_hazard:
            if self._hazard_frames <= self.HAZARD_BRAKE_FRAMES:
                gas = 0.0
                brake = max(brake, self.HAZARD_BRAKE)
            else:
                # Continue moving at a controlled crawl while the bounded
                # avoidance steer clears the bright shoulder/obstacle.  This
                # prevents the safety layer from becoming a permanent stop.
                gas = max(self.HAZARD_CRAWL_GAS, min(self.MAX_GAS, gas))
                brake = 0.0
            self._brake_frames = 0
        elif gas <= 0.0 and brake > 0.0:
            self._brake_frames += 1
            if self._brake_frames > self.SPEED_BRAKE_FRAMES:
                # A stale speed bar or a long, conservative bend must not
                # turn ordinary speed regulation into a permanent stop.
                gas = self.SPEED_WATCHDOG_GAS
                brake = 0.0
        else:
            self._brake_frames = 0
        if corridor_blocked and self._hazard_frames <= self.GREEN_BRAKE_FRAMES:
            # When the road width itself is unsafe, a short braking window is
            # preferable to a large recovery turn that could cross grass or an
            # unseen obstacle.  After that window, the hazard branch above
            # deliberately returns a crawl command instead of stopping.
            gas = 0.0
            brake = max(brake, min(self.MAX_BRAKE, self.HAZARD_BRAKE * 1.25))
        elif corridor_blocked:
            gas = max(self.HAZARD_CRAWL_GAS, min(self.MAX_GAS, gas))
            brake = 0.0

        # A detected compact object is a collision risk even when it is still
        # far enough away that the corridor-wide bright-pixel gate has not
        # fired.  Remove forward throttle immediately so the selected escape
        # side has room to clear it; close objects already carry a brake from
        # the corridor safety branch above.
        if obstacle is not None:
            gas = 0.0

        # Slow down before a bend and rate-limit steering so one noisy frame
        # cannot turn the car around or induce a drift-like correction.
        if curve_mode and not corridor_hazard and brake <= 0.0:
            curve_gas_scale = float(
                np.clip(
                    1.0 - self.CURVE_GAS_REDUCTION * road_sweep,
                    self.CURVE_MIN_GAS_SCALE,
                    1.0,
                )
            )
            gas = min(gas, self.MAX_GAS * curve_gas_scale)
        if curve_mode and obstacle is None:
            steer_limit = min(steer_limit, self.CURVE_MAX_STEER)
        if not straight and abs(steering) > 0.28:
            gas = min(gas, self.MAX_GAS * 0.5)
        if curve_mode and obstacle is None:
            # This vehicle has little rotational inertia: the same steering
            # value produces a much sharper yaw response than a conventional
            # car.  Attenuate the curve gain, and attenuate it further while
            # the measured speed is still above the curve target.  Once the
            # brake has brought speed down, the full bounded correction can
            # return without a large lateral impulse.
            speed_ratio = float(
                np.clip(
                    target_speed / max(speed, target_speed, 1.0),
                    self.CURVE_HIGH_SPEED_STEER_FLOOR,
                    1.0,
                )
            )
            steering *= self.CURVE_STEER_GAIN_SCALE * speed_ratio
        steering = float(np.clip(steering, -steer_limit, steer_limit))
        if self._last_steer != 0.0 and steering * self._last_steer < 0.0:
            # Even a small command must cross zero before changing sides.  A
            # rate limit alone still allows a tiny positive steer to become a
            # negative command in one frame, which is enough to start the
            # counter-steer/drift seen in visual replays.
            steering = 0.0
        steer_step = (
            self.CURVE_STEER_STEP
            if curve_mode and obstacle is None
            else self.MAX_STEER_STEP
        )
        steering = self._last_steer + float(
            np.clip(steering - self._last_steer, -steer_step, steer_step)
        )
        steering = float(np.clip(steering, -steer_limit, steer_limit))
        self._last_steer = steering
        # Keep the float32 wire value strictly within the declared action
        # bound; rounding 0.28 to float32 can otherwise compare just above
        # MAX_BRAKE in downstream validators.
        brake = min(
            brake,
            float(np.nextafter(np.float32(self.MAX_BRAKE), np.float32(0.0))),
        )
        return np.asarray([steering, gas, brake], dtype=np.float32)


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


class _CompoundSpeedMarginController(_LaunchThrottleController):
    """Reduce speed only for an already detected obstacle in a sharp bend."""

    COMPOUND_TARGET_SPEED = 24.0


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


class _TranslationInvariantExitController(_FastCornerCarryController):
    """Accelerate on aligned curve exits before recentering is complete.

    The inherited straight predicate intentionally stays unchanged.  This
    layer adds only a longitudinal gate whose road-shape measurement compares
    sampled centers with another sampled center, so lateral translation is not
    mistaken for curvature.  Steering and all obstacle states remain inherited.
    """

    EXIT_SHAPE_ROWS = (30, 34, 38, 42, 46, 50)
    EXIT_SHAPE_SWEEP_MAX = 1.5
    EXIT_HEADING_MAX = 1.5
    EXIT_CENTER_OFFSET_MAX = 6.0
    EXIT_STEER_REQUEST_MAX = 0.28

    def __init__(self, *, cruise_speed: float = 68.0) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._exit_shape_sweep = None
        self._clear_exit = False

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._exit_shape_sweep = None
        self._clear_exit = False

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
        self._exit_shape_sweep = None
        self._clear_exit = False
        required_rows = (*self.EXIT_SHAPE_ROWS, 54)
        if any(row not in centers for row in required_rows):
            return adjusted

        near = float(centers[54])
        far = float(centers[42])
        shape_sweep = max(
            abs(float(centers[row]) - near) for row in self.EXIT_SHAPE_ROWS
        )
        self._exit_shape_sweep = float(shape_sweep)
        self._clear_exit = bool(
            not straight
            and obstacle is None
            and not self._carry_latched_at_frame_start
            and self._obstacle_side == 0.0
            and shape_sweep <= self.EXIT_SHAPE_SWEEP_MAX
            and abs(far - near) <= self.EXIT_HEADING_MAX
            and abs(far - self.IMAGE_CENTER) <= self.EXIT_CENTER_OFFSET_MAX
            and abs(adjusted) <= self.EXIT_STEER_REQUEST_MAX
        )
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
            self._clear_exit
            and obstacle is None
            and not self._carry_latched_at_frame_start
            and self._obstacle_side == 0.0
        ):
            self._pace_command_target = self.STRAIGHT_TARGET_SPEED
            return self.STRAIGHT_TARGET_SPEED
        return inherited

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
            not self._clear_exit
            or obstacle is not None
            or self._carry_latched_at_frame_start
            or self._obstacle_side != 0.0
            or self._pace_effective_target is None
        ):
            return inherited_gas, inherited_brake

        target = self._pace_effective_target
        if self._pace_speed > target + 1.0:
            exit_brake = float(
                np.clip(
                    (self._pace_speed - target) * 0.012,
                    0.04,
                    self.MAX_BRAKE,
                )
            )
            exit_brake = min(
                exit_brake,
                float(
                    np.nextafter(
                        np.float32(self.MAX_BRAKE), np.float32(0.0)
                    )
                ),
            )
            return 0.0, exit_brake
        if self._pace_speed < target - 8.0:
            return self.STRAIGHT_GAS_FULL, 0.0
        if self._pace_speed < target - 3.0:
            return self.STRAIGHT_GAS_MID, 0.0
        return self.STRAIGHT_GAS_HOLD, 0.0


class _CoherentCurveAttackController(_TranslationInvariantExitController):
    """Strengthen only a visually coherent, obstacle-free curve request."""

    ATTACK_GAIN = 1.20
    ATTACK_MAX_INCREMENT = 0.06
    ATTACK_DELTA_MIN = 1.5
    ATTACK_ROWS = (30, 34, 38, 42, 46, 50, 54)

    def __init__(self, *, cruise_speed: float = 68.0) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._curve_attack_active = False
        self._curve_attack_direction = 0.0
        self._curve_attack_control_request = 0.0

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._curve_attack_active = False
        self._curve_attack_direction = 0.0
        self._curve_attack_control_request = 0.0

    def _lost_road_action(self) -> np.ndarray:
        self._curve_attack_active = False
        self._curve_attack_direction = 0.0
        self._curve_attack_control_request = 0.0
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
        self._curve_attack_active = False
        self._curve_attack_direction = 0.0
        self._curve_attack_control_request = float(inherited)

        if (
            straight
            or obstacle is not None
            or self._clear_exit
            or any(row not in centers for row in self.ATTACK_ROWS)
        ):
            return inherited

        near = float(centers[54])
        delta30 = float(centers[30]) - near
        delta34 = float(centers[34]) - near
        if (
            abs(delta30) <= self.ATTACK_DELTA_MIN
            or abs(delta34) <= self.ATTACK_DELTA_MIN
            or delta30 * delta34 <= 0.0
        ):
            return inherited

        direction = 1.0 if delta30 > 0.0 else -1.0
        if (
            inherited * direction <= 0.0
            or any(
                (float(centers[row]) - near) * direction < 0.0
                for row in self.ATTACK_ROWS[:-1]
            )
        ):
            return inherited

        magnitude = min(
            self.MAX_STEER,
            self.ATTACK_GAIN * abs(inherited),
            abs(inherited) + self.ATTACK_MAX_INCREMENT,
        )
        if abs(inherited) <= self.MAX_STEER_STEP <= magnitude:
            return inherited
        candidate = direction * magnitude
        self._curve_attack_active = True
        self._curve_attack_direction = direction
        self._carry_steer_request = candidate
        if self._preview_requested != 0.0 and self._preview_requested == inherited:
            self._preview_requested = candidate
        return candidate


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


class _CompoundBrakeOnsetController(_CompoundClearingBrakeCarryController):
    """Ramp supplemental braking for small visible-compound speed errors.

    The base speed brake remains a floor. Targets, lateral actions, detector
    misses, and large overspeed retain the control policy. This is a diagnostic
    candidate. The registered comparison changed no actions on seven cells;
    this class is retained for audit and is not the active Agent route.
    """

    COMPOUND_BRAKE_RAMP_EXCESS = 2.0

    def _curve_brake_envelope(
        self, *, excess: float, base_brake: float,
    ) -> float:
        inherited = super()._curve_brake_envelope(
            excess=excess, base_brake=base_brake,
        )
        visible_adaptive_compound = (
            self._obstacle_side != 0.0
            and self._obstacle_missing == 0
            and self._pace_latched_target == self.COMPOUND_TARGET_SPEED
            and self._pace_command_target is not None
            and self.COMPOUND_TARGET_SPEED < self._pace_command_target
            <= self.ADAPTIVE_COMPOUND_MAX_TARGET
            and self.COMPOUND_SWEEP_THRESHOLD <= self._pace_sweep
            < self.ADAPTIVE_COMPOUND_EXTREME_SWEEP
        )
        if not visible_adaptive_compound or excess >= self.COMPOUND_BRAKE_RAMP_EXCESS:
            return inherited
        ramp = float(np.clip(excess / self.COMPOUND_BRAKE_RAMP_EXCESS, 0.0, 1.0))
        compound = float(np.clip(
            self.COMPOUND_CARRY_BRAKE_BASE + self.COMPOUND_CARRY_BRAKE_GAIN * excess,
            self.COMPOUND_CARRY_BRAKE_BASE,
            self.COMPOUND_CARRY_BRAKE_MAX,
        ))
        return max(base_brake, ramp * compound)


class _VisibleCompoundBaseBrakeController(_CompoundClearingBrakeCarryController):
    """Evaluate inherited base braking during visible target30 compounds.

    This diagnostic candidate changes only the supplemental brake, including
    near obstacles and extreme curves. Rejected by the registered comparison:
    seed17 lost a clean finish. Kept for reproduction, never the active route.
    """

    def _curve_brake_envelope(
        self, *, excess: float, base_brake: float,
    ) -> float:
        visible_compound = (
            self._obstacle_side != 0.0
            and self._obstacle_missing == 0
            and self._pace_latched_target == self.COMPOUND_TARGET_SPEED
            and self._pace_command_target == self.COMPOUND_TARGET_SPEED
            and self._pace_sweep >= self.COMPOUND_SWEEP_THRESHOLD
        )
        if visible_compound:
            return base_brake
        return super()._curve_brake_envelope(
            excess=excess, base_brake=base_brake,
        )


class _ObservedRoadTargetController(_CompoundClearingBrakeCarryController):
    """Use observed road geometry when the steering reference row drops out.

    A partially visible bend can retain three road rows while row42 leaves the
    image. Replacing that row with image center invents a straight-ahead target
    and can reverse the requested turn. Use the observed path's interpolated or
    clamped endpoint instead, without inventing another detected road row.
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
            not straight
            and 42 not in centers
            and 54 in centers
            and len(centers) >= 3
            and min(centers) <= 46
        ):
            far = self._center_at(42, centers)
            near = float(centers[54])
            steering = 0.016 * (far - self.IMAGE_CENTER) + 0.012 * (far - near)
        return super()._adjust_road_steering(
            steering=steering,
            straight=straight,
            centers=centers,
            obstacle=obstacle,
        )


class _ObservedRoadSideCommitController(_ObservedRoadTargetController):
    """Keep the chosen passing side once a detected obstacle becomes near.

    Use the existing row44 near-obstacle boundary. The inherited detector-miss
    latch still releases the decision, and distant obstacles may reselect it.
    """

    SIDE_SWITCH_ROW = 44.0


class _ObservedCurveArbitrationController(_ObservedRoadSideCommitController):
    """Retain opposing curve steering only when the observed bend supports it."""

    def __init__(self, *, cruise_speed: float = 68.0) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._observed_bend_displacement = None

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._observed_bend_displacement = None

    def act(self, observation) -> np.ndarray:
        self._observed_bend_displacement = None
        return super().act(observation)

    def _adjust_road_steering(
        self,
        *,
        steering: float,
        straight: bool,
        centers: dict[int, float],
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        self._observed_bend_displacement = None
        if len(centers) >= 2:
            self._observed_bend_displacement = float(
                centers[min(centers)] - centers[max(centers)]
            )
        return super()._adjust_road_steering(
            steering=steering, straight=straight, centers=centers, obstacle=obstacle
        )

    def _adjust_obstacle_steering(
        self, *, base_steering: float, obstacle_bias: float, straight: bool,
    ) -> float:
        unsupported_turn = (
            not straight
            and base_steering * obstacle_bias < 0.0
            and self._observed_bend_displacement is not None
            and base_steering * self._observed_bend_displacement <= 0.0
        )
        # The inherited straight arbitration branch is additive. Selecting it
        # here also preserves the parent's previous-preview transition guard;
        # the actual road classification and longitudinal state stay intact.
        return super()._adjust_obstacle_steering(
            base_steering=base_steering,
            obstacle_bias=obstacle_bias,
            straight=straight or unsupported_turn,
        )


class _ObservedCenterlineArbitrationController(_ObservedCurveArbitrationController):
    """Begin opposing avoidance only when the observed road supports it.

    A tiny far-near bend is insufficient if the observed centerline remains on
    the other side of the vehicle. Once a supported pass begins, preserve its
    direction while a matching avoidance side remains continuously visible.
    """

    MIN_OBSERVED_CROSSING = 1.0

    def __init__(self, *, cruise_speed: float = 68.0) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._observed_far_center = None
        self._centerline_override_side = 0.0

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._observed_far_center = None
        self._centerline_override_side = 0.0

    def act(self, observation) -> np.ndarray:
        self._observed_far_center = None
        action = super().act(observation)
        if not self.road_visible:
            self._centerline_override_side = 0.0
        return action

    def _adjust_road_steering(
        self,
        *,
        steering: float,
        straight: bool,
        centers: dict[int, float],
        obstacle: tuple[float, float, float] | None,
    ) -> float:
        self._observed_far_center = (
            float(centers[min(centers)]) if centers else None
        )
        if obstacle is None:
            self._centerline_override_side = 0.0
        return super()._adjust_road_steering(
            steering=steering, straight=straight, centers=centers, obstacle=obstacle
        )

    def _adjust_obstacle_steering(
        self, *, base_steering: float, obstacle_bias: float, straight: bool,
    ) -> float:
        side = float(np.sign(obstacle_bias))
        if side != self._centerline_override_side:
            self._centerline_override_side = 0.0
        supported = bool(
            not straight
            and base_steering * obstacle_bias < 0.0
            and self._observed_far_center is not None
            and self._observed_bend_displacement is not None
            and side * (self._observed_far_center - self.IMAGE_CENTER)
            >= self.MIN_OBSERVED_CROSSING
            and side * self._observed_bend_displacement
            >= self.MIN_OBSERVED_CROSSING
        )
        if supported:
            self._centerline_override_side = side
        allow_override = bool(
            not straight
            and base_steering * obstacle_bias < 0.0
            and side == self._centerline_override_side
        )
        # Call the pre-arbitration parent to retain its preview transition
        # guard without re-enabling the rejected flat-bend rule.
        return _ObservedRoadSideCommitController._adjust_obstacle_steering(
            self,
            base_steering=base_steering,
            obstacle_bias=obstacle_bias,
            straight=straight or allow_override,
        )


class _ObservedMarginArbitrationController(_ObservedCenterlineArbitrationController):
    """Require visible side-edge room before initiating opposing avoidance."""

    MIN_PASSAGE_MARGIN = 7.0

    def __init__(self, *, cruise_speed: float = 68.0) -> None:
        super().__init__(cruise_speed=cruise_speed)
        self._observed_side_margins = None

    def reset(self, observation=None) -> None:
        super().reset(observation)
        self._observed_side_margins = None

    def act(self, observation) -> np.ndarray:
        self._observed_side_margins = None
        return super().act(observation)

    @classmethod
    def _ego_road_margins(
        cls, frame: np.ndarray, centers: dict[int, float], obstacle_y: float,
    ) -> tuple[float, float] | None:
        if not centers:
            return None
        # A near sampled row is preferable when obstacle_y lies between rows.
        row = min(centers, key=lambda y: (abs(y - obstacle_y), -y))
        asphalt = (frame[row] >= cls.ROAD_LOW) & (frame[row] <= cls.ROAD_HIGH)
        pivot = next((x for x in (41, 42) if asphalt[x]), None)
        if pivot is None:
            return None
        left = right = pivot
        while left > 0 and asphalt[left - 1]:
            left -= 1
        while right < frame.shape[1] - 1 and asphalt[right + 1]:
            right += 1
        return cls.IMAGE_CENTER - left, right - cls.IMAGE_CENTER

    def _nearest_obstacle(
        self,
        frame: np.ndarray,
        centers: dict[int, float],
        spans: dict[int, tuple[float, float]] | None = None,
    ) -> tuple[float, float, float] | None:
        obstacle = super()._nearest_obstacle(frame, centers, spans)
        self._observed_side_margins = (
            self._ego_road_margins(frame, centers, obstacle[0])
            if obstacle is not None else None
        )
        return obstacle

    def _adjust_obstacle_steering(
        self, *, base_steering: float, obstacle_bias: float, straight: bool,
    ) -> float:
        side = float(np.sign(obstacle_bias))
        if side != self._centerline_override_side:
            self._centerline_override_side = 0.0
        if self._centerline_override_side == 0.0:
            margin = None
            if self._observed_side_margins is not None:
                margin = self._observed_side_margins[1 if side > 0.0 else 0]
            if margin is None or margin < self.MIN_PASSAGE_MARGIN:
                return _ObservedRoadSideCommitController._adjust_obstacle_steering(
                    self,
                    base_steering=base_steering,
                    obstacle_bias=obstacle_bias,
                    straight=straight,
                )
        return super()._adjust_obstacle_steering(
            base_steering=base_steering,
            obstacle_bias=obstacle_bias,
            straight=straight,
        )


class _ObservedEgoSideSwitchController(_ObservedMarginArbitrationController):
    """Reconsider a nearby pass only when the obstacle is across the ego."""

    EARLY_SIDE_SWITCH_ROW = 38.0
    LATE_SIDE_SWITCH_ROW = 52.0

    def _allow_obstacle_side_switch(
        self,
        *,
        obstacle_y: float,
        obstacle_x: float,
        candidate_side: float,
    ) -> bool:
        if obstacle_y < self.EARLY_SIDE_SWITCH_ROW:
            return True
        return bool(
            obstacle_y < self.LATE_SIDE_SWITCH_ROW
            and candidate_side * (self.IMAGE_CENTER - obstacle_x) > 0.0
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


class _FeasibleCorridorSideVetoPaceController(_FeasibleCorridorRoadDropoutController):
    """Keep a visibly open obstacle side and use a modest no-corridor pace."""

    NO_CORRIDOR_TARGET_SPEED = 20.0
    SIDE_VETO_ROAD_ROWS = 3
    SIDE_VETO_ROW_RADIUS = 13.0

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
        bbox = self._corridor_bbox
        if (
            not inherited
            or current_side == 0.0
            or candidate_side == current_side
            or bbox is None
        ):
            return inherited
        if (
            self._corridor_candidate(-1.0) is not None
            or self._corridor_candidate(1.0) is not None
        ):
            return inherited
        nearest_rows = sorted(
            (
                row for row in self._corridor_edges
                if abs(row - obstacle_y) <= self.SIDE_VETO_ROW_RADIUS
            ),
            key=lambda row: (abs(row - obstacle_y), row),
        )[:self.SIDE_VETO_ROAD_ROWS]
        if (
            len(nearest_rows) < self.SIDE_VETO_ROAD_ROWS
            or max(nearest_rows) - min(nearest_rows)
            > self.SIDE_VETO_ROAD_ROWS - 1
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


class _RacingLineController(_ForwardCorridorController):
    """Fresh F1-inspired controller for the bare baseline checkpoint.

    This controller intentionally does not accumulate the old collection of
    curve-specific steering branches.  Each frame follows the same pipeline:
    perceive the asphalt envelope, select a preview point, plan a speed from
    curvature/obstacle distance, then apply one geometric safety envelope.
    The parent class supplies only camera parsing, road geometry, obstacle
    components, and bounded recovery primitives; its former ``act`` policy is
    not used by the Agent runtime.
    """

    MAX_GAS = 0.18
    STRAIGHT_CRUISE_SPEED = 72.0
    MAX_STEER = 0.44
    CURVE_MAX_STEER = 0.36
    MAX_STEER_STEP = 0.06
    CURVE_STEER_STEP = 0.05
    CURVE_SPEED_FLOOR = 30.0
    CURVE_SPEED_PENALTY = 2.25
    CURVE_ENTRY_SPEED_PENALTY = 0.75
    CURVE_GAS_MIN = 0.48
    SHARP_CURVE_SWEEP = 8.0
    SHARP_CURVE_SPEED_FLOOR = 24.0
    SHARP_CURVE_STEER_GAIN = 0.64
    SHARP_CURVE_STEER_LIMIT = 0.32
    SHARP_CURVE_STEER_STEP = 0.04
    SHARP_CURVE_GAS_MIN = 0.32
    # Move toward the wider edge before a straight-road obstacle becomes
    # urgent, but stop short of the asphalt boundary by the vehicle margin.
    OBSTACLE_EDGE_FRACTION = 0.78
    OBSTACLE_EDGE_GAIN = 0.025
    HAZARD_TARGET_SPEED = 22.0
    HAZARD_BRAKE_FRAMES = 2
    HAZARD_CRAWL_GAS = 0.04
    SPEED_WATCHDOG_GAS = 0.04
    SPEED_BRAKE_FRAMES = 7

    def __init__(self, *, cruise_speed: float = STRAIGHT_CRUISE_SPEED) -> None:
        super().__init__(cruise_speed=cruise_speed)

    @classmethod
    def _suppress_outward_steering(
        cls,
        steering: float,
        spans: dict[int, tuple[float, float]],
    ) -> float:
        """Remove the final command component that points beyond the track."""
        if not spans:
            return float(steering)
        near_span = spans.get(54) or spans.get(50)
        if near_span is None:
            return float(steering)
        left = near_span[0] + cls.MIN_EDGE_CLEARANCE
        right = near_span[1] - cls.MIN_EDGE_CLEARANCE
        target = float(np.clip(cls.IMAGE_CENTER, left, right))
        envelope_error = target - cls.IMAGE_CENTER
        if envelope_error > 0.25 and steering < 0.0:
            return 0.0
        if envelope_error < -0.25 and steering > 0.0:
            return 0.0
        return float(steering)

    @classmethod
    def _straight_obstacle_edge_bias(
        cls,
        obstacle: tuple[float, float, float],
        spans: dict[int, tuple[float, float]],
        side: float,
    ) -> float:
        """Aim toward the wider safe edge before a straight-road obstacle."""
        if not spans or side == 0.0:
            return 0.0
        near_span = spans.get(54) or spans.get(50)
        if near_span is None:
            return 0.0
        left = near_span[0] + cls.MIN_EDGE_CLEARANCE
        right = near_span[1] - cls.MIN_EDGE_CLEARANCE
        if left >= right:
            return 0.0
        safe_width = right - left
        target = (
            left + cls.OBSTACLE_EDGE_FRACTION * safe_width
            if side > 0.0
            else right - cls.OBSTACLE_EDGE_FRACTION * safe_width
        )
        del obstacle
        return float(cls.OBSTACLE_EDGE_GAIN * (target - cls.IMAGE_CENTER))

    def act(self, observation) -> np.ndarray:
        frame = self._frame(observation)
        if frame is None:
            self.road_visible = False
            if self.has_seen_road:
                return self._lost_road_action()
            return np.zeros(3, dtype=np.float32)

        centers, spans = self._road_geometry(frame)
        self.road_visible = len(centers) >= 3
        if not self.road_visible:
            if not self.has_seen_road:
                return np.zeros(3, dtype=np.float32)
            recovery_centers, recovery_spans = self._wide_road_geometry(frame)
            if len(recovery_centers) >= 2:
                self._remember_road(recovery_centers, recovery_spans)
                return self._lost_road_action(recovery_centers)
            return self._lost_road_action()

        self.has_seen_road = True
        self._recovery_frames = 0
        self._remember_road(centers, spans)

        # Racing-line preview: use a mid point for turn-in and a farther point
        # for heading anticipation, both clipped to the visible body-safe road.
        near = self._safe_center_at(54.0, centers, spans)
        mid = self._safe_center_at(38.0, centers, spans)
        far = self._safe_center_at(22.0, centers, spans)
        heading = far - near
        preview_error = 0.58 * (mid - self.IMAGE_CENTER) + 0.42 * (
            far - self.IMAGE_CENTER
        )
        road_sweep = self._road_sweep(centers)
        entry_sweep = abs(far - mid)
        curve_strength = max(road_sweep, entry_sweep)
        curve_mode = curve_strength >= self.CURVE_TRIGGER_SWEEP
        sharp_curve = curve_strength >= self.SHARP_CURVE_SWEEP

        corridor_hazard, corridor_blocked, corridor_hint = self._corridor_hazard(
            frame, centers, spans
        )
        if corridor_hazard:
            self._hazard_frames += 1
        else:
            self._hazard_frames = 0

        # One preview-based path tracker.  The geometric envelope below is
        # applied after obstacle and shoulder proposals, so no racing-line
        # shortcut can point outside the asphalt.
        steering = 0.017 * preview_error + 0.010 * heading
        if curve_mode:
            steering *= self.SHARP_CURVE_STEER_GAIN if sharp_curve else 0.78
        steer_limit = self.CURVE_MAX_STEER if curve_mode else self.MAX_STEER
        if sharp_curve:
            steer_limit = min(steer_limit, self.SHARP_CURVE_STEER_LIMIT)

        target_speed = self.cruise_speed
        target_speed -= (
            self.CURVE_SPEED_PENALTY * (1.15 if sharp_curve else 1.0) * road_sweep
        )
        target_speed -= self.CURVE_ENTRY_SPEED_PENALTY * entry_sweep
        speed_floor = (
            self.SHARP_CURVE_SPEED_FLOOR if sharp_curve else self.CURVE_SPEED_FLOOR
        )
        target_speed = float(
            np.clip(target_speed, speed_floor, self.cruise_speed)
        )

        obstacle = self._nearest_obstacle(frame, centers, spans)
        if obstacle is not None:
            obstacle_y, obstacle_x, road_center = obstacle
            obstacle_span = self._span_at(float(obstacle_y), spans)
            if obstacle_span is not None:
                left_clearance = max(0.0, obstacle_x - obstacle_span[0])
                right_clearance = max(0.0, obstacle_span[1] - obstacle_x)
                candidate_side = 1.0 if right_clearance >= left_clearance else -1.0
            else:
                candidate_side = 1.0 if obstacle_x < road_center else -1.0
            if self._obstacle_side == 0.0 or obstacle_y < 48.0:
                self._obstacle_side = candidate_side
            self._obstacle_missing = 0
            urgency = float(np.clip((obstacle_y - 22.0) / 24.0, 0.0, 1.0))
            steering += self._obstacle_side * 0.22 * urgency
            straight_context = (
                abs(heading) <= self.STRAIGHT_SWEEP_DEADBAND
                and road_sweep <= self.STRAIGHT_SWEEP_DEADBAND
                and abs(mid - near) <= self.STRAIGHT_SWEEP_DEADBAND
            )
            if straight_context:
                # Start the lateral move while the obstacle is still distant;
                # the target is a safe inner-edge point, not the green pixels.
                steering += self._straight_obstacle_edge_bias(
                    obstacle, spans, self._obstacle_side
                )
            target_speed = min(target_speed, self._obstacle_speed_limit(obstacle_y))
        elif self._obstacle_side != 0.0:
            self._obstacle_missing += 1
            if self._obstacle_missing > self.OBSTACLE_MISS_LIMIT:
                self._obstacle_side = 0.0
                self._obstacle_missing = 0

        if corridor_hazard:
            steering += 0.16 * corridor_hint
            target_speed = min(target_speed, self.HAZARD_TARGET_SPEED)
            if corridor_blocked:
                steering = 0.20 * corridor_hint
                steer_limit = min(steer_limit, 0.25)

        steering = self._track_bound_steering(steering, spans)
        if self._target_speed is not None:
            # Brake promptly for a corner, but restore straight-line speed
            # quickly once curvature has fallen instead of carrying stale
            # conservative targets around the whole lap.
            blend = 0.35 if target_speed < self._target_speed else 0.20
            target_speed = blend * self._target_speed + (1.0 - blend) * target_speed
        self._target_speed = target_speed

        speed = self._estimate_speed(frame)
        gas, brake = self._pedals(speed, target_speed)
        if corridor_hazard:
            if self._hazard_frames <= self.HAZARD_BRAKE_FRAMES:
                gas = 0.0
                brake = max(brake, self.HAZARD_BRAKE)
            else:
                gas = max(self.HAZARD_CRAWL_GAS, min(self.MAX_GAS, gas))
                brake = 0.0
        elif gas <= 0.0 and brake > 0.0:
            self._brake_frames += 1
            if self._brake_frames > self.SPEED_BRAKE_FRAMES:
                gas, brake = self.SPEED_WATCHDOG_GAS, 0.0
        else:
            self._brake_frames = 0

        if corridor_blocked and self._hazard_frames <= self.HAZARD_BRAKE_FRAMES:
            gas = 0.0
            brake = max(brake, self.HAZARD_BRAKE)
        if obstacle is not None:
            # Reserve longitudinal margin for the selected escape side.
            gas = 0.0
        if curve_mode and not corridor_hazard and obstacle is None and brake <= 0.0:
            curve_gas_scale = float(
                np.clip(
                    1.0 - 0.045 * curve_strength,
                    self.SHARP_CURVE_GAS_MIN if sharp_curve else self.CURVE_GAS_MIN,
                    1.0,
                )
            )
            gas = min(gas, self.MAX_GAS * curve_gas_scale)

        steering = float(np.clip(steering, -steer_limit, steer_limit))
        if self._last_steer != 0.0 and steering * self._last_steer < 0.0:
            steering = 0.0
        if sharp_curve:
            step = self.SHARP_CURVE_STEER_STEP
        else:
            step = self.CURVE_STEER_STEP if curve_mode else self.MAX_STEER_STEP
        steering = self._last_steer + float(
            np.clip(steering - self._last_steer, -step, step)
        )
        steering = float(np.clip(steering, -steer_limit, steer_limit))
        steering = self._suppress_outward_steering(steering, spans)
        self._last_steer = steering
        brake = min(
            brake,
            float(np.nextafter(np.float32(self.MAX_BRAKE), np.float32(0.0))),
        )
        return np.asarray([steering, gas, brake], dtype=np.float32)


class Agent:
    """Load either the root baseline model or the packaged HAIC visual policy."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        project_root: str | None = None,
        policy=None,
        dynamics=None,
        planner=None,
        planner_enabled: bool | None = None,
        strict_checkpoint_loading: bool | None = None,
        plan_budget: float | None = None,
        clock=None,
        policy_checkpoint: str | None = None,
        dynamics_checkpoint: str | None = None,
    ):
        resolved_project_root = (
            str(project_root).rstrip("/\\") if project_root is not None else None
        )
        resolved_policy_checkpoint = policy_checkpoint
        if resolved_policy_checkpoint is None and resolved_project_root is not None:
            resolved_policy_checkpoint = f"{resolved_project_root}/{POLICY_MODEL_FILENAME}"
        resolved_dynamics_checkpoint = dynamics_checkpoint
        if resolved_dynamics_checkpoint is None and resolved_project_root is not None:
            resolved_dynamics_checkpoint = f"{resolved_project_root}/{DYNAMICS_MODEL_FILENAME}"
        haic_options_requested = (
            any(
                value is not None
                for value in (
                    policy,
                    dynamics,
                    planner,
                    planner_enabled,
                    strict_checkpoint_loading,
                    clock,
                    policy_checkpoint,
                    dynamics_checkpoint,
                )
            )
            or (plan_budget is not None and float(plan_budget) != 4.5)
        )
        if not haic_options_requested:
            resolved_model_path = model_path
            if resolved_model_path is None:
                resolved_model_path = (
                    f"{resolved_project_root}/{MODEL_FILENAME}"
                    if resolved_project_root is not None
                    else MODEL_FILENAME
                )
            try:
                payload = torch.load(
                    resolved_model_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except FileNotFoundError:
                if model_path is not None:
                    raise
                # The inference-only HAIC archive contains policy.pt instead of
                # the root baseline model.pt.
                self._init_haic(
                    policy_checkpoint=resolved_policy_checkpoint,
                    dynamics_checkpoint=resolved_dynamics_checkpoint,
                )
            else:
                model_format = payload.get("format") if isinstance(payload, dict) else None
                if model_format == DRQ_ACTOR_FORMAT:
                    self._init_drq(payload)
                elif model_format is not None:
                    raise ValueError(f"unsupported model format: {model_format}")
                else:
                    self._init_baseline(payload)
            return
        self._init_haic(
            policy=policy,
            dynamics=dynamics,
            planner=planner,
            planner_enabled=planner_enabled,
            strict_checkpoint_loading=strict_checkpoint_loading,
            plan_budget=plan_budget,
            clock=clock,
            policy_checkpoint=resolved_policy_checkpoint,
            dynamics_checkpoint=resolved_dynamics_checkpoint,
        )

    def _init_drq(self, payload):
        # These helpers are intentionally lazy: HAIC-only submissions do not
        # include the baseline action-contract modules.
        from action_smoothing import normalize_action_control, normalize_action_smoothing
        from action_representation import normalize_action_representation

        config = payload.get("config", {})
        observation_spec = payload.get("observation_spec", {})
        action_spec = payload.get("action_spec", {})
        expected_observation = {
            "shape": (4, 84, 84), "dtype": "float32", "channel_order": "CHW",
            "low": 0.0, "high": 1.0, "uint8_scale": 255,
            "control_plane_fingerprint": None,
        }
        expected_action = {
            "native_low": (-1.0, -1.0, -1.0), "native_high": (1.0, 1.0, 1.0),
            "official_low": (-1.0, 0.0, 0.0), "official_high": (1.0, 1.0, 1.0),
            "frame_skip": 4, "order": ("steer", "gas", "brake"),
            "method": "symmetric-native-to-haic-box",
        }
        for name, actual, expected in (
            ("observation", observation_spec, expected_observation),
            ("action", action_spec, expected_action),
        ):
            if not isinstance(actual, dict):
                raise ValueError(f"invalid DrQ {name} spec")
            for key, value in expected.items():
                recorded = actual.get(key)
                if isinstance(value, tuple) and isinstance(recorded, (list, tuple)):
                    recorded = tuple(recorded)
                if key not in actual or recorded != value:
                    raise ValueError(f"unsupported DrQ {name} spec: {key}")
        if (
            not isinstance(config, dict)
            or config.get("observation_shape") not in ((4, 84, 84), [4, 84, 84])
            or config.get("action_dim") != 3
            or any(
                type(config.get(key)) is not int or config[key] <= 0
                for key in ("feature_dim", "hidden_dim")
            )
        ):
            raise ValueError("invalid DrQ actor architecture config")
        if (
            normalize_action_smoothing(payload.get("action_smoothing"))
            != normalize_action_smoothing()
            or normalize_action_control(payload.get("action_control"))
            != normalize_action_control()
            or normalize_action_representation(payload.get("action_representation"))
            != normalize_action_representation()
        ):
            raise ValueError("DrQ export must use the frozen unsmoothed action contract")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict) or not state_dict or any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
            for value in state_dict.values()
        ):
            raise ValueError("invalid or non-finite DrQ actor state")
        for key, shape in (
            ("encoder.linear.1.weight", (config["feature_dim"], 512)),
            ("trunk.0.weight", (config["hidden_dim"], config["feature_dim"])),
            ("policy.weight", (3, config["hidden_dim"])),
        ):
            if key not in state_dict or tuple(state_dict[key].shape) != shape:
                raise ValueError(f"DrQ actor config does not match state: {key}")
        with torch.random.fork_rng(devices=[]):
            self.model = DrQActor(config["feature_dim"], config["hidden_dim"])
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.format = DRQ_ACTOR_FORMAT
        self._runtime_mode = "drq"
        self.export_metadata = {
            key: value for key, value in payload.items() if key != "state_dict"
        }
        self.reset(None)

    def _init_baseline(self, payload):
        # Keep these imports local: the HAIC-only archive does not ship the
        # baseline helpers, and never needs them when loading policy.pt.
        from action_smoothing import (
            append_action_control_plane,
            action_control_fingerprint,
            action_smoothing_fingerprint,
            build_action_smoother,
            normalize_action_smoothing,
            normalize_action_control,
        )
        from action_representation import (
            action_representation_fingerprint,
            map_policy_action,
            normalize_action_representation,
        )

        # A bare OrderedDict is the original baseline export.  That actor has
        # no longitudinal/direction safety contract, so route it through the
        # source-pinned completion controller below.  Explicit exports retain
        # their recorded action contract and continue to use the neural actor.
        use_forward_controller = not (
            isinstance(payload, dict) and "state_dict" in payload
        )
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
            action_smoothing = payload.get("action_smoothing")
            smoothing_fingerprint = payload.get("action_smoothing_fingerprint")
            action_control = payload.get("action_control")
            control_fingerprint = payload.get("action_control_fingerprint")
            input_channels = payload.get("input_channels", 4)
            action_representation = payload.get("action_representation")
            representation_fingerprint = payload.get("action_representation_fingerprint")
        else:
            state_dict = payload
            action_smoothing = None
            smoothing_fingerprint = None
            action_control = None
            control_fingerprint = None
            input_channels = 4
            action_representation = None
            representation_fingerprint = None
        action_smoothing = normalize_action_smoothing(action_smoothing)
        action_control = normalize_action_control(action_control)
        action_representation = normalize_action_representation(action_representation)
        if (
            smoothing_fingerprint is not None
            and smoothing_fingerprint != action_smoothing_fingerprint(action_smoothing)
        ):
            raise ValueError("model action smoothing fingerprint does not match config")
        if (
            control_fingerprint is not None
            and control_fingerprint != action_control_fingerprint(action_control)
        ):
            raise ValueError("model action control fingerprint does not match config")
        if int(input_channels) != action_control["input_channels"]:
            raise ValueError("model input channels do not match action control config")
        if (
            representation_fingerprint is not None
            and representation_fingerprint != action_representation_fingerprint(action_representation)
        ):
            raise ValueError("model action representation fingerprint does not match config")
        action_dimensions = (
            sum(action_representation["nvec"])
            if action_representation["method"] != "continuous_box"
            else 3
        )
        self.model = Baseline1Actor(
            input_channels=int(input_channels), action_dimensions=action_dimensions
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.action_smoothing = action_smoothing
        self.action_control = action_control
        self.action_representation = action_representation
        self._append_action_control_plane = append_action_control_plane
        self._map_policy_action = map_policy_action
        self.smoother = build_action_smoother(self.action_smoothing)
        self._forward_controller = (
            _CompoundClearingBrakeCarryController()
            if use_forward_controller
            else None
        )
        self._runtime_mode = "baseline"
        self.format = None
        self.reset(None)

    def _init_haic(
        self,
        *,
        policy=None,
        dynamics=None,
        planner=None,
        planner_enabled: bool | None = None,
        strict_checkpoint_loading: bool | None = None,
        plan_budget: float | None = None,
        clock=None,
        policy_checkpoint: str | None = None,
        dynamics_checkpoint: str | None = None,
    ):
        from haic_agent.dynamics import LatentDynamicsEnsemble
        from haic_agent.networks import VisualActorCritic
        from haic_agent.planner import CEMPlanner
        from haic_agent.runtime_config import (
            PLANNER_ENABLED,
            PLANNER_SETTINGS,
            STRICT_CHECKPOINT_LOADING,
        )

        resolved_clock = time.monotonic if clock is None else clock
        self.clock = resolved_clock
        self.strict_checkpoint_loading = (
            STRICT_CHECKPOINT_LOADING
            if strict_checkpoint_loading is None
            else bool(strict_checkpoint_loading)
        )
        requested_plan_budget = float(4.5 if plan_budget is None else plan_budget)
        if not math.isfinite(requested_plan_budget):
            raise ValueError("plan_budget must be finite")
        self.plan_budget = min(max(requested_plan_budget, 0.0), 4.5)
        resolved_policy_checkpoint = (
            POLICY_MODEL_FILENAME if policy_checkpoint is None else policy_checkpoint
        )
        configured_planner_enabled = (
            PLANNER_ENABLED if planner_enabled is None else bool(planner_enabled)
        )
        resolved_dynamics_checkpoint = (
            DYNAMICS_MODEL_FILENAME
            if dynamics_checkpoint is None and configured_planner_enabled
            else dynamics_checkpoint
        )
        self.policy = policy if policy is not None else self._load_policy(
            resolved_policy_checkpoint,
            strict=self.strict_checkpoint_loading,
        )
        self.planner_enabled = configured_planner_enabled
        self.dynamics = None
        self._visual_actor_critic_class = VisualActorCritic
        if self.planner_enabled:
            self.dynamics = (
                dynamics
                if dynamics is not None
                else self._load_dynamics(
                    resolved_dynamics_checkpoint,
                    strict=self.strict_checkpoint_loading,
                )
            )
        self.planner = (
            planner if planner is not None else CEMPlanner(**PLANNER_SETTINGS, clock=resolved_clock)
        )
        if self.policy is not None and hasattr(self.policy, "eval"):
            self.policy.eval()
        if self.dynamics is not None and hasattr(self.dynamics, "eval"):
            self.dynamics.eval()
        self._runtime_mode = "haic"

    @staticmethod
    def _load_policy_with_status(checkpoint: str | None, *, strict: bool = False):
        from haic_agent.networks import VisualActorCritic

        if checkpoint is None:
            if strict:
                raise RuntimeError("failed to load required policy checkpoint: None")
            return None, False
        try:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = saved.get("model_state") if isinstance(saved, dict) else None
            if isinstance(state, dict):
                policy = VisualActorCritic()
                policy.load_state_dict(state, strict=True)
                return policy, True
            if strict:
                raise ValueError("policy checkpoint lacks model_state")
        except (
            FileNotFoundError,
            RuntimeError,
            ValueError,
            OSError,
            pickle.UnpicklingError,
            EOFError,
        ) as error:
            if strict:
                raise RuntimeError(
                    f"failed to load required policy checkpoint: {checkpoint}"
                ) from error
        return None, False

    @staticmethod
    def _load_policy(checkpoint: str | None, *, strict: bool = False):
        policy, _loaded = Agent._load_policy_with_status(checkpoint, strict=strict)
        if policy is not None:
            return policy
        from haic_agent.networks import VisualActorCritic

        return VisualActorCritic()

    @staticmethod
    def _load_dynamics(checkpoint: str | None, *, strict: bool = False):
        from haic_agent.dynamics import LatentDynamicsEnsemble

        if checkpoint is None:
            return None
        dynamics = LatentDynamicsEnsemble()
        try:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = saved.get("model") if isinstance(saved, dict) else None
            if isinstance(state, dict):
                dynamics.load_state_dict(state, strict=True)
                return dynamics
            if strict:
                raise ValueError("dynamics checkpoint lacks model")
        except (
            FileNotFoundError,
            RuntimeError,
            ValueError,
            OSError,
            pickle.UnpicklingError,
            EOFError,
        ) as error:
            if strict:
                raise RuntimeError(
                    f"failed to load required dynamics checkpoint: {checkpoint}"
                ) from error
        return None

    @staticmethod
    def _safe_action(action) -> np.ndarray | None:
        try:
            candidate = np.asarray(action, dtype=np.float32).reshape(3)
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(candidate)):
            return None
        return np.clip(candidate, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]).astype(
            np.float32, copy=False
        )

    @staticmethod
    def _observation_tensor(observation):
        try:
            pixels = np.asarray(observation, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            return None
        if pixels.shape != (4, 84, 84) or not np.all(np.isfinite(pixels)):
            return None
        if float(pixels.min()) < 0.0 or float(pixels.max()) > 1.0:
            return None
        return torch.from_numpy(pixels).unsqueeze(0)

    def reset(self, observation):
        """Clear the state owned by the selected runtime."""
        if self._runtime_mode == "drq":
            # The deterministic actor is feed-forward.
            return
        if self._runtime_mode == "baseline":
            self.smoother.reset(initial_action=self.action_smoothing["initial_action"])
            if self._forward_controller is not None:
                self._forward_controller.reset(observation)
            return
        reset = getattr(self.planner, "reset", None)
        if callable(reset):
            reset()
        del observation

    @torch.inference_mode()
    def act(self, observation) -> np.ndarray:
        if self._runtime_mode == "drq":
            observation = np.asarray(observation)
            if (
                observation.shape != (4, 84, 84)
                or observation.dtype != np.float32
                or not np.isfinite(observation).all()
                or np.any(observation < 0.0)
                or np.any(observation > 1.0)
            ):
                raise ValueError(
                    "DrQ observation must be float32 CHW (4, 84, 84) in [0, 1]"
                )
            native = self.model(
                torch.as_tensor(np.ascontiguousarray(observation)).unsqueeze(0)
            ).squeeze(0).numpy()
            action = np.clip(native, -1.0, 1.0).astype(np.float32)
            action[1:] = (action[1:] + 1.0) * 0.5
            if not np.isfinite(action).all():
                raise ValueError("DrQ actor produced a non-finite action")
            return action
        if self._runtime_mode == "baseline":
            if self._forward_controller is not None:
                controlled = self._forward_controller.act(observation)
                if (
                    self._forward_controller.road_visible
                    or self._forward_controller.has_seen_road
                ):
                    return controlled
            controlled = self._append_action_control_plane(
                observation,
                self.action_control,
                self.smoother.last_action,
            )
            state = torch.as_tensor(controlled).unsqueeze(0)
            raw_action = self.model(state).squeeze(0).numpy()
            if self.action_representation["method"] != "continuous_box":
                steering_dimensions, longitudinal_dimensions = self.action_representation["nvec"]
                raw_action = self._map_policy_action(
                    np.asarray(
                        (
                            np.argmax(raw_action[:steering_dimensions]),
                            np.argmax(raw_action[
                                steering_dimensions:steering_dimensions + longitudinal_dimensions
                            ]),
                        ),
                        dtype=np.int64,
                    ),
                    self.action_representation,
                )
            else:
                raw_action = np.asarray(
                    [
                        np.clip(raw_action[0], -1.0, 1.0),
                        np.clip(raw_action[1], 0.0, 1.0),
                        np.clip(raw_action[2], 0.0, 1.0),
                    ],
                    dtype=np.float32,
                )
            action = self.smoother.smooth(raw_action)
            return np.asarray(action, dtype=np.float32)

        """Return a finite action before the end-to-end 4.5 second deadline."""
        deadline = self.clock() + self.plan_budget
        no_op = np.zeros(3, dtype=np.float32)
        state = self._observation_tensor(observation)
        if state is None:
            return no_op
        try:
            latent = self.policy.encode_observation(state)
            policy_mean, policy_log_std = self.policy.action_parameters(latent)
            fallback = self._visual_actor_critic_class._bound_actions(policy_mean)
            action = self._safe_action(fallback.squeeze(0).cpu().numpy())
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return no_op
        if action is None:
            return no_op
        if not self.planner_enabled or self.dynamics is None or self.clock() >= deadline:
            return action
        try:
            result = self.planner.plan(
                latent,
                policy_mean,
                policy_log_std,
                self.dynamics,
                deadline=deadline,
            )
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return action
        if result is None or self.clock() >= deadline:
            return action
        planned_action = self._safe_action(result.action)
        return action if planned_action is None else planned_action
