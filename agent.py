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
                if not (4 <= area <= 80 and 2 <= box_width <= 9 and 2 <= box_height <= 10):
                    continue
                center_x = sum_x / area
                center_y = sum_y / area
                road_center = self._center_at(center_y, centers)
                if abs(center_x - road_center) <= 24.0:
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
        # forward corridor controller below.  Explicit exports retain their
        # recorded action contract and continue to use the neural actor.
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
            _ForwardCorridorController() if use_forward_controller else None
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
