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
    STRAIGHT_SWEEP_DEADBAND = 1.5
    MAX_STEER_STEP = 0.07
    MAX_GAS = 0.08
    MAX_BRAKE = 0.28
    OBSTACLE_MISS_LIMIT = 4
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

    def _nearest_obstacle(
        self, frame: np.ndarray, centers: dict[int, float]
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
    def _estimate_speed(cls, frame: np.ndarray) -> float:
        top, bottom, left, right = cls.SPEED_ROI
        mass = float(frame[top:bottom, left:right].sum())
        return float(np.clip((mass - cls.SPEED_BASELINE) / cls.SPEED_PER_UNIT, 0.0, 80.0))

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

    def _pedals(self, speed: float, target_speed: float) -> tuple[float, float]:
        if speed > target_speed + 1.0:
            brake = float(np.clip((speed - target_speed) * 0.012, 0.04, self.MAX_BRAKE))
            return 0.0, brake
        if speed < target_speed - 8.0:
            return self.MAX_GAS, 0.0
        if speed < target_speed - 3.0:
            return self.MAX_GAS * (2.0 / 3.0), 0.0
        return self.MAX_GAS * (5.0 / 12.0), 0.0

    def act(self, observation) -> np.ndarray:
        frame = self._frame(observation)
        if frame is None:
            self.road_visible = False
            if self.has_seen_road:
                self._last_steer = float(
                    self._last_steer
                    + np.clip(-self._last_steer, -self.MAX_STEER_STEP, self.MAX_STEER_STEP)
                )
                return np.asarray(
                    [self._last_steer, 0.0, min(self.MAX_BRAKE, 0.06)],
                    dtype=np.float32,
                )
            return np.zeros(3, dtype=np.float32)

        centers = self._road_centers(frame)
        self.road_visible = len(centers) >= 3
        if not self.road_visible:
            if self.has_seen_road:
                # If the corridor temporarily disappears, brake gently and
                # decay the last steer instead of asking the neural baseline
                # to make an unconstrained recovery turn.
                self._last_steer = float(
                    self._last_steer
                    + np.clip(-self._last_steer, -self.MAX_STEER_STEP, self.MAX_STEER_STEP)
                )
                return np.asarray(
                    [self._last_steer, 0.0, min(self.MAX_BRAKE, 0.06)],
                    dtype=np.float32,
                )
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
            # Quantization noise on a straight road should not create a
            # persistent weave.  A small residual correction remains only
            # outside the deadband.
            steering = 0.0
            steer_limit = self.STRAIGHT_MAX_STEER
        else:
            steering = 0.016 * center_offset + 0.012 * (far - near)
            steer_limit = self.MAX_STEER
        target_speed = float(
            np.clip(self.cruise_speed - 2.0 * road_sweep, 36.0, self.cruise_speed)
        )

        obstacle = self._nearest_obstacle(frame, centers)
        if obstacle is not None:
            obstacle_y, obstacle_x, road_center = obstacle
            # A straight road normally has a tight steering limit; a detected
            # obstacle is the explicit exception, but still stays well below
            # a drift-sized command.
            steer_limit = min(self.MAX_STEER, 0.32)
            side_offset = obstacle_x - road_center
            candidate_side = 1.0 if side_offset < 0.0 else -1.0
            if self._obstacle_side == 0.0:
                self._obstacle_side = candidate_side
            elif obstacle_y < 52.0:
                self._obstacle_side = candidate_side
            self._last_obstacle_side_offset = side_offset
            urgency = float(np.clip((obstacle_y - 22.0) / 18.0, 0.0, 1.0))
            steering += self._obstacle_side * 0.24 * urgency
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
        if not straight and abs(steering) > 0.28:
            target_speed = min(target_speed, 44.0)
        speed = self._estimate_speed(frame)
        gas, brake = self._pedals(speed, target_speed)

        # Slow down before a bend and rate-limit steering so one noisy frame
        # cannot turn the car around or induce a drift-like correction.
        if not straight and abs(steering) > 0.28:
            gas = min(gas, self.MAX_GAS * 0.5)
        steering = float(np.clip(steering, -steer_limit, steer_limit))
        if (
            self._last_steer != 0.0
            and steering * self._last_steer < 0.0
            and abs(self._last_steer) > self.MAX_STEER_STEP
        ):
            # Cross zero before changing sides.  This prevents an S-curve or
            # one noisy centerline estimate from becoming a snap reversal.
            steering = 0.0
        steering = self._last_steer + float(
            np.clip(steering - self._last_steer, -self.MAX_STEER_STEP, self.MAX_STEER_STEP)
        )
        steering = float(np.clip(steering, -steer_limit, steer_limit))
        self._last_steer = steering
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
