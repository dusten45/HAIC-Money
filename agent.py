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
