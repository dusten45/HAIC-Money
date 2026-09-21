import time

import numpy as np
import torch
from torch import nn


MODEL_FILENAME = "model.pt"
POLICY_MODEL_FILENAME = "policy.pt"
DYNAMICS_MODEL_FILENAME = "dynamics.pt"

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


class Agent:
    """Load either the root baseline model or the packaged HAIC visual policy."""

    def __init__(
        self,
        *,
        policy=None,
        dynamics=None,
        planner=None,
        planner_enabled: bool | None = None,
        strict_checkpoint_loading: bool | None = None,
        controller_mode: str | None = None,
        plan_budget: float | None = None,
        clock=None,
        policy_checkpoint: str | None = None,
        dynamics_checkpoint: str | None = None,
    ):
        haic_options_requested = (
            any(
                value is not None
                for value in (
                    policy,
                    dynamics,
                    planner,
                    planner_enabled,
                    strict_checkpoint_loading,
                    controller_mode,
                    clock,
                    policy_checkpoint,
                    dynamics_checkpoint,
                )
            )
            or (plan_budget is not None and float(plan_budget) != 4.5)
        )
        if not haic_options_requested:
            try:
                payload = torch.load(
                    MODEL_FILENAME,
                    map_location="cpu",
                    weights_only=True,
                )
            except FileNotFoundError:
                # The inference-only HAIC archive contains policy.pt instead of
                # the root baseline model.pt.
                self._init_haic()
            else:
                self._init_baseline(payload)
            return
        self._init_haic(
            policy=policy,
            dynamics=dynamics,
            planner=planner,
            planner_enabled=planner_enabled,
            strict_checkpoint_loading=strict_checkpoint_loading,
            controller_mode=controller_mode,
            plan_budget=plan_budget,
            clock=clock,
            policy_checkpoint=policy_checkpoint,
            dynamics_checkpoint=dynamics_checkpoint,
        )

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
        self.reset(None)

    def _init_haic(
        self,
        *,
        policy=None,
        dynamics=None,
        planner=None,
        planner_enabled: bool | None = None,
        strict_checkpoint_loading: bool | None = None,
        controller_mode: str | None = None,
        plan_budget: float | None = None,
        clock=None,
        policy_checkpoint: str | None = None,
        dynamics_checkpoint: str | None = None,
    ):
        from haic_agent.corridor_agent import VisionCorridorAgent
        from haic_agent.dynamics import LatentDynamicsEnsemble
        from haic_agent.networks import VisualActorCritic
        from haic_agent.planner import CEMPlanner
        from haic_agent.runtime_config import (
            CONTROLLER_MODE,
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
        self.plan_budget = min(max(float(4.5 if plan_budget is None else plan_budget), 0.0), 4.5)
        resolved_policy_checkpoint = (
            POLICY_MODEL_FILENAME if policy_checkpoint is None else policy_checkpoint
        )
        requested_controller_mode = (
            CONTROLLER_MODE if controller_mode is None else controller_mode
        )
        if requested_controller_mode not in {"auto", "learned", "corridor"}:
            raise ValueError("controller_mode must be 'auto', 'learned', or 'corridor'")
        configured_planner_enabled = (
            PLANNER_ENABLED if planner_enabled is None else bool(planner_enabled)
        )
        resolved_dynamics_checkpoint = (
            DYNAMICS_MODEL_FILENAME
            if dynamics_checkpoint is None and configured_planner_enabled
            else dynamics_checkpoint
        )
        self.controller_mode = requested_controller_mode
        self.corridor_controller = None
        self.policy = None
        self.dynamics = None
        self._visual_actor_critic_class = VisualActorCritic
        if requested_controller_mode == "corridor":
            self.controller_mode = "corridor"
            self.corridor_controller = VisionCorridorAgent()
            self.planner_enabled = False
        else:
            if policy is None:
                self.policy, policy_loaded = self._load_policy_with_status(
                    resolved_policy_checkpoint,
                    strict=self.strict_checkpoint_loading,
                )
            else:
                self.policy = policy
                policy_loaded = True
            if requested_controller_mode == "learned" and not policy_loaded:
                raise RuntimeError("learned controller requires a valid policy checkpoint")
            if requested_controller_mode == "auto" and not policy_loaded:
                self.controller_mode = "corridor"
                self.corridor_controller = VisionCorridorAgent()
                self.policy = None
                self.planner_enabled = False
            else:
                self.controller_mode = "learned"
                self.planner_enabled = configured_planner_enabled
                self.dynamics = (
                    dynamics
                    if dynamics is not None
                    else self._load_dynamics(
                        resolved_dynamics_checkpoint,
                        strict=self.strict_checkpoint_loading,
                    )
                ) if self.planner_enabled else None
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

        policy = VisualActorCritic()
        if checkpoint is None:
            if strict:
                raise RuntimeError("failed to load required policy checkpoint: None")
            return policy, False
        try:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = saved.get("model_state") if isinstance(saved, dict) else None
            if isinstance(state, dict):
                policy = VisualActorCritic()
                policy.load_state_dict(state, strict=True)
                return policy, True
            if strict:
                raise ValueError("policy checkpoint lacks model_state")
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
            if strict:
                raise RuntimeError(
                    f"failed to load required policy checkpoint: {checkpoint}"
                ) from error
        return policy, False

    @staticmethod
    def _load_policy(checkpoint: str | None, *, strict: bool = False):
        policy, _loaded = Agent._load_policy_with_status(checkpoint, strict=strict)
        return policy

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
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
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
        pixels = np.asarray(observation, dtype=np.float32)
        if pixels.shape != (4, 84, 84) or not np.all(np.isfinite(pixels)):
            return None
        if float(pixels.min()) < 0.0 or float(pixels.max()) > 1.0:
            return None
        return torch.from_numpy(pixels).unsqueeze(0)

    def reset(self, observation):
        """Clear the state owned by the selected runtime."""
        if self._runtime_mode == "baseline":
            self.smoother.reset(initial_action=self.action_smoothing["initial_action"])
            return
        reset = getattr(self.planner, "reset", None)
        if callable(reset):
            reset()
        if self.corridor_controller is not None:
            self.corridor_controller.reset(observation)
        del observation

    @torch.inference_mode()
    def act(self, observation) -> np.ndarray:
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
        if self.controller_mode == "corridor":
            action = self._safe_action(self.corridor_controller.act(observation))
            return np.zeros(3, dtype=np.float32) if action is None else action
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
