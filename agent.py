import time

import numpy as np
import torch
from torch import nn

from haic_agent.dynamics import LatentDynamicsEnsemble
from haic_agent.networks import VisualActorCritic
from haic_agent.planner import CEMPlanner
from haic_agent.runtime_config import (
    PLANNER_ENABLED,
    PLANNER_SETTINGS,
    STRICT_CHECKPOINT_LOADING,
)


POLICY_MODEL_FILENAME = "policy.pt"
DYNAMICS_MODEL_FILENAME = "dynamics.pt"
# Task 5's archive validator reads this conventional primary model filename.
MODEL_FILENAME = "policy.pt"

# Small single-observation CNN inference is faster and more predictable without
# the default large CPU thread pool.
torch.set_num_threads(1)


class NatureFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
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

    def __init__(self):
        super().__init__()
        self.features_extractor = NatureFeatures()
        self.policy_net = nn.Sequential()
        self.action_net = nn.Linear(512, 3)

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
    """Pixel-only driving using a trained visual policy and optional learned planner."""

    def __init__(
        self,
        *,
        policy=None,
        dynamics=None,
        planner=None,
        planner_enabled: bool | None = None,
        strict_checkpoint_loading: bool | None = None,
        plan_budget: float = 4.5,
        clock=time.monotonic,
        policy_checkpoint: str | None = None,
        dynamics_checkpoint: str | None = None,
    ):
        """Build the visual policy and optional planning models."""
        torch.set_num_threads(1)
        self.clock = clock
        self.strict_checkpoint_loading = (
            STRICT_CHECKPOINT_LOADING if strict_checkpoint_loading is None else bool(strict_checkpoint_loading)
        )
        self.plan_budget = min(max(float(plan_budget), 0.0), 4.5)
        resolved_policy_checkpoint = POLICY_MODEL_FILENAME if policy_checkpoint is None else policy_checkpoint
        configured_planner_enabled = PLANNER_ENABLED if planner_enabled is None else bool(planner_enabled)
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
        if self.planner_enabled:
            self.dynamics = (
                dynamics
                if dynamics is not None
                else self._load_dynamics(
                    resolved_dynamics_checkpoint,
                    strict=self.strict_checkpoint_loading,
                )
            )
        self.planner = planner if planner is not None else CEMPlanner(**PLANNER_SETTINGS, clock=clock)
        if self.policy is not None and hasattr(self.policy, "eval"):
            self.policy.eval()
        if self.dynamics is not None and hasattr(self.dynamics, "eval"):
            self.dynamics.eval()

    @staticmethod
    def _load_policy_with_status(checkpoint: str | None, *, strict: bool = False):
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
            elif strict:
                raise ValueError("policy checkpoint lacks model_state")
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
            if strict:
                raise RuntimeError(f"failed to load required policy checkpoint: {checkpoint}") from error
        return None, False

    @staticmethod
    def _load_policy(checkpoint: str | None, *, strict: bool = False):
        policy, _loaded = Agent._load_policy_with_status(checkpoint, strict=strict)
        return policy if policy is not None else VisualActorCritic()

    @staticmethod
    def _load_dynamics(checkpoint: str | None, *, strict: bool = False):
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
                raise RuntimeError(f"failed to load required dynamics checkpoint: {checkpoint}") from error
        return None

    @staticmethod
    def _safe_action(action) -> np.ndarray | None:
        try:
            candidate = np.asarray(action, dtype=np.float32).reshape(3)
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(candidate)):
            return None
        return np.clip(candidate, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]).astype(np.float32, copy=False)

    @staticmethod
    def _observation_tensor(observation):
        pixels = np.asarray(observation, dtype=np.float32)
        if pixels.shape != (4, 84, 84) or not np.all(np.isfinite(pixels)):
            return None
        if float(pixels.min()) < 0.0 or float(pixels.max()) > 1.0:
            return None
        return torch.from_numpy(pixels).unsqueeze(0)

    def reset(self, observation):
        """Clear all per-episode plan state; reset needs no simulator data."""
        reset = getattr(self.planner, "reset", None)
        if callable(reset):
            reset()
        del observation

    @torch.inference_mode()
    def act(self, observation) -> np.ndarray:
        """Return a finite action before the end-to-end 4.5 second deadline."""
        deadline = self.clock() + self.plan_budget
        no_op = np.zeros(3, dtype=np.float32)
        state = self._observation_tensor(observation)
        if state is None:
            return no_op
        try:
            latent = self.policy.encode_observation(state)
            policy_mean, policy_log_std = self.policy.action_parameters(latent)
            fallback = VisualActorCritic._bound_actions(policy_mean)
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
