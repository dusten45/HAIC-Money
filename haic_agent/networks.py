"""Pixel-only visual actor-critic network used by PPO and later planners."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F

from haic_agent.observation import HUD_ROIS


LATENT_SIZE = 128
POLICY_ACTION_SIZE = 2
ACTION_SIZE = 3
AUXILIARY_SIZE = 10
MAX_GAS = 0.02
MAX_BRAKE = 0.03
INITIAL_STEERING_COMMAND = -0.12
INITIAL_LONGITUDINAL_COMMAND = 0.55
_ACTION_EPSILON = 1e-6
MIN_LOG_STD = -5.0
MAX_LOG_STD = 2.0


@dataclass(frozen=True)
class PolicyOutput:
    """Policy prediction for a pixel batch.

    ``action_mean`` and ``action_log_std`` parameterize a two-dimensional
    Normal prior over steering and signed longitudinal control. ``value`` is
    shaped ``(B,)`` and the ten auxiliary targets are speed, four wheel
    angular velocities, steering angle, yaw rate, normalized lateral offset,
    and sine/cosine of heading error in that order.
    """

    latent: Tensor
    action_mean: Tensor
    action_log_std: Tensor
    value: Tensor
    auxiliary_predictions: Tensor


class VisualActorCritic(nn.Module):
    """A compact full-frame CNN fused with a fixed-position HUD CNN."""

    def __init__(self, *, use_hud: bool = True) -> None:
        super().__init__()
        self.use_hud = bool(use_hud)
        self.full_frame_encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
        )
        self.hud_encoder = nn.Sequential(
            nn.Conv2d(len(HUD_ROIS), 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 64),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(128, LATENT_SIZE), nn.ReLU())
        self.policy_mean = nn.Linear(LATENT_SIZE, POLICY_ACTION_SIZE)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.zeros_(self.policy_mean.bias)
        with torch.no_grad():
            self.policy_mean.bias[0] = torch.atanh(
                torch.tensor(INITIAL_STEERING_COMMAND)
            )
            self.policy_mean.bias[1] = torch.atanh(
                torch.tensor(INITIAL_LONGITUDINAL_COMMAND)
            )
        self.policy_log_std = nn.Parameter(
            torch.log(torch.tensor((0.35, 0.4), dtype=torch.float32))
        )
        self.value_head = nn.Linear(LATENT_SIZE, 1)
        self.auxiliary_head = nn.Linear(LATENT_SIZE, AUXILIARY_SIZE)

    @staticmethod
    def _validate_observation(observation_tensor: Tensor) -> None:
        if observation_tensor.ndim != 4 or observation_tensor.shape[1:] != (4, 84, 84):
            raise ValueError("observation_tensor must have shape (B, 4, 84, 84)")

    def _hud_pixels(self, observation_tensor: Tensor) -> Tensor:
        """Return each named fixed HUD ROI resized into one visual channel."""
        last_frame = observation_tensor[:, -1:, :, :]
        crops = []
        for top, bottom, left, right in HUD_ROIS.values():
            crop = last_frame[:, :, top:bottom, left:right]
            crops.append(F.interpolate(crop, size=(8, 8), mode="bilinear", align_corners=False))
        return torch.cat(crops, dim=1)

    def encode_observation(self, observation_tensor: Tensor) -> Tensor:
        """Encode a batch of four float pixel frames as stable ``(B, 128)`` latents."""
        self._validate_observation(observation_tensor)
        pixels = observation_tensor.float()
        full_features = self.full_frame_encoder(pixels)
        if self.use_hud:
            hud_features = self.hud_encoder(self._hud_pixels(pixels))
        else:
            hud_features = torch.zeros_like(full_features)
        return self.fusion(torch.cat((full_features, hud_features), dim=1))

    def action_parameters(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """Return unconstrained Normal mean and log standard deviation for actions."""
        if latent.ndim != 2 or latent.shape[1] != LATENT_SIZE:
            raise ValueError("latent must have shape (B, 128)")
        action_mean = self.policy_mean(latent)
        action_log_std = self.policy_log_std.clamp(MIN_LOG_STD, MAX_LOG_STD)
        return action_mean, action_log_std.expand_as(action_mean)

    def forward(self, observation_tensor: Tensor) -> PolicyOutput:
        latent = self.encode_observation(observation_tensor)
        action_mean, action_log_std = self.action_parameters(latent)
        return PolicyOutput(
            latent=latent,
            action_mean=action_mean,
            action_log_std=action_log_std,
            value=self.value_head(latent).squeeze(-1),
            auxiliary_predictions=self.auxiliary_head(latent),
        )

    @staticmethod
    def _bound_actions(unconstrained_actions: Tensor) -> Tensor:
        if unconstrained_actions.ndim < 2 or unconstrained_actions.shape[-1] != POLICY_ACTION_SIZE:
            raise ValueError("policy action coordinates must end in width 2")
        steering = torch.tanh(unconstrained_actions[..., :1])
        longitudinal = torch.tanh(unconstrained_actions[..., 1:2])
        gas = MAX_GAS * longitudinal.clamp(min=0.0)
        brake = MAX_BRAKE * (-longitudinal).clamp(min=0.0)
        return torch.cat((steering, gas, brake), dim=-1)

    @staticmethod
    def _unbound_actions(actions: Tensor) -> tuple[Tensor, Tensor]:
        if actions.ndim != 2 or actions.shape[1] != ACTION_SIZE:
            raise ValueError("actions must have shape (B, 3)")
        if not torch.isfinite(actions).all():
            raise ValueError("actions must be finite")
        gas_values = actions[:, 1:2]
        brake_values = actions[:, 2:3]
        if torch.any(gas_values > MAX_GAS + _ACTION_EPSILON):
            raise ValueError(f"gas must not exceed the learned throttle limit {MAX_GAS}")
        if torch.any(brake_values > MAX_BRAKE + _ACTION_EPSILON):
            raise ValueError(f"brake must not exceed the calibrated limit {MAX_BRAKE}")
        if torch.any((gas_values > _ACTION_EPSILON) & (brake_values > _ACTION_EPSILON)):
            raise ValueError("gas and brake cannot both be active")
        steer = actions[:, :1].clamp(-1.0 + _ACTION_EPSILON, 1.0 - _ACTION_EPSILON)
        gas = gas_values.clamp(min=0.0, max=MAX_GAS)
        brake = brake_values.clamp(min=0.0, max=MAX_BRAKE)
        longitudinal = torch.where(
            gas > 0.0, gas / MAX_GAS, -brake / MAX_BRAKE
        )
        longitudinal = longitudinal.clamp(-1.0 + _ACTION_EPSILON, 1.0 - _ACTION_EPSILON)
        unconstrained = torch.cat((torch.atanh(steer), torch.atanh(longitudinal)), dim=1)
        return unconstrained, VisualActorCritic._log_abs_det_jacobian(unconstrained)

    @staticmethod
    def _log_abs_det_jacobian(unconstrained_actions: Tensor) -> Tensor:
        """Compute log-Jacobians for steering and signed pedal coordinates."""
        steer = unconstrained_actions[:, :1]
        longitudinal = unconstrained_actions[:, 1:2]
        steer_log_det = 2.0 * (
            torch.log(torch.tensor(2.0, device=steer.device, dtype=steer.dtype))
            - steer
            - F.softplus(-2.0 * steer)
        )
        longitudinal_log_det = 2.0 * (
            torch.log(torch.tensor(2.0, device=longitudinal.device, dtype=longitudinal.dtype))
            - longitudinal
            - F.softplus(-2.0 * longitudinal)
        )
        pedal_scale_log_det = torch.where(
            longitudinal > 0.0,
            torch.full_like(longitudinal, math.log(MAX_GAS)),
            torch.where(
                longitudinal < 0.0,
                torch.full_like(longitudinal, math.log(MAX_BRAKE)),
                torch.zeros_like(longitudinal),
            ),
        )
        return (steer_log_det + longitudinal_log_det + pedal_scale_log_det).sum(dim=1)

    def sample_actions(self, output: PolicyOutput) -> tuple[Tensor, Tensor]:
        """Sample bounded simulator actions and their matching transformed log probability."""
        actions, log_probability, _ = self.sample_actions_with_pretransform(output)
        return actions, log_probability

    def sample_actions_with_pretransform(self, output: PolicyOutput) -> tuple[Tensor, Tensor, Tensor]:
        """Sample actions while retaining exact Normal draws for PPO rollout storage."""
        distribution = Normal(output.action_mean, output.action_log_std.exp())
        unconstrained = distribution.rsample()
        actions = self._bound_actions(unconstrained)
        log_probability = self.log_probability_from_pretransform(
            unconstrained, output.action_mean, output.action_log_std
        )
        return actions, log_probability, unconstrained

    def deterministic_actions(self, output: PolicyOutput) -> Tensor:
        """Transform policy means to bounded evaluation actions."""
        return self._bound_actions(output.action_mean)

    def log_probability(self, actions: Tensor, action_mean: Tensor, action_log_std: Tensor) -> Tensor:
        """Evaluate bounded environment actions under the transformed Normal policy."""
        unconstrained, log_abs_det_jacobian = self._unbound_actions(actions)
        distribution = Normal(action_mean, action_log_std.exp())
        return distribution.log_prob(unconstrained).sum(dim=1) - log_abs_det_jacobian

    def log_probability_from_pretransform(
        self, pretransform_actions: Tensor, action_mean: Tensor, action_log_std: Tensor
    ) -> Tensor:
        """Evaluate exact rollout Normal draws without lossy action inverse transforms."""
        distribution = Normal(action_mean, action_log_std.exp())
        return distribution.log_prob(pretransform_actions).sum(dim=-1) - self._log_abs_det_jacobian(
            pretransform_actions
        )

    @staticmethod
    def entropy(action_log_std: Tensor) -> Tensor:
        """Entropy of the policy's unconstrained Normal prior for PPO regularization."""
        return Normal(torch.zeros_like(action_log_std), action_log_std.exp()).entropy().sum(dim=1)
