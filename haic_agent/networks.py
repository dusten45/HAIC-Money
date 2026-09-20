"""Pixel-only visual actor-critic network used by PPO and later planners."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F

from haic_agent.observation import HUD_ROIS


LATENT_SIZE = 128
ACTION_SIZE = 3
AUXILIARY_SIZE = 7
_ACTION_EPSILON = 1e-6


@dataclass(frozen=True)
class PolicyOutput:
    """Policy prediction for a pixel batch.

    ``action_mean`` and ``action_log_std`` parameterize an unconstrained
    three-dimensional Normal prior.  ``value`` is shaped ``(B,)`` and the
    seven auxiliary targets are speed, four wheel angular velocities,
    steering angle, and yaw rate in that order.
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
        self.policy_mean = nn.Linear(LATENT_SIZE, ACTION_SIZE)
        self.policy_log_std = nn.Parameter(torch.zeros(ACTION_SIZE))
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
        return action_mean, self.policy_log_std.expand_as(action_mean)

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
        return torch.cat(
            (torch.tanh(unconstrained_actions[:, :1]), torch.sigmoid(unconstrained_actions[:, 1:])),
            dim=1,
        )

    @staticmethod
    def _unbound_actions(actions: Tensor) -> tuple[Tensor, Tensor]:
        if actions.ndim != 2 or actions.shape[1] != ACTION_SIZE:
            raise ValueError("actions must have shape (B, 3)")
        steer = actions[:, :1].clamp(-1.0 + _ACTION_EPSILON, 1.0 - _ACTION_EPSILON)
        pedals = actions[:, 1:].clamp(_ACTION_EPSILON, 1.0 - _ACTION_EPSILON)
        unconstrained = torch.cat((torch.atanh(steer), torch.logit(pedals)), dim=1)
        log_abs_det_jacobian = torch.cat(
            (torch.log1p(-steer.square()), torch.log(pedals * (1.0 - pedals))), dim=1
        ).sum(dim=1)
        return unconstrained, log_abs_det_jacobian

    def sample_actions(self, output: PolicyOutput) -> tuple[Tensor, Tensor]:
        """Sample bounded simulator actions and their matching transformed log probability."""
        distribution = Normal(output.action_mean, output.action_log_std.exp())
        unconstrained = distribution.rsample()
        actions = self._bound_actions(unconstrained)
        log_probability = distribution.log_prob(unconstrained).sum(dim=1)
        log_probability -= torch.cat(
            (torch.log1p(-actions[:, :1].square()), torch.log(actions[:, 1:] * (1.0 - actions[:, 1:]))),
            dim=1,
        ).sum(dim=1)
        return actions, log_probability

    def deterministic_actions(self, output: PolicyOutput) -> Tensor:
        """Transform policy means to bounded evaluation actions."""
        return self._bound_actions(output.action_mean)

    def log_probability(self, actions: Tensor, action_mean: Tensor, action_log_std: Tensor) -> Tensor:
        """Evaluate bounded environment actions under the transformed Normal policy."""
        unconstrained, log_abs_det_jacobian = self._unbound_actions(actions)
        distribution = Normal(action_mean, action_log_std.exp())
        return distribution.log_prob(unconstrained).sum(dim=1) - log_abs_det_jacobian

    @staticmethod
    def entropy(action_log_std: Tensor) -> Tensor:
        """Entropy of the policy's unconstrained Normal prior for PPO regularization."""
        return Normal(torch.zeros_like(action_log_std), action_log_std.exp()).entropy().sum(dim=1)
