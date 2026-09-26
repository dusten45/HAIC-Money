"""Torch pixel-SAC networks for the isolated HAIC RLPD research lane."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


OBSERVATION_SHAPE = (4, 84, 84)
ACTION_DIM = 3
NUM_QS = 10
LATENT_DIM = 50


class PixelEncoder(nn.Module):
    """D4PG-style pixel encoder used by the pinned RLPD pixel configuration."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        if type(latent_dim) is not int or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer")
        self.latent_dim = latent_dim
        layers = []
        channels = (4, 32, 64, 128, 256)
        for in_channels, out_channels in zip(channels, channels[1:]):
            layers.extend((nn.Conv2d(in_channels, out_channels, 3, stride=2), nn.ReLU()))
        self.convolution = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

    @staticmethod
    def normalize(observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 4 or tuple(observation.shape[1:]) != OBSERVATION_SHAPE:
            raise ValueError("pixel observations must have shape (batch, 4, 84, 84)")
        if observation.dtype == torch.uint8:
            observation = observation.float().div(255.0)
        elif not observation.is_floating_point():
            raise TypeError("pixel observations must be uint8 or floating point")
        return observation

    def convolutional_features(self, observation: torch.Tensor) -> torch.Tensor:
        return self.convolution(self.normalize(observation))

    def project(self, convolutional_features: torch.Tensor) -> torch.Tensor:
        return self.projection(convolutional_features)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.project(self.convolutional_features(observation))


def squashed_normal_log_prob(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    pre_tanh: torch.Tensor,
) -> torch.Tensor:
    """Stable transformed-Normal log density from the original sample ``u``."""
    log_std = log_std.clamp(-20.0, 2.0)
    standardized = (pre_tanh - mean) / log_std.exp()
    normal_log_prob = -0.5 * (standardized.square() + 2.0 * log_std + math.log(2.0 * math.pi))
    log_tanh_jacobian = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
    return (normal_log_prob - log_tanh_jacobian).sum(dim=-1, keepdim=True)


def sample_squashed_normal(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return native action, its exact log density, and retained pre-tanh sample."""
    if mean.shape != log_std.shape or mean.shape[-1] != ACTION_DIM:
        raise ValueError("mean and log_std must have matching native 3-action shapes")
    log_std = log_std.clamp(-20.0, 2.0)
    noise = torch.randn(
        mean.shape,
        dtype=mean.dtype,
        device=mean.device,
        generator=generator,
    )
    pre_tanh = mean + log_std.exp() * noise
    action = torch.tanh(pre_tanh)
    return action, squashed_normal_log_prob(mean, log_std, pre_tanh), pre_tanh


class PixelActor(nn.Module):
    """Squashed diagonal-Gaussian actor; its encoder is synchronized by learner."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = PixelEncoder(latent_dim)
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, ACTION_DIM)
        self.log_std = nn.Linear(256, ACTION_DIM)

    def distribution(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The pinned pixel multiplexer stops gradients after its CNN encoder;
        # latent projection, LayerNorm/Tanh, actor trunk, and policy heads train.
        with torch.no_grad():
            convolutional_features = self.encoder.convolutional_features(observation)
        features = self.encoder.project(convolutional_features)
        hidden = self.trunk(features)
        return self.mean(hidden), self.log_std(hidden).clamp(-20.0, 2.0)

    def sample(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        mean, log_std = self.distribution(observation)
        if deterministic:
            return torch.tanh(mean), None, None
        action, log_prob, pre_tanh = sample_squashed_normal(
            mean, log_std, generator=generator
        )
        return action, log_prob, pre_tanh


class PixelCritic(nn.Module):
    """One image trunk and ten independently initialized, normalized Q heads."""

    def __init__(self, latent_dim: int = LATENT_DIM, num_qs: int = NUM_QS):
        super().__init__()
        if type(num_qs) is not int or num_qs <= 0:
            raise ValueError("num_qs must be a positive integer")
        self.encoder = PixelEncoder(latent_dim)
        self.num_qs = num_qs
        self.q_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(latent_dim + ACTION_DIM, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
            for _ in range(num_qs)
        )

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action.ndim != 2 or action.shape[-1] != ACTION_DIM:
            raise ValueError("native actions must have shape (batch, 3)")
        if observation.shape[0] != action.shape[0]:
            raise ValueError("observation and action batch sizes must match")
        latent = self.encoder(observation) if features is None else features
        if latent.shape != (action.shape[0], self.encoder.latent_dim):
            raise ValueError("critic features have the wrong batch or latent shape")
        value_input = torch.cat((latent, action), dim=-1)
        return torch.cat([head(value_input) for head in self.q_heads], dim=-1)


def select_target_q_values(
    q_values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Take the minimum of the selected Q heads per sample."""
    if q_values.ndim != 2:
        raise ValueError("q_values must have shape (batch, num_qs)")
    if indices.ndim == 1:
        if indices.numel() == 0:
            raise ValueError("at least one target Q index is required")
        selected = q_values.index_select(1, indices.long())
    elif indices.ndim == 2:
        if indices.shape[0] != q_values.shape[0] or indices.shape[1] == 0:
            raise ValueError("per-sample Q indices must have shape (batch, num_min_qs)")
        selected = q_values.gather(1, indices.long())
    else:
        raise ValueError("indices must be a 1D or 2D tensor")
    return selected.min(dim=1, keepdim=True).values
