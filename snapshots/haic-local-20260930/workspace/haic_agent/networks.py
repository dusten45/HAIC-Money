"""Pixel-only visual actor-critic network used by PPO and later planners."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F

from haic_agent.observation import HUD_ROIS
from haic_agent.pixel_features import (
    TEMPORAL_FEATURE_SIZE,
    VISUAL_FEATURE_SIZE,
    extract_temporal_features_batch,
    extract_visual_features_batch,
)


LATENT_SIZE = 128
CPU_INFERENCE_THREADS = 1
POLICY_ACTION_SIZE = 2
ACTION_SIZE = 3
AUXILIARY_SIZE = 10
# Match the calibrated teacher's usable actuator range. Earlier PPO caps of
# 0.02/0.03 made the actor much slower and weaker at braking than the simulator
# and teacher: on the site map, gas 0.02 reached 10.5 speed in 20 decisions,
# while gas 0.12 reached 29.0; brake 0.03 barely slowed a 36.3-speed car.
MAX_GAS = 0.12
MAX_BRAKE = 0.28
# Pedal expansion widens the usable pedal range without moving the cruise
# operating point. The longitudinal coordinate becomes ``tanh(u / expansion)``
# while both pedal scales are multiplied by ``expansion``: near ``u = 0`` the
# two factors cancel, so the cruise throttle a checkpoint already produces is
# preserved, and only the saturated end of the range grows. Rollout evidence
# motivated this: the selected actor's sampled ``gas_max`` was 0.1153 against a
# 0.12 ceiling (96% saturation) while the offline reference teacher laps the
# same map 2.3s faster using gas up to 0.24, so lap time was limited by the
# action transform rather than by the network. Expansion 1.0 reproduces every
# existing checkpoint exactly.
DEFAULT_PEDAL_EXPANSION = 1.0
# The throttle and brake branches saturate at different base scales, so a single
# symmetric factor cannot open both to the simulator's full [0, 1] pedal range.
# Expanding each branch separately can: gas reaches 1.0 at 1/MAX_GAS and brake
# at 1/MAX_BRAKE. Track measurements force this. A 15s lap needs a 64-80 mean
# speed (track length 962-1197 over 15s) against a car that tops out at 100,
# while gas 0.12 is still at 42.5 after 40 decisions and peaks at 59-71 on a
# straight. Gas 0.42 reaches 80.0 after 40 decisions, so the target is only
# inside the envelope once the throttle branch opens well past 0.12.
MAX_THROTTLE_EXPANSION = 1.0 / MAX_GAS
MAX_BRAKE_EXPANSION = 1.0 / MAX_BRAKE
# A symmetric ``pedal_expansion`` sets both branches, so it is bounded by the
# tighter of the two.
MAX_PEDAL_EXPANSION = MAX_BRAKE_EXPANSION
INITIAL_STEERING_COMMAND = -0.12
INITIAL_LONGITUDINAL_COMMAND = 0.55
_ACTION_EPSILON = 1e-6
MIN_LOG_STD = -5.0
MAX_LOG_STD = 2.0


@dataclass(frozen=True)
class PedalScale:
    """How far each pedal branch is opened past its calibrated base scale.

    The longitudinal coordinate is squashed by the branch's own expansion, and
    the branch scale is multiplied by it: ``MAX * k * tanh(u / k)``. Near
    ``u = 0`` the two factors cancel, so the cruise command a trained
    checkpoint already produces is preserved and only the saturated end of the
    range grows. Expansion 1.0 on both branches is the original transform.
    """

    throttle_expansion: float = DEFAULT_PEDAL_EXPANSION
    brake_expansion: float = DEFAULT_PEDAL_EXPANSION

    def __post_init__(self) -> None:
        for name, value, ceiling in (
            ("throttle_expansion", self.throttle_expansion, MAX_THROTTLE_EXPANSION),
            ("brake_expansion", self.brake_expansion, MAX_BRAKE_EXPANSION),
        ):
            if not math.isfinite(float(value)) or not 1.0 <= float(value) <= ceiling + 1e-9:
                raise ValueError(
                    f"{name} must be finite and between 1.0 and {ceiling}"
                )

    @classmethod
    def resolve(cls, value: "PedalScale | float | int | None") -> "PedalScale":
        """Accept a PedalScale, a symmetric float, or None for the default."""
        if value is None:
            return cls()
        if isinstance(value, PedalScale):
            return value
        expansion = float(value)
        return cls(expansion, expansion)

    @property
    def throttle_limit(self) -> float:
        return MAX_GAS * self.throttle_expansion

    @property
    def brake_limit(self) -> float:
        return MAX_BRAKE * self.brake_expansion

    @property
    def is_symmetric(self) -> bool:
        return self.throttle_expansion == self.brake_expansion


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

    def __init__(
        self,
        *,
        use_hud: bool = True,
        use_visual_features: bool = False,
        use_temporal_features: bool = False,
        pedal_expansion: float = DEFAULT_PEDAL_EXPANSION,
        throttle_expansion: float | None = None,
        brake_expansion: float | None = None,
    ) -> None:
        super().__init__()
        self.pedal_scale = PedalScale(
            float(pedal_expansion if throttle_expansion is None else throttle_expansion),
            float(pedal_expansion if brake_expansion is None else brake_expansion),
        )
        # ``pedal_expansion`` stays meaningful only while both branches match.
        self.pedal_expansion = (
            self.pedal_scale.throttle_expansion
            if self.pedal_scale.is_symmetric
            else float("nan")
        )
        self.throttle_limit = self.pedal_scale.throttle_limit
        self.brake_limit = self.pedal_scale.brake_limit
        self.use_hud = bool(use_hud)
        self.use_visual_features = bool(use_visual_features)
        self.use_temporal_features = bool(use_temporal_features)
        # The collector uses the same pixel-derived features seen by the actor
        # to shape train-only obstacle feedback. Keep them out of state_dict.
        self.last_visual_features: Tensor | None = None
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
        self.visual_feature_encoder = (
            nn.Sequential(nn.Linear(VISUAL_FEATURE_SIZE, 32), nn.ReLU())
            if self.use_visual_features
            else None
        )
        self.temporal_feature_encoder = (
            nn.Linear(TEMPORAL_FEATURE_SIZE, LATENT_SIZE, bias=True)
            if self.use_visual_features and self.use_temporal_features
            else None
        )
        if self.temporal_feature_encoder is not None:
            nn.init.zeros_(self.temporal_feature_encoder.weight)
            nn.init.zeros_(self.temporal_feature_encoder.bias)
        fusion_input_size = 128 + (32 if self.use_visual_features else 0)
        self.fusion = nn.Sequential(nn.Linear(fusion_input_size, LATENT_SIZE), nn.ReLU())
        self.policy_mean = nn.Linear(LATENT_SIZE, POLICY_ACTION_SIZE)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.zeros_(self.policy_mean.bias)
        with torch.no_grad():
            self.policy_mean.bias[0] = torch.atanh(
                torch.tensor(INITIAL_STEERING_COMMAND)
            )
            # Solve ``k * tanh(bias / k) == INITIAL_LONGITUDINAL`` on the
            # throttle branch so a fresh actor starts at the same absolute gas
            # whatever the expansion.
            throttle_expansion = self.pedal_scale.throttle_expansion
            self.policy_mean.bias[1] = throttle_expansion * torch.atanh(
                torch.tensor(INITIAL_LONGITUDINAL_COMMAND / throttle_expansion)
            )
        self.policy_log_std = nn.Parameter(
            torch.log(torch.tensor((0.35, 0.4), dtype=torch.float32))
        )
        self.value_head = nn.Linear(LATENT_SIZE, 1)
        self.auxiliary_head = nn.Linear(LATENT_SIZE, AUXILIARY_SIZE)
        self.last_temporal_features: Tensor | None = None

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
        fused_features = [full_features, hud_features]
        if self.use_visual_features:
            feature_values = extract_visual_features_batch(pixels.detach().cpu().numpy())
            visual_values = torch.as_tensor(
                feature_values, device=pixels.device, dtype=pixels.dtype
            )
            self.last_visual_features = visual_values.detach()
            if self.visual_feature_encoder is None:
                raise RuntimeError("visual feature encoder was not initialized")
            fused_features.append(self.visual_feature_encoder(visual_values))
        else:
            self.last_visual_features = None
        latent = self.fusion(torch.cat(fused_features, dim=1))
        if self.temporal_feature_encoder is not None:
            temporal_values = extract_temporal_features_batch(
                pixels.detach().cpu().numpy()
            )
            temporal_tensor = torch.as_tensor(
                temporal_values, device=pixels.device, dtype=pixels.dtype
            )
            self.last_temporal_features = temporal_tensor.detach()
            latent = latent + self.temporal_feature_encoder(temporal_tensor)
        else:
            self.last_temporal_features = None
        return latent

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
    def pedal_limits(
        pedal_expansion: "PedalScale | float" = DEFAULT_PEDAL_EXPANSION,
    ) -> tuple[float, float]:
        """Return the (throttle, brake) ceilings implied by a pedal scale."""
        scale = PedalScale.resolve(pedal_expansion)
        return scale.throttle_limit, scale.brake_limit

    @staticmethod
    def _bound_actions(
        unconstrained_actions: Tensor,
        pedal_expansion: "PedalScale | float" = DEFAULT_PEDAL_EXPANSION,
    ) -> Tensor:
        if unconstrained_actions.ndim < 2 or unconstrained_actions.shape[-1] != POLICY_ACTION_SIZE:
            raise ValueError("policy action coordinates must end in width 2")
        scale = PedalScale.resolve(pedal_expansion)
        longitudinal_coordinate = unconstrained_actions[..., 1:2]
        steering = torch.tanh(unconstrained_actions[..., :1])
        # Each branch squashes with its own expansion; both map 0 to 0, so the
        # piecewise map stays continuous and strictly monotone in ``u``.
        gas = scale.throttle_limit * torch.tanh(
            longitudinal_coordinate / scale.throttle_expansion
        ).clamp(min=0.0)
        brake = scale.brake_limit * (
            -torch.tanh(longitudinal_coordinate / scale.brake_expansion)
        ).clamp(min=0.0)
        return torch.cat((steering, gas, brake), dim=-1)

    @staticmethod
    def _unbound_actions(
        actions: Tensor,
        pedal_expansion: "PedalScale | float" = DEFAULT_PEDAL_EXPANSION,
    ) -> tuple[Tensor, Tensor]:
        if actions.ndim != 2 or actions.shape[1] != ACTION_SIZE:
            raise ValueError("actions must have shape (B, 3)")
        if not torch.isfinite(actions).all():
            raise ValueError("actions must be finite")
        scale = PedalScale.resolve(pedal_expansion)
        throttle_limit, brake_limit = scale.throttle_limit, scale.brake_limit
        gas_values = actions[:, 1:2]
        brake_values = actions[:, 2:3]
        if torch.any(gas_values > throttle_limit + _ACTION_EPSILON):
            raise ValueError(
                f"gas must not exceed the learned throttle limit {throttle_limit}"
            )
        if torch.any(brake_values > brake_limit + _ACTION_EPSILON):
            raise ValueError(f"brake must not exceed the calibrated limit {brake_limit}")
        if torch.any((gas_values > _ACTION_EPSILON) & (brake_values > _ACTION_EPSILON)):
            raise ValueError("gas and brake cannot both be active")
        steer = actions[:, :1].clamp(-1.0 + _ACTION_EPSILON, 1.0 - _ACTION_EPSILON)
        gas = gas_values.clamp(min=0.0, max=throttle_limit)
        brake = brake_values.clamp(min=0.0, max=brake_limit)
        accelerating = gas > 0.0
        branch_expansion = torch.where(
            accelerating,
            torch.full_like(gas, scale.throttle_expansion),
            torch.full_like(gas, scale.brake_expansion),
        )
        longitudinal = torch.where(
            accelerating, gas / throttle_limit, -brake / brake_limit
        )
        longitudinal = longitudinal.clamp(-1.0 + _ACTION_EPSILON, 1.0 - _ACTION_EPSILON)
        unconstrained = torch.cat(
            (torch.atanh(steer), branch_expansion * torch.atanh(longitudinal)), dim=1
        )
        return unconstrained, VisualActorCritic._log_abs_det_jacobian(
            unconstrained, scale
        )

    @staticmethod
    def _log_abs_det_jacobian(
        unconstrained_actions: Tensor,
        pedal_expansion: "PedalScale | float" = DEFAULT_PEDAL_EXPANSION,
    ) -> Tensor:
        """Compute log-Jacobians for steering and signed pedal coordinates.

        With branch expansion ``k`` the pedal map is ``MAX * k * tanh(u / k)``
        whose derivative is ``MAX * sech^2(u / k)``: the expansion cancels and
        only the point where the squashing term is evaluated changes. Each
        branch uses its own ``k``, chosen by the sign of ``u``.
        """
        scale = PedalScale.resolve(pedal_expansion)
        raw_longitudinal = unconstrained_actions[:, 1:2]
        branch_expansion = torch.where(
            raw_longitudinal > 0.0,
            torch.full_like(raw_longitudinal, scale.throttle_expansion),
            torch.full_like(raw_longitudinal, scale.brake_expansion),
        )
        steer = unconstrained_actions[:, :1]
        longitudinal = raw_longitudinal / branch_expansion
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

    def bound_actions(self, unconstrained_actions: Tensor) -> Tensor:
        """Apply this actor's own pedal expansion to unconstrained coordinates."""
        return self._bound_actions(unconstrained_actions, self.pedal_scale)

    def unbound_actions(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Invert this actor's own bounded simulator actions."""
        return self._unbound_actions(actions, self.pedal_scale)

    def sample_actions(self, output: PolicyOutput) -> tuple[Tensor, Tensor]:
        """Sample bounded simulator actions and their matching transformed log probability."""
        actions, log_probability, _ = self.sample_actions_with_pretransform(output)
        return actions, log_probability

    def sample_actions_with_pretransform(self, output: PolicyOutput) -> tuple[Tensor, Tensor, Tensor]:
        """Sample actions while retaining exact Normal draws for PPO rollout storage."""
        distribution = Normal(output.action_mean, output.action_log_std.exp())
        unconstrained = distribution.rsample()
        actions = self._bound_actions(unconstrained, self.pedal_scale)
        log_probability = self.log_probability_from_pretransform(
            unconstrained, output.action_mean, output.action_log_std
        )
        return actions, log_probability, unconstrained

    def deterministic_actions(self, output: PolicyOutput) -> Tensor:
        """Transform policy means to bounded evaluation actions."""
        return self._bound_actions(output.action_mean, self.pedal_scale)

    def log_probability(self, actions: Tensor, action_mean: Tensor, action_log_std: Tensor) -> Tensor:
        """Evaluate bounded environment actions under the transformed Normal policy."""
        unconstrained, log_abs_det_jacobian = self._unbound_actions(
            actions, self.pedal_scale
        )
        distribution = Normal(action_mean, action_log_std.exp())
        return distribution.log_prob(unconstrained).sum(dim=1) - log_abs_det_jacobian

    def log_probability_from_pretransform(
        self, pretransform_actions: Tensor, action_mean: Tensor, action_log_std: Tensor
    ) -> Tensor:
        """Evaluate exact rollout Normal draws without lossy action inverse transforms."""
        distribution = Normal(action_mean, action_log_std.exp())
        return distribution.log_prob(pretransform_actions).sum(dim=-1) - self._log_abs_det_jacobian(
            pretransform_actions, self.pedal_scale
        )

    @staticmethod
    def entropy(action_log_std: Tensor) -> Tensor:
        """Entropy of the policy's unconstrained Normal prior for PPO regularization."""
        return Normal(torch.zeros_like(action_log_std), action_log_std.exp()).entropy().sum(dim=1)
