"""Native PyTorch DreamerV3 implementation for HAIC visual control.

Strictly adheres to:
1. Frozen HAIC observation spec (4x84x84 float32 CHW in [0, 1]).
2. Frozen HAIC continuous action spec (symmetric native [-1, 1]^3 -> [steer, gas, brake]).
3. Categorical RSSM with unimix=0.01 and straight-through gradient estimation.
4. Scale-free symlog/symexp transformations for rewards and value targets.
5. Latent imagination (horizon H=15) with lambda-returns and stopped-feature score-function actor updates.
6. Sequence replay buffer with explicit is_first episode reset masking and circular buffer boundary safety.
7. Standalone CPU exportable actor container ("haic-dreamerv3-actor-v1") compatible with Torch 2.1 CPU.
"""

from dataclasses import asdict, dataclass
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_adapter import ActionAdapter, ActionSpec, ObservationSpec, Transition

DREAMERV3_ACTOR_FORMAT = "haic-dreamerv3-actor-v1"


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Scale-free symmetric logarithm: sign(x) * ln(|x| + 1)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Scale-free symmetric exponential: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def balanced_kl_loss(
    prior_probs: torch.Tensor,
    posterior_probs: torch.Tensor,
    *,
    dyn_weight: float,
    rep_weight: float,
    free_nats: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1e-7
    prior = prior_probs.clamp(min=eps)
    posterior = posterior_probs.clamp(min=eps)
    dyn = (
        posterior.detach()
        * (torch.log(posterior.detach()) - torch.log(prior))
    ).sum(dim=-1).sum(dim=-1)
    rep = (
        posterior * (torch.log(posterior) - torch.log(prior.detach()))
    ).sum(dim=-1).sum(dim=-1)
    dyn = dyn.clamp(min=free_nats)
    rep = rep.clamp(min=free_nats)
    loss = (dyn_weight * dyn + rep_weight * rep).mean()
    return loss, dyn, rep


def continuation_weights(continues: torch.Tensor, gamma: float) -> torch.Tensor:
    if continues.ndim < 1 or continues.shape[0] < 1:
        raise ValueError("continues must contain at least one imagination step")
    discounts = gamma * continues.detach()
    first = torch.ones_like(discounts[:1])
    return torch.cat([first, torch.cumprod(discounts[:-1], dim=0)], dim=0)


def lambda_returns(
    rewards: torch.Tensor,
    continues: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    if rewards.shape != continues.shape or values.shape[0] != rewards.shape[0] + 1:
        raise ValueError("lambda-return inputs must have T rewards/continues and T+1 values")
    result = [None] * rewards.shape[0]
    last = values[-1]
    for t in reversed(range(rewards.shape[0])):
        discount = gamma * continues[t]
        target = rewards[t] + discount * (
            (1.0 - lambda_) * values[t + 1] + lambda_ * last
        )
        result[t] = target
        last = target
    return torch.stack(result)


def continue_prediction_loss(
    logits: torch.Tensor, targets: torch.Tensor, positive_weight: float
) -> torch.Tensor:
    if positive_weight <= 0.0:
        raise ValueError("continue positive_weight must be positive")
    weight = torch.as_tensor(positive_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)


class LayerNormGRUCell(nn.Module):
    """GRUCell with LayerNorm applied to input and hidden projections for scale-invariance."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.fc_x = nn.Linear(input_dim, 3 * hidden_dim, bias=False)
        self.fc_h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.ln_x = nn.LayerNorm(3 * hidden_dim)
        self.ln_h = nn.LayerNorm(3 * hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gx = self.ln_x(self.fc_x(x))
        gh = self.ln_h(self.fc_h(h))
        xr, xu, xh = torch.chunk(gx, 3, dim=-1)
        hr, hu, hh = torch.chunk(gh, 3, dim=-1)
        r = torch.sigmoid(xr + hr)
        u = torch.sigmoid(xu + hu)
        h_tilde = torch.tanh(xh + r * hh)
        return (1.0 - u) * h_tilde + u * h


class ConvEncoder(nn.Module):
    """Encodes (4, 84, 84) pixel observations into a compact embedding."""

    def __init__(self, in_channels: int = 4, embed_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 84 -> 42
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 42 -> 21
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 21 -> 10
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),  # 10 -> 5
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x in [0, 1]
        return self.net(x.float())


class ConvDecoder(nn.Module):
    """Predicts pixel channels from an RSSM state [h, z]."""

    def __init__(self, in_features: int = 512, out_channels: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_features, 128 * 7 * 7)
        self.net = nn.Sequential(
            nn.Upsample(size=(14, 14), mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(size=(28, 28), mode="nearest"),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(size=(42, 42), mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(size=(84, 84), mode="nearest"),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self.fc(state).view(-1, 128, 7, 7)
        return self.net(h)


class CategoricalRSSM(nn.Module):
    """Recurrent State Space Model with discrete unimix categoricals and straight-through gradients."""

    def __init__(
        self,
        action_dim: int = 3,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        num_categoricals: int = 16,
        num_classes: int = 16,
        unimix: float = 0.01,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.k = int(num_categoricals)
        self.c = int(num_classes)
        self.stoch_dim = self.k * self.c
        self.state_dim = self.hidden_dim + self.stoch_dim
        self.unimix = float(unimix)

        self.act_embed = nn.Linear(self.stoch_dim + self.action_dim, self.hidden_dim)
        self.cell = LayerNormGRUCell(self.hidden_dim, self.hidden_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.stoch_dim),
        )
        self.post_net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.embed_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.stoch_dim),
        )

    def get_dist(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reshaped = logits.reshape(-1, self.k, self.c)
        probs = F.softmax(reshaped, dim=-1)
        if self.unimix > 0:
            probs = (1.0 - self.unimix) * probs + (self.unimix / self.c)
        return reshaped, probs

    def sample_st(self, probs: torch.Tensor) -> torch.Tensor:
        """Straight-through categorical sampling: sample discrete one-hot, gradient flows to probs."""
        dist = torch.distributions.OneHotCategorical(probs=probs)
        sample = dist.sample()
        st_sample = sample + probs - probs.detach()
        return st_sample.view(-1, self.stoch_dim)

    def step_prior(
        self, prev_h: torch.Tensor, prev_z: torch.Tensor, prev_a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.silu(self.act_embed(torch.cat([prev_z, prev_a], dim=-1)))
        next_h = self.cell(x, prev_h)
        logits, probs = self.get_dist(self.prior_net(next_h))
        next_z = self.sample_st(probs)
        return next_h, next_z, logits, probs

    def step_post(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        prev_a: torch.Tensor,
        embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.silu(self.act_embed(torch.cat([prev_z, prev_a], dim=-1)))
        next_h = self.cell(x, prev_h)
        logits, probs = self.get_dist(
            self.post_net(torch.cat([next_h, embed], dim=-1))
        )
        next_z = self.sample_st(probs)
        return next_h, next_z, logits, probs

    def step_post_deterministic(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        prev_a: torch.Tensor,
        embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fast deterministic step for evaluation/deployment using mode (argmax)."""
        x = F.silu(self.act_embed(torch.cat([prev_z, prev_a], dim=-1)))
        next_h = self.cell(x, prev_h)
        logits = self.post_net(torch.cat([next_h, embed], dim=-1)).view(-1, self.k, self.c)
        idx = torch.argmax(logits, dim=-1)
        next_z = F.one_hot(idx, num_classes=self.c).float().view(-1, self.stoch_dim)
        return next_h, next_z

    def observe_sequence(
        self,
        embeds: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor,
        *,
        burnin: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Infer posterior states for T+1 observations and T transitions."""
        batch_size, observation_count = embeds.shape[:2]
        transition_count = actions.shape[1]
        if observation_count != transition_count + 1:
            raise ValueError("an observation sequence must have one more frame than transitions")
        if is_first.shape != actions.shape[:2]:
            raise ValueError("is_first must have one flag per transition")
        if burnin < 0 or burnin >= transition_count:
            raise ValueError("burnin must be nonnegative and leave at least one learning transition")

        h = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)
        z = torch.zeros(batch_size, self.stoch_dim, device=embeds.device)
        prev_a = torch.zeros(batch_size, self.action_dim, device=embeds.device)
        states, prior_logits, posterior_logits = [], [], []

        for t in range(observation_count):
            if t < transition_count:
                first_mask = is_first[:, t].unsqueeze(-1).float()
            else:
                first_mask = torch.zeros(batch_size, 1, device=embeds.device)
            h = h * (1.0 - first_mask)
            z = z * (1.0 - first_mask)
            prev_a = prev_a * (1.0 - first_mask)

            if t < burnin:
                with torch.no_grad():
                    _, _, pr_logits, _ = self.step_prior(h, z, prev_a)
                    post_h, post_z, po_logits, _ = self.step_post(
                        h, z, prev_a, embeds[:, t]
                    )
            else:
                _, _, pr_logits, _ = self.step_prior(h, z, prev_a)
                post_h, post_z, po_logits, _ = self.step_post(
                    h, z, prev_a, embeds[:, t]
                )
            h, z = post_h, post_z
            states.append(torch.cat([h, z], dim=-1))
            prior_logits.append(pr_logits)
            posterior_logits.append(po_logits)

            if t < transition_count:
                prev_a = actions[:, t]

        return (
            torch.stack(states, dim=1),
            torch.stack(prior_logits, dim=1),
            torch.stack(posterior_logits, dim=1),
        )


class TwoHotHead(nn.Module):
    """Categorical prediction over the Dreamer symlog-spaced raw-value bins."""

    def __init__(self, in_features: int = 512, bins: int = 255):
        super().__init__()
        if bins < 3 or bins % 2 != 1:
            raise ValueError("two-hot bin count must be an odd integer of at least 3")
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, bins),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        half = torch.linspace(-20.0, 0.0, (bins - 1) // 2 + 1)
        negative_half = torch.sign(half) * torch.expm1(torch.abs(half))
        centers = torch.cat([negative_half, -negative_half[:-1].flip(0)])
        self.register_buffer("bin_centers", centers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def pred_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        midpoint = self.bin_centers.numel() // 2
        center = probs[..., midpoint] * self.bin_centers[midpoint]
        negative_pairs = (
            probs[..., :midpoint] * self.bin_centers[:midpoint]
        ).flip(-1)
        positive_pairs = probs[..., midpoint + 1:] * self.bin_centers[midpoint + 1:]
        return center + (negative_pairs + positive_pairs).sum(dim=-1)

    def pred(self, state: torch.Tensor) -> torch.Tensor:
        return self.pred_from_logits(self(state))

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=logits.dtype)
        upper = torch.searchsorted(self.bin_centers, target.contiguous()).clamp(
            min=0, max=self.bin_centers.numel() - 1
        )
        lower = (upper - 1).clamp(min=0)
        lower_value = self.bin_centers[lower]
        upper_value = self.bin_centers[upper]
        fraction = ((target - lower_value) / (upper_value - lower_value).clamp_min(1e-8)).clamp(0.0, 1.0)
        log_probs = F.log_softmax(logits, dim=-1)
        lower_log_prob = log_probs.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
        upper_log_prob = log_probs.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
        return -((1.0 - fraction) * lower_log_prob + fraction * upper_log_prob)


class RewardHead(TwoHotHead):
    """Predicts raw rewards with Dreamer's two-hot categorical target."""


class Critic(TwoHotHead):
    """Predicts raw state values with Dreamer's two-hot categorical target."""


class ContinueHead(nn.Module):
    """Predicts continuation probability (1 - terminal)."""

    def __init__(self, in_features: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Actor(nn.Module):
    """Dreamer bounded-normal policy; sampled actions are clipped at execution."""

    def __init__(self, in_features: int = 512, action_dim: int = 3):
        super().__init__()
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(256, action_dim)
        self.std_head = nn.Linear(256, action_dim)
        nn.init.normal_(self.mean_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.normal_(self.std_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.std_head.bias)

    def distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        features = self.trunk(state)
        mean = torch.tanh(self.mean_head(features))
        std = torch.sigmoid(self.std_head(features) + 2.0) * 0.9 + 0.1
        return torch.distributions.Normal(mean, std)

    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(state)
        if deterministic:
            return dist.mean, dist.mean
        action = dist.rsample()
        return action, action

    def log_prob(self, state: torch.Tensor, sampled_action: torch.Tensor) -> torch.Tensor:
        return self.distribution(state).log_prob(sampled_action.detach()).sum(dim=-1)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        return self.distribution(state).entropy().sum(dim=-1)


class NoValidSequenceError(ValueError):
    """Raised when replay has no same-episode window with a successor frame."""


class Uint8SequenceReplay:
    """Ring buffer storing transitions and sampling (T+1 observations, T actions)."""

    def __init__(self, capacity: int = 100000):
        self.capacity = int(capacity)
        self.observations = np.empty((capacity, 4, 84, 84), dtype=np.uint8)
        self.actions = np.empty((capacity, 3), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.is_first = np.empty(capacity, dtype=bool)
        self.is_last = np.empty(capacity, dtype=bool)
        self.is_terminal = np.empty(capacity, dtype=bool)
        self.sequence_ids = np.empty(capacity, dtype=np.int64)
        self._boundary_observations: dict[int, np.ndarray] = {}
        self._terminal_sequence_ids: set[int] = set()
        self._episode_start_sequence_ids: set[int] = set()

        self.cursor = 0
        self.size = 0
        self.total_steps = 0

    def add(self, transition: Transition, observation_spec: ObservationSpec | None = None) -> None:
        spec = observation_spec or ObservationSpec()
        obs = transition.observation
        if obs.dtype != np.uint8:
            obs = spec.to_uint8(obs)
        if obs.shape != self.observations.shape[1:]:
            raise ValueError("transition observation has the wrong shape")

        next_obs = None
        if transition.is_last or transition.is_terminal:
            next_obs = transition.next_observation
            if next_obs.dtype != np.uint8:
                next_obs = spec.to_uint8(next_obs)
            if next_obs.shape != self.observations.shape[1:]:
                raise ValueError("transition next_observation has the wrong shape")

        if self.size == self.capacity:
            overwritten_id = int(self.sequence_ids[self.cursor])
            self._boundary_observations.pop(overwritten_id, None)
            self._terminal_sequence_ids.discard(overwritten_id)
            self._episode_start_sequence_ids.discard(overwritten_id)

        self.observations[self.cursor] = obs
        self.actions[self.cursor] = transition.action
        self.rewards[self.cursor] = transition.reward
        self.is_first[self.cursor] = transition.is_first
        self.is_last[self.cursor] = transition.is_last
        self.is_terminal[self.cursor] = transition.is_terminal
        self.sequence_ids[self.cursor] = self.total_steps
        if next_obs is not None:
            self._boundary_observations[self.total_steps] = next_obs.copy()
        if transition.is_terminal:
            self._terminal_sequence_ids.add(self.total_steps)
        if transition.is_first:
            self._episode_start_sequence_ids.add(self.total_steps)

        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_steps += 1

    def sample_sequence(
        self,
        batch_size: int,
        seq_len: int,
        *,
        burnin: int = 0,
        terminal_fraction: float = 0.0,
        reset_start_fraction: float = 0.0,
        short_episode_fraction: float = 0.0,
    ) -> dict[str, torch.Tensor | int | bool]:
        if (
            batch_size <= 0
            or seq_len <= 0
            or burnin < 0
            or not 0.0 <= terminal_fraction <= 1.0
            or not 0.0 <= reset_start_fraction <= 1.0
            or not 0.0 <= short_episode_fraction <= 1.0
        ):
            raise ValueError("invalid sequence sampling size, burnin, or sampling fraction")
        if self.size < (1 if short_episode_fraction else seq_len):
            raise NoValidSequenceError(f"replay has {self.size} steps; need at least {seq_len}")

        def is_valid_start(start: int, count: int) -> bool:
            if self.size < self.capacity and start + count > self.size:
                return False
            idx = (np.arange(start, start + count) % self.capacity).astype(np.int64)
            ids = self.sequence_ids[idx]
            if ids[-1] - ids[0] != count - 1 or np.any(np.diff(ids) != 1):
                return False
            if np.any(self.is_last[idx[:-1]] | self.is_terminal[idx[:-1]]):
                return False
            if count > 1 and np.any(self.is_first[idx[1:]]):
                return False

            endpoint_id = int(ids[-1])
            if self.is_last[idx[-1]] or self.is_terminal[idx[-1]]:
                return endpoint_id in self._boundary_observations

            successor = (int(idx[-1]) + 1) % self.capacity
            if self.size < self.capacity and successor >= self.size:
                return False
            return (
                self.sequence_ids[successor] == endpoint_id + 1
                and not self.is_first[successor]
            )

        short_options: dict[int, list[int]] = {}
        if short_episode_fraction:
            for endpoint_id in sorted(self._boundary_observations):
                endpoint = endpoint_id % self.capacity
                if self.sequence_ids[endpoint] != endpoint_id or not self.is_last[endpoint]:
                    continue
                for length in range(1, min(seq_len, self.size + 1)):
                    start_id = endpoint_id - length + 1
                    if start_id in self._episode_start_sequence_ids:
                        start = start_id % self.capacity
                        if is_valid_start(start, length):
                            short_options.setdefault(length, []).append(start)

        if short_episode_fraction == 1.0 and not short_options:
            raise NoValidSequenceError("replay has no valid complete short episode")
        use_short = bool(short_options) and (
            self.size < seq_len + burnin
            or short_episode_fraction == 1.0
            or np.random.random() < short_episode_fraction
        )
        short_length = int(np.random.choice(sorted(short_options))) if use_short else seq_len

        reset_options: dict[int, list[int]] = {}
        if reset_start_fraction and not use_short:
            for sequence_id in sorted(self._episode_start_sequence_ids):
                start = sequence_id % self.capacity
                if self.sequence_ids[start] != sequence_id or not self.is_first[start]:
                    continue
                for prefix in range(min(burnin, self.size - seq_len) + 1):
                    if is_valid_start(start, seq_len + prefix):
                        reset_options.setdefault(prefix, []).append(start)

        if reset_start_fraction == 1.0 and not reset_options and not use_short:
            raise NoValidSequenceError("replay has no valid reset-origin sequence")
        use_reset = not use_short and bool(reset_options) and (
            self.size < seq_len + burnin
            or reset_start_fraction == 1.0
            or np.random.random() < reset_start_fraction
        )
        effective_burnin = (
            0 if use_short else int(np.random.choice(sorted(reset_options))) if use_reset else burnin
        )
        transition_count = short_length + effective_burnin
        if self.size < transition_count:
            raise NoValidSequenceError(f"replay has {self.size} steps; need at least {transition_count}")
        reset_starts = reset_options[effective_burnin] if use_reset else []
        reset_start_set = set(reset_starts)
        short_starts = short_options[short_length] if use_short else []
        short_start_set = set(short_starts)

        available_starts = (
            self.size - transition_count + 1 if self.size < self.capacity else self.capacity
        )
        terminal_starts = []
        if terminal_fraction:
            for sequence_id in sorted(self._terminal_sequence_ids):
                endpoint = sequence_id % self.capacity
                if (
                    int(self.sequence_ids[endpoint]) != sequence_id
                    or not self.is_terminal[endpoint]
                ):
                    continue
                start_id = sequence_id - transition_count + 1
                if start_id < 0:
                    continue
                start = start_id % self.capacity
                if is_valid_start(start, transition_count) and (
                    (not use_reset or start in reset_start_set)
                    and (not use_short or start in short_start_set)
                ):
                    terminal_starts.append(start)

        terminal_count = (
            min(batch_size, int(np.ceil(batch_size * terminal_fraction)))
            if terminal_fraction > 0.0 and terminal_starts
            else 0
        )
        start_indices = (
            np.random.choice(terminal_starts, size=terminal_count, replace=True).tolist()
            if terminal_count
            else []
        )
        terminal_anchored = [True] * len(start_indices)

        # Special batches keep a common shape using only genuine episode starts.
        uniform_needed = batch_size - terminal_count
        uniform_starts = []
        uniform_anchored = []
        if (use_reset or use_short) and uniform_needed:
            uniform_starts = np.random.choice(
                short_starts if use_short else reset_starts,
                size=uniform_needed, replace=True,
            ).tolist()
        max_attempts = max(uniform_needed * 50, 100)
        attempts = 0
        while not (use_reset or use_short) and len(uniform_starts) < uniform_needed and attempts < max_attempts:
            attempts += 1
            start = np.random.randint(0, available_starts)
            if is_valid_start(start, transition_count):
                uniform_starts.append(start)

        if uniform_needed and not uniform_starts:
            valid_starts = [
                start for start in range(available_starts) if is_valid_start(start, transition_count)
            ]
            if valid_starts:
                uniform_starts = np.random.choice(
                    valid_starts, size=uniform_needed, replace=True
                ).tolist()
                uniform_anchored = [False] * uniform_needed
            elif terminal_starts:
                uniform_starts = np.random.choice(
                    terminal_starts, size=uniform_needed, replace=True
                ).tolist()
                uniform_anchored = [True] * uniform_needed
            elif reset_options:
                return self.sample_sequence(
                    batch_size, seq_len, burnin=burnin,
                    terminal_fraction=terminal_fraction, reset_start_fraction=1.0,
                    short_episode_fraction=short_episode_fraction,
                )
            elif short_options:
                return self.sample_sequence(
                    batch_size, seq_len, burnin=burnin,
                    terminal_fraction=terminal_fraction, reset_start_fraction=reset_start_fraction,
                    short_episode_fraction=1.0,
                )
            else:
                raise NoValidSequenceError(
                    "replay has no valid transition sequence with a successor observation"
                )
        elif uniform_needed:
            if len(uniform_starts) < uniform_needed:
                uniform_starts.extend(
                    np.random.choice(
                        uniform_starts,
                        size=uniform_needed - len(uniform_starts),
                        replace=True,
                    ).tolist()
                )
            uniform_anchored = [False] * len(uniform_starts)

        start_indices.extend(uniform_starts)
        terminal_anchored.extend(uniform_anchored)
        if terminal_fraction > 0.0:
            order = np.random.permutation(batch_size)
            start_indices = [start_indices[index] for index in order]
            terminal_anchored = [terminal_anchored[index] for index in order]

        batch_idx = np.empty((batch_size, transition_count), dtype=np.int64)
        for b, start in enumerate(start_indices):
            batch_idx[b] = (np.arange(start, start + transition_count)) % self.capacity

        obs = np.empty(
            (batch_size, transition_count + 1, *self.observations.shape[1:]), dtype=np.uint8
        )
        obs[:, :transition_count] = self.observations[batch_idx]
        for b in range(batch_size):
            endpoint = int(batch_idx[b, -1])
            endpoint_id = int(self.sequence_ids[endpoint])
            if self.is_last[endpoint] or self.is_terminal[endpoint]:
                obs[b, transition_count] = self._boundary_observations[endpoint_id]
            else:
                successor = (endpoint + 1) % self.capacity
                obs[b, transition_count] = self.observations[successor]

        obs = torch.from_numpy(obs).float() / 255.0
        actions = torch.from_numpy(self.actions[batch_idx])
        rewards = torch.from_numpy(self.rewards[batch_idx])
        is_first = torch.from_numpy(self.is_first[batch_idx])
        is_last = torch.from_numpy(self.is_last[batch_idx])
        is_terminal = torch.from_numpy(self.is_terminal[batch_idx])

        return {
            "observations": obs,
            "actions": actions,
            "rewards": rewards,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "sequence_ids": torch.from_numpy(self.sequence_ids[batch_idx]),
            "terminal_anchored": torch.tensor(terminal_anchored, dtype=torch.bool),
            "effective_burnin": effective_burnin,
            "effective_seq_len": short_length,
            "reset_start_anchored": use_reset,
            "short_episode_anchored": use_short,
        }

    @property
    def memory_bytes(self) -> int:
        return (
            self.observations.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.is_first.nbytes
            + self.is_last.nbytes
            + self.is_terminal.nbytes
            + self.sequence_ids.nbytes
            + sum(obs.nbytes for obs in self._boundary_observations.values())
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "cursor": self.cursor,
            "size": self.size,
            "total_steps": self.total_steps,
            "observations": self.observations[:self.size].copy(),
            "actions": self.actions[:self.size].copy(),
            "rewards": self.rewards[:self.size].copy(),
            "is_first": self.is_first[:self.size].copy(),
            "is_last": self.is_last[:self.size].copy(),
            "is_terminal": self.is_terminal[:self.size].copy(),
            "sequence_ids": self.sequence_ids[:self.size].copy(),
            "boundary_observations": {
                sequence_id: obs.copy()
                for sequence_id, obs in self._boundary_observations.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.cursor = int(state["cursor"])
        self.size = int(state["size"])
        self.total_steps = int(state["total_steps"])
        n = self.size
        self.observations[:n] = state["observations"]
        self.actions[:n] = state["actions"]
        self.rewards[:n] = state["rewards"]
        self.is_first[:n] = state["is_first"]
        self.is_last[:n] = state["is_last"]
        self.is_terminal[:n] = state["is_terminal"]
        self.sequence_ids[:n] = state["sequence_ids"]
        self._boundary_observations = {
            int(sequence_id): np.asarray(obs, dtype=np.uint8).copy()
            for sequence_id, obs in state.get("boundary_observations", {}).items()
        }
        self._terminal_sequence_ids = {
            int(self.sequence_ids[index])
            for index in range(self.size)
            if self.is_terminal[index]
        }
        self._episode_start_sequence_ids = {
            int(self.sequence_ids[index])
            for index in range(self.size)
            if self.is_first[index]
        }


@dataclass
class DreamerV3Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    embed_dim: int = 512
    hidden_dim: int = 256
    num_categoricals: int = 16
    num_classes: int = 16
    unimix: float = 0.01
    imagination_horizon: int = 15
    gamma: float = 0.99
    td_lambda: float = 0.95
    critic_ema_tau: float = 0.02
    actor_entropy_coeff: float = 3e-4
    continue_positive_weight: float = 1.0
    slow_critic_weight: float = 1.0
    replay_value_weight: float = 1.0
    twohot_bins: int = 255
    observation_loss_scale: float = 1.0
    overshoot_horizon: int = 1
    overshoot_kl_weight: float = 0.0
    overshoot_free_nats: float = 0.0
    kl_dyn_weight: float = 1.0
    kl_rep_weight: float = 0.1
    kl_free_nats: float = 1.0
    model_lr: float = 1e-4
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    grad_clip_norm: float = 100.0
    batch_size: int = 16
    seq_len: int = 32
    burnin_steps: int = 8
    terminal_window_fraction: float = 0.0
    reset_start_fraction: float = 0.0
    short_episode_fraction: float = 0.0
    replay_capacity: int = 100000
    warmup_steps: int = 10000
    updates_per_step: int = 1
    seed: int = 0


class ExportedDreamerV3Actor(nn.Module):
    """Self-contained recurrent CPU inference actor for HAIC evaluation and submission."""

    def __init__(self, encoder: ConvEncoder, rssm: CategoricalRSSM, actor: Actor):
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.actor = actor
        self.hidden_dim = rssm.hidden_dim
        self.stoch_dim = rssm.stoch_dim
        self.register_buffer("h", torch.zeros(1, self.hidden_dim))
        self.register_buffer("z", torch.zeros(1, self.stoch_dim))
        self.register_buffer("prev_a", torch.zeros(1, 3))
        self.is_first = True

    def reset_episode(self) -> None:
        self.h.zero_()
        self.z.zero_()
        self.prev_a.zero_()
        self.is_first = True

    @torch.inference_mode()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> np.ndarray:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        e = self.encoder(obs)
        if self.is_first:
            self.h.zero_()
            self.z.zero_()
            self.prev_a.zero_()
            self.is_first = False

        next_h, next_z = self.rssm.step_post_deterministic(
            self.h, self.z, self.prev_a, e
        )
        self.h.copy_(next_h)
        self.z.copy_(next_z)

        state = torch.cat([self.h, self.z], dim=-1)
        action, _ = self.actor(state, deterministic=deterministic)
        self.prev_a.copy_(action)
        return action.squeeze(0).cpu().numpy()


class DreamerV3Agent:
    """DreamerV3 World Model and Actor-Critic Agent for HAIC visual control."""

    def __init__(self, config: DreamerV3Config | None = None, seed: int = 0):
        self.config = config or DreamerV3Config()
        self.seed = int(seed)
        self.device = torch.device(self.config.device)
        self.action_adapter = ActionAdapter(ActionSpec())
        self.observation_spec = ObservationSpec()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.encoder = ConvEncoder(in_channels=4, embed_dim=self.config.embed_dim).to(self.device)
        self.rssm = CategoricalRSSM(
            action_dim=3,
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            num_categoricals=self.config.num_categoricals,
            num_classes=self.config.num_classes,
            unimix=self.config.unimix,
        ).to(self.device)
        self.decoder = ConvDecoder(
            in_features=self.rssm.state_dim, out_channels=1
        ).to(self.device)
        self.reward_head = RewardHead(
            in_features=self.rssm.state_dim, bins=self.config.twohot_bins
        ).to(self.device)
        self.continue_head = ContinueHead(in_features=self.rssm.state_dim).to(self.device)

        self.actor = Actor(in_features=self.rssm.state_dim, action_dim=3).to(self.device)
        self.critic = Critic(
            in_features=self.rssm.state_dim, bins=self.config.twohot_bins
        ).to(self.device)
        self.critic_target = Critic(
            in_features=self.rssm.state_dim, bins=self.config.twohot_bins
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        wm_params = (
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_head.parameters())
            + list(self.continue_head.parameters())
        )
        self.wm_optimizer = torch.optim.Adam(wm_params, lr=self.config.model_lr, eps=1e-8)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr, eps=1e-8)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr, eps=1e-8)

        # Replay
        self.replay = Uint8SequenceReplay(capacity=self.config.replay_capacity)

        # Online recurrent state
        self._online_h = torch.zeros(1, self.rssm.hidden_dim, device=self.device)
        self._online_z = torch.zeros(1, self.rssm.stoch_dim, device=self.device)
        self._online_prev_a = torch.zeros(1, 3, device=self.device)
        self._online_first = True
        self._online_observation_pending = False
        self._last_policy_stats: dict[str, np.ndarray] | None = None

        # Counters & telemetry
        self.environment_steps = 0
        self.gradient_steps = 0
        self._ret_p5 = 0.0
        self._ret_p95 = 1.0

    def reset_episode(self) -> None:
        self._online_h.zero_()
        self._online_z.zero_()
        self._online_prev_a.zero_()
        self._online_first = True
        self._online_observation_pending = False
        self._last_policy_stats = None

    def observe(self, transition: Transition) -> None:
        self.replay.add(transition, self.observation_spec)
        self.environment_steps += 1

        if not self._online_observation_pending:
            if transition.is_first:
                self.reset_episode()
            self._advance_online_observation(transition.observation, deterministic=False)

        executed_action = torch.as_tensor(
            transition.action, dtype=torch.float32, device=self.device
        ).view(1, -1)
        if executed_action.shape != self._online_prev_a.shape:
            raise ValueError("transition action has the wrong shape for Dreamer online state")
        self._online_prev_a.copy_(executed_action)
        self._online_observation_pending = False

    @torch.inference_mode()
    def _advance_online_observation(
        self, observation: np.ndarray, *, deterministic: bool
    ) -> torch.Tensor:
        obs_t = torch.from_numpy(np.ascontiguousarray(observation)).float().unsqueeze(0).to(self.device)
        if obs_t.max() > 1.0:
            obs_t = obs_t / 255.0

        embed = self.encoder(obs_t)
        if self._online_first:
            self._online_h.zero_()
            self._online_z.zero_()
            self._online_prev_a.zero_()
            self._online_first = False

        if deterministic:
            h, z = self.rssm.step_post_deterministic(
                self._online_h, self._online_z, self._online_prev_a, embed
            )
        else:
            h, z, _, _ = self.rssm.step_post(
                self._online_h, self._online_z, self._online_prev_a, embed
            )

        self._online_h.copy_(h)
        self._online_z.copy_(z)
        return torch.cat([self._online_h, self._online_z], dim=-1)

    def _overshooting_kl_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        posterior_logits: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, transition_count = actions.shape[:2]
        horizon = min(self.config.overshoot_horizon, transition_count)
        if horizon <= 1:
            return states.new_zeros(())
        if states.shape[1] != transition_count + 1 or posterior_logits.shape[1] != transition_count + 1:
            raise ValueError("overshooting requires T+1 posterior states for T actions")

        posterior_h, posterior_z = torch.split(
            states[:, :-1],
            [self.rssm.hidden_dim, self.rssm.stoch_dim],
            dim=-1,
        )
        prior_h = prior_z = None
        losses = []
        for step in range(horizon):
            start_count = transition_count - step
            if step == 0:
                start_h = posterior_h[:, :start_count]
                start_z = posterior_z[:, :start_count]
            else:
                start_h = prior_h[:, :start_count]
                start_z = prior_z[:, :start_count]
            step_actions = actions[:, step:step + start_count]
            next_h, next_z, prior_logits, _ = self.rssm.step_prior(
                start_h.reshape(batch_size * start_count, -1),
                start_z.reshape(batch_size * start_count, -1),
                step_actions.reshape(batch_size * start_count, -1),
            )
            prior_h = next_h.reshape(batch_size, start_count, -1)
            prior_z = next_z.reshape(batch_size, start_count, -1)
            if step == 0:
                continue

            target_logits = posterior_logits[:, step + 1:step + 1 + start_count]
            _, prior_probs = self.rssm.get_dist(prior_logits)
            _, target_probs = self.rssm.get_dist(
                target_logits.reshape(batch_size * start_count, *target_logits.shape[2:])
            )
            prior = prior_probs.clamp(min=1e-7)
            target = target_probs.detach().clamp(min=1e-7)
            kl = (
                target * (torch.log(target) - torch.log(prior))
            ).sum(dim=-1).sum(dim=-1)
            losses.append(kl.clamp(min=self.config.overshoot_free_nats).mean())

        return torch.stack(losses).mean() if losses else states.new_zeros(())

    @torch.inference_mode()
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state = self._advance_online_observation(observation, deterministic=deterministic)
        distribution = self.actor.distribution(state)
        action = distribution.mean if deterministic else distribution.rsample()
        self._last_policy_stats = {
            "mean": distribution.mean.squeeze(0).cpu().numpy().copy(),
            "std": distribution.stddev.squeeze(0).cpu().numpy().copy(),
            "entropy": distribution.entropy().squeeze(0).cpu().numpy().copy(),
        }
        self._online_prev_a.copy_(action)
        self._online_observation_pending = True
        return action.squeeze(0).cpu().numpy()

    def update(self, *, model_only: bool = False) -> dict[str, float]:
        total_transitions = self.config.burnin_steps + self.config.seq_len
        minimum_transitions = (
            1 if self.config.short_episode_fraction else
            self.config.seq_len if self.config.reset_start_fraction else total_transitions
        )
        if self.replay.size < minimum_transitions or (
            not model_only and self.environment_steps < self.config.warmup_steps
        ):
            return {}

        try:
            batch = self.replay.sample_sequence(
                self.config.batch_size,
                self.config.seq_len,
                burnin=self.config.burnin_steps,
                terminal_fraction=self.config.terminal_window_fraction,
                reset_start_fraction=self.config.reset_start_fraction,
                short_episode_fraction=self.config.short_episode_fraction,
            )
        except NoValidSequenceError:
            return {}
        obs = batch["observations"].to(self.device)  # (B, T+1, 4, 84, 84)
        actions = batch["actions"].to(self.device)    # (B, burnin+T, 3)
        rewards = batch["rewards"].to(self.device)    # (B, burnin+T)
        is_first = batch["is_first"].to(self.device)  # (B, burnin+T)
        is_terminal = batch["is_terminal"].to(self.device)  # (B, burnin+T)

        B, transition_count = actions.shape[:2]
        burnin = batch["effective_burnin"]
        T = batch["effective_seq_len"]
        if transition_count != burnin + T or obs.shape[1] != transition_count + 1:
            raise ValueError("replay must provide one successor observation per transition sequence")

        # 1. Encode current and successor observations.
        observation_count = transition_count + 1
        embeds = self.encoder(
            obs.view(B * observation_count, 4, 84, 84)
        ).view(B, observation_count, -1)

        # Burn-in reconstructs the sampled prefix without training through it.
        states, pr_logits_stack, po_logits_stack = self.rssm.observe_sequence(
            embeds, actions, is_first, burnin=burnin
        )  # states/logits: (B, T+1, ...)

        learning_states = states[:, burnin:burnin + T + 1]
        learning_observations = obs[:, burnin:burnin + T + 1]
        learning_rewards = rewards[:, burnin:]
        learning_terminals = is_terminal[:, burnin:]
        learning_prior_logits = pr_logits_stack[:, burnin:burnin + T + 1]
        learning_posterior_logits = po_logits_stack[:, burnin:burnin + T + 1]
        terminal_labels = batch["is_terminal"][:, burnin:]
        terminal_ids = batch["sequence_ids"][:, burnin:][terminal_labels]
        sampling_metrics = {
            "effective_burnin": float(burnin),
            "effective_seq_len": float(T),
            "reset_start_anchored": float(batch["reset_start_anchored"]),
            "short_episode_anchored": float(batch["short_episode_anchored"]),
            "unique_short_episodes": float(torch.unique(batch["sequence_ids"][:, 0]).numel())
            if batch["short_episode_anchored"] else 0.0,
            "terminal_label_count": float(terminal_labels.sum().item()),
            "terminal_label_fraction": float(terminal_labels.float().mean().item()),
            "unique_terminal_events": float(torch.unique(terminal_ids).numel()),
        }

        # 3. World Model Losses
        # A. Predict only the action-conditioned change in the newest frame.
        result_states = learning_states[:, 1:]
        predicted_delta = self.decoder(
            result_states.reshape(B * T, -1)
        ).view(B, T, 1, 84, 84)
        previous_frame = learning_observations[:, :-1, -1:]
        next_frame = learning_observations[:, 1:, -1:]
        target_delta = next_frame - previous_frame
        obs_loss = (
            0.5
            * self.config.observation_loss_scale
            * F.mse_loss(predicted_delta, target_delta)
        )

        # B. Reward and continuation belong to the state reached by each action.
        pred_reward_logits = self.reward_head(result_states)
        reward_loss = self.reward_head.loss(pred_reward_logits, learning_rewards).mean()

        pred_cont = self.continue_head(result_states).squeeze(-1)
        cont_target = (1.0 - learning_terminals.float())
        continue_loss = continue_prediction_loss(
            pred_cont, cont_target, self.config.continue_positive_weight
        )

        # D. KL balancing loss
        _, pr_probs = self.rssm.get_dist(learning_prior_logits)
        _, po_probs = self.rssm.get_dist(learning_posterior_logits)
        kl_loss, kl_dyn, kl_rep = balanced_kl_loss(
            pr_probs,
            po_probs,
            dyn_weight=self.config.kl_dyn_weight,
            rep_weight=self.config.kl_rep_weight,
            free_nats=self.config.kl_free_nats,
        )

        overshoot_loss = (
            self._overshooting_kl_loss(
                learning_states,
                actions[:, burnin:],
                learning_posterior_logits,
            )
            if self.config.overshoot_kl_weight > 0.0
            else states.new_zeros(())
        )
        wm_loss = (
            obs_loss
            + reward_loss
            + continue_loss
            + kl_loss
            + self.config.overshoot_kl_weight * overshoot_loss
        )

        self.wm_optimizer.zero_grad()
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_head.parameters())
            + list(self.continue_head.parameters()),
            self.config.grad_clip_norm,
        )
        self.wm_optimizer.step()

        if model_only:
            self.gradient_steps += 1
            return {
                "loss_wm": float(wm_loss.item()),
                "loss_obs": float(obs_loss.item()),
                "loss_reward": float(reward_loss.item()),
                "loss_continue": float(continue_loss.item()),
                "loss_kl": float(kl_loss.item()),
                "loss_overshoot_kl": float(overshoot_loss.item()),
                "kl_dyn_mean": float(kl_dyn.mean().item()),
                "kl_rep_mean": float(kl_rep.mean().item()),
                "terminal_anchored_fraction": float(
                    batch["terminal_anchored"].float().mean().item()
                ),
                **sampling_metrics,
            }

        # 4. Latent Imagination and Actor-Critic
        # Sample starting states from detached posterior states
        flat_states = learning_states[:, :-1].detach().reshape(B * T, -1)
        # Select subset of states for imagination to keep budget fast
        sample_size = min(B * T, 128)
        perm = torch.randperm(B * T, device=self.device)[:sample_size]
        init_s = flat_states[perm]
        init_h, init_z = torch.split(init_s, [self.rssm.hidden_dim, self.rssm.stoch_dim], dim=-1)

        H = self.config.imagination_horizon
        im_h, im_z = init_h, init_z
        im_states = [torch.cat([im_h, im_z], dim=-1)]
        im_raw_actions = []
        im_rewards = []
        im_continues = []

        for step in range(H):
            curr_s = torch.cat([im_h, im_z], dim=-1)
            sampled_action, raw_sample = self.actor(curr_s, deterministic=False)
            executed_action = sampled_action.clamp(-1.0, 1.0)
            next_h, next_z, _, _ = self.rssm.step_prior(
                im_h, im_z, executed_action
            )
            next_s = torch.cat([next_h, next_z], dim=-1)

            r = self.reward_head.pred(next_s)
            c = torch.sigmoid(self.continue_head(next_s).squeeze(-1))

            im_h, im_z = next_h, next_z
            im_states.append(next_s)
            im_raw_actions.append(raw_sample)
            im_rewards.append(r)
            im_continues.append(c)

        im_states = torch.stack(im_states)  # (H+1, sample_size, state_dim)
        im_rewards = torch.stack(im_rewards)  # (H, sample_size)
        im_continues = torch.stack(im_continues)  # (H, sample_size)

        # Value targets via lambda-return
        with torch.no_grad():
            values = self.critic.pred(im_states)  # (H+1, sample_size)
            targets = lambda_returns(
                im_rewards,
                im_continues,
                values,
                self.config.gamma,
                self.config.td_lambda,
            )

            current_values = values[:-1]
            flat_targets = targets.reshape(-1)
            p5 = torch.quantile(flat_targets, 0.05).item()
            p95 = torch.quantile(flat_targets, 0.95).item()
            self._ret_p5 = 0.9 * self._ret_p5 + 0.1 * p5
            self._ret_p95 = 0.9 * self._ret_p95 + 0.1 * p95
            return_scale = max(self._ret_p95 - self._ret_p5, 1.0)
            advantages = ((targets - current_values) / return_scale).detach()

            weights = continuation_weights(im_continues, self.config.gamma)
            replay_values = self.critic.pred(learning_states).transpose(0, 1)
            replay_continues = (1.0 - learning_terminals.float()).transpose(0, 1)
            replay_targets = lambda_returns(
                learning_rewards.transpose(0, 1),
                replay_continues,
                replay_values,
                self.config.gamma,
                self.config.td_lambda,
            )
            replay_weights = continuation_weights(
                replay_continues, self.config.gamma
            ).transpose(0, 1)

        # Critic update
        critic_states = im_states[:-1].detach()
        pred_value_logits = self.critic(critic_states)
        target_loss = self.critic.loss(pred_value_logits, targets)
        with torch.no_grad():
            slow_values = self.critic_target.pred(im_states[:-1])
        slow_value_loss = self.critic.loss(pred_value_logits, slow_values)
        imagination_critic_loss = (
            weights.detach()
            * (target_loss + self.config.slow_critic_weight * slow_value_loss)
        ).mean()
        replay_value_logits = self.critic(learning_states[:, :-1].detach())
        replay_value_loss = (
            replay_weights.detach()
            * self.critic.loss(replay_value_logits, replay_targets.transpose(0, 1))
        ).mean()
        critic_loss = imagination_critic_loss + self.config.replay_value_weight * replay_value_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_norm)
        self.critic_optimizer.step()

        # Only the actor parameters at each imagined state receive score/entropy gradients.
        score_states = im_states[:-1].detach()
        log_prob = self.actor.log_prob(score_states, torch.stack(im_raw_actions))
        entropy = self.actor.entropy(score_states)
        actor_loss = -(
            weights.detach()
            * (log_prob * advantages + self.config.actor_entropy_coeff * entropy)
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_grads = torch.autograd.grad(actor_loss, tuple(self.actor.parameters()))
        for parameter, gradient in zip(self.actor.parameters(), actor_grads):
            parameter.grad = gradient
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
        self.actor_optimizer.step()

        # Soft update target critic
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.copy_(
                    (1.0 - self.config.critic_ema_tau) * p_targ.data
                    + self.config.critic_ema_tau * p.data
                )

        self.gradient_steps += 1

        return {
            "loss_wm": float(wm_loss.item()),
            "loss_obs": float(obs_loss.item()),
            "loss_reward": float(reward_loss.item()),
            "loss_continue": float(continue_loss.item()),
            "loss_kl": float(kl_loss.item()),
            "loss_overshoot_kl": float(overshoot_loss.item()),
            "kl_dyn_mean": float(kl_dyn.mean().item()),
            "kl_rep_mean": float(kl_rep.mean().item()),
            "loss_critic": float(critic_loss.item()),
            "loss_replay_value": float(replay_value_loss.item()),
            "loss_actor": float(actor_loss.item()),
            "reward_pred_mean": float(self.reward_head.pred(result_states).mean().item()),
            "value_pred_mean": float(current_values.mean().item()),
            "actor_advantage_mean": float(advantages.mean().item()),
            "actor_entropy_mean": float(entropy.mean().item()),
            "imagination_weight_mean": float(weights.mean().item()),
            "terminal_anchored_fraction": float(
                batch["terminal_anchored"].float().mean().item()
            ),
            **sampling_metrics,
        }

    def save_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        source_paths: list[str | Path] | None = None,
        run_metadata: dict[str, Any] | None = None,
        trainer_state: dict[str, Any] | None = None,
    ) -> Path:
        path = Path(checkpoint_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format": "haic-dreamerv3-checkpoint-v3",
            "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
            "config": asdict(self.config),
            "encoder": self.encoder.state_dict(),
            "rssm": self.rssm.state_dict(),
            "decoder": self.decoder.state_dict(),
            "reward_head": self.reward_head.state_dict(),
            "continue_head": self.continue_head.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "return_scale": {"p5": self._ret_p5, "p95": self._ret_p95},
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "replay": self.replay.state_dict(),
            "run_metadata": run_metadata,
            "trainer_state": trainer_state,
        }
        if torch.cuda.is_available() and self.device.type == "cuda":
            payload["torch_cuda_rng_state"] = torch.cuda.get_rng_state(self.device)

        torch.save(payload, path)
        return path

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any] | None:
        path = Path(checkpoint_path).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_format = payload.get("format")
        if checkpoint_format in {
            "haic-dreamerv3-checkpoint-v1",
            "haic-dreamerv3-checkpoint-v2",
        }:
            raise ValueError(
                f"checkpoint {checkpoint_format} uses the full-stack decoder and is incompatible with residual-frame v3"
            )
        if checkpoint_format != "haic-dreamerv3-checkpoint-v3":
            raise ValueError(f"unsupported checkpoint format: {checkpoint_format}")

        self.environment_steps = int(payload["environment_steps"])
        self.gradient_steps = int(payload["gradient_steps"])
        self.encoder.load_state_dict(payload["encoder"])
        self.rssm.load_state_dict(payload["rssm"])
        self.decoder.load_state_dict(payload["decoder"])
        self.reward_head.load_state_dict(payload["reward_head"])
        self.continue_head.load_state_dict(payload["continue_head"])
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        return_scale = payload.get("return_scale", {})
        self._ret_p5 = float(return_scale.get("p5", 0.0))
        self._ret_p95 = float(return_scale.get("p95", 1.0))

        self.wm_optimizer.load_state_dict(payload["wm_optimizer"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])

        torch.set_rng_state(payload["torch_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        if "torch_cuda_rng_state" in payload and torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.set_rng_state(payload["torch_cuda_rng_state"], self.device)

        if "replay" in payload:
            self.replay.load_state_dict(payload["replay"])

        return payload.get("trainer_state")

    def export_actor(self, export_path: str | Path) -> Path:
        """Export standalone CPU actor payload for submission and CPU evaluation."""
        path = Path(export_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        encoder_cpu = ConvEncoder(in_channels=4, embed_dim=self.config.embed_dim)
        encoder_cpu.load_state_dict(self.encoder.state_dict())
        encoder_cpu.eval()

        rssm_cpu = CategoricalRSSM(
            action_dim=3,
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            num_categoricals=self.config.num_categoricals,
            num_classes=self.config.num_classes,
            unimix=self.config.unimix,
        )
        rssm_cpu.load_state_dict(self.rssm.state_dict())
        rssm_cpu.eval()

        actor_cpu = Actor(in_features=self.rssm.state_dim, action_dim=3)
        actor_cpu.load_state_dict(self.actor.state_dict())
        actor_cpu.eval()

        exported_model = ExportedDreamerV3Actor(encoder_cpu, rssm_cpu, actor_cpu)
        exported_model.eval()

        payload = {
            "format": DREAMERV3_ACTOR_FORMAT,
            "config": asdict(self.config),
            "observation_spec": asdict(self.observation_spec),
            "action_spec": asdict(self.action_adapter.spec),
            "state_dict": exported_model.state_dict(),
        }
        torch.save(payload, path)
        return path


def load_exported_actor(actor_path: str | Path, device: str = "cpu") -> ExportedDreamerV3Actor:
    payload = torch.load(actor_path, map_location="cpu", weights_only=True)
    if payload.get("format") != DREAMERV3_ACTOR_FORMAT:
        raise ValueError(f"expected format {DREAMERV3_ACTOR_FORMAT}, got {payload.get('format')}")
    cfg = payload["config"]
    encoder = ConvEncoder(in_channels=4, embed_dim=cfg["embed_dim"])
    rssm = CategoricalRSSM(
        action_dim=3,
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_categoricals=cfg["num_categoricals"],
        num_classes=cfg["num_classes"],
        unimix=cfg["unimix"],
    )
    actor = Actor(in_features=rssm.state_dim, action_dim=3)
    model = ExportedDreamerV3Actor(encoder, rssm, actor)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.to(device)
    return model
