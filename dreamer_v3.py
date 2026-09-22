"""Native PyTorch DreamerV3 implementation for HAIC visual control.

Strictly adheres to:
1. Frozen HAIC observation spec (4x84x84 float32 CHW in [0, 1]).
2. Frozen HAIC continuous action spec (symmetric native [-1, 1]^3 -> [steer, gas, brake]).
3. Categorical RSSM with unimix=0.01 and straight-through gradient estimation.
4. Scale-free symlog/symexp transformations for rewards and value targets.
5. Latent imagination (horizon H=15) with lambda-returns, percentile return normalization, and dynamics backprop.
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
    """Reconstructs (4, 84, 84) pixel observations from RSSM state [h, z]."""

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
        reshaped = logits.view(-1, self.k, self.c)
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


class RewardHead(nn.Module):
    """Predicts expected reward in symlog space."""

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


class Critic(nn.Module):
    """Predicts state value in symlog space."""

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
    """Outputs continuous actions in [-1, 1]^3 with Gaussian parameterization."""

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

    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(state)
        mean = self.mean_head(features)
        if deterministic:
            action = torch.tanh(mean)
            return action, mean

        std = torch.sigmoid(self.std_head(features) + 2.0) * (1.0 - 0.1) + 0.1
        dist = torch.distributions.Normal(mean, std)
        raw_sample = dist.rsample()
        action = torch.tanh(raw_sample)
        return action, raw_sample


class Uint8SequenceReplay:
    """Ring buffer storing transitions and sampling continuous sequence batches (B, L)."""

    def __init__(self, capacity: int = 100000):
        self.capacity = int(capacity)
        self.observations = np.empty((capacity, 4, 84, 84), dtype=np.uint8)
        self.actions = np.empty((capacity, 3), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.is_first = np.empty(capacity, dtype=bool)
        self.is_last = np.empty(capacity, dtype=bool)
        self.is_terminal = np.empty(capacity, dtype=bool)
        self.sequence_ids = np.empty(capacity, dtype=np.int64)

        self.cursor = 0
        self.size = 0
        self.total_steps = 0

    def add(self, transition: Transition, observation_spec: ObservationSpec | None = None) -> None:
        spec = observation_spec or ObservationSpec()
        obs = transition.observation
        if obs.dtype != np.uint8:
            obs = spec.to_uint8(obs)

        self.observations[self.cursor] = obs
        self.actions[self.cursor] = transition.action
        self.rewards[self.cursor] = transition.reward
        self.is_first[self.cursor] = transition.is_first
        self.is_last[self.cursor] = transition.is_last
        self.is_terminal[self.cursor] = transition.is_terminal
        self.sequence_ids[self.cursor] = self.total_steps

        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_steps += 1

    def sample_sequence(self, batch_size: int, seq_len: int) -> dict[str, torch.Tensor]:
        if self.size < seq_len + 1:
            raise ValueError(f"replay has {self.size} steps; need at least {seq_len + 1}")

        # Uniform rejection sampling of valid sequence starts
        start_indices = []
        max_attempts = batch_size * 50
        attempts = 0
        while len(start_indices) < batch_size and attempts < max_attempts:
            attempts += 1
            if self.size < self.capacity:
                start = np.random.randint(0, self.size - seq_len + 1)
            else:
                start = np.random.randint(0, self.capacity)

            end = (start + seq_len - 1) % self.capacity
            # Check contiguity in logical sequence_ids to avoid wrap-around boundary collisions
            if self.sequence_ids[end] - self.sequence_ids[start] == seq_len - 1:
                start_indices.append(start)

        if len(start_indices) < batch_size:
            # Fallback for small initial fills: tiled first valid index
            start_indices.extend([start_indices[0]] * (batch_size - len(start_indices)))

        batch_idx = np.empty((batch_size, seq_len), dtype=np.int64)
        for b, start in enumerate(start_indices):
            batch_idx[b] = (np.arange(start, start + seq_len)) % self.capacity

        obs = torch.from_numpy(self.observations[batch_idx]).float() / 255.0
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
    kl_dyn_weight: float = 0.8
    kl_rep_weight: float = 0.2
    kl_free_nats: float = 1.0
    model_lr: float = 1e-4
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    grad_clip_norm: float = 100.0
    batch_size: int = 16
    seq_len: int = 32
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
            in_features=self.rssm.state_dim, out_channels=4
        ).to(self.device)
        self.reward_head = RewardHead(in_features=self.rssm.state_dim).to(self.device)
        self.continue_head = ContinueHead(in_features=self.rssm.state_dim).to(self.device)

        self.actor = Actor(in_features=self.rssm.state_dim, action_dim=3).to(self.device)
        self.critic = Critic(in_features=self.rssm.state_dim).to(self.device)
        self.critic_target = Critic(in_features=self.rssm.state_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        wm_params = (
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_head.parameters())
            + list(self.continue_head.parameters())
        )
        self.wm_optimizer = torch.optim.AdamW(wm_params, lr=self.config.model_lr, eps=1e-8)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.config.actor_lr, eps=1e-8)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.config.critic_lr, eps=1e-8)

        # Replay
        self.replay = Uint8SequenceReplay(capacity=self.config.replay_capacity)

        # Online recurrent state
        self._online_h = torch.zeros(1, self.rssm.hidden_dim, device=self.device)
        self._online_z = torch.zeros(1, self.rssm.stoch_dim, device=self.device)
        self._online_prev_a = torch.zeros(1, 3, device=self.device)
        self._online_first = True

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

    def observe(self, transition: Transition) -> None:
        self.replay.add(transition, self.observation_spec)
        self.environment_steps += 1

    @torch.inference_mode()
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.from_numpy(np.ascontiguousarray(observation)).float().unsqueeze(0).to(self.device)
        if obs_t.max() > 1.0:
            obs_t = obs_t / 255.0

        e = self.encoder(obs_t)
        if self._online_first:
            self._online_h.zero_()
            self._online_z.zero_()
            self._online_prev_a.zero_()
            self._online_first = False

        if deterministic:
            h, z = self.rssm.step_post_deterministic(
                self._online_h, self._online_z, self._online_prev_a, e
            )
        else:
            h, z, _, _ = self.rssm.step_post(
                self._online_h, self._online_z, self._online_prev_a, e
            )

        self._online_h.copy_(h)
        self._online_z.copy_(z)

        s = torch.cat([self._online_h, self._online_z], dim=-1)
        action, _ = self.actor(s, deterministic=deterministic)
        self._online_prev_a.copy_(action)
        return action.squeeze(0).cpu().numpy()

    def update(self) -> dict[str, float]:
        if self.replay.size < self.config.seq_len + 1 or self.environment_steps < self.config.warmup_steps:
            return {}

        batch = self.replay.sample_sequence(self.config.batch_size, self.config.seq_len)
        obs = batch["observations"].to(self.device)  # (B, T, 4, 84, 84)
        actions = batch["actions"].to(self.device)    # (B, T, 3)
        rewards = batch["rewards"].to(self.device)    # (B, T)
        is_first = batch["is_first"].to(self.device)  # (B, T)
        is_terminal = batch["is_terminal"].to(self.device)  # (B, T)

        B, T = obs.shape[:2]

        # 1. Encode observations
        embeds = self.encoder(obs.view(B * T, 4, 84, 84)).view(B, T, -1)

        # 2. RSSM Sequence Rollout
        h = torch.zeros(B, self.rssm.hidden_dim, device=self.device)
        z = torch.zeros(B, self.rssm.stoch_dim, device=self.device)
        prev_a = torch.zeros(B, 3, device=self.device)

        h_seq, z_seq = [], []
        prior_logits_seq, post_logits_seq = [], []

        for t in range(T):
            first_mask = is_first[:, t].unsqueeze(-1).float()
            h = h * (1.0 - first_mask)
            z = z * (1.0 - first_mask)
            prev_a = prev_a * (1.0 - first_mask)

            prior_h, prior_z, pr_logits, pr_probs = self.rssm.step_prior(h, z, prev_a)
            post_h, post_z, po_logits, po_probs = self.rssm.step_post(h, z, prev_a, embeds[:, t])

            h, z = post_h, post_z
            prev_a = actions[:, t]

            h_seq.append(h)
            z_seq.append(z)
            prior_logits_seq.append(pr_logits)
            post_logits_seq.append(po_logits)

        h_stack = torch.stack(h_seq, dim=1)  # (B, T, hidden_dim)
        z_stack = torch.stack(z_seq, dim=1)  # (B, T, stoch_dim)
        states = torch.cat([h_stack, z_stack], dim=-1)  # (B, T, state_dim)

        pr_logits_stack = torch.stack(prior_logits_seq, dim=1)  # (B, T, K, C)
        po_logits_stack = torch.stack(post_logits_seq, dim=1)  # (B, T, K, C)

        # 3. World Model Losses
        # A. Observation reconstruction
        rec_obs = self.decoder(states.view(B * T, -1)).view(B, T, 4, 84, 84)
        obs_loss = 0.5 * F.mse_loss(rec_obs, obs)

        # B. Reward loss (symlog MSE)
        pred_reward = self.reward_head(states).squeeze(-1)
        target_reward = symlog(rewards)
        reward_loss = 0.5 * F.mse_loss(pred_reward, target_reward)

        # C. Continue loss
        pred_cont = self.continue_head(states).squeeze(-1)
        cont_target = (1.0 - is_terminal.float())
        continue_loss = F.binary_cross_entropy_with_logits(pred_cont, cont_target)

        # D. KL balancing loss
        _, pr_probs = self.rssm.get_dist(pr_logits_stack)
        _, po_probs = self.rssm.get_dist(po_logits_stack)
        eps = 1e-7
        po_p = po_probs.clamp(min=eps)
        pr_p = pr_probs.clamp(min=eps)
        kl = (po_p * (torch.log(po_p) - torch.log(pr_p))).sum(dim=-1).sum(dim=-1)  # (B, T)
        kl_free = torch.clamp(kl, min=self.config.kl_free_nats)

        # Balance dynamics (prior) and representation (posterior)
        kl_dyn = (po_p.detach() * (torch.log(po_p.detach()) - torch.log(pr_p))).sum(dim=-1).sum(dim=-1)
        kl_rep = (po_p * (torch.log(po_p) - torch.log(pr_p.detach()))).sum(dim=-1).sum(dim=-1)
        kl_loss = (self.config.kl_dyn_weight * kl_dyn + self.config.kl_rep_weight * kl_rep).mean()

        wm_loss = obs_loss + reward_loss + continue_loss + kl_loss

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

        # 4. Latent Imagination and Actor-Critic
        # Sample starting states from detached posterior states
        flat_states = states.detach().view(B * T, -1)
        # Select subset of states for imagination to keep budget fast
        sample_size = min(B * T, 128)
        perm = torch.randperm(B * T, device=self.device)[:sample_size]
        init_s = flat_states[perm]
        init_h, init_z = torch.split(init_s, [self.rssm.hidden_dim, self.rssm.stoch_dim], dim=-1)

        H = self.config.imagination_horizon
        im_h, im_z = init_h, init_z
        im_states = [torch.cat([im_h, im_z], dim=-1)]
        im_actions = []
        im_rewards = []
        im_continues = []

        for step in range(H):
            curr_s = torch.cat([im_h, im_z], dim=-1)
            act, _ = self.actor(curr_s, deterministic=False)
            next_h, next_z, _, _ = self.rssm.step_prior(im_h, im_z, act)
            next_s = torch.cat([next_h, next_z], dim=-1)

            r = symexp(self.reward_head(next_s).squeeze(-1))
            c = torch.sigmoid(self.continue_head(next_s).squeeze(-1))

            im_h, im_z = next_h, next_z
            im_states.append(next_s)
            im_actions.append(act)
            im_rewards.append(r)
            im_continues.append(c)

        im_states = torch.stack(im_states)  # (H+1, sample_size, state_dim)
        im_rewards = torch.stack(im_rewards)  # (H, sample_size)
        im_continues = torch.stack(im_continues)  # (H, sample_size)

        # Value targets via lambda-return
        with torch.no_grad():
            values = symexp(self.critic_target(im_states).squeeze(-1))  # (H+1, sample_size)
            targets = []
            last_v = values[-1]
            for tau in reversed(range(H)):
                gamma = self.config.gamma * im_continues[tau]
                v_target = im_rewards[tau] + gamma * (
                    (1.0 - self.config.td_lambda) * values[tau + 1]
                    + self.config.td_lambda * last_v
                )
                targets.append(v_target)
                last_v = v_target
            targets.reverse()
            targets = torch.stack(targets)  # (H, sample_size)

        # Critic update
        pred_values = self.critic(im_states[:-1].detach()).squeeze(-1)
        critic_loss = 0.5 * F.mse_loss(pred_values, symlog(targets))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_norm)
        self.critic_optimizer.step()

        # Actor update via dynamics backpropagation
        with torch.no_grad():
            flat_targets = targets.view(-1)
            p5 = torch.quantile(flat_targets, 0.05).item()
            p95 = torch.quantile(flat_targets, 0.95).item()
            self._ret_p5 = 0.9 * self._ret_p5 + 0.1 * p5
            self._ret_p95 = 0.9 * self._ret_p95 + 0.1 * p95
            scale = max(self._ret_p95 - self._ret_p5, 1.0)
            offset = self._ret_p5

        for p in self.critic.parameters():
            p.requires_grad = False

        diff_values = self.critic(im_states[1:]).squeeze(-1)
        norm_values = (diff_values - offset) / scale
        actor_loss = -norm_values.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
        self.actor_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True

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
            "loss_critic": float(critic_loss.item()),
            "loss_actor": float(actor_loss.item()),
            "reward_pred_mean": float(pred_reward.mean().item()),
            "value_pred_mean": float(pred_values.mean().item()),
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
            "format": "haic-dreamerv3-checkpoint-v1",
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
        if payload.get("format") != "haic-dreamerv3-checkpoint-v1":
            raise ValueError(f"unsupported checkpoint format: {payload.get('format')}")

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
