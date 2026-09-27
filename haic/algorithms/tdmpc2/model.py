"""Single-task TD-MPC2 pixel world model (official 5M architecture).

Source: nicklashansen/tdmpc2, commit e9f59321933cbc8e11a002b842adc7d4ffae8ff1,
tdmpc2/common/{world_model,layers,math,init}.py and tdmpc2/config.yaml.
Upstream RGB observations stack three 3-channel, 64x64 uint8 frames. HAIC's
four grayscale frames are explicitly adapted outside this model to 4x64x64.
The environment/replay, optimizer, and planner are outside this module.
"""

from copy import deepcopy
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TDMPC2ModelConfig:
    """Official single-task 5M pixel defaults; action_dim is environment-specific."""

    action_dim: int = 3
    obs: str = "rgb"
    obs_shape: dict = field(default_factory=lambda: {"rgb": (9, 64, 64)})
    multitask: bool = False
    task_dim: int = 0
    episodic: bool = False
    num_channels: int = 32
    latent_dim: int = 512
    mlp_dim: int = 512
    num_q: int = 5
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0
    simnorm_dim: int = 8
    dropout: float = 0.01
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    tau: float = 0.01

    @property
    def bin_size(self):
        return (self.vmax - self.vmin) / (self.num_bins - 1) if self.num_bins > 1 else 0.0


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


def two_hot(x, cfg):
    """Interpolate a raw scalar between adjacent bins in symlog space."""
    if cfg.num_bins == 0:
        return x
    if cfg.num_bins == 1:
        return symlog(x)
    if x.shape[-1] != 1:
        raise ValueError("two_hot expects a trailing scalar dimension")
    bin_size = getattr(cfg, "bin_size", (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1))
    position = ((symlog(x).clamp(cfg.vmin, cfg.vmax) - cfg.vmin) / bin_size).clamp(0, cfg.num_bins - 1)
    lower = position.floor().long().clamp(max=cfg.num_bins - 2)
    upper_weight = position - lower
    target = torch.zeros(*x.shape[:-1], cfg.num_bins, device=x.device, dtype=x.dtype)
    target.scatter_(-1, lower, 1 - upper_weight)
    target.scatter_(-1, lower + 1, upper_weight)
    return target


def two_hot_inv(logits, cfg):
    """Decode a predicted categorical distribution back to raw reward/value."""
    if cfg.num_bins == 0:
        return logits
    if cfg.num_bins == 1:
        return symexp(logits)
    bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=logits.device, dtype=logits.dtype)
    return symexp((F.softmax(logits, dim=-1) * bins).sum(-1, keepdim=True))


def soft_ce(logits, target, cfg):
    """Per-sample soft-target cross-entropy, with unreduced trailing dimension."""
    if cfg.num_bins < 2:
        raise ValueError("categorical cross-entropy requires at least two bins")
    return -(two_hot(target, cfg) * F.log_softmax(logits, dim=-1)).sum(-1, keepdim=True)


class SimNorm(nn.Module):
    """Softmax each consecutive group of simnorm_dim latent coordinates."""

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim

    def forward(self, x):
        shape = x.shape
        return F.softmax(x.reshape(*shape[:-1], -1, self.dim), dim=-1).reshape(shape)


class ShiftAug(nn.Module):
    """Upstream pad-3 random shift, including its independent per-frame draws."""

    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, _, h, w = x.shape
        if h != w:
            raise ValueError("pixel augmentation requires square images")
        x = F.pad(x, (self.pad,) * 4, mode="replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1 + eps, 1 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(0, 1)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, (n, 1, 1, 2), device=x.device, dtype=x.dtype)
        grid = base_grid + shift * (2.0 / (h + 2 * self.pad))
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class PixelPreprocess(nn.Module):
    def forward(self, x):
        return x.div(255.0).sub(0.5)


class NormedLinear(nn.Linear):
    def __init__(self, in_features, out_features, *, dropout=0.0, act=None):
        super().__init__(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.act = nn.Mish(inplace=False) if act is None else act
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.act(self.ln(x))


def _mlp(in_dim, hidden_dim, out_dim, *, act=None, dropout=0.0):
    return nn.Sequential(
        NormedLinear(in_dim, hidden_dim, dropout=dropout),
        NormedLinear(hidden_dim, hidden_dim),
        NormedLinear(hidden_dim, out_dim, act=act) if act is not None else nn.Linear(hidden_dim, out_dim),
    )


def _weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class WorldModel(nn.Module):
    """Official TD-MPC2 implicit model specialized to single-task pixel control.

    Accepts TDMPC2ModelConfig or a namespace with the same official attributes.
    Observations are BCHW or TBCHW uint8 in [0,255]. No frame stack is formed
    inside the model: the caller supplies the full channel stack declared by cfg.
    """

    def __init__(self, cfg):
        super().__init__()
        if getattr(cfg, "multitask", False) or getattr(cfg, "task_dim", 0) != 0:
            raise ValueError("this model supports single-task control only")
        if getattr(cfg, "obs", "rgb") != "rgb":
            raise ValueError("this model requires rgb observations")
        shape = cfg.obs_shape["rgb"]
        if len(shape) != 3 or tuple(shape[-2:]) != (64, 64):
            raise ValueError("TD-MPC2 pixel encoder requires CHW 64x64 observations")
        if cfg.latent_dim != 16 * cfg.num_channels:
            raise ValueError("pixel encoder output must equal latent_dim (16 * num_channels)")
        if cfg.latent_dim % cfg.simnorm_dim or cfg.num_q < 2 or cfg.action_dim < 1:
            raise ValueError("invalid simnorm_dim, num_q, or action_dim")
        self.cfg = cfg
        channels = cfg.num_channels
        self._encoder = nn.ModuleDict({"rgb": nn.Sequential(
            ShiftAug(), PixelPreprocess(),
            nn.Conv2d(shape[0], channels, 7, stride=2), nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 5, stride=2), nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, stride=2), nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, stride=1), nn.Flatten(), SimNorm(cfg),
        )})
        in_dim = cfg.latent_dim + cfg.action_dim
        self._dynamics = _mlp(in_dim, cfg.mlp_dim, cfg.latent_dim, act=SimNorm(cfg))
        self._reward = _mlp(in_dim, cfg.mlp_dim, max(cfg.num_bins, 1))
        self._termination = _mlp(cfg.latent_dim, cfg.mlp_dim, 1) if cfg.episodic else None
        self._pi = _mlp(cfg.latent_dim, cfg.mlp_dim, 2 * cfg.action_dim)
        self._Qs = nn.ModuleList([
            _mlp(in_dim, cfg.mlp_dim, max(cfg.num_bins, 1), dropout=cfg.dropout)
            for _ in range(cfg.num_q)
        ])
        self.apply(_weight_init)
        with torch.no_grad():
            self._reward[-1].weight.zero_()
            for q in self._Qs:
                q[-1].weight.zero_()
        self._target_Qs = deepcopy(self._Qs)
        self._target_Qs.requires_grad_(False)
        self._target_Qs.eval()
        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max - cfg.log_std_min))

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train(self, mode=True):
        super().train(mode)
        self._target_Qs.eval()
        return self

    @torch.no_grad()
    def soft_update_target_Q(self):
        for online, target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
            target.lerp_(online, self.cfg.tau)

    def encode(self, obs, task=None):
        if task is not None:
            raise ValueError("single-task model does not accept task IDs")
        if obs.ndim == 5:
            return torch.stack([self._encoder["rgb"](frame) for frame in obs])
        return self._encoder["rgb"](obs)

    def next(self, z, a, task=None):
        if task is not None:
            raise ValueError("single-task model does not accept task IDs")
        return self._dynamics(torch.cat([z, a], dim=-1))

    def reward(self, z, a, task=None):
        if task is not None:
            raise ValueError("single-task model does not accept task IDs")
        return self._reward(torch.cat([z, a], dim=-1))

    def termination(self, z, task=None, unnormalized=False):
        if task is not None:
            raise ValueError("single-task model does not accept task IDs")
        if self._termination is None:
            raise RuntimeError("termination head requires episodic=True")
        logits = self._termination(z)
        return logits if unnormalized else torch.sigmoid(logits)

    def pi(self, z, task=None):
        if task is not None:
            raise ValueError("single-task model does not accept task IDs")
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = self.log_std_min + 0.5 * self.log_std_dif * (torch.tanh(log_std) + 1)
        eps = torch.randn_like(mean)
        raw_log_prob = (-0.5 * eps.square() - log_std - 0.9189385175704956).sum(-1, keepdim=True)
        action = torch.tanh(mean + eps * log_std.exp())
        mean = torch.tanh(mean)
        log_prob = raw_log_prob - torch.log(F.relu(1 - action.square()) + 1e-6).sum(-1, keepdim=True)
        entropy_scale = (raw_log_prob * self.cfg.action_dim) / (log_prob + 1e-8)
        return action, {
            "mean": mean,
            "log_std": log_std,
            "action_prob": 1.0,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        }

    def Q(self, z, a, task=None, return_type="min", target=False, detach=False):
        if task is not None:
            raise ValueError("single-task model does not accept task IDs")
        if return_type not in {"min", "avg", "all"}:
            raise ValueError("return_type must be min, avg, or all")
        if target and detach:
            raise ValueError("target and detach are mutually exclusive")
        x = torch.cat([z, a], dim=-1)
        critics = self._target_Qs if target else self._Qs
        if detach and not target:
            from torch.func import functional_call

            out = torch.stack([
                functional_call(q, {k: p.detach() for k, p in q.named_parameters()}, (x,))
                for q in critics
            ])
        else:
            out = torch.stack([q(x) for q in critics])
        if return_type == "all":
            return out
        pair = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        values = two_hot_inv(out[pair], self.cfg)
        return values.min(0).values if return_type == "min" else values.mean(0)
