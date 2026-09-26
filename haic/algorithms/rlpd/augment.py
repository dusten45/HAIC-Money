"""Pixel augmentation utilities for RLPD."""

from numbers import Integral

import torch
import torch.nn.functional as F


def random_shift(
    observations: torch.Tensor,
    *,
    pad: int = 4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply a per-image integer translation shared by all stacked frames."""
    if not isinstance(observations, torch.Tensor):
        raise TypeError("observations must be a torch.Tensor")
    if isinstance(pad, bool) or not isinstance(pad, Integral):
        raise TypeError("pad must be an integer")
    pad = int(pad)
    if pad < 0:
        raise ValueError("pad must be non-negative")
    if observations.ndim != 4 or tuple(observations.shape[1:]) != (4, 84, 84):
        raise ValueError("observations must have shape [B, 4, 84, 84]")
    if observations.dtype not in (torch.float32, torch.uint8):
        raise TypeError("observations must have dtype float32 or uint8")
    if observations.dtype == torch.float32:
        if (
            not bool(torch.isfinite(observations).all())
            or bool((observations < 0).any())
            or bool((observations > 1).any())
        ):
            raise ValueError("float32 observations must be finite and in [0, 1]")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None")

    batch, channels, height, width = observations.shape
    rng_device = observations.device if generator is None else generator.device
    x_offsets = torch.randint(
        0, 2 * pad + 1, (batch,), device=rng_device, generator=generator
    ).to(device=observations.device)
    y_offsets = torch.randint(
        0, 2 * pad + 1, (batch,), device=rng_device, generator=generator
    ).to(device=observations.device)

    padded = F.pad(observations, (pad, pad, pad, pad), mode="replicate")
    rows = y_offsets[:, None] + torch.arange(height, device=observations.device)
    columns = x_offsets[:, None] + torch.arange(width, device=observations.device)
    linear_indices = (
        rows[:, :, None] * padded.shape[-1] + columns[:, None, :]
    ).reshape(batch, 1, height * width)
    linear_indices = linear_indices.expand(-1, channels, -1)
    return padded.flatten(2).gather(2, linear_indices).reshape_as(observations)
