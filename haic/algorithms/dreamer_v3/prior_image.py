"""A separate, free-running prior image update for the frozen Dreamer learner."""

from __future__ import annotations

import math
from numbers import Real

import torch

from dreamer_v3 import DreamerV3Agent


def prior_image_aux_update(
    agent: DreamerV3Agent, *, horizon: int, weight: float
) -> dict[str, float]:
    """Take one additional world-model step, without advancing learner counters.

    Anchors count sampled batch rows (including replay duplicates); targets count
    action-to-successor-frame predictions, not distinct replay transitions.
    """
    if type(horizon) is not int or not 1 <= horizon <= 8:
        raise ValueError("prior image horizon must be an integer in [1, 8]")
    if (isinstance(weight, bool) or not isinstance(weight, Real)
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError("prior image weight must be finite and positive")

    config = agent.config
    batch = agent.replay.sample_sequence(
        config.batch_size,
        config.seq_len,
        burnin=config.burnin_steps,
        terminal_fraction=config.terminal_window_fraction,
        reset_start_fraction=config.reset_start_fraction,
        short_episode_fraction=config.short_episode_fraction,
    )
    obs = batch["observations"].to(agent.device).detach()
    actions = batch["actions"].to(agent.device).detach()
    first = batch["is_first"].to(agent.device)
    last = batch["is_last"].to(agent.device)
    terminal = batch["is_terminal"].to(agent.device)
    sequence_ids = batch["sequence_ids"].to(agent.device)
    burnin = batch["effective_burnin"]
    learning_length = batch["effective_seq_len"]
    batch_size, transitions = actions.shape[:2]
    if (
        batch_size == 0 or learning_length < 1 or burnin < 0
        or transitions != burnin + learning_length
        or obs.shape != (batch_size, transitions + 1, 4, 84, 84)
        or actions.shape != (batch_size, transitions, 3)
        or any(flag.shape != (batch_size, transitions) for flag in (first, last, terminal, sequence_ids))
    ):
        raise ValueError("prior image requires a valid T+1 replay batch and learning window")
    if (
        (last[:, :-1] | terminal[:, :-1]).any().item()
        or first[:, 1:].any().item()
        or (sequence_ids[:, 1:] != sequence_ids[:, :-1] + 1).any().item()
    ):
        raise ValueError("prior image replay batch crosses an episode boundary")

    late_anchor = max(burnin, transitions - horizon) if learning_length >= horizon else transitions - 1
    anchors = tuple(dict.fromkeys((burnin, late_anchor)))
    with torch.no_grad():
        # The last (possibly terminal) successor is a target only, never an embed.
        embeds = agent.encoder(obs[:, :-1].reshape(batch_size * transitions, 4, 84, 84))
        embeds = embeds.reshape(batch_size, transitions, -1)
        h = embeds.new_zeros(batch_size, agent.rssm.hidden_dim)
        z = embeds.new_zeros(batch_size, agent.rssm.stoch_dim)
        prev_action = actions.new_zeros(batch_size, actions.shape[-1])
        posterior_anchors = {}
        for t in range(transitions):
            keep = (~first[:, t]).unsqueeze(-1)
            h, z, _, _ = agent.rssm.step_post(
                h * keep, z * keep, prev_action * keep, embeds[:, t]
            )
            if t in anchors:
                posterior_anchors[t] = (h.detach(), z.detach())
            prev_action = actions[:, t]

    pixel_error_sum = obs.new_zeros(())
    clipped_pixels = 0
    target_count = 0
    with torch.enable_grad():
        for anchor in anchors:
            h, z = posterior_anchors[anchor]
            stack = obs[:, anchor]
            for t in range(anchor, min(anchor + horizon, transitions)):
                h, z, _, _ = agent.rssm.step_prior(h, z, actions[:, t])
                delta = agent.decoder(torch.cat((h, z), dim=-1))
                unclamped = stack[:, -1:] + delta
                clipped_pixels += int(((unclamped < 0) | (unclamped > 1)).sum().item())
                latest = unclamped.clamp(0.0, 1.0)
                target = obs[:, t + 1, -1:]
                pixel_error_sum = pixel_error_sum + (
                    (latest - target).square().mean(dim=(1, 2, 3)).sum()
                )
                target_count += batch_size
                stack = torch.cat((stack[:, 1:], latest), dim=1)

        if not target_count:
            raise ValueError("prior image replay batch has no valid target")
        frame_mse = pixel_error_sum / target_count
        loss = weight * 0.5 * config.observation_loss_scale * frame_mse
        if not torch.isfinite(loss).item():
            raise ValueError("non-finite prior image loss")
        agent.wm_optimizer.zero_grad()
        loss.backward()
        params = [param for group in agent.wm_optimizer.param_groups for param in group["params"]]
        torch.nn.utils.clip_grad_norm_(params, config.grad_clip_norm, error_if_nonfinite=True)
        agent.wm_optimizer.step()

    return {
        "aux_loss": float(loss.item()),
        "frame_mse": float(frame_mse.item()),
        "anchor_count": float(batch_size * len(anchors)),
        "target_count": float(target_count),
        "effective_horizon": float(min(horizon, learning_length)),
        "clipped_pixel_fraction": float(clipped_pixels / (target_count * 84 * 84)),
    }
