"""r7-only fixed-quota source replay and optional source-action preservation."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch.nn import functional as F

from drq_v2 import DrQv2Agent, DrQActor
from haic.algorithms.drq_v2.teacher_study import RNGStreams


class RetentionReplaySampler:
    """Draw 32 immutable source and 32 online n-step starts per minibatch."""

    def __init__(self, online_replay, source_replay, *, seed: int):
        if (online_replay.n_step, online_replay.gamma) != (source_replay.n_step, source_replay.gamma):
            raise ValueError("source and online replay returns differ")
        self.online = online_replay
        self.source = source_replay
        self.rng = np.random.default_rng(seed)
        self.source_valid = np.asarray(source_replay.valid_indices(), dtype=np.int64)
        if len(self.source_valid) < 32:
            raise ValueError("source replay has fewer than 32 valid n-step starts")
        self.online_valid: list[int] = []
        self.online_seen_next = 0
        self.online_seen_oldest = int(online_replay.oldest_sequence)

    def _refresh_online(self) -> None:
        next_sequence = int(self.online._next_sequence)
        oldest = int(self.online.oldest_sequence)
        if oldest != self.online_seen_oldest or next_sequence < self.online_seen_next:
            self.online_valid = self.online.valid_indices()
        elif next_sequence != self.online_seen_next:
            known = set(self.online_valid)
            for sequence in range(max(oldest, self.online_seen_next - self.online.n_step), next_sequence):
                if sequence not in known and self.online._build_n_step(sequence) is not None:
                    self.online_valid.append(sequence)
            self.online_valid.sort()
        self.online_seen_next = next_sequence
        self.online_seen_oldest = oldest

    def sample(self, batch_size: int = 64) -> dict[str, np.ndarray]:
        if batch_size != 64:
            raise ValueError("r7 fixes batch size at 64, exactly 32 source and 32 online")
        self._refresh_online()
        if len(self.online_valid) < 32:
            raise ValueError("online replay has fewer than 32 valid n-step starts")
        online_indices = self.rng.choice(self.online_valid, 32, replace=False).astype(np.int64)
        source_indices = self.rng.choice(self.source_valid, 32, replace=False).astype(np.int64)
        online = self.online.sample(32, indices=online_indices)
        source = self.source.sample(32, indices=source_indices)
        permutation = self.rng.permutation(64)
        fields = ("observation", "next_observation", "action", "reward", "discount",
                  "terminated", "truncated", "terminal", "horizon")
        batch = {}
        for key in fields:
            values = np.concatenate((online[key], source[key]), axis=0)
            if key in ("observation", "next_observation"):
                values = values.astype(np.float32) / np.float32(255.0)
            batch[key] = values[permutation]
        batch["source"] = np.asarray([0] * 32 + [1] * 32, dtype=np.uint8)[permutation]
        batch["source_indices"] = np.concatenate((online_indices, source_indices))[permutation]
        batch["episode_id"] = np.concatenate((online["episode_id"], source["episode_id"]))[permutation]
        return batch

    def state_dict(self) -> dict:
        return {
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "online_valid": list(self.online_valid),
            "online_seen_next": self.online_seen_next,
            "online_seen_oldest": self.online_seen_oldest,
            "source_valid_count": len(self.source_valid),
        }


def update_retained(
    agent: DrQv2Agent, batch: dict[str, np.ndarray], rng: RNGStreams,
    *, teacher: DrQActor | None, lambda_preserve: float,
) -> dict[str, float]:
    """Keep r6 critic and Q1 actor objectives; add only r7b's action MSE."""
    source = np.asarray(batch["source"])
    if source.shape != (64,) or np.count_nonzero(source == 1) != 32 or np.count_nonzero(source == 0) != 32:
        raise ValueError("r7 update requires precisely 32 source and 32 online rows")
    if not np.isfinite(lambda_preserve) or lambda_preserve < 0:
        raise ValueError("lambda_preserve must be finite and nonnegative")
    if (teacher is None) != (lambda_preserve == 0):
        raise ValueError("teacher must be present only for positive preservation weight")
    if teacher is not None and (teacher.training or any(p.requires_grad for p in teacher.parameters())):
        raise ValueError("source teacher must be eval and frozen")

    values = agent._batch_tensors(batch)
    observation = rng.shift(values["observation"], "critic_current", agent.config.augmentation_pad)
    next_observation = rng.shift(values["next_observation"], "critic_next", agent.config.augmentation_pad)
    with torch.no_grad():
        next_action = agent.actor(next_observation)
        noise = torch.randn(
            next_action.shape, dtype=next_action.dtype, device=next_action.device,
            generator=rng.target_noise,
        ) * agent.config.target_policy_noise
        noise = noise.clamp(-agent.config.target_policy_noise_clip, agent.config.target_policy_noise_clip)
        next_action = (next_action + noise).clamp(-1.0, 1.0)
        target_q = torch.minimum(
            agent.target_one(next_observation, next_action),
            agent.target_two(next_observation, next_action),
        )
        target = values["reward"] + values["discount"] * target_q
    q1 = agent.critic_one(observation, values["action"])
    q2 = agent.critic_two(observation, values["action"])
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    agent.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
        list(agent.critic_one.parameters()) + list(agent.critic_two.parameters()), 10.0
    )
    agent.critic_optimizer.step()
    agent.gradient_steps += 1

    actor_loss = torch.zeros((), device=agent.device)
    preserve_loss = torch.zeros((), device=agent.device)
    actor_q_loss = torch.zeros((), device=agent.device)
    actor_updated = agent.gradient_steps % agent.config.actor_update_frequency == 0
    actor_grad_norm = torch.zeros((), device=agent.device)
    if actor_updated:
        critic_parameters = list(agent.critic_one.parameters()) + list(agent.critic_two.parameters())
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        try:
            actor_observation = rng.shift(values["observation"], "actor_view", agent.config.augmentation_pad)
            actor_action = agent.actor(actor_observation)
            actor_q_loss = -agent.critic_one(actor_observation, actor_action).mean()
            actor_loss = actor_q_loss
            if teacher is not None:
                source_observation = values["observation"][torch.as_tensor(source == 1, device=agent.device)]
                with torch.no_grad():
                    reference = teacher(source_observation)
                preserve_loss = F.mse_loss(agent.actor(source_observation), reference)
                actor_loss = actor_q_loss + lambda_preserve * preserve_loss
            agent.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
            agent.actor_optimizer.step()
        finally:
            for parameter in critic_parameters:
                parameter.requires_grad_(True)
    if agent.gradient_steps % agent.config.target_update_frequency == 0:
        agent._soft_update_targets()
    metrics = {
        "critic_loss": float(critic_loss.detach().cpu()),
        "critic_grad_norm": float(critic_grad_norm.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
        "actor_q_loss": float(actor_q_loss.detach().cpu()),
        "preservation_loss": float(preserve_loss.detach().cpu()),
        "actor_grad_norm": float(actor_grad_norm.detach().cpu()),
        "actor_updated": float(actor_updated),
        "q1_mean": float(q1.detach().mean().cpu()),
        "q2_mean": float(q2.detach().mean().cpu()),
        "target_mean": float(target.detach().mean().cpu()),
        "sampled_source_count": 32.0,
        "sampled_online_count": 32.0,
        "gradient_steps": float(agent.gradient_steps),
    }
    if not np.isfinite(list(metrics.values())).all():
        raise FloatingPointError("non-finite r7 learner metrics")
    return metrics
