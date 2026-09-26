"""Weight-only DrQ forks and study-specific update/RNG primitives."""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from common_adapter import ActionSpec, ObservationSpec
from drq_v2 import DrQv2Agent, DrQv2Config, load_exported_actor, random_shift


SOURCE_ENVIRONMENT_STEPS = 131_072


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_source_config(config: DrQv2Config) -> dict[str, Any]:
    expected = asdict(config)
    expected["observation_shape"] = list(config.observation_shape)
    return expected


def audit_source_actor_pair(
    checkpoint_path: str | Path,
    actor_path: str | Path,
    *,
    learner_seed: int,
    source_revision: str,
    expected_checkpoint_sha256: str,
    expected_actor_sha256: str,
) -> dict[str, Any]:
    """Revalidate full checkpoint/export identity and contract before data collection."""
    checkpoint_path = Path(checkpoint_path)
    actor_path = Path(actor_path)
    checkpoint_hash = file_sha256(checkpoint_path)
    actor_hash = file_sha256(actor_path)
    if checkpoint_hash != expected_checkpoint_sha256:
        raise ValueError("source full-checkpoint SHA-256 mismatch")
    if actor_hash != expected_actor_sha256:
        raise ValueError("source actor-export SHA-256 mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "haic-drq-v2-checkpoint-v1":
        raise ValueError("unsupported source full-checkpoint format")
    if int(payload.get("environment_steps", -1)) != SOURCE_ENVIRONMENT_STEPS:
        raise ValueError("source checkpoint is not the declared 131072-decision checkpoint")
    if payload.get("manifest", {}).get("algorithm") != "drq-v2":
        raise ValueError("source checkpoint manifest does not identify DrQ-v2")
    run_metadata = payload.get("manifest", {}).get("extra", {})
    if int(run_metadata.get("environment_steps", -1)) != SOURCE_ENVIRONMENT_STEPS:
        raise ValueError("source manifest step count does not match the checkpoint")
    source_hashes = payload.get("manifest", {}).get("source_hashes", {})
    if not source_hashes or not all(
        isinstance(name, str) and isinstance(value, str) and len(value) == 64
        for name, value in source_hashes.items()
    ):
        raise ValueError("source checkpoint manifest has no complete code snapshot hashes")

    config = DrQv2Config(**payload["config"])
    expected = {
        "observation_shape": (4, 84, 84),
        "action_dim": 3,
        "feature_dim": 256,
        "hidden_dim": 256,
        "replay_capacity": 100_000,
        "batch_size": 64,
        "warmup_steps": 10_000,
        "n_step": 3,
        "gamma": 0.99,
        "actor_learning_rate": 0.0001,
        "critic_learning_rate": 0.0001,
        "steering_logit_l2": 0.0,
        "tau": 0.01,
        "actor_update_frequency": 2,
        "target_update_frequency": 2,
        "augmentation_pad": 4,
        "exploration_initial_std": 0.2,
        "exploration_final_std": 0.05,
        "exploration_duration": 100_000,
        "target_policy_noise": 0.2,
        "target_policy_noise_clip": 0.5,
    }
    if any(getattr(config, name) != value for name, value in expected.items()):
        raise ValueError("source checkpoint config differs from the frozen pad-4 source contract")
    if learner_seed not in (0, 1):
        raise ValueError("source learner seed must be 0 or 1")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("source revision is absent from the protocol")

    actor_state = payload.get("actor")
    if not isinstance(actor_state, dict):
        raise ValueError("source checkpoint is missing actor weights")
    actor, action_adapter, observation_spec = load_exported_actor(actor_path, device="cpu")
    if action_adapter.spec.fingerprint != ActionSpec().fingerprint:
        raise ValueError("source actor action fingerprint differs from the study")
    if observation_spec.fingerprint != ObservationSpec().fingerprint:
        raise ValueError("source actor observation fingerprint differs from the study")
    actor_hash_from_weights = _state_dict_sha256(actor_state)
    if actor_hash_from_weights != _state_dict_sha256(actor.state_dict()):
        raise ValueError("source actor export weights differ from the full checkpoint")
    with torch.inference_mode():
        for fill in (0.0, 0.5, 1.0):
            observation = torch.full((1, 4, 84, 84), fill, dtype=torch.float32)
            action = actor(observation)
            if action.shape != (1, 3) or not torch.isfinite(action).all():
                raise ValueError("source actor failed the frozen CPU action smoke check")
    return {
        "learner_seed": int(learner_seed),
        "source_revision": source_revision,
        "checkpoint_sha256": checkpoint_hash,
        "actor_sha256": actor_hash,
        "actor_weights_sha256": actor_hash_from_weights,
        "checkpoint_source_hash_count": len(source_hashes),
        "action_fingerprint": action_adapter.spec.fingerprint,
        "observation_fingerprint": observation_spec.fingerprint,
        "cpu_smoke_observations": 3,
    }


def fork_from_source(
    checkpoint_path: str | Path,
    actor_path: str | Path,
    config: DrQv2Config,
    *,
    learner_seed: int,
    study_seed: int,
    expected_checkpoint_sha256: str,
    expected_actor_sha256: str,
) -> tuple[DrQv2Agent, dict[str, Any]]:
    """Copy all five network states but none of the source learner's mutable state."""
    checkpoint_path = Path(checkpoint_path)
    actor_path = Path(actor_path)
    actual_checkpoint_hash = file_sha256(checkpoint_path)
    actual_actor_hash = file_sha256(actor_path)
    if actual_checkpoint_hash != expected_checkpoint_sha256:
        raise ValueError("source full-checkpoint SHA-256 mismatch")
    if actual_actor_hash != expected_actor_sha256:
        raise ValueError("source actor-export SHA-256 mismatch")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "haic-drq-v2-checkpoint-v1":
        raise ValueError("unsupported source full-checkpoint format")
    saved_config = dict(payload.get("config", {}))
    expected_config = _expected_source_config(config)
    for key, expected in expected_config.items():
        saved = saved_config.get(key)
        if key == "observation_shape":
            saved = list(saved) if saved is not None else saved
        if key != "device" and saved != expected:
            raise ValueError(f"source DrQ config mismatch: {key}")
    if int(payload.get("environment_steps", -1)) != SOURCE_ENVIRONMENT_STEPS:
        raise ValueError("source checkpoint is not the declared 131072-decision checkpoint")
    if int(learner_seed) not in (0, 1):
        raise ValueError("source learner seed must be 0 or 1")

    state_names = ("actor", "critic_one", "critic_two", "target_one", "target_two")
    states = {name: payload.get(name) for name in state_names}
    if any(not isinstance(state, dict) for state in states.values()):
        raise ValueError("source full checkpoint is missing a network state")
    weight_hashes = {name: _state_dict_sha256(state) for name, state in states.items()}

    source_actor, source_action, source_observation = load_exported_actor(actor_path, device="cpu")
    if source_action.spec.fingerprint != ActionSpec().fingerprint:
        raise ValueError("source actor action fingerprint does not match the study")
    if source_observation.fingerprint != ObservationSpec().fingerprint:
        raise ValueError("source actor observation fingerprint does not match the study")
    if _state_dict_sha256(source_actor.state_dict()) != weight_hashes["actor"]:
        raise ValueError("source actor export does not match the full checkpoint actor")

    python_rng_state = random.getstate()
    cuda_devices = (
        list(range(torch.cuda.device_count()))
        if torch.device(config.device).type == "cuda" else []
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            agent = DrQv2Agent(config, seed=study_seed)
    finally:
        random.setstate(python_rng_state)
    agent.actor.load_state_dict(states["actor"], strict=True)
    agent.critic_one.load_state_dict(states["critic_one"], strict=True)
    agent.critic_two.load_state_dict(states["critic_two"], strict=True)
    agent.target_one.load_state_dict(states["target_one"], strict=True)
    agent.target_two.load_state_dict(states["target_two"], strict=True)
    agent.actor.eval()
    agent.critic_one.train()
    agent.critic_two.train()
    agent.target_one.eval()
    agent.target_two.eval()

    # This is inherited exploration time, not study progress or optimizer history.
    agent.environment_steps = SOURCE_ENVIRONMENT_STEPS
    agent.gradient_steps = 0
    if agent.actor_optimizer.state or agent.critic_optimizer.state:
        raise RuntimeError("fork optimizers must start with fresh Adam moments")
    if agent.replay.size != 0:
        raise RuntimeError("fork replay must start empty")

    cpu_fork = copy.deepcopy(agent.actor).cpu().eval()
    observations = (
        np.zeros((4, 84, 84), dtype=np.float32),
        np.full((4, 84, 84), 0.5, dtype=np.float32),
        np.ones((4, 84, 84), dtype=np.float32),
    )
    with torch.inference_mode():
        for observation in observations:
            value = torch.as_tensor(observation).unsqueeze(0)
            source_action = source_actor(value).squeeze(0).cpu().numpy()
            fork_action = cpu_fork(value).squeeze(0).cpu().numpy()
            np.testing.assert_array_equal(fork_action, source_action)

    identity = {
        "learner_seed": int(learner_seed),
        "study_seed": int(study_seed),
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": actual_checkpoint_hash,
        "source_actor_path": str(actor_path),
        "source_actor_sha256": actual_actor_hash,
        "source_environment_steps": SOURCE_ENVIRONMENT_STEPS,
        "initial_weight_sha256": weight_hashes,
        "optimizer_state_empty": True,
        "online_replay_empty": True,
        "cpu_actor_parity_observations": len(observations),
    }
    del states, payload, source_actor
    return agent, identity


class RNGStreams:
    """Independent NumPy shift seeds and Torch target-noise generator."""

    _SHIFT_NAMES = ("critic_current", "critic_next", "actor_view")

    def __init__(self, seed: int, device: torch.device | str):
        children = np.random.SeedSequence(int(seed)).spawn(4)
        self.shift_rngs = {
            name: np.random.default_rng(child)
            for name, child in zip(self._SHIFT_NAMES, children[:3])
        }
        self.target_noise = torch.Generator(device=device)
        self.target_noise.manual_seed(int(children[3].generate_state(1, dtype=np.uint64)[0]))

    def shift(self, observation: torch.Tensor, name: str, pad: int) -> torch.Tensor:
        if name not in self.shift_rngs:
            raise ValueError(f"unknown augmentation RNG stream: {name}")
        seed = int(self.shift_rngs[name].integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        return random_shift(observation, pad=pad, seed=seed)

    def state_dict(self) -> dict[str, Any]:
        return {
            "shift_rngs": {
                name: copy.deepcopy(rng.bit_generator.state)
                for name, rng in self.shift_rngs.items()
            },
            "target_noise_state": self.target_noise.get_state().cpu().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != {"shift_rngs", "target_noise_state"}:
            raise ValueError("unsupported learner RNG checkpoint fields")
        if set(state["shift_rngs"]) != set(self.shift_rngs):
            raise ValueError("augmentation RNG stream set does not match")
        for name, rng in self.shift_rngs.items():
            rng.bit_generator.state = copy.deepcopy(state["shift_rngs"][name])
        self.target_noise.set_state(state["target_noise_state"].cpu())


def update_from_mixture(
    agent: DrQv2Agent,
    batch: dict[str, np.ndarray],
    rng: RNGStreams,
) -> dict[str, float]:
    """Run one unchanged DrQ loss update on an explicitly mixed replay batch."""
    required = {
        "observation", "next_observation", "action", "reward", "discount", "source",
    }
    if not required.issubset(batch):
        raise ValueError(f"mixed batch lacks fields: {sorted(required - set(batch))}")
    source = np.asarray(batch["source"], dtype=np.int8)
    batch_size = int(agent.config.batch_size)
    if source.shape != (batch_size,) or not np.isin(source, (0, 1)).all():
        raise ValueError("batch source tags must have one online/teacher marker per row")
    teacher_count = int(np.count_nonzero(source == 1))
    expected_teacher = 16 if teacher_count else 0
    if teacher_count not in (0, expected_teacher) or batch_size - teacher_count not in (48, 64):
        raise ValueError("batch quota must be exactly online-only 0:64 or treatment 16:48")
    if teacher_count and (teacher_count != 16 or np.count_nonzero(source == 0) != 48):
        raise ValueError("teacher treatment batch must contain exactly 48 online and 16 teacher rows")
    if not teacher_count and np.count_nonzero(source == 0) != 64:
        raise ValueError("online-only batch must contain exactly 64 online rows")

    values = agent._batch_tensors(batch)
    observation = rng.shift(values["observation"], "critic_current", agent.config.augmentation_pad)
    next_observation = rng.shift(values["next_observation"], "critic_next", agent.config.augmentation_pad)
    with torch.no_grad():
        next_action = agent.actor(next_observation)
        noise = torch.randn(
            next_action.shape,
            dtype=next_action.dtype,
            device=next_action.device,
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
    td1, td2 = q1.detach() - target.detach(), q2.detach() - target.detach()
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    agent.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
        list(agent.critic_one.parameters()) + list(agent.critic_two.parameters()), 10.0
    )
    agent.critic_optimizer.step()
    agent.gradient_steps += 1

    actor_loss = torch.zeros((), device=agent.device)
    actor_updated = agent.gradient_steps % agent.config.actor_update_frequency == 0
    actor_metrics: dict[str, float] = {}
    if actor_updated:
        critic_parameters = list(agent.critic_one.parameters()) + list(agent.critic_two.parameters())
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        try:
            actor_observation = rng.shift(values["observation"], "actor_view", agent.config.augmentation_pad)
            actor_action, actor_logits = agent.actor(actor_observation, return_logits=True)
            actor_loss = -agent.critic_one(actor_observation, actor_action).mean()
            agent.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            steering_head_grad_norm = (
                agent.actor.policy.weight.grad[0].square().sum()
                + agent.actor.policy.bias.grad[0].square()
            ).sqrt()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
            agent.actor_optimizer.step()
            with torch.no_grad():
                actor_metrics = {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "actor_q_loss": float(actor_loss.detach().cpu()),
                    "actor_grad_norm": float(actor_grad_norm.detach().cpu()),
                    "actor_steering_head_grad_norm": float(steering_head_grad_norm.detach().cpu()),
                    "steering_logit_l2_penalty": 0.0,
                    "steering_logit_mean_square": float(actor_logits[:, 0].square().mean().detach().cpu()),
                    "steering_logit_abs_mean": float(actor_logits[:, 0].abs().mean().detach().cpu()),
                    "steering_logit_abs_max": float(actor_logits[:, 0].abs().max().detach().cpu()),
                    "steering_saturation_fraction": float((actor_action[:, 0].abs() >= 0.99).float().mean().cpu()),
                    "steering_abs_ge_0_46_fraction": float((actor_action[:, 0].abs() >= 0.46).float().mean().cpu()),
                    "steering_tanh_derivative_mean": float((1.0 - actor_action[:, 0].square()).mean().cpu()),
                    "steering_tanh_zero_fraction": float(((1.0 - actor_action[:, 0].square()) == 0.0).float().mean().cpu()),
                }
        finally:
            for parameter in critic_parameters:
                parameter.requires_grad_(True)

    if agent.gradient_steps % agent.config.target_update_frequency == 0:
        agent._soft_update_targets()

    online = torch.as_tensor(source == 0, device=agent.device)
    teacher = torch.as_tensor(source == 1, device=agent.device)
    with torch.inference_mode():
        teacher_actions = agent.actor(values["observation"])[teacher]
        teacher_reference = values["action"][teacher]
        output = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "critic_grad_norm": float(critic_grad_norm.detach().cpu()),
            "q1_mean": float(q1.detach().mean().cpu()),
            "q2_mean": float(q2.detach().mean().cpu()),
            "q1_q2_abs_gap_mean": float((q1.detach() - q2.detach()).abs().mean().cpu()),
            "target_mean": float(target.detach().mean().cpu()),
            "return_mean": float(values["reward"].mean().detach().cpu()),
            "td_error_online": float((td1[online].abs().mean() + td2[online].abs().mean()).cpu() / 2.0),
            "sampled_online_count": float(online.sum().cpu()),
            "sampled_teacher_count": float(teacher.sum().cpu()),
            "sampled_teacher_fraction": float(teacher.float().mean().cpu()),
            "actor_updated": float(actor_updated),
            "gradient_steps": float(agent.gradient_steps),
            "replay_size": float(agent.replay.size),
            **actor_metrics,
        }
        if teacher_count:
            td_teacher = td1[teacher].abs().mean() + td2[teacher].abs().mean()
            actor_output = {
                "td_error_teacher": float(td_teacher.cpu() / 2.0),
                "teacher_state_actor_abs_delta_mean": float((teacher_actions - teacher_reference).abs().mean().cpu()),
                "teacher_state_actor_saturation_fraction": float((teacher_actions.abs() >= 0.99).float().mean().cpu()),
            }
            output.update(actor_output)
        finite_metrics = [value for value in output.values() if isinstance(value, (float, int))]
        if not np.isfinite(finite_metrics).all():
            raise FloatingPointError(f"non-finite teacher-replay update metrics: {output}")
    return output


class GeometryPoolRNG:
    """Reset-seam RNG that cycles through a frozen geometry pool without replacement."""

    def __init__(self, track_ids: list[int], geometry_seeds: list[int], seed: int):
        if track_ids != [1, 2, 3, 4]:
            raise ValueError("teacher study sampler must use track IDs 1-4")
        if not geometry_seeds or len(geometry_seeds) != len(set(geometry_seeds)):
            raise ValueError("geometry pool must be non-empty and duplicate-free")
        self.track_ids = tuple(track_ids)
        self.geometry_seeds = tuple(int(value) for value in geometry_seeds)
        self.rng = np.random.default_rng(seed)
        self._sequence: np.ndarray = np.empty(0, dtype=np.uint32)
        self._position = 0
        self._cycle = 0

    def choice(self, values: Any) -> Any:
        return self.rng.choice(values)

    def integers(self, low: int, high: int, *, dtype: Any = np.int64) -> int:
        if int(low) != 0 or int(high) != 2**32:
            raise ValueError("controlled geometry sampler received an unexpected integer range")
        if self._position >= len(self._sequence):
            self._sequence = self.rng.permutation(self.geometry_seeds).astype(np.uint32)
            self._position = 0
            self._cycle += 1
        value = int(self._sequence[self._position])
        self._position += 1
        return int(np.asarray(value, dtype=dtype))

    def state_dict(self) -> dict[str, Any]:
        return {
            "track_ids": list(self.track_ids),
            "geometry_seeds": list(self.geometry_seeds),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "sequence": self._sequence.copy(),
            "position": self._position,
            "cycle": self._cycle,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != {"track_ids", "geometry_seeds", "rng_state", "sequence", "position", "cycle"}:
            raise ValueError("unsupported geometry-pool RNG checkpoint fields")
        if state["track_ids"] != list(self.track_ids) or state["geometry_seeds"] != list(self.geometry_seeds):
            raise ValueError("geometry-pool RNG source does not match this run")
        sequence = np.asarray(state["sequence"], dtype=np.uint32)
        if sequence.size and set(map(int, sequence)) != set(self.geometry_seeds):
            raise ValueError("checkpoint geometry cycle does not match the frozen seed pool")
        position = int(state["position"])
        if not 0 <= position <= sequence.size:
            raise ValueError("checkpoint geometry cycle position is invalid")
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        self._sequence = sequence.copy()
        self._position = position
        self._cycle = int(state["cycle"])


def find_sampled_track_env(env: Any) -> Any:
    """Find the training-only sampled-road wrapper without using `.unwrapped`."""
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if all(hasattr(current, name) for name in ("_track_ids", "_rng", "_excluded_seeds")):
            return current
        child = getattr(current, "env", None)
        if child is None or child is current:
            break
        current = child
    raise RuntimeError("sampled environment wrapper chain has no audited SampledHaicTrack")


def learner_rng_state(agent: DrQv2Agent) -> dict[str, Any]:
    """Capture global RNG state owned by model initialization and optimizer runtime."""
    return {
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": (
            torch.cuda.get_rng_state(agent.device).cpu().clone()
            if agent.device.type == "cuda" else None
        ),
        "python": random.getstate(),
    }


def restore_learner_rng_state(agent: DrQv2Agent, state: dict[str, Any]) -> None:
    if set(state) != {"torch_cpu", "torch_cuda", "python"}:
        raise ValueError("unsupported global learner RNG checkpoint fields")
    torch.set_rng_state(state["torch_cpu"].cpu())
    if agent.device.type == "cuda":
        if state["torch_cuda"] is None:
            raise ValueError("CUDA learner checkpoint has no CUDA RNG state")
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device=agent.device)
    elif state["torch_cuda"] is not None:
        raise ValueError("CPU learner checkpoint unexpectedly contains CUDA RNG state")
    random.setstate(state["python"])
