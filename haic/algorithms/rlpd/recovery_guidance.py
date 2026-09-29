"""Joint failure-Oracle guidance plus original-mean prior/handoff retention."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from haic.algorithms.rlpd.agent import PixelRLPDAgent
from haic.algorithms.rlpd.model import PixelActor
from haic.algorithms.rlpd.recovery import FINISH_POLICY, file_sha256
from scripts.rlpd_common import canonical_sha256


@dataclass(frozen=True)
class GuidanceConfig:
    guide_weight: float = 1.0
    retention_weight: float = 0.1
    gradient_clip: float = 10.0

    def __post_init__(self):
        for value in asdict(self).values():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("guidance weights and clip must be finite and positive")


def load_role_lookup(dataset: Path, manifest_sha256: str):
    """Bind accepted (episode, step) roles to hashed traces and actual actions."""
    dataset = Path(dataset)
    manifest_path = dataset / "manifest.json"
    if manifest_path.is_symlink() or file_sha256(manifest_path) != manifest_sha256:
        raise ValueError("guidance manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("eligibility_policy") != FINISH_POLICY:
        raise ValueError("guidance requires paired-finish windows")
    lookup = {}
    trace_rows = []
    for episode in manifest["episodes"]:
        if not episode.get("paired_finish_qualified"):
            if episode.get("accepted_transitions") != 0:
                raise ValueError("unqualified guidance window")
            continue
        relative = Path(episode["path"])
        if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                or any(dataset.joinpath(*relative.parts[:i]).is_symlink()
                       for i in range(1, len(relative.parts) + 1))):
            raise ValueError("unsafe guidance trace path")
        trace = dataset / relative
        if file_sha256(trace) != episode["sha256"]:
            raise ValueError("guidance trace changed")
        start = episode["training_window_start"]
        end = episode["training_window_end"]
        handoff = episode["anchor_step"] + episode["horizon"]
        if (episode["stratum"] not in ("failure", "finish-control")
                or episode.get("censored") is not False
                or episode.get("finished") is not True
                or episode.get("local_recovery_qualified") is not True
                or start != episode["anchor_step"] or episode["horizon"] not in (12, 25)
                or not 0 <= start < handoff < end <= episode["steps"]
                or end != handoff + 63 or end - start != episode["accepted_transitions"]):
            raise ValueError("invalid guidance window")
        with np.load(trace, allow_pickle=False) as data:
            expected_mask = np.zeros(episode["steps"], dtype=bool)
            expected_mask[start:end] = True
            if (not np.array_equal(data["recovery_mask"], expected_mask)
                    or not (data["role"][start:handoff] == "oracle").all()
                    or not (data["role"][handoff:end] == "actor").all()):
                raise ValueError("guidance roles disagree with executed window")
            for step in range(start, end):
                key = (episode["episode_id"], step)
                if key in lookup:
                    raise ValueError("duplicate guidance row identity")
                action = np.asarray(data["executed_action"][step], dtype=np.float32)
                if action.shape != (3,) or not np.isfinite(action).all() or (abs(action) > 1).any():
                    raise ValueError("invalid executed guidance action")
                row = {"episode_id": key[0], "step": step,
                       "stratum": episode["stratum"],
                       "role": "oracle" if step < handoff else "actor",
                       "trace_sha256": episode["sha256"],
                       "action": action.tolist()}
                lookup[key] = row
                trace_rows.append(row)
    counts = {f"{stratum}/{role}": sum(row["stratum"] == stratum and row["role"] == role
                                       for row in trace_rows)
              for stratum in ("failure", "finish-control") for role in ("oracle", "actor")}
    receipt = {"manifest_sha256": manifest_sha256,
               "role_lookup_trace_sha256": canonical_sha256(trace_rows),
               "accepted_rows": len(lookup), "counts": counts,
               "target": "executed_action", "role_rule": "episode-step-window"}
    return lookup, receipt


class RecoveryGuidedRLPDAgent(PixelRLPDAgent):
    def __init__(self, *args, reference_state: dict, role_lookup: dict,
                 guidance_receipt: dict, guidance_config: GuidanceConfig | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.guidance_config = guidance_config or GuidanceConfig()
        self.role_lookup = copy.deepcopy(role_lookup)
        self.guidance_receipt = copy.deepcopy(guidance_receipt)
        self.reference_actor = PixelActor().to(self.device)
        self.reference_actor.load_state_dict(reference_state, strict=True)
        self.reference_actor.eval().requires_grad_(False)
        self.extra_actor_steps = 0

    def guidance_specification(self):
        return {"variant": "joint-native-recovery-guided-v1",
                **asdict(self.guidance_config), "guidance_source": "recovery/failure/oracle",
                "retention_source": ["ordinary-prior", "recovery/actor-handoff",
                                     "recovery/finish-control/oracle"],
                "retention_target": "original-V5-source-actor-mean-on-same-images",
                "guidance_retention_masks_disjoint": True,
                "q_filter": False, "encoder_detached": True,
                "raw_sac_unchanged": True, "matched_compute_to_v1": False,
                "extra_actor_optimizer_steps": self.extra_actor_steps,
                "source_receipt": copy.deepcopy(self.guidance_receipt)}

    def masks(self, batch):
        source = np.asarray(batch["source"])
        guide = np.zeros(len(source), dtype=bool)
        retention = source == "offline"
        roles = {f"{stratum}/{role}": 0 for stratum in ("failure", "finish-control")
                 for role in ("oracle", "actor")}
        if int(retention.sum()) < 16:
            raise ValueError("guidance requires at least sixteen ordinary prior rows")
        for index in np.flatnonzero(source == "recovery"):
            key = (int(batch["episode_id"][index]), int(batch["step"][index]))
            row = self.role_lookup.get(key)
            if row is None or not np.array_equal(np.asarray(row["action"], dtype=np.float32),
                                                 batch["action"][index]):
                raise ValueError("sampled recovery row/action lacks frozen role provenance")
            roles[f"{row['stratum']}/{row['role']}"] += 1
            guide[index] = row["stratum"] == "failure" and row["role"] == "oracle"
            retention[index] = row["role"] == "actor" or row["stratum"] == "finish-control"
        return guide, retention, roles

    def guidance_update(self, batch):
        guide, retention, roles = self.masks(batch)
        observation = torch.as_tensor(batch["observation"], device=self.device)
        executed = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        # Detach the complete encoder, including SAC's trainable projection.
        with torch.no_grad():
            features = self.actor.encoder(observation).detach()
            reference = self.reference_actor.sample(observation, deterministic=True)[0]
        mean = self.actor.mean(self.actor.trunk(features)).tanh()
        guide_mask = torch.as_tensor(guide, device=self.device)
        retention_mask = torch.as_tensor(retention, device=self.device)
        guide_components = (mean[guide_mask] - executed[guide_mask]).square().mean(0) \
            if guide.any() else mean.new_zeros(3)
        retention_components = (mean[retention_mask] - reference[retention_mask]).square().mean(0)
        guide_loss = guide_components.mean()
        retention_loss = retention_components.mean()
        loss = (self.guidance_config.guide_weight * guide_loss
                + self.guidance_config.retention_weight * retention_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite guidance loss")
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [parameter for group in self.actor_optimizer.param_groups
                      for parameter in group["params"]]
        norm = self._finite_grad_norm(parameters)
        clip_grad_norm_(parameters, self.guidance_config.gradient_clip)
        self.actor_optimizer.step()
        self.extra_actor_steps += 1
        return {"guidance_loss": float(guide_loss.detach()),
                "retention_loss": float(retention_loss.detach()),
                "aux_actor_loss": float(loss.detach()), "aux_actor_grad_norm": norm,
                "eligible_oracle_failure_rows": int(guide.sum()),
                "retention_rows": int(retention.sum()), "ordinary_prior_rows": int((np.asarray(batch['source']) == 'offline').sum()),
                "recovery_role_counts": roles, "extra_actor_optimizer_steps": self.extra_actor_steps,
                **{f"{prefix}_{name}_loss": float(components[i].detach())
                   for prefix, components in (("guidance", guide_components), ("retention", retention_components))
                   for i, name in enumerate(("steer", "gas", "brake"))}}

    def update(self, batch):
        self.masks(batch)  # Fail closed before any SAC mutation on unknown roles.
        metrics = super().update(batch)
        return {**metrics, **self.guidance_update(batch), "sac_gradient_steps": self.gradient_steps}

    def export_actor(self, path, **kwargs):
        kwargs["environment_contract"] = {**kwargs["environment_contract"],
                                           "guidance": self.guidance_specification()}
        return super().export_actor(path, **kwargs)

    def checkpoint_state(self, **kwargs):
        state = super().checkpoint_state(**kwargs)
        state["guidance"] = self.guidance_specification()
        return state
