"""Pixel-only local-support wrapper, not a SAC export or an Oracle lookup.

Calibration on consumed training images is a memorization upper bound, not a
generalization guarantee. Only the frozen actors, feature support and a decision
counter participate in inference.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch

from common_adapter import ActionAdapter, ObservationSpec
from haic.algorithms.rlpd.model import PixelActor


FORMAT = "haic-rlpd-local-recovery-gate-v1"
POLICY = {"hold_decisions": 12, "radius_fraction": 0.5,
          "distance": "euclidean-source-encoder", "zero_radius_disabled": True,
          "hold_extension": False, "runtime_inputs": "pixels-and-counter-only"}


def tensor_state_sha(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def feature_distances(queries, prototypes):
    """Direct differences retain exact zeros, unlike dot-product cdist shortcuts."""
    return torch.linalg.vector_norm(
        queries.to(torch.float64)[:, None, :] - prototypes.to(torch.float64)[None, :, :], dim=-1)


def calibrate_support(prototypes, protected, *, chunk_size=512):
    prototypes = torch.as_tensor(prototypes).detach()
    protected = torch.as_tensor(protected, device=prototypes.device).detach()
    if (prototypes.ndim != 2 or protected.ndim != 2 or not len(prototypes)
            or not len(protected) or prototypes.shape[1] != protected.shape[1]
            or not torch.isfinite(prototypes).all() or not torch.isfinite(protected).all()
            or type(chunk_size) is not int or chunk_size <= 0):
        raise ValueError("nonempty finite matching feature matrices required")
    nearest = torch.full((len(prototypes),), float("inf"), dtype=torch.float64,
                         device=prototypes.device)
    for start in range(0, len(protected), chunk_size):
        distances = feature_distances(protected[start:start + chunk_size], prototypes)
        nearest = torch.minimum(nearest, distances.min(dim=0).values)
    radii = nearest * 0.5
    enabled = radii > 0
    triggered = 0
    for start in range(0, len(protected), chunk_size):
        distances = feature_distances(protected[start:start + chunk_size], prototypes)
        triggered += int(((distances <= radii) & enabled).any(dim=1).sum())
    if triggered:
        raise RuntimeError("protected feature activates calibrated support")
    coverage = ((feature_distances(prototypes, prototypes) <= radii) & enabled).any(dim=1)
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    return radii.cpu(), {"protected_points": len(protected), "protected_triggers": triggered,
        "training_prototypes": len(prototypes), "training_prototype_coverage": int(coverage.sum()),
        "enabled_prototypes": int(enabled.sum()), "disabled_zero_distance": int((~enabled).sum()),
        "quantiles": quantiles,
        "nearest_protected_distance_quantiles": torch.quantile(nearest, torch.tensor(
            quantiles, device=nearest.device, dtype=torch.float64)).cpu().tolist(),
        "radius_quantiles": torch.quantile(radii, torch.tensor(
            quantiles, device=radii.device, dtype=torch.float64)).cpu().tolist()}


class LocalRecoveryGate:
    """Own two immutable PixelActors; preserve source actions outside held support."""

    def __init__(self, source_actor, corrected_actor, prototypes, radii, *, device="cpu"):
        self.device = torch.device(device)
        self.source_actor = copy.deepcopy(source_actor).to(self.device).eval().requires_grad_(False)
        self.corrected_actor = copy.deepcopy(corrected_actor).to(self.device).eval().requires_grad_(False)
        if tensor_state_sha(self.source_actor.encoder.state_dict()) != tensor_state_sha(
                self.corrected_actor.encoder.state_dict()):
            raise ValueError("source/corrected encoders must be bitwise identical")
        self.prototypes = torch.as_tensor(prototypes, dtype=torch.float32,
                                          device=self.device).detach().clone()
        self.radii = torch.as_tensor(radii, dtype=torch.float64,
                                    device=self.device).detach().clone()
        if (self.prototypes.ndim != 2 or not len(self.prototypes)
                or self.prototypes.shape[1] != self.source_actor.encoder.latent_dim
                or self.radii.shape != (len(self.prototypes),)
                or not torch.isfinite(self.prototypes).all() or not torch.isfinite(self.radii).all()
                or (self.radii < 0).any()):
            raise ValueError("invalid prototype/radius schema")
        self.observation_spec = ObservationSpec()
        self.action_adapter = ActionAdapter()
        self.export_metadata = {}
        self.reset()

    def reset(self, observation=None):
        self.remaining = 0
        self.triggers = 0
        self.prototype_id = None
        self.distance = None
        self.active = False
        self.source_action = None
        self.corrected_action = None

    @property
    def diagnostics(self):
        return {"active": self.active, "triggers": self.triggers,
                "prototype_id": self.prototype_id, "distance": self.distance,
                "remaining": self.remaining,
                "source_action": None if self.source_action is None else self.source_action.copy(),
                "corrected_action": None if self.corrected_action is None else self.corrected_action.copy()}

    def _select(self, feature):
        distances = feature_distances(feature.reshape(1, -1), self.prototypes)[0]
        if not torch.isfinite(distances).all():
            raise ValueError("nonfinite query features")
        near = (self.radii > 0) & (distances <= self.radii)
        if self.remaining == 0:
            self.prototype_id = None
            if near.any():
                candidates = torch.where(near, distances, float("inf"))
                self.prototype_id = int(candidates.argmin())
                self.remaining = POLICY["hold_decisions"]
                self.triggers += 1
        selected = self.prototype_id if self.prototype_id is not None else int(distances.argmin())
        self.distance = float(distances[selected])
        self.active = self.remaining > 0
        if self.active:
            self.remaining -= 1
        return self.active

    @torch.inference_mode()
    def act(self, observation):
        observation = self.observation_spec.validate(observation)
        tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        feature = self.source_actor.encoder(tensor)
        corrected = self._select(feature[0])
        # Both immutable actors have the exact same encoder. Reuse its output,
        # but select the complete mean vector, never stored labels or action axes.
        source_native = self.source_actor.mean(self.source_actor.trunk(feature)).tanh()[0]
        corrected_native = self.corrected_actor.mean(self.corrected_actor.trunk(feature)).tanh()[0]
        self.source_action = self.action_adapter.to_official(source_native.cpu().numpy(), clip=False)
        self.corrected_action = self.action_adapter.to_official(corrected_native.cpu().numpy(), clip=False)
        return (self.corrected_action if corrected else self.source_action).copy()

    def artifact(self, *, provenance):
        return {"format": FORMAT, "config": {"latent_dim": self.source_actor.encoder.latent_dim},
                "policy": copy.deepcopy(POLICY),
                "actor_state_dicts": {name: {key: value.detach().cpu().clone()
                    for key, value in actor.state_dict().items()} for name, actor in
                    (("source", self.source_actor), ("corrected", self.corrected_actor))},
                "prototypes": self.prototypes.cpu().clone(), "radii": self.radii.cpu().clone(),
                "encoder_sha256": tensor_state_sha(self.source_actor.encoder.state_dict()),
                "provenance": copy.deepcopy(provenance)}

    @classmethod
    def load(cls, path, *, expected_sha256=None, device="cpu"):
        path = Path(path)
        if expected_sha256 is not None:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_sha256:
                raise ValueError("gate artifact hash changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (set(payload) != {"format", "config", "policy", "actor_state_dicts", "prototypes",
                             "radii", "encoder_sha256", "provenance"}
                or payload["format"] != FORMAT or payload["policy"] != POLICY
                or set(payload["config"]) != {"latent_dim"}
                or set(payload["actor_state_dicts"]) != {"source", "corrected"}
                or not isinstance(payload["provenance"], dict)):
            raise ValueError("not a standalone local-recovery wrapper artifact")
        actors = []
        for name in ("source", "corrected"):
            actor = PixelActor(payload["config"]["latent_dim"])
            actor.load_state_dict(payload["actor_state_dicts"][name], strict=True)
            if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):
                raise ValueError("nonfinite actor weights")
            actors.append(actor)
        if tensor_state_sha(actors[0].encoder.state_dict()) != payload["encoder_sha256"]:
            raise ValueError("encoder hash mismatch")
        result = cls(actors[0], actors[1], payload["prototypes"], payload["radii"], device=device)
        result.export_metadata = {key: payload[key] for key in
                                  ("format", "config", "policy", "encoder_sha256", "provenance")}
        return result
