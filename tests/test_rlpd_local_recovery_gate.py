"""Synthetic CPU tests only: no environment construction or reset."""

import copy
import hashlib
import importlib

import numpy as np
import pytest
import torch
from torch import nn

from common_adapter import ActionAdapter
from haic.algorithms.rlpd.local_recovery_gate import (
    FORMAT, POLICY, LocalRecoveryGate, calibrate_support, tensor_state_sha)
from haic.algorithms.rlpd.model import PixelActor, PixelEncoder


torch.set_num_threads(1)


class TinyEncoder(PixelEncoder):
    def __init__(self):
        nn.Module.__init__(self)
        self.latent_dim = 2

    def forward(self, observation):
        return observation[:, 0, 0, :2]

    def convolutional_features(self, observation):
        return self(observation)

    def project(self, convolutional_features):
        return convolutional_features


def actors(*, tiny=True):
    torch.manual_seed(7)
    source = PixelActor(2)
    if tiny:
        source.encoder = TinyEncoder()
    corrected = copy.deepcopy(source)
    with torch.no_grad():
        corrected.mean.weight.zero_()
        corrected.mean.bias.copy_(torch.tensor([0.8, -0.4, 0.3]))
    return source, corrected


def observation(x=0.0, y=0.0):
    result = np.zeros((4, 84, 84), dtype=np.float32)
    result[0, 0, :2] = [x, y]
    return result


def make_gate():
    return LocalRecoveryGate(*actors(), [[0.0, 0.0]], [0.25])


def test_radius_exact_half_nearest_and_protected_exclusion():
    prototypes = torch.tensor([[0., 0.], [3., 4.]])
    protected = torch.tensor([[2., 0.], [3., 4.], [20., 10.]])
    radii, receipt = calibrate_support(prototypes, protected, chunk_size=1)
    assert torch.equal(radii, torch.tensor([1., 0.], dtype=torch.float64))
    assert receipt["disabled_zero_distance"] == 1
    assert receipt["protected_triggers"] == 0
    assert receipt["training_prototype_coverage"] == 1
    gate = LocalRecoveryGate(*actors(), prototypes, radii)
    for point in protected:
        gate.reset()
        assert not gate._select(point)


def test_boundary_inclusive_zero_disabled():
    gate = make_gate()
    assert gate._select(torch.tensor([0.25, 0.0]))
    gate.reset()
    assert not gate._select(torch.tensor([0.250001, 0.0]))
    disabled = LocalRecoveryGate(*actors(), [[0., 0.]], [0.])
    assert not disabled._select(torch.tensor([0., 0.]))


def test_hold_exactly_twelve_no_extension_and_rearm_after_hold():
    gate = make_gate()
    assert gate._select(torch.tensor([0., 0.]))
    assert gate.diagnostics == {"active": True, "triggers": 1, "prototype_id": 0,
                                "distance": 0., "remaining": 11,
                                "source_action": None, "corrected_action": None}
    for remaining in range(10, -1, -1):
        assert gate._select(torch.tensor([0., 0.]))
        assert gate.remaining == remaining
        assert gate.triggers == 1
    assert not gate._select(torch.tensor([1., 1.]))
    assert gate.prototype_id is None
    assert gate._select(torch.tensor([0., 0.]))
    assert gate.triggers == 2


def test_reset_clears_hold_and_diagnostics_without_actor_mutation():
    gate = make_gate()
    before = tensor_state_sha(gate.source_actor.state_dict())
    gate.act(observation())
    gate.reset()
    assert gate.diagnostics == {"active": False, "triggers": 0, "prototype_id": None,
                                "distance": None, "remaining": 0,
                                "source_action": None, "corrected_action": None}
    actual = gate.act(observation(1., 1.))
    native = gate.source_actor.sample(torch.from_numpy(observation(1., 1.)).unsqueeze(0),
                                      deterministic=True)[0][0].numpy()
    assert np.array_equal(actual, ActionAdapter().to_official(native))
    assert before == tensor_state_sha(gate.source_actor.state_dict())
    assert all(not p.requires_grad for p in gate.source_actor.parameters())
    assert all(not p.requires_grad for p in gate.corrected_actor.parameters())


def test_full_corrected_vector_during_hold_and_source_exact_outside():
    gate = make_gate()
    for index in range(12):
        obs = observation() if index == 0 else observation(1., 1.)
        native = gate.corrected_actor.sample(torch.from_numpy(obs).unsqueeze(0),
                                             deterministic=True)[0][0].numpy()
        actual = gate.act(obs)
        assert actual.shape == (3,) and actual.dtype == np.float32
        assert np.isfinite(actual).all()
        assert np.array_equal(actual, ActionAdapter().to_official(native))
    obs = observation(1., 1.)
    native = gate.source_actor.sample(torch.from_numpy(obs).unsqueeze(0),
                                      deterministic=True)[0][0].numpy()
    assert np.array_equal(gate.act(obs), ActionAdapter().to_official(native))
    assert not gate.active


def test_outside_support_exact_root_actor_actions_and_export_roundtrip(tmp_path):
    from agent import RLPDActor
    source, corrected = actors(tiny=False)
    root = RLPDActor(2).eval().requires_grad_(False)
    root.load_state_dict(source.state_dict(), strict=True)
    gate = LocalRecoveryGate(source, corrected, [[100., 100.]], [0.01])
    before = tensor_state_sha(source.state_dict())
    for obs in [observation(), np.random.default_rng(0).random((4, 84, 84), dtype=np.float32)]:
        with torch.inference_mode():
            official = ActionAdapter().to_official(root(torch.from_numpy(obs).unsqueeze(0))[0].numpy())
        assert np.array_equal(gate.act(obs), official)
    payload = gate.artifact(provenance={"source_sha256": "a" * 64})
    assert payload["format"] == FORMAT and payload["policy"] == POLICY
    assert set(payload["actor_state_dicts"]) == {"source", "corrected"}
    assert all(v.device.type == "cpu" for state in payload["actor_state_dicts"].values()
               for v in state.values())
    assert tensor_state_sha(payload["actor_state_dicts"]["source"]) == before
    path = tmp_path / "gate.pt"
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = LocalRecoveryGate.load(path, expected_sha256=digest)
    assert np.array_equal(loaded.act(observation()), gate.act(observation()))
    assert loaded.triggers == 0
    assert tensor_state_sha(source.state_dict()) == before
    with pytest.raises(ValueError, match="hash changed"):
        LocalRecoveryGate.load(path, expected_sha256="0" * 64)
    payload["policy"] = {**POLICY, "hold_decisions": 13}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="standalone"):
        LocalRecoveryGate.load(path)


@pytest.mark.parametrize("obs", [np.zeros((84, 84), np.float32),
    np.zeros((4, 84, 84), np.uint8), np.full((4, 84, 84), np.nan, np.float32),
    np.full((4, 84, 84), 1.1, np.float32)])
def test_canonical_observation_required(obs):
    with pytest.raises(ValueError):
        make_gate().act(obs)


def test_encoder_mismatch_fails_closed():
    source, corrected = actors(tiny=False)
    with torch.no_grad():
        corrected.encoder.projection[1].bias.add_(1.)
    with pytest.raises(ValueError, match="bitwise"):
        LocalRecoveryGate(source, corrected, [[0., 0.]], [1.])


@pytest.mark.parametrize("prototypes,radii", [([[float("nan"), 0.]], [1.]),
    ([[0., 0.]], [-1.]), ([[0., 0.]], [float("inf")]), ([[0.]], [1.])])
def test_invalid_support_schema(prototypes, radii):
    with pytest.raises(ValueError, match="schema"):
        LocalRecoveryGate(*actors(), prototypes, radii)


def test_builder_source_closure_and_no_old_geometry_validator():
    builder = importlib.import_module("scripts.build_rlpd_local_recovery_gate")
    assert builder.SOURCE_FILES == builder.actor_only.SOURCE_FILES | builder.NEW_FILES
    assert len(builder.NEW_FILES) == 3
    assert "scripts/train_rlpd_newhost.py" not in builder.SOURCE_FILES
    assert "scripts/diagnose_rlpd.py" not in builder.SOURCE_FILES
    assert builder.CORRECTED_SHA == "9198802b7cd573bf359fa1217d1edcaec3496a68e7e99d712fd1ef4135e2553f"


def test_encoding_is_source_features_on_exact_uint8_stacks():
    builder = importlib.import_module("scripts.build_rlpd_local_recovery_gate")
    source, _ = actors()
    images = np.stack([(observation(0.2, 0.4) * 255).astype(np.uint8),
                       (observation(0.6, 0.8) * 255).astype(np.uint8)])
    actual = builder.encode_images(source, images, chunk_size=1)
    assert torch.equal(actual, torch.from_numpy(images[:, 0, 0, :2].astype(np.float32) / 255.))


def test_protected_no_trigger_does_not_veto_existing_hold():
    source, corrected = actors()
    radii, _ = calibrate_support(torch.tensor([[0., 0.]]), torch.tensor([[1., 1.]]))
    gate = LocalRecoveryGate(source, corrected, [[0., 0.]], radii)
    gate.act(observation(1., 1.))
    assert not gate.active
    gate.act(observation())
    protected_during_hold = gate.act(observation(1., 1.))
    assert gate.active and gate.triggers == 1
    assert gate.corrected_action is not None
    assert np.array_equal(protected_during_hold, gate.corrected_action)


def test_synthetic_builder_export_loader_and_exclusive_output(tmp_path, monkeypatch):
    builder = importlib.import_module("scripts.build_rlpd_local_recovery_gate")
    (tmp_path / "experiments").mkdir()
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(builder.parent, "ROOT", tmp_path)
    protocol_path = tmp_path / "experiments" / "synthetic.json"
    protocol_path.write_text("{}")
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    source, corrected = actors(tiny=False)
    gate = LocalRecoveryGate(source, corrected, [[100., 100.]], [0.1])
    protocol = {"source_hashes": {}, "source_actor": {"state_sha256": tensor_state_sha(source.state_dict())},
                "corrected_actor": {}, "inputs": {}, "primary_result": {}, "limitations": []}
    monkeypatch.setattr(builder, "preflight", lambda *args, **kwargs:
                        (protocol, gate, {"protected_triggers": 0}, {"environment_resets": 0}))
    receipt = builder.build("experiments/synthetic.json", digest, "runs/synthetic-gate")
    loaded = LocalRecoveryGate.load(tmp_path / receipt["artifact_path"],
                                    expected_sha256=receipt["artifact_sha256"])
    assert receipt["environment_resets"] == 0 and receipt["critic_updates"] == 0
    assert loaded.export_metadata["provenance"]["protocol_sha256"] == digest
    assert np.array_equal(loaded.act(observation()), gate.act(observation()))
    with pytest.raises(FileExistsError):
        builder.build("experiments/synthetic.json", digest, "runs/synthetic-gate")
