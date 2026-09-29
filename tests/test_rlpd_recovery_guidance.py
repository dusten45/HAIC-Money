"""CPU-only guidance gradients and child protocol/source closure tests."""

import copy
import json
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from haic.algorithms.rlpd.agent import PixelRLPDAgent, RLPDConfig
from haic.algorithms.rlpd.recovery import FINISH_POLICY, file_sha256
from haic.algorithms.rlpd.recovery_guidance import (
    GuidanceConfig, RecoveryGuidedRLPDAgent, load_role_lookup,
)
from scripts import train_rlpd_recovery_guided as trainer


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.trunk = nn.Linear(2, 2)
        self.mean = nn.Linear(2, 3)
        self.log_std = nn.Linear(2, 3)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)

    def sample(self, observation, *, deterministic):
        return self.mean(self.trunk(self.encoder(observation))).tanh(), None, None


def tiny_agent():
    agent: Any = object.__new__(RecoveryGuidedRLPDAgent)
    agent.actor = TinyActor()
    agent.reference_actor = copy.deepcopy(agent.actor).requires_grad_(False)
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=0.01)
    agent.device = torch.device("cpu")
    agent.guidance_config = GuidanceConfig()
    agent.guidance_receipt = {"role_lookup_trace_sha256": "a" * 64}
    agent.role_lookup = {}
    agent.extra_actor_steps = 0
    agent.gradient_steps = 0
    return agent


def batch_for(agent, *, eligible=True):
    source = np.array(["offline"] * 16 + ["online"] * 2 + ["recovery"] * 4)
    batch = {"source": source, "observation": np.ones((22, 2), dtype=np.float32),
             "action": np.zeros((22, 3), dtype=np.float32),
             "proposed_action": np.full((22, 3), -0.9, dtype=np.float32),
             "episode_id": np.arange(22), "step": np.zeros(22, dtype=int)}
    for index, (stratum, role) in enumerate((("failure", "oracle" if eligible else "actor"),
                                           ("failure", "actor"),
                                           ("finish-control", "oracle"),
                                           ("finish-control", "actor")), 18):
        batch["action"][index] = [0.6, -0.8, 0.4] if index == 18 else [-0.9, 0.9, -0.9]
        agent.role_lookup[(index, 0)] = {"stratum": stratum, "role": role,
                                       "action": batch["action"][index].tolist()}
    return batch


def test_only_failure_oracle_guides_all_three_native_coordinates():
    agent = tiny_agent()
    batch = batch_for(agent)
    metrics = agent.guidance_update(batch)
    assert metrics["eligible_oracle_failure_rows"] == 1
    assert metrics["retention_rows"] == 19
    guide, retain, _ = agent.masks(batch)
    assert not (guide & retain).any()
    assert not guide[19] and retain[19]  # Failure actor handoff retains the source mean.
    assert guide.tolist() == [False] * 18 + [True, False, False, False]
    assert retain.tolist() == [True] * 16 + [False] * 3 + [True] * 3
    assert metrics["retention_loss"] == 0
    assert metrics["guidance_steer_loss"] == pytest.approx(0.36)
    assert metrics["guidance_gas_loss"] == pytest.approx(0.64)
    assert metrics["guidance_brake_loss"] == pytest.approx(0.16)
    assert torch.all(agent.actor.mean.bias.detach() * torch.tensor([1., -1., 1.]) > 0)
    assert metrics["recovery_role_counts"] == {"failure/oracle": 1, "failure/actor": 1,
                                                "finish-control/oracle": 1, "finish-control/actor": 1}


def test_encoder_reference_and_logstd_unchanged_by_auxiliary_step():
    agent = tiny_agent()
    batch = batch_for(agent)
    encoder = copy.deepcopy(agent.actor.encoder.state_dict())
    reference = copy.deepcopy(agent.reference_actor.state_dict())
    logstd = copy.deepcopy(agent.actor.log_std.state_dict())
    agent.guidance_update(batch)
    for before, after in ((encoder, agent.actor.encoder.state_dict()),
                          (reference, agent.reference_actor.state_dict()),
                          (logstd, agent.actor.log_std.state_dict())):
        assert all(torch.equal(before[name], after[name]) for name in before)
    assert all(parameter.grad is None for parameter in agent.actor.encoder.parameters())


def test_no_eligible_rows_finite_zero_guidance_retention_still_steps():
    agent = tiny_agent()
    metrics = agent.guidance_update(batch_for(agent, eligible=False))
    assert metrics["guidance_loss"] == 0
    assert metrics["retention_loss"] == 0
    assert metrics["extra_actor_optimizer_steps"] == 1
    assert metrics["eligible_oracle_failure_rows"] == 0


def test_retention_restores_original_not_executed_handoff_or_control_actions():
    agent = tiny_agent()
    batch = batch_for(agent, eligible=False)
    with torch.no_grad():
        agent.actor.mean.bias.fill_(0.2)
    metrics = agent.guidance_update(batch)
    assert metrics["retention_loss"] == pytest.approx(float(torch.tanh(torch.tensor(.2)) ** 2))
    assert (agent.actor.mean.bias.detach() < .2).all()


def test_failure_handoff_changes_retention_not_guidance_target():
    agent = tiny_agent()
    batch = batch_for(agent)
    batch["observation"][19] = [2., 3.]
    with torch.no_grad():
        agent.actor.encoder.weight.copy_(torch.eye(2))
        agent.actor.encoder.bias.zero_()
        agent.actor.trunk.weight.copy_(torch.eye(2))
        agent.actor.trunk.bias.zero_()
        agent.actor.mean.weight.fill_(.1)
    with torch.no_grad():
        current = agent.actor.sample(torch.from_numpy(batch["observation"]), deterministic=True)[0]
        reference = agent.reference_actor.sample(torch.from_numpy(batch["observation"]), deterministic=True)[0]
    guide, retain, _ = agent.masks(batch)
    expected_retention = (current[retain] - reference[retain]).square().mean().item()
    oracle_label_loss = (current[retain] - torch.from_numpy(batch["action"])[retain]).square().mean().item()
    assert expected_retention != pytest.approx(oracle_label_loss)
    assert retain[19] and not guide[19]
    metrics = agent.guidance_update(batch)
    assert metrics["retention_loss"] == pytest.approx(expected_retention)
    assert metrics["guidance_loss"] == pytest.approx(
        (current[guide] - torch.from_numpy(batch["action"])[guide]).square().mean().item())


@pytest.mark.parametrize("corruption", ["missing", "action", "prior"])
def test_role_and_prior_validation_fail_before_base_update(corruption):
    agent = tiny_agent()
    batch = batch_for(agent)
    if corruption == "missing":
        del agent.role_lookup[(18, 0)]
    elif corruption == "action":
        batch["action"][18, 0] = -0.5
    else:
        batch["source"][0] = "online"
    with patch.object(PixelRLPDAgent, "update") as base:
        with pytest.raises(ValueError):
            agent.update(batch)
        base.assert_not_called()


def test_base_sac_called_once_then_separate_aux_count():
    agent = tiny_agent()
    batch = batch_for(agent)
    def base(self, passed):
        assert passed is batch
        assert self.extra_actor_steps == 0
        self.gradient_steps += 1
        return {"gradient_steps": self.gradient_steps, "critic_loss": 7.0}
    with patch.object(PixelRLPDAgent, "update", base):
        metrics = agent.update(batch)
    assert metrics["critic_loss"] == 7
    assert metrics["sac_gradient_steps"] == metrics["gradient_steps"] == 1
    assert metrics["extra_actor_optimizer_steps"] == 1


@pytest.mark.parametrize("field,value", [("guide_weight", float("nan")),
                                         ("retention_weight", 0), ("gradient_clip", -1)])
def test_invalid_guidance_config(field, value):
    with pytest.raises(ValueError):
        GuidanceConfig(**{field: value})


def write_dataset(tmp_path, *, wrong_role=False):
    n = 76
    roles = np.array(["oracle"] * 12 + ["actor"] * 64)
    if wrong_role:
        roles[0] = "actor"
    mask = np.arange(n) < 75
    trace = tmp_path / "episode.npz"
    np.savez(trace, role=roles, recovery_mask=mask,
             executed_action=np.tile(np.array([.4, -.5, .3], dtype=np.float32), (n, 1)))
    row = {"episode_id": 4, "path": trace.name, "sha256": file_sha256(trace),
           "paired_finish_qualified": True, "accepted_transitions": 75,
           "training_window_start": 0, "training_window_end": 75,
           "anchor_step": 0, "horizon": 12, "steps": n, "stratum": "failure",
           "censored": False, "finished": True, "local_recovery_qualified": True}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"eligibility_policy": FINISH_POLICY, "episodes": [row]}))
    return file_sha256(manifest)


def test_window_lookup_hash_and_actual_native_actions(tmp_path):
    digest = write_dataset(tmp_path)
    lookup, receipt = load_role_lookup(tmp_path, digest)
    assert len(lookup) == 75
    assert lookup[(4, 11)]["role"] == "oracle"
    assert lookup[(4, 12)]["role"] == "actor"
    assert receipt["counts"]["failure/oracle"] == 12
    assert receipt["counts"]["failure/actor"] == 63
    assert receipt["target"] == "executed_action"
    assert load_role_lookup(tmp_path, digest)[1] == receipt
    with (tmp_path / "episode.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="trace changed"):
        load_role_lookup(tmp_path, digest)


def test_lookup_rejects_wrong_executed_role(tmp_path):
    with pytest.raises(ValueError, match="roles disagree"):
        load_role_lookup(tmp_path, write_dataset(tmp_path, wrong_role=True))


def parent_fixture():
    return {"seed": 60, "total_steps": 8192, "first_update_step": 2,
            "arms": copy.deepcopy(trainer.ARMS), "runtime": {"fake": "CPU-test"},
            "recovery_dataset": trainer.DATASET,
            "recovery_receipt": {"manifest_sha256": trainer.DATASET_SHA},
            "source_hashes": {"old.py": "a" * 64}, "cells": [{"geometry_seed": 1}]}


def role_receipt():
    return {"accepted_rows": 665, "counts": {"failure/oracle": 87, "failure/actor": 252,
                                              "finish-control/oracle": 74,
                                              "finish-control/actor": 252}}


def test_child_contract_predeclared_weights_budget_and_compute_disclosure():
    child = trainer.child_contract(parent_fixture(), role_receipt(), {})
    assert child["guidance_config"] == {"guide_weight": 1., "retention_weight": .1, "gradient_clip": 10.}
    assert child["sac_updates"] == child["extra_actor_updates"] == 8191
    assert child["matched_compute_to_v1"] is False
    assert child["source_checkpoint_sha256"] == trainer.parent.CHECKPOINT_SHA
    assert child["evaluation"]["launch_automatically"] is False
    assert child["objective"]["retention_rows"] == ["ordinary-prior", "recovery/actor-handoff",
                                                     "recovery/finish-control/oracle"]
    assert child["objective"]["retention_target"] == "original-V5-source-actor-mean-on-same-images"
    assert child["objective"]["guidance_retention_masks_disjoint"] is True


@pytest.mark.parametrize("key,value", [("seed", 61), ("total_steps", 16384),
                                       ("first_update_step", 1), ("recovery_dataset", "other")])
def test_child_rejects_budget_or_dataset_drift(key, value):
    protocol = parent_fixture()
    protocol[key] = value
    with pytest.raises(ValueError):
        trainer.child_contract(protocol, role_receipt(), {})


def test_preflight_runtime_copy_and_child_code_freeze(tmp_path):
    protocol = parent_fixture()
    frozen_parent = copy.deepcopy(protocol)
    hashes = {name: "a" * 64 for name in trainer.SOURCE_FILES}
    child = trainer.child_contract(protocol, role_receipt(), hashes)
    child["created_at_utc"] = "synthetic"
    path = tmp_path / "child.json"
    path.write_text(json.dumps(child))
    with patch.object(trainer.parent, "pinned", return_value=path), \
            patch.object(trainer.parent, "preflight", return_value=(protocol, None, None, None)), \
            patch.object(trainer.parent, "located", return_value=tmp_path), \
            patch.object(trainer, "load_role_lookup", return_value=({}, role_receipt())), \
            patch.object(trainer, "source_hashes", return_value=hashes):
        _, runtime, *_ = trainer.preflight("unused", "b" * 64)
        assert protocol == frozen_parent
        assert runtime["arms"] == trainer.ARMS
        assert runtime["guidance"]["child_protocol_sha256"] == "b" * 64
        assert runtime["guidance"]["objective"] == child["objective"]
        assert set(runtime["source_hashes"]) == trainer.SOURCE_FILES
        runtime["arms"]["treatment"]["prior"] = 99
        assert protocol == frozen_parent
        changed = {**hashes, "haic/algorithms/rlpd/recovery_guidance.py": "c" * 64}
        with patch.object(trainer, "source_hashes", return_value=changed):
            with pytest.raises(ValueError, match="changed after freeze"):
                trainer.preflight("unused", "b" * 64)


def test_checkpoint_and_export_confess_guidance_and_aux_steps():
    agent = tiny_agent()
    agent.extra_actor_steps = 19
    with patch.object(PixelRLPDAgent, "checkpoint_state", return_value={"gradient_steps": 19}):
        state = agent.checkpoint_state()
    assert state["guidance"]["extra_actor_optimizer_steps"] == 19
    assert state["guidance"]["matched_compute_to_v1"] is False
    assert state["guidance"]["retention_source"] == ["ordinary-prior", "recovery/actor-handoff",
                                                     "recovery/finish-control/oracle"]
    assert state["guidance"]["guidance_retention_masks_disjoint"] is True
    with patch.object(PixelRLPDAgent, "export_actor", return_value="actor") as export:
        agent.export_actor("unused", environment_contract={"arm": "treatment"})
    assert export.call_args.kwargs["environment_contract"]["guidance"] == state["guidance"]


def test_real_pixel_auxiliary_encoder_unchanged_and_reference_readonly():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    base = PixelRLPDAgent(RLPDConfig(), device="cpu")
    agent = RecoveryGuidedRLPDAgent(RLPDConfig(), device="cpu",
        reference_state=base.actor.state_dict(), role_lookup={}, guidance_receipt={})
    agent.actor.load_state_dict(base.actor.state_dict())
    batch = batch_for(agent)
    batch["observation"] = np.zeros((22, 4, 84, 84), dtype=np.uint8)
    before = copy.deepcopy(agent.actor.encoder.state_dict())
    metrics = agent.guidance_update(batch)
    assert all(torch.equal(value, agent.actor.encoder.state_dict()[name]) for name, value in before.items())
    assert metrics["retention_loss"] == 0
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in agent.reference_actor.parameters())
    assert len(agent.actor_optimizer.state) > 0
    assert all(parameter not in agent.actor_optimizer.state for parameter in agent.actor.encoder.parameters())


def test_real_raw_sac_update_followed_by_auxiliary_step_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    base = PixelRLPDAgent(RLPDConfig(), device="cpu")
    agent = RecoveryGuidedRLPDAgent(RLPDConfig(), device="cpu",
        reference_state=base.actor.state_dict(), role_lookup={}, guidance_receipt={})
    agent.actor.load_state_dict(base.actor.state_dict())
    n = 64
    batch = {"source": np.array(["offline"] * 16 + ["online"] * 32 + ["recovery"] * 16),
             "observation": np.zeros((n, 4, 84, 84), dtype=np.uint8),
             "next_observation": np.zeros((n, 4, 84, 84), dtype=np.uint8),
             "action": np.zeros((n, 3), dtype=np.float32),
             "proposed_action": np.zeros((n, 3), dtype=np.float32),
             "reward": np.ones(n, dtype=np.float32), "terminal": np.zeros(n, dtype=bool),
             "episode_id": np.arange(n), "step": np.zeros(n, dtype=int)}
    for i in range(48, n):
        agent.role_lookup[(i, 0)] = {"role": "oracle", "stratum": "failure", "action": [0., 0., 0.]}
    metrics = agent.update(batch)
    assert metrics["sac_gradient_steps"] == metrics["extra_actor_optimizer_steps"] == 1
    assert metrics["offline_samples"] == 16 and metrics["online_samples"] == 32
    assert metrics["eligible_oracle_failure_rows"] == 16
    assert metrics["retention_rows"] == 16
    assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, (float, int)))
    assert agent.critic_optimizer.state and agent.temperature_optimizer.state
    assert all(parameter.grad is None for parameter in agent.actor.encoder.parameters())


def test_run_uses_original_state_fixed_treatment_and_exclusive_outputs(tmp_path):
    protocol = parent_fixture()
    receipt = role_receipt()
    source = tmp_path / "source.py"
    source.write_text("synthetic source")
    child = trainer.child_contract(protocol, receipt, {"synthetic.py": file_sha256(source)})
    child_path = tmp_path / "protocol.json"
    child_path.write_text(json.dumps(child))
    protocol_sha = file_sha256(child_path)
    protocol["guidance"] = {"child_protocol_sha256": protocol_sha}
    output = tmp_path / "run"
    original_state = {"actor": {"original": "not-recovery-v1"},
                      "actor_optimizer": {}, "critic_optimizer": {}, "temperature_optimizer": {}}

    class FakeGuided:
        def __init__(self, config, **kwargs):
            assert kwargs["reference_state"] is original_state["actor"]
            assert kwargs["seed"] == 60
            assert kwargs["device"] == "cuda"
            self.gradient_steps = self.extra_actor_steps = 8191

        def guidance_specification(self):
            return {"extra_actor_optimizer_steps": self.extra_actor_steps}

    def fake_loop(path, runtime, sha, config, prior, recovery, **kwargs):
        assert path == output and runtime is protocol and sha == protocol_sha
        assert kwargs["arm"] == "treatment" and runtime["arms"] == trainer.ARMS
        assert kwargs["learning_state"] is original_state
        assert {"actor_optimizer", "critic_optimizer", "temperature_optimizer"} <= set(kwargs["learning_state"])
        kwargs["agent_factory"](config, seed=60, device="cuda")
        return {"environment_steps": 8192, "gradient_steps": 8191}

    with patch.object(trainer, "preflight", return_value=(child, protocol, None, None, None, {})), \
            patch.object(trainer.parent, "located", return_value=output), \
            patch.object(trainer.parent, "source_path", return_value=source), \
            patch.object(trainer.parent, "pinned", return_value=child_path), \
            patch.object(trainer.parent, "source_checkpoint", return_value=source), \
            patch.object(trainer.torch, "load", return_value=original_state) as load, \
            patch.object(trainer, "RecoveryGuidedRLPDAgent", FakeGuided), \
            patch.object(trainer.parent, "train_loop", side_effect=fake_loop) as loop:
        stats = trainer.run("child", protocol_sha, "runs/synthetic")
        assert stats["sac_gradient_steps"] == stats["extra_actor_optimizer_steps"] == 8191
        assert stats["matched_compute_to_v1"] is False
        assert json.loads((output / "finalstats.json").read_text()) == stats
        assert (output / "source" / "synthetic.py").read_bytes() == source.read_bytes()
        assert load.call_args.kwargs == {"map_location": "cpu", "weights_only": False}
        with pytest.raises(FileExistsError):
            trainer.run("child", protocol_sha, "runs/synthetic")
        assert loop.call_count == 1


def test_freeze_exclusive_output_and_parent_preflight(tmp_path):
    target = tmp_path / "synthetic.json"
    with patch.object(trainer.parent, "located", return_value=target), \
            patch.object(trainer.parent, "preflight", return_value=(parent_fixture(), None, None, None)) as base, \
            patch.object(trainer, "load_role_lookup", return_value=({}, role_receipt())), \
            patch.object(trainer, "source_hashes", return_value={}):
        result = trainer.freeze("unused")
        base.assert_called_once_with(trainer.PARENT_PROTOCOL, trainer.PARENT_SHA)
        assert result["sha256"] == file_sha256(target)
        with pytest.raises(FileExistsError):
            trainer.freeze("unused")
