"""Synthetic-only source gates and fixed-candidate planner arithmetic."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haic.algorithms.tdmpc2.model import two_hot_inv
from scripts import diagnose_tdmpc2_overshoot_planner_terms as diag


class TinyModel:
    cfg = SimpleNamespace(num_bins=101, num_q=5, vmin=-10., vmax=10.)

    def __init__(self, terminal_step):
        self.terminal_step = terminal_step
        self.draws = []

    def encode(self, pixels, task):
        return torch.zeros((1, 1))

    def next(self, z, action, task):
        return z + 1

    def reward(self, z, action, task):
        logits = torch.full((z.shape[0], 101), -100.)
        logits[:, 51] = 100.
        return logits

    def termination(self, z, task):
        return torch.full((z.shape[0], 1), .9 if int(z[0, 0]) >= self.terminal_step else .1)

    def pi(self, z, task):
        sample = torch.randn((z.shape[0], 3)).clamp(-1, 1)
        self.draws.append(sample.clone())
        return sample, {}

    def Q(self, z, action, task, *, return_type):
        assert return_type == "all"
        bins = (51 + torch.arange(5))[:, None, None].expand(5, z.shape[0], 1)
        logits = torch.full((5, z.shape[0], 101), -100.)
        return logits.scatter(2, bins, 100.)


def _actions():
    return {"logged": np.zeros((5, 3), np.float32), **{
        name: np.asarray(value, np.float32) for name, value in diag.branch.FIXED_SUFFIXES.items()}}


@pytest.mark.parametrize("terminal_step,horizon,alive_before,alive_after", [
    (1, 3, [True, False, False], [False, False, False]),
    (5, 5, [True] * 5, [True] * 4 + [False]),
    (6, 5, [True] * 5, [True] * 5),
])
def test_terminal_t0_t4_reward_retained_and_exact_planner_parity(
    terminal_step, horizon, alive_before, alive_after, monkeypatch,
):
    model = TinyModel(terminal_step)
    monkeypatch.setattr(torch, "randperm", lambda *a, **k: pytest.fail("random Q pair chosen"))
    prior_rng = torch.get_rng_state().clone()
    result = diag._terms(model, np.zeros(diag.branch.PIXELS, np.uint8), _actions(), 834, horizon)
    torch.testing.assert_close(torch.get_rng_state(), prior_rng, atol=0, rtol=0)
    assert result["bootstrap_seed"] == 100834
    assert len(model.draws) == (11 if alive_after[-1] else 1)
    for draw in model.draws[1:]:
        torch.testing.assert_close(draw, model.draws[0], atol=0, rtol=0)
    row = result["candidates"]["logged"]
    assert row["alive_before_reward"] == alive_before
    assert row["alive_after_transition"] == alive_after
    assert len(row["raw_reward_sequence"]) == horizon
    assert row["gated_reward"] == pytest.approx(sum(.995**t * row["raw_reward_sequence"][t]
                                                     for t, is_alive in enumerate(alive_before) if is_alive), abs=1e-5)
    assert row["bootstrap_alive"] == alive_after[-1]
    logits = model.Q(torch.zeros((5, 1)), torch.zeros((5, 3)), None, return_type="all")
    expected_q = two_hot_inv(logits, model.cfg)[:, 0, 0]
    assert row["decoded_online_q_heads"] == pytest.approx(expected_q.tolist(), abs=1e-6)
    assert len(row["fixed_pairs"]) == 10
    term = row["fixed_pairs"]["0,4"]["bootstrap_q_term"]
    assert term == pytest.approx((.995 ** horizon) * float((expected_q[0] + expected_q[4]) / 2)
                                 if alive_after[-1] else 0, abs=1e-5)


def test_reward_ungated_vs_terminal_gated_order_and_identical_q_heads():
    model = TinyModel(1)
    row = diag._terms(model, np.zeros(diag.branch.PIXELS, np.uint8), _actions(), 834, 5)["candidates"]["logged"]
    assert row["ungated_reward"] > row["gated_reward"]
    assert row["fixed_pairs"]["0,1"]["full_score"] == pytest.approx(row["gated_reward"])


@pytest.mark.parametrize("bad", ["actions", "pixels", "sixth", "horizon"])
def test_reject_input_shape_before_encoding(bad):
    actions = _actions()
    pixels = np.zeros(diag.branch.PIXELS, np.uint8)
    horizon = 3
    if bad == "actions":
        actions["logged"] = actions["logged"][:3]
    elif bad == "pixels":
        pixels = pixels[:2]
    elif bad == "sixth":
        actions["extra"] = np.zeros((5, 3), np.float32)
    else:
        horizon = 4
    with pytest.raises(ValueError, match="invalid"):
        diag._terms(TinyModel(6), pixels, actions, 834, horizon)


def _archived_fixture(monkeypatch):
    monkeypatch.setattr(diag.fixed, "_verify_receipt", lambda old, spec: old["anchors"])
    original = {"anchors": []}
    new = {"format": diag.score.RESULT_FORMAT, "status": "FAIL", "protocol":
           diag._ref(diag.score.PROTOCOL, diag.SCORE_PROTOCOL_SHA),
           "original_branch": {"primary": diag._ref(diag.score.OLD_PRIMARY, diag.score.OLD_PRIMARY_SHA)},
           "environment_resets": 0, "optimizer_updates": 0,
           "scope": "reused_TRAIN_archived_branches_only", "gate": {"passed": False,
           "h5_normalized_error": {"passed": False}}, "anchors": []}
    pixel = np.zeros(diag.branch.PIXELS, np.uint8).tobytes()
    for episode, start in diag.branch.ANCHORS:
        actions = _actions()
        old_rows, new_rows = [], []
        for name in diag.NAMES:
            action_hex = actions[name].tobytes().hex()
            old_rows.append({"candidate": name, "predicted_reward_return": 1.,
                             "h3_predicted_reward_return": .6, "real_discounted_raw_return": 2.,
                             "real_h3_prefix_raw_return": 1., "model_action_bytes_hex": action_hex})
            new_rows.append({"candidate": name, "old_h5_reward_return": 1.,
                             "new_h3_reward_return": .8, "new_h5_reward_return": 1.5,
                             "real_h5_raw_return": 2., "real_h3_prefix_raw_return": 1.,
                             "model_action_bytes_hex": action_hex, "steps": 5, "ending": None})
        original["anchors"].append({"anchor_model_observation_hex": pixel.hex(),
            "anchor_model_observation_sha256": hashlib.sha256(pixel).hexdigest(),
            "candidate_action_bytes_hex": {name: actions[name].tobytes().hex() for name in diag.NAMES},
            "candidates": old_rows})
        new["anchors"].append({"episode_id": episode, "start_step": start,
            "geometry_seed": diag.branch.ROADS[episode],
            "anchor_model_observation_sha256": hashlib.sha256(pixel).hexdigest(), "candidates": new_rows})
    new["body_sha256"] = diag._body(new)
    return original, new


@pytest.mark.parametrize("tamper", [None, "body", "pixel", "action", "outcome", "return", "short"])
def test_archived_candidate_bytes_and_body_fail_closed(monkeypatch, tamper):
    old, new = _archived_fixture(monkeypatch)
    if tamper == "body":
        new["body_sha256"] = "0" * 64
    elif tamper == "pixel":
        new["anchors"][0]["anchor_model_observation_sha256"] = "0" * 64
    elif tamper == "action":
        new["anchors"][0]["candidates"][0]["model_action_bytes_hex"] = "01" * 60
    elif tamper == "outcome":
        new["anchors"][0]["candidates"][0]["real_h5_raw_return"] += .1
    elif tamper == "return":
        new["anchors"][0]["candidates"][0]["old_h5_reward_return"] += .1
    elif tamper == "short":
        new["anchors"][0]["candidates"][0]["steps"] = 4
    if tamper not in (None, "body"):
        new["body_sha256"] = diag._body({key: value for key, value in new.items() if key != "body_sha256"})
    if tamper is None:
        assert len(diag._archived_inputs(old, new, {})) == 12
    else:
        with pytest.raises(ValueError, match="body|anchor|candidate"):
            diag._archived_inputs(old, new, {})


def test_missing_frozen_protocol_prevents_deserialization_and_output(monkeypatch, tmp_path):
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("checkpoint deserialized"))
    with pytest.raises(ValueError, match="missing or changed pinned source"):
        diag.run(tmp_path, "a" * 64, execute=True)
    assert not (tmp_path / diag.OUTPUT).exists()


def test_source_drift_and_checkpoint_pin_before_load(monkeypatch, tmp_path):
    """Synthetic protocol checks fail before the checkpoint loader is reached."""
    (tmp_path / "runs").mkdir()
    (tmp_path / "experiments").mkdir()
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("checkpoint deserialized"))
    # Even a well-formed protocol cannot replace the pinned original score protocol.
    spec = {"format": "haic-tdmpc2-overshoot-planner-terms-v1",
            "purpose": "read-only-consumed-TRAIN-fixed-candidate-mechanism",
            "score_source": {"protocol": diag._ref(diag.score.PROTOCOL, diag.SCORE_PROTOCOL_SHA),
                             "primary": diag._ref(diag.score.OUTPUT, diag.SCORE_SHA)},
            "fixed_q_summary": diag._ref("experiments/tdmpc2-h5-fixed-q-v1-result.json", diag.FIXED_SHA),
            "source_sha256": {}, "runtime": {}, "output": diag.OUTPUT,
            "augmentation_seed": 834, "bootstrap_offset": diag.BOOTSTRAP_OFFSET,
            "discount": diag.branch.DISCOUNT, "tie_tolerance": diag.branch.TIE_TOLERANCE,
            "anchors": [{"episode_id": ep, "start_step": start} for ep, start in diag.branch.ANCHORS],
            "candidates": list(diag.NAMES), "policy_release": False}
    path = tmp_path / diag.PROTOCOL
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(diag.score, "preflight", lambda *a: (_ for _ in ()).throw(ValueError("checkpoint SHA drift")))
    with pytest.raises(ValueError, match="checkpoint SHA drift"):
        diag.run(tmp_path, hashlib.sha256(path.read_bytes()).hexdigest(), execute=True)
    spec["score_source"]["primary"]["sha256"] = "a" * 64
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="separately frozen"):
        diag.run(tmp_path, hashlib.sha256(path.read_bytes()).hexdigest(), execute=True)
    assert not (tmp_path / diag.OUTPUT).exists()


@pytest.mark.parametrize("drift", ["source", "receipt_body", "score_sha"])
def test_complete_preflight_rejects_resealed_source_and_receipt_drift_before_load(monkeypatch, tmp_path, drift):
    (tmp_path / "runs").mkdir()
    (tmp_path / "experiments").mkdir()
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("checkpoint deserialized"))
    module = "synthetic.py"
    (tmp_path / module).write_bytes(b"original source")
    fixed_path = tmp_path / "scripts/diagnose_tdmpc2_h5_fixed_q.py"
    fixed_path.parent.mkdir()
    fixed_path.write_bytes(b"synthetic fixed operator")
    self_path = tmp_path / diag.SELF
    self_path.write_bytes(b"synthetic diagnostic")
    test_path = tmp_path / diag.TEST
    test_path.parent.mkdir()
    test_path.write_bytes(b"synthetic tests")
    digest = diag.branch.digest
    sources = {module: digest(tmp_path / module),
               "scripts/diagnose_tdmpc2_h5_fixed_q.py": digest(fixed_path),
               diag.SELF: digest(self_path), diag.TEST: digest(test_path)}
    fixed_summary = {"operator": "scripts/diagnose_tdmpc2_h5_fixed_q.py",
                     "operator_sha256": digest(fixed_path),
                     "source_h5_primary_receipt_sha256": diag.score.OLD_PRIMARY_SHA,
                     "source_h5_branch_protocol_sha256": diag.score.OLD_PROTOCOL_SHA,
                     "environment_resets": 0, "optimizer_steps": 0, "informative_real_h5_pairs": 91}
    summary_path = tmp_path / "experiments/tdmpc2-h5-fixed-q-v1-result.json"
    summary_path.write_text(json.dumps(fixed_summary))
    monkeypatch.setattr(diag, "FIXED_SHA", digest(summary_path))
    score_receipt = {"status": "FAIL", "protocol": diag._ref(diag.score.PROTOCOL, diag.SCORE_PROTOCOL_SHA),
                     "training_source": {}, "original_branch": {},
                     "source_sha256": {module: sources[module]}, "runtime": {"python": "synthetic"}}
    score_receipt["body_sha256"] = diag._body(score_receipt)
    score_path = tmp_path / diag.score.OUTPUT
    score_path.write_text(json.dumps(score_receipt))
    monkeypatch.setattr(diag, "SCORE_SHA", digest(score_path))
    spec = {"format": "haic-tdmpc2-overshoot-planner-terms-v1",
            "purpose": "read-only-consumed-TRAIN-fixed-candidate-mechanism",
            "score_source": {"protocol": diag._ref(diag.score.PROTOCOL, diag.SCORE_PROTOCOL_SHA),
                             "primary": diag._ref(diag.score.OUTPUT, diag.SCORE_SHA)},
            "fixed_q_summary": diag._ref("experiments/tdmpc2-h5-fixed-q-v1-result.json", diag.FIXED_SHA),
            "source_sha256": sources, "runtime": {"python": "synthetic"}, "output": diag.OUTPUT,
            "augmentation_seed": 834, "bootstrap_offset": diag.BOOTSTRAP_OFFSET,
            "discount": diag.branch.DISCOUNT, "tie_tolerance": diag.branch.TIE_TOLERANCE,
            "anchors": [{"episode_id": ep, "start_step": start} for ep, start in diag.branch.ANCHORS],
            "candidates": list(diag.NAMES), "policy_release": False}
    protocol = tmp_path / diag.PROTOCOL
    protocol.write_text(json.dumps(spec))
    bundle = {"spec": {"source_sha256": {module: sources[module]}, "runtime": spec["runtime"],
                       "training_source": {}, "original_branch": {}},
              "protocol_sha256": diag.SCORE_PROTOCOL_SHA, "primary": {}}
    monkeypatch.setattr(diag.score, "preflight", lambda *a: bundle)
    monkeypatch.setattr(diag.branch, "runtime_identity", lambda: spec["runtime"])
    monkeypatch.setattr(diag, "_archived_inputs", lambda *a: [])
    if drift == "source":
        (tmp_path / module).write_bytes(b"changed source")
        match = "source/runtime pins|missing or changed pinned source"
    elif drift == "receipt_body":
        score_receipt["body_sha256"] = "0" * 64
        score_path.write_text(json.dumps(score_receipt))
        monkeypatch.setattr(diag, "SCORE_SHA", digest(score_path))
        spec["score_source"]["primary"]["sha256"] = diag.SCORE_SHA
        protocol.write_text(json.dumps(spec))
        match = "archived branch body SHA differs"
    else:
        score_path.write_bytes(score_path.read_bytes() + b" ")
        match = "missing or changed pinned source"
    with pytest.raises(ValueError, match=match):
        diag.preflight(tmp_path, digest(protocol))
    assert not (tmp_path / diag.OUTPUT).exists()


def test_exclusive_receipt_and_body_sha(tmp_path):
    (tmp_path / "runs").mkdir()
    report = {"status": "complete_diagnostic_only", "policy_release": False,
              "environment_resets": 0, "optimizer_steps": 0}
    diag._write_exclusive(tmp_path, report)
    path = tmp_path / diag.OUTPUT
    first = path.read_bytes()
    decoded = json.loads(first)
    assert decoded["body_sha256"] == diag._body({k: v for k, v in decoded.items() if k != "body_sha256"})
    with pytest.raises(ValueError, match="absent"):
        diag._write_exclusive(tmp_path, copy.deepcopy(report))
    assert path.read_bytes() == first
