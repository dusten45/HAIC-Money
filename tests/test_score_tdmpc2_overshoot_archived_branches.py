"""Synthetic-only archived-branch scoring contracts; never load a trained model."""

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import score_tdmpc2_overshoot_archived_branches as score


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _put(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _fixed_protocol(monkeypatch, root):
    training = {"format": "haic-tdmpc2-reward-overshoot-train-v1", "run_dir": score.RUN,
                "source_sha256": {"learner.py": _sha(b"learner")},
                "reward_overshoot": {"enabled": True}, "runtime": {"python": "synthetic"},
                "seed_schedule": "synthetic-fixed", "baseline_training_result": {},
                "replay_audit": {}, "throughput_benchmark": {},
                "checkpoint_targets": [20000, 40000, 70000, 100000],
                "cells": [{"track_id": 1, "geometry_seed": seed} for seed in score.branch.ROADS],
                "episode_schedule": [0, 1, 2, 3], "training": {"horizon": 3, "discount": .995}}
    old = {"source_sha256": {"physics.py": _sha(b"physics")}, "augmentation_seed": 834,
           "anchors": [{"episode_id": ep, "start_step": start} for ep, start in score.branch.ANCHORS],
           "candidates": {"logged": None, **score.branch.FIXED_SUFFIXES}, "runtime": {"python": "synthetic"}}
    _put(root, "learner.py", b"learner")
    _put(root, "physics.py", b"physics")
    _put(root, score.SELF, b"synthetic evaluator")
    _put(root, score.TEST, b"synthetic tests")
    _put(root, score.TRAIN_PROTOCOL, json.dumps(training).encode())
    _put(root, score.OLD_PROTOCOL, json.dumps(old).encode())
    monkeypatch.setattr(score, "TRAIN_PROTOCOL_SHA", score.branch.digest(root / score.TRAIN_PROTOCOL))
    monkeypatch.setattr(score, "OLD_PROTOCOL_SHA", score.branch.digest(root / score.OLD_PROTOCOL))
    monkeypatch.setattr(score.branch, "runtime_identity", lambda: {"python": "synthetic"})
    original = {"primary": score._ref(score.OLD_PRIMARY, score.OLD_PRIMARY_SHA),
                "protocol": score._ref(score.OLD_PROTOCOL, score.OLD_PROTOCOL_SHA),
                "operator": score._ref(score.OLD_OPERATOR, score.OLD_OPERATOR_SHA),
                "model": score._ref(score.OLD_CHECKPOINT, score.OLD_CHECKPOINT_SHA),
                "summary": score._ref(score.OLD_SUMMARY, score.OLD_SUMMARY_SHA)}
    refs = {"protocol": score._ref(score.TRAIN_PROTOCOL, score.TRAIN_PROTOCOL_SHA),
            "result": score._ref(f"{score.RUN}/result.json", "1" * 64),
            "checkpoint": score._ref(f"{score.RUN}/checkpoint-at-least-100000-step-100001.pt", "2" * 64),
            "training_ledger": score._ref(f"{score.RUN}/training.jsonl", "3" * 64),
            "step_ledger": score._ref(f"{score.RUN}/steps.jsonl", "4" * 64)}
    spec = {"format": score.FORMAT, "purpose": "consumed-TRAIN-archived-branch-reward-only",
            "training_source": refs, "original_branch": original, "output": score.OUTPUT,
            "source_sha256": {**training["source_sha256"], **old["source_sha256"],
                              score.SELF: score.branch.digest(root / score.SELF),
                              score.TEST: score.branch.digest(root / score.TEST)},
            "runtime": {"python": "synthetic"},
            "gate": {"min_h5_concordant": 66, "h5_informative_pairs": 91,
                     "min_nonregressing_roads": 3, "min_h3_concordant": 31,
                     "h3_informative_pairs": 40, "h5_error_exclusive": score.ERROR_LIMIT}}
    path = _put(root, score.PROTOCOL, json.dumps(spec).encode())
    return spec, training, score.branch.digest(path)


def test_missing_score_protocol_never_touches_partial_ledger_or_untrusted_pickle(monkeypatch, tmp_path):
    _put(tmp_path, f"{score.RUN}/training.jsonl", b"partial live ledger\n")
    _put(tmp_path, f"{score.RUN}/steps.jsonl", b"partial live steps\n")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("untrusted torch.load"))
    monkeypatch.setattr(score, "_ledgers", lambda *a: pytest.fail("opened partial ledger"))
    with pytest.raises(ValueError, match="missing or changed pinned source"):
        score.run(tmp_path, "a" * 64, score=True)
    assert not (tmp_path / score.OUTPUT).exists()


def test_source_drift_after_first_preflight_blocks_torch_load(monkeypatch, tmp_path):
    _, _, sha = _fixed_protocol(monkeypatch, tmp_path)
    monkeypatch.setattr(score, "_archived", lambda *a: ({}, {}))
    monkeypatch.setattr(score, "_finished_result", lambda *a: ({}, {}))
    monkeypatch.setattr(score, "_ledgers", lambda *a: [])
    bundle = score.preflight(tmp_path, sha)
    _put(tmp_path, "physics.py", b"changed after first source gate")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("untrusted torch.load"))
    with pytest.raises(ValueError, match="missing or changed pinned source"):
        score._models(tmp_path, bundle)
    assert not (tmp_path / score.OUTPUT).exists()


@pytest.mark.parametrize("tamper", ["source", "missing_result", "bad_result", "missing_checkpoint", "checkpoint_sha"])
def test_source_and_complete_result_gate_before_any_ledger_or_torch_load(monkeypatch, tmp_path, tamper):
    spec, training, sha = _fixed_protocol(monkeypatch, tmp_path)
    _put(tmp_path, f"{score.RUN}/training.jsonl", b"partial but never read\n")
    monkeypatch.setattr(score, "_archived", lambda *a: ({}, {}))
    monkeypatch.setattr(score, "_ledgers", lambda *a: pytest.fail("read an in-progress ledger"))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("untrusted torch.load"))
    if tamper == "source":
        _put(tmp_path, "physics.py", b"changed")
        match = "missing or changed pinned source"
    else:
        result = {"format": training["format"], "status": "completed_boundary_at_least_100k",
                  "protocol_sha256": score.TRAIN_PROTOCOL_SHA,
                  "source_sha256": training["source_sha256"], "reward_overshoot": training["reward_overshoot"],
                  "runtime": training["runtime"], "seed_schedule": training["seed_schedule"],
                  "baseline_training_result": {}, "replay_audit": {}, "throughput_benchmark": {},
                  "pretrain_updates": 10000, "action_dim": 3, "reused_train_only": True,
                  "resume_supported": False, "evaluation": None,
                  "training_ledger_sha256": "3" * 64, "step_ledger_sha256": "4" * 64,
                  "decisions": 100001, "updates": 100001, "episodes": 4,
                  "checkpoints": [{"target": target, "decisions": target + 1, "updates": target + 1,
                                   "episodes": i + 1, "path": f"checkpoint-at-least-{target:06d}-step-{target + 1:06d}.pt",
                                   "sha256": "2" * 64}
                                  for i, target in enumerate((20000, 40000, 70000, 100000))]}
        if tamper == "bad_result":
            result["status"] = "partial"
        if tamper != "missing_result":
            path = _put(tmp_path, f"{score.RUN}/result.json", json.dumps(result).encode())
            spec["training_source"]["result"]["sha256"] = score.branch.digest(path)
        if tamper == "checkpoint_sha":
            _put(tmp_path, spec["training_source"]["checkpoint"]["path"], b"wrong checkpoint")
        if tamper == "missing_checkpoint":
            # Complete-looking result without a final model still cannot open the live ledger.
            assert result["checkpoints"][-1]["target"] == 100000
        _put(tmp_path, score.PROTOCOL, json.dumps(spec).encode())
        sha = score.branch.digest(tmp_path / score.PROTOCOL)
        match = "missing or changed pinned source|not complete and bound|four completed"
    with pytest.raises(ValueError, match=match):
        score.preflight(tmp_path, sha)
    assert not (tmp_path / score.OUTPUT).exists()


def _ledger_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "RUN", "runs/synthetic")
    seeds = (11, 12, 13, 14)
    monkeypatch.setattr(score.branch, "ROADS", seeds)
    training = {"source_sha256": {"learner.py": "a" * 64}, "reward_overshoot": {"enabled": True},
                "runtime": {"python": "synthetic"}, "checkpoint_targets": [2, 4, 6, 8],
                "cells": [{"track_id": 1, "geometry_seed": x} for x in seeds],
                "training": {"max_steps": 5}}
    action = np.array([0, -1, -1], np.float32).tobytes()
    native = score.environment_action(np.frombuffer(action, dtype=np.float32)).tobytes()
    rows = [{"event": "start", "protocol_sha256": score.TRAIN_PROTOCOL_SHA,
             "source_sha256": training["source_sha256"], "reward_overshoot": training["reward_overshoot"],
             "runtime": training["runtime"], "resume_supported": False}]
    checkpoints, steps, episodes = [], [], []
    for ep, seed in enumerate(seeds):
        cell = {"track_id": 1, "geometry_seed": seed}
        for event in ("reset_intent", "reset"):
            rows.append({"event": event, "episode": ep, "decisions": ep * 2, **cell})
        for offset in range(2):
            last = offset == 1
            steps.append({"episode": ep, "decision": ep * 2 + offset + 1, **cell,
                          "action_f32_hex": action.hex(), "native_action_f32_hex": native.hex(),
                          "reward": .5, "terminated": last, "truncated": False, "terminal": last})
        episode = {"event": "episode", "episode": ep, **cell, "decisions": (ep + 1) * 2,
                   "updates": 0, "length": 2, "return": 1.0, "progress": 0., "damage": 0.,
                   "finished": False, "terminated": True, "truncated": False, "terminal": True,
                   "action_trace_sha256": _sha(action * 2), "native_action_trace_sha256": _sha(native * 2)}
        rows.append(episode)
        episodes.append(episode)
        path = f"checkpoint-at-least-{(ep + 1) * 2:06d}-step-{(ep + 1) * 2:06d}.pt"
        data = f"synthetic checkpoint {ep}".encode()
        _put(tmp_path, f"{score.RUN}/{path}", data)
        previous = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
        prefix = b"".join(json.dumps(row).encode() + b"\n" for row in steps)
        report = {"target": (ep + 1) * 2, "decisions": (ep + 1) * 2,
                  "updates": 0, "episodes": ep + 1, "rolling_update_count": 0,
                  "training_ledger_sha256_before_checkpoint": _sha(previous),
                  "step_ledger_sha256": _sha(prefix), "path": path, "sha256": _sha(data),
                  "train_episodes_since_previous": [{key: episodes[ep][key] for key in (
                      "episode", "track_id", "geometry_seed", "decisions", "length", "return",
                      "progress", "damage", "finished")}]}
        rows.append({"event": "checkpoint", **report})
        checkpoints.append(report)
    ledger = _put(tmp_path, f"{score.RUN}/training.jsonl", b"".join(
        json.dumps(row).encode() + b"\n" for row in rows))
    step_path = _put(tmp_path, f"{score.RUN}/steps.jsonl", b"".join(
        json.dumps(row).encode() + b"\n" for row in steps))
    refs = {"training_ledger": score._ref(f"{score.RUN}/training.jsonl", score.branch.digest(ledger)),
            "step_ledger": score._ref(f"{score.RUN}/steps.jsonl", score.branch.digest(step_path))}
    result = {"decisions": 8, "updates": 0, "episodes": 4, "checkpoints": checkpoints}
    return {"training_source": refs}, training, result, ledger, step_path


@pytest.mark.parametrize("damage", ["truncated_training", "truncated_steps", "wrong_step_action", "wrong_checkpoint_prefix", "extra_step"])
def test_complete_ledger_refuses_resealed_corrupt_or_torn_records(monkeypatch, tmp_path, damage):
    spec, training, result, ledger, steps = _ledger_fixture(tmp_path, monkeypatch)
    assert len(score._ledgers(tmp_path, spec, training, result)) == 4
    if damage == "truncated_training":
        ledger.write_bytes(ledger.read_bytes()[:-1])
        spec["training_source"]["training_ledger"]["sha256"] = score.branch.digest(ledger)
    elif damage == "truncated_steps":
        steps.write_bytes(steps.read_bytes()[:-1])
        spec["training_source"]["step_ledger"]["sha256"] = score.branch.digest(steps)
    elif damage == "wrong_step_action":
        data = steps.read_text().replace('"native_action_f32_hex": "', '"native_action_f32_hex": "00', 1)
        steps.write_text(data)
        spec["training_source"]["step_ledger"]["sha256"] = score.branch.digest(steps)
    elif damage == "wrong_checkpoint_prefix":
        rows = ledger.read_text().replace('"rolling_update_count": 0', '"rolling_update_count": 1', 1)
        ledger.write_text(rows)
        spec["training_source"]["training_ledger"]["sha256"] = score.branch.digest(ledger)
    else:
        with steps.open("ab") as stream:
            stream.write(steps.read_bytes().splitlines(keepends=True)[-1])
        spec["training_source"]["step_ledger"]["sha256"] = score.branch.digest(steps)
    with pytest.raises(ValueError, match="truncated|lineage|checkpoint|ledger|cursor"):
        score._ledgers(tmp_path, spec, training, result)
    assert not (tmp_path / score.OUTPUT).exists()


def test_complete_checkpoint_gate_rejects_missing_model_even_when_result_claims_100k(tmp_path):
    root = tmp_path
    train = {"format": "haic-tdmpc2-reward-overshoot-train-v1", "source_sha256": {},
             "reward_overshoot": {}, "runtime": {}, "seed_schedule": "fixed",
             "baseline_training_result": {}, "replay_audit": {}, "throughput_benchmark": {}}
    result = {"format": train["format"], "status": "completed_boundary_at_least_100k",
              "protocol_sha256": score.TRAIN_PROTOCOL_SHA, "source_sha256": {},
              "reward_overshoot": {}, "runtime": {}, "seed_schedule": "fixed",
              "baseline_training_result": {}, "replay_audit": {}, "throughput_benchmark": {},
              "pretrain_updates": 10000, "action_dim": 3, "reused_train_only": True,
              "resume_supported": False, "evaluation": None, "decisions": 100001,
              "updates": 100001, "episodes": 4, "training_ledger_sha256": "1" * 64,
              "step_ledger_sha256": "2" * 64}
    result["checkpoints"] = [{"target": target, "decisions": target + 1,
                              "updates": target + 1, "episodes": n + 1,
                              "path": f"checkpoint-at-least-{target:06d}-step-{target + 1:06d}.pt",
                              "sha256": "3" * 64} for n, target in enumerate((20000, 40000, 70000, 100000))]
    path = _put(root, f"{score.RUN}/result.json", json.dumps(result).encode())
    last = result["checkpoints"][-1]
    spec = {"training_source": {"result": score._ref(f"{score.RUN}/result.json", score.branch.digest(path)),
                                "checkpoint": score._ref(f"{score.RUN}/{last['path']}", "3" * 64),
                                "training_ledger": score._ref(f"{score.RUN}/training.jsonl", "1" * 64),
                                "step_ledger": score._ref(f"{score.RUN}/steps.jsonl", "2" * 64)}}
    with pytest.raises(ValueError, match="missing or changed pinned source"):
        score._finished_result(root, spec, train)


class FakeModel:
    cfg = SimpleNamespace(num_bins=0)

    def eval(self):
        return self

    def encode(self, observation, task):
        assert tuple(observation.shape) == (1, *score.branch.PIXELS)
        return torch.zeros((1, 1))

    def reward(self, z, action, task):
        return z + action[:, 1:2]

    def next(self, z, action, task):
        return z + action[:, :1]

    def Q(self, *args, **kwargs):
        pytest.fail("reward-only score must not use Q")

    def pi(self, *args, **kwargs):
        pytest.fail("reward-only score must not use sampled policy")


def test_fake_model_h3_h5_decoding_same_action_and_crop_rng(monkeypatch):
    monkeypatch.setattr(score, "two_hot_inv", lambda tensor, config: tensor)
    model = FakeModel()
    pixels = np.zeros(score.branch.PIXELS, np.uint8)
    action = np.array([[1, 0, 0], [0, 1, 0], [0, -1, 0],
                       [0, 0, 0], [0, 0, 0]], np.float32)
    state = torch.get_rng_state().clone()
    predicted = score._predict_both(model, pixels, {"logged": action}, 834)["logged"]
    rewards = (0, 2, 0, 1, 1)
    assert predicted["h3"] == pytest.approx(sum(.995**t * r for t, r in enumerate(rewards[:3])))
    assert predicted["h5"] == pytest.approx(sum(.995**t * r for t, r in enumerate(rewards)))
    torch.testing.assert_close(torch.get_rng_state(), state)


def test_fake_five_action_pairs_count_real_ties_predicted_ties_and_terminal():
    rows = [{"real_h5_raw_return": actual, "new_h5_reward_return": predicted,
             "real_h3_prefix_raw_return": h3, "new_h3_reward_return": guess,
             "ending": ending, "steps": 5} for actual, predicted, h3, guess, ending in (
                 (4., 1., 0., 0., None), (3., 1., 2., 2., None),
                 (2., 2., 0., 0., "terminated"), (4. + 1e-10, 2., 0., 0., "finished"),
                 (1., 3., 0., 0., "timeout"))]
    counts = score._pairs(rows, "new_h5_reward_return", "real_h5_raw_return")
    assert counts == {"concordant": 1, "discordant": 6, "real_tie": 1, "predicted_tie": 2}
    prefix = score._pairs(rows, "new_h3_reward_return", "real_h3_prefix_raw_return")
    assert prefix["real_tie"] == 6 and prefix["concordant"] == 4
    assert sum(counts.values()) == sum(prefix.values()) == 10


def test_synthetic_ideal_new_predictions_pass_all_four_archived_gates(monkeypatch):
    # Immutable consumed-TRAIN branch outcomes only; all model predictions below are synthetic.
    primary = json.loads((score.ROOT / score.OLD_PRIMARY).read_text())
    old_summary = json.loads((score.ROOT / score.OLD_SUMMARY).read_text())
    old_by_seed = {}
    new_by_seed = {}
    for entry in primary["anchors"]:
        key = 834 + entry["episode_id"] * 1000 + entry["start_step"]
        old_by_seed[key] = {row["candidate"]: row["predicted_reward_return"]
                            for row in entry["candidates"]}
        new_by_seed[key] = {row["candidate"]: {f"h{h}": math.fsum(
            .995**t * value for t, value in enumerate(row["raw_rewards"][:h]))
            for h in (3, 5)} for row in entry["candidates"]}
    old_called = set()

    def old_predict(model, pixels, candidates, seed):
        old_called.add(seed)
        return old_by_seed[seed]

    def new_predict(model, pixels, candidates, seed):
        assert len(old_called) == 12, "all old-model parity must pass before new-model scoring"
        return new_by_seed[seed]

    monkeypatch.setattr(score.branch, "predicted_returns", old_predict)
    monkeypatch.setattr(score, "_predict_both", new_predict)
    entries, summary, gate = score.score_rows(primary, old_summary, FakeModel(), FakeModel())
    assert len(entries) == 12 and sum(len(e["candidates"]) for e in entries) == 60
    assert summary["h5"]["pairs"] == {"concordant": 91, "discordant": 0,
                                      "real_tie": 29, "predicted_tie": 0}
    assert summary["h3"]["pairs"] == {"concordant": 40, "discordant": 0,
                                      "real_tie": 80, "predicted_tie": 0}
    assert summary["h5_absolute_error_per_discounted_step"] == 0
    assert gate["road_nonregression"]["nonregressing_roads"] == 4
    assert gate["passed"] is True
    for bad in ("early_terminal", "different_pixel", "different_action", "old_prediction"):
        changed = copy.deepcopy(primary)
        row = changed["anchors"][0]["candidates"][0]
        if bad == "early_terminal":
            row["steps"] = 3
            row["ending"] = "terminated"
        elif bad == "different_pixel":
            changed["anchors"][0]["anchor_model_observation_hex"] = "00"
        elif bad == "different_action":
            row["model_action_bytes_hex"] = "00" * 60
        else:
            row["predicted_reward_return"] += 1e-4
        with pytest.raises(ValueError, match="archived"):
            score.score_rows(changed, old_summary, FakeModel(), FakeModel())
    for entry in primary["anchors"]:
        key = 834 + entry["episode_id"] * 1000 + entry["start_step"]
        for estimate in new_by_seed[key].values():
            estimate["h5"] += 10
    _, _, failed = score.score_rows(primary, old_summary, FakeModel(), FakeModel())
    assert not failed["h5_normalized_error"]["passed"] and not failed["passed"]


def test_receipt_exclusive_fsync_body_sha_and_no_overwrite(monkeypatch, tmp_path):
    (tmp_path / "runs").mkdir()
    report = {"status": "FAIL", "environment_resets": 0, "optimizer_updates": 0,
              "gate": {"passed": False}}
    score._write_exclusive(tmp_path, report)
    path = tmp_path / score.OUTPUT
    data = path.read_bytes()
    written = json.loads(data)
    assert written["body_sha256"] == score._body_sha({key: value for key, value in written.items()
                                                       if key != "body_sha256"})
    with pytest.raises(ValueError, match="absent"):
        score._write_exclusive(tmp_path, {"status": "PASS"})
    assert path.read_bytes() == data


def test_no_output_when_no_frozen_model_or_if_source_drifts_after_first_preflight(monkeypatch, tmp_path):
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(score, "preflight", lambda *args: (_ for _ in ()).throw(
        ValueError("complete checkpoint not available")))
    monkeypatch.setattr(torch, "load", lambda *args, **kw: pytest.fail("loaded partial model"))
    with pytest.raises(ValueError, match="complete checkpoint"):
        score.run(tmp_path, "a" * 64, score=True)
    assert not (tmp_path / score.OUTPUT).exists()


def test_optimizer_step_metadata_cannot_claim_completed_model_with_short_updates():
    def state(step):
        return {"state": {0: {"step": torch.tensor(step), "exp_avg": torch.tensor(1.),
                              "exp_avg_sq": torch.tensor(1.)}},
                "param_groups": [{"params": [0]}]}
    score._optimizer(state(100001), 100001)
    for invalid in (state(70000), {"state": {}, "param_groups": []},
                    {"state": {0: {"step": torch.tensor(100001)}}, "param_groups": [{"params": [0]}]}):
        with pytest.raises(ValueError, match="optimizer"):
            score._optimizer(invalid, 100001)


@pytest.mark.parametrize("tamper", [None, "action", "reward", "mask", "boundary", "extra_key"])
def test_overshoot_checkpoint_probe_binds_masked_suffix_to_saved_replay(monkeypatch, tamper):
    from scripts import diagnose_tdmpc2_checkpoint_losses as base_probe

    base_keys = {"obs", "action", "reward", "terminated", "truncated", "terminal",
                 "bootstrap_mask", "episode_id", "start_step"}
    observed = []

    def fake_base(state, training, row):
        observed.append((training, row))
        assert set(state["probe"]) == base_keys

    monkeypatch.setattr(base_probe, "_probe", fake_base)
    probe = {key: torch.zeros(1) for key in base_keys}
    probe["episode_id"] = torch.zeros(256, dtype=torch.int64)
    probe["start_step"] = torch.zeros(256, dtype=torch.int64)
    actions = np.tile(np.array([0., -1., 0.], np.float32), (5, 1))
    rewards = np.array([0., 1., 2., 3., 4.], np.float32)
    episode = {"start_step": 0, "actions": actions, "rewards": rewards,
               "terminated": np.array([False, False, False, False, True]),
               "truncated": np.zeros(5, np.bool_)}
    probe["overshoot_action"] = torch.from_numpy(np.tile(actions[3:5, None, :], (1, 256, 1)).copy())
    probe["overshoot_reward"] = torch.from_numpy(np.tile(rewards[3:5, None, None], (1, 256, 1)).copy())
    probe["overshoot_mask"] = torch.ones((2, 256, 1), dtype=torch.float32)
    state = {"probe": probe, "replay": {"episodes": [episode]}}
    if tamper == "action":
        probe["overshoot_action"][0, 0, 0] = .25
    elif tamper == "reward":
        probe["overshoot_reward"][1, 0, 0] += 1
    elif tamper == "mask":
        probe["overshoot_mask"][0, 0, 0] = 0
    elif tamper == "boundary":
        episode["terminated"][2] = True
    elif tamper == "extra_key":
        probe["forged"] = torch.tensor(1)
    if tamper is None:
        score._overshoot_probe(state, {"training": "synthetic"}, {"checkpoint": "synthetic"})
        assert observed == [({"training": "synthetic"}, {"checkpoint": "synthetic"})]
    else:
        with pytest.raises(ValueError, match="probe|boundary"):
            score._overshoot_probe(state, {}, {})


@pytest.mark.parametrize("tamper", [None, "action", "reward", "terminal", "pixels", "incomplete"])
def test_complete_checkpoint_replay_binds_every_episode_to_hashed_steps(monkeypatch, tmp_path, tamper):
    spec, _, result, _, _ = _ledger_fixture(tmp_path, monkeypatch)
    action = np.array([0, -1, -1], np.float32)
    stored = []
    for eid in range(4):
        term = np.array([False, True], np.bool_)
        stored.append({"episode_id": eid, "start_step": 0,
                       "actions": np.tile(action, (2, 1)),
                       "rewards": np.array([.5, .5], np.float32),
                       "observations": np.zeros((3, *score.branch.PIXELS), np.uint8),
                       "terminated": term.copy(), "truncated": np.zeros(2, np.bool_),
                       "terminal": term.copy()})
    if tamper == "action":
        stored[1]["actions"][0, 0] = .5
    elif tamper == "reward":
        stored[2]["rewards"][0] = 5
    elif tamper == "terminal":
        stored[0]["terminal"][-1] = False
    elif tamper == "pixels":
        stored[3]["observations"] = stored[3]["observations"][:2]
    elif tamper == "incomplete":
        stored.pop()
    native = score.environment_action(action).tobytes()
    bundle = {"spec": spec, "episodes": [{"length": 2, "decisions": (eid + 1) * 2,
                                            "action_trace_sha256": _sha(action.tobytes() * 2),
                                            "native_action_trace_sha256": _sha(native * 2),
                                            "terminated": True, "truncated": False, "terminal": True}
                                           for eid in range(4)]}
    if tamper is None:
        score._bind_checkpoint_replay(tmp_path, {"replay": {"episodes": stored}}, bundle)
    else:
        with pytest.raises(ValueError, match="replay"):
            score._bind_checkpoint_replay(tmp_path, {"replay": {"episodes": stored}}, bundle)
