"""Synthetic-only frozen-branch checkpoint scorer tests; no HAIC environment."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import diagnose_tdmpc2_branch_checkpoint_ranking as scorer


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _fixtures():
    protocol = {
        "format": "haic-tdmpc2-v2-prefix-branches-v1", "purpose": "consumed-TRAIN-same-state-diagnostic",
        "anchors": [{"episode_id": ep, "start_step": step} for ep, step in scorer.ANCHORS],
        "candidates": {"logged": None, **{name: [list(action)] * 3
                                         for name, action in scorer.FIXED_ACTIONS.items()}},
        "environment": {"track_id": 1, "geometry_seeds": list(scorer.ROADS),
                        "max_steps": 2000, "frame_skip": 4, "obstacles": True, "reward_shaping": False},
        "horizon": 3, "discount": .995, "tie_tolerance": 1e-6,
        "augmentation_seed": 73301, "max_resets": 72,
        "source_sha256": {"source.py": _hash(b"synthetic source")},
        "v2_sha256": {"v2.pt": "b" * 64}, "runtime": {"torch": "synthetic"},
    }
    anchors = []
    for index, (episode, step) in enumerate(scorer.ANCHORS):
        pixels = np.full(scorer.PIXELS, index, dtype=np.uint8).tobytes()
        # 5 all-tied anchors, six 3:2 splits, one 4:1 split:
        # 50 + 24 + 6 = 80 real ties, leaving 40 informative pairs.
        if index < 5:
            actual, predicted = [0] * 5, list(range(5))
        elif index < 11:
            actual = [0, 0, 0, 1, 1]
            predicted = ([0, 2, 3, 1, 4] if index < 9 else [0, 2, 4, 1, 3])
        else:
            actual, predicted = [0, 0, 0, 0, 1], [0, 1, 2, 4, 3]
        candidates = []
        for name, real, pred in zip(scorer.NAMES, actual, predicted):
            action = ([.25, .5, -.5] if name == "logged" else scorer.FIXED_ACTIONS[name])
            actions = np.asarray([action] * 3, dtype="<f4")
            candidates.append({"candidate": name, "actions_model": actions.tolist(),
                               "model_action_bytes_hex": actions.tobytes().hex(),
                               "full_h3": True, "ranking_eligible": True, "steps": 3,
                               "terminal_excluded": False, "terminated": False, "truncated": False,
                               "raw_rewards": [real, 0, 0], "real_discounted_raw_return": real,
                               "predicted_reward_return": pred})
        counts = scorer._pairs(candidates)
        comparable = counts["concordant"] + counts["discordant"]
        anchors.append({"episode_id": episode, "start_step": step, "track_id": 1,
                        "geometry_seed": scorer.ROADS[episode],
                        "anchor_model_observation_hex": pixels.hex(),
                        "anchor_model_observation_sha256": _hash(pixels),
                        "anchor_observation_sha256": "a" * 64,
                        "anchor_accessible_state_sha256": "c" * 64,
                        "road_sha256": _hash(str(episode).encode()),
                        "candidates": candidates,
                        "ranking": {"pairs": counts, "comparable_pairs": comparable,
                                    "concordance": counts["concordant"] / comparable if comparable else None,
                                    "ranking_identifiable": comparable > 0, "full_h3_candidates": 5,
                                    "ranking_eligible_candidates": 5, "terminal_candidates": 0}})
    total = {key: sum(anchor["ranking"]["pairs"][key] for anchor in anchors)
             for key in scorer.PAIR_KEYS}
    assert total == scorer.BASELINE_PAIRS
    receipt = {
        "aggregate_pair_counts": total, "anchor_count": 12, "anchors": anchors,
        "candidate_count_per_anchor": 5, "concordance_on_comparable_pairs": .625,
        "discount": .995, "environment_resets_attempted": 72,
        "format": "haic-tdmpc2-v2-prefix-branches-result-v1",
        "historical_road_or_hidden_box2d_state_proven": False,
        "horizon": 3, "official_or_protected_result": False,
        "protocol_sha256": scorer.BRANCH_PROTOCOL_SHA256, "runtime": protocol["runtime"],
        "same_state_counterfactual_ranking": True,
        "scope": "reused_consumed_TRAIN_same_reconstructed_accessible_state_only",
        "source_sha256": protocol["source_sha256"], "status": "complete",
        "tie_tolerance": 1e-6, "v2_sha256": protocol["v2_sha256"],
    }
    summary = {"status": "complete", "protocol_sha256": scorer.BRANCH_PROTOCOL_SHA256,
               "primary_receipt": scorer.BRANCH_RECEIPT,
               "primary_receipt_sha256": scorer.BRANCH_RECEIPT_SHA256,
               "predeclared_anchors": 12, "completed_anchors": 12, "candidate_sequences": 60,
               "full_h3_candidates": 60, "execution_environment_resets_attempted": 72,
               "real_return_ties_at_1e-6": 80, "concordant_pairs": 25,
               "discordant_pairs": 15, "informative_pairs": 40, "informative_anchors": 7,
               "terminal_or_short_excluded_candidates": 0}
    return protocol, receipt, summary


def test_synthetic_baseline_reconstructs_25_15_80_from_all_60_suffixes():
    protocol, receipt, summary = _fixtures()
    anchors = scorer._validate_branch(protocol, receipt, summary)
    assert len(anchors) == 12
    assert sum(a["ranking"]["comparable_pairs"] > 0 for a in anchors) == 7
    assert anchors[0]["_decoded"][0].shape == scorer.PIXELS
    assert len([row for anchor in anchors for row in anchor["_decoded"][1]]) == 60


@pytest.mark.parametrize("change,match", [
    (lambda p, r, s: r.update(status="running"), "incomplete"),
    (lambda p, r, s: r.update(environment_resets_attempted=71), "72-reset"),
    (lambda p, r, s: r["anchors"].pop(), "72-reset"),
    (lambda p, r, s: r["anchors"][0].update(anchor_model_observation_sha256="0" * 64), "observation"),
    (lambda p, r, s: r["anchors"][0]["candidates"][0].update(model_action_bytes_hex="0" * 72), "actions"),
    (lambda p, r, s: r["anchors"][0]["candidates"][0]["raw_rewards"].__setitem__(0, 1.), "return"),
    (lambda p, r, s: r["anchors"][0]["candidates"][0].update(terminated=True), "terminal"),
    (lambda p, r, s: r["anchors"][0]["candidates"].reverse(), "reordered"),
    (lambda p, r, s: r["anchors"][0]["ranking"]["pairs"].update(concordant=1), "pair counts"),
    (lambda p, r, s: r["anchors"][1].update(road_sha256="d" * 64), "road signature"),
    (lambda p, r, s: s.update(informative_anchors=8), "summary"),
    (lambda p, r, s: r["anchors"][0]["candidates"][0].update(predicted_reward_return=float("nan")), "nonnumeric"),
])
def test_branch_tamper_fails_closed(change, match):
    protocol, receipt, summary = _fixtures()
    change(protocol, receipt, summary)
    with pytest.raises(ValueError, match=match):
        scorer._validate_branch(protocol, receipt, summary)


def test_frozen_sha_duplicate_keys_and_symlink_fail_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / scorer.BRANCH_RECEIPT
    path.parent.mkdir()
    path.write_bytes(b'{"a":1,"a":2}')
    monkeypatch.setattr(scorer, "BRANCH_RECEIPT_SHA256", _hash(path.read_bytes()))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        scorer._pinned_json(tmp_path, scorer.BRANCH_RECEIPT, scorer.BRANCH_RECEIPT_SHA256)
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="SHA mismatch"):
        scorer._pinned_json(tmp_path, scorer.BRANCH_RECEIPT, scorer.BRANCH_RECEIPT_SHA256)
    path.unlink()
    path.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="symlink"):
        scorer._pinned_json(tmp_path, scorer.BRANCH_RECEIPT, scorer.BRANCH_RECEIPT_SHA256)


class FakeRewardModel:
    def __init__(self):
        self.cfg = SimpleNamespace(num_bins=0)
        self.encodes = 0
        self.training = True

    def eval(self):
        self.training = False
        return self

    def encode(self, pixels, task):
        assert not self.training and pixels.shape == (1, *scorer.PIXELS)
        self.encodes += 1
        return torch.rand(1, 1) + pixels[:, 0, 0, 0:1].float() / 255

    def reward(self, z, a, task):
        return z + a[:, :1]

    def next(self, z, a, task):
        return z + .1 * a[:, 1:2]

    def Q(self, *args, **kwargs):
        pytest.fail("reward-only score must not use Q bootstrap")


def test_reward_only_same_anchor_one_encode_shared_seed_and_no_rng_change():
    protocol, receipt, summary = _fixtures()
    anchor = scorer._validate_branch(protocol, receipt, summary)[5]
    pixels, rows = anchor["_decoded"]
    before = torch.get_rng_state().clone()
    first_model, second_model = FakeRewardModel(), FakeRewardModel()
    first = scorer._predictions(first_model, pixels, rows, 73301 + 50)
    second = scorer._predictions(second_model, pixels, rows, 73301 + 50)
    assert first == second and torch.equal(torch.get_rng_state(), before)
    assert first_model.encodes == second_model.encodes == 1
    assert first[1]["real_discounted_raw_return"] == 0
    assert first[1]["predicted_reward_return"] != first[2]["predicted_reward_return"]
    assert scorer._predictions(FakeRewardModel(), pixels, rows, 73302 + 50) != first


def test_real_and_predicted_ties_are_separate_denominators():
    rows = [{"real_discounted_raw_return": 0., "predicted_reward_return": 0.},
            {"real_discounted_raw_return": .5e-6, "predicted_reward_return": 1.},
            {"real_discounted_raw_return": 2., "predicted_reward_return": .5e-6},
            {"real_discounted_raw_return": -2., "predicted_reward_return": -3.}]
    pairs = scorer._pairs(rows)
    assert pairs == {"concordant": 3, "discordant": 1, "real_tie": 1,
                     "predicted_tie": 1, "terminal_or_short_excluded": 0}


@pytest.mark.parametrize("targets", [(), (20000, 20000), (40000, 20000), (10001,), (True,)])
def test_invalid_checkpoint_selection_fails_before_any_artifact_load(monkeypatch, targets):
    monkeypatch.setattr(scorer, "_pinned_json", lambda *args: pytest.fail("read unselected artifacts"))
    with pytest.raises(ValueError, match="ascending subset"):
        scorer.score(targets=targets)


def test_all_selected_seals_checked_before_first_torch_load(tmp_path, monkeypatch):
    protocol, receipt, summary = _fixtures()
    (tmp_path / "source.py").write_bytes(b"synthetic source")
    artifacts = {scorer.BRANCH_PROTOCOL: protocol, scorer.BRANCH_RECEIPT: receipt,
                 scorer.BRANCH_SUMMARY: summary}
    monkeypatch.setattr(scorer, "_pinned_json", lambda root, name, expected: copy.deepcopy(artifacts[name]))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("unsealed checkpoint deserialized"))
    attempted = []

    def bind(root, targets):
        for target in targets:
            attempted.append(target)
            if target == 40000:
                raise ValueError("second checkpoint has no complete episode-boundary ledger row")
        pytest.fail("incomplete checkpoint accepted")

    monkeypatch.setattr(scorer, "_bind_long", bind)
    with pytest.raises(ValueError, match="complete episode-boundary"):
        scorer.score(root=tmp_path, targets=(20000, 40000))
    assert attempted == [20000, 40000]


def test_read_only_score_reports_checkpoint_and_v2_on_same_40_synthetic_pairs(tmp_path, monkeypatch):
    protocol, receipt, summary = _fixtures()
    (tmp_path / "source.py").write_bytes(b"synthetic source")
    artifacts = {scorer.BRANCH_PROTOCOL: protocol, scorer.BRANCH_RECEIPT: receipt,
                 scorer.BRANCH_SUMMARY: summary}
    monkeypatch.setattr(scorer, "_pinned_json", lambda root, name, sha: copy.deepcopy(artifacts[name]))
    synthetic = tmp_path / "synthetic.pt"
    synthetic.write_bytes(b"sealed synthetic model")
    pin = {"checkpoint": synthetic, "checkpoint_sha256": _hash(synthetic.read_bytes()),
           "row": {"target": 20000, "decisions": 20001, "updates": 20001, "episodes": 4},
           "line": 50, "ledger_prefix_sha256": "a" * 64, "step_prefix_sha256": "b" * 64}
    monkeypatch.setattr(scorer, "_bind_long", lambda root, targets: (
        {"source_sha256": {"runner.py": "c" * 64}}, [pin]))
    monkeypatch.setattr(scorer, "_checkpoint_model", lambda selected, bound: FakeRewardModel())
    report = scorer.score(root=tmp_path, targets=(20000,))
    assert report["environment_resets"] == report["optimizer_steps"] == 0
    assert report["baseline_v2"]["concordant_on_fixed_informative_pairs"] == {
        "numerator": 25, "denominator": 40}
    row = report["checkpoints"][0]
    assert row["checkpoint_sha256"] == pin["checkpoint_sha256"]
    assert row["pairs"]["real_tie"] == 80
    assert row["concordant_on_fixed_informative_pairs"]["denominator"] == 40
    assert row["concordant_on_fixed_informative_pairs"]["numerator"] == row["pairs"]["concordant"]
    assert row["concordant_on_nonpredicted_ties"]["denominator"] == (
        row["pairs"]["concordant"] + row["pairs"]["discordant"])
    assert len(row["anchors"]) == 12 and len(row["anchors"][0]["candidates"]) == 5


def test_real_pinned_cursor_rejects_unsealed_and_accepts_sealed_prefix_with_live_tail(tmp_path, monkeypatch):
    from scripts import diagnose_tdmpc2_checkpoint_losses as cursor

    monkeypatch.setattr(cursor, "TARGETS", (6,))
    monkeypatch.setattr(scorer, "TARGETS", (6,))
    p = {"source_sha256": {"scripts/train_tdmpc2_long.py":
                            "9121fee37b5beeeaf7f22f253ab0500896e514fa7e5a140a8257483116df9849"},
         "training": {"max_steps": 3, "seed_steps": 3, "pretrain_updates": 3},
         "cells": [{"track_id": 1, "geometry_seed": seed} for seed in scorer.ROADS]}
    monkeypatch.setattr(cursor, "_protocol", lambda root: p)
    real_file = scorer._file
    monkeypatch.setattr(scorer, "_file", lambda root, name: (
        scorer.ROOT / scorer.CURSOR_SOURCE if name == scorer.CURSOR_SOURCE else real_file(root, name)))
    run = tmp_path / scorer.RUN
    run.mkdir(parents=True)
    rows = [{"event": "start", "protocol_sha256": scorer.LONG_PROTOCOL_SHA256,
             "source_sha256": p["source_sha256"], "resume_supported": False}]
    steps = []
    action = np.zeros(3, dtype="<f4").tobytes()
    for episode in range(2):
        cell = p["cells"][episode]
        rows += [{"event": event, "episode": episode, "decisions": episode * 3, **cell}
                 for event in ("reset_intent", "reset")]
        for offset in range(3):
            steps.append({"episode": episode, "decision": episode * 3 + offset + 1,
                          **cell, "action_f32_hex": action.hex(), "native_action_f32_hex": action.hex(),
                          "reward": .25, "terminated": False, "truncated": offset == 2,
                          "terminal": False})
        rows.append({"event": "episode", "episode": episode, "decisions": (episode + 1) * 3,
                     **cell, "updates": (episode + 1) * 3, "length": 3, "return": .75,
                     "terminated": False, "truncated": True, "terminal": False,
                     "action_trace_sha256": _hash(action * 3),
                     "native_action_trace_sha256": _hash(action * 3)})
    line = lambda row: (json.dumps(row, sort_keys=True) + "\n").encode()
    step_bytes = b"".join(map(line, steps))
    (run / "steps.jsonl").write_bytes(step_bytes)
    name = "checkpoint-at-least-000006-step-000006.pt"
    (run / name).write_bytes(b"sealed synthetic checkpoint (no pickle)")
    checkpoint_row = {"event": "checkpoint", "target": 6, "decisions": 6, "updates": 6,
                      "episodes": 2, "path": name, "sha256": _hash((run / name).read_bytes()),
                      "step_ledger_sha256": _hash(step_bytes), "rolling_update_count": 6,
                      "train_episodes_since_previous": [{}, {}]}
    ledger = run / "training.jsonl"
    ledger.write_bytes(b"".join(map(line, rows)))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("unsealed pickle"))
    with pytest.raises(ValueError, match="complete checkpoint ledger cursor"):
        scorer._bind_long(tmp_path, (6,))
    ledger.write_bytes(b"".join(map(line, rows + [checkpoint_row])) + b'{"event":"reset_intent"')
    _, pins = scorer._bind_long(tmp_path, (6,))
    assert pins[0]["row"]["target"] == 6
    assert pins[0]["checkpoint_sha256"] == checkpoint_row["sha256"]
    (run / name).write_bytes(b"partial checkpoint")
    with pytest.raises(ValueError, match="checkpoint SHA"):
        scorer._bind_long(tmp_path, (6,))


def test_unpinned_cursor_helper_refused_without_torch_load(tmp_path, monkeypatch):
    source = tmp_path / scorer.CURSOR_SOURCE
    source.parent.mkdir()
    source.write_bytes(b"not the pinned helper")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("unverified pickle"))
    with pytest.raises(ValueError, match="cursor executable SHA"):
        scorer._bind_long(tmp_path, (20000,))


def test_sealed_checkpoint_metadata_replay_and_sha_precede_model_load(tmp_path, monkeypatch):
    path = tmp_path / "sealed.pt"
    path.write_bytes(b"synthetic, never deserialized")
    pin = {"checkpoint": path, "checkpoint_sha256": "a" * 64,
           "row": {"target": 20000, "decisions": 20001, "updates": 20001, "episodes": 2}}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("bad SHA deserialized"))
    with pytest.raises(ValueError, match="changed before torch.load"):
        scorer._checkpoint_model(pin, {"source_sha256": {}})
    pin["checkpoint_sha256"] = _hash(path.read_bytes())
    state = {"format": "haic-tdmpc2-long-train-v1", "protocol_sha256": scorer.LONG_PROTOCOL_SHA256,
             "source_sha256": {"source.py": "a" * 64}, "resume_supported": False,
             "target": 20000, "decisions": 20001, "updates": 20001, "episodes": 2,
             "action_dim": 3, "learner": {"q_scale": torch.ones(1), "model.weight": torch.ones(1)},
             "optim": {}, "pi_optim": {}, "probe": {}, "rng": {},
             "replay": {"format": "haic-tdmpc2-episode-replay-v1", "active": None,
                        "size": 20001, "next_episode_id": 2, "episodes": [{}, {}],
                        "capacity": 120000, "horizon": 3, "action_dim": 3,
                        "observation_shape": scorer.PIXELS}}
    def load(stream, **kwargs):
        assert stream.read() == b"synthetic, never deserialized"
        assert kwargs == {"map_location": "cpu", "weights_only": False}
        return state
    monkeypatch.setattr(torch, "load", load)
    state["replay"]["active"] = {}
    protocol = {"format": state["format"], "source_sha256": state["source_sha256"]}
    with pytest.raises(ValueError, match="whole-episode"):
        scorer._checkpoint_model(pin, protocol)
    state["replay"]["active"] = None
    class DummyModel:
        def __init__(self, cfg):
            self.cfg = cfg
        def load_state_dict(self, weights, *, strict):
            assert strict and list(weights) == ["weight"]
        def to(self, device):
            assert device == "cpu"
            return self
        def eval(self):
            return self
    monkeypatch.setattr("haic.algorithms.tdmpc2.model.WorldModel", DummyModel)
    assert isinstance(scorer._checkpoint_model(pin, protocol), DummyModel)


def test_cli_help_is_read_only(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["python -m scripts.diagnose_tdmpc2_branch_checkpoint_ranking", "--help"])
    with pytest.raises(SystemExit) as ex:
        scorer.main()
    assert ex.value.code == 0
    assert "--targets" in capsys.readouterr().out
