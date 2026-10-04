"""Synthetic-only Q0 real-branch operator checks. Never reset a real HAIC env."""

import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haic.algorithms.rlpd import prefix_parity as parity
from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner
from scripts import diagnose_tdmpc2_h5_branches as original
from scripts import diagnose_tdmpc2_overshoot_q0_branches as q0
from tests.test_diagnose_tdmpc2_h5_branches import FakeEnv, _mock_parity, _anchor


def _spec(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir(exist_ok=True)
    (tmp_path / "experiments").mkdir(exist_ok=True)
    for path in (q0.SELF, q0.TEST, q0.Q_MODULE):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(path)
    sources = {path: original.digest(tmp_path / path) for path in (q0.SELF, q0.TEST, q0.Q_MODULE)}
    source = {"source_sha256": {}, "runtime": {"synthetic": "1"}, "training_source": {},
              "original_branch": {}}
    episodes = [{"length": 200} for _ in range(8)]
    for ep, length in enumerate(q0.LENGTHS, 4):
        episodes[ep]["length"] = length
    monkeypatch.setattr(q0.score, "preflight", lambda *a: {"spec": source, "episodes": episodes})
    monkeypatch.setattr(q0.branch, "preflight", lambda *a: {"episodes": episodes, "raw": {}})
    monkeypatch.setattr(q0.branch, "runtime_identity", lambda: {"synthetic": "1"})
    spec = {"format": "haic-tdmpc2-overshoot-q0-real-branches-v1",
            "purpose": "consumed-TRAIN-disjoint-RAW-replay-real-five-action-Q0-screen",
            "score_source": q0.ref(q0.score.PROTOCOL, q0.SCORE_PROTOCOL_SHA),
            "old_primary": q0.ref(q0.score.OLD_PRIMARY, q0.score.OLD_PRIMARY_SHA),
            "source_sha256": sources, "runtime": source["runtime"],
            "anchors": [{"episode_id": ep, "start_step": step} for ep, step in q0.ANCHORS],
            "episode_lengths": list(q0.LENGTHS), "candidates": {"logged": None, **original.FIXED_SUFFIXES},
            "environment": original.ENVIRONMENT, "horizons": [3, 5],
            "discount": .995, "tie_tolerance": 1e-6, "augmentation_seed": 834,
            "bootstrap_offset": q0.BOOTSTRAP_OFFSET, "q_weights": {"q0": 0, "q1": 1},
            "fixed_q_pairs": [list(pair) for pair in q0.PAIRS], "max_resets": 72,
            "quality_gate": q0.GATE, "output": "runs/tdmpc2-overshoot-q0-branches-synthetic.json",
            "policy_release": False}
    path = tmp_path / q0.PROTOCOL
    path.write_text(json.dumps(spec))
    return spec, path, original.digest(path)


@pytest.mark.parametrize("field", ["anchors", "episode_lengths", "candidates", "max_resets",
                                    "quality_gate", "old_primary", "source_sha256", "output"])
def test_preflight_fails_closed_on_resealed_protocol_changes(tmp_path, monkeypatch, field):
    spec, path, sha = _spec(tmp_path, monkeypatch)
    assert q0.preflight(tmp_path, sha)["spec"] == spec
    changed = copy.deepcopy(spec)
    if field == "anchors":
        changed[field][0]["episode_id"] = 0
    elif field == "episode_lengths":
        changed[field][0] = 1
    elif field == "candidates":
        changed[field]["coast"] = [[0, 0, 0]] * 5
    elif field == "max_resets":
        changed[field] = 73
    elif field == "quality_gate":
        changed[field]["min_q0_concordance"] = .69
    elif field == "old_primary":
        changed[field]["sha256"] = "a" * 64
    elif field == "source_sha256":
        changed[field][q0.Q_MODULE] = "b" * 64
    else:
        changed[field] = "runs/../runs/tdmpc2-overshoot-q0-branches-bad.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        q0.preflight(tmp_path, original.digest(path))


def test_preflight_protocol_only_never_deserializes_or_resets(tmp_path, monkeypatch):
    _, _, sha = _spec(tmp_path, monkeypatch)
    monkeypatch.setattr(q0.torch, "load", lambda *a, **k: pytest.fail("deserialized"))
    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(build_env=lambda *a, **k: pytest.fail("env")))
    assert q0.run(tmp_path, sha) == {"status": "preflight_only", "environment_resets_attempted": 0,
                                    "optimizer_updates": 0, "protocol_sha256": sha, "policy_release": False}
    assert not (tmp_path / "runs/tdmpc2-overshoot-q0-branches-synthetic.json").exists()


def test_source_byte_tamper_blocks_even_resealed_protocol(tmp_path, monkeypatch):
    spec, path, _ = _spec(tmp_path, monkeypatch)
    (tmp_path / q0.Q_MODULE).write_text("changed sibling source bytes")
    with pytest.raises(ValueError, match="source/runtime"):
        q0.preflight(tmp_path, original.digest(path))
    spec["source_sha256"][q0.Q_MODULE] = original.digest(tmp_path / q0.Q_MODULE)
    path.write_text(json.dumps(spec))
    assert q0.preflight(tmp_path, original.digest(path))["spec"] == spec


def test_recheck_rehashes_original_RAW_artifacts_before_each_reset_and_completion(tmp_path, monkeypatch):
    spec, _, sha = _spec(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(q0.branch, "_recheck", lambda *args, artifacts=False: calls.append(artifacts))
    bound = {"spec": spec, "score": {"spec": {"training_source": {}, "original_branch": {}}},
             "raw": {"spec": {"source": {}}}, "protocol_sha256": sha}
    q0.recheck(tmp_path, bound)
    q0.recheck(tmp_path, bound)
    assert calls == [True, True]
    monkeypatch.setattr(q0.branch, "_recheck", lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("RAW ledger changed")))
    with pytest.raises(ValueError, match="RAW ledger changed"):
        q0.recheck(tmp_path, bound)


def test_disjoint_replay_binds_exact_lengths_roads_rewards_and_step_cursor(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    lengths = [110] * 4 + list(q0.LENGTHS)
    monkeypatch.setattr(q0.branch, "FINAL", {**original.FINAL, "decisions": sum(lengths),
                                           "updates": sum(lengths), "episodes": 8})
    replay, ledgers, steps = [], [], []
    decision = 0
    pixels = np.zeros(original.PIXELS, np.uint8)
    for ep, length in enumerate(lengths):
        actions = np.broadcast_to(np.array([0., -1., 0.], np.float32), (length, 3))
        observations = np.broadcast_to(pixels, (length + 1, *original.PIXELS))
        flags = np.zeros(length, np.bool_)
        flags[-1] = True
        native = b"".join(original.environment_action(action).tobytes() for action in actions)
        for offset in range(length):
            decision += 1
            steps.append({"episode": ep, "decision": decision, "track_id": 1,
                          "geometry_seed": original.ROADS[ep % 4],
                          "action_f32_hex": actions[offset].tobytes().hex(),
                          "native_action_f32_hex": original.environment_action(actions[offset]).tobytes().hex(),
                          "reward": .25, "terminated": bool(flags[offset]),
                          "truncated": False, "terminal": bool(flags[offset])})
        replay.append({"episode_id": ep, "start_step": 0, "actions": actions,
                       "observations": observations, "rewards": np.full(length, .25, np.float32),
                       "terminated": flags.copy(), "truncated": np.zeros(length, np.bool_),
                       "terminal": flags.copy()})
        ledgers.append({"length": length, "geometry_seed": original.ROADS[ep % 4],
                        "terminated": True, "truncated": False, "terminal": True,
                        "return": length * .25, "decisions": decision,
                        "action_trace_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                        "native_action_trace_sha256": hashlib.sha256(native).hexdigest()})
    step_path = tmp_path / original.RUN / "steps.jsonl"
    step_path.parent.mkdir(parents=True)
    step_path.write_text("".join(json.dumps(row) + "\n" for row in steps))
    bound = {"raw": {"raw": {"format": "raw", "source_sha256": {}}, "episodes": ledgers}}
    state = {"format": "raw", "protocol_sha256": original.RAW_PROTOCOL_SHA,
             "source_sha256": {}, "learner": {"q_scale": torch.tensor(1.)}, **original.FINAL,
             "replay": {"format": "haic-tdmpc2-episode-replay-v1", "capacity": 120000,
                        "horizon": 3, "action_dim": 3, "augmentation_pad": 3,
                        "include_partial": False, "bootstrap_on_truncation": True,
                        "observation_shape": original.PIXELS, "next_episode_id": 8,
                        "size": decision, "active": None, "episodes": replay}}
    anchors = q0.bind_anchors(state, bound, tmp_path)
    assert [(a.episode_id, a.step) for a in anchors] == list(q0.ANCHORS)
    assert all(a.rewards == (.25,) * a.step and a.logged_rewards == (.25,) * 5 for a in anchors)
    assert [a.seed for a in anchors[::3]] == list(original.ROADS)
    ledgers[4]["length"] += 1
    with pytest.raises(ValueError, match="length/road|shape|cursor"):
        q0.bind_anchors(state, bound, tmp_path)
    ledgers[4]["length"] -= 1
    steps[0]["decision"] = -1
    step_path.write_text("".join(json.dumps(row) + "\n" for row in steps))
    with pytest.raises(ValueError, match="lineage"):
        q0.bind_anchors(state, bound, tmp_path)


def _entries(*, informative_anchors=12, q0_order=(0, 1, 2, 3, 4)):
    entries = []
    for index, (ep, start) in enumerate(q0.ANCHORS):
        actual = list(range(5)) if index in range(0, informative_anchors * 3, 3) else [0] * 5
        rows = []
        for name, value in zip(q0.NAMES, actual):
            raw = [value / sum(.995**t for t in range(5))] * 5
            rows.append({"candidate": name, "steps": 5, "raw_rewards": raw, "full_h5": True,
                         "ranking_eligible": True, "ending": None,
                         "real_discounted_raw_return": value})
        scores = {"h5_q0": dict(zip(q0.NAMES, q0_order)),
                  "h3_q0": dict(zip(q0.NAMES, reversed(range(5))))}
        scores.update({f"h5_q1_{a}_{b}": dict(zip(q0.NAMES, reversed(range(5)))) for a, b in q0.PAIRS})
        entries.append({"episode_id": ep, "start_step": start, "geometry_seed": original.ROADS[ep % 4],
                        "candidates": rows, "scores": scores})
    return entries


def test_gate_exact_40_pairs_70_percent_and_all_ten_pair_comparisons():
    entries = _entries(informative_anchors=4, q0_order=(2, 1, 0, 3, 4))
    summary = q0.aggregate(entries)
    assert summary["informative_real_pairs"] == 40
    assert summary["informative_roads"] == 4
    assert summary["scores"]["h5_q0"]["pairs"]["concordant"] == 28
    assert summary["scores"]["h5_q0"]["concordance"] == .7
    assert summary["quality_gate_passed"]
    assert len(summary["scores"]) == 12
    for changed in ("concordance", "q1_tie", "q1_regret", "regret", "h3", "road"):
        variant = copy.deepcopy(entries)
        if changed == "concordance":
            variant[0]["scores"]["h5_q0"] = dict(zip(q0.NAMES, (3, 2, 1, 0, 4)))
        elif changed == "q1_tie":
            for index in (0, 3, 6, 9):
                variant[index]["scores"]["h5_q1_3_4"] = variant[index]["scores"]["h5_q0"].copy()
        elif changed == "q1_regret":
            for index in (0, 3, 6, 9):
                variant[index]["scores"]["h5_q1_3_4"] = dict(zip(q0.NAMES, (3, 2, 1, 0, 4)))
        elif changed == "regret":
            for entry in variant:
                entry["scores"]["h5_q0"] = dict(zip(q0.NAMES, (2, 1, 0, 4, 3)))
        elif changed == "h3":
            for entry in variant:
                entry["scores"]["h3_q0"] = dict(zip(q0.NAMES, range(5)))
        else:
            for index in (3, 6, 9):
                variant[index]["geometry_seed"] = variant[0]["geometry_seed"]
        assert not q0.aggregate(variant)["quality_gate_passed"], changed


def test_h5_q0_regret_must_not_exceed_h3_on_same_real_h5_outcomes():
    entries = _entries(informative_anchors=4, q0_order=(0, 1, 2, 4, 3))
    for entry in entries:
        entry["scores"]["h3_q0"] = dict(zip(q0.NAMES, (3, 2, 1, 0, 4)))
    summary = q0.aggregate(entries)
    assert summary["scores"]["h5_q0"]["concordance"] > summary["scores"]["h3_q0"]["concordance"]
    assert summary["scores"]["h5_q0"]["mean_real_h5_regret"] > summary["scores"]["h3_q0"]["mean_real_h5_regret"]
    assert not summary["quality_gate_passed"]


def test_short_endings_are_preserved_but_not_promoted_as_gate():
    entries = _entries()
    entries[0]["candidates"][0]["steps"] = 3
    entries[0]["candidates"][0]["ending"] = "terminated"
    entries[0]["candidates"][0]["ranking_eligible"] = False
    with pytest.raises(ValueError, match="sixty real five-step"):
        q0.aggregate(entries)


class FakeModel:
    cfg = SimpleNamespace(num_bins=101, vmin=-10., vmax=10.)

    def encode(self, obs, task):
        return torch.ones((1, 2))

    def reward(self, z, action, task):
        return torch.zeros((len(z), 101))

    def next(self, z, action, task):
        return z + action[:, :2] * .01

    def termination(self, z, task):
        return torch.zeros((len(z), 1))

    def pi(self, z, task):
        return torch.randn((len(z), 3)).tanh(), {}

    def Q(self, z, action, task, *, return_type):
        assert return_type == "all"
        return torch.stack([torch.ones((len(z), 101)) * i for i in range(5)])


def test_q0_consumes_matching_bootstrap_draws_and_matches_all_fixed_pairs(monkeypatch):
    model = FakeModel()
    actions = {name: np.full((5, 3), .2, np.float32) for name in q0.NAMES}
    pixels = np.zeros(original.PIXELS, np.uint8)
    outcomes = q0.model_scores(model, pixels, actions, 33)
    assert len(outcomes) == 12
    for h in (3, 5):
        cfg = PlannerConfig(action_dim=3, discount=.995, horizon=h, episodic=True,
                            num_samples=5, num_elites=1, num_pi_trajs=0, num_bins=101)
        seq = torch.from_numpy(np.stack([actions[name][:h] for name in q0.NAMES], axis=1))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(33 + q0.BOOTSTRAP_OFFSET + h)
            before = torch.random.get_rng_state().clone()
            weighted = q0.QWeightedPlanner(q0.fixed._FixedPair(model, (0, 1)), cfg, q_weight=1)
            one = weighted._estimate_value(torch.ones((5, 2)), seq)
            after = torch.random.get_rng_state().clone()
            torch.random.set_rng_state(before)
            zero = q0.QWeightedPlanner(q0.fixed._FixedPair(model, (0, 1)), cfg, q_weight=0)
            other = zero._estimate_value(torch.ones((5, 2)), seq)
            assert torch.equal(after, torch.random.get_rng_state())
            torch.random.set_rng_state(before)
            baseline = TDMPC2Planner(q0.fixed._FixedPair(model, (0, 1)), cfg)._estimate_value(
                torch.ones((5, 2)), seq)
            torch.testing.assert_close(one, baseline)
            assert torch.allclose(other, torch.zeros_like(other), atol=1e-5)
            assert list(outcomes[f"h{h}_q0"].values()) == pytest.approx(other[:, 0].tolist())


def _synthetic_run(tmp_path, monkeypatch, *, fail_at=None):
    spec, _, sha = _spec(tmp_path, monkeypatch)
    anchors = []
    for ep, start in q0.ANCHORS:
        template = _anchor(ep % 4, start)
        anchors.append(original.BoundAnchor(ep, template.track_id, template.seed, template.step,
                                           template.observations, template.actions, template.rewards,
                                           template.logged_actions, template.logged_observations,
                                           template.logged_rewards, template.logged_flags))
    anchors = tuple(anchors)
    monkeypatch.setattr(q0, "preflight", lambda *a: {"spec": spec, "score": {"spec": {
        "original_branch": {"model": {"path": "dummy", "sha256": "a" * 64}}}},
        "protocol_sha256": sha})
    monkeypatch.setattr(q0, "recheck", lambda *a, **kw: None)
    monkeypatch.setattr(q0.score, "_models", lambda *a: (None, FakeModel()))
    monkeypatch.setattr(q0.branch, "digest_stream", lambda *a: "a" * 64)
    monkeypatch.setattr(q0.torch, "load", lambda *a, **k: {})
    monkeypatch.setattr(q0, "bind_anchors", lambda *a: anchors)
    monkeypatch.setattr(q0, "model_scores", lambda *a: {
        "h3_q0": dict(zip(q0.NAMES, range(5))), "h5_q0": dict(zip(q0.NAMES, range(5))),
        **{f"h5_q1_{a}_{b}": dict(zip(q0.NAMES, reversed(range(5)))) for a, b in q0.PAIRS}})
    (tmp_path / "dummy").write_bytes(b"checkpoint")
    _mock_parity(monkeypatch)
    instances = []
    def factory(*args, **kwargs):
        path = tmp_path / spec["output"]
        saved = json.loads(path.read_text())
        assert saved["reset_intents"][-1]["ordinal"] == len(instances) + 1
        assert saved["environment_resets_attempted"] == len(instances) + 1
        env = FakeEnv(args[1], end_at=18 if fail_at == len(instances) else None)
        instances.append(env)
        return env
    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(build_env=factory))
    return sha, instances, tmp_path / spec["output"]


def test_synthetic_72_resets_prefix_parity_and_exclusive_receipt(tmp_path, monkeypatch):
    sha, instances, path = _synthetic_run(tmp_path, monkeypatch)
    report = q0.run(tmp_path, sha, execute=True)
    assert len(instances) == 72 and all(env.closed for env in instances)
    assert report["environment_resets_attempted"] == 72
    assert report["replay_bound_before_first_reset"] is True
    assert all(len(entry["candidates"]) == 5 for entry in report["anchors"])
    assert report["body_sha256"] == q0.body_sha({k: v for k, v in report.items() if k != "body_sha256"})
    with pytest.raises(ValueError, match="exclusive"):
        q0.run(tmp_path, sha, execute=True)
    assert json.loads(path.read_text()) == report


def test_parity_failure_preserves_partial_intent_before_second_reset(tmp_path, monkeypatch):
    sha, instances, path = _synthetic_run(tmp_path, monkeypatch, fail_at=1)
    with pytest.raises(parity.ParityError):
        q0.run(tmp_path, sha, execute=True)
    partial = json.loads(path.read_text())
    assert partial["status"] == "stopped_partial_no_resume"
    assert partial["environment_resets_attempted"] == 2
    assert len(partial["reset_intents"]) == 2
    assert len(instances) == 2 and all(env.closed for env in instances)
    assert partial["body_sha256"] == q0.body_sha({k: v for k, v in partial.items() if k != "body_sha256"})
