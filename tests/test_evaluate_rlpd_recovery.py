"""Synthetic evaluator gates; no environment construction or resets."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import evaluate_rlpd_recovery as evaluator


def trace(archived=False):
    values = {
        "policy_or_oracle_action": np.tile([-.7, .5, 0], (5, 1)),
        "oracle_action_at_state": np.tile([.5, .1, .2], (5, 1)),
        "pre_speed": np.full(5, 15.), "center_error": np.full(5, 7.),
        "heading_error": np.full(5, .6), "curvature": np.full(5, .03),
    }
    if archived:
        return {"commanded_official_action": values["policy_or_oracle_action"],
                "speed_m_s": values["pre_speed"], "centerline_distance_m": values["center_error"],
                "heading_error_rad": values["heading_error"]}
    return values


def test_archive_never_invents_oracle_signals():
    result = evaluator.events(trace(True), archived=True)
    assert set(result["unassessed"]) == {"curve-entry-overspeed", "steering-opposition"}
    assert "curve-entry-overspeed" not in result["first_events"]
    assert result["state_timing"] == "post-decision"
    assert result["lateral_signed"] is False


def test_all_event_types_in_success_controls():
    result = evaluator.events(trace())
    assert result["first_events"]["curve-entry-overspeed"]["decision_index_zero_based"] == 0
    assert result["first_events"]["steering-opposition"]["decision_index_zero_based"] == 0
    counts = evaluator.census([{"finished": True, "events": result}])
    assert counts["successful_controls"]["event_episode_counts"]["heading-error"] == 1
    assert counts["nonfinishes"]["denominator"] == 0


def test_sustained_events_require_consecutive_decisions():
    values = trace()
    values["heading_error"] = np.array([.6, 0, .6, .6, .6])
    values["center_error"] = np.array([7, 0, 7, 0, 7])
    result = evaluator.events(values)
    assert result["first_events"]["heading-error"]["decision_index_zero_based"] == 2
    assert result["first_events"]["lateral-excursion"] is None


@pytest.mark.parametrize("key", ["pre_speed", "oracle_action_at_state", "curvature"])
def test_nonfinite_telemetry_rejected(key):
    values = trace()
    values[key].flat[0] = np.nan
    with pytest.raises(ValueError):
        evaluator.events(values)


def test_event_fields_aligned():
    values = trace()
    values["center_error"] = np.zeros(4)
    with pytest.raises(ValueError, match="misaligned"):
        evaluator.events(values)


def test_label_validation(tmp_path):
    actor = tmp_path / "actor.pt"
    actor.touch()
    assert evaluator.parse_actors(["a=actor.pt"], tmp_path) == {"a": actor}
    for values in (["a=actor.pt", "a=actor.pt"], ["../a=actor.pt"], ["a=missing"], []):
        with pytest.raises(ValueError):
            evaluator.parse_actors(values, tmp_path)


class SyntheticActor:
    def reset(self, observation):
        assert observation.shape == (4, 84, 84)

    def act(self, observation):
        return np.array([0, .4, .1], np.float32)


def test_double_reload_zero_reset(tmp_path, monkeypatch):
    actor = tmp_path / "actor.pt"
    torch.save({"format": "haic-rlpd-pixel-actor-v1"}, actor)
    monkeypatch.setattr(evaluator.diagnosis, "_make_env", lambda *args: pytest.fail("reset forbidden"))
    result = evaluator.reload_preflight(actor, lambda path: SyntheticActor())
    assert result["double_reload_bit_exact"]
    assert result["environment_resets"] == 0


def test_double_reload_mismatch_rejected(tmp_path):
    actor = tmp_path / "actor.pt"
    torch.save({"format": "haic-rlpd-pixel-actor-v1"}, actor)
    class Different(SyntheticActor):
        def act(self, observation):
            return np.array([.1, .4, .1], np.float32)
    made = []
    def factory(path):
        made.append(path)
        return SyntheticActor() if len(made) == 1 else Different()
    with pytest.raises(ValueError, match="reload"):
        evaluator.reload_preflight(actor, factory)


def rows():
    return [{"track_id": 1, "geometry_seed": seed, "finished": False,
             "road_centerline_sha256": str(seed)} for seed in evaluator.SEEDS]


def test_paired_finish_transition_and_censor():
    left, right = rows(), rows()
    left[0]["finished"] = right[0]["finished"] = True
    left[1]["finished"] = True
    right[2]["finished"] = True
    result = evaluator.transition_table(left, right)
    assert [result[k] for k in ("kept", "lost", "gained", "neither")] == [1, 1, 1, 9]
    right[3]["censored"] = True
    censored = evaluator.transition_table(left, right)
    assert censored["full_finish_comparison_valid"] is False
    assert censored["neither"] == 8
    assert evaluator.census([{"finished": False, "censored": True, "events": evaluator.events(trace())}])["nonfinishes"]["denominator"] == 0
    with pytest.raises(ValueError):
        evaluator.transition_table(left, right[:-1])
    right[0]["road_centerline_sha256"] = "changed"
    with pytest.raises(ValueError, match="road"):
        evaluator.transition_table(left, right)


def fake_inputs(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    actor = tmp_path / "actor.pt"
    actor.touch()
    baseline = [{**r, "actor_id": "entropy-v5-author-seed50", "actor_sha256": evaluator.V5_SHA} for r in rows()]
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("\n".join(json.dumps(r) for r in rows()))
    monkeypatch.setattr(evaluator, "G0", "ledger.jsonl")
    monkeypatch.setattr(evaluator, "G0_SHA", evaluator.sha(ledger))
    monkeypatch.setattr(evaluator, "archived_screen", lambda root: (baseline, {}))
    monkeypatch.setattr(evaluator, "source_hashes", lambda root: {})
    monkeypatch.setattr(evaluator, "runtime", lambda: {"synthetic": True})
    monkeypatch.setattr(evaluator, "reload_preflight", lambda path: {"environment_resets": 0})
    return {"a": actor}


def test_preflight_freezes_and_never_resets(tmp_path, monkeypatch):
    actors = fake_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluator.diagnosis, "_collect_episode", lambda *args: pytest.fail("reset forbidden"))
    result = evaluator.evaluate(actors, "runs/newslug", root=tmp_path, preflight_only=True)
    assert result["environment_resets"] == 0
    protocol = json.loads((tmp_path / "runs/newslug/protocol.json").read_text())
    assert protocol["contract"] == evaluator.CONTRACT
    manifest = json.loads((tmp_path / "runs/newslug/manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        assert evaluator.sha(tmp_path / "runs/newslug" / name) == digest
    with pytest.raises(ValueError, match="new"):
        evaluator.evaluate(actors, "runs/newslug", root=tmp_path, preflight_only=True)


def test_full_arm_synthetic_runner_and_receipts(tmp_path, monkeypatch):
    import agent
    actors = fake_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(agent, "Agent", lambda **kwargs: SyntheticActor())
    calls = []
    def collect(case, mode, actor):
        calls.append(case)
        assert mode == "policy" and case["track_id"] == 1 and case["max_steps"] == 2000
        assert (tmp_path / f"runs/newslug/a-seed-{case['geometry_seed']}-reset-intent.json").is_file()
        return {"track_id": 1, "geometry_seed": case["geometry_seed"], "steps": 5,
                "terminated": True, "truncated": False, "finished": True,
                "initial_observation_sha256": "fixture", "road_centerline_sha256": str(case["geometry_seed"]),
                "total_reward": 12.34}, trace()
    monkeypatch.setattr(evaluator.diagnosis, "_collect_episode", collect)
    result = evaluator.evaluate(actors, "runs/newslug", root=tmp_path)
    assert len(calls) == 12
    assert [c["geometry_seed"] for c in calls] == list(evaluator.SEEDS)
    assert result["per_actor"]["a"]["finish_count"] == 12
    assert all(r["total_reward"] == 12.34 for r in result["episodes"])


def test_actor_drift_halts_before_first_reset(tmp_path, monkeypatch):
    actors = fake_inputs(tmp_path, monkeypatch)
    def preflight(path):
        path.write_bytes(b"changed")
        return {}
    monkeypatch.setattr(evaluator, "reload_preflight", preflight)
    monkeypatch.setattr(evaluator.diagnosis, "_collect_episode", lambda *args: pytest.fail("reset forbidden"))
    with pytest.raises(ValueError, match="actor changed"):
        evaluator.evaluate(actors, "runs/newslug", root=tmp_path)


def test_failed_attempt_preserves_reset_intent_and_hashed_receipt(tmp_path, monkeypatch):
    import agent
    actors = fake_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(agent, "Agent", lambda **kwargs: SyntheticActor())
    def fail(*args):
        raise RuntimeError("synthetic rollout failure")
    monkeypatch.setattr(evaluator.diagnosis, "_collect_episode", fail)
    with pytest.raises(RuntimeError):
        evaluator.evaluate(actors, "runs/newslug", root=tmp_path)
    manifest = json.loads((tmp_path / "runs/newslug/manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "failure.json" in manifest["files_sha256"]
    assert any("reset-intent" in name for name in manifest["files_sha256"])


@pytest.mark.skipif(not (evaluator.ROOT / evaluator.SCREEN / "manifest.json").is_file(),
                    reason="optional frozen run artifacts are not installed")
def test_requested_archived_review_is_zero_reset(monkeypatch):
    monkeypatch.setattr(evaluator.diagnosis, "_make_env", lambda *args: pytest.fail("reset forbidden"))
    result = evaluator.review()
    assert result["environment_resets"] == 0
    assert result["episodes"] == 24
    assert len(result["source_artifacts_sha256"]) == 49
    assert result["per_actor"]["entropy-v5-author-seed50"]["successful_controls"]["denominator"] == 5
    assert result["per_actor"]["newhost-seed52-final"]["successful_controls"]["denominator"] == 1
