"""Fake loaders/collectors only: no real actor artifacts or environment resets."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import evaluate_rlpd_local_recovery_gate as evaluator
from scripts import summarize_rlpd_recovery_evaluation as summarizer


class Source:
    def reset(self, obs):
        self.calls = 0

    def act(self, obs):
        assert obs.shape == (4, 84, 84)
        self.calls += 1
        return np.array([0, .4, .1], np.float32)


class Gate:
    def __init__(self, trigger_at=None):
        self.trigger_at = trigger_at
        self.reset()

    def reset(self):
        self.calls = self.triggers = self.remaining = 0

    def act(self, obs):
        assert obs.shape == (4, 84, 84)
        trigger = self.calls == self.trigger_at
        if trigger:
            self.triggers += 1
            self.remaining = 12
        active = self.remaining > 0
        if active:
            self.remaining -= 1
        source = np.array([0, .4, .1], np.float32)
        corrected = np.array([.2, .1, .4], np.float32)
        self.diagnostics = {"active": active, "triggers": self.triggers, "remaining": self.remaining,
                            "prototype_id": 0 if active else None, "distance": .1,
                            "source_action": source, "corrected_action": corrected}
        self.calls += 1
        return corrected if active else source


def make(kind, path, digest):
    return evaluator.AuditActor(Source(), False) if kind == "original" else evaluator.AuditActor(Gate(), True)


def inputs(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    source, corrected, gate = (tmp_path / name for name in ("source.pt", "corrected.pt", "gate.pt"))
    for p in (source, corrected, gate):
        p.write_bytes(p.name.encode())
    own = tmp_path / "scripts/evaluate_rlpd_local_recovery_gate.py"
    own.parent.mkdir()
    own.write_text("synthetic evaluator source")
    rows = [{"track_id": 1, "geometry_seed": seed, "finished": False,
             "road_centerline_sha256": str(seed), "actor_sha256": evaluator.original.sha(source)}
            for seed in evaluator.original.SEEDS]
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr(evaluator.original, "G0", ledger.name)
    monkeypatch.setattr(evaluator.original, "G0_SHA", evaluator.original.sha(ledger))
    monkeypatch.setattr(evaluator.original, "V5_SHA", evaluator.original.sha(source))
    monkeypatch.setattr(evaluator.original, "archived_screen", lambda root: (rows, {}))
    monkeypatch.setattr(evaluator.original, "runtime", lambda: {"synthetic": True})
    monkeypatch.setattr(evaluator.original, "source_hashes", lambda root: {})
    metadata = {"source_actor": {"path": source.name, "sha256": evaluator.original.sha(source)},
                "corrected_actor": {"path": corrected.name, "sha256": evaluator.original.sha(corrected)},
                "sources_sha256": {}, "evidence_sha256": {},
                "policy": {"hold_decisions": 12, "radius_fraction": .5},
                "calibration": {"protected_triggers": 0}}
    monkeypatch.setattr(evaluator, "inspect_gate", lambda *args: metadata)
    return gate, evaluator.original.sha(gate)


def collect(case, mode, actor, *, n=5, censored=False, obs_hash="fixture"):
    assert mode == "policy" and case["track_id"] == 1 and case["max_steps"] == 2000
    obs = np.zeros((4, 84, 84), np.float32)
    actor.reset(obs)
    actions = np.stack([actor.act(obs) for _ in range(n)])
    trace = {"step": np.arange(n), "policy_or_oracle_action": actions,
             "oracle_action_at_state": np.tile([.5, .1, .2], (n, 1)),
             "pre_speed": np.full(n, 15.), "center_error": np.full(n, 7.),
             "pre_position": np.zeros((n, 2)), "pre_velocity": np.zeros((n, 2)),
             "pre_heading": np.zeros(n),
             "heading_error": np.full(n, .6), "curvature": np.full(n, .03),
             "damage": np.zeros(n), "progress": np.linspace(0, .5, n),
             "off_track_counter": np.zeros(n), "collision": np.zeros(n),
             "reward": np.full(n, 2.)}
    summary = {"track_id": 1, "geometry_seed": case["geometry_seed"], "steps": n,
               "finished": not censored, "terminated": not censored, "truncated": False,
               "reason": "runner_max_steps" if censored else "finished", "damage": 0.,
               "progress": .5, "total_reward": 2. * n, "initial_observation_sha256": obs_hash,
               "road_centerline_sha256": str(case["geometry_seed"])}
    return summary, trace


def test_preflight_two_reloads_source_exact():
    actors = {kind: {"path": "fake", "sha256": "a" * 64} for kind in evaluator.ARMS}
    result = evaluator.preflight(actors, make)
    assert result["environment_resets"] == 0
    assert result["initial_and_outside_source_exact"]


def test_preflight_rejects_triggered_synthetic_initial():
    def bad(kind, path, digest):
        return evaluator.AuditActor(Gate(0), True) if kind == "local-gate" else make(kind, path, digest)
    with pytest.raises(ValueError, match="initial synthetic"):
        evaluator.preflight({kind: {"path": "fake", "sha256": "a" * 64} for kind in evaluator.ARMS}, bad)


def test_proxy_twelve_hold_then_fallback_reset_no_crossroad():
    proxy = evaluator.AuditActor(Gate(1), True)
    obs = np.zeros((4, 84, 84), np.float32)
    proxy.reset(obs)
    for _ in range(15):
        proxy.act(obs)
    assert [r["gate_active"] for r in proxy.audit] == [False] + [True] * 12 + [False] * 2
    assert sum(r["gate_trigger"] for r in proxy.audit) == 1
    assert [r["gate_holdcounter"] for r in proxy.audit[1:13]] == list(range(11, -1, -1))
    proxy.reset(obs)
    proxy.act(obs)
    assert len(proxy.audit) == 1 and not proxy.audit[0]["gate_active"]


def test_proxy_disagreement_and_extension_fail_closed():
    class Wrong(Gate):
        def act(self, obs):
            result = super().act(obs)
            self.diagnostics["source_action"] = np.array([1, 1, 1], np.float32)
            return result
    proxy = evaluator.AuditActor(Wrong(), True)
    with pytest.raises(ValueError, match="audit disagree"):
        proxy.act(np.zeros((4, 84, 84), np.float32))
    class Extended(Gate):
        def act(self, obs):
            result = super().act(obs)
            if self.calls == 2:
                self.diagnostics["remaining"] = 11
            return result
    proxy = evaluator.AuditActor(Extended(0), True)
    proxy.act(np.zeros((4, 84, 84), np.float32))
    with pytest.raises(ValueError, match="extended"):
        proxy.act(np.zeros((4, 84, 84), np.float32))


@pytest.mark.parametrize("value", [np.zeros(3, np.float64), np.full(3, np.nan, np.float32),
                                  np.array([0, -1, 0], np.float32), np.zeros(2, np.float32)])
def test_bad_action_rejected(value):
    with pytest.raises(ValueError, match="float32"):
        evaluator.action(value)


def test_zero_reset_freeze_exclusive_and_manifest(tmp_path, monkeypatch):
    gate, digest = inputs(tmp_path, monkeypatch)
    def forbidden(*args):
        pytest.fail("environment reset forbidden")
    result = evaluator.evaluate(gate, digest, "runs/preflight", preflight_only=True,
                                root=tmp_path, make=make, collect=forbidden)
    assert result["environment_resets"] == 0
    directory = tmp_path / "runs/preflight"
    protocol = json.loads((directory / "protocol.json").read_text())
    assert protocol["contract"] == evaluator.original.CONTRACT
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["reset_intents"] == 0
    for name, expected in manifest["files_sha256"].items():
        assert evaluator.original.sha(directory / name) == expected
    with pytest.raises(ValueError, match="new"):
        evaluator.evaluate(gate, digest, "runs/preflight", root=tmp_path, make=make, collect=forbidden)


def test_full_24_episodes_factory_per_road_and_existing_summarizer(tmp_path, monkeypatch):
    gate, digest = inputs(tmp_path, monkeypatch)
    loaded, calls = [], []
    def fresh(kind, path, sha):
        actor = make(kind, path, sha)
        loaded.append(actor)
        return actor
    def collector(case, mode, actor):
        kind = "local-gate" if actor.gated else "original"
        assert (tmp_path / f"runs/full/{kind}-seed-{case['geometry_seed']}-reset-intent.json").is_file()
        calls.append((kind, case["geometry_seed"]))
        return collect(case, mode, actor)
    result = evaluator.evaluate(gate, digest, "runs/full", root=tmp_path, make=fresh, collect=collector)
    assert len(loaded) == 28 and len({id(x) for x in loaded}) == 28
    assert calls == [(kind, seed) for kind in evaluator.ARMS for seed in evaluator.original.SEEDS]
    assert result["environment_resets"] == 24
    assert result["contemporaneous_pairs"]["original->local-gate"]["kept"] == 12
    for kind in evaluator.ARMS:
        assert result["per_actor"][kind]["finish_count"] == 12
        assert result["per_actor"][kind]["guard_active_fraction"] == 0
    audited = summarizer.summarize(tmp_path / "runs/full")
    assert audited["episodes"] == 24
    with np.load(tmp_path / "runs/full/local-gate-seed-4272000001.npz") as trace:
        assert trace["gate_trigger"].shape == (5,)
        assert trace["gate_source_action"].shape == (5, 3)


def test_closed_loop_followup_primary_trace_metrics():
    proxy = evaluator.AuditActor(Gate(1), True)
    summary, trace = collect({"track_id": 1, "geometry_seed": 1, "max_steps": 2000}, "policy", proxy, n=66)
    trace = proxy.merge(trace)
    result = evaluator.telemetry(trace)
    assert result["active_decisions"] == 12 and result["trigger_count"] == 1
    window = result["trigger_followups"][0]
    assert window["complete_5s_window"] and window["end_index"] == 64
    assert window["elapsed_seconds"] == 5.04
    assert window["source_action"] != window["switched_action"]
    assert result["raw_reward_sum"] == summary["total_reward"]
    assert not result["independent_active_episodes"]
    assert window["handoff_index"] == 13 and not window["complete_post_hold_5s_window"]
    assert window["post_hold_later_index"] == 65


def test_censored_pairs_excluded(tmp_path, monkeypatch):
    gate, digest = inputs(tmp_path, monkeypatch)
    def collector(case, mode, actor):
        capped = actor.gated and case["geometry_seed"] == evaluator.original.SEEDS[0]
        return collect(case, mode, actor, n=2000 if capped else 5, censored=capped)
    result = evaluator.evaluate(gate, digest, "runs/censored", root=tmp_path, make=make, collect=collector)
    table = result["contemporaneous_pairs"]["original->local-gate"]
    assert table["kept"] == 11 and table["lost"] == 0 and table["censored_pairs"] == 1
    assert not table["full_finish_comparison_valid"]


@pytest.mark.parametrize("drift", ["gate", "source", "corrected", "protocol", "evaluator"])
def test_drift_after_first_reset_preserves_intent(tmp_path, monkeypatch, drift):
    gate, digest = inputs(tmp_path, monkeypatch)
    calls = []
    def collector(case, mode, actor):
        calls.append(case)
        target = {"gate": gate, "source": tmp_path / "source.pt", "corrected": tmp_path / "corrected.pt",
                  "protocol": tmp_path / "runs/drift/protocol.json",
                  "evaluator": tmp_path / "scripts/evaluate_rlpd_local_recovery_gate.py"}[drift]
        result = collect(case, mode, actor)
        target.write_bytes(b"changed")
        return result
    with pytest.raises((ValueError, json.JSONDecodeError)):
        evaluator.evaluate(gate, digest, "runs/drift", root=tmp_path, make=make, collect=collector)
    assert len(calls) == 1
    manifest = json.loads((tmp_path / "runs/drift/manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["reset_intents"] == 1
    assert "failure.json" in manifest["files_sha256"]


def test_reset_image_mismatch_fails_after_preserving_24_receipts(tmp_path, monkeypatch):
    gate, digest = inputs(tmp_path, monkeypatch)
    def collector(case, mode, actor):
        return collect(case, mode, actor, obs_hash="gate" if actor.gated else "source")
    with pytest.raises(ValueError, match="reset image"):
        evaluator.evaluate(gate, digest, "runs/mismatch", root=tmp_path, make=make, collect=collector)
    manifest = json.loads((tmp_path / "runs/mismatch/manifest.json").read_text())
    assert manifest["episodes_complete"] == 24 and manifest["status"] == "failed"


def test_reset_accessible_state_mismatch_fails(tmp_path, monkeypatch):
    gate, digest = inputs(tmp_path, monkeypatch)
    def collector(case, mode, actor):
        summary, trace = collect(case, mode, actor)
        if actor.gated:
            trace["pre_position"][0, 0] = 1
        return summary, trace
    with pytest.raises(ValueError, match="accessible reset state"):
        evaluator.evaluate(gate, digest, "runs/state-mismatch", root=tmp_path, make=make, collect=collector)


def test_failed_collector_keeps_intent_no_retry(tmp_path, monkeypatch):
    gate, digest = inputs(tmp_path, monkeypatch)
    def fail(*args):
        raise RuntimeError("collector failed")
    with pytest.raises(RuntimeError, match="collector failed"):
        evaluator.evaluate(gate, digest, "runs/failure", root=tmp_path, make=make, collect=fail)
    manifest = json.loads((tmp_path / "runs/failure/manifest.json").read_text())
    assert manifest["episodes_complete"] == 0 and manifest["reset_intents"] == 1
    assert any("reset-intent" in name for name in manifest["files_sha256"])


def test_standalone_loader_rejects_wrong_hash_and_pixel_export(tmp_path):
    path = tmp_path / "gate.pt"
    torch.save({"format": "haic-rlpd-pixel-actor-v1"}, path)
    with pytest.raises(ValueError, match="SHA mismatch"):
        evaluator.inspect_gate(path, "a" * 64, tmp_path)
    with pytest.raises(ValueError, match="not a local"):
        evaluator.inspect_gate(path, evaluator.original.sha(path), tmp_path)


def artifact(tmp_path, monkeypatch):
    from haic.algorithms.rlpd.local_recovery_gate import tensor_state_sha
    state = {"synthetic": torch.zeros(1)}
    source = tmp_path / "source.pt"
    corrected = tmp_path / "corrected.pt"
    for path in (source, corrected):
        torch.save({"actor_state_dict": state}, path)
    monkeypatch.setattr(evaluator.original, "V5_SHA", evaluator.original.sha(source))
    gate_module = tmp_path / evaluator.GATE_MODULE
    gate_module.parent.mkdir(parents=True)
    gate_module.write_text("synthetic gate source")
    primary = tmp_path / "primary.json"
    repair = tmp_path / "repair.json"
    primary.write_text("{}")
    repair.write_text("{}")
    protocol = {"source_actor": {"path": source.name, "sha256": evaluator.original.sha(source),
                                 "state_sha256": tensor_state_sha(state)},
                "corrected_actor": {"path": corrected.name, "sha256": evaluator.original.sha(corrected),
                                    "state_sha256": tensor_state_sha(state)},
                "policy": {"hold_decisions": 12, "radius_fraction": .5},
                "source_hashes": {evaluator.GATE_MODULE: evaluator.original.sha(gate_module)},
                "inputs": {"consumed": True},
                "primary_result": {"path": primary.name, "sha256": evaluator.original.sha(primary)},
                "repair_receipt": {"path": repair.name, "sha256": evaluator.original.sha(repair)}}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    payload = {"format": "haic-rlpd-local-recovery-gate-v1", "policy": protocol["policy"],
               "actor_state_dicts": {"source": state, "corrected": state},
               "provenance": {**protocol, "protocol_sha256": evaluator.original.sha(protocol_path),
                   "calibration": {"protected_triggers": 0, "protected_points": 3,
                       "protected_images_exact_source_action_checked": 3, "guide_rows": 87}}}
    gate = tmp_path / "gate.pt"
    torch.save(payload, gate)
    return gate, payload


def test_inspect_standalone_provenance_and_embedded_states(tmp_path, monkeypatch):
    gate, payload = artifact(tmp_path, monkeypatch)
    metadata = evaluator.inspect_gate(gate, evaluator.original.sha(gate), tmp_path)
    assert metadata["calibration"]["protected_triggers"] == 0
    assert metadata["source_actor"]["sha256"] == evaluator.original.V5_SHA
    assert set(metadata["evidence_sha256"]) == {"protocol.json", "primary.json", "repair.json"}
    payload["actor_state_dicts"]["source"] = {"synthetic": torch.ones(1)}
    torch.save(payload, gate)
    with pytest.raises(ValueError, match="embedded actor"):
        evaluator.inspect_gate(gate, evaluator.original.sha(gate), tmp_path)


@pytest.mark.parametrize("kind", ["calibration", "menu", "source", "builder", "unpinned"])
def test_artifact_preflight_contract_fail_closed(tmp_path, monkeypatch, kind):
    gate, payload = artifact(tmp_path, monkeypatch)
    if kind == "calibration":
        payload["provenance"]["calibration"]["protected_triggers"] = 1
    elif kind == "menu":
        payload["policy"]["hold_decisions"] = 25
    elif kind == "source":
        payload["provenance"]["source_actor"]["sha256"] = "a" * 64
    elif kind == "builder":
        (tmp_path / "protocol.json").write_text("{}")
    else:
        payload["provenance"]["source_hashes"] = {}
    torch.save(payload, gate)
    with pytest.raises(ValueError):
        evaluator.inspect_gate(gate, evaluator.original.sha(gate), tmp_path)


def test_cli_return_codes(monkeypatch):
    arguments = ["--gate-artifact", "fake", "--gate-sha256", "a" * 64,
                 "--output", "runs/test", "--preflight-only"]
    monkeypatch.setattr(evaluator, "evaluate", lambda *args, **kwargs: {"status": "preflight_only", "environment_resets": 0})
    assert evaluator.main(arguments) == 0
    def fail(*args, **kwargs):
        raise ValueError("fail closed")
    monkeypatch.setattr(evaluator, "evaluate", fail)
    assert evaluator.main(arguments) == 2
