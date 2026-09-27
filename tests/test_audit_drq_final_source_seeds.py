"""Synthetic-only historical TRAIN schedules; never load a real model or reset env."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_drq_final_source_seeds as audit


def _write(root: Path, name: str, data: dict | list[dict] | bytes) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        raw = data
    elif isinstance(data, list):
        raw = b"".join(json.dumps(row).encode() + b"\n" for row in data)
    else:
        raw = json.dumps(data).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _rows(seed: int) -> list[dict]:
    rows = []
    for ep, track, geometry in [(0, 1, 101), (1, 2, 102), (2, 3, 103), (3, 4, 104)][:4 - seed]:
        rows.append({"event": "reset", "episode_id": ep, "track_id": track, "seed": geometry})
        if ep < 2 or seed == 0 and ep == 2:
            length, end = ((2, 2), (3, 5), (2, 7))[ep]
            rows.append({"event": "end", "episode_id": ep, "track_id": track,
                         "seed": geometry, "steps": length, "global_step": end})
    return rows


def _state(seed: int) -> dict:
    ids = np.arange(3, 8, dtype=np.int64)
    slots = ids % 5
    sequence = np.empty(5, dtype=np.int64)
    sequence[slots] = ids
    episodes = np.empty(5, dtype=np.int64)
    steps = np.empty(5, dtype=np.int64)
    for seq, slot in zip(ids, slots):
        if seq < 5:
            episode, start = 1, 2
        elif seq < 7 or seed == 1:
            episode, start = 2, 5
        else:
            episode, start = 3, 7
        episodes[slot] = episode
        steps[slot] = seq - start
    return {"capacity": 5, "size": 5, "next_sequence": 8, "action_dim": 3,
            "n_step": 3, "gamma": .99, "sequence_ids": sequence,
            "episode_ids": episodes, "episode_steps": steps}


@pytest.fixture
def source(tmp_path, monkeypatch):
    for key, value in (("FIRST_SEQUENCE", 3), ("LAST_SEQUENCE", 7),
                       ("CAPACITY", 5), ("TOTAL_STEPS", 8), ("SOURCE_COUNTS", (4, 3))):
        monkeypatch.setattr(audit, key, value)
    root = tmp_path
    (root / audit.CLAIMS).mkdir(parents=True)
    _write(root, f"{audit.CLAIMS}/.gitkeep", b"")
    catalog = {"format": "haic-drq-training-geometry-protocol-v1",
               "exclusion_seed_ids": {"reserved": [101, 102, 103, 104],
                                      "heldout": [8001], "blind": [8002]}}
    catalog_protocol_sha = _write(root, audit.CATALOG_PROTOCOL, catalog)
    _write(root, "experiments/drqv2-teacher-replay-v1-r3.json", {
        "reserved_training_seeds": [101, 102, 103],
        "source_actors": [{"training_geometry_seeds": [102, 103]}]})
    _write(root, "experiments/pixel-rlpd-entropy-target-ablation-v5.json", {
        "reserved_training_seeds": [104],
        "partitions": {"confirmation": {"seeds": [8101]}, "blind": {"seeds": [8102]}}})
    catalog_path = "runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json"
    diagnostic = [{"track_id": 1, "geometry_seed": 9000 + i} for i in range(16)]
    catalog_sha = _write(root, catalog_path, {"train_diagnostic": diagnostic,
                                              "protocol_sha256": catalog_protocol_sha})
    r6_path = "experiments/drqv2-geometry-mix-v1-r6.json"
    r6_sha = _write(root, r6_path, {
        "format": "haic-drq-geometry-mix-study-v1",
        "environment": {"partition": "TRAIN", "obstacles": True},
        "diagnostic_pool": {"partition": "TRAIN-DIAGNOSTIC",
                            "geometry_seeds": [row["geometry_seed"] for row in diagnostic]},
        "catalog": {"path": catalog_path, "sha256": catalog_sha},
        "source_actors": [{"learner_seed": seed, "checkpoint_sha256":
                           hashlib.sha256(f"synthetic-checkpoint-{seed}".encode()).hexdigest(),
                           "actor_sha256": "a" * 64} for seed in (0, 1)]})
    src = {}
    for seed in (0, 1):
        base = f"runs/20260922-drq-augmentation-pad-v1-restart/control-seed{seed}"
        _write(root, f"{base}/config.json", {"config": {
            "training_obstacles": "official", "track_ids": [1, 2, 3, 4],
            "frame_skip": 4, "max_steps": 2000,
            "reward_contract": {"reward_shaping": False, "norm_reward": False},
            "protocol": {"training_track_ids": [1, 2, 3, 4],
                         "frame_skip": 4, "max_steps": 2000}}})
        ledger = f"{base}/episodes.jsonl"
        ledger_sha = _write(root, ledger, _rows(seed))
        checkpoint = f"{base}/checkpoints/step-000131072/checkpoint.pt"
        checkpoint_sha = _write(root, checkpoint, f"synthetic-checkpoint-{seed}".encode())
        src[str(seed)] = {"checkpoint_path": checkpoint, "checkpoint_sha256": checkpoint_sha,
                          "episode_ledger_path": ledger, "episode_ledger_sha256": ledger_sha}
    _write(root, audit.R7_PROTOCOL, {"format": "haic-drq-retention-study-v1",
        "r6_protocol_path": r6_path, "r6_protocol_sha256": r6_sha,
        "source_replay": src})
    monkeypatch.setattr(audit, "_checkpoint_state",
                        lambda path: _state(0 if "seed0" in str(path) else 1))
    protected = root / "runs/anything/blind/episodes.jsonl"
    protected.parent.mkdir(parents=True)
    protected.write_text("must never open", encoding="utf-8")
    return root


def test_containing_then_suffix_then_earlier_once_and_known_source_exclusions(source, monkeypatch):
    original = Path.read_bytes

    def guard(path: Path):
        assert "blind" not in path.parts
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    report = audit.audit(source)
    assert report["format"] == audit.FORMAT
    assert report["passed"] and report["partition"] == "TRAIN"
    assert report["protected_or_reserved_overlap"] == report["ambiguous_records"] == []
    assert report["evidence"]["distinct_historical_source_train_cells"] == 4
    assert report["evidence"]["expected_source_exclusions"]["distinct_geometry_seeds"] == 4
    assert report["evidence"]["source_schedule"]["0"] == {
        "containing_episode": 1, "primary_count": 3, "earlier_once_count": 1,
        "total_count": 4}
    expected = [{"original_episode_id": ep, "track_id": ep + 1,
                 "geometry_seed": 101 + ep} for ep in (1, 2, 3, 0)]
    assert report["schedule_sha256"]["0"] == audit.schedule_sha256(expected)
    assert report["schedule_sha256"]["1"] == audit.schedule_sha256(
        [expected[0], expected[1], expected[-1]])


@pytest.mark.parametrize("file,value,reason", [
    ("experiments/protected.json", {"partitions": {"blind": {"seeds": [102]}}}, "protected"),
    ("experiments/protected.json", {"train_diagnostic": {"geometry_seeds": [102]}}, "protected"),
    ("experiments/foreign.json", {"reserved_training_seeds": [102]}, "foreign"),
    ("experiments/pixel-rlpd-entropy-target-ablation-v6.json",
     {"reserved_training_seeds": [102]}, "foreign"),
    ("experiments/unknown.json", {"road_seed_range": {"start": 100, "count": 4}}, "ambiguous"),
    ("experiments/unknown.json", b'{"geometry_seed":102,', "ambiguous"),
])
def test_protected_foreign_and_ambiguous_experiment_sources_block(source, file, value, reason):
    _write(source, file, value)
    with pytest.raises(ValueError, match=reason):
        audit.audit(source)


def test_foreign_cross_track_train_claim_blocks_even_when_status_unknown(source):
    _write(source, f"{audit.CLAIMS}/seed-102.json", b"{malformed partial claim")
    with pytest.raises(ValueError, match="immutable TRAIN claim"):
        audit.audit(source)


def test_cross_track_actual_train_ledger_and_incomplete_candidate_record_block(source):
    path = "runs/foreign-training/episodes.jsonl"
    _write(source, path, [{"event": "reset", "track_id": 4, "seed": 102}])
    with pytest.raises(ValueError, match="cross-track"):
        audit.audit(source)
    _write(source, path, [{"event": "reset", "seed": 102}])
    with pytest.raises(ValueError, match="ambiguous TRAIN road identity"):
        audit.audit(source)
    _write(source, path, [{"event": "reset", "track_id": 2, "seed": 999,
                           "road_ids": [102]}])
    with pytest.raises(ValueError, match="ambiguous additional road field"):
        audit.audit(source)


def test_known_source_exclusion_missing_is_not_silently_passing(source):
    path = audit.CATALOG_PROTOCOL
    obj = json.loads((source / path).read_text())
    obj["exclusion_seed_ids"]["reserved"].remove(104)
    _write(source, path, obj)
    with pytest.raises(ValueError, match="missing from known-used"):
        audit.audit(source)


def test_unrelated_malformed_metadata_is_not_a_global_collision_gate(source):
    _write(source, "experiments/unrelated.json", b'{"unrelated":')
    assert audit.audit(source)["passed"]


def test_explicit_unrelated_range_requires_complete_ordered_candidates(source):
    path = "experiments/other-seed-audit.json"
    rule = "seed_start + offset for offsets 0..2, in that order; no replacement on collision"
    _write(source, path, {"seed_start": 4272000001,
                          "candidate_seeds": [4272000001, 4272000002, 4272000003],
                          "candidate_rule": rule})
    assert audit.audit(source)["passed"]
    _write(source, path, {"seed_start": 4272000001,
                          "candidate_seeds": [4272000001, 4272000003],
                          "candidate_rule": rule})
    with pytest.raises(ValueError, match="candidate-ambiguous"):
        audit.audit(source)


def test_checkpoint_retained_road_mismatch_is_not_a_ledger_only_clearance(source, monkeypatch):
    state = _state(0)
    state["episode_steps"][5 % 5] = 12
    monkeypatch.setattr(audit, "_checkpoint_state",
                        lambda path: state if "seed0" in str(path) else _state(1))
    with pytest.raises(ValueError, match="checkpoint episode identity/steps"):
        audit.audit(source)


def test_completed_own_seed_zero_pool_recheck_requires_hash_bound_receipt(source):
    base = "runs/20260927-drqv2-final-source-replay-v1/collection/seed0"
    protocol_sha = _write(source, "experiments/drqv2-final-source-replay-collection-v1.json",
                          {"format": "haic-drq-final-source-collection-v1"})
    ledger = f"{base}/episodes.jsonl"
    ledger_sha = _write(source, ledger, [
        {"event": "reset", "episode_id": 0, "schedule_index": 0,
         "original_episode_id": 1, "source_seed": 0, "track_id": 2,
         "seed": 102, "geometry_seed": 102, "partition": "TRAIN"},
        {"event": "capped_partial", "episode_id": 0, "schedule_index": 0,
         "original_episode_id": 1, "source_seed": 0, "track_id": 2,
         "seed": 102, "geometry_seed": 102}])
    step_sha = _write(source, f"{base}/steps.jsonl", [
        {"decision": n + 1, "sequence_id": n, "schedule_phase": "retained_window",
         "schedule_index": 0, "episode_id": 0, "source_seed": 0, "partition": "TRAIN",
         "original_episode_id": 1, "track_id": 2, "geometry_seed": 102} for n in range(5)])
    pool_sha = _write(source, f"{base}/pool.pt", b"synthetic replay pool")
    r7 = json.loads((source / audit.R7_PROTOCOL).read_text())
    receipt = {"format": "haic-drq-final-source-pool-v1", "completed": True,
               "source_seed": 0, "partition": "TRAIN", "excluded_diagnostic_roads": True,
               "collection_protocol_sha256": protocol_sha,
               "schedule_sha256": audit.schedule_sha256([
                   {"original_episode_id": ep, "track_id": ep + 1,
                    "geometry_seed": 101 + ep} for ep in (1, 2, 3, 0)]),
               "original_ledger_sha256": r7["source_replay"]["0"]["episode_ledger_sha256"],
               "source_checkpoint_sha256": r7["source_replay"]["0"]["checkpoint_sha256"],
               "source_actor_sha256": "a" * 64,
               "episode_ledger_path": ledger, "episode_ledger_sha256": ledger_sha,
               "step_ledger_path": f"{base}/steps.jsonl", "step_ledger_sha256": step_sha,
               "pool_path": f"{base}/pool.pt", "pool_sha256": pool_sha,
               "decisions": 5, "capacity": 5, "scheduled_episodes_consumed": 1,
               "geometry_seeds": [102], "retained_window_schedule_episodes": 3,
               "historical_prefix_episodes_consumed": 0, "historical_prefix_decisions": 0}
    _write(source, f"{base}/receipt.json", receipt)
    report = audit.audit(source)
    assert report["evidence"]["actual_train_ledgers"]["verified_own_pool_receipt_sha256"]["0"]
    receipt["episode_ledger_sha256"] = "0" * 64
    _write(source, f"{base}/receipt.json", receipt)
    with pytest.raises(ValueError, match="pinned source SHA mismatch"):
        audit.audit(source)


def test_preflight_is_read_only_and_blocked_audit_never_writes_receipt(source, monkeypatch, capsys):
    output = "experiments/historical-source-audit.json"
    argv = ["audit", "--repo-root", str(source), "--output", output]
    monkeypatch.setattr(sys, "argv", argv + ["--preflight-only"])
    audit.main()
    assert json.loads(capsys.readouterr().out)["passed"]
    assert not (source / output).exists()
    _write(source, "experiments/other.json", {"partitions": {"confirmation": {"seeds": [103]}}})
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="protected"):
        audit.main()
    assert not (source / output).exists()
    (source / "experiments/other.json").unlink()
    audit.main()
    assert json.loads((source / output).read_text())["passed"]
    with pytest.raises(ValueError, match="already exists"):
        audit.main()


def test_output_rejects_protected_and_claim_registry_destinations(source, monkeypatch):
    for output in ("experiments/train-seed-claims/seed-102.json",
                   "runs/20260927-drqv2-final-source-replay-v1/train-diagnostic/audit.json"):
        monkeypatch.setattr(sys, "argv", ["audit", "--repo-root", str(source), "--output", output])
        with pytest.raises(ValueError, match="new unprotected"):
            audit.main()
        assert not (source / output).exists()
