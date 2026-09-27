"""Receipt-only checks; no HAIC environment or model is loaded."""

import json

import pytest

from scripts.summarize_tdmpc2_pilot import digest, summarize


def _records(tmp_path):
    run_dir = tmp_path / "pilot"
    run_dir.mkdir()
    protocol_path = tmp_path / "protocol.json"
    protocol = {"run_dir": str(run_dir), "cells": [{"track_id": 1, "geometry_seed": 1}],
                "source_sha256": {"model": "abc"},
                "training": {"seed_steps": 2, "decision_cap": 8},
                "evaluation": {"repeats": 1}}
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    sha = digest(protocol_path)
    rows = [{"event": "start", "protocol_sha256": sha, "source_sha256": protocol["source_sha256"]}]
    for episode in range(4):
        rows.append({"event": "step", "decisions": episode + 1})
        rows.append({"event": "episode", "decisions": episode + 1, "length": 1,
                     "progress": (episode + 1) / 10, "reward": float(episode), "finished": episode == 3})
    rows.append({"event": "partial", "reason": "bounded_no_checkpoint"})
    (run_dir / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return protocol_path, run_dir, sha


def test_episode_denominators_and_partial_evidence(tmp_path):
    protocol, run_dir, sha = _records(tmp_path)
    report = summarize(protocol, run_dir, sha)
    assert report["recorded_step_count"] == 4
    assert report["complete_episode_count"] == 4
    assert report["random_episodes"]["episodes"] == 3
    assert report["planned_episodes"]["episodes"] == 1
    assert report["all_episodes"]["finishes"] == 1
    assert report["last_ledger_event"] == "partial"
    assert report["latest_checkpoint_hash_valid"] is False
    assert report["exact_boundary_resume_allowed"] is False


def test_receipt_summary_rejects_protocol_drift_and_step_gaps(tmp_path):
    protocol, run_dir, sha = _records(tmp_path)
    with pytest.raises(ValueError, match="protocol SHA"):
        summarize(protocol, run_dir, "0" * 64)
    ledger = run_dir / "training.jsonl"
    rows = [json.loads(row) for row in ledger.read_text().splitlines()]
    rows[1]["decisions"] = 7
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="consecutive"):
        summarize(protocol, run_dir, sha)
