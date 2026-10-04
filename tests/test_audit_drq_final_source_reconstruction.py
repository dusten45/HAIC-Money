from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import audit_drq_final_source_reconstruction as audit

AuditError = audit.AuditError
summarize_metrics = audit.summarize_metrics


def _row(step: int, *, seed: int = 10, family: str = "easy") -> dict:
    updates = max(0, step - 2)
    return {
        "additional_online_step": step,
        "episode_id": 0,
        "geometry_family": family,
        "geometry_seed": seed,
        "gradient_steps": updates,
        "online_samples": updates * 32,
        "source_samples": updates * 32,
        "track_id": 1,
    }


def test_summarize_metrics_accepts_reused_train_cells_and_budget():
    rows = [_row(step) for step in range(1, 5)]
    report = summarize_metrics(rows, {10: "easy"}, expected_steps=4, warmup_steps=2)

    assert report["decisions"] == 4
    assert report["gradient_steps"] == 2
    assert report["episodes"] == 1
    assert report["unique_track_seed_cells"] == 1
    assert report["unique_geometry_seeds"] == 1
    assert report["by_track_geometry_seed_decisions"] == {"1:10": 4}


@pytest.mark.parametrize(
    ("rows", "train_families", "expected_steps"),
    [
        ([_row(1), _row(3)], {10: "easy"}, 2),
        ([_row(1, seed=11)], {10: "easy"}, 1),
        ([_row(1, family="wrong")], {10: "easy"}, 1),
    ],
)
def test_summarize_metrics_fails_closed(rows, train_families, expected_steps):
    with pytest.raises(AuditError):
        summarize_metrics(rows, train_families, expected_steps=expected_steps, warmup_steps=2)


def test_tdmpc2_active_shared_gpu_blocks_drq_snapshot(tmp_path, monkeypatch):
    (tmp_path / audit.TDMP2_V2_RUN_ROOT).mkdir(parents=True)
    monkeypatch.setattr(audit, "_active_gpu_apps", lambda: ["123, learner, 500 MiB"])

    with pytest.raises(AuditError, match="still using the shared GPU"):
        audit._tdmpc2_v2_snapshot(tmp_path, {}, {(1, 10)}, {10})


def test_tdmpc2_stable_reused_train_ledger_is_included(tmp_path, monkeypatch):
    run_dir = tmp_path / audit.TDMP2_V2_RUN_ROOT
    run_dir.mkdir(parents=True)
    protocol = {"source_sha256": {"script.py": "abc"}}
    records = [
        {"event": "start", "protocol_sha256": audit.TDMP2_V2_PROTOCOL_SHA,
         "source_sha256": protocol["source_sha256"]},
        {"event": "reset_intent", "track_id": 1, "geometry_seed": 10},
        {"event": "reset", "track_id": 1, "geometry_seed": 10},
        {"event": "step", "decisions": 1},
        {"event": "partial", "decisions": 1},
    ]
    (run_dir / "training.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    monkeypatch.setattr(audit, "_active_gpu_apps", lambda: [])

    snapshot = audit._tdmpc2_v2_snapshot(tmp_path, protocol, {(1, 10)}, {10})

    assert snapshot["recorded_decisions"] == 1
    assert snapshot["status"] == "partial"
    assert snapshot["actual_reset_cells"] == [{"track_id": 1, "geometry_seed": 10}]
    assert snapshot["reset_intent_cells"] == [{"track_id": 1, "geometry_seed": 10}]
    assert snapshot["fresh_seed_claim"] is False


def _tdmpc2_eval_fixture(tmp_path: Path) -> tuple[dict, dict, str]:
    run_dir = tmp_path / audit.TDMP2_V2_RUN_ROOT
    run_dir.mkdir(parents=True)
    protocol_sha = audit.TDMP2_V2_PROTOCOL_SHA
    source_sha = {"trainer.py": "abc"}
    cells = [{"track_id": 1, "geometry_seed": seed} for seed in (10, 11, 12, 13)]
    boundary = b"boundary-checkpoint"
    cpu_model = b"cpu-model-export"
    checkpoint_sha = hashlib.sha256(boundary).hexdigest()
    cpu_model_sha = hashlib.sha256(cpu_model).hexdigest()
    (run_dir / "boundary.pt").write_bytes(boundary)
    (run_dir / "cpu-model.pt").write_bytes(cpu_model)
    (run_dir / "cpu-model-export.json").write_text(json.dumps({
        "format": "haic-tdmpc2-cpu-export-v1",
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": checkpoint_sha,
        "sha256": cpu_model_sha,
        "environment_resets": 0,
        "decisions": 12058,
        "updates": 12057,
    }), encoding="utf-8")
    ledger = []
    for repeat in range(2):
        for index, cell in enumerate(cells):
            episode_seed = 73301 + repeat * len(cells) + index
            for mode in ("prior", "mppi"):
                identity = {"mode": mode, "repeat": repeat, "episode_seed": episode_seed,
                            "track_id": cell["track_id"], "geometry_seed": cell["geometry_seed"]}
                ledger.append({"event": "reset_intent", **identity})
                ledger.append({"event": "episode", "decisions": 500, **identity})
    (run_dir / "train-evaluation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ledger), encoding="utf-8"
    )
    protocol = {
        "source_sha256": source_sha,
        "evaluation": {"seed": 73301, "repeats": 2, "max_episodes": 16, "max_steps": 500},
        "cells": cells,
    }
    training_result = {"evaluation": None, "decisions": 12058, "updates": 12057}
    evaluation_result = {
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": checkpoint_sha,
        "cpu_export_sha256": cpu_model_sha,
        "source_sha256": source_sha,
        "reused_train_only": True,
        "evaluation_max_steps": 500,
        "capped_finish_comparison_valid": False,
        "episodes": 16,
        "per_mode": {"prior": {"episodes": 8}, "mppi": {"episodes": 8}},
    }
    (run_dir / "evaluation-result.json").write_text(
        json.dumps(evaluation_result), encoding="utf-8"
    )
    return protocol, training_result, checkpoint_sha


def test_tdmpc2_completed_cpu_evaluation_is_counted_as_reused_train(tmp_path):
    protocol, training_result, checkpoint_sha = _tdmpc2_eval_fixture(tmp_path)

    snapshot = audit._verify_tdmpc2_v2_cpu_evaluation(
        tmp_path, protocol, training_result, checkpoint_sha
    )

    assert snapshot["status"] == "completed_reused_TRAIN_only"
    assert snapshot["episodes"] == 16
    assert snapshot["unique_track_seed_cells"] == 4
    assert snapshot["fresh_cells"] == 0


def test_tdmpc2_in_progress_cpu_evaluation_blocks_snapshot(tmp_path):
    protocol, training_result, checkpoint_sha = _tdmpc2_eval_fixture(tmp_path)
    (tmp_path / audit.TDMP2_V2_RUN_ROOT / "evaluation-result.json").unlink()

    with pytest.raises(AuditError, match="pending or has no terminal result"):
        audit._verify_tdmpc2_v2_cpu_evaluation(
            tmp_path, protocol, training_result, checkpoint_sha
        )
