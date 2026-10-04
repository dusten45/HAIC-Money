"""Read-only exposure audit for a fresh matched DrQ six-arm reconstruction.

This records consumed history from the paused final-source study. It does not
claim fresh TRAIN cells, reserve seeds, build an environment, or run a learner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, cast


PARENT_PROTOCOL = "experiments/drqv2-final-source-replay-v1.json"
PARENT_PROTOCOL_SHA = "1d891d82e8a5d04be1fed53265df3b36d96d76d2e3317b110626f5f15d4f4e64"
COLLECTION_PROTOCOL = "experiments/drqv2-final-source-replay-collection-v1.json"
COLLECTION_PROTOCOL_SHA = "1fd60b9ab8a6d37171e5ed3989c63ba3cee059262159a68952563e99a01c28a2"
STOP_RECEIPT = "experiments/drqv2-final-source-replay-v1-migration-stop.json"
STOP_RECEIPT_SHA = "d85bf27716650be2fef02942a9d1cabe4666995ac2d94996091a42c50792a390"
PARENT_RUN_ROOT = "runs/20260927-drqv2-final-source-replay-v1"
CATALOG_PATH = "runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json"
CATALOG_SHA = "a8cbe5d2445563f9065a944254f6410486135b20eb8574ec1cf9eb611ebbf00b"
CROSS_LANE_AUDIT = f"{PARENT_RUN_ROOT}/cross-lane-audit-v1.json"
CROSS_LANE_AUDIT_SHA = "29ea04a47ce8fc732c75e4424f650a2d44420d001f08e4f87ca9f0d865f735f4"
NEW_RUN_ROOT = "runs/20260928-drqv2-final-source-replay-reconstruction-r1"
CLAIMS_DIR = "experiments/train-seed-claims"
TDMP2_V1_PROTOCOL = "experiments/tdmpc2-reused-train-pilot-v1.json"
TDMP2_V1_PROTOCOL_SHA = "e671cb916ea07c7c902b029393d5728560cfe9b48b04bade3a8c5604d54ef748"
TDMP2_V1_FAILURE = "experiments/tdmpc2-reused-train-pilot-v1-failure.json"
TDMP2_V1_FAILURE_SHA = "0c5a4c999e751bb7a2e0858f2c649a1ddf90a4441dd45507a6dfe122bf4d6df9"
TDMP2_V1_LEDGER = "runs/tdmpc2-reused-train-20260927-v1/training.jsonl"
TDMP2_V1_LEDGER_SHA = "ff909545fe87fac34e649e016586b8babc22ab296aa33ad935ca4a1e13871961"
TDMP2_V1_BOUNDARY = "runs/tdmpc2-reused-train-20260927-v1/boundary.pt"
TDMP2_V1_BOUNDARY_SHA = "d3502e430a0c4bb3be9a993ab16434d6b5400afdc201177080f8cc4d035d8b4c"
TDMP2_V2_PROTOCOL = "experiments/tdmpc2-reused-train-pilot-v2.json"
TDMP2_V2_PROTOCOL_SHA = "209f8a752a6ef61a3dcec6c770fe10fd71158779b83afbf72f9b5d72410c8bfb"
TDMP2_V2_RUN_ROOT = "runs/tdmpc2-reused-train-20260927-v2"
RLPD_RESERVED_PROTOCOLS = (
    ("experiments/pixel-rlpd-entropy-target-ablation-v1.json",
     "cbbe3039d8bec0564d3673f3e920540348c1e6bf5ae9a419a2c641e0b3a160a9"),
    ("experiments/pixel-rlpd-entropy-target-ablation-v2.json",
     "313bbff2164ff8f0a8efac7db2ad818628e00ee818237c09fd4fad3461e0ec98"),
    ("experiments/pixel-rlpd-entropy-target-ablation-v3.json",
     "541b86c8726efdb4c5c9c4c445aef75d96885adf64632626aaaa1924449eca41"),
    ("experiments/pixel-rlpd-entropy-target-ablation-v4.json",
     "86f525d4ac8ae1fb0af623e7824ed4970d30540cb74b1ca0423bc180cc9109d3"),
    ("experiments/pixel-rlpd-entropy-target-ablation-v5.json",
     "2e7df4152e2282bcd34a6d2a160737cc13c6f1a03d769bd69832e1fc51326546"),
)
WARMUP_STEPS = 10_000
ONLINE_STEPS = 32_768
UPDATES = 22_768
VARIANTS = ("uniform", "failure_weighted", "easy_retention")
COMPLETE_ARMS = (
    (0, "uniform"), (0, "failure_weighted"), (0, "easy_retention"),
    (1, "uniform"), (1, "failure_weighted"),
)
PARTIAL_ARM = (1, "easy_retention")
_CLAIM_NAME = re.compile(r"seed-(0|[1-9][0-9]*)\.json\Z")


class AuditError(ValueError):
    """Raised when any candidate-relevant evidence is missing or ambiguous."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _json(raw: str | bytes, source: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               AuditError(f"{source}: invalid JSON constant {value}")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{source}: invalid JSON: {exc}") from exc
    _require(type(value) is dict, f"{source}: expected a JSON object")
    return value


def _safe_file(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    _require(type(relative) is str and not path.is_absolute()
             and "\\" not in relative
             and all(part not in ("", ".", "..") for part in path.parts),
             f"unsafe repository path: {relative}")
    current = root
    for part in path.parts:
        current = current / part
        _require(not current.is_symlink(), f"symlink in evidence path: {relative}")
    _require(current.is_file(), f"missing evidence file: {relative}")
    return current


def _pinned_json(root: Path, relative: str, expected_sha: str) -> tuple[dict[str, Any], str]:
    path = _safe_file(root, relative)
    actual = _sha(path)
    _require(actual == expected_sha, f"SHA-256 mismatch: {relative}")
    return _json(path.read_bytes(), relative), actual


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise AuditError(f"{path}:{number}: blank JSONL row")
            yield _json(line, f"{path}:{number}")


def _catalog(root: Path) -> tuple[dict[str, Any], dict[int, str], set[int]]:
    catalog, _ = _pinned_json(root, CATALOG_PATH, CATALOG_SHA)
    train_rows = catalog.get("train")
    diagnostic_rows = catalog.get("train_diagnostic")
    if type(train_rows) is not list or len(train_rows) != 120:
        raise AuditError("expected the original 120-road TRAIN catalog")
    if type(diagnostic_rows) is not list or len(diagnostic_rows) != 16:
        raise AuditError("expected the disjoint 16-road TRAIN-DIAGNOSTIC catalog")
    train: dict[int, str] = {}
    for row in train_rows:
        if (type(row) is not dict or type(row.get("geometry_seed")) is not int
                or type(row.get("family")) is not str):
            raise AuditError("malformed TRAIN catalog row")
        seed = cast(int, row["geometry_seed"])
        family = cast(str, row["family"])
        _require(seed not in train, f"duplicate TRAIN geometry seed {seed}")
        train[seed] = family
    diagnostic: set[int] = set()
    for row in diagnostic_rows:
        if type(row) is not dict or type(row.get("geometry_seed")) is not int:
            raise AuditError("malformed TRAIN-DIAGNOSTIC catalog row")
        diagnostic.add(cast(int, row["geometry_seed"]))
    _require(len(diagnostic) == 16 and not (set(train) & diagnostic),
             "TRAIN and TRAIN-DIAGNOSTIC catalog identities overlap or are malformed")
    seed_audit = catalog.get("seed_audit", {})
    _require(seed_audit.get("passed") is True and seed_audit.get("matched_collisions") == [],
             "original TRAIN catalog seed audit did not pass cleanly")
    return catalog, train, diagnostic


def summarize_metrics(
    rows: Iterable[dict[str, Any]],
    train_families: dict[int, str],
    *,
    expected_steps: int,
    warmup_steps: int = WARMUP_STEPS,
) -> dict[str, Any]:
    """Verify the append-only online ledger and summarize TRAIN road exposure."""
    episode_cells: dict[int, tuple[int, int, str]] = {}
    cells: dict[tuple[int, int], int] = {}
    family_by_cell: dict[tuple[int, int], str] = {}
    steps = 0
    last_episode = -1
    last = None
    for steps, row in enumerate(rows, 1):
        step = row.get("additional_online_step")
        episode = row.get("episode_id")
        track = row.get("track_id")
        seed = row.get("geometry_seed")
        family = row.get("geometry_family")
        if type(step) is not int or step != steps:
            raise AuditError("online step sequence is not contiguous")
        if type(episode) is not int or episode < 0:
            raise AuditError("invalid episode_id in TRAIN ledger")
        if type(track) is not int or track not in (1, 2, 3, 4):
            raise AuditError("non-protocol track in TRAIN ledger")
        if type(seed) is not int or seed not in train_families:
            raise AuditError("non-TRAIN geometry seed in learner ledger")
        step = cast(int, step)
        episode = cast(int, episode)
        track = cast(int, track)
        seed = cast(int, seed)
        family = cast(str, family)
        _require(family == train_families[seed], "geometry family differs from frozen TRAIN catalog")
        cell = (track, seed, family)
        if episode in episode_cells:
            _require(episode_cells[episode] == cell, "episode changes TRAIN identity mid-episode")
        else:
            _require(episode == last_episode + 1, "episode IDs are not contiguous in TRAIN ledger")
            episode_cells[episode] = cell
            last_episode = episode
        updates = max(0, step - warmup_steps)
        _require(row.get("gradient_steps") == updates, "gradient-step cursor mismatch")
        _require(row.get("source_samples") == updates * 32
                 and row.get("online_samples") == updates * 32,
                 "per-update 32:32 replay sample cursor mismatch")
        pair = (track, seed)
        cells[pair] = cells.get(pair, 0) + 1
        family_by_cell[pair] = family
        last = row
    _require(steps == expected_steps, f"expected {expected_steps} step rows, found {steps}")
    if last is None:
        raise AuditError("empty learner ledger")
    return {
        "decisions": steps,
        "gradient_steps": last["gradient_steps"],
        "episodes": len(episode_cells),
        "unique_track_seed_cells": len(cells),
        "unique_geometry_seeds": len({seed for _, seed in cells}),
        "by_track_geometry_seed_decisions": {
            f"{track}:{seed}": count for (track, seed), count in sorted(cells.items())
        },
        "last_row": {
            key: last[key] for key in (
                "additional_online_step", "gradient_steps", "episode_id", "track_id", "geometry_seed"
            )
        },
        "episode_cells": episode_cells,
        "family_by_cell": family_by_cell,
    }


def _verify_episodes(path: Path, summary: dict[str, Any], catalog_sha: str) -> dict[str, Any]:
    reset_cells: dict[int, tuple[int, int, str]] = {}
    event_count = 0
    last_reset_id = -1
    for row in _jsonl(path):
        event_count += 1
        episode = row.get("episode_id")
        event = row.get("event")
        if event == "reset":
            if type(episode) is not int or episode != last_reset_id + 1:
                raise AuditError(f"{path}: reset episode IDs are not contiguous")
            episode = cast(int, episode)
            _require(row.get("catalog_sha256") == catalog_sha
                     and row.get("seed") == row.get("geometry_seed"),
                     f"{path}: reset identity differs from frozen catalog")
            track = row.get("track_id")
            seed = row.get("geometry_seed")
            family = row.get("geometry_family")
            _require(type(track) is int and type(seed) is int and type(family) is str,
                     f"{path}: malformed reset identity")
            cell = (cast(int, track), cast(int, seed), cast(str, family))
            _require(summary["episode_cells"].get(episode) == cell,
                     f"{path}: reset does not match per-step exposure")
            reset_cells[episode] = cell
            last_reset_id = episode
        elif event == "end":
            if type(episode) is not int or episode not in reset_cells:
                raise AuditError(f"{path}: end event has no matching reset")
            episode = cast(int, episode)
            _require((row.get("track_id"), row.get("geometry_seed", row.get("seed")),
                      row.get("geometry_family"))
                     == reset_cells[episode], f"{path}: end event changes road identity")
        else:
            raise AuditError(f"{path}: unknown event type {event!r}")
    _require(reset_cells == summary["episode_cells"],
             f"{path}: episode reset ledger does not cover the per-step ledger")
    return {"events": event_count, "resets": len(reset_cells), "sha256": _sha(path)}


def _verify_result_artifacts(root: Path, run_dir: str, result: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, field in (("online_replay_manifest_sha256", "step-metrics.jsonl"),
                       ("diagnostic_trace_sha256", "drift.jsonl")):
        path = _safe_file(root, f"{run_dir}/{field}")
        actual = _sha(path)
        _require(actual == result.get(key), f"result does not bind {run_dir}/{field}")
        hashes[field] = actual
    candidates = result.get("candidates")
    if type(candidates) is not list or len(candidates) != 2:
        raise AuditError(f"{run_dir}: expected both 16,384 and 32,768 checkpoint receipts")
    for candidate in cast(list[dict[str, Any]], candidates):
        _require(type(candidate) is dict, f"{run_dir}: malformed checkpoint receipt")
        _require(candidate.get("checkpoint_online_step") in (16_384, 32_768),
                 f"{run_dir}: unexpected checkpoint step")
        _require(candidate.get("study_gradient_steps") == candidate["checkpoint_online_step"] - WARMUP_STEPS,
                 f"{run_dir}: checkpoint gradient count mismatch")
        for field in ("checkpoint", "actor", "sample_trace"):
            path_field = f"{field}_path"
            sha_field = f"{field}_sha256"
            actual = _sha(_safe_file(root, candidate[path_field]))
            _require(actual == candidate.get(sha_field), f"{run_dir}: {field} SHA mismatch")
            hashes[candidate[path_field]] = actual
    return hashes


def _verify_complete_arm(
    root: Path,
    seed: int,
    variant: str,
    train_families: dict[int, str],
    catalog_sha: str,
    parent_sha: str,
) -> dict[str, Any]:
    name = f"learner-{seed}-{variant}-final_source"
    run_dir = f"{PARENT_RUN_ROOT}/{name}"
    result_path = f"{run_dir}/result.json"
    result, result_sha = _pinned_json(root, result_path, _sha(_safe_file(root, result_path)))
    _require(result.get("completed") is True and result.get("study_protocol_sha256") == parent_sha
             and result.get("source_seed") == seed and result.get("variant") == variant
             and result.get("condition") == "final_source"
             and result.get("additional_online_steps") == ONLINE_STEPS
             and result.get("study_gradient_steps") == UPDATES
             and result.get("source_samples") == UPDATES * 32
             and result.get("online_samples") == UPDATES * 32,
             f"{name}: completed-run receipt differs from frozen design")
    run_config_path = _safe_file(root, f"{run_dir}/run-config.json")
    run_config = _json(run_config_path.read_bytes(), f"{run_dir}/run-config.json")
    _require(run_config.get("protocol_sha256") == parent_sha
             and run_config.get("source_seed") == seed
             and run_config.get("variant") == variant
             and run_config.get("condition") == "final_source",
             f"{name}: run config does not bind the parent protocol arm")
    metrics_path = _safe_file(root, f"{run_dir}/step-metrics.jsonl")
    summary = summarize_metrics(_jsonl(metrics_path), train_families, expected_steps=ONLINE_STEPS)
    episodes_path = _safe_file(root, f"{run_dir}/episodes.jsonl")
    episodes = _verify_episodes(episodes_path, summary, catalog_sha)
    artifact_hashes = _verify_result_artifacts(root, run_dir, result)
    return {
        "arm": name,
        "result_sha256": result_sha,
        "run_config_sha256": _sha(run_config_path),
        "step_metrics_sha256": _sha(metrics_path),
        "episodes": episodes,
        "decisions": summary["decisions"],
        "gradient_steps": summary["gradient_steps"],
        "unique_track_seed_cells": summary["unique_track_seed_cells"],
        "unique_geometry_seeds": summary["unique_geometry_seeds"],
        "by_track_geometry_seed_decisions": summary["by_track_geometry_seed_decisions"],
        "artifact_hashes": artifact_hashes,
    }


def _verify_partial_arm(
    root: Path,
    stop: dict[str, Any],
    train_families: dict[int, str],
    parent_sha: str,
) -> dict[str, Any]:
    name = "learner-1-easy_retention-final_source"
    partial_dir = f"{PARENT_RUN_ROOT}/{name}"
    run_config_path = _safe_file(root, f"{partial_dir}/run-config.json")
    run_config = _json(run_config_path.read_bytes(), f"{partial_dir}/run-config.json")
    _require(run_config.get("protocol_sha256") == parent_sha
             and run_config.get("source_seed") == 1
             and run_config.get("variant") == "easy_retention"
             and run_config.get("condition") == "final_source",
             "partial arm run config differs from frozen protocol")
    exposure = stop.get("exposure_after_checkpoint", {})
    metrics_relative = exposure.get("step_metrics_path")
    _require(type(metrics_relative) is str, "stop receipt has no partial step-ledger path")
    metrics_path = _safe_file(root, cast(str, metrics_relative))
    actual_metrics_sha = _sha(metrics_path)
    _require(actual_metrics_sha == exposure.get("step_metrics_sha256"),
             "partial step ledger differs from migration-stop receipt")
    summary = summarize_metrics(_jsonl(metrics_path), train_families, expected_steps=21_037)
    _require(summary["gradient_steps"] == 11_037
             and summary["last_row"]["episode_id"] == 52
             and summary["last_row"]["track_id"] == 3
             and summary["last_row"]["geometry_seed"] == 3_910_800_146
             and exposure.get("post_checkpoint_consumed_decisions") == 4_653,
             "partial step cursor differs from immutable stop receipt")
    checkpoint = stop.get("last_saved_checkpoint", {})
    checkpoint_sha = _sha(_safe_file(root, checkpoint.get("path")))
    actor_sha = _sha(_safe_file(root, checkpoint.get("actor_path")))
    _require(checkpoint_sha == checkpoint.get("sha256") and actor_sha == checkpoint.get("actor_sha256"),
             "partial checkpoint or actor differs from migration-stop receipt")
    drift = exposure.get("drift_path")
    _require(type(drift) is str, "stop receipt has no partial drift-ledger path")
    drift_sha = _sha(_safe_file(root, cast(str, drift)))
    _require(drift_sha == exposure.get("drift_sha256"), "partial drift ledger differs from stop receipt")
    _require(not (root / partial_dir / "result.json").exists()
             and not (root / partial_dir / "checkpoint-catalog.json").exists()
             and not (root / partial_dir / "episodes.jsonl").exists(),
             "partial run unexpectedly has a completion artifact")
    return {
        "arm": name,
        "state": "partial; immutable; not a resumable checkpoint",
        "steps": summary["decisions"],
        "updates": summary["gradient_steps"],
        "post_checkpoint_consumed_decisions": 4_653,
        "episodes": summary["episodes"],
        "unique_track_seed_cells": summary["unique_track_seed_cells"],
        "unique_geometry_seeds": summary["unique_geometry_seeds"],
        "by_track_geometry_seed_decisions": summary["by_track_geometry_seed_decisions"],
        "step_metrics_sha256": actual_metrics_sha,
        "checkpoint_sha256": checkpoint_sha,
        "actor_sha256": actor_sha,
        "drift_sha256": drift_sha,
    }


def _claim_snapshot(root: Path, candidate_seeds: set[int]) -> dict[str, Any]:
    directory = root / CLAIMS_DIR
    _require(directory.is_dir() and not directory.is_symlink(), "shared TRAIN claim directory missing/unsafe")
    files = []
    candidate_claims = []
    for path in sorted(directory.iterdir()):
        if path.name == ".gitkeep":
            _require(path.is_file() and not path.is_symlink() and path.stat().st_size == 0,
                     "unsafe TRAIN claim marker")
            continue
        match = _CLAIM_NAME.fullmatch(path.name)
        if match is None or not path.is_file() or path.is_symlink():
            raise AuditError(f"unknown shared TRAIN claim entry: {path.name}")
        seed = int(match[1])
        claim = _json(path.read_bytes(), path.as_posix())
        _require(claim.get("geometry_seed") == seed, f"claim filename and geometry seed disagree: {path}")
        files.append({"path": path.relative_to(root).as_posix(), "sha256": _sha(path), "geometry_seed": seed})
        if seed in candidate_seeds:
            candidate_claims.append(seed)
    _require(not candidate_claims, "candidate TRAIN catalog seeds now have shared claims; re-audit required")
    return {"directory": CLAIMS_DIR, "claims": files, "candidate_claims": candidate_claims}


def _cells_from_protocol(protocol: dict[str, Any]) -> set[tuple[int, int]]:
    rows = protocol.get("cells")
    if type(rows) is not list:
        raise AuditError("cross-lane protocol has no explicit cell list")
    cells: set[tuple[int, int]] = set()
    for row in rows:
        if (type(row) is not dict or type(row.get("track_id")) is not int
                or type(row.get("geometry_seed")) is not int):
            raise AuditError("malformed cross-lane TRAIN cell")
        cells.add((cast(int, row["track_id"]), cast(int, row["geometry_seed"])))
    return cells


def _active_gpu_apps() -> list[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuditError(f"cannot verify shared GPU process state: {exc}") from exc
    return [line.strip() for line in completed.stdout.splitlines()
            if line.strip() and not line.lower().startswith("no running processes")]


def _verify_tdmpc2_v2_cpu_evaluation(
    root: Path,
    protocol: dict[str, Any],
    training_result: dict[str, Any],
    checkpoint_sha: str,
) -> dict[str, Any]:
    directory = root / TDMP2_V2_RUN_ROOT
    export_path = _safe_file(root, f"{TDMP2_V2_RUN_ROOT}/cpu-model.pt")
    export_receipt_path = _safe_file(root, f"{TDMP2_V2_RUN_ROOT}/cpu-model-export.json")
    export_receipt = _json(export_receipt_path.read_bytes(), export_receipt_path.as_posix())
    export_sha = _sha(export_path)
    _require(export_receipt.get("format") == "haic-tdmpc2-cpu-export-v1"
             and export_receipt.get("protocol_sha256") == TDMP2_V2_PROTOCOL_SHA
             and export_receipt.get("checkpoint_sha256") == checkpoint_sha
             and export_receipt.get("sha256") == export_sha
             and export_receipt.get("environment_resets") == 0
             and export_receipt.get("decisions") == training_result.get("decisions")
             and export_receipt.get("updates") == training_result.get("updates"),
             "TD-MPC2 v2 CPU export does not bind the completed TRAIN checkpoint")
    _require(training_result.get("evaluation") is None,
             "TD-MPC2 v2 training result unexpectedly mixes evaluation with TRAIN")
    ledger_relative = f"{TDMP2_V2_RUN_ROOT}/train-evaluation.jsonl"
    result_relative = f"{TDMP2_V2_RUN_ROOT}/evaluation-result.json"
    ledger_exists = (root / ledger_relative).exists()
    result_exists = (root / result_relative).exists()
    _require(ledger_exists and result_exists,
             "TD-MPC2 v2 CPU TRAIN evaluation is pending or has no terminal result")
    ledger_path = _safe_file(root, ledger_relative)
    before = ledger_path.stat()
    raw = ledger_path.read_bytes()
    after = ledger_path.stat()
    _require((before.st_size, before.st_mtime_ns, before.st_ino)
             == (after.st_size, after.st_mtime_ns, after.st_ino)
             and raw.endswith(b"\n"),
             "TD-MPC2 v2 CPU evaluation ledger is still changing or truncated")
    lines = raw.splitlines()
    _require(all(line.strip() for line in lines), "TD-MPC2 v2 evaluation ledger contains blank rows")
    rows = [_json(line, f"{ledger_relative}:{index}") for index, line in enumerate(lines, 1)]
    result_path = _safe_file(root, result_relative)
    result = _json(result_path.read_bytes(), result_relative)
    evaluation = protocol.get("evaluation")
    cells = protocol.get("cells")
    if (type(evaluation) is not dict or type(cells) is not list
            or any(type(cell) is not dict for cell in cells)):
        raise AuditError("TD-MPC2 v2 frozen evaluation/cell contract is malformed")
    _require(result.get("protocol_sha256") == TDMP2_V2_PROTOCOL_SHA
             and result.get("checkpoint_sha256") == checkpoint_sha
             and result.get("cpu_export_sha256") == export_sha
             and result.get("source_sha256") == protocol.get("source_sha256")
             and result.get("reused_train_only") is True
             and result.get("evaluation_max_steps") == evaluation.get("max_steps")
             and result.get("capped_finish_comparison_valid") is False,
             "TD-MPC2 v2 CPU evaluation result differs from its frozen TRAIN-only contract")
    expected = set()
    for repeat in range(evaluation["repeats"]):
        for index, cell in enumerate(cells):
            episode_seed = evaluation["seed"] + repeat * len(cells) + index
            for mode in ("prior", "mppi"):
                expected.add((mode, repeat, episode_seed, cell["track_id"], cell["geometry_seed"]))
    intents = set()
    episodes = set()
    for row in rows:
        event = row.get("event")
        if event not in ("reset_intent", "episode"):
            raise AuditError(f"unexpected TD-MPC2 v2 evaluation ledger event: {event!r}")
        key = (row.get("mode"), row.get("repeat"), row.get("episode_seed"),
               row.get("track_id"), row.get("geometry_seed"))
        _require(key in expected, "TD-MPC2 v2 evaluation used a non-protocol TRAIN cell")
        if event == "reset_intent":
            _require(key not in intents, "duplicate TD-MPC2 v2 evaluation reset intent")
            intents.add(key)
        else:
            _require(key not in episodes and type(row.get("decisions")) is int
                     and 0 < row["decisions"] <= evaluation["max_steps"],
                     "duplicate or invalid TD-MPC2 v2 evaluation episode")
            episodes.add(key)
    summary = result.get("per_mode")
    _require(episodes == intents == expected and result.get("episodes") == len(expected)
             and type(summary) is dict
             and all(summary.get(mode, {}).get("episodes") == 8 for mode in ("prior", "mppi")),
             "TD-MPC2 v2 CPU evaluation is incomplete or its ledger/result disagree")
    return {
        "status": "completed_reused_TRAIN_only",
        "cpu_model_sha256": export_sha,
        "cpu_model_export_receipt_sha256": _sha(export_receipt_path),
        "evaluation_ledger_sha256": hashlib.sha256(raw).hexdigest(),
        "evaluation_result_sha256": _sha(result_path),
        "episodes": len(episodes),
        "unique_track_seed_cells": len({(row["track_id"], row["geometry_seed"])
                                         for row in rows if row["event"] == "episode"}),
        "fresh_cells": 0,
    }


def _tdmpc2_v2_snapshot(
    root: Path,
    protocol: dict[str, Any],
    declared_cells: set[tuple[int, int]],
    catalog_seeds: set[int],
) -> dict[str, Any]:
    run_root = root / TDMP2_V2_RUN_ROOT
    if not run_root.exists():
        return {
            "run_root": TDMP2_V2_RUN_ROOT,
            "run_root_exists": False,
            "status": "frozen_not_started",
            "declared_cells": [
                {"track_id": track, "geometry_seed": seed}
                for track, seed in sorted(declared_cells)
            ],
            "fresh_seed_claim": False,
        }
    _require(run_root.is_dir() and not run_root.is_symlink(),
             "TD-MPC2 v2 run path is not a real directory")
    apps = _active_gpu_apps()
    _require(not apps, f"TD-MPC2 v2 is still using the shared GPU: {apps}")
    ledger = _safe_file(root, f"{TDMP2_V2_RUN_ROOT}/training.jsonl")
    before = ledger.stat()
    raw = ledger.read_bytes()
    after = ledger.stat()
    _require((before.st_size, before.st_mtime_ns, before.st_ino)
             == (after.st_size, after.st_mtime_ns, after.st_ino),
             "TD-MPC2 v2 ledger changed during exposure audit")
    _require(raw.endswith(b"\n"), "TD-MPC2 v2 ledger has a truncated final row")
    lines = raw.splitlines()
    _require(all(line.strip() for line in lines), "TD-MPC2 v2 ledger contains blank rows")
    rows = [_json(line, f"{TDMP2_V2_RUN_ROOT}/training.jsonl:{index}")
            for index, line in enumerate(lines, 1)]
    if not rows:
        raise AuditError("TD-MPC2 v2 ledger is empty")
    _require(rows[0].get("event") == "start"
             and rows[0].get("protocol_sha256") == TDMP2_V2_PROTOCOL_SHA
             and rows[0].get("source_sha256") == protocol.get("source_sha256"),
             "TD-MPC2 v2 run start record does not bind the frozen protocol/source")
    actual_cells: set[tuple[int, int]] = set()
    intent_cells: set[tuple[int, int]] = set()
    decisions = resets = reset_intents = 0
    for row in rows:
        event = row.get("event")
        if event in ("reset", "reset_intent"):
            track = row.get("track_id")
            seed = row.get("geometry_seed")
            _require(type(track) is int and type(seed) is int and track == 1
                     and seed in catalog_seeds,
                     "TD-MPC2 v2 ledger contains a non-TRAIN/foreign reset cell")
            cell = (cast(int, track), cast(int, seed))
            _require(cell in declared_cells, "TD-MPC2 v2 reset left its frozen reused cell set")
            if event == "reset":
                actual_cells.add(cell)
                resets += 1
            else:
                intent_cells.add(cell)
                reset_intents += 1
        elif event == "step":
            decisions += 1
    result_path = root / TDMP2_V2_RUN_ROOT / "result.json"
    result_sha = None
    training_result: dict[str, Any] | None = None
    if result_path.exists():
        result = _json(_safe_file(root, f"{TDMP2_V2_RUN_ROOT}/result.json").read_bytes(),
                       f"{TDMP2_V2_RUN_ROOT}/result.json")
        _require(result.get("protocol_sha256") == TDMP2_V2_PROTOCOL_SHA
                 and result.get("decisions") == decisions,
                 "TD-MPC2 v2 result does not reconcile with its immutable ledger")
        status = result.get("status")
        _require(type(status) is str and bool(status),
                 "TD-MPC2 v2 result has no terminal status")
        training_result = result
        result_sha = _sha(result_path)
    else:
        last_event = rows[-1]
        _require(last_event.get("event") == "partial"
                 and last_event.get("decisions") == decisions,
                 "TD-MPC2 v2 has no stable result or final partial event")
        status = "partial"
    cpu_evaluation = None
    if training_result is not None:
        boundary_path = _safe_file(root, f"{TDMP2_V2_RUN_ROOT}/boundary.pt")
        boundary_sha = _sha(boundary_path)
        _require(rows[-1].get("event") == "checkpoint"
                 and rows[-1].get("sha256") == boundary_sha,
                 "TD-MPC2 v2 result is not bound to the final episode-boundary checkpoint")
        cpu_evaluation = _verify_tdmpc2_v2_cpu_evaluation(
            root, protocol, training_result, boundary_sha
        )
    return {
        "run_root": TDMP2_V2_RUN_ROOT,
        "run_root_exists": True,
        "status": status,
        "run_ledger_sha256": hashlib.sha256(raw).hexdigest(),
        "result_sha256": result_sha,
        "cpu_evaluation": cpu_evaluation,
        "recorded_decisions": decisions,
        "successful_resets": resets,
        "reset_intents": reset_intents,
        "actual_reset_cells": [
            {"track_id": track, "geometry_seed": seed}
            for track, seed in sorted(actual_cells)
        ],
        "reset_intent_cells": [
            {"track_id": track, "geometry_seed": seed}
            for track, seed in sorted(intent_cells)
        ],
        "fresh_seed_claim": False,
        "classification": "consumed TRAIN reuse in another lane",
    }


def _other_lane_snapshot(root: Path, train_families: dict[int, str]) -> dict[str, Any]:
    catalog_seeds = set(train_families)
    td_v1, td_v1_sha = _pinned_json(root, TDMP2_V1_PROTOCOL, TDMP2_V1_PROTOCOL_SHA)
    td_v1_cells = _cells_from_protocol(td_v1)
    _require({seed for _, seed in td_v1_cells} <= catalog_seeds,
             "TD-MPC2 v1 cell declaration is outside the DrQ TRAIN catalog")
    td_failure, td_failure_sha = _pinned_json(root, TDMP2_V1_FAILURE, TDMP2_V1_FAILURE_SHA)
    _require(td_failure.get("protocol_sha256") == td_v1_sha
             and td_failure.get("observations", {}).get("recorded_decisions") == 10_061
             and td_failure.get("observations", {}).get("complete_episodes") == 29,
             "TD-MPC2 v1 receipt does not bind the expected partial TRAIN exposure")
    ledger_path = _safe_file(root, TDMP2_V1_LEDGER)
    _require(_sha(ledger_path) == TDMP2_V1_LEDGER_SHA,
             "TD-MPC2 v1 ledger differs from its failure receipt")
    boundary_path = _safe_file(root, TDMP2_V1_BOUNDARY)
    _require(_sha(boundary_path) == TDMP2_V1_BOUNDARY_SHA,
             "TD-MPC2 v1 boundary checkpoint differs from its failure receipt")
    actual_cells: set[tuple[int, int]] = set()
    decisions = resets = reset_intents = 0
    for row in _jsonl(ledger_path):
        if row.get("event") == "reset":
            track = row.get("track_id")
            seed = row.get("geometry_seed")
            if type(track) is not int or type(seed) is not int:
                raise AuditError("malformed TD-MPC2 reset identity")
            actual_cells.add((cast(int, track), cast(int, seed)))
            resets += 1
        elif row.get("event") == "reset_intent":
            reset_intents += 1
        elif row.get("event") == "step":
            decisions += 1
    _require(actual_cells == td_v1_cells and decisions == 10_061 and resets == 29
             and reset_intents == 30,
             "TD-MPC2 v1 observed cells/cursor differ from its frozen protocol and failure receipt")

    td_v2, td_v2_sha = _pinned_json(root, TDMP2_V2_PROTOCOL, TDMP2_V2_PROTOCOL_SHA)
    td_v2_cells = _cells_from_protocol(td_v2)
    _require(td_v2_cells == td_v1_cells and {seed for _, seed in td_v2_cells} <= catalog_seeds,
             "TD-MPC2 v2 does not declare the known reused four-cell TRAIN set")
    td_v2_snapshot = _tdmpc2_v2_snapshot(root, td_v2, td_v2_cells, catalog_seeds)

    rlpd_reservations = []
    for relative, expected_sha in RLPD_RESERVED_PROTOCOLS:
        protocol, actual_sha = _pinned_json(root, relative, expected_sha)
        reserved = protocol.get("reserved_training_seeds")
        if type(reserved) is not list or any(type(seed) is not int for seed in reserved):
            raise AuditError(f"{relative}: malformed RLPD reserved-training seed list")
        reserved = cast(list[int], reserved)
        intersection = sorted(set(reserved) & catalog_seeds)
        rlpd_reservations.append({
            "path": relative,
            "sha256": actual_sha,
            "reserved_training_seed_count": len(reserved),
            "drq_catalog_seed_intersection_count": len(intersection),
            "classification": "RLPD prior exclusion list; not an active shared TRAIN claim",
        })
    return {
        "tdmpc2_v1": {
            "protocol_path": TDMP2_V1_PROTOCOL,
            "protocol_sha256": td_v1_sha,
            "failure_receipt_path": TDMP2_V1_FAILURE,
            "failure_receipt_sha256": td_failure_sha,
            "ledger_path": TDMP2_V1_LEDGER,
            "ledger_sha256": TDMP2_V1_LEDGER_SHA,
            "boundary_path": TDMP2_V1_BOUNDARY,
            "boundary_sha256": TDMP2_V1_BOUNDARY_SHA,
            "actual_reset_cells": [
                {"track_id": track, "geometry_seed": seed}
                for track, seed in sorted(actual_cells)
            ],
            "recorded_decisions": decisions,
            "completed_episodes": resets,
            "reset_intents": reset_intents,
            "classification": "consumed TRAIN reuse in another lane",
        },
        "tdmpc2_v2": {
            "protocol_path": TDMP2_V2_PROTOCOL,
            "protocol_sha256": td_v2_sha,
            **td_v2_snapshot,
        },
        "rlpd_prior_seed_exclusions": rlpd_reservations,
    }


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    parent, parent_sha = _pinned_json(root, PARENT_PROTOCOL, PARENT_PROTOCOL_SHA)
    _require(parent.get("study_id") == "drqv2-final-source-replay-v1",
             "parent final-source protocol identity changed")
    collection, collection_sha = _pinned_json(root, COLLECTION_PROTOCOL, COLLECTION_PROTOCOL_SHA)
    stop, stop_sha = _pinned_json(root, STOP_RECEIPT, STOP_RECEIPT_SHA)
    _require(stop.get("study_protocol_sha256") == parent_sha
             and stop.get("completed_other_arms") == 5
             and stop.get("sixth_arm_completed") is False,
             "migration-stop receipt does not bind the expected partial study")
    catalog, train_families, diagnostic_seeds = _catalog(root)
    cross_lane, cross_lane_sha = _pinned_json(root, CROSS_LANE_AUDIT, CROSS_LANE_AUDIT_SHA)
    cross_evidence = cross_lane.get("evidence", {})
    _require(cross_lane.get("passed") is True and cross_lane.get("ambiguous_records") == []
             and cross_lane.get("protected_or_reserved_overlap") == []
             and cross_evidence.get("actual_train_ledgers", {}).get("protected_outcome_directories_opened") == 0,
             "parent source-pool cross-lane audit is not clean")
    _require(collection.get("cross_lane_audit", {}).get("sha256") == cross_lane_sha,
             "collection protocol no longer binds the cross-lane audit")

    complete = [_verify_complete_arm(root, seed, variant, train_families,
                                     CATALOG_SHA, parent_sha)
                for seed, variant in COMPLETE_ARMS]
    partial = _verify_partial_arm(root, stop, train_families, parent_sha)
    other_lanes = _other_lane_snapshot(root, train_families)
    candidate_seeds = set(train_families)
    claims = _claim_snapshot(root, candidate_seeds)
    prior_by_cell: dict[tuple[int, int], int] = {}
    for arm in [*complete, partial]:
        for key, count in arm["by_track_geometry_seed_decisions"].items():
            track, seed = (int(value) for value in key.split(":"))
            prior_by_cell[(track, seed)] = prior_by_cell.get((track, seed), 0) + count
    _require(diagnostic_seeds.isdisjoint(seed for _, seed in prior_by_cell),
             "a prior learner used a TRAIN-DIAGNOSTIC geometry seed")
    return {
        "format": "haic-drq-final-source-reconstruction-exposure-audit-v1",
        "audit_status": "clear_for_reuse_only",
        "freshness_status": "not_fresh; all 120 TRAIN seeds were allocated by the frozen parent catalog",
        "partition": "TRAIN",
        "candidate": {
            "study_id": "drqv2-final-source-replay-reconstruction-r1",
            "train_catalog_path": CATALOG_PATH,
            "train_catalog_sha256": CATALOG_SHA,
            "train_geometry_seed_count": len(train_families),
            "train_track_ids": [1, 2, 3, 4],
            "train_diagnostic_seed_count": len(diagnostic_seeds),
            "train_diagnostic_used_for_training": False,
            "new_geometry_seeds": [],
            "shared_registry_claim_required": False,
            "claim_reason": "same parent-allocated TRAIN catalog; no fresh cells or new allocation",
        },
        "parent_evidence": {
            "protocol": {"path": PARENT_PROTOCOL, "sha256": parent_sha},
            "collection_protocol": {"path": COLLECTION_PROTOCOL, "sha256": collection_sha},
            "migration_stop_receipt": {"path": STOP_RECEIPT, "sha256": stop_sha},
            "cross_lane_audit": {"path": CROSS_LANE_AUDIT, "sha256": cross_lane_sha,
                                 "passed": True, "ambiguous_records": 0,
                                 "protected_or_reserved_overlap": 0},
            "other_lane_reuse_snapshot": other_lanes,
            "shared_train_claims": claims,
        },
        "previous_learner_exposure": {
            "complete_arms": complete,
            "partial_arm": partial,
            "total_recorded_decisions": sum(arm["decisions"] for arm in complete) + partial["steps"],
            "total_recorded_updates": sum(arm["gradient_steps"] for arm in complete) + partial["updates"],
            "unique_prior_track_seed_cells": len(prior_by_cell),
            "by_track_geometry_seed_decisions": {
                f"{track}:{seed}": count for (track, seed), count in sorted(prior_by_cell.items())
            },
        },
        "limitations": [
            "TRAIN roads are intentionally reused; this is not fresh geometry or generalization evidence.",
            "The source-pool audit is candidate-scoped and does not prove unrecorded external exposure.",
            "The interrupted sixth arm is preserved as exposure history, never treated as resumable or complete.",
            "Recheck active claims and resource/process gates immediately before each new run/reset.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preflight-only", action="store_true", help="print the audit; never write")
    parser.add_argument("--output", type=Path, help="new receipt under the reconstruction run root")
    args = parser.parse_args(argv)
    _require(args.preflight_only or args.output is not None,
             "--output is required unless --preflight-only is used")
    report = audit(args.repo_root)
    if args.preflight_only:
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
        return 0
    if args.output is None:
        raise AuditError("--output is required unless --preflight-only is used")
    output = args.output
    if output.is_absolute():
        output = output.relative_to(args.repo_root.resolve(strict=True))
    _require(not output.is_absolute() and all(part not in (".", "..") for part in output.parts)
             and output.parts[0:1] == (NEW_RUN_ROOT.split("/")[0],)
             and output.as_posix().startswith(NEW_RUN_ROOT + "/"),
             "audit receipt must be a new file under the dedicated reconstruction run root")
    destination = args.repo_root.resolve(strict=True) / output
    _require(destination.parent.is_dir() and not destination.parent.is_symlink(),
             "receipt parent must already exist and be a real directory")
    _require(not destination.exists() and not destination.is_symlink(), "refusing to overwrite receipt")
    raw = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"audit_status": report["audit_status"],
                      "receipt": output.as_posix(),
                      "sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
