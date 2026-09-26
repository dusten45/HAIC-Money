"""Join sealed per-actor DrQ teacher-replay evaluator receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.validate_drq_teacher_protocol import ProtocolError, validate_protocol_file


DECISION_TYPE = "haic-drq-teacher-replay-paired-decision-v1"
FINALIST_TYPE = "haic-drq-teacher-replay-finalist-v1"
SELECTION_FORMAT = "haic-drq-teacher-replay-screen-selection-v1"
ARMS = ("unchanged-source", "online-only", "teacher-replay")
STUDENT_ARMS = ("online-only", "teacher-replay")
WRAPPER_KEYS = {
    "study_protocol_sha256",
    "source_learner_seed",
    "arm",
    "actor_sha256",
    "checkpoint_sha256",
    "checkpoint_online_step",
    "partition",
    "evaluator_receipt_path",
    "evaluator_receipt_sha256",
}
WRAPPER_LINEAGE_KEYS = {"screen_receipt_sha256", "screen_candidate_identity"}
POINTER_KEYS = {
    "evaluation_dir", "protocol_name", "protocol_sha256", "partition",
    "diagnostic_only", "ranked",
}
MANIFEST_KEYS = {
    "schema_version", "protocol", "protocol_sha256", "partition", "diagnostic_only",
    "run_dir", "previous_evaluation", "started_at_utc", "cell_matrix", "frame_skip",
    "max_steps", "action_smoothing", "action_smoothing_fingerprints",
    "action_smoothing_by_candidate", "timeout_seconds", "workers", "limits", "git",
    "coordinator_runtime", "worker_runtime", "runtime", "finished_at_utc",
}
SUMMARY_KEYS = {
    "action_control", "action_control_fingerprint", "action_control_present",
    "action_representation", "action_representation_fingerprint",
    "action_representation_present", "action_smoothing", "action_smoothing_fingerprint",
    "action_smoothing_present", "algorithm", "aliases", "archive_sha256", "by_track",
    "candidate_id", "canonical_episodes", "cpu_reload_matches", "determinism_audited",
    "diagnostic_only", "eligible", "evaluation_archive_path", "evaluation_archive_sha256",
    "evaluation_run_config_path", "expected_cells", "expected_results", "export_metadata",
    "export_spec_fingerprints", "is_comparator", "norm_obs", "operational_failures",
    "policy_sha256", "resources", "run_config_path", "run_config_sha256", "run_frame_skip",
    "run_max_steps", "source_path", "summary", "vecnormalize_path", "vecnormalize_sha256",
}
EPISODE_KEYS = {
    "status", "track_id", "seed", "steps", "reward", "progress", "finished", "terminated",
    "truncated", "retire_reason", "finish_qualified", "finish_time_s", "lap_time_ms", "damage",
    "termination_class", "collision_actions", "steering_delta_abs_mean", "action_smoothing",
    "action_smoothing_fingerprint", "action_control", "action_control_fingerprint",
    "action_representation", "action_representation_fingerprint", "raw_time_s",
    "action_trace_sha256", "actions", "model_load_seconds", "process_initialization_seconds",
    "reset_seconds", "agent_reset_seconds", "first_action_seconds", "mean_action_seconds",
    "p95_action_seconds", "max_action_seconds", "actor_load_peak_rss_bytes", "peak_rss_bytes",
    "rss_scope", "episode_wall_seconds", "loaded_archive_sha256", "runtime", "worker_pid",
    "candidate_id", "source_path", "repeat", "parent_wall_seconds",
}
TRACE_FIELDS = (
    "status", "finished", "termination_class", "steps", "lap_time_ms", "progress", "damage",
    "collision_actions", "action_trace_sha256", "steering_delta_abs_mean",
    "action_smoothing_fingerprint", "action_control_fingerprint",
    "action_representation_fingerprint", "terminated", "truncated", "retire_reason",
    "loaded_archive_sha256",
)
EVALUATOR_FILES = {
    "manifest.json", "summary.json", "episodes.jsonl", "determinism.json", "protocol.json",
    "protocol_spec.json", "candidates.json",
}
RESOURCE_LIMITS = {
    "process_initialization_seconds": 10.0,
    "agent_reset_seconds": 5.0,
    "max_action_seconds": 5.0,
    "peak_rss_bytes": 1024**3,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_bytes(), parse_constant=_reject_constant)


def _digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
             f"{label} must be a lowercase SHA-256 digest")
    return value


def _sealed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    _require(sidecar.is_file(), f"{label} lacks its SHA-256 sidecar: {path}")
    expected = sidecar.read_text(encoding="ascii").strip()
    _require(SHA256_RE.fullmatch(expected) is not None and expected == actual,
             f"{label} hash mismatch: {path}")
    value = json.loads(raw, parse_constant=_reject_constant)
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value, actual


def _read_wrapper(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    value = json.loads(raw, parse_constant=_reject_constant)
    _require(isinstance(value, dict), f"actor receipt must be a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite")
    if minimum is not None:
        _require(value >= minimum, f"{label} must be at least {minimum}")
    return float(value)


def _identity(wrapper: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_learner_seed": wrapper["source_learner_seed"],
        "arm": wrapper["arm"],
        "actor_sha256": wrapper["actor_sha256"],
        "checkpoint_sha256": wrapper["checkpoint_sha256"],
        "checkpoint_online_step": wrapper["checkpoint_online_step"],
    }


def _identity_key(wrapper: dict[str, Any]) -> tuple[int, str, int | None]:
    return (
        wrapper["source_learner_seed"],
        wrapper["arm"],
        wrapper["checkpoint_online_step"],
    )


def _protocol_sources(protocol: dict[str, Any]) -> dict[int, dict[str, Any]]:
    sources = protocol.get("source_actors")
    _require(isinstance(sources, list) and len(sources) == 2, "protocol must define two source actors")
    indexed = {actor.get("learner_seed"): actor for actor in sources if isinstance(actor, dict)}
    _require(set(indexed) == {0, 1}, "protocol source learner identities must be 0 and 1")
    return indexed


def _expected_wrapper_identities(
    partition: str, protocol: dict[str, Any]
) -> set[tuple[int, str, int | None]]:
    if partition == "screen":
        return {
            (seed, "unchanged-source", None) for seed in (0, 1)
        } | {
            (seed, arm, step)
            for seed in (0, 1)
            for arm in STUDENT_ARMS
            for step in protocol["budgets"]["checkpoint_online_steps"]
        }
    return set()


def _validate_wrapper(
    path: Path, protocol: dict[str, Any], protocol_sha256: str, partition: str
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    wrapper, wrapper_sha256 = _read_wrapper(path)
    keys = set(wrapper)
    _require(keys == WRAPPER_KEYS or keys == WRAPPER_KEYS | WRAPPER_LINEAGE_KEYS,
             f"actor receipt has missing or unsupported schema fields: {path}")
    _require(wrapper["partition"] == partition, f"actor receipt has wrong partition: {path}")
    _require(_digest(wrapper["study_protocol_sha256"], "study_protocol_sha256") == protocol_sha256,
             f"actor receipt protocol SHA differs: {path}")
    learner = wrapper["source_learner_seed"]
    _require(type(learner) is int and learner in (0, 1), f"invalid source learner identity: {path}")
    arm = wrapper["arm"]
    _require(arm in ARMS, f"unsupported arm identity: {path}")
    actor_sha = _digest(wrapper["actor_sha256"], "actor_sha256")
    sources = _protocol_sources(protocol)
    if arm == "unchanged-source":
        source = sources[learner]
        _require(wrapper["checkpoint_online_step"] is None,
                 f"unchanged source must not have a checkpoint_online_step: {path}")
        checkpoint_sha = wrapper["checkpoint_sha256"]
        _require(checkpoint_sha is None or _digest(checkpoint_sha, "checkpoint_sha256") == source["checkpoint_sha256"],
                 f"unchanged source checkpoint identity differs from protocol: {path}")
        _require(actor_sha == source["actor_sha256"], f"unchanged source actor hash differs from protocol: {path}")
    else:
        checkpoint_sha = _digest(wrapper["checkpoint_sha256"], "checkpoint_sha256")
        step = wrapper["checkpoint_online_step"]
        _require(type(step) is int and step in protocol["budgets"]["checkpoint_online_steps"],
                 f"student checkpoint step is not predeclared: {path}")
    pointer_path = Path(wrapper["evaluator_receipt_path"]).resolve()
    _require(pointer_path.is_file(), f"evaluator receipt is missing: {pointer_path}")
    expected_pointer_sha = _digest(wrapper["evaluator_receipt_sha256"], "evaluator_receipt_sha256")
    _require(sha256_file(pointer_path) == expected_pointer_sha,
             f"evaluator receipt hash mismatch: {pointer_path}")
    pointer = _read_json(pointer_path)
    _require(set(pointer) == POINTER_KEYS, f"unsupported evaluator pointer schema: {pointer_path}")
    _require(pointer["partition"] == partition and pointer["protocol_sha256"] == protocol_sha256,
             f"evaluator pointer partition/protocol mismatch: {pointer_path}")
    _require(pointer["protocol_name"] == f"{protocol['name']}-{partition}",
             f"evaluator pointer name mismatch: {pointer_path}")
    _require(pointer["diagnostic_only"] is False, f"diagnostic evaluator receipts are not admissible: {pointer_path}")
    _require(isinstance(pointer["evaluation_dir"], str) and Path(pointer["evaluation_dir"]).is_absolute(),
             f"evaluator directory must be absolute: {pointer_path}")
    directory = Path(pointer["evaluation_dir"]).resolve()
    _require(directory.is_dir(), f"evaluator directory is missing: {directory}")
    _require(isinstance(pointer["ranked"], list) and len(pointer["ranked"]) == 1,
             f"actor evaluator pointer must contain one candidate: {pointer_path}")
    report = _validate_evaluator_report(directory, pointer, protocol, protocol_sha256, partition, wrapper)
    return wrapper, wrapper_sha256, report


def _read_evaluator_files(
    directory: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _read_json(directory / "manifest.json")
    summary = _read_json(directory / "summary.json")
    _require(isinstance(summary, list) and len(summary) == 1,
             f"evaluator summary must contain one actor: {directory}")
    episodes: list[dict[str, Any]] = []
    with (directory / "episodes.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line, parse_constant=_reject_constant)
            _require(isinstance(value, dict), f"invalid episode row at {directory}:{line_number}")
            episodes.append(value)
    determinism = _read_json(directory / "determinism.json")
    _require(isinstance(determinism, list), f"invalid determinism audit: {directory}")
    return manifest, summary, episodes, determinism


def _validate_previous_lineage(
    directory: Path,
    previous: Any,
    wrapper: dict[str, Any],
    partition: str,
    expected_pointer_path: Path | None = None,
) -> None:
    if partition == "screen":
        _require(previous is None, "screen evaluator must not have prior-evaluation lineage")
        return
    _require(isinstance(previous, dict) and set(previous) == {"evaluation_dir", "partition", "actor_sha256", "files"},
             "missing or unsupported evaluator previous_evaluation schema")
    previous_partition = "screen" if partition == "confirmation" else "confirmation"
    _require(previous["partition"] == previous_partition
             and previous["actor_sha256"] == wrapper["actor_sha256"],
             "evaluator previous_evaluation has wrong partition or actor")
    files = previous["files"]
    names = {"previous_evaluation.json", "previous_summary.json", "previous_manifest.json"}
    _require(isinstance(files, dict) and set(files) == names,
             "evaluator previous_evaluation has missing or extra files")
    expected_previous = {}
    if expected_pointer_path is not None:
        expected_previous["previous_evaluation.json"] = expected_pointer_path.resolve()
    previous_directory = Path(previous["evaluation_dir"]).resolve()
    if expected_pointer_path is not None:
        prior_pointer = _read_json(expected_pointer_path)
        _require(prior_pointer.get("evaluation_dir") == str(previous_directory),
                 "evaluator previous_evaluation directory differs from declared lineage")
    expected_previous.update({
        "previous_summary.json": previous_directory / "summary.json",
        "previous_manifest.json": previous_directory / "manifest.json",
    })
    for name in names:
        record = files[name]
        _require(set(record) == {"source_path", "sha256", "snapshot_path"},
                 f"unsupported prior-evaluation file schema: {name}")
        expected_sha = _digest(record["sha256"], f"previous_evaluation.{name}.sha256")
        source = Path(record["source_path"]).resolve()
        if name in expected_previous:
            _require(source == expected_previous[name],
                     "evaluator prior pointer does not match declared lineage")
        else:
            _require(source.is_file(), f"evaluator prior-evaluation source is missing: {name}")
        snapshot = (directory / record["snapshot_path"]).resolve()
        _require(snapshot.is_relative_to(directory) and snapshot.is_file(),
                 f"missing/escaped prior-evaluation snapshot: {name}")
        _require(sha256_file(source) == expected_sha == sha256_file(snapshot),
                 f"prior-evaluation file hash mismatch: {name}")


def _validate_runtime(runtime: Any) -> None:
    _require(isinstance(runtime, dict), "missing worker runtime metadata")
    packages = runtime.get("packages")
    _require(isinstance(packages, dict), "worker runtime packages are missing")
    _require(runtime.get("python_version") == [3, 11]
             and runtime.get("sys_platform") == "linux"
             and packages.get("torch", "").split("+")[0] == "2.1.0"
             and packages.get("numpy") == "1.26.0"
             and packages.get("gymnasium") == "0.29.1"
             and packages.get("opencv-python") == "4.8.1.78"
             and runtime.get("torch_cuda") is None
             and runtime.get("cuda_available") is False
             and runtime.get("torch_threads") == runtime.get("torch_interop_threads") == 1,
             "actor evaluator did not use the pinned CPU runtime")


def _validate_evaluator_report(
    directory: Path,
    pointer: dict[str, Any],
    protocol: dict[str, Any],
    protocol_sha256: str,
    partition: str,
    wrapper: dict[str, Any],
) -> dict[str, Any]:
    required_files = EVALUATOR_FILES.copy()
    manifest, ranked, episodes, audits = _read_evaluator_files(directory)
    _require(set(manifest) == MANIFEST_KEYS, f"unsupported evaluator manifest schema: {directory}")
    _require(manifest["schema_version"] == 2 and manifest["partition"] == partition
             and manifest["protocol"] == f"{protocol['name']}-{partition}"
             and manifest["protocol_sha256"] == protocol_sha256
             and manifest["diagnostic_only"] is False,
             f"evaluator manifest identity mismatch: {directory}")
    matrix = protocol["partitions"][partition]
    _require(manifest["cell_matrix"] == matrix
             and manifest["frame_skip"] == protocol["frame_skip"]
             and manifest["max_steps"] == protocol["max_steps"],
             f"evaluator manifest grid/horizon mismatch: {directory}")
    _require(_read_json(directory / "protocol.json") == matrix,
             f"evaluator cell protocol snapshot mismatch: {directory}")
    protocol_snapshot = directory / "protocol_spec.json"
    _require(sha256_file(protocol_snapshot) == protocol_sha256
             and _read_json(protocol_snapshot) == protocol,
             f"evaluator study protocol snapshot mismatch: {directory}")
    _require(pointer["ranked"] == ranked and manifest["previous_evaluation"] is not None
             if partition != "screen" else pointer["ranked"] == ranked and manifest["previous_evaluation"] is None,
             f"evaluator pointer/manifest mismatch: {directory}")
    if partition != "screen":
        _validate_previous_lineage(directory, manifest["previous_evaluation"], wrapper, partition)

    result = ranked[0]
    _require(set(result) == SUMMARY_KEYS, f"unsupported evaluator summary schema: {directory}")
    _require(result["algorithm"] == "drq-v2" and result["eligible"] is True
             and result["determinism_audited"] is True and result["cpu_reload_matches"] is True
             and result["operational_failures"] == 0 and result["diagnostic_only"] is False
             and result["is_comparator"] is False,
             f"actor is operationally ineligible: {directory}")
    actor_sha = wrapper["actor_sha256"]
    _require(result["archive_sha256"] == actor_sha
             and result["policy_sha256"] == actor_sha
             and result["evaluation_archive_sha256"] == actor_sha
             and result["candidate_id"] is not None,
             f"evaluator actor identity differs from wrapper: {directory}")
    _require(result["expected_cells"] == result["canonical_episodes"]
             and result["expected_cells"] == len(matrix["track_ids"]) * len(matrix["seeds"])
             and result["expected_results"] == result["expected_cells"] * 2,
             f"evaluator summary has wrong cell counts: {directory}")
    _require(result["run_frame_skip"] == protocol["frame_skip"]
             and result["run_max_steps"] == protocol["max_steps"],
             f"actor run config horizon differs from protocol: {directory}")
    _require(result["export_metadata"].get("format") == "haic-drq-v2-actor-v1"
             and result["export_metadata"].get("action_spec", {}).get("frame_skip") == protocol["frame_skip"],
             f"actor export format/action contract mismatch: {directory}")

    for name in required_files:
        _require((directory / name).is_file(), f"missing evaluator artifact: {directory / name}")
    candidate_archive = (directory / result["evaluation_archive_path"]).resolve()
    candidate_config = (directory / result["evaluation_run_config_path"]).resolve()
    _require(candidate_archive.is_relative_to(directory) and candidate_archive.is_file()
             and sha256_file(candidate_archive) == actor_sha,
             f"evaluator actor snapshot mismatch: {directory}")
    _require(candidate_config.is_relative_to(directory) and candidate_config.is_file()
             and sha256_file(candidate_config) == result["run_config_sha256"],
             f"evaluator config snapshot mismatch: {directory}")
    _require(result["run_config_sha256"] == sha256_file(Path(result["run_config_path"])),
             f"actor source config hash mismatch: {directory}")

    runtimes = manifest["worker_runtime"]
    _require(isinstance(runtimes, list) and bool(runtimes), f"missing worker runtime receipts: {directory}")
    for runtime in runtimes:
        _validate_runtime(runtime)
    grid = {
        (track, seed, repeat)
        for track in matrix["track_ids"]
        for seed in matrix["seeds"]
        for repeat in range(2)
    }
    observed: dict[tuple[int, int, int], dict[str, Any]] = {}
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        _require(set(row) == EPISODE_KEYS, f"unsupported evaluator episode schema: {directory}")
        key = (row["track_id"], row["seed"], row["repeat"])
        _require(key in grid and key not in observed, f"extra/duplicate evaluator cell {key}: {directory}")
        _require(row["candidate_id"] == result["candidate_id"]
                 and row["loaded_archive_sha256"] == actor_sha
                 and row["runtime"] in runtimes and row["status"] == "ok",
                 f"invalid evaluator worker result at {key}: {directory}")
        _require(type(row["finished"]) is bool and type(row["terminated"]) is bool
                 and type(row["truncated"]) is bool,
                 f"invalid terminal flags at {key}: {directory}")
        _require(type(row["track_id"]) is int and type(row["seed"]) is int
                 and type(row["repeat"]) is int and type(row["steps"]) is int
                 and 0 < row["steps"] <= protocol["max_steps"],
                 f"invalid cell coordinates/steps at {key}: {directory}")
        for field in ("progress", "reward", "damage", "steering_delta_abs_mean"):
            _finite_number(row[field], f"episode {field}")
        for field, limit in RESOURCE_LIMITS.items():
            _require(_finite_number(row[field], f"episode {field}", minimum=0) <= limit,
                     f"CPU operational limit exceeded ({field}) at {key}: {directory}")
        _require(isinstance(row["action_trace_sha256"], str)
                 and SHA256_RE.fullmatch(row["action_trace_sha256"]) is not None,
                 f"invalid action trace digest at {key}: {directory}")
        action_digest = hashlib.sha256()
        _require(isinstance(row["actions"], list), f"invalid action trace at {key}: {directory}")
        _require(len(row["actions"]) == row["steps"],
                 f"executed action count differs from episode steps at {key}: {directory}")
        for action in row["actions"]:
            _require(isinstance(action, list) and len(action) == 3,
                     f"invalid executed action at {key}: {directory}")
            values = [_finite_number(value, "action value") for value in action]
            action_digest.update(struct.pack("<3f", *values))
        _require(action_digest.hexdigest() == row["action_trace_sha256"],
                 f"action trace digest does not match actions at {key}: {directory}")
        if row["finished"]:
            _finite_number(row["lap_time_ms"], f"finished cell lap time {key}", minimum=0)
        else:
            _require(row["lap_time_ms"] is None, f"non-finish has a lap time at {key}: {directory}")
        observed[key] = row
        grouped[key[:2]].append(row)
    _require(set(observed) == grid, f"incomplete evaluator cell matrix: {directory}")
    _require(len(audits) == len(grid) // 2, f"wrong determinism audit count: {directory}")
    audit_keys = set()
    for audit in audits:
        _require(set(audit) == {"audited", "candidate_id", "matches_canonical", "repeats", "seed", "track_id"},
                 f"unsupported determinism audit schema: {directory}")
        key = (audit["track_id"], audit["seed"])
        _require(key not in audit_keys and key in grouped and audit["candidate_id"] == result["candidate_id"]
                 and audit["repeats"] == 2 and audit["audited"] is True
                 and audit["matches_canonical"] is True,
                 f"invalid or incomplete reload audit for {key}: {directory}")
        audit_keys.add(key)
        pair = sorted(grouped[key], key=lambda row: row["repeat"])
        _require([row["repeat"] for row in pair] == [0, 1]
                 and tuple(pair[0][field] for field in TRACE_FIELDS)
                 == tuple(pair[1][field] for field in TRACE_FIELDS),
                 f"reload outcome/trace mismatch for {key}: {directory}")

    canonical = [row for key, row in observed.items() if key[2] == 0]
    finishes = sum(row["finished"] for row in canonical)
    progress = sum(row["progress"] for row in canonical) / len(canonical)
    lap_times = [row["lap_time_ms"] for row in canonical if row["finished"]]
    reported = result["summary"]
    _require(reported["n_episodes"] == len(canonical)
             and math.isclose(reported["finish_rate"], finishes / len(canonical), abs_tol=1e-12)
             and math.isclose(reported["avg_progress"], progress, abs_tol=1e-12)
             and (reported["avg_lap_time_ms"] is None if not lap_times else
                  math.isclose(reported["avg_lap_time_ms"], sum(lap_times) / len(lap_times), abs_tol=1e-12)),
             f"evaluator canonical summary disagrees with episode rows: {directory}")
    return {
        "wrapper": wrapper,
        "wrapper_sha256": sha256_file(Path(wrapper["evaluator_receipt_path"]).resolve()),
        "evaluator_receipt_path": str(Path(wrapper["evaluator_receipt_path"]).resolve()),
        "evaluation_dir": str(directory),
        "candidate_id": result["candidate_id"],
        "result": result,
        "canonical": {(row["track_id"], row["seed"]): row for row in canonical},
        "finishes": finishes,
        "cell_count": len(canonical),
        "manifest": manifest,
    }


def _load_receipts(
    paths: list[Path], protocol: dict[str, Any], protocol_sha256: str, partition: str
) -> list[dict[str, Any]]:
    expected_count = 10 if partition == "screen" else 6 if partition == "confirmation" else 1
    _require(len(paths) == expected_count, f"{partition} requires exactly {expected_count} actor receipts")
    resolved_paths = [Path(path).resolve() for path in paths]
    _require(len(set(resolved_paths)) == len(resolved_paths), "duplicate actor receipt file paths")
    reports = []
    for path in resolved_paths:
        wrapper, wrapper_sha, report = _validate_wrapper(path, protocol, protocol_sha256, partition)
        report["wrapper"] = wrapper
        report["wrapper_sha256"] = wrapper_sha
        report["path"] = str(path)
        reports.append(report)
        if partition == "screen":
            _require(set(wrapper) == WRAPPER_KEYS,
                     "screen actor receipt must not carry preceding-partition lineage fields")
    identities = [_identity_key(report["wrapper"]) for report in reports]
    _require(len(identities) == len(set(identities)), "duplicate source/arm/checkpoint actor roles")
    if partition == "screen":
        _require(set(identities) == _expected_wrapper_identities(partition, protocol),
                 f"{partition} receipts have missing or extra source/arm/checkpoint identities")
    elif partition == "confirmation":
        expected_roles = {(learner, arm) for learner in (0, 1) for arm in ARMS}
        actual_roles = {(learner, arm) for learner, arm, _ in identities}
        _require(actual_roles == expected_roles,
                 "confirmation receipts have missing or duplicate source/arm roles")
        _require(all(step is None if arm == "unchanged-source" else
                     type(step) is int and step in protocol["budgets"]["checkpoint_online_steps"]
                     for _, arm, step in identities),
                 "confirmation checkpoint identity is not screen-predeclared")
    else:
        only = reports[0]["wrapper"]
        _require(only["arm"] == "teacher-replay", "blind requires one teacher-replay treatment actor")
    return reports


def _role_map(reports: list[dict[str, Any]]) -> dict[tuple[int, str, int | None], dict[str, Any]]:
    return {_identity_key(report["wrapper"]): report for report in reports}


def _pair_stats(
    left: dict[tuple[int, int], dict[str, Any]],
    right: dict[tuple[int, int], dict[str, Any]],
    seeds: list[int],
    track_ids: list[int],
) -> dict[str, Any]:
    wins = losses = ties = 0
    geometry = []
    details = []
    per_seed: dict[int, list[int]] = {seed: [] for seed in seeds}
    for track in track_ids:
        for seed in seeds:
            cell = (track, seed)
            left_finish = left[cell]["finished"]
            right_finish = right[cell]["finished"]
            delta = int(right_finish) - int(left_finish)
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1
            per_seed[seed].append(delta)
            details.append({
                "track_id": track,
                "seed": seed,
                "reference_finished": left_finish,
                "arm_finished": right_finish,
                "finish_delta": delta,
            })
    for seed in seeds:
        deltas = per_seed[seed]
        cluster_mean = sum(deltas) / len(deltas)
        geometry.append({"seed": seed, "wins": sum(delta > 0 for delta in deltas),
                         "losses": sum(delta < 0 for delta in deltas),
                         "ties": sum(delta == 0 for delta in deltas),
                         "mean_finish_delta": cluster_mean})
    cluster_values = [row["mean_finish_delta"] for row in geometry]
    mean = sum(cluster_values) / len(cluster_values)
    sd = (sum((value - mean) ** 2 for value in cluster_values) / (len(cluster_values) - 1)) ** 0.5 \
        if len(cluster_values) > 1 else None
    return {
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "canonical_cells": len(details),
        "cells": details,
        "geometry_seed_clusters": geometry,
        "geometry_clustered_mean_finish_delta": mean,
        "geometry_clustered_sample_sd": sd,
        "geometry_clustered_standard_error": sd / math.sqrt(len(cluster_values)) if sd is not None else None,
        "geometry_clustered_range": [min(cluster_values), max(cluster_values)],
        "uncertainty_interpretation": "descriptive geometry-cluster variation only; not a significance test",
    }


def _screen_result(reports: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    roles = _role_map(reports)
    matrix = protocol["partitions"]["screen"]
    result = {"selection_performed": False, "sources": []}
    for learner in (0, 1):
        source = roles[(learner, "unchanged-source", None)]
        arms = []
        for arm in STUDENT_ARMS:
            for step in protocol["budgets"]["checkpoint_online_steps"]:
                candidate = roles[(learner, arm, step)]
                record = {
                    "arm": arm,
                    "checkpoint_online_step": step,
                    "actor_sha256": candidate["wrapper"]["actor_sha256"],
                    "checkpoint_sha256": candidate["wrapper"]["checkpoint_sha256"],
                    "finish_count": candidate["finishes"],
                    "canonical_cells": candidate["cell_count"],
                    "source_comparison": _pair_stats(
                        source["canonical"], candidate["canonical"], matrix["seeds"], matrix["track_ids"]
                    ),
                }
                if arm == "teacher-replay":
                    online = roles[(learner, "online-only", step)]
                    record["online_only_comparison"] = _pair_stats(
                        online["canonical"], candidate["canonical"], matrix["seeds"], matrix["track_ids"]
                    )
                    record["screen_only_eligible"] = (
                        candidate["finishes"] > 0 and candidate["finishes"] >= online["finishes"]
                    )
                arms.append(record)
        result["sources"].append({
            "source_learner_seed": learner,
            "unchanged_source_finish_count": source["finishes"],
            "canonical_cells": source["cell_count"],
            "candidates": arms,
        })
    return result


def _read_lineages(paths: list[Path], expected: int, label: str) -> list[tuple[Path, dict[str, Any], str]]:
    _require(len(paths) == expected, f"{label} requires exactly {expected} lineage pointers")
    resolved = [Path(path).resolve() for path in paths]
    _require(len(set(resolved)) == len(resolved), f"duplicate {label} lineage pointers")
    result = []
    for path in resolved:
        wrapper, digest = _read_wrapper(path)
        result.append((path, wrapper, digest))
    return result


def _confirmation_result(
    reports: list[dict[str, Any]],
    screen_reports: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    roles = {(report["wrapper"]["source_learner_seed"], report["wrapper"]["arm"]): report
             for report in reports}
    screen_roles = _role_map(screen_reports)
    matrix = protocol["partitions"]["confirmation"]
    sources = []
    passed_all = True
    for learner in (0, 1):
        source = roles[(learner, "unchanged-source")]
        online = roles[(learner, "online-only")]
        teacher = roles[(learner, "teacher-replay")]
        screen_teacher = screen_roles[(learner, "teacher-replay", teacher["wrapper"]["checkpoint_online_step"])]
        screen_online = screen_roles[(learner, "online-only", online["wrapper"]["checkpoint_online_step"])]
        # Screen checkpoint identity is fixed externally; this comparator only checks that binding.
        screen_eligible = screen_teacher["finishes"] > 0 and screen_teacher["finishes"] >= screen_online["finishes"]
        against_online = _pair_stats(online["canonical"], teacher["canonical"], matrix["seeds"], matrix["track_ids"])
        against_source = _pair_stats(source["canonical"], teacher["canonical"], matrix["seeds"], matrix["track_ids"])
        gain_seeds = [row["seed"] for row in against_online["geometry_seed_clusters"] if row["wins"] > 0]
        hurdle_online = teacher["finishes"] >= online["finishes"] + 2
        hurdle_source = teacher["finishes"] >= source["finishes"]
        nonzero = teacher["finishes"] > 0
        passed = screen_eligible and hurdle_online and hurdle_source and len(gain_seeds) >= 2 and nonzero
        passed_all = passed_all and passed
        sources.append({
            "source_learner_seed": learner,
            "screen_only_eligible": screen_eligible,
            "finish_counts": {
                "unchanged_source": source["finishes"],
                "online_only": online["finishes"],
                "teacher_replay": teacher["finishes"],
                "canonical_cells": teacher["cell_count"],
            },
            "hurdles": {
                "teacher_at_least_online_only_plus_2": hurdle_online,
                "teacher_at_least_unchanged_source": hurdle_source,
                "gains_on_at_least_two_distinct_geometry_seeds": len(gain_seeds) >= 2,
                "nonzero_teacher_finishes": nonzero,
                "no_operational_failure": True,
            },
            "gain_geometry_seeds_vs_online_only": gain_seeds,
            "teacher_vs_online_only": against_online,
            "teacher_vs_unchanged_source": against_source,
            "passed": passed,
        })
    return {"passed": passed_all, "sources": sources}


def _validate_decision_receipt(path: Path, protocol_sha256: str, partition: str) -> tuple[dict[str, Any], str]:
    decision, digest = _sealed_json(path, "paired decision receipt")
    _require(set(decision) == {
        "schema_version", "receipt_type", "partition", "study_protocol_sha256", "input_receipts",
        "lineage_receipts", "selection_performed", "result",
    }, "paired decision receipt has missing or unsupported schema fields")
    _require(decision.get("schema_version") == 1 and decision.get("receipt_type") == DECISION_TYPE,
             "unsupported paired decision receipt schema")
    _require(decision.get("study_protocol_sha256") == protocol_sha256
             and decision.get("partition") == partition,
             "paired decision lineage has wrong protocol or partition")
    _require(decision["selection_performed"] is False and isinstance(decision["input_receipts"], list)
             and isinstance(decision["lineage_receipts"], list) and isinstance(decision["result"], dict),
             "paired decision receipt has invalid selection or evidence fields")
    _require(all(isinstance(item, dict) and set(item) == {
        "path", "sha256", "identity", "evaluator_receipt_path", "evaluator_receipt_sha256",
    } and SHA256_RE.fullmatch(item["sha256"]) is not None
                and SHA256_RE.fullmatch(item["evaluator_receipt_sha256"]) is not None
                and set(item["identity"]) == {
                    "source_learner_seed", "arm", "actor_sha256", "checkpoint_sha256",
                    "checkpoint_online_step",
                }
                for item in decision["input_receipts"]),
             "paired decision receipt has invalid input receipt hashes")
    for item in decision["input_receipts"]:
        identity = item["identity"]
        _require(type(identity["source_learner_seed"]) is int and identity["source_learner_seed"] in (0, 1)
                 and identity["arm"] in ARMS,
                 "paired decision contains invalid source/arm identity")
        _digest(identity["actor_sha256"], "decision actor_sha256")
        if identity["arm"] == "unchanged-source":
            _require(identity["checkpoint_online_step"] is None
                     and (identity["checkpoint_sha256"] is None
                          or SHA256_RE.fullmatch(identity["checkpoint_sha256"]) is not None),
                     "paired decision contains invalid unchanged-source checkpoint identity")
        else:
            _digest(identity["checkpoint_sha256"], "decision checkpoint_sha256")
            _require(type(identity["checkpoint_online_step"]) is int
                     and identity["checkpoint_online_step"] in (16384, 32768),
                     "paired decision contains an unpredeclared checkpoint step")
        _require(all(isinstance(item[field], str) and Path(item[field]).is_absolute()
                     for field in ("path", "evaluator_receipt_path")),
                 "paired decision input paths must be absolute")
    _require(all(isinstance(item, dict) and set(item) == {"path", "sha256"}
                 and SHA256_RE.fullmatch(item["sha256"]) is not None
                 for item in decision["lineage_receipts"]),
             "paired decision receipt has invalid lineage hashes")
    if partition == "confirmation":
        _require(set(decision["result"]) == {"passed", "sources", "screen_eligibility"}
                 and type(decision["result"]["passed"]) is bool
                 and len(decision["input_receipts"]) == 6,
                 "confirmation decision result schema mismatch")
        roles = {(item["identity"]["source_learner_seed"], item["identity"]["arm"])
                 for item in decision["input_receipts"]}
        _require(len(roles) == 6 and roles == {(learner, arm) for learner in (0, 1) for arm in ARMS},
                 "confirmation decision does not contain all six source/arm roles")
    elif partition == "screen":
        _require(set(decision["result"]) == {"selection_performed", "sources"}
                 and decision["result"]["selection_performed"] is False,
                 "screen decision result schema mismatch")
    return decision, digest


def _validate_finalist_pointer(
    path: Path,
    protocol: dict[str, Any],
    protocol_sha256: str,
    screen_decision: dict[str, Any],
    confirmation_started_at: str,
) -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any]]:
    finalist, digest = _sealed_json(path, "screen-fixed finalist pointer")
    expected_keys = {
        "schema_version", "receipt_type", "study_protocol_sha256", "partition",
        "source_learner_seed", "arm", "actor_sha256", "checkpoint_sha256",
        "checkpoint_online_step", "screen_receipt_sha256", "screen_decision_path",
        "screen_decision_sha256", "selection_file_path", "selection_file_sha256", "fixed_at_utc",
    }
    _require(set(finalist) == expected_keys, "finalist pointer has missing or unsupported schema fields")
    _require(finalist["schema_version"] == 1 and finalist["receipt_type"] == FINALIST_TYPE
             and finalist["study_protocol_sha256"] == protocol_sha256
             and finalist["partition"] == "screen"
             and finalist["arm"] == "teacher-replay",
             "finalist pointer is not a screen-fixed teacher-replay candidate")
    _require(Path(finalist["screen_decision_path"]).resolve() == Path(screen_decision["_path"]).resolve()
             and finalist["screen_decision_sha256"] == screen_decision["_sha256"],
             "finalist pointer does not bind the screen paired decision")
    selection_path = Path(finalist["selection_file_path"]).resolve()
    selection_sha = _digest(finalist["selection_file_sha256"], "selection_file_sha256")
    _require(selection_path.is_file() and sha256_file(selection_path) == selection_sha,
             "selected-candidates file hash mismatch")
    selection = _validate_selection_record(selection_path, protocol, protocol_sha256,
                                           screen_decision, finalist)
    screen_record = selection["blind_finalist"]
    screen_path = Path(screen_record["screen_wrapper_path"]).resolve()
    screen_wrapper, screen_sha, screen_report = _validate_wrapper(
        screen_path, protocol, protocol_sha256, "screen"
    )
    _require(screen_sha == finalist["screen_receipt_sha256"]
             and screen_sha == screen_record["screen_wrapper_sha256"]
             and _identity(screen_wrapper) == {
                 "source_learner_seed": finalist["source_learner_seed"],
                 "arm": finalist["arm"],
                 "actor_sha256": finalist["actor_sha256"],
                 "checkpoint_sha256": finalist["checkpoint_sha256"],
                 "checkpoint_online_step": finalist["checkpoint_online_step"],
             }, "finalist pointer identity differs from its screen actor receipt")
    source_result = next((item for item in screen_decision["result"]["sources"]
                          if item["source_learner_seed"] == finalist["source_learner_seed"]), None)
    candidate_result = next((item for item in (source_result or {}).get("candidates", [])
                             if item["arm"] == "teacher-replay"
                             and item["checkpoint_online_step"] == finalist["checkpoint_online_step"]), None)
    _require(candidate_result is not None and candidate_result["screen_only_eligible"] is True,
             "screen-finalist was not screen-only eligible")
    try:
        fixed_at = datetime.fromisoformat(finalist["fixed_at_utc"].replace("Z", "+00:00"))
        confirmation_at = datetime.fromisoformat(confirmation_started_at.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("invalid finalist/confirmation timestamp") from error
    _require(fixed_at.tzinfo is not None and confirmation_at.tzinfo is not None,
             "finalist/confirmation timestamps must include a timezone")
    _require(fixed_at < confirmation_at, "blind finalist was not fixed before confirmation began")
    return finalist, digest, screen_wrapper, screen_sha, screen_report


def _validate_selection_record(
    path: Path,
    protocol: dict[str, Any],
    protocol_sha256: str,
    screen_decision: dict[str, Any],
    finalist: dict[str, Any],
) -> dict[str, Any]:
    selection = _read_json(path)
    expected_keys = {
        "format", "study_protocol_sha256", "screen_decision_path", "screen_decision_sha256",
        "candidate_receipts", "selected_identities", "selection_rule", "blind_finalist",
    }
    _require(isinstance(selection, dict) and set(selection) == expected_keys,
             "selected-candidates record has missing or unsupported schema fields")
    _require(selection["format"] == SELECTION_FORMAT
             and selection["study_protocol_sha256"] == protocol_sha256
             and Path(selection["screen_decision_path"]).resolve() == Path(screen_decision["_path"]).resolve()
             and selection["screen_decision_sha256"] == screen_decision["_sha256"],
             "selected-candidates record has wrong protocol or screen decision lineage")
    _require(isinstance(selection["selection_rule"], str) and bool(selection["selection_rule"].strip()),
             "selected-candidates record lacks its frozen selection rule")

    candidate_keys = {"candidate_identity", "wrapper_path", "wrapper_sha256"}
    candidates = selection["candidate_receipts"]
    _require(isinstance(candidates, list) and len(candidates) == 10,
             "selected-candidates record must bind all ten screen receipts")
    candidate_map: dict[str, tuple[dict[str, Any], str]] = {}
    candidate_roles = set()
    for candidate in candidates:
        _require(isinstance(candidate, dict) and set(candidate) == candidate_keys,
                 "selected-candidates record has unsupported candidate receipt schema")
        wrapper_path = Path(candidate["wrapper_path"]).resolve()
        wrapper_sha = _digest(candidate["wrapper_sha256"], "candidate wrapper_sha256")
        _require(wrapper_path.is_file() and sha256_file(wrapper_path) == wrapper_sha,
                 "selected-candidates screen wrapper hash mismatch")
        wrapper, _ = _read_wrapper(wrapper_path)
        _require(set(wrapper) == WRAPPER_KEYS and wrapper.get("partition") == "screen"
                 and wrapper.get("study_protocol_sha256") == protocol_sha256,
                 "selected-candidates entry is not a screen actor receipt for this protocol")
        identity = _identity(wrapper)
        _require(candidate["candidate_identity"] == identity,
                 "selected-candidates actor identity differs from its wrapper")
        key = _identity_key(wrapper)
        _require(key not in candidate_roles, "selected-candidates has duplicate source/arm/checkpoint roles")
        candidate_roles.add(key)
        candidate_map[wrapper_sha] = (wrapper, str(wrapper_path))
    _require(candidate_roles == _expected_wrapper_identities("screen", protocol),
             "selected-candidates record has missing or extra screen actor roles")
    decision_inputs = {item["sha256"]: item for item in screen_decision["input_receipts"]}
    _require(set(decision_inputs) == set(candidate_map),
             "screen decision and selected-candidates receipt sets differ")
    for wrapper_sha, (wrapper, wrapper_path) in candidate_map.items():
        item = decision_inputs[wrapper_sha]
        _require(Path(item["path"]).resolve() == Path(wrapper_path)
                 and item["identity"] == _identity(wrapper)
                 and Path(item["evaluator_receipt_path"]).resolve()
                 == Path(wrapper["evaluator_receipt_path"]).resolve()
                 and item["evaluator_receipt_sha256"] == wrapper["evaluator_receipt_sha256"],
                 "screen decision does not bind its selected-candidates actor receipt")

    selected_keys = {
        "source_learner_seed", "arm", "checkpoint_online_step", "actor_sha256", "checkpoint_sha256",
        "screen_wrapper_path", "screen_wrapper_sha256", "score_tuple",
    }
    selected = selection["selected_identities"]
    _require(isinstance(selected, list) and len(selected) == 6,
             "selected-candidates record must freeze six source/arm actors")
    selected_roles = set()
    selected_by_role: dict[tuple[int, str], dict[str, Any]] = {}
    for item in selected:
        _require(isinstance(item, dict) and set(item) == selected_keys,
                 "selected-candidates identity has missing or unsupported fields")
        learner, arm, step = item["source_learner_seed"], item["arm"], item["checkpoint_online_step"]
        _require(type(learner) is int and learner in (0, 1) and arm in ARMS
                 and (step is None if arm == "unchanged-source" else
                      type(step) is int and step in protocol["budgets"]["checkpoint_online_steps"]),
                 "selected-candidates identity has invalid source/arm/checkpoint")
        role = (learner, arm)
        _require(role not in selected_roles, "selected-candidates repeats a source/arm role")
        selected_roles.add(role)
        wrapper_sha = _digest(item["screen_wrapper_sha256"], "screen_wrapper_sha256")
        _require(wrapper_sha in candidate_map, "selected actor is not one of the screen candidate receipts")
        wrapper, wrapper_path = candidate_map[wrapper_sha]
        _require(Path(item["screen_wrapper_path"]).resolve() == Path(wrapper_path)
                 and item["source_learner_seed"] == wrapper["source_learner_seed"]
                 and item["arm"] == wrapper["arm"]
                 and item["checkpoint_online_step"] == wrapper["checkpoint_online_step"]
                 and item["actor_sha256"] == wrapper["actor_sha256"]
                 and item["checkpoint_sha256"] == wrapper["checkpoint_sha256"],
                 "selected-candidates identity does not match its frozen screen wrapper")
        _require(isinstance(item["score_tuple"], list) and bool(item["score_tuple"])
                 and all(type(value) in (int, float) and math.isfinite(value) for value in item["score_tuple"]),
                 "selected-candidates score_tuple must contain finite numbers")
        selected_by_role[role] = item
    _require(selected_roles == {(learner, arm) for learner in (0, 1) for arm in ARMS},
             "selected-candidates lacks a source/arm role")

    blind = selection["blind_finalist"]
    _require(isinstance(blind, dict) and set(blind) == selected_keys
             and blind["arm"] == "teacher-replay",
             "selected-candidates blind_finalist has invalid schema or arm")
    identity = (blind["source_learner_seed"], blind["arm"])
    selected_blind = selected_by_role.get(identity)
    _require(selected_blind is not None and blind == selected_blind,
             "blind finalist is not the exact screen-selected treatment actor")
    _require(identity == (finalist["source_learner_seed"], "teacher-replay")
             and blind["screen_wrapper_sha256"] == finalist["screen_receipt_sha256"]
             and blind["actor_sha256"] == finalist["actor_sha256"]
             and blind["checkpoint_sha256"] == finalist["checkpoint_sha256"]
             and blind["checkpoint_online_step"] == finalist["checkpoint_online_step"],
             "finalist pointer differs from the frozen blind finalist")
    other_teacher = selected_by_role[(1 - identity[0], "teacher-replay")]
    if blind["score_tuple"] == other_teacher["score_tuple"]:
        _require(blind["source_learner_seed"] == 0,
                 "blind finalist violates the learner-0 exact-tie breaker")
    return selection


def _blind_result(report: dict[str, Any], confirmation_decision: dict[str, Any]) -> dict[str, Any]:
    finishes = report["finishes"]
    accepted = confirmation_decision["result"].get("passed") is True and finishes > 0
    return {
        "confirmation_passed": confirmation_decision["result"].get("passed") is True,
        "screen_fixed_treatment_finalist": True,
        "canonical_blind_finishes": finishes,
        "canonical_blind_cells": report["cell_count"],
        "canonical_outcomes": [
            {"track_id": track, "seed": seed, "finished": row["finished"],
             "progress": row["progress"], "lap_time_ms": row["lap_time_ms"]}
            for (track, seed), row in sorted(report["canonical"].items())
        ],
        "two_repeat_trace_agreement": True,
        "accepted": accepted,
        "reason": "accepted" if accepted else "no canonical blind finish" if finishes == 0
                  else "confirmation did not pass",
    }


def compare_receipts(
    protocol: dict[str, Any],
    protocol_sha256: str,
    partition: str,
    receipt_paths: list[Path],
    lineage_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Validate sealed individual receipts and return a separate paired decision."""
    _digest(protocol_sha256, "study protocol SHA-256")
    _require(partition in ("screen", "confirmation", "blind"), "unsupported comparison partition")
    lineage_paths = list(lineage_paths or [])
    reports = _load_receipts(receipt_paths, protocol, protocol_sha256, partition)
    result: dict[str, Any]
    lineage_records: list[dict[str, str]] = []

    if partition == "screen":
        _require(not lineage_paths, "screen comparison must not have lineage pointers")
        result = _screen_result(reports, protocol)
    elif partition == "confirmation":
        lineages = _read_lineages(lineage_paths, 6, "confirmation")
        screen_reports = []
        for path, wrapper, digest in lineages:
            _require(set(wrapper) in (WRAPPER_KEYS, WRAPPER_KEYS | WRAPPER_LINEAGE_KEYS),
                     f"unsupported screen actor receipt schema: {path}")
            _require(wrapper.get("partition") == "screen", f"confirmation lineage is not a screen receipt: {path}")
            verified, verified_digest, report = _validate_wrapper(path, protocol, protocol_sha256, "screen")
            _require(verified_digest == digest, f"confirmation screen lineage hash mismatch: {path}")
            screen_reports.append({**report, "wrapper": verified, "wrapper_sha256": verified_digest,
                                   "path": str(path)})
            lineage_records.append({"path": str(path), "sha256": digest})
        screen_roles = _role_map(screen_reports)
        selected_roles = {(learner, arm) for learner, arm, _ in screen_roles}
        _require(selected_roles == {(learner, arm) for learner in (0, 1) for arm in ARMS}
                 and len(screen_roles) == 6,
                 "confirmation requires one screen-frozen actor for every source/arm role")
        seen_screen_hashes = set()
        for report in reports:
            wrapper = report["wrapper"]
            _require(WRAPPER_LINEAGE_KEYS <= set(wrapper),
                     "confirmation receipt lacks screen lineage fields")
            screen_sha = wrapper["screen_receipt_sha256"]
            _digest(screen_sha, "screen_receipt_sha256")
            step_key = None if wrapper["arm"] == "unchanged-source" else wrapper["checkpoint_online_step"]
            screen = screen_roles.get((wrapper["source_learner_seed"], wrapper["arm"], step_key))
            _require(screen is not None and screen_sha == screen["wrapper_sha256"],
                     "confirmation actor was re-selected or has the wrong screen lineage")
            _require(wrapper["screen_candidate_identity"] == _identity(screen["wrapper"]),
                     "confirmation screen_candidate_identity mismatch")
            _require(_identity(wrapper) == _identity(screen["wrapper"]),
                     "confirmation actor/checkpoint identity differs from screen")
            _require(screen_sha not in seen_screen_hashes, "duplicate screen lineage role")
            seen_screen_hashes.add(screen_sha)
            _validate_previous_lineage(
                Path(report["evaluation_dir"]), report["manifest"]["previous_evaluation"], wrapper,
                "confirmation", Path(screen["evaluator_receipt_path"]),
            )
        result = _confirmation_result(reports, screen_reports, protocol)
        screen_selection = {(report["wrapper"]["source_learner_seed"], report["wrapper"]["arm"]): report
                            for report in screen_reports}
        result["screen_eligibility"] = {}
        for learner in (0, 1):
            screen_teacher = screen_selection[(learner, "teacher-replay")]
            screen_online = screen_selection[(learner, "online-only")]
            result["screen_eligibility"][str(learner)] = (
                screen_teacher["finishes"] > 0 and screen_teacher["finishes"] >= screen_online["finishes"]
            )
        result["passed"] = result["passed"] and all(result["screen_eligibility"].values())
    else:
        lineages = _read_lineages(lineage_paths, 2, "blind")
        decision_entries = [(path, value, digest) for path, value, digest in lineages
                            if value.get("receipt_type") == DECISION_TYPE
                            and value.get("partition") == "confirmation"]
        finalist_entries = [(path, value, digest) for path, value, digest in lineages
                            if value.get("receipt_type") == FINALIST_TYPE]
        _require(len(decision_entries) == len(finalist_entries) == 1,
                 "blind requires one screen-finalist pointer and one confirmation decision receipt")
        decision_path, _, decision_sha = decision_entries[0]
        decision, verified_decision_sha = _validate_decision_receipt(
            decision_path, protocol_sha256, "confirmation"
        )
        _require(verified_decision_sha == decision_sha and decision["result"]["passed"] is True,
                 "blind requires a passed paired confirmation decision")
        finalist_path, _, finalist_sha = finalist_entries[0]
        finalist_value, _ = _sealed_json(finalist_path, "screen-fixed finalist pointer")
        _require(len(decision["lineage_receipts"]) == 6,
                 "confirmation decision lacks the six selected screen lineages")
        _require(any(item["sha256"] == finalist_value["screen_receipt_sha256"]
                     for item in decision["lineage_receipts"]),
                 "confirmation decision does not carry the screen-fixed finalist lineage")
        finalist_identity = {
            "source_learner_seed": finalist_value["source_learner_seed"],
            "arm": finalist_value["arm"],
            "actor_sha256": finalist_value["actor_sha256"],
            "checkpoint_sha256": finalist_value["checkpoint_sha256"],
            "checkpoint_online_step": finalist_value["checkpoint_online_step"],
        }
        confirmation_matches = [item for item in decision["input_receipts"]
                                if item["identity"] == finalist_identity]
        _require(len(confirmation_matches) == 1,
                 "passed confirmation decision does not contain the screen finalist identity")
        confirmation_input = confirmation_matches[0]
        confirmation_path = Path(confirmation_input["path"]).resolve()
        prior_confirmation, confirmation_sha, confirmation_report = _validate_wrapper(
            confirmation_path, protocol, protocol_sha256, "confirmation"
        )
        _require(confirmation_sha == confirmation_input["sha256"]
                 and prior_confirmation["evaluator_receipt_path"] == confirmation_input["evaluator_receipt_path"]
                 and prior_confirmation["evaluator_receipt_sha256"] == confirmation_input["evaluator_receipt_sha256"],
                 "confirmation decision receipt does not match its sealed actor wrapper")
        screen_decision_path = Path(finalist_value["screen_decision_path"]).resolve()
        screen_decision, screen_decision_sha = _validate_decision_receipt(
            screen_decision_path, protocol_sha256, "screen"
        )
        screen_decision["_path"] = str(screen_decision_path)
        screen_decision["_sha256"] = screen_decision_sha
        finalist, checked_finalist_sha, screen, screen_sha, screen_report = _validate_finalist_pointer(
            finalist_path, protocol, protocol_sha256, screen_decision,
            confirmation_report["manifest"]["started_at_utc"],
        )
        _require(checked_finalist_sha == finalist_sha
                 and finalist["screen_receipt_sha256"] == screen_sha,
                 "blind finalist pointer is not the exact screen actor lineage")
        blind = reports[0]
        wrapper = blind["wrapper"]
        _require(screen["arm"] == prior_confirmation["arm"] == wrapper["arm"] == "teacher-replay"
                 and _identity(screen) == _identity(prior_confirmation) == _identity(wrapper),
                 "blind actor differs from the screen-fixed, confirmed treatment")
        _require(prior_confirmation.get("screen_receipt_sha256") == screen_sha
                 and prior_confirmation.get("screen_candidate_identity") == _identity(screen),
                 "confirmation actor does not bind the screen-fixed finalist")
        _validate_previous_lineage(
            Path(confirmation_report["evaluation_dir"]),
            confirmation_report["manifest"]["previous_evaluation"],
            prior_confirmation,
            "confirmation",
            Path(screen_report["evaluator_receipt_path"]),
        )
        _require(WRAPPER_LINEAGE_KEYS <= set(wrapper)
                 and wrapper["screen_receipt_sha256"] == screen_sha
                 and wrapper["screen_candidate_identity"] == _identity(screen),
                 "blind receipt has wrong screen finalist lineage")
        confirmation_pointer_path = Path(prior_confirmation["evaluator_receipt_path"]).resolve()
        _validate_previous_lineage(
            Path(blind["evaluation_dir"]), blind["manifest"]["previous_evaluation"], wrapper, "blind",
            confirmation_pointer_path,
        )
        result = _blind_result(blind, decision)
        lineage_records = [
            {"path": str(decision_path), "sha256": decision_sha},
            {"path": str(finalist_path), "sha256": finalist_sha},
        ]

    inputs = [
        {"path": report["path"], "sha256": report["wrapper_sha256"], "identity": _identity(report["wrapper"]),
         "evaluator_receipt_path": report["evaluator_receipt_path"],
         "evaluator_receipt_sha256": report["wrapper"]["evaluator_receipt_sha256"]}
        for report in reports
    ]
    inputs.sort(key=lambda item: (item["identity"]["source_learner_seed"], item["identity"]["arm"],
                                  item["identity"]["checkpoint_online_step"] or -1))
    return {
        "schema_version": 1,
        "receipt_type": DECISION_TYPE,
        "partition": partition,
        "study_protocol_sha256": protocol_sha256,
        "input_receipts": inputs,
        "lineage_receipts": lineage_records,
        "selection_performed": False,
        "result": result,
    }


def load_protocol(path: Path, repo_root: Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    protocol = json.loads(raw, parse_constant=_reject_constant)
    _require(isinstance(protocol, dict), "study protocol must be a JSON object")
    try:
        validate_protocol_file(path, repo_root.resolve())
    except ProtocolError as error:
        raise ValueError(f"invalid frozen study protocol: {error}") from error
    _require(sha256_file(path) == digest, "study protocol changed while it was being validated")
    return protocol, digest


def write_sealed(path: Path, value: dict[str, Any]) -> str:
    path = Path(path).resolve()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    _require(path.parent.is_dir(), f"output parent directory does not exist: {path.parent}")
    _require(not path.exists() and not sidecar.exists(), f"refusing to overwrite output: {path}")
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    with path.open("xb") as target:
        target.write(raw)
    with sidecar.open("xb") as target:
        target.write((digest + "\n").encode("ascii"))
    return digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-file", required=True, type=Path)
    parser.add_argument("--partition", required=True, choices=("screen", "confirmation", "blind"))
    parser.add_argument("--receipt", required=True, action="append", type=Path,
                        help="individual actor wrapper binding an evaluator receipt SHA; repeat per actor")
    parser.add_argument("--lineage-pointer", action="append", default=[], type=Path,
                        help="predeclared earlier actor/decision/finalist receipt required outside screen")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        protocol, protocol_sha = load_protocol(args.protocol_file, args.repo_root)
        output = args.output.resolve()
        input_paths = [Path(path).resolve() for path in (args.protocol_file, *args.receipt, *args.lineage_pointer)]
        _require(output not in input_paths, "decision receipt output cannot overwrite an input")
        decision = compare_receipts(
            protocol, protocol_sha, args.partition, args.receipt, args.lineage_pointer
        )
        output_sha = write_sealed(output, decision)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(output), "sha256": output_sha, "partition": args.partition}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
