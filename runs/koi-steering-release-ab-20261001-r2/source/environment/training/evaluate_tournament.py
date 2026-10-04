"""Run an immutable completion-first batch on identical local map/seed cells."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from haic_research.models import ExperimentResult, GateStatus, WorkflowState
from haic_research.policy import rank_candidates
from haic_research.tournament import referenced_files, validate_candidate_manifest
from haic_research.commands import load_run_plan, replay_run_history
from haic_research.config import load_config
from training.evaluate_closed_loop import evaluate_mode
from training.site_maps import SiteMapEpisode, load_site_map_group


_CANDIDATE_FAILURES = frozenset({
    "agent_setup_error", "agent_reset_error", "reset_timeout", "invalid_action", "act_timeout",
})
_MISSING_TELEMETRY_PENALTY = 1_000_000.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _project_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("candidate input must remain inside the repository") from error
    if not candidate.is_file():
        raise ValueError("candidate input must be a regular file")
    return candidate


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _registered_tournament_plan(
    *,
    config: Any,
    output_directory: Path,
    candidate_manifest: Path,
    split_path: Path,
    split_group: str,
    control_policy_checkpoint: Path,
    control_dynamics_checkpoint: Path,
    plan_budget_seconds: float,
    max_decisions: int,
) -> dict[str, Any]:
    """Bind runtime inputs to the current approved, already-claimed harness run."""
    if output_directory.parent.parent != config.artifact_root:
        raise ValueError("tournament output must be exactly one registered run below the artifact root")
    run_id = output_directory.parent.name
    run_dir = config.run_root / run_id
    plan = load_run_plan(config, run_dir)
    history = replay_run_history(config, run_dir)
    if (plan.get("profile_id") != "evaluate_closed_loop"
            or history.state != WorkflowState.EVALUATE
            or not history.execution_started
            or history.execution_finished
            or {approval.stage for approval in history.approvals} != {"design", "implementation", "execution"}):
        raise ValueError("tournament execution requires its current fully approved harness run")
    arguments = plan.get("arguments", {})
    expected_paths = {
        "output": output_directory,
        "candidate-manifest": candidate_manifest,
        "site-map-split": split_path,
        "policy-checkpoint": control_policy_checkpoint,
        "dynamics-checkpoint": control_dynamics_checkpoint,
    }
    for name, path in expected_paths.items():
        if Path(arguments.get(name, "")).resolve() != path:
            raise ValueError(f"runtime {name} differs from the approved tournament plan")
    if (arguments.get("split-group") != split_group
            or arguments.get("plan-budget") != float(plan_budget_seconds)
            or arguments.get("max-decisions") != max_decisions):
        raise ValueError("runtime split or evaluation limits differ from the approved tournament plan")
    return plan


def _verify_frozen_tune(
    *,
    root: Path,
    config: Any,
    manifest: Mapping[str, Any],
    control_policy_checkpoint: Path,
    control_dynamics_checkpoint: Path,
    current_split_path: Path,
    plan_budget_seconds: float,
    max_decisions: int,
) -> dict[str, Any] | None:
    """Require held-out to be the byte-bound winner from an approved tune run."""
    if manifest["stage"] != "held_out":
        return None
    source = manifest["tune_source"]
    run_dir = config.run_root / source["run_id"]
    plan = load_run_plan(config, run_dir)
    history = replay_run_history(config, run_dir)
    if plan.get("plan_hash") != source["plan_hash"]:
        raise ValueError("tune source run plan hash does not match the held-out manifest")
    if not history.execution_finished or not history.execution_succeeded:
        raise ValueError("held-out requires a successfully completed, approved tune execution")
    if (history.state != WorkflowState.EVALUATE or history.outcome is not None
            or {approval.stage for approval in history.approvals} != {"design", "implementation", "execution"}):
        raise ValueError("tune source must be fully approved and awaiting its evaluation decision")
    if plan.get("profile_id") != "evaluate_closed_loop":
        raise ValueError("tune source must be a registered closed-loop evaluation")
    run_manifest = plan.get("manifest", {})
    if (run_manifest.get("comparison_id") != manifest["comparison_id"]
            or run_manifest.get("control_revision") != manifest["control_id"]
            or run_manifest.get("candidate_revision") != f"tournament-{manifest['comparison_id']}"
            or run_manifest.get("split_ids") != ["tune"]):
        raise ValueError("tune source run manifest does not match this tournament identity")
    arguments = plan.get("arguments", {})
    if arguments.get("split-group") != "tune":
        raise ValueError("tune source run must use the tune split")
    prior_split = Path(arguments.get("site-map-split", "")).resolve(strict=True)
    if prior_split != current_split_path:
        raise ValueError("held-out must use the exact split manifest registered for tune")
    if (arguments.get("plan-budget") != float(plan_budget_seconds)
            or arguments.get("max-decisions") != max_decisions):
        raise ValueError("held-out must keep the tune planning budget and decision cap")
    source_manifest_path = Path(arguments.get("candidate-manifest", ""))
    if not source_manifest_path.is_file():
        raise ValueError("tune source candidate manifest is unavailable")
    source_manifest = validate_candidate_manifest(
        json.loads(source_manifest_path.read_text(encoding="utf-8")), "tune"
    )
    if (source_manifest["comparison_id"] != manifest["comparison_id"]
            or source_manifest["control_id"] != manifest["control_id"]):
        raise ValueError("held-out comparison/control identity differs from the tune source")
    selected = manifest["candidates"][0]
    source_candidate = next(
        (item for item in source_manifest["candidates"] if item["candidate_id"] == selected["candidate_id"]),
        None,
    )
    if source_candidate != selected:
        raise ValueError("held-out candidate definition must exactly match its registered tune arm")
    for argument_name, current_path in (
        ("policy-checkpoint", control_policy_checkpoint),
        ("dynamics-checkpoint", control_dynamics_checkpoint),
    ):
        prior_path = Path(arguments.get(argument_name, "")).resolve(strict=True)
        if prior_path != current_path:
            raise ValueError("held-out control checkpoints must match the tune comparison")
    summary_path = _project_file(root, source["summary_ref"])
    if _sha256(summary_path) != source["summary_sha256"]:
        raise ValueError("tune summary bytes do not match the held-out manifest")
    expected_summary = (Path(arguments["output"]).resolve(strict=True) / "summary.json")
    if summary_path != expected_summary:
        raise ValueError("tune summary must be the output of the referenced tune run")
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    input_files = plan.get("input_identity", {}).get("files", {})
    source_manifest_identity = input_files.get("candidate-manifest", {})
    split_identity = input_files.get("site-map-split", {})
    if (report.get("candidate_manifest_sha256") != source_manifest_identity.get("sha256")
            or report.get("site_map_split_sha256") != split_identity.get("sha256")
            or report.get("inputs_stable_during_evaluation") is not True):
        raise ValueError("tune summary input fingerprints do not match the approved tune plan")
    if (report.get("checkpoint_sha256", {}).get("control_policy")
            != input_files.get("policy-checkpoint", {}).get("sha256")
            or report.get("checkpoint_sha256", {}).get("control_dynamics")
            != input_files.get("dynamics-checkpoint", {}).get("sha256")):
        raise ValueError("tune summary control checkpoint hashes do not match the approved tune plan")
    for reference in source_manifest_identity.get("references", []):
        relative = reference.get("path")
        expected_hash = reference.get("sha256")
        if (not relative
                or report.get("input_sha256_before", {}).get(f"reference:{relative}") != expected_hash
                or report.get("input_sha256_after", {}).get(f"reference:{relative}") != expected_hash):
            raise ValueError("tune summary hypothesis/checkpoint inputs do not match the approved tune plan")
    for map_input in split_identity.get("maps", []):
        relative = map_input.get("path")
        expected_hash = map_input.get("sha256")
        if (not relative
                or report.get("input_sha256_before", {}).get(f"map:{relative}") != expected_hash
                or report.get("input_sha256_after", {}).get(f"map:{relative}") != expected_hash):
            raise ValueError("tune summary selected map hashes do not match the approved tune plan")
    for reference in source_manifest_identity.get("references", []):
        relative = reference.get("path")
        if relative and relative.lower().endswith((".pt", ".pth")):
            if report.get("checkpoint_sha256", {}).get(relative) != reference.get("sha256"):
                raise ValueError("tune summary candidate checkpoint hashes do not match the approved tune plan")
    if (report.get("status") != "RANKED" or report.get("stage") != "tune"
            or report.get("comparison_id") != manifest["comparison_id"]
            or report.get("control_id") != manifest["control_id"]
            or report.get("plan_budget_seconds") != arguments.get("plan-budget")
            or report.get("max_decisions") != arguments.get("max-decisions")
            or report.get("winner_candidate_id") != selected["candidate_id"]
            or not report.get("ranking")
            or report["ranking"][0].get("candidate_id") != selected["candidate_id"]):
        raise ValueError("tune summary does not record this candidate as the frozen winner")
    return {
        "run_id": source["run_id"],
        "plan_hash": source["plan_hash"],
        "summary_ref": source["summary_ref"],
        "summary_sha256": source["summary_sha256"],
        "winner_candidate_id": selected["candidate_id"],
    }


def _cell_id(episode: SiteMapEpisode) -> str:
    return f"{episode.map_id}@{episode.seed}"


def _as_result(
    *,
    candidate_id: str,
    comparison_id: str,
    split_group: str,
    episodes: tuple[SiteMapEpisode, ...],
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> ExperimentResult:
    map_ids = tuple(sorted({episode.map_id for episode in episodes}))
    seed_ids = tuple(sorted({str(episode.seed) for episode in episodes}))
    return ExperimentResult(
        candidate_id=candidate_id,
        comparison_id=comparison_id,
        split_id=split_group,
        map_ids=map_ids,
        seed_ids=seed_ids,
        completion_count=metrics["completed_episodes"],
        episode_count=len(episodes),
        median_finished_lap_ms=metrics["median_finished_lap_time_ms"],
        mean_incomplete_progress=metrics["mean_incomplete_progress"],
        p90_finished_lap_ms=metrics["p90_finished_lap_time_ms"],
        collisions=metrics["mean_collisions_per_episode"],
        damage=metrics["mean_damage_per_episode"],
        act_latency_p95_ms=metrics["act_latency_p95_ms"],
        rule_compliance=GateStatus.UNKNOWN,
        mechanism_activation=GateStatus.UNKNOWN,
        eligibility="candidate",
    )


def _summarize_arm(records: list[dict[str, Any]], expected_cells: tuple[str, ...]) -> dict[str, Any]:
    actual_cells = [str(record.get("cell_id", "")) for record in records]
    coverage_ok = len(records) == len(expected_cells) and actual_cells == list(expected_cells)
    completed = [record for record in records if record.get("completed") is True]
    incomplete = [record for record in records if record.get("completed") is not True]
    finished_laps = [
        float(record["lapTimeMs"])
        for record in completed
        if isinstance(record.get("lapTimeMs"), (int, float))
        and math.isfinite(float(record["lapTimeMs"]))
        and float(record["lapTimeMs"]) > 0.0
    ]
    valid_finishes = len(finished_laps) == len(completed)
    progress: list[float] = []
    collisions: list[float] = []
    damage: list[float] = []
    latencies: list[float] = []
    telemetry_fields_valid = True
    for record in records:
        candidate_fault = record.get("retire_reason") in _CANDIDATE_FAILURES
        if record.get("completed") is not True:
            value = record.get("progress")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0:
                progress.append(float(value))
            elif candidate_fault:
                progress.append(0.0)
            else:
                telemetry_fields_valid = False
        for name, target, validity_field in (
            ("collisions", collisions, "collision_telemetry_valid"),
            ("damage", damage, "damage_telemetry_valid"),
        ):
            value = record.get(name)
            measurement_valid = (
                record.get(validity_field) is True
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0.0
            )
            if measurement_valid:
                target.append(float(value))
            elif candidate_fault:
                target.append(_MISSING_TELEMETRY_PENALTY)
            else:
                telemetry_fields_valid = False
        raw_latencies = record.get("act_latency_ms")
        if isinstance(raw_latencies, list) and raw_latencies:
            if all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(float(value)) and float(value) >= 0.0 for value in raw_latencies):
                latencies.extend(float(value) for value in raw_latencies)
            else:
                telemetry_fields_valid = False
        elif candidate_fault:
            latencies.append(_MISSING_TELEMETRY_PENALTY)
        else:
            telemetry_fields_valid = False
        if record.get("nonfinite_measurement") is True:
            telemetry_fields_valid = False
    telemetry_ok = coverage_ok and valid_finishes and telemetry_fields_valid
    return {
        "episodes": len(records),
        "registered_denominator": len(expected_cells),
        "completed_episodes": len(completed),
        "completion_rate": len(completed) / len(expected_cells) if expected_cells else 0.0,
        "median_finished_lap_time_ms": float(np.percentile(finished_laps, 50)) if finished_laps else None,
        "mean_incomplete_progress": float(np.mean(progress)) if progress else 0.0,
        "p90_finished_lap_time_ms": float(np.percentile(finished_laps, 90)) if finished_laps else None,
        "mean_collisions_per_episode": float(np.mean(collisions)) if collisions else _MISSING_TELEMETRY_PENALTY,
        "mean_damage_per_episode": float(np.mean(damage)) if damage else _MISSING_TELEMETRY_PENALTY,
        "act_latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "attempt_cell_ids": actual_cells,
        "cell_coverage_valid": coverage_ok,
        "measurement_telemetry_valid": telemetry_ok,
        "candidate_failure_count": sum(record.get("retire_reason") in _CANDIDATE_FAILURES for record in records),
        "infrastructure_invalid": any(
            record.get("retire_reason") == "evaluation_error"
            for record in records
        ),
    }


def _result_payload(result: ExperimentResult) -> dict[str, Any]:
    return {
        "candidate_id": result.candidate_id,
        "comparison_id": result.comparison_id,
        "split_id": result.split_id,
        "map_ids": list(result.map_ids),
        "seed_ids": list(result.seed_ids),
        "completion_count": result.completion_count,
        "episode_count": result.episode_count,
        "median_finished_lap_ms": result.median_finished_lap_ms,
        "mean_incomplete_progress": result.mean_incomplete_progress,
        "p90_finished_lap_ms": result.p90_finished_lap_ms,
        "collisions": result.collisions,
        "damage": result.damage,
        "act_latency_p95_ms": result.act_latency_p95_ms,
        "rule_compliance": result.rule_compliance.value,
        "mechanism_activation": result.mechanism_activation.value,
        "official_score": result.official_score,
        "eligibility": result.eligibility,
    }


def run_tournament(
    *,
    output_directory: Path,
    candidate_manifest: Path,
    split_path: Path,
    split_group: str,
    control_policy_checkpoint: Path,
    control_dynamics_checkpoint: Path,
    plan_budget_seconds: float,
    max_decisions: int,
) -> dict[str, Any]:
    """Evaluate every registered arm once on each selected, shared episode cell."""
    if (isinstance(plan_budget_seconds, bool) or not isinstance(plan_budget_seconds, (int, float))
            or not math.isfinite(float(plan_budget_seconds)) or float(plan_budget_seconds) <= 0.0):
        raise ValueError("plan budget must be finite and positive")
    if type(max_decisions) is not int or max_decisions <= 0:
        raise ValueError("max_decisions must be a positive integer")
    root = Path(__file__).resolve().parents[1]
    config = load_config(root)
    candidate_manifest = Path(candidate_manifest).resolve(strict=True)
    split_path = Path(split_path).resolve(strict=True)
    control_policy_checkpoint = Path(control_policy_checkpoint).resolve(strict=True)
    control_dynamics_checkpoint = Path(control_dynamics_checkpoint).resolve(strict=True)
    try:
        candidate_manifest.relative_to(root)
        split_path.relative_to(root)
        control_policy_checkpoint.relative_to(root)
        control_dynamics_checkpoint.relative_to(root)
    except ValueError as error:
        raise ValueError("tournament inputs must remain inside the repository") from error
    if not _is_within(candidate_manifest, root / "docs"):
        raise ValueError("candidate manifests must be stored under docs/")
    if not _is_within(split_path, root / "training" / "maps" / "site"):
        raise ValueError("tournament splits must be stored under training/maps/site/")
    if (not _is_within(control_policy_checkpoint, config.artifact_root)
            or not _is_within(control_dynamics_checkpoint, config.artifact_root)):
        raise ValueError("control checkpoints must come from the new artifact root")
    output_directory = Path(output_directory).resolve()
    if not _is_within(output_directory, config.artifact_root):
        raise ValueError("tournament output must remain under the new artifact root")
    registered_plan = _registered_tournament_plan(
        config=config,
        output_directory=output_directory,
        candidate_manifest=candidate_manifest,
        split_path=split_path,
        split_group=split_group,
        control_policy_checkpoint=control_policy_checkpoint,
        control_dynamics_checkpoint=control_dynamics_checkpoint,
        plan_budget_seconds=plan_budget_seconds,
        max_decisions=max_decisions,
    )
    approved_inputs = registered_plan.get("input_identity", {}).get("files", {})
    input_paths: dict[str, Path] = {
        "candidate_manifest": candidate_manifest,
        "site_map_split": split_path,
        "control_policy_checkpoint": control_policy_checkpoint,
        "control_dynamics_checkpoint": control_dynamics_checkpoint,
    }
    approved_named = {
        "candidate-manifest": candidate_manifest,
        "site-map-split": split_path,
        "policy-checkpoint": control_policy_checkpoint,
        "dynamics-checkpoint": control_dynamics_checkpoint,
    }
    for name, path in approved_named.items():
        entry = approved_inputs.get(name, {})
        if entry.get("path") != path.relative_to(root).as_posix() or entry.get("sha256") != _sha256(path):
            raise ValueError(f"runtime input {name} differs from its approved byte fingerprint")
    input_hashes_before = {
        "candidate_manifest": _sha256(candidate_manifest),
        "site_map_split": _sha256(split_path),
        "control_policy_checkpoint": _sha256(control_policy_checkpoint),
        "control_dynamics_checkpoint": _sha256(control_dynamics_checkpoint),
    }
    payload = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    manifest = validate_candidate_manifest(payload, split_group)
    if not control_policy_checkpoint.is_file() or not control_dynamics_checkpoint.is_file():
        raise ValueError("registered control checkpoints must exist as files")

    candidate_paths = {
        relative: _project_file(root, relative)
        for relative in referenced_files(manifest)
    }
    approved_references = approved_inputs.get("candidate-manifest", {}).get("references", [])
    reference_hashes = {item.get("path"): item.get("sha256") for item in approved_references}
    if set(reference_hashes) != {path.relative_to(root).as_posix() for path in candidate_paths.values()}:
        raise ValueError("runtime candidate references differ from the approved manifest")
    for relative, candidate_path in candidate_paths.items():
        if relative.lower().endswith((".pt", ".pth")) and not _is_within(candidate_path, config.artifact_root):
            raise ValueError("candidate checkpoints must come from the new artifact root")
        if relative == manifest.get("tune_source", {}).get("summary_ref") and not _is_within(candidate_path, config.artifact_root):
            raise ValueError("tune source summary must come from the new artifact root")
        if relative.endswith(".md") and not _is_within(candidate_path, root / "docs"):
            raise ValueError("candidate hypothesis documents must be stored under docs/")
        reference_hash = reference_hashes.get(candidate_path.relative_to(root).as_posix())
        if reference_hash != _sha256(candidate_path):
            raise ValueError(f"runtime candidate reference changed after approval: {relative}")
        input_paths[f"reference:{relative}"] = candidate_path
        input_hashes_before[f"reference:{relative}"] = reference_hash
    frozen_tune_source = _verify_frozen_tune(
        root=root,
        config=config,
        manifest=manifest,
        control_policy_checkpoint=control_policy_checkpoint,
        control_dynamics_checkpoint=control_dynamics_checkpoint,
        current_split_path=split_path,
        plan_budget_seconds=plan_budget_seconds,
        max_decisions=max_decisions,
    )
    approved_maps = approved_inputs.get("site-map-split", {}).get("maps", [])
    expected_map_hashes = {item.get("path"): item.get("sha256") for item in approved_maps}
    if not expected_map_hashes:
        raise ValueError("approved plan must fingerprint the selected map files")
    map_paths: dict[str, Path] = {}
    for relative, expected_hash in expected_map_hashes.items():
        map_path = _project_file(root, relative)
        if not _is_within(map_path, root / "training" / "maps" / "site"):
            raise ValueError("selected map must remain under training/maps/site/")
        if _sha256(map_path) != expected_hash:
            raise ValueError(f"runtime selected map changed after approval: {relative}")
        map_paths[relative] = map_path
        input_paths[f"map:{relative}"] = map_path
        input_hashes_before[f"map:{relative}"] = expected_hash
    episodes = load_site_map_group(split_path, split_group)
    loaded_map_paths = {episode.source_path.relative_to(root).as_posix() for episode in episodes}
    if loaded_map_paths != set(expected_map_hashes):
        raise ValueError("loaded map files differ from the approved selected-map set")
    if any(_sha256(path) != expected_map_hashes[relative] for relative, path in map_paths.items()):
        raise ValueError("a selected map changed while it was being loaded")
    cells = tuple(_cell_id(episode) for episode in episodes)
    if not episodes or len(cells) != len(set(cells)):
        raise ValueError("selected split must contain unique, nonempty map/seed cells")
    if split_group == "held_out" and (
        len(episodes) < 10 or len({episode.seed for episode in episodes}) < 2
    ):
        raise ValueError("held_out comparison requires at least ten attempts across two seeds")

    output_directory.mkdir(parents=True, exist_ok=False)
    episodes_path = output_directory / "episodes.jsonl"
    summary_path = output_directory / "summary.json"
    episode_stream = episodes_path.open("x", encoding="utf-8")
    arms = [{
        "candidate_id": manifest["control_id"],
        "direction": "control",
        "hypothesis_id": "CONTROL",
        "mode": "ppo_only",
        "policy_checkpoint": control_policy_checkpoint,
        "dynamics_checkpoint": None,
        "planner_settings": {},
        "role": "control",
    }]
    arms.extend({
        **candidate,
        "policy_checkpoint": candidate_paths[candidate["policy_checkpoint"]],
        "dynamics_checkpoint": (
            candidate_paths[candidate["dynamics_checkpoint"]]
            if candidate["dynamics_checkpoint"] is not None else None
        ),
        "role": "candidate",
    } for candidate in manifest["candidates"])
    arm_records: dict[str, list[dict[str, Any]]] = {arm["candidate_id"]: [] for arm in arms}
    arm_summaries: dict[str, dict[str, Any]] = {}
    result_objects: list[ExperimentResult] = []
    execution_rounds: list[dict[str, Any]] = []

    def persist(arm: Mapping[str, Any], cell_index: int, raw_record: dict[str, Any]) -> dict[str, Any]:
        record = dict(raw_record)
        episode = episodes[cell_index]
        identity_ok = (
            record.get("map_id") == episode.map_id
            and type(record.get("seed")) is int
            and record.get("seed") == episode.seed
        )
        reported_completed = record.get("completed") is True
        lap_time = record.get("lapTimeMs")
        valid_finish = (
            reported_completed and record.get("invalid_actions", 0) == 0
            and record.get("retire_reason") is None
            and isinstance(lap_time, (int, float)) and not isinstance(lap_time, bool)
            and math.isfinite(float(lap_time)) and float(lap_time) > 0.0
        )
        record.update({
            "candidate_id": arm["candidate_id"],
            "role": arm["role"],
            "direction": arm["direction"],
            "hypothesis_id": arm["hypothesis_id"],
            "cell_id": cells[cell_index] if identity_ok else None,
            "cell_index": cell_index,
            "registered_map_id": episode.map_id,
            "registered_seed": episode.seed,
            "identity_matches_registered_cell": identity_ok,
            "simulator_reported_completed": reported_completed,
            "completed": valid_finish,
            "terminal_status": "FINISHED" if valid_finish else "DNF",
        })
        nonfinite = False

        def json_safe(value: Any) -> Any:
            nonlocal nonfinite
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                nonfinite = True
                return None
            if isinstance(value, Mapping):
                return {str(key): json_safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(item) for item in value]
            return value

        safe_record = json_safe(record)
        safe_record["nonfinite_measurement"] = nonfinite
        arm_records[arm["candidate_id"]].append(safe_record)
        episode_stream.write(json.dumps(safe_record, sort_keys=True, allow_nan=False) + "\n")
        episode_stream.flush()
        return safe_record

    try:
        for cell_index, episode in enumerate(episodes):
            rotation = cell_index % len(arms)
            round_arms = arms[rotation:] + arms[:rotation]
            execution_rounds.append({"cell_id": cells[cell_index], "arms": [arm["candidate_id"] for arm in round_arms]})
            for arm in round_arms:
                callback_records: list[dict[str, Any]] = []

                def on_record(raw_record: dict[str, Any], current_arm=arm, current_cell=cell_index) -> None:
                    callback_records.append(persist(current_arm, current_cell, raw_record))

                records = evaluate_mode(
                    mode=arm["candidate_id"],
                    agent_mode=arm["mode"],
                    episodes=(episode,),
                    policy_checkpoint=arm["policy_checkpoint"],
                    dynamics_checkpoint=arm["dynamics_checkpoint"],
                    plan_budget_seconds=plan_budget_seconds,
                    max_decisions=max_decisions,
                    planner_settings=dict(arm["planner_settings"]),
                    fail_on_invalid_action=True,
                    on_record=on_record,
                )
                if len(records) != 1 or len(callback_records) != 1:
                    raise RuntimeError("each tournament arm must persist exactly one row per registered cell")
        for arm in arms:
            candidate_id = arm["candidate_id"]
            enriched = arm_records[candidate_id]
            summary = _summarize_arm(enriched, cells)
            summary["control_agent_setup_failed"] = (
                arm["role"] == "control"
                and any(record.get("retire_reason") == "agent_setup_error" for record in enriched)
            )
            arm_summaries[candidate_id] = summary
            if summary["measurement_telemetry_valid"]:
                result_objects.append(_as_result(
                    candidate_id=candidate_id,
                    comparison_id=manifest["comparison_id"],
                    split_group=split_group,
                    episodes=episodes,
                    records=enriched,
                    metrics=summary,
                ))
    finally:
        episode_stream.close()

    input_hashes_after = {name: _sha256(path) for name, path in input_paths.items()}
    inputs_stable = input_hashes_before == input_hashes_after

    invalid = any(
        not summary["cell_coverage_valid"]
        or not summary["measurement_telemetry_valid"]
        or summary["infrastructure_invalid"]
        or summary["control_agent_setup_failed"]
        for summary in arm_summaries.values()
    ) or not inputs_stable
    ranking = []
    winner_id = None
    if not invalid:
        ranked = rank_candidates(result_objects)
        ranking = [
            {"rank": index + 1, "candidate_id": result.candidate_id,
             "completion_count": result.completion_count, "episode_count": result.episode_count,
             "completion_rate": result.completion_count / result.episode_count,
             "metrics": arm_summaries[result.candidate_id]}
            for index, result in enumerate(ranked)
        ]
        if split_group == "tune" and ranked and ranked[0].candidate_id != manifest["control_id"]:
            winner_id = ranked[0].candidate_id

    report = {
        "schema_version": 1,
        "comparison_id": manifest["comparison_id"],
        "stage": split_group,
        "plan_budget_seconds": float(plan_budget_seconds),
        "max_decisions": max_decisions,
        "status": "INVALID_COMPARISON" if invalid else "RANKED",
        "denominator": len(episodes),
        "registered_cells": list(cells),
        "execution_order": "cell-major with rotating arm order",
        "execution_rounds": execution_rounds,
        "control_id": manifest["control_id"],
        "winner_candidate_id": winner_id,
        "ranking": ranking,
        "arms": arm_summaries,
        "frozen_tune_source": frozen_tune_source,
        "input_sha256_before": input_hashes_before,
        "input_sha256_after": input_hashes_after,
        "inputs_stable_during_evaluation": inputs_stable,
        "checkpoint_sha256": {
            "control_policy": input_hashes_before["control_policy_checkpoint"],
            "control_dynamics": input_hashes_before["control_dynamics_checkpoint"],
            **{relative: input_hashes_before[f"reference:{relative}"] for relative in candidate_paths
               if relative.endswith(".pt") or relative.endswith(".pth")},
        },
        "candidate_manifest_sha256": input_hashes_before["candidate_manifest"],
        "site_map_split_sha256": input_hashes_before["site_map_split"],
        "episode_jsonl": episodes_path.name,
        "ranking_policy": [
            "completion_rate_descending", "median_finished_lap_time_ms_ascending",
            "mean_incomplete_progress_descending", "p90_finished_lap_time_ms_ascending",
            "mean_collisions_per_episode_ascending", "mean_damage_per_episode_ascending",
            "act_latency_p95_ms_ascending",
        ],
        "invalid_attempts_are_noncompletions": True,
        "candidate_failures_are_noncompletions": True,
        "missing_candidate_failure_telemetry_penalty": _MISSING_TELEMETRY_PENALTY,
        "tune_does_not_open_held_out_map_files": split_group == "tune",
    }
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    if split_group == "held_out" and not invalid:
        selected = next(candidate for candidate in manifest["candidates"])
        by_id = {result.candidate_id: result for result in result_objects}
        (output_directory / "control_result.json").write_text(
            json.dumps(_result_payload(by_id[manifest["control_id"]]), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_directory / "candidate_result.json").write_text(
            json.dumps(_result_payload(by_id[selected["candidate_id"]]), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if invalid:
        raise RuntimeError(f"tournament comparison is invalid; inspect {summary_path}")
    return {"summary_path": str(summary_path), "episodes_path": str(episodes_path), **report}
