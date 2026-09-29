from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


GENERATED_START = "<!-- BEGIN GENERATED RESULTS -->"
GENERATED_END = "<!-- END GENERATED RESULTS -->"
JSON_START = "<!-- RESULTS_JSON_START -->"
JSON_END = "<!-- RESULTS_JSON_END -->"


@dataclass(frozen=True)
class ResultRecord:
    record_id: str
    strategy: str
    split: str
    episodes: int
    completion_rate: float | None
    valid_under_13_fraction: float | None
    mean_progress: float | None
    median_finished_lap_ms: float | None
    p90_finished_lap_ms: float | None
    collisions: float | None
    damage: float | None
    act_latency_p95_ms: float | None
    restriction_status: str
    source: str
    status: str = "observed"
    variant: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int:
    number = _number(value)
    return max(0, int(number or 0))


def _metric(summary: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in summary:
            return _number(summary[name])
    return None


def _latency_p95(summary: dict[str, Any]) -> float | None:
    direct = _metric(summary, "act_latency_p95_ms", "act_p95_ms")
    if direct is not None:
        return direct
    latency = summary.get("act_latency_ms")
    if isinstance(latency, dict):
        return _number(latency.get("p95"))
    return None


def infer_strategy(path: Path, label: str = "", payload: dict[str, Any] | None = None) -> str:
    text = f"{path.as_posix()} {label} {json.dumps(payload or {}, sort_keys=True)}".lower()
    if any(token in text for token in ("corridor", "vision-racing", "teacher")):
        return "vision_corridor_teacher"
    if any(token in text for token in ("ppo+c em", "ppo+cem", "ppo_cem", "ppo-cem", "cem", "planner")):
        return "ppo_cem"
    if any(
        token in text
        for token in (
            "ppo_only",
            "ppo-only",
            "actor_only",
            "actor-only",
            "actor-selection",
            "obstacle-risk",
            'runtime_policy": "ppo_actor_only',
        )
    ) or "ppo" in text:
        return "ppo_actor_only"
    return "unknown"


def infer_split(path: Path, label: str = "") -> str:
    text = f"{path.as_posix()} {label}".lower().replace("-", "_")
    if "held_out" in text or "heldout" in text or "fullcap" in text:
        return "held_out"
    if "candidate" in text or "actual_agent" in text:
        return "tune"
    if "tune" in text or "validation" in text or "valid" in text:
        return "tune"
    if "official" in text:
        return "official"
    if "train" in text:
        return "train"
    return "unknown"


def _record_id(source: Path, payload: Any, label: str) -> str:
    digest = hashlib.sha256(
        (source.as_posix() + "\n" + label + "\n" + json.dumps(payload, sort_keys=True)).encode("utf-8")
    ).hexdigest()
    return digest[:12]


def _make_record(source: Path, label: str, summary: dict[str, Any], payload: Any) -> ResultRecord:
    completion = _metric(summary, "completion_rate", "completed_rate", "finish_rate")
    completed = _metric(summary, "completed", "finished", "successes", "finish_count", "finished_count")
    episodes = _integer(_metric(summary, "episodes", "episode_count", "n", "total_episodes"))
    if episodes == 0 and completed is not None and completion is not None and completion > 0:
        episodes = max(1, round(completed / completion))
    # Counts are authoritative.  Recomputing avoids ranking 2/3 as different
    # values when one artifact serialized it as 0.6666666667.
    if episodes and completed is not None:
        completion = completed / episodes

    inferred_strategy = infer_strategy(source, label, summary)
    restriction = summary.get("restriction_status")
    if restriction is None:
        source_text = source.as_posix().lower()
        if "final-ppo-actor-selection" in source_text:
            restriction = "pass"
        elif inferred_strategy == "vision_corridor_teacher":
            restriction = "fail"
        else:
            restriction = "unknown"
    inference_latency = _metric(summary, "actor_inference_p95_ms_max")
    if inference_latency is None:
        inference_latency = _latency_p95(summary)
    return ResultRecord(
        record_id=_record_id(source, payload, label),
        strategy=inferred_strategy,
        split=infer_split(source, label),
        episodes=episodes,
        completion_rate=completion,
        valid_under_13_fraction=_metric(summary, "valid_under_13_fraction"),
        mean_progress=_metric(summary, "mean_progress", "average_progress", "progress_mean"),
        median_finished_lap_ms=_metric(
            summary,
            "median_finished_lap_ms",
            "median_finished_lap_time_ms",
            "median_lap_ms",
            "median_finished_lap",
        ),
        p90_finished_lap_ms=_metric(
            summary,
            "p90_finished_lap_ms",
            "p90_finished_lap_time_ms",
            "p90_lap_ms",
            "p90_finished_lap",
        ),
        collisions=_metric(summary, "collisions", "collision_count", "mean_collisions", "collision_episodes"),
        damage=_metric(summary, "damage", "damage_count", "mean_damage", "mean_final_damage"),
        act_latency_p95_ms=inference_latency,
        restriction_status=str(restriction),
        source=source.as_posix(),
        variant=summary.get("variant"),
    )


def _aggregate_episodes(source: Path, payload: list[Any]) -> dict[str, Any]:
    episodes = [item for item in payload if isinstance(item, dict)]
    completed = [item for item in episodes if item.get("completed") or item.get("finished")]
    progresses = [_metric(item, "progress", "progress_fraction") for item in episodes]
    progresses = [value for value in progresses if value is not None]
    laps = [_metric(item, "lapTimeMs", "lap_time_ms", "lap_ms") for item in completed]
    laps = sorted(value for value in laps if value is not None)
    collisions = [_metric(item, "collisions", "collision_count") for item in episodes]
    collisions = [value for value in collisions if value is not None]
    damage = [_metric(item, "damage", "damage_count") for item in episodes]
    damage = [value for value in damage if value is not None]

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        index = min(len(values) - 1, round((len(values) - 1) * fraction))
        return values[index]

    return {
        "episodes": len(episodes),
        "completed": len(completed),
        "completion_rate": len(completed) / len(episodes) if episodes else None,
        "mean_progress": sum(progresses) / len(progresses) if progresses else None,
        "median_finished_lap_ms": percentile(laps, 0.5),
        "p90_finished_lap_ms": percentile(laps, 0.9),
        "collisions": sum(collisions) if collisions else None,
        "damage": sum(damage) if damage else None,
    }


def normalize_json_record(source: Path) -> list[ResultRecord]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    records: list[ResultRecord] = []
    if isinstance(payload, list):
        summary = _aggregate_episodes(source, payload)
        records.append(_make_record(source, "episodes", summary, payload))
        return records
    if not isinstance(payload, dict):
        return records

    # Paired runners may wrap their normalized arm summaries in an outer
    # result.json object that also contains training provenance. Index the
    # evaluation payload so provenance references (for example a train-only
    # teacher source file hash) cannot override the submitted strategy label.
    nested_evaluation = payload.get("evaluation")
    if (
        isinstance(nested_evaluation, dict)
        and isinstance(nested_evaluation.get("arm_summaries"), dict)
        and isinstance(nested_evaluation.get("episodes"), list)
    ):
        payload = nested_evaluation

    # Paired speed-target screens keep one episode list and a compact summary
    # for each treatment arm. Preserve one DB row per arm instead of treating
    # the whole payload as an unknown single experiment.
    target_summaries = payload.get("target_summaries")
    episode_rows = payload.get("episodes")
    if isinstance(target_summaries, dict) and isinstance(episode_rows, list):
        split = str(payload.get("split") or infer_split(source))
        for variant, target_summary in target_summaries.items():
            if not isinstance(target_summary, dict):
                continue
            match = re.fullmatch(r"target[_-]?(\d+(?:\.\d+)?)", str(variant).lower())
            target_speed = _number(target_summary.get("speed_target_m_s"))
            if target_speed is None and match:
                target_speed = float(match.group(1))
            selected_episodes = [
                item
                for item in episode_rows
                if isinstance(item, dict)
                and (
                    target_speed is None
                    or _number(item.get("speed_target_m_s")) == target_speed
                )
            ]
            episode_metrics = _aggregate_episodes(source, selected_episodes)
            normalized = dict(target_summary)
            normalized["variant"] = str(variant)
            normalized["episodes"] = target_summary.get("episode_count", len(selected_episodes))
            normalized["completed"] = target_summary.get("finish_count")
            normalized["completion_rate"] = target_summary.get("finish_rate")
            normalized["valid_under_13_fraction"] = target_summary.get("valid_under_13_fraction")
            normalized["mean_progress"] = episode_metrics.get("mean_progress")
            normalized["collisions"] = target_summary.get("collision_episodes")
            normalized["damage"] = target_summary.get("mean_final_damage")
            normalized["act_latency_p95_ms"] = target_summary.get("actor_inference_p95_ms_max")
            for source_key, target_key in (
                ("median_finished_lap_time_s", "median_finished_lap_ms"),
                ("p90_finished_lap_time_s", "p90_finished_lap_ms"),
            ):
                value = _number(target_summary.get(source_key))
                if value is not None:
                    normalized[target_key] = value * 1000.0
            if payload.get("restriction_status") is not None:
                normalized["restriction_status"] = payload["restriction_status"]
            label = f"{split}:{variant}"
            records.append(
                _make_record(
                    source,
                    label,
                    normalized,
                    {"variant": variant, "summary": target_summary, "episodes": selected_episodes},
                )
            )
        if records:
            return records

    # Reference-KL and other paired policy screens summarize each arm under
    # its name, while episode rows carry the matching arm label. Preserve the
    # arms as separate tune records and retain the raw rows for provenance.
    arm_summaries = payload.get("arm_summaries")
    if isinstance(arm_summaries, dict) and isinstance(episode_rows, list):
        split = str(payload.get("split") or infer_split(source))
        for arm, arm_summary in arm_summaries.items():
            if not isinstance(arm_summary, dict):
                continue
            selected_episodes = [
                item
                for item in episode_rows
                if isinstance(item, dict) and str(item.get("arm")) == str(arm)
            ]
            episode_metrics = _aggregate_episodes(source, selected_episodes)
            normalized = dict(arm_summary)
            normalized["variant"] = str(arm)
            normalized["episodes"] = arm_summary.get("episode_count", len(selected_episodes))
            normalized["completed"] = arm_summary.get("finish_count")
            normalized["completion_rate"] = arm_summary.get("finish_rate")
            normalized["valid_under_13_fraction"] = arm_summary.get("valid_under_13_fraction")
            normalized["mean_progress"] = episode_metrics.get("mean_progress")
            normalized["collisions"] = arm_summary.get("collision_episodes")
            normalized["damage"] = arm_summary.get("mean_final_damage")
            normalized["act_latency_p95_ms"] = arm_summary.get("actor_inference_p95_ms_max")
            for source_key, target_key in (
                ("median_finished_lap_time_s", "median_finished_lap_ms"),
                ("p90_finished_lap_time_s", "p90_finished_lap_ms"),
            ):
                value = _number(arm_summary.get(source_key))
                if value is not None:
                    normalized[target_key] = value * 1000.0
            if payload.get("restriction_status") is not None:
                normalized["restriction_status"] = payload["restriction_status"]
            label = f"{split}:{arm}"
            records.append(
                _make_record(
                    source,
                    label,
                    normalized,
                    {"variant": arm, "summary": arm_summary, "episodes": selected_episodes},
                )
            )
        if records:
            return records

    containers: list[tuple[str, dict[str, Any]]] = []
    if isinstance(payload.get("by_mode"), dict):
        containers.append(("root", payload))
    for parent_label in ("summary", "held_out_final_comparison", "tune", "held_out"):
        nested = payload.get(parent_label)
        if isinstance(nested, dict) and isinstance(nested.get("by_mode"), dict):
            containers.append((parent_label, nested))
    if containers:
        for parent_label, container in containers:
            for label, summary in container["by_mode"].items():
                if isinstance(summary, dict):
                    records.append(_make_record(source, f"{parent_label}:{label}", summary, summary))
        return records

    # Actual-agent comparison files keep the selected candidate under a
    # nested key rather than using the summary schema.  Keep the candidate and
    # baseline as separate tune records so a later run can compare them without
    # rewriting the raw artifact.
    for label, summary in (("candidate", payload.get("candidate")), ("baseline", payload.get("baseline"))):
        if isinstance(summary, dict) and any(
            key in summary for key in ("completion_rate", "mean_progress", "completed_episodes")
        ):
            records.append(_make_record(source, label, summary, summary))
    for label, summary in (
        ("candidate_heldout", payload.get("candidate_heldout")),
        ("baseline_heldout", payload.get("baseline_heldout")),
    ):
        if isinstance(summary, dict) and any(
            key in summary for key in ("completion_rate", "mean_progress", "completed_episodes")
        ):
            records.append(_make_record(source, label, summary, summary))
    if records:
        return records

    # Training result files use tune_metrics/full_tune_metrics and, in newer
    # runs, a held-out summary.  Normalize those fields into the same DB row.
    for label in ("tune_metrics", "full_tune_metrics", "held_out", "heldout"):
        summary = payload.get(label)
        if isinstance(summary, dict) and any(
            key in summary for key in ("completion_rate", "mean_progress", "completed_episodes", "episodes")
        ):
            records.append(_make_record(source, label, summary, summary))
    if records:
        return records

    episode_records = payload.get("records")
    if isinstance(episode_records, list):
        records.append(_make_record(source, "episodes", _aggregate_episodes(source, episode_records), episode_records))
        return records

    for label in ("held_out", "heldout", "tune", "official", "summary", "metrics"):
        summary = payload.get(label)
        if isinstance(summary, dict) and any(
            key in summary for key in ("completion_rate", "mean_progress", "completed", "episodes")
        ):
            records.append(_make_record(source, label, summary, summary))

    if not records:
        records.append(_make_record(source, "root", payload, payload))
    return records


RESULT_FILENAMES = {
    "summary.json",
    "evaluation.json",
    "benchmark.json",
    "result.json",
    "results.json",
    "training-result.json",
    "final-ppo-actor-selection.json",
    "manifest.json",
}
RESULT_ROOTS = (Path("artifacts") / "haic", Path("runs"), Path("submissions"))
IGNORED_PATH_TOKENS = {
    "trace",
    "decision_trace",
    "observations",
    "replay",
    "events",
    "state",
}


def discover_result_files(root: Path) -> list[Path]:
    """Find small summary JSON files without indexing raw traces or weights."""
    discovered: list[Path] = []
    for relative_root in RESULT_ROOTS:
        directory = root / relative_root
        if not directory.exists():
            continue
        for path in directory.rglob("*.json"):
            if path.name not in RESULT_FILENAMES:
                continue
            if path.stat().st_size > 4 * 1024 * 1024:
                continue
            lower_parts = {part.lower() for part in path.parts}
            if lower_parts.intersection(IGNORED_PATH_TOKENS):
                continue
            discovered.append(path)
    return sorted(set(discovered))


def scan_workspace(root: Path) -> list[ResultRecord]:
    """Normalize allowlisted result files and make sources workspace-relative."""
    records: list[ResultRecord] = []
    for path in discover_result_files(root):
        try:
            normalized = normalize_json_record(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        relative_source = path.relative_to(root).as_posix()
        for record in normalized:
            records.append(
                ResultRecord(
                    record_id=record.record_id,
                    strategy=record.strategy,
                    split=record.split,
                    episodes=record.episodes,
                    completion_rate=record.completion_rate,
                    mean_progress=record.mean_progress,
                    median_finished_lap_ms=record.median_finished_lap_ms,
                    p90_finished_lap_ms=record.p90_finished_lap_ms,
                    collisions=record.collisions,
                    damage=record.damage,
                    act_latency_p95_ms=record.act_latency_p95_ms,
                    restriction_status=record.restriction_status,
                    source=relative_source,
                    status=record.status,
                    variant=record.variant,
                    valid_under_13_fraction=record.valid_under_13_fraction,
                )
            )
    # Keep the small hand-maintained strategy registry in the same comparison
    # plane as discovered artifacts.  Raw artifacts remain authoritative; the
    # registry supplies explicit S1/S2/S3 labels and restriction decisions for
    # records whose historical JSON predates the control-plane schema.
    manual_path = root / "results" / "experiments.jsonl"
    if manual_path.exists():
        strategy_map = {
            "ppo_visual": "ppo_actor_only",
            "ppo_obstacle_risk": "ppo_actor_only",
            "ppo_dagger_failure": "ppo_actor_only",
            "ppo_actor_only": "ppo_actor_only",
            "ppo_cem": "ppo_cem",
            "vision_corridor_teacher": "vision_corridor_teacher",
        }
        for line in manual_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or not payload.get("experiment_id"):
                continue
            strategy = strategy_map.get(str(payload.get("strategy")), str(payload.get("strategy", "unknown")))
            for split in ("tune", "held_out", "official"):
                summary = payload.get(split)
                if not isinstance(summary, dict) or not summary:
                    continue
                episodes = _integer(summary.get("episodes"))
                completed = _number(summary.get("completed_episodes"))
                completion = _number(summary.get("completion_rate"))
                if episodes and completed is not None:
                    completion = completed / episodes
                records.append(
                    ResultRecord(
                        record_id=f"{payload['experiment_id']}:{split}",
                        strategy=strategy,
                        split=split,
                        episodes=episodes,
                        completion_rate=completion,
                        mean_progress=_number(summary.get("mean_progress")),
                        median_finished_lap_ms=_number(summary.get("median_finished_lap_ms")),
                        p90_finished_lap_ms=_number(summary.get("p90_finished_lap_ms")),
                        collisions=_number(summary.get("collisions")),
                        damage=_number(summary.get("damage")),
                        act_latency_p95_ms=_latency_p95(summary),
                        restriction_status=str(payload.get("restriction_status", "unknown")),
                        source=str(payload.get("source", manual_path.as_posix())),
                        status=str(payload.get("status", "observed")),
                        valid_under_13_fraction=_number(summary.get("valid_under_13_fraction")),
                    )
                )
    return records


def _as_dict(record: ResultRecord | dict[str, Any]) -> dict[str, Any]:
    return record.to_dict() if isinstance(record, ResultRecord) else dict(record)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}" if not value.is_integer() else str(int(value))
    return str(value)


def _render_generated(records: Iterable[ResultRecord | dict[str, Any]]) -> str:
    rows = [_as_dict(record) for record in records]
    rows.sort(key=lambda row: (row.get("strategy", ""), row.get("split", ""), row.get("record_id", "")))
    lines = [
        GENERATED_START,
        "<!-- Generated by `python -m research_ops.cli sync-results`; raw artifacts remain unchanged. -->",
        JSON_START,
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True),
        JSON_END,
        "",
        "| record_id | strategy | split | variant | episodes | completion | under 13 s | mean progress | median lap ms | p90 lap ms | collisions | damage | act p95 ms | restrictions | source |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _fmt(row.get(key))
                for key in (
                    "record_id",
                    "strategy",
                    "split",
                    "variant",
                    "episodes",
                    "completion_rate",
                    "valid_under_13_fraction",
                    "mean_progress",
                    "median_finished_lap_ms",
                    "p90_finished_lap_ms",
                    "collisions",
                    "damage",
                    "act_latency_p95_ms",
                    "restriction_status",
                    "source",
                )
            )
            + " |"
        )
    lines.extend([GENERATED_END, ""])
    return "\n".join(lines)


def sync_results_document(target: Path, records: Iterable[ResultRecord | dict[str, Any]]) -> None:
    if target.exists():
        content = target.read_text(encoding="utf-8")
    else:
        content = "# HAIC Results Database\n\nHuman notes and experiment decisions live below the generated section.\n\n"
    generated = _render_generated(records)
    pattern = re.compile(re.escape(GENERATED_START) + r".*?" + re.escape(GENERATED_END) + r"\n?", re.S)
    if pattern.search(content):
        content = pattern.sub(generated, content)
    else:
        content = content.rstrip() + "\n\n" + generated
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def read_results_document(target: Path) -> list[dict[str, Any]]:
    if not target.exists():
        return []
    content = target.read_text(encoding="utf-8")
    match = re.search(re.escape(JSON_START) + r"\s*(.*?)\s*" + re.escape(JSON_END), content, re.S)
    if not match:
        return []
    data = json.loads(match.group(1))
    return data if isinstance(data, list) else []
