from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .index import ResultRecord
from .policy import compare_records, promotion_gate


TRIGGER = "성능 개선해 줘"
REQUIRED_CONTROL_FILES = (
    "RULES.md",
    "SOTA.md",
    "RESULTS.md",
    "COMPETITION_INFO.md",
    "RESTRICTIONS.md",
    "report.pdf",
    "research/papers.json",
)
STRATEGY_LANES = {
    "ppo_actor_only": {
        "role": "submission_candidate",
        "description": "현재 제출 가능한 순수 시각 PPO actor",
        "promotion_rule": "held_out 또는 official에서 SOTA를 엄격히 개선할 때만 승격",
    },
    "ppo_cem": {
        "role": "model_based_candidate",
        "description": "PPO actor에 latent dynamics와 CEM planner를 결합한 후보",
        "promotion_rule": "planner 비용과 패키지 검증을 포함해 actor-only보다 좋아야 함",
    },
    "vision_corridor_teacher": {
        "role": "teacher_diagnostic",
        "description": "픽셀 기반 corridor·장애물 회피 교사 및 진단 기준선",
        "promotion_rule": "제출 actor에 직접 포함하지 않고 학습·진단 근거로만 승격",
    },
}
PAPER_QUERIES = (
    "PPO visual control partial observability racing reinforcement learning",
    "DAgger learner induced distribution shift imitation learning",
    "MBPO short model rollouts model bias policy optimization",
    "domain randomization simulation visual control generalization",
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def audit_research(root: Path) -> dict[str, Any]:
    registry = root / "research" / "papers.json"
    report = root / "report.pdf"
    try:
        payload = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        payload = []
    papers = [
        item for item in payload
        if isinstance(item, dict) and item.get("id") and item.get("url") and item.get("haic_hypothesis")
    ] if isinstance(payload, list) else []
    strategy_links = {
        strategy: sorted(
            str(item["id"])
            for item in papers
            if strategy in item.get("related_strategies", [])
        )
        for strategy in STRATEGY_LANES
    }
    return {
        "registry": registry.as_posix(),
        "papers": len(papers),
        "paper_ids": [str(item["id"]) for item in papers],
        "strategy_links": strategy_links,
        "report_pdf": report.as_posix(),
        "report_pdf_sha256": _sha256(report),
        "matched": bool(papers) and report.is_file(),
    }


def _dict(record: ResultRecord | dict[str, Any]) -> dict[str, Any]:
    return record.to_dict() if isinstance(record, ResultRecord) else dict(record)


def audit_control_plane(root: Path) -> dict[str, Any]:
    present = {name: (root / name).is_file() for name in REQUIRED_CONTROL_FILES}
    research = audit_research(root)
    return {
        "required_files": present,
        "missing_files": [name for name, exists in present.items() if not exists],
        "research": research,
        "ready": all(present.values()) and research["matched"],
    }


def _split_priority(row: dict[str, Any]) -> tuple[int, tuple[float, ...]]:
    split_priority = {"held_out": 0, "official": 1, "tune": 2, "train": 3, "unknown": 4}
    values = (
        -float(row.get("completion_rate") or 0.0),
        float(row.get("median_finished_lap_ms") or float("inf")),
        -float(row.get("mean_progress") or 0.0),
    )
    return split_priority.get(str(row.get("split", "unknown")), 4), values


def best_records_by_strategy(records: Iterable[ResultRecord | dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATEGY_LANES}
    for raw in records:
        row = _dict(raw)
        if row.get("strategy") in grouped:
            grouped[str(row["strategy"])].append(row)
    selected: dict[str, dict[str, Any]] = {}
    for strategy, rows in grouped.items():
        if not rows:
            continue
        best_split = min(_split_priority(row)[0] for row in rows)
        same_split = [row for row in rows if _split_priority(row)[0] == best_split]
        selected[strategy] = compare_records(same_split)[0]
    return selected


def build_improvement_plan(
    root: Path,
    trigger: str,
    *,
    records: Iterable[ResultRecord | dict[str, Any]],
) -> dict[str, Any]:
    if trigger.strip() != TRIGGER:
        raise ValueError(f"unsupported trigger; expected exactly: {TRIGGER}")
    normalized = [_dict(record) for record in records]
    selected = best_records_by_strategy(normalized)
    ranking = compare_records(selected.values())
    control_plane = audit_control_plane(root)
    actor_reference = selected.get("ppo_actor_only")
    promotion: dict[str, dict[str, Any]] = {}
    for strategy, candidate in selected.items():
        if strategy == "ppo_actor_only":
            continue
        if actor_reference is None:
            promotion[strategy] = {
                "promote": False,
                "reason": "No actor-only baseline is available for comparison.",
            }
            continue
        decision = promotion_gate(candidate, actor_reference)
        promotion[strategy] = {"promote": decision.promote, "reason": decision.reason}
    return {
        "schema_version": 1,
        "trigger": TRIGGER,
        "control_plane": control_plane,
        "strategy_lanes": STRATEGY_LANES,
        "paper_search": {
            "queries": list(PAPER_QUERIES),
            "primary_sources_only": True,
            "record_against": "report.pdf",
            "registry": control_plane["research"],
        },
        "evidence": {
            "records_indexed": len(normalized),
            "best_by_strategy": selected,
            "ranking": ranking,
        },
        "promotion": promotion,
        "experiment_protocol": {
            "split_order": ["train", "tune", "held_out", "official"],
            "promotion_evidence": "held_out 또는 official",
            "minimum_episodes": 3,
            "compare_order": [
                "completion_rate",
                "median_finished_lap_ms",
                "mean_progress",
                "p90_finished_lap_ms",
                "collisions",
                "damage",
                "act_latency_ms",
            ],
            "never_promote_from": ["smoke", "train", "tune_only"],
        },
        "execution_policy": {
            "default": "plan_only",
            "execute_requires_explicit_command": True,
            "uses_shell": False,
            "writes_new_timestamped_artifacts": True,
            "resync_after_execution": True,
        },
        "submission_policy": {
            "default": "disabled",
            "requires_explicit_confirmation": True,
            "requires_restriction_pass": True,
            "requires_package_smoke": True,
            "requires_open_submission_command": True,
        },
        "next_actions": [
            "sync RESULTS.md from allowlisted raw summaries",
            "search primary papers and map claims to report.pdf",
            "design a fixed-seed experiment for the largest remaining gap",
            "run only after explicit --execute and record stdout/stderr",
            "compare held-out/official evidence and update SOTA.md only on strict promotion",
        ],
    }


def write_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _previous_sota(root: Path) -> dict[str, Any] | None:
    latest = root / "artifacts" / "haic" / "rules-runs" / "latest.json"
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    actor = payload.get("sota", {}).get("current") if isinstance(payload.get("sota"), dict) else None
    if isinstance(actor, dict):
        return actor
    evidence = payload.get("evidence", {})
    selected = evidence.get("best_by_strategy", {}) if isinstance(evidence, dict) else {}
    actor = selected.get("ppo_actor_only") if isinstance(selected, dict) else None
    return actor if isinstance(actor, dict) else None


def apply_sota_decision(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    selected = plan.get("evidence", {}).get("best_by_strategy", {})
    current = selected.get("ppo_actor_only") if isinstance(selected, dict) else None
    previous = _previous_sota(root)
    if not isinstance(current, dict):
        decision = {"promoted": False, "reason": "no actor-only held-out/official record", "current": None, "previous": previous}
    elif not isinstance(previous, dict):
        decision = {
            "promoted": current.get("restriction_status") == "pass" and current.get("split") in {"held_out", "official"} and float(current.get("episodes") or 0) >= 3,
            "reason": "initial actor-only record" if current.get("restriction_status") == "pass" else "restriction preflight did not pass",
            "current": current,
            "previous": None,
        }
    else:
        gate = promotion_gate(current, previous)
        decision = {
            "promoted": gate.promote,
            "reason": gate.reason,
            "current": current,
            "previous": previous,
        }
    plan["sota"] = decision
    return decision


def sync_sota_document(root: Path, plan: dict[str, Any]) -> None:
    """Write the actor-only SOTA pointer after a complete control-plane run.

    Tune rows select candidates, while the pointer is allowed to reference only
    an independent held-out/official record that passed the restriction audit.
    The teacher lane is intentionally excluded even when its lap time is lower.
    """
    decision = plan.get("sota", {})
    if decision.get("promoted") is False and (root / "SOTA.md").is_file():
        return
    selected = plan.get("evidence", {}).get("best_by_strategy", {})
    actor = selected.get("ppo_actor_only") if isinstance(selected, dict) else None
    lines = [
        "# SOTA",
        "",
        "현재 로컬 DB에서 held-out/official 근거와 제한 검사를 통과한 PPO actor-only 기록이다.",
        "Tune은 후보 선택에만 사용하고, corridor teacher 기록은 이 포인터에 올리지 않는다.",
        "",
    ]
    if not isinstance(actor, dict) or actor.get("restriction_status") != "pass":
        lines.append("현재 기록에는 제한 검사를 통과한 독립 SOTA가 없다.")
    else:
        checkpoint = None
        submission_archive = None
        source = actor.get("source")
        if isinstance(source, str):
            source_path = root / source
            try:
                source_payload = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                source_payload = {}
            if isinstance(source_payload, dict):
                checkpoint = source_payload.get("selected_policy")
                submission_archive = source_payload.get("submission_archive")
        lines.extend(
            [
                f"- record: `{actor.get('record_id')}`",
                f"- strategy: `{actor.get('strategy')}`",
                f"- split: `{actor.get('split')}`",
                f"- completion: `{actor.get('completion_rate')}`",
                f"- mean progress: `{actor.get('mean_progress')}`",
                f"- median lap ms: `{actor.get('median_finished_lap_ms')}`",
                f"- p90 lap ms: `{actor.get('p90_finished_lap_ms')}`",
                f"- source: `{actor.get('source')}`",
                f"- checkpoint: `{checkpoint}`" if checkpoint else "- checkpoint: `source JSON에 기록 없음`",
                f"- submission archive: `{submission_archive}`" if submission_archive else "- submission archive: `source JSON에 기록 없음`",
            ]
        )
    (root / "SOTA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def timestamped_plan_path(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "artifacts" / "haic" / "rules-runs" / f"improvement-plan-{stamp}.json"
