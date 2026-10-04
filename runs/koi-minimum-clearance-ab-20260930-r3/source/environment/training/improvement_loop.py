"""Run the repository's repeatable performance-improvement bookkeeping loop.

The loop is deliberately actor-only at submission time.  It indexes the local
experiment records, checks the competition documents, ranks the three allowed
training strategies on tune metrics, and writes a dated audit record.  It does
not silently upload a submission; ``--execute`` only runs the explicitly
provided local command and the package is still checked by the restrictions
audit before it can be recorded as a candidate.

Example::

    python -m training.improvement_loop "성능 개선해 줘"
    python -m training.improvement_loop "성능 개선해 줘" --execute \
        --command "python training/train_policy.py --help"
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Iterable, Mapping
from zipfile import ZipFile, BadZipFile


TRIGGER = "성능 개선해 줘"
REQUIRED_DOCS = (
    "RULES.md",
    "SOTA.md",
    "RESULTS.md",
    "COMPETITION_INFO.md",
    "RESTRICTIONS.md",
)
STRATEGIES = {
    "ppo_actor_only": "S1 PPO visual actor (제출 가능)",
    "ppo_cem": "S2 PPO + latent-dynamics CEM (평가 전용)",
    "vision_corridor_teacher": "S3 pixel corridor teacher (학습·진단 전용)",
}
STRATEGY_ORDER = tuple(STRATEGIES)
RESEARCH_PATH = Path("research") / "papers.json"
SUBMISSION_DIR = Path("artifacts") / "haic" / "submission"
CURRENT_SUBMISSION_NAME = "haic-obstacle-risk-ppo-actor.zip"
FORBIDDEN_IMPORTS = {
    "ctypes",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "resource",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
}
FORBIDDEN_CALLS = {"compile", "eval", "exec", "__import__"}
MAX_ARCHIVE_BYTES = 500 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_FILES = 1_000
REQUIRED_SUBMISSION_FILES = {
    "agent.py",
    "haic_agent/__init__.py",
    "haic_agent/observation.py",
    "haic_agent/pixel_features.py",
    "haic_agent/networks.py",
    "haic_agent/dynamics.py",
    "haic_agent/planner.py",
    "haic_agent/runtime_config.py",
    "policy.pt",
    "dynamics.pt",
}


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    strategy: str
    checkpoint: str | None
    source: str
    tune: dict[str, Any]
    held_out: dict[str, Any]
    official: dict[str, Any]
    status: str = "candidate"


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default


def _metric(payload: Mapping[str, Any] | None, *names: str) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    for name in names:
        value = _number(payload.get(name))
        if value is not None:
            return value
    return None


def _normalise_metrics(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Convert the several historical JSON metric spellings to one schema."""
    payload = payload if isinstance(payload, Mapping) else {}
    completed = _metric(payload, "completed_episodes", "completed")
    episodes = _metric(payload, "episodes")
    completion_rate = _metric(payload, "completion_rate", "finish_rate")
    # Counts are authoritative.  Recompute the fraction whenever both are
    # present so 0.6666666666 and 2/3 do not become different rankings.
    if completed is not None and episodes:
        completion_rate = completed / episodes
    return {
        "episodes": int(episodes) if episodes is not None else None,
        "completed_episodes": int(completed) if completed is not None else None,
        "completion_rate": completion_rate,
        "mean_progress": _metric(payload, "mean_progress", "progress"),
        "median_finished_lap_time_ms": _metric(
            payload, "median_finished_lap_time_ms"
        ),
        "median_finished_lap_time_s": _metric(payload, "median_finished_lap_time_s"),
        "p90_finished_lap_time_ms": _metric(payload, "p90_finished_lap_time_ms"),
        "p90_finished_lap_time_s": _metric(payload, "p90_finished_lap_time_s"),
        "collisions": _metric(payload, "collisions"),
        "act_p95_ms": _metric(payload, "act_p95_ms"),
    }


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


def load_research_registry(path: Path) -> list[dict[str, Any]]:
    """Read the checked-in primary-paper registry used by the next plan."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    papers: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        if not item.get("id") or not item.get("url") or not item.get("haic_hypothesis"):
            continue
        papers.append(dict(item))
    return papers


def audit_research(root: Path) -> dict[str, Any]:
    registry = root / RESEARCH_PATH
    report = root / "report.pdf"
    papers = load_research_registry(registry)
    strategy_links = {
        strategy: sorted(
            paper["id"]
            for paper in papers
            if strategy in paper.get("related_strategies", [])
        )
        for strategy in STRATEGY_ORDER
    }
    return {
        "registry": str(registry),
        "papers": len(papers),
        "paper_ids": [paper["id"] for paper in papers],
        "strategy_links": strategy_links,
        "report_pdf": str(report),
        "report_pdf_sha256": _sha256(report),
        "matched": bool(papers) and report.is_file(),
    }


def _held_out_key(record: Experiment) -> tuple[float, float, float]:
    metrics = record.held_out
    completion_value = _number(metrics.get("completion_rate"), -1.0)
    completion = round(completion_value if completion_value is not None else -1.0, 6)
    median = _as_ms(metrics)
    progress_value = _number(metrics.get("mean_progress"), -1.0)
    progress = progress_value if progress_value is not None else -1.0
    return (
        completion,
        -(median if median is not None else float("inf")),
        progress,
    )


def selection_key(record: Experiment) -> tuple[float, ...]:
    """Rank tune first, then prefer records with measured held-out evidence."""
    return (*rank_key(record.tune), *_held_out_key(record))


def sota_candidates(records: Iterable[Experiment]) -> list[Experiment]:
    candidates = [
        record for record in records
        if record.strategy == "ppo_actor_only"
        and record.checkpoint
        and (_number(record.held_out.get("episodes"), 0.0) or 0.0) >= 1
    ]
    if candidates:
        return candidates
    return [record for record in records if record.strategy == "ppo_actor_only" and record.checkpoint]


def next_experiment_plan(records: list[Experiment], research: Mapping[str, Any]) -> dict[str, Any]:
    """Create a deterministic, review-free next hypothesis from the DB."""
    by_strategy: dict[str, list[Experiment]] = {name: [] for name in STRATEGY_ORDER}
    for record in records:
        by_strategy.setdefault(record.strategy, []).append(record)
    coverage = {name: len(by_strategy.get(name, [])) for name in STRATEGY_ORDER}
    selected = max(sota_candidates(records), key=selection_key, default=None)
    # Continue the least covered and weakest observed strategy first; ties
    # follow the fixed order so repeated heartbeat runs remain reproducible.
    def strategy_quality(name: str) -> tuple[float, ...]:
        observed = by_strategy.get(name, [])
        return max((selection_key(item) for item in observed), default=(float("-inf"),))

    target = min(
        STRATEGY_ORDER,
        key=lambda name: (coverage.get(name, 0), strategy_quality(name), STRATEGY_ORDER.index(name)),
    )
    paper_ids = list(research.get("strategy_links", {}).get(target, []))
    return {
        "strategy": target,
        "strategy_label": STRATEGIES[target],
        "reason": "coverage-first hypothesis selection; use the same train/tune/held-out protocol",
        "current_selected": selected.experiment_id if selected else None,
        "supporting_papers": paper_ids,
        "required_metrics": [
            "completion_rate", "median_finished_lap_time_ms", "p90_finished_lap_time_ms",
            "mean_progress", "collisions", "act_p95_ms", "mean_abs_acceleration",
            "peak_deceleration", "collision_peak_deceleration", "obstacle_urgency",
        ],
    }


def _as_ms(metrics: Mapping[str, Any]) -> float | None:
    ms = _number(metrics.get("median_finished_lap_time_ms"))
    if ms is not None:
        return ms
    seconds = _number(metrics.get("median_finished_lap_time_s"))
    return seconds * 1000.0 if seconds is not None else None


def rank_key(metrics: Mapping[str, Any]) -> tuple[float, float, float, float, float, float]:
    """The competition order with progress/collisions as tie breakers.

    Completion rate is primary.  Faster completed laps win ties; incomplete
    runs then use mean progress.  Missing values are always ranked last.
    """
    completion_value = _number(metrics.get("completion_rate"), 0.0)
    completion = round(completion_value if completion_value is not None else 0.0, 6)
    median = _as_ms(metrics)
    p90 = _number(metrics.get("p90_finished_lap_time_ms"))
    if p90 is None:
        seconds = _number(metrics.get("p90_finished_lap_time_s"))
        p90 = seconds * 1000.0 if seconds is not None else None
    progress_value = _number(metrics.get("mean_progress"), 0.0)
    progress = progress_value if progress_value is not None else 0.0
    collisions_value = _number(metrics.get("collisions"), 0.0)
    collisions = collisions_value if collisions_value is not None else 0.0
    latency_value = _number(metrics.get("act_p95_ms"), 5_000.0)
    latency = latency_value if latency_value is not None else 5_000.0
    return (
        completion,
        -(median if median is not None else float("inf")),
        -(p90 if p90 is not None else float("inf")),
        progress,
        -collisions,
        -latency,
    )


def _strategy_from_text(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("corridor", "vision-racing", "teacher")):
        return "vision_corridor_teacher"
    if any(token in lowered for token in ("ppo+c em", "ppo+cem", "ppo_cem", "ppo-cem", "cem", "planner")):
        return "ppo_cem"
    return "ppo_actor_only"


def _canonical_strategy(value: Any) -> str:
    strategy = str(value or "").strip().lower()
    if strategy in STRATEGIES:
        return strategy
    if strategy in {"ppo_visual", "ppo_obstacle_risk", "ppo_dagger_failure", "ppo-only", "ppo_actor"}:
        return "ppo_actor_only"
    return _strategy_from_text(strategy)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _comparison_experiment(directory: Path, payload: Mapping[str, Any]) -> Experiment | None:
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        return None
    held = payload.get("candidate_heldout", {})
    strategy = _strategy_from_text(directory.name)
    return Experiment(
        experiment_id=directory.name,
        strategy=strategy,
        checkpoint=str(directory / "policy.pt") if (directory / "policy.pt").exists() else None,
        source=str(directory / "actual-agent-comparison.json"),
        tune=_normalise_metrics(candidate),
        held_out=_normalise_metrics(held if isinstance(held, Mapping) else None),
        official={},
    )


def _training_experiment(directory: Path, payload: Mapping[str, Any]) -> Experiment | None:
    tune = payload.get("tune_metrics") or payload.get("full_tune_metrics")
    if not isinstance(tune, Mapping):
        return None
    strategy = _strategy_from_text(directory.name + " " + str(payload.get("phase", "")))
    config = _read_json(directory / "experiment_config.json") or {}
    runtime = str(config.get("runtime_policy", ""))
    # Historical artifact folders contain smoke runs and evaluators with a
    # different split/timing contract.  Only records that explicitly declare
    # the current actor-only runtime are eligible for SOTA selection.
    if runtime not in {"ppo_actor_only", "ppo_only"}:
        return None
    return Experiment(
        experiment_id=directory.name,
        strategy=strategy,
        checkpoint=str(directory / "policy.pt") if (directory / "policy.pt").exists() else None,
        source=str(directory / "training-result.json"),
        tune=_normalise_metrics(tune),
        held_out=_normalise_metrics(payload.get("held_out") if isinstance(payload.get("held_out"), Mapping) else None),
        official={},
    )


def index_artifacts(artifact_root: Path) -> list[Experiment]:
    """Index authoritative JSON records without treating every debug file as a run."""
    records: list[Experiment] = []
    for directory in sorted(path for path in artifact_root.iterdir() if path.is_dir()):
        comparison_path = directory / "actual-agent-comparison.json"
        training_path = directory / "training-result.json"
        experiment: Experiment | None = None
        if comparison_path.exists():
            payload = _read_json(comparison_path)
            if payload:
                experiment = _comparison_experiment(directory, payload)
        if experiment is None and training_path.exists():
            payload = _read_json(training_path)
            if payload:
                experiment = _training_experiment(directory, payload)
        if experiment is not None:
            records.append(experiment)
    return records


def load_manual_results(path: Path) -> list[Experiment]:
    """Load the small JSONL registry used for candidates without full artifacts."""
    if not path.exists():
        return []
    records: list[Experiment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping) or not payload.get("experiment_id"):
            continue
        records.append(
            Experiment(
                experiment_id=str(payload["experiment_id"]),
                strategy=_canonical_strategy(payload.get("strategy")),
                checkpoint=payload.get("checkpoint"),
                source=str(payload.get("source", path)),
                tune=_normalise_metrics(payload.get("tune")),
                held_out=_normalise_metrics(payload.get("held_out")),
                official=dict(payload.get("official", {})) if isinstance(payload.get("official"), Mapping) else {},
                status=str(payload.get("status", "candidate")),
            )
        )
    return records


def _deduplicate(records: Iterable[Experiment]) -> list[Experiment]:
    by_id: dict[str, Experiment] = {}
    for record in records:
        current = by_id.get(record.experiment_id)
        if current is None or record.source.endswith("experiments.jsonl"):
            by_id[record.experiment_id] = record
    return sorted(by_id.values(), key=lambda item: item.experiment_id)


def audit_documents(root: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_DOCS if not (root / name).exists()]
    report_path = root / "report.pdf"
    research = audit_research(root)
    return {
        "required_documents": list(REQUIRED_DOCS),
        "missing_documents": missing,
        "report_pdf_present": report_path.exists(),
        "research": research,
        "rules_ready": not missing and report_path.exists() and research["matched"],
    }


def audit_submission(zip_path: Path | None) -> dict[str, Any]:
    if zip_path is None:
        return {"checked": False, "reason": "no package path supplied"}
    result: dict[str, Any] = {
        "checked": True,
        "path": str(zip_path),
        "violations": [],
        "forbidden_imports": [],
        "forbidden_calls": [],
    }
    if not zip_path.exists():
        result["violations"].append("package does not exist")
        result["pass"] = False
        return result
    result["archive_bytes"] = zip_path.stat().st_size
    if result["archive_bytes"] > MAX_ARCHIVE_BYTES:
        result["violations"].append("archive exceeds 500 MB")
    try:
        with ZipFile(zip_path) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            names = [info.filename for info in infos]
            result["file_count"] = len(names)
            result["uncompressed_bytes"] = sum(info.file_size for info in infos)
            if result["file_count"] > MAX_ARCHIVE_FILES:
                result["violations"].append("archive exceeds 1,000 files")
            if result["uncompressed_bytes"] > MAX_UNCOMPRESSED_BYTES:
                result["violations"].append("archive exceeds 2 GiB uncompressed")
            missing = sorted(REQUIRED_SUBMISSION_FILES - set(names))
            extra = sorted(set(names) - REQUIRED_SUBMISSION_FILES)
            result["missing_files"] = missing
            result["extra_files"] = extra
            if missing or extra:
                result["violations"].append("archive must contain exactly the inference files")
            if any("corridor_agent" in name.lower() for name in names):
                result["violations"].append("training-only corridor module is packaged")
            native_suffixes = (".dll", ".dylib", ".exe", ".so", ".pyd")
            if any(name.lower().endswith(native_suffixes) for name in names):
                result["violations"].append("native executable is packaged")
            forbidden_hits: list[str] = []
            forbidden_call_hits: list[str] = []
            for name in names:
                if not name.endswith(".py"):
                    continue
                try:
                    tree = ast.parse(archive.read(name).decode("utf-8"), filename=name)
                except (UnicodeDecodeError, SyntaxError):
                    forbidden_hits.append(name + ": parse error")
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        forbidden_hits.extend(
                            f"{name}: import {item.name}"
                            for item in node.names
                            if item.name.split(".")[0] in FORBIDDEN_IMPORTS
                        )
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        if node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                            forbidden_hits.append(f"{name}: from {node.module}")
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        if node.func.id in FORBIDDEN_CALLS:
                            forbidden_call_hits.append(f"{name}: call {node.func.id}")
            result["forbidden_imports"] = forbidden_hits
            result["forbidden_calls"] = forbidden_call_hits
            result["violations"].extend(forbidden_hits)
            result["violations"].extend(forbidden_call_hits)
            if "haic_agent/runtime_config.py" in names:
                runtime_tree = ast.parse(
                    archive.read("haic_agent/runtime_config.py").decode("utf-8"),
                    filename="haic_agent/runtime_config.py",
                )
                values: dict[str, Any] = {}
                for node in runtime_tree.body:
                    if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                        continue
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            values[target.id] = node.value.value
                result["planner_enabled"] = values.get("PLANNER_ENABLED")
                result["strict_checkpoint_loading"] = values.get("STRICT_CHECKPOINT_LOADING")
                if result["strict_checkpoint_loading"] is not True:
                    result["violations"].append("strict checkpoint loading is not enabled")
                if result["planner_enabled"] is not False:
                    result["violations"].append("submission planner must be disabled")
    except (OSError, BadZipFile) as error:
        result["violations"].append(f"cannot read package: {error}")
    result["pass"] = not result["violations"]
    return result


def _markdown_metrics(metrics: Mapping[str, Any]) -> str:
    rate = metrics.get("completion_rate")
    progress = metrics.get("mean_progress")
    median = _as_ms(metrics)
    p90 = metrics.get("p90_finished_lap_time_ms")
    if p90 is None:
        seconds = _number(metrics.get("p90_finished_lap_time_s"))
        p90 = seconds * 1000.0 if seconds is not None else None
    def fmt(value: Any, digits: int = 3) -> str:
        return "-" if value is None else f"{float(value):.{digits}f}"
    return f"{fmt(rate)} | {fmt(progress)} | {fmt(median, 0)} | {fmt(p90, 0)}"


def _best_record(records: Iterable[Experiment]) -> Experiment | None:
    candidates = sota_candidates(records)
    return max(candidates, key=selection_key, default=None)


def write_results_markdown(root: Path, records: list[Experiment], audit: Mapping[str, Any]) -> None:
    ranked = sorted(records, key=selection_key, reverse=True)
    lines = [
        "# RESULTS",
        "",
        "이 파일은 `training.improvement_loop`가 로컬 JSON 실험 DB를 읽어 갱신하는 결과 표입니다.",
        "완주율을 첫 기준으로 하고, 같은 완주율에서는 완주 시간, 진행도, 충돌, act p95 순으로 비교합니다.",
        "완료율은 완료 수/전체 수로 다시 계산해 기록 간 부동소수점 표기 차이를 제거합니다.",
        "",
        "## 세 전략",
        "",
        "| 전략 ID | 학습 방식 | 기록 수 |",
        "|---|---|---:|",
    ]
    for strategy in STRATEGY_ORDER:
        lines.append(f"| `{strategy}` | {STRATEGIES[strategy]} | {sum(item.strategy == strategy for item in records)} |")
    lines.extend(
        [
            "",
            "## Tune 비교",
            "",
        "| 순위 | 실험 | 전략 | Tune 완주율 | Tune 평균 진행도 | 중앙 완주 ms | P90 완주 ms | 상태 |",
        "|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for index, item in enumerate(ranked, 1):
        metrics = item.tune
        lines.append(
            f"| {index} | `{item.experiment_id}` | {STRATEGIES.get(item.strategy, item.strategy)} | "
            f"{_markdown_metrics(metrics)} | {item.status} |"
        )
    if not ranked:
        lines.append("| - | 실험 기록 없음 | - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "## 자동 점검",
            "",
            f"- 필수 문서 누락: `{len(audit.get('missing_documents', []))}`개",
            f"- `report.pdf`: `{'있음' if audit.get('report_pdf_present') else '없음'}`",
            f"- 논문 registry: `{audit.get('research', {}).get('papers', 0)}편`, PDF 대조: `{'통과' if audit.get('research', {}).get('matched') else '실패'}`",
            "- Tune만 후보 선택에 사용하고 held-out/공식 맵은 최종 확인용으로 기록합니다.",
            "- teacher, corridor, map geometry는 제출 actor의 실행 경로에 포함하지 않습니다.",
            "",
            "## 기록 위치",
            "",
            "각 실험의 원본은 `artifacts/haic/` 아래 JSON과 checkpoint입니다. 수동 후보는 "
            "`results/experiments.jsonl`에 한 줄 JSON으로 추가합니다.",
            "",
        ]
    )
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def write_sota_markdown(root: Path, records: list[Experiment], *, promoted: bool, previous: Experiment | None) -> None:
    best = _best_record(records)
    lines = [
        "# SOTA",
        "",
        "현재 로컬 실험 DB에서 같은 프로토콜의 Tune과 held-out을 모두 가진 actor-only PPO 후보입니다.",
        "공식 Track1 seed42와 추가 장애물 맵은 별도 진단이며 Tune 순위에 섞지 않습니다.",
        "사용자 목표 기록: Track1 17초, Track2 20초.",
        "",
    ]
    if not promoted and previous is not None:
        best = previous
        lines.append("이번 실행에서는 현재 SOTA보다 엄격히 좋은 후보가 없어 기존 기록을 유지했습니다.")
        lines.append("")
    if best is None:
        lines.append("아직 기록된 실험이 없습니다.")
    else:
        lines.extend(
            [
                f"- 실험: `{best.experiment_id}`",
                f"- 전략: {STRATEGIES.get(best.strategy, best.strategy)}",
                f"- checkpoint: `{best.checkpoint or '없음'}`",
                f"- Tune 지표: `{json.dumps(best.tune, ensure_ascii=False, sort_keys=True)}`",
                f"- held-out 지표: `{json.dumps(best.held_out, ensure_ascii=False, sort_keys=True)}`",
                f"- 공식 진단: `{json.dumps(best.official, ensure_ascii=False, sort_keys=True)}`",
                f"- SOTA 갱신: `{'예' if promoted else '아니오'}`",
                "- 제출 런타임: PPO actor only, planner/teacher/corridor fallback 비활성화",
            ]
        )
    lines.extend(
        [
            "",
            "## 갱신 규칙",
            "",
            "`RULES.md`의 트리거를 실행했을 때 Tune 순위가 엄격히 좋아지고 held-out 기록이 있으면 이 항목을 교체합니다.",
            "개선이 없으면 SOTA를 교체하지 않고 결과 DB와 자동 실행 기록에 후보로 남깁니다.",
            "",
        ]
    )
    (root / "SOTA.md").write_text("\n".join(lines), encoding="utf-8")


def _previous_selection(root: Path, records: Iterable[Experiment]) -> Experiment | None:
    latest = _read_json(root / "artifacts" / "haic" / "improvement-loop" / "latest.json")
    if not latest:
        return None
    selected_id = latest.get("sota_selected") or latest.get("selected")
    if not selected_id:
        return None
    return next((record for record in records if record.experiment_id == selected_id), None)


def _default_submission(root: Path) -> Path | None:
    preferred = root / SUBMISSION_DIR / CURRENT_SUBMISSION_NAME
    if preferred.is_file():
        return preferred
    archives = sorted((root / SUBMISSION_DIR).glob("*.zip")) if (root / SUBMISSION_DIR).is_dir() else []
    return archives[-1] if archives else None


def write_run_record(
    root: Path,
    *,
    records: list[Experiment],
    audit: Mapping[str, Any],
    submission: Mapping[str, Any],
    command: str | None,
    command_result: Mapping[str, Any] | None,
    plan: Mapping[str, Any],
    sota_selected: Experiment | None,
    previous_sota: Experiment | None,
    promoted: bool,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = root / "artifacts" / "haic" / "improvement-loop"
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(records, key=selection_key, reverse=True)
    tune_ranked = sorted(records, key=lambda item: rank_key(item.tune), reverse=True)
    payload = {
        "schema_version": 2,
        "trigger": TRIGGER,
        "timestamp_utc": timestamp,
        "document_audit": dict(audit),
        "submission_audit": dict(submission),
        "ranking": [
            {
                "experiment_id": item.experiment_id,
                "strategy": item.strategy,
                "checkpoint": item.checkpoint,
                "tune": item.tune,
                "held_out": item.held_out,
                "official": item.official,
                "rank_key": rank_key(item.tune),
                "selection_key": selection_key(item),
            }
            for item in ranked
        ],
        "selected": tune_ranked[0].experiment_id if tune_ranked else None,
        "sota_selected": sota_selected.experiment_id if sota_selected else None,
        "previous_sota": previous_sota.experiment_id if previous_sota else None,
        "sota_promoted": promoted,
        "research_plan": dict(plan),
        "command": command,
        "command_result": command_result,
    }
    path = output_dir / f"run-{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "latest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def run_command(command: str, root: Path) -> dict[str, Any]:
    args = shlex.split(command, posix=False)
    completed = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=3_600)
    return {
        "args": args,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-8_000:],
        "stderr_tail": completed.stderr[-8_000:],
    }


def run(*, root: Path, trigger: str, execute: bool = False, command: str | None = None, submission: Path | None = None) -> dict[str, Any]:
    if trigger.strip() != TRIGGER:
        raise ValueError(f"unsupported trigger; expected exactly: {TRIGGER}")
    audit = audit_documents(root)
    artifact_records = index_artifacts(root / "artifacts" / "haic")
    manual_records = load_manual_results(root / "results" / "experiments.jsonl")
    records = _deduplicate([*artifact_records, *manual_records])
    plan = next_experiment_plan(records, audit.get("research", {}))
    command_result = None
    if execute:
        if not command:
            raise ValueError("--execute requires --command; no command is inferred")
        command_result = run_command(command, root)
        # A train/evaluate command may have created a new JSON record.  Reindex
        # after it finishes so this same invocation compares the new result.
        artifact_records = index_artifacts(root / "artifacts" / "haic")
        manual_records = load_manual_results(root / "results" / "experiments.jsonl")
        records = _deduplicate([*artifact_records, *manual_records])
    previous_sota = _previous_selection(root, records)
    best = _best_record(records)
    promoted = previous_sota is None or (best is not None and selection_key(best) > selection_key(previous_sota))
    if not promoted and previous_sota is not None:
        best_for_sota = previous_sota
    else:
        best_for_sota = best
    write_results_markdown(root, records, audit)
    write_sota_markdown(root, records, promoted=promoted, previous=best_for_sota if not promoted else None)
    submission_path = submission or _default_submission(root)
    submission_audit = audit_submission(submission_path)
    record_path = write_run_record(
        root,
        records=records,
        audit=audit,
        submission=submission_audit,
        command=command,
        command_result=command_result,
        plan=plan,
        sota_selected=best_for_sota,
        previous_sota=previous_sota,
        promoted=promoted,
    )
    return {
        "record": str(record_path),
        "selected": best_for_sota.experiment_id if best_for_sota else None,
        "sota_promoted": promoted,
        "next_strategy": plan.get("strategy"),
        "experiments": len(records),
        "document_audit": audit,
        "submission_audit": submission_audit,
        "command_result": command_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trigger", help=f"must be exactly: {TRIGGER}")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true", help="run the explicit local command after the audit")
    parser.add_argument("--command", help="local training/evaluation/package command used with --execute")
    parser.add_argument("--submission", type=Path, help="ZIP to audit and record")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        root=args.root.resolve(),
        trigger=args.trigger,
        execute=args.execute,
        command=args.command,
        submission=args.submission.resolve() if args.submission else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
