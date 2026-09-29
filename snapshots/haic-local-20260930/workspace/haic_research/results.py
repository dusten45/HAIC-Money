"""New-run result evidence and append-only local research reporting.

Metrics are supplied observations, not independently measured races. Historical
records are never discovered, opened, imported or promoted by this module.
"""
from __future__ import annotations

import json
import html
import os
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .commands import load_run_plan, replay_run_history, report_run
from .config import HarnessConfig
from .models import ExperimentResult, GateStatus, IntegrationReport, PromotionDecision, WorkflowState
from .policy import ComparisonMismatchError, promotion_decision
from .records import (RecordError, _canonical, _config_roots, _json_value, _lock,
                      _open, _typed, _validate_report, run_transaction)

_BEGIN = b"<!-- BEGIN GENERATED RESULTS -->"
_END = b"<!-- END GENERATED RESULTS -->"
_INDEPENDENT = frozenset({"held_out", "confirmation", "blind", "official"})


def _result(value: ExperimentResult) -> dict[str, object]:
    if not isinstance(value, ExperimentResult):
        raise RecordError("a typed ExperimentResult is required")
    # Revalidate even if an object was constructed outside the normal boundary.
    return _json_value(_typed(ExperimentResult, _json_value(value)))


def _decision(candidate, control):
    try:
        return promotion_decision(candidate, control)
    except ComparisonMismatchError as exc:
        return PromotionDecision(False, str(exc))


def _evidence_payloads(plan, candidate, control):
    identity = {"run_id": plan["manifest"]["run_id"], "plan_hash": plan["plan_hash"]}
    result = dict(identity, result=_result(candidate))
    comparison = None if control is None else dict(identity, candidate=_result(candidate), control=_result(control),
                                                  decision=_json_value(_decision(candidate, control)))
    return result, comparison


def persist_result_evidence(config: HarnessConfig, run_dir: Path, candidate: ExperimentResult,
                            control: ExperimentResult | None = None) -> None:
    """Persist exact supplied payloads before the immutable integration report."""
    plan = load_run_plan(config, run_dir)
    history = replay_run_history(config, run_dir)
    if history.state != WorkflowState.EVALUATE or not history.execution_finished:
        raise RecordError("result evidence requires a finished execution in EVALUATE")
    result, comparison = _evidence_payloads(plan, candidate, control)
    with run_transaction(run_dir, config=config) as run:
        if run.report is not None or any((run_dir / name).exists() for name in ("result_evidence.json", "comparison_evidence.json")):
            raise RecordError("result/comparison evidence already exists or gate report is frozen")
        run.write_result_evidence(result)
        if comparison is not None:
            run.write_result_evidence(comparison, comparison=True)


def _verified_evidence(config, run_dir, plan, candidate, control=None):
    expected, comparison = _evidence_payloads(plan, candidate, control)
    with run_transaction(run_dir, config=config) as run:
        actual = run.read_result_evidence()
        if actual != expected:
            raise RecordError("candidate differs from this run's immutable result evidence")
        if control is not None and run.read_result_evidence(comparison=True) != comparison:
            raise RecordError("candidate/control differ from this run's immutable comparison evidence")
        if run.report is None:
            raise RecordError("integration report is required before document reporting")
        return run.report


def _document(config, supplied, relative):
    root, _, _ = _config_roots(config)
    expected = root / relative
    # Compare lexical paths before stat/open: no arbitrary or historical targets.
    if Path(supplied) != expected:
        raise RecordError(f"reporting path must be {relative}")
    return _canonical(expected)


def _read_bytes(path):
    with _open(path, "r") as stream:
        return stream.buffer.read()


def _replace_bytes(path, content):
    """Replace the human document under the caller's shared document lock."""
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with _open(temporary, "x") as stream:
            stream.buffer.write(content)
        _canonical(path)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _newline(content):
    return b"\r\n" if b"\r\n" in content else b"\n"


def _block(text, content):
    return text.replace("\n", _newline(content).decode('ascii')).encode('utf-8')


def _reference(config, path):
    return Path(path).relative_to(config.repo_root).as_posix()


def _marker(run_id):
    return f"<!-- HAIC V2 RUN {run_id} -->"


def _cell(value):
    return html.escape(str(value).replace('\r', ' ').replace('\n', ' ')).replace('|', '&#124;')


def append_experiment_summary(results_path: Path, experiment_path: Path, result: ExperimentResult,
                              *, config: HarnessConfig, run_dir: Path) -> None:
    """Write immutable detail and append run entries, preserving generated bytes."""
    plan = load_run_plan(config, run_dir)
    history = replay_run_history(config, run_dir)
    report = _verified_evidence(config, run_dir, plan, result)
    run_id = plan['manifest']['run_id']
    results_path = _document(config, results_path, 'RESULTS.md')
    experiment_path = _document(config, experiment_path, f'docs/experiments/{run_id}.md')
    index_path = _document(config, config.repo_root / 'docs/experiments/INDEX.md', 'docs/experiments/INDEX.md')
    marker = _marker(run_id)
    date = datetime.now(timezone.utc).date().isoformat()
    manifest_ref = _reference(config, run_dir / 'run_manifest.json')
    report_ref = _reference(config, run_dir / 'integration_report.json')
    detail_ref = _reference(config, experiment_path)
    observation = f"{result.completion_count}/{result.episode_count} completions; {history.outcome.value if history.outcome else 'unfinished'}"
    summary = f"\n{marker}\n### {date}: `{run_id}`\n\nSupplied observation: {_cell(result.candidate_id)}; {observation}; split `{_cell(result.split_id)}`. [Full experiment]({detail_ref}). Official ranking is separate.\n\n"
    research = plan['research']
    detail = (f"# Experiment {run_id}\n\nRecorded: {date} UTC. Metrics are supplied evidence; this writer does not independently run or validate races.\n\n"
              f"- [Manifest](../../{manifest_ref})\n- [Approved plan](../../{_reference(config, run_dir / 'execution_plan.json')})\n"
              f"- [Result evidence](../../{_reference(config, run_dir / 'result_evidence.json')})\n"
              f"- [Integration report](../../{report_ref})\n\n## Registered research and protocol\n\n```json\n"
              + json.dumps(dict(research=research, manifest=plan['manifest']), ensure_ascii=False, sort_keys=True, indent=2)
              + "\n```\n\n## Supplied result and gates\n\n```json\n"
              + json.dumps(dict(result=_result(result), report=_json_value(report), outcome=_json_value(history.outcome), released=history.released), ensure_ascii=False, sort_keys=True, indent=2)
              + "\n```\n")
    if (run_dir / 'comparison_evidence.json').exists():
        detail += f"\n[Immutable comparison evidence](../../{_reference(config, run_dir / 'comparison_evidence.json')}).\n"
    row = (f"\n{marker}\n### {run_id}\n\n| Run ID | Hypothesis | Registered split and control | Decision | Manifest | Integration report |\n|---|---|---|---|---|---|\n| [{run_id}]({run_id}.md) | {_cell(research['hypothesis_id'])} | {_cell(result.split_id)} / {_cell(research['control'])} | "
           f"{history.outcome.value if history.outcome else 'unfinished'} | [manifest](../../{manifest_ref}) | [report](../../{report_ref}) |\n")
    with _lock(config.repo_root):
        before, index = _read_bytes(results_path), _read_bytes(index_path)
        if experiment_path.exists() or marker.encode() in before or marker.encode() in index:
            raise RecordError("run already appears in reporting history; use a new correction record")
        if before.count(_BEGIN) != before.count(_END) or before.count(_BEGIN) > 1:
            raise RecordError("generated RESULTS boundaries must be unique and complete")
        if _BEGIN in before:
            start, end = before.index(_BEGIN), before.index(_END)
            if end < start:
                raise RecordError("generated RESULTS boundaries are reordered")
            after = before[:start] + _block(summary, before) + before[start:]
        else:
            after = before + _block(summary, before)
        # Preflight every document before the immutable detail is created.
        with _open(experiment_path, "x") as stream:
            stream.write(detail)
        _replace_bytes(results_path, after)
        with _open(index_path, "a") as stream:
            stream.buffer.write(_block(row, index))


def promote_sota(sota_path: Path, candidate: ExperimentResult, control: ExperimentResult,
                 report_path: Path, *, config: HarnessConfig, run_dir: Path) -> PromotionDecision:
    """Append a local SOTA pointer only for this exact released comparison."""
    sota_path = _document(config, sota_path, 'SOTA.md')
    if Path(report_path) != Path(run_dir) / 'integration_report.json':
        raise RecordError("promotion must use this named run's integration report")
    plan = load_run_plan(config, run_dir)
    report = _verified_evidence(config, run_dir, plan, candidate, control)
    history = replay_run_history(config, run_dir)
    if not history.released:
        return PromotionDecision(False, "run did not replay an ADVANCE gate review and release")
    if any(gate.status != GateStatus.PASS for gate in report.gate_results):
        return PromotionDecision(False, "all three integration gates must PASS")
    evidence = {_reference(config, run_dir / name) for name in ('result_evidence.json', 'comparison_evidence.json')}
    if not evidence.issubset(report.evidence_paths):
        return PromotionDecision(False, "integration report must reference this exact result and comparison evidence")
    profile = plan['profile']
    if profile.get('sota_eligible', True) is not True or profile['operation'] != 'local_evaluation' or plan['profile_id'] != 'evaluate_closed_loop':
        return PromotionDecision(False, "only an eligible independent evaluation profile can promote")
    manifest = plan['manifest']
    for result, revision in ((candidate, manifest['candidate_revision']), (control, manifest['control_revision'])):
        if (result.candidate_id != revision or result.comparison_id != manifest.get('comparison_id')
                or set(result.map_ids) != set(manifest['map_ids']) or [result.split_id] != manifest['split_ids']
                or set(result.seed_ids) != set(manifest.get('seed_ids', []))
                or result.episode_count != manifest.get('comparison_episode_count')):
            return PromotionDecision(False, "result identity/protocol must match the preregistered manifest")
        if result.split_id not in _INDEPENDENT or len(result.seed_ids) < 2 or result.episode_count < len(result.seed_ids):
            return PromotionDecision(False, "promotion requires a replicated independent evaluation, never train/tune")
    decision = _decision(candidate, control)
    if not decision.eligible:
        return decision
    run_id = manifest['run_id']
    marker = _marker(run_id)
    pointer = (f"\n{marker}\n## Local v2 SOTA pointer: {run_id}\n\n"
               f"Recorded {datetime.now(timezone.utc).date().isoformat()} UTC. Supplied matched metrics selected by completion-first policy after the recorded release gates. Official standing is separate.\n\n"
               f"- Candidate: `{_cell(candidate.candidate_id)}`; control: `{_cell(control.candidate_id)}`\n"
               f"- Completion: {candidate.completion_count}/{candidate.episode_count}; control: {control.completion_count}/{control.episode_count}\n"
               f"- Split: `{_cell(candidate.split_id)}`; seeds: {_cell(', '.join(candidate.seed_ids))}\n"
               f"- [Manifest]({_reference(config, run_dir / 'run_manifest.json')})\n"
               f"- [Comparison evidence]({_reference(config, run_dir / 'comparison_evidence.json')})\n"
               f"- [Integration report]({_reference(config, report_path)})\n")
    with _lock(config.repo_root):
        before = _read_bytes(sota_path)
        if marker.encode() in before:
            raise RecordError("run already has a SOTA pointer; history cannot be overwritten")
        with _open(sota_path, "a") as stream:
            stream.buffer.write(_block(pointer, before))
    return decision


def report_experiment(config: HarnessConfig, run_dir: Path, outcome: str, report: IntegrationReport,
                      candidate: ExperimentResult, control: ExperimentResult | None = None,
                      *, promote: bool = False) -> PromotionDecision:
    """CLI integration: evidence -> immutable gate report -> documents -> pointer."""
    if promote and control is None:
        raise RecordError("SOTA promotion requires explicit control result input")
    _validate_report(report)
    requested = WorkflowState(outcome)
    if requested not in {WorkflowState.ADVANCE, WorkflowState.REJECT, WorkflowState.REVISE, WorkflowState.PIVOT}:
        raise RecordError("unknown evaluation outcome")
    history = replay_run_history(config, run_dir)
    if requested == WorkflowState.ADVANCE and not history.execution_succeeded:
        raise RecordError("failed execution cannot advance")
    persist_result_evidence(config, run_dir, candidate, control)
    paths = tuple(_reference(config, run_dir / name) for name in ('result_evidence.json', 'comparison_evidence.json') if (run_dir / name).exists())
    report = replace(report, evidence_paths=tuple(dict.fromkeys((*report.evidence_paths, *paths))))
    report_run(config, run_dir, outcome, report)
    append_experiment_summary(config.repo_root / 'RESULTS.md', config.repo_root / f'docs/experiments/{run_dir.name}.md', candidate, config=config, run_dir=run_dir)
    if promote:
        return promote_sota(config.repo_root / 'SOTA.md', candidate, control, run_dir / 'integration_report.json', config=config, run_dir=run_dir)
    return PromotionDecision(False, "observation recorded; SOTA promotion was not requested")
