"""Plan-only local HAIC research CLI: python -m haic_research.cli."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

from .commands import (
    CommandError, approve_run, execute_approved, load_run_plan, register_plan,
    replay_run_history, report_run,
)
from .config import ConfigError, load_config
from .models import ExperimentResult, IntegrationReport
from .results import report_experiment
from .records import RecordError, _canonical, _identifier, _json_value, _read_json, _typed
from .state import TransitionError


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise CommandError(message)


def parser() -> argparse.ArgumentParser:
    result = _Parser(description=__doc__)
    result.add_argument("--root", type=Path, default=Path.cwd(), help="repository containing harness.config.json")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="run the Task 8 repository validator when available")
    plan = commands.add_parser("plan", help="register one immutable command plan; no subprocess")
    plan.add_argument("--manifest", type=Path, required=True, help="explicit complete manifest metadata JSON")
    plan.add_argument("--research", type=Path, required=True, help="complete Hypothesis JSON")
    plan.add_argument("--profile", required=True)
    plan.add_argument("--arguments", type=Path, required=True, help="typed argument JSON with keys without --")
    plan.add_argument("--previous-run", help="explicit v2 predecessor identifier for a continuation")
    approve = commands.add_parser("approve")
    approve.add_argument("run_id")
    approve.add_argument("stage", choices=("design", "implementation", "execution"))
    approve.add_argument("--source-ref", required=True, help="reference explaining explicit user authorization")
    commands.add_parser("status").add_argument("run_id")
    run = commands.add_parser("run", help="preview the persisted plan; --execute consumes approval once")
    run.add_argument("run_id")
    run.add_argument("--execute", action="store_true")
    report = commands.add_parser("report", help="record evaluation outcome and the three gates")
    report.add_argument("run_id")
    report.add_argument("--outcome", choices=("ADVANCE", "REJECT", "REVISE", "PIVOT"), required=True)
    report.add_argument("--report", type=Path, required=True, help="IntegrationReport JSON")
    report.add_argument("--result", type=Path, help="explicit candidate ExperimentResult JSON")
    report.add_argument("--control", type=Path, help="explicit control ExperimentResult JSON; requires --result")
    report.add_argument("--promote-sota", action="store_true", help="request local SOTA gate review; requires --result/--control")
    return result


def _run_dir(config, run_id):
    _identifier(run_id)
    return config.run_root / run_id


def _input_json(config, filename):
    path = filename if filename.is_absolute() else config.repo_root / filename
    if ".." in path.parts or (path != config.repo_root and config.repo_root not in path.parents):
        raise CommandError("metadata file must remain in this repository")
    for legacy in config.legacy_paths:
        if (path == legacy or legacy in path.parents) and config.run_root not in path.parents:
            raise CommandError("historical data cannot be used as CLI metadata")
    return _read_json(_canonical(path))


def _status(history):
    return dict(state=history.state, approvals=[approval.stage for approval in history.approvals],
                execution_started=history.execution_started, execution_finished=history.execution_finished,
                execution_succeeded=history.execution_succeeded, outcome=history.outcome,
                released=history.released)


def main(argv=None, *, runner=subprocess.run, stdout=None, stderr=None) -> int:
    """Return an exit code; an injected runner keeps integration tests inert."""
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    try:
        args = parser().parse_args(argv)
        config = load_config(args.root)
        if args.command == "validate":
            try:
                module = importlib.import_module("haic_research.validation")
                validator = getattr(module, "validate_project")
            except (ImportError, AttributeError) as exc:
                raise CommandError("Task 8 repository validator is unavailable") from exc
            issues = validator(config.repo_root)
            output = {"valid": not issues, "issues": _json_value(issues)}
            exit_code = 0 if not issues else 1
        elif args.command == "plan":
            path = register_plan(config, _input_json(config, args.manifest), args.profile,
                                 _input_json(config, args.arguments), research=_input_json(config, args.research),
                                 previous_run_dir=_run_dir(config, args.previous_run) if args.previous_run else None)
            output = load_run_plan(config, path)
            exit_code = 0
        elif args.command == "approve":
            output = approve_run(config, _run_dir(config, args.run_id), args.stage, source_ref=args.source_ref)
            exit_code = 0
        elif args.command == "status":
            output = _status(replay_run_history(config, _run_dir(config, args.run_id)))
            exit_code = 0
        elif args.command == "run":
            path = _run_dir(config, args.run_id)
            plan = load_run_plan(config, path)
            if args.execute:
                output = execute_approved(config, path, plan["plan_hash"], plan["profile_id"], plan["arguments"], runner=runner)
                exit_code = 0 if output.succeeded else 1
            else:
                output = dict(mode="plan-only", plan_hash=plan["plan_hash"], profile_id=plan["profile_id"],
                              arguments=plan["arguments"], artifact_destination=plan["artifact_destination"],
                              sota_eligible=plan["profile"].get("sota_eligible", True))
                exit_code = 0
        else:
            report = _typed(IntegrationReport, _input_json(config, args.report))
            path = _run_dir(config, args.run_id)
            if (args.control or args.promote_sota) and not args.result:
                raise CommandError("--control/--promote-sota require --result")
            if args.result:
                candidate = _typed(ExperimentResult, _input_json(config, args.result))
                control = _typed(ExperimentResult, _input_json(config, args.control)) if args.control else None
                decision = report_experiment(config, path, args.outcome, report, candidate, control, promote=args.promote_sota)
                output = dict(_status(replay_run_history(config, path)), promotion=_json_value(decision))
            else:
                output = _status(report_run(config, path, args.outcome, report))
            exit_code = 0
        print(json.dumps(_json_value(output), sort_keys=True, ensure_ascii=False, allow_nan=False), file=stdout)
        return exit_code
    except (CommandError, ConfigError, RecordError, TransitionError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
