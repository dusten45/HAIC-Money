from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any
from zipfile import BadZipFile, ZipFile

from .index import scan_workspace, sync_results_document
from .orchestrator import (
    TRIGGER,
    build_improvement_plan,
    sync_sota_document,
    timestamped_plan_path,
    write_plan,
)


FORBIDDEN_IMPORTS = {
    "ctypes", "importlib", "multiprocessing", "os", "pathlib", "resource",
    "shutil", "signal", "socket", "subprocess", "sys",
}
FORBIDDEN_CALLS = {"compile", "eval", "exec", "__import__"}
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
MAX_ARCHIVE_BYTES = 500 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_FILES = 1_000
DEFAULT_TIMEOUT_SECONDS = 3_600


def _positive_timeout_seconds(value: str | int) -> int:
    try:
        seconds = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("timeout must be a positive number of seconds") from error
    if seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive number of seconds")
    return seconds


def _run_explicit_command(
    command: str,
    root: Path,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    timeout_seconds = _positive_timeout_seconds(timeout_seconds)
    args = shlex.split(command, posix=False)
    # With posix=False, shlex preserves Windows double-quote delimiters. Passing
    # those tokens to subprocess(list, shell=False) turns the delimiters into
    # literal argument characters, breaking quoted executable and path values.
    args = [
        token[1:-1]
        if len(token) >= 2 and token.startswith('"') and token.endswith('"')
        else token
        for token in args
    ]
    completed = subprocess.run(
        args,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return {
        "args": args,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_tail": completed.stdout[-8_000:],
        "stderr_tail": completed.stderr[-8_000:],
    }


def _audit_submission_package(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"checked": False, "pass": False, "reason": "no package supplied"}
    result: dict[str, Any] = {
        "checked": True,
        "path": path.as_posix(),
        "violations": [],
        "forbidden_imports": [],
        "forbidden_calls": [],
    }
    if not path.is_file():
        result["violations"].append("package does not exist")
        result["pass"] = False
        return result
    result["archive_bytes"] = path.stat().st_size
    if result["archive_bytes"] > MAX_ARCHIVE_BYTES:
        result["violations"].append("archive exceeds 500 MB")
    try:
        with ZipFile(path) as archive:
            infos = [item for item in archive.infolist() if not item.is_dir()]
            names = [item.filename for item in infos]
            result["file_count"] = len(names)
            result["uncompressed_bytes"] = sum(item.file_size for item in infos)
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
            if any(name.lower().endswith((".dll", ".dylib", ".exe", ".so", ".pyd")) for name in names):
                result["violations"].append("native executable is packaged")
            forbidden_imports: list[str] = []
            forbidden_calls: list[str] = []
            for name in names:
                if not name.endswith(".py"):
                    continue
                try:
                    tree = ast.parse(archive.read(name).decode("utf-8"), filename=name)
                except (UnicodeDecodeError, SyntaxError):
                    result["violations"].append(f"{name}: parse error")
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        forbidden_imports.extend(
                            f"{name}: import {item.name}"
                            for item in node.names
                            if item.name.split(".")[0] in FORBIDDEN_IMPORTS
                        )
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        if node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                            forbidden_imports.append(f"{name}: from {node.module}")
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        if node.func.id in FORBIDDEN_CALLS:
                            forbidden_calls.append(f"{name}: call {node.func.id}")
            result["forbidden_imports"] = forbidden_imports
            result["forbidden_calls"] = forbidden_calls
            result["violations"].extend(forbidden_imports)
            result["violations"].extend(forbidden_calls)
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
        result["violations"].append(str(error))
    result["pass"] = not result["violations"]
    return result


def run_improvement(
    root: Path,
    trigger: str,
    *,
    execute: bool = False,
    command: str | None = None,
    submission: Path | None = None,
    submit: bool = False,
    submission_command: str | None = None,
    confirm_submit: bool = False,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if trigger.strip() != TRIGGER:
        raise ValueError(f"unsupported trigger; expected exactly: {TRIGGER}")
    try:
        timeout_seconds = _positive_timeout_seconds(timeout_seconds)
    except argparse.ArgumentTypeError as error:
        raise ValueError(str(error)) from error
    root = root.resolve()
    if submission is None:
        preferred = root / "artifacts" / "haic" / "submission" / "haic-obstacle-risk-ppo-actor.zip"
        if preferred.is_file():
            submission = preferred
    records = scan_workspace(root)
    sync_results_document(root / "RESULTS.md", records)
    plan = build_improvement_plan(root, trigger, records=records)
    plan["execution"] = {
        "requested": execute,
        "timeout_seconds": timeout_seconds,
        "command_result": None,
    }
    plan["submission"] = {
        "requested": submit,
        "command_result": None,
        "package_audit": _audit_submission_package(submission),
    }
    plan_path = timestamped_plan_path(root)

    if execute:
        if not command:
            raise ValueError("--execute requires --command; no experiment command is inferred")
        execution_result = _run_explicit_command(command, root, timeout_seconds=timeout_seconds)
        log_stem = plan_path.with_suffix("")
        stdout_path = log_stem.with_name(f"{log_stem.name}.stdout.log")
        stderr_path = log_stem.with_name(f"{log_stem.name}.stderr.log")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(str(execution_result.pop("stdout", "")), encoding="utf-8")
        stderr_path.write_text(str(execution_result.pop("stderr", "")), encoding="utf-8")
        execution_result["stdout_log_path"] = stdout_path.relative_to(root).as_posix()
        execution_result["stderr_log_path"] = stderr_path.relative_to(root).as_posix()
        plan["execution"]["command_result"] = execution_result
        records = scan_workspace(root)
        sync_results_document(root / "RESULTS.md", records)
        plan = build_improvement_plan(root, trigger, records=records)
        plan["execution"] = {
            "requested": True,
            "command": command,
            "timeout_seconds": timeout_seconds,
            "command_result": execution_result,
        }
        plan["submission"] = {
            "requested": submit,
            "command_result": None,
            "package_audit": _audit_submission_package(submission),
        }

    if submit:
        if not confirm_submit:
            raise ValueError("--submit requires --confirm-submit")
        if not submission_command:
            raise ValueError("--submit requires --submission-command")
        if not plan["control_plane"]["ready"]:
            raise ValueError("submission blocked: control-plane documents are not ready")
        if not plan["submission"]["package_audit"].get("pass"):
            raise ValueError("submission blocked: package restriction audit did not pass")
        plan["submission"]["command_result"] = _run_explicit_command(
            submission_command,
            root,
            timeout_seconds=timeout_seconds,
        )

    from .orchestrator import apply_sota_decision

    apply_sota_decision(root, plan)
    plan["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    plan["plan_path"] = plan_path.as_posix()
    write_plan(plan_path, plan)
    sync_sota_document(root, plan)
    latest = plan_path.parent / "latest.json"
    write_plan(latest, plan)
    return plan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAIC result DB and strategy improvement control plane")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    sync = subparsers.add_parser("sync-results", help="index allowlisted result summaries into RESULTS.md")
    sync.add_argument("--root", type=Path, default=Path.cwd())

    compare = subparsers.add_parser("compare", help="compare the best observed record per strategy")
    compare.add_argument("--root", type=Path, default=Path.cwd())

    improve = subparsers.add_parser("improve", help=f"run the '{TRIGGER}' control-plane flow")
    improve.add_argument("trigger")
    improve.add_argument("--root", type=Path, default=Path.cwd())
    improve.add_argument("--execute", action="store_true")
    improve.add_argument("--command")
    improve.add_argument(
        "--timeout-seconds",
        type=_positive_timeout_seconds,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="maximum runtime for the explicit command (default: 3600)",
    )
    improve.add_argument("--submission", type=Path)
    improve.add_argument("--submit", action="store_true")
    improve.add_argument("--submission-command")
    improve.add_argument("--confirm-submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    if args.subcommand == "sync-results":
        records = scan_workspace(root)
        target = root / "RESULTS.md"
        sync_results_document(target, records)
        print(json.dumps({"records": len(records), "target": target.as_posix()}, ensure_ascii=False, indent=2))
        return
    if args.subcommand == "compare":
        plan = build_improvement_plan(root, TRIGGER, records=scan_workspace(root))
        print(json.dumps(plan["evidence"], ensure_ascii=False, indent=2, sort_keys=True))
        return
    result = run_improvement(
        root,
        args.trigger,
        execute=args.execute,
        command=args.command,
        submission=args.submission.resolve() if args.submission else None,
        submit=args.submit,
        submission_command=args.submission_command,
        confirm_submit=args.confirm_submit,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
