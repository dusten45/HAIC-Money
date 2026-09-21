import argparse
import ast
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from action_smoothing import (
    action_control_fingerprint,
    action_smoothing_fingerprint,
    normalize_action_smoothing,
    normalize_action_control,
    read_embedded_action_smoothing,
)
from train import find_vecnormalize_path


MAX_ARCHIVE_BYTES = 500 * 1024 * 1024
MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024
MAX_FILE_COUNT = 1_000
MAX_COMPRESSION_RATIO = 100
BANNED_IMPORTS = {
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
BANNED_CALLS = {"compile", "eval", "exec", "__import__"}
BANNED_SUFFIXES = {
    ".com",
    ".dll",
    ".dylib",
    ".exe",
    ".msi",
    ".scr",
    ".so",
}


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path):
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def git_provenance():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def relative_path(path: Path):
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def safe_label(label: str):
    normalized = "".join(
        character.lower() if character.isalnum() else "-" for character in label
    ).strip("-")
    if not normalized:
        raise ValueError("--label must contain at least one alphanumeric character")
    return normalized


def resolve_python_executable(python_executable: str):
    # ZIP smoke tests run from a temporary extraction directory. Preserve bare
    # PATH commands, but anchor a supplied relative path before changing CWD.
    # Do not resolve symlinks: virtualenv launchers rely on their .venv path to
    # locate pyvenv.cfg and site-packages.
    if "/" not in python_executable or Path(python_executable).is_absolute():
        return python_executable
    return str(Path.cwd() / python_executable)


def source_model_metadata(source_model_path: Path | None):
    if source_model_path is None:
        return None
    if not source_model_path.is_file():
        raise FileNotFoundError(source_model_path)

    metadata = file_metadata(source_model_path)
    vecnormalize_path = find_vecnormalize_path(source_model_path, "")
    metadata["vecnormalize"] = (
        file_metadata(vecnormalize_path)
        if vecnormalize_path is not None and vecnormalize_path.is_file()
        else None
    )

    run_dir = (
        source_model_path.parent.parent
        if source_model_path.parent.name == "checkpoints"
        else source_model_path.parent
    )
    metadata["run_config"] = (
        file_metadata(run_dir / "config.json")
        if (run_dir / "config.json").is_file()
        else None
    )
    metadata["evaluation_metrics"] = (
        file_metadata(run_dir / "best_model_metrics.json")
        if (run_dir / "best_model_metrics.json").is_file()
        else None
    )
    return metadata


def model_action_smoothing(model_path: Path, source_model_path: Path | None = None):
    """Resolve the exact smoother config carried by the model or its run."""

    source_config = None
    embedded_config = (
        read_embedded_action_smoothing(source_model_path)
        if source_model_path is not None
        else None
    )
    if source_model_path is not None:
        run_dir = (
            source_model_path.parent.parent
            if source_model_path.parent.name == "checkpoints"
            else source_model_path.parent
        )
        config_path = run_dir / "config.json"
        if config_path.is_file():
            recorded = json.loads(config_path.read_text())
            source_config = normalize_action_smoothing(
                recorded.get("config", {}).get("action_smoothing")
            )
    if embedded_config is not None and source_config is not None:
        if action_smoothing_fingerprint(embedded_config) != action_smoothing_fingerprint(source_config):
            raise ValueError("checkpoint and source run action smoothing configs do not match")
    source_config = embedded_config or source_config

    payload_config = None
    try:
        import torch

        payload = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        if payload.get("action_smoothing") is not None:
            payload_config = normalize_action_smoothing(payload["action_smoothing"])
    if source_config is not None and payload_config is not None:
        if action_smoothing_fingerprint(source_config) != action_smoothing_fingerprint(payload_config):
            raise ValueError("model and source run action smoothing configs do not match")
    if payload_config is None and source_config is not None:
        # A raw actor state dict has no place to carry the transform that Agent
        # must execute.  Refuse to misreport a smoothed source as a no-op model.
        if action_smoothing_fingerprint(source_config) != action_smoothing_fingerprint(
            normalize_action_smoothing()
        ):
            raise ValueError(
                "raw model payload cannot carry the source action smoothing config"
            )
    return payload_config or source_config or normalize_action_smoothing()


def model_action_control(model_path: Path, source_model_path: Path | None = None):
    source_config = None
    if source_model_path is not None:
        run_dir = (
            source_model_path.parent.parent
            if source_model_path.parent.name == "checkpoints"
            else source_model_path.parent
        )
        config_path = run_dir / "config.json"
        if config_path.is_file():
            recorded = json.loads(config_path.read_text())
            source_config = normalize_action_control(
                recorded.get("config", {}).get("action_control")
            )
    try:
        import torch

        payload = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = None
    payload_config = (
        normalize_action_control(payload.get("action_control"))
        if isinstance(payload, dict) and payload.get("action_control") is not None
        else None
    )
    if source_config is not None and payload_config is not None:
        if action_control_fingerprint(source_config) != action_control_fingerprint(payload_config):
            raise ValueError("model and source run action control configs do not match")
    action_control = payload_config or source_config or normalize_action_control()
    if isinstance(payload, dict) and payload.get("input_channels", action_control["input_channels"]) != action_control["input_channels"]:
        raise ValueError("model input channels do not match action control config")
    return action_control


def model_filename(agent_path: Path):
    tree = ast.parse(agent_path.read_text(), filename=str(agent_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "MODEL_FILENAME"
            for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            filename = node.value.value
            if Path(filename).name != filename:
                raise ValueError("MODEL_FILENAME must be a ZIP-root relative path")
            return filename
    raise ValueError("agent.py must define a string MODEL_FILENAME")


def agent_static_violations(agent_path: Path):
    tree = ast.parse(agent_path.read_text(), filename=str(agent_path))
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BANNED_IMPORTS:
                    violations.append(f"banned import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in BANNED_IMPORTS:
                violations.append(f"banned import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_CALLS:
                violations.append(f"banned call: {node.func.id}")
    return sorted(set(violations))


def validate_agent_source(agent_path: Path):
    violations = agent_static_violations(agent_path)
    if violations:
        raise ValueError("; ".join(violations))

    tree = ast.parse(agent_path.read_text(), filename=str(agent_path))
    agent_classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Agent"]
    if len(agent_classes) != 1:
        raise ValueError("agent.py must define exactly one Agent class")
    methods = {
        node.name for node in agent_classes[0].body if isinstance(node, ast.FunctionDef)
    }
    if "act" not in methods:
        raise ValueError("Agent must define act()")
    return model_filename(agent_path)


def agent_dependency_paths(agent_path: Path):
    """Resolve local Python modules imported by the submission agent."""

    tree = ast.parse(agent_path.read_text(), filename=str(agent_path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    dependencies = []
    if "action_smoothing" in modules:
        dependency = agent_path.with_name("action_smoothing.py")
        if not dependency.is_file():
            raise FileNotFoundError(dependency)
        violations = agent_static_violations(dependency)
        if violations:
            raise ValueError("; ".join(violations))
        dependencies.append(dependency)
    if "action_representation" in modules:
        dependency = agent_path.with_name("action_representation.py")
        if not dependency.is_file():
            raise FileNotFoundError(dependency)
        violations = agent_static_violations(dependency)
        if violations:
            raise ValueError("; ".join(violations))
        dependencies.append(dependency)
    return dependencies


def validate_submission_archive(
    archive_path: Path,
    expected_model_filename: str,
    expected_modules=(),
):
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("submission ZIP exceeds 500 MiB")

    with zipfile.ZipFile(archive_path) as archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in files]
        if len(files) > MAX_FILE_COUNT:
            raise ValueError("submission ZIP exceeds 1,000 files")
        expected_names = {"agent.py", expected_model_filename, *expected_modules}
        if set(names) != expected_names:
            raise ValueError(
                "submission ZIP must contain agent, model, and declared modules"
            )
        if any("/" in name or "\\" in name for name in names):
            raise ValueError("submission files must be at ZIP root")

        extracted_bytes = 0
        for info in files:
            if Path(info.filename).suffix.lower() in BANNED_SUFFIXES:
                raise ValueError(f"banned archive suffix: {info.filename}")
            if info.file_size > MAX_ARCHIVE_BYTES:
                raise ValueError(f"file exceeds 500 MiB: {info.filename}")
            if info.compress_size == 0 and info.file_size > 0:
                raise ValueError(f"invalid compressed size: {info.filename}")
            if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
                raise ValueError(f"compression ratio exceeds 100x: {info.filename}")
            extracted_bytes += info.file_size
        if extracted_bytes > MAX_EXTRACTED_BYTES:
            raise ValueError("submission archive exceeds 2 GiB extracted")


def build_submission(agent_path: Path, model_path: Path, archive_path: Path):
    expected_model_filename = validate_agent_source(agent_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    dependencies = agent_dependency_paths(agent_path)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(agent_path, arcname="agent.py")
        archive.write(model_path, arcname=expected_model_filename)
        for dependency in dependencies:
            archive.write(dependency, arcname=dependency.name)
    validate_submission_archive(
        archive_path,
        expected_model_filename,
        expected_modules=[dependency.name for dependency in dependencies],
    )
    return archive_path


def smoke_submission(archive_path: Path, python_executable: str):
    python_executable = resolve_python_executable(python_executable)
    child_code = """
import json
import time
import numpy as np

start = time.perf_counter()
from agent import Agent
agent = Agent()
init_seconds = time.perf_counter() - start
observation = np.zeros((4, 84, 84), dtype=np.float32)
unreset_action = agent.act(observation)
agent.reset(observation)
start = time.perf_counter()
first_action = agent.act(observation)
actions = [first_action, agent.act(observation)]
agent.reset(observation)
reset_action = agent.act(observation)
act_seconds = time.perf_counter() - start
print(json.dumps({
    "init_seconds": init_seconds,
    "act_seconds": act_seconds,
    "shape": list(actions[-1].shape),
    "finite": bool(all(np.isfinite(action).all() for action in actions)),
    "reset_matches_first": bool(np.array_equal(first_action, reset_action)),
    "unreset_matches_first": bool(np.array_equal(unreset_action, first_action)),
}))
"""
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(directory)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        completed = subprocess.run(
            [python_executable, "-c", child_code],
            cwd=directory,
            env=environment,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())

    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if result["init_seconds"] > 10:
        raise RuntimeError(f"Agent import and construction exceeded 10 s: {result}")
    if result["act_seconds"] > 5:
        raise RuntimeError(f"Agent.act() exceeded 5 s: {result}")
    if (
        result["shape"] != [3]
        or not result["finite"]
        or not result["reset_matches_first"]
        or not result["unreset_matches_first"]
    ):
        raise RuntimeError(f"Agent returned an invalid action: {result}")
    return result


def create_submission_record(
    agent_path: Path,
    model_path: Path,
    source_model_path: Path | None,
    submissions_dir: Path,
    label: str,
    smoke_test: bool,
    python_executable: str,
    command_line: str | None = None,
):
    python_executable = resolve_python_executable(python_executable)
    created_at = datetime.now(timezone.utc)
    submission_id = f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}_{safe_label(label)}"
    record_dir = submissions_dir / submission_id
    if record_dir.exists():
        raise FileExistsError(record_dir)

    expected_model_filename = validate_agent_source(agent_path)
    if model_path.name != expected_model_filename:
        raise ValueError(
            f"agent expects {expected_model_filename}, but model path is {model_path.name}"
        )
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    source_model = source_model_metadata(source_model_path)
    action_smoothing = model_action_smoothing(model_path, source_model_path)
    action_control = model_action_control(model_path, source_model_path)
    provenance = git_provenance()
    submissions_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".pending-", dir=submissions_dir))
    try:
        archive_path = build_submission(agent_path, model_path, temporary_dir / "submission.zip")
        smoke_result = (
            smoke_submission(archive_path, python_executable) if smoke_test else None
        )
        manifest = {
            "schema_version": 2,
            "submission_id": submission_id,
            "created_at_utc": created_at.isoformat().replace("+00:00", "Z"),
            "git": provenance,
            "package_command": command_line,
            "runtime": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "python_executable": python_executable,
            },
            "agent": file_metadata(agent_path),
            "dependencies": [file_metadata(path) for path in agent_dependency_paths(agent_path)],
            "action_smoothing": action_smoothing,
            "action_smoothing_fingerprint": action_smoothing_fingerprint(action_smoothing),
            "action_control": action_control,
            "action_control_fingerprint": action_control_fingerprint(action_control),
            "model": file_metadata(model_path),
            "source_model": source_model,
            "submission_zip": {
                "path": "submission.zip",
                "sha256": sha256_file(archive_path),
                "bytes": archive_path.stat().st_size,
            },
            "smoke_test": smoke_result,
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        temporary_dir.replace(record_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return record_dir, manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Build and validate a HAIC submission ZIP")
    parser.add_argument("--agent", type=Path, default=Path("agent.py"))
    parser.add_argument("--model", type=Path, default=Path("model.pt"))
    parser.add_argument("--source-model", type=Path)
    parser.add_argument("--submissions-dir", type=Path, default=Path("submissions"))
    parser.add_argument("--label", required=True, help="immutable submission record label")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--python", default=sys.executable, help="CPU Python interpreter")
    return parser.parse_args()


def main():
    args = parse_args()
    record_dir, manifest = create_submission_record(
        args.agent,
        args.model,
        args.source_model,
        args.submissions_dir,
        args.label,
        args.smoke_test,
        args.python,
        shlex.join(sys.argv),
    )
    print(
        f"built submission record: {record_dir} "
        f"({manifest['submission_zip']['bytes']} bytes)"
    )
    if manifest["smoke_test"] is not None:
        result = manifest["smoke_test"]
        print(
            f"cpu smoke passed: init_ms={result['init_seconds'] * 1000:.1f} "
            f"act_ms={result['act_seconds'] * 1000:.1f}"
        )


if __name__ == "__main__":
    main()
