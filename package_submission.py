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


def source_model_metadata(source_model_path: Path | None):
    if source_model_path is None:
        return None
    if not source_model_path.is_file():
        raise FileNotFoundError(source_model_path)

    metadata = file_metadata(source_model_path)
    if source_model_path.stem == "best_model":
        vecnormalize_path = source_model_path.with_name("best_model_vecnormalize.pkl")
    else:
        vecnormalize_path = source_model_path.with_name("model_vecnormalize.pkl")
    metadata["vecnormalize"] = (
        file_metadata(vecnormalize_path) if vecnormalize_path.is_file() else None
    )

    run_dir = source_model_path.parent
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


def validate_submission_archive(archive_path: Path, expected_model_filename: str):
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("submission ZIP exceeds 500 MiB")

    with zipfile.ZipFile(archive_path) as archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in files]
        if len(files) > MAX_FILE_COUNT:
            raise ValueError("submission ZIP exceeds 1,000 files")
        if set(names) != {"agent.py", expected_model_filename}:
            raise ValueError("submission ZIP must contain only root agent.py and model")
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
    if model_path.name != expected_model_filename:
        raise ValueError(
            f"agent expects {expected_model_filename}, but model path is {model_path.name}"
        )
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(agent_path, arcname="agent.py")
        archive.write(model_path, arcname=expected_model_filename)
    validate_submission_archive(archive_path, expected_model_filename)
    return archive_path


def smoke_submission(archive_path: Path, python_executable: str):
    child_code = """
import json
import time
import numpy as np

start = time.perf_counter()
from agent import Agent
agent = Agent()
init_seconds = time.perf_counter() - start
observation = np.zeros((4, 84, 84), dtype=np.float32)
start = time.perf_counter()
action = agent.act(observation)
act_seconds = time.perf_counter() - start
print(json.dumps({
    "init_seconds": init_seconds,
    "act_seconds": act_seconds,
    "shape": list(action.shape),
    "finite": bool(np.isfinite(action).all()),
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
    if result["shape"] != [3] or not result["finite"]:
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
