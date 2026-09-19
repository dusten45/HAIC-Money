import json
import subprocess
import sys
import time
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def git_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip())
    except Exception:
        dirty = False
    return {"commit": commit, "dirty": dirty}


def pip_freeze() -> list:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        return sorted(line for line in out.splitlines() if line.strip())
    except Exception:
        return []


def new_run(name: str, config: dict, command_line: str = "") -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    run_dir = RUNS_DIR / f"{stamp}_{safe_name}"
    counter = 1
    while run_dir.exists():
        run_dir = RUNS_DIR / f"{stamp}_{safe_name}_{counter}"
        counter += 1
    run_dir.mkdir(parents=True)

    meta = {
        "name": name,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_info(),
        "command_line": command_line,
        "config": config,
        "pip_freeze": pip_freeze(),
    }
    (run_dir / "config.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n"
    )

    (run_dir / "metrics.jsonl").write_text("")
    (run_dir / ".gitignore").write_text("*.zip\ncheckpoints/\n")

    latest = RUNS_DIR / "_latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name, target_is_directory=True)
    except OSError:
        pass

    return run_dir


def log_metrics(run_dir: Path, metrics: dict) -> None:
    line = json.dumps(metrics, default=str)
    with open(run_dir / "metrics.jsonl", "a") as handle:
        handle.write(line + "\n")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")
