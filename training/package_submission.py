"""Build a minimal, CPU-only HAIC archive and verify it in a clean directory."""

import argparse
import ast
import json
import math
import pickle
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import torch

from haic_agent.dynamics import LatentDynamicsEnsemble
from haic_agent.networks import VisualActorCritic


INFERENCE_FILES = (
    "agent.py",
    "haic_agent/__init__.py",
    "haic_agent/observation.py",
    "haic_agent/networks.py",
    "haic_agent/dynamics.py",
    "haic_agent/planner.py",
    "haic_agent/runtime_config.py",
    "policy.pt",
    "dynamics.pt",
)
BANNED_IMPORTS = {"ctypes", "importlib", "multiprocessing", "os", "pathlib", "resource", "shutil", "signal", "socket", "subprocess", "sys"}
BANNED_CALLS = {"compile", "eval", "exec", "__import__"}
MAX_MEMORY_BYTES = 1_024 * 1024 * 1024
DEFAULT_PACKAGED_PLANNER_SETTINGS = {
    "horizon": 4,
    "population": 16,
    "iterations": 2,
    "candidate_batch_size": 8,
    "uncertainty_cost": 1.0,
}


def validate_planner_settings(settings: Any) -> dict[str, Any]:
    """Validate planner values before writing or accepting an inference archive."""
    required = {
        "horizon",
        "population",
        "iterations",
        "candidate_batch_size",
        "uncertainty_cost",
    }
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError("planner settings must define exactly the required fields")
    minimums = {
        "horizon": 1,
        "population": 2,
        "iterations": 1,
        "candidate_batch_size": 1,
    }
    for name, minimum in minimums.items():
        value = settings[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"planner settings {name} must be an integer >= {minimum}")
    uncertainty_cost = settings["uncertainty_cost"]
    try:
        finite_uncertainty_cost = float(uncertainty_cost)
    except (TypeError, ValueError, OverflowError):
        finite_uncertainty_cost = float("nan")
    if (
        isinstance(uncertainty_cost, bool)
        or not isinstance(uncertainty_cost, (int, float))
        or not math.isfinite(finite_uncertainty_cost)
        or finite_uncertainty_cost < 0.0
    ):
        raise ValueError("planner settings uncertainty_cost must be finite and non-negative")
    return dict(settings)


def static_violations(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BANNED_IMPORTS:
                    violations.append(f"banned import: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in BANNED_IMPORTS:
            violations.append(f"banned import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            violations.append(f"banned call: {node.func.id}")
    return sorted(set(violations))


def _archive_sources(source_root: Path) -> dict[str, Path]:
    paths = {name: source_root / name for name in INFERENCE_FILES if name.endswith(".py")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    for name, path in paths.items():
        violations = static_violations(path.read_text(encoding="utf-8"), name)
        if violations:
            raise ValueError(f"{name}: {'; '.join(violations)}")
    return paths


def validate_submission_archive(archive_path: Path) -> dict[str, Any]:
    """Check the exact root layout and statically audit every packaged Python file."""
    with zipfile.ZipFile(archive_path) as archive:
        names = tuple(sorted(info.filename for info in archive.infolist() if not info.is_dir()))
        if set(names) != set(INFERENCE_FILES):
            raise ValueError("submission archive must contain only the required inference files")
        if any(name.startswith("training/") or ".venv" in name or "labels" in name for name in names):
            raise ValueError("submission archive contains forbidden training content")
        for name in names:
            if name.endswith(".py"):
                violations = static_violations(archive.read(name).decode("utf-8"), name)
                if violations:
                    raise ValueError(f"{name}: {'; '.join(violations)}")
        runtime_config = archive.read("haic_agent/runtime_config.py").decode("utf-8")
        tree = ast.parse(runtime_config)
        planner_enabled = next(
            (bool(node.value.value) for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "PLANNER_ENABLED" for target in node.targets)
             and isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool)),
            None,
        )
        if planner_enabled is None:
            raise ValueError("runtime planner selection is missing")
        settings = next(
            (ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "PLANNER_SETTINGS" for target in node.targets)),
            None,
        )
        strict_loading = next(
            (bool(node.value.value) for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "STRICT_CHECKPOINT_LOADING" for target in node.targets)
             and isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool)),
            None,
        )
        if strict_loading is not True:
            raise ValueError("runtime inference settings or strict checkpoint loading are invalid")
        settings = validate_planner_settings(settings)
    return {"root_agent": "agent.py", "files": list(names), "archive_bytes": archive_path.stat().st_size,
            "planner_enabled": planner_enabled, "planner_settings": settings, "strict_checkpoint_loading": strict_loading,
            "runtime_policy": "trained_visual_actor"}


def smoke_submission(
    archive_path: Path, *, expected_planner_enabled: bool, expected_planner_settings: dict[str, Any],
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Unpack into a clean directory and enforce CPU import/reset/act limits."""
    child = """
import json
import time
import tracemalloc
import ctypes
from ctypes import wintypes
import numpy as np
class _Counters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
        ('PrivateUsage', ctypes.c_size_t)]
def process_rss_bytes():
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _Counters(); counters.cb = ctypes.sizeof(_Counters)
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
    except AttributeError:
        pass
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * 1024)
    except ImportError:
        return None
started = time.monotonic()
from agent import Agent
agent = Agent()
init_s = time.monotonic() - started
obs = np.zeros((4, 84, 84), dtype=np.float32)
reset_times = []
actions = []
act_times = []
for _ in range(2):
    started = time.monotonic(); agent.reset(obs); reset_times.append(time.monotonic() - started)
    started = time.monotonic(); action = agent.act(obs); act_times.append(time.monotonic() - started); actions.append(action)
print(json.dumps({
  'init_s': init_s, 'reset_s': reset_times, 'act_s': act_times,
  'actions': [np.asarray(action, dtype=np.float32).tolist() for action in actions],
  'finite_action': all(bool(np.isfinite(action).all()) and np.asarray(action).shape == (3,) for action in actions),
  'process_rss_bytes': process_rss_bytes(),
  'planner_enabled': agent.planner_enabled,
  'planner_settings': {name: getattr(agent.planner, name) for name in ('horizon', 'population', 'iterations', 'candidate_batch_size', 'uncertainty_cost')},
}))
"""
    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(directory_path)
        completed = subprocess.run(
            [python_executable, "-c", child], cwd=directory_path, text=True, capture_output=True, timeout=25, check=False
        )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if result["init_s"] > 10.0 or max(result["reset_s"]) > 5.0 or max(result["act_s"]) > 5.0:
        raise RuntimeError(f"CPU inference deadline exceeded: {result}")
    if result["process_rss_bytes"] is None:
        raise RuntimeError(f"unable to measure process RSS: {result}")
    if not result["finite_action"] or result["process_rss_bytes"] > MAX_MEMORY_BYTES:
        raise RuntimeError(f"CPU inference validity or memory check failed: {result}")
    if result["planner_enabled"] != expected_planner_enabled or result["planner_settings"] != expected_planner_settings:
        raise RuntimeError(f"packaged planner runtime differs from requested selection: {result}")
    return result


def validate_checkpoints(policy_checkpoint: Path, dynamics_checkpoint: Path) -> None:
    """Fail packaging if either CPU inference checkpoint cannot load exactly."""
    try:
        policy_payload = torch.load(policy_checkpoint, map_location="cpu", weights_only=True)
        dynamics_payload = torch.load(dynamics_checkpoint, map_location="cpu", weights_only=True)
        policy = VisualActorCritic()
        dynamics = LatentDynamicsEnsemble()
        policy.load_state_dict(policy_payload["model_state"], strict=True)
        dynamics.load_state_dict(dynamics_payload["model"], strict=True)
    except (
        KeyError,
        RuntimeError,
        ValueError,
        OSError,
        FileNotFoundError,
        pickle.UnpicklingError,
        EOFError,
    ) as error:
        raise ValueError("policy.pt and dynamics.pt must be strict-loadable CPU checkpoints") from error


def resolve_package_selection(
    evaluation_summary: Path | None, *, planner_override: bool | None = None
) -> tuple[bool, dict[str, Any]]:
    """Use recorded evaluation selection; absent evidence packages PPO-only safely."""
    if evaluation_summary is None:
        enabled = False
        settings = dict(DEFAULT_PACKAGED_PLANNER_SETTINGS)
    else:
        report = json.loads(evaluation_summary.read_text(encoding="utf-8"))
        settings = report.get("selected_planner_settings")
        if not isinstance(settings, dict):
            raise ValueError("evaluation summary lacks selected_planner_settings")
        enabled = bool(report.get("package_planner_enabled", False))
    if planner_override is not None:
        enabled = bool(planner_override)
    return enabled, dict(settings)


def build_submission(
    *, source_root: Path, policy_checkpoint: Path, dynamics_checkpoint: Path, archive_path: Path, smoke_test: bool = False,
    planner_enabled: bool = True, planner_settings: dict[str, Any] | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Package the two trained CPU checkpoints with only their inference code."""
    sources = _archive_sources(source_root)
    checkpoints = {"policy.pt": policy_checkpoint, "dynamics.pt": dynamics_checkpoint}
    for target, path in checkpoints.items():
        if path.name != target:
            raise ValueError(f"{target} checkpoint must retain its inference filename")
        if not path.is_file():
            raise FileNotFoundError(path)
    selected_settings = validate_planner_settings(
        DEFAULT_PACKAGED_PLANNER_SETTINGS if planner_settings is None else planner_settings
    )
    validate_checkpoints(policy_checkpoint, dynamics_checkpoint)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sources.items():
            if name == "haic_agent/runtime_config.py":
                archive.writestr(
                    name,
                    '"""Package-selected inference mode."""\n\n'
                    f'PLANNER_ENABLED = {bool(planner_enabled)!r}\n'
                    f'PLANNER_SETTINGS = {selected_settings!r}\n'
                    'STRICT_CHECKPOINT_LOADING = True\n',
                )
            else:
                archive.write(path, arcname=name)
        for name, path in checkpoints.items():
            archive.write(path, arcname=name)
    manifest = validate_submission_archive(archive_path)
    smoke = (
        smoke_submission(archive_path, expected_planner_enabled=bool(planner_enabled),
                         expected_planner_settings=selected_settings,
                         python_executable=python_executable)
        if smoke_test else None
    )
    if manifest["planner_enabled"] != bool(planner_enabled):
        raise RuntimeError("submission planner configuration does not match packaging selection")
    if manifest["planner_settings"] != selected_settings:
        raise RuntimeError("submission planner settings do not match packaging selection")
    return {"archive": str(archive_path), "layout": manifest, "smoke": smoke, "planner_enabled": bool(planner_enabled),
            "planner_settings": selected_settings}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--dynamics-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/haic/submission/submission.zip"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--evaluation-summary", type=Path)
    planner_group = parser.add_mutually_exclusive_group()
    planner_group.add_argument("--enable-planner", action="store_true")
    planner_group.add_argument("--disable-planner", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    override = True if args.enable_planner else False if args.disable_planner else None
    planner_enabled, planner_settings = resolve_package_selection(
        args.evaluation_summary, planner_override=override
    )
    result = build_submission(
        source_root=args.source_root,
        policy_checkpoint=args.policy_checkpoint,
        dynamics_checkpoint=args.dynamics_checkpoint,
        archive_path=args.output,
        smoke_test=args.smoke_test,
        planner_enabled=planner_enabled,
        planner_settings=planner_settings,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
