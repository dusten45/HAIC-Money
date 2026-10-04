"""Run only the separately frozen six-arm final-source reconstruction.

The original trainer/protocol stay immutable. This adapter pins the new run root,
rechecks prior exposure and host gates, then delegates to the source-hashed parent
trainer after replacing only its two study/path constants in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts import audit_drq_final_source_reconstruction as exposure_audit
from scripts import train_drq_final_source_replay as trainer


PROTOCOL_PATH = "experiments/drqv2-final-source-replay-reconstruction-r1.json"
STUDY_ID = "drqv2-final-source-replay-reconstruction-r1"
RUN_ROOT = "runs/20260928-drqv2-final-source-replay-reconstruction-r1"
PARENT_PROTOCOL_SHA = "1d891d82e8a5d04be1fed53265df3b36d96d76d2e3317b110626f5f15d4f4e64"
SUPERSEDED_PROTOCOL_SHA = "797fd79a6b2ae8f35dd6db9ac4d25d5868273ec1cab70e9f095a7491700c6900"
PREFLIGHT_FAILURE_RECEIPT = (
    f"{RUN_ROOT}/preflight-failure-20260928T104204Z-seed0-uniform.json"
)
PREFLIGHT_FAILURE_RECEIPT_SHA = "1e10e41146fd82fa24b4322309290d9c1274bf67a945f4676e8c6a30fea3e2a4"
EXPOSURE_RECEIPT = f"{RUN_ROOT}/preflight-exposure-audit.json"
SAMPLE_AUDIT_PATH = f"{RUN_ROOT}/pre-evaluation-sample-audit.json"
DIAGNOSTIC_OUTPUT_ROOT = f"{RUN_ROOT}/train-diagnostic"
GPU_NAME = "NVIDIA GeForce RTX 4060 Ti"
GPU_DRIVER = "580.173.02"
GPU_CAPABILITY = (8, 9)
TORCH_CUDA_BUILD = "12.8"
REQUIRED_PYTHON = "3.11.14"
REQUIRED_PACKAGES = {
    "torch": "2.11.0+cu128",
    "gymnasium": "0.29.1",
    "box2d-py": "2.3.5",
    "numpy": "1.26.0",
    "opencv-python": "4.8.1.78",
    "pygame": "2.6.1",
    "stable-baselines3": "2.2.1",
    "cuda-bindings": "12.9.7",
    "cuda-pathfinder": "1.6.0",
    "filelock": "3.32.3",
    "fsspec": "2026.7.0",
}
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
RESOURCE_FLOORS = {
    "min_disk_available_bytes": 20 * 1024**3,
    "min_cgroup_available_bytes": 16 * 1024**3,
    "min_gpu_free_bytes": 8 * 1024**3,
    "max_parallel_gpu_apps": 0,
    "max_parallel_study_arms": 1,
}


class GateError(RuntimeError):
    """Raised when a frozen reconstruction gate no longer matches."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


@contextmanager
def _parent_pool_receipt_context():
    original_receipt = trainer._receipt
    child_run_root = trainer.RUN_ROOT

    def verify_parent_pool(root: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
        trainer.RUN_ROOT = Path(exposure_audit.PARENT_RUN_ROOT)
        try:
            return original_receipt(root, *args, **kwargs)
        finally:
            trainer.RUN_ROOT = child_run_root

    trainer._receipt = verify_parent_pool
    try:
        yield
    finally:
        trainer._receipt = original_receipt
        trainer.RUN_ROOT = child_run_root


def _canonical_json_sha(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    return hashlib.sha256(raw).hexdigest()


def _validate_protocol(root: Path, protocol: dict[str, Any]) -> None:
    _require(protocol.get("study_id") == STUDY_ID and protocol.get("run_root") == RUN_ROOT,
             "not the dedicated DrQ reconstruction protocol")
    parent_path = root / exposure_audit.PARENT_PROTOCOL
    _require(parent_path.is_file() and not parent_path.is_symlink()
             and hashlib.sha256(parent_path.read_bytes()).hexdigest() == PARENT_PROTOCOL_SHA,
             "frozen parent protocol identity changed")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    for key, value in parent.items():
        if key not in {"study_id", "run_root", "runs", "code_sha256"}:
            _require(protocol.get(key) == value, f"reconstruction changed inherited field {key}")
    reconstruction = protocol.get("reconstruction")
    if type(reconstruction) is not dict:
        raise GateError("missing reconstruction declaration")
    _require(reconstruction.get("method") == "fresh_matched_six_arm_from_source_forks"
             and reconstruction.get("parent_protocol_sha256") == PARENT_PROTOCOL_SHA
             and reconstruction.get("partition") == "TRAIN"
             and reconstruction.get("fresh_geometry_seeds") == []
             and reconstruction.get("all_six_arms_rerun") is True
             and reconstruction.get("partial_arm_resume") is False,
             "reconstruction method/exposure declaration changed")
    revision = reconstruction.get("protocol_revision")
    _require(revision == {
        "revision": 2,
        "supersedes_protocol_sha256": SUPERSEDED_PROTOCOL_SHA,
        "preflight_failure_receipt": {
            "path": PREFLIGHT_FAILURE_RECEIPT,
            "sha256": PREFLIGHT_FAILURE_RECEIPT_SHA,
        },
    }, "protocol revision does not preserve the zero-interaction setup failure")
    failure_path = root / PREFLIGHT_FAILURE_RECEIPT
    _require(failure_path.is_file() and not failure_path.is_symlink()
             and hashlib.sha256(failure_path.read_bytes()).hexdigest() == PREFLIGHT_FAILURE_RECEIPT_SHA,
             "superseded preflight-failure receipt is missing or changed")
    _require(reconstruction.get("postrun_gates") == {
        "sample_audit_path": SAMPLE_AUDIT_PATH,
        "sample_audit_format": "haic-drq-final-source-sample-audit-v1",
        "diagnostic_output_root": DIAGNOSTIC_OUTPUT_ROOT,
        "diagnostic_partition": "TRAIN-DIAGNOSTIC",
        "diagnostic_episode_count": 192,
        "diagnostic_repeats": 2,
        "fixed_gate": "kept >=9/11 AND gained >=2/21",
    }, "post-run sample/diagnostic gates changed")
    exposure_ref = reconstruction.get("exposure_audit")
    if type(exposure_ref) is not dict:
        raise GateError("missing exposure audit reference")
    receipt_path = root / EXPOSURE_RECEIPT
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest() if receipt_path.is_file() else None
    _require(exposure_ref == {
        "path": EXPOSURE_RECEIPT,
        "sha256": receipt_sha,
    }, "reconstruction exposure receipt reference changed or is missing")
    _require(reconstruction.get("training_runtime") == {
        "python": REQUIRED_PYTHON,
        "packages": REQUIRED_PACKAGES,
        "torch_cuda_build": TORCH_CUDA_BUILD,
        "gpu_name": GPU_NAME,
        "gpu_compute_capability": list(GPU_CAPABILITY),
        "gpu_driver": GPU_DRIVER,
        "thread_env": {name: "1" for name in THREAD_ENV},
    }, "frozen training runtime differs from the isolated verified runtime")
    _require(reconstruction.get("resource_floors") == RESOURCE_FLOORS,
             "resource floors differ from the frozen reconstruction protocol")
    rows = protocol.get("runs")
    if type(rows) is not list or len(rows) != 6:
        raise GateError("reconstruction must contain exactly six matched arms")
    expected = {
        (seed, variant, "final_source", f"{RUN_ROOT}/learner-{seed}-{variant}-final_source")
        for seed in (0, 1)
        for variant in trainer.r6.VARIANTS
    }
    actual = {
        (row.get("source_seed"), row.get("variant"), row.get("condition"), row.get("run_dir"))
        for row in rows if type(row) is dict
    }
    _require(actual == expected, "new run paths/arms do not form the fixed six-arm matrix")
    code_hashes = protocol.get("code_sha256")
    if type(code_hashes) is not dict:
        raise GateError("reconstruction code hash map missing")
    parent_hashes = parent.get("code_sha256")
    if type(parent_hashes) is not dict:
        raise GateError("parent code hash map missing")
    for name, digest in parent_hashes.items():
        _require(code_hashes.get(name) == digest,
                 f"parent source pin changed in reconstruction: {name}")
    new_sources = {
        "scripts/audit_drq_final_source_reconstruction.py",
        "scripts/train_drq_final_source_reconstruction.py",
        "scripts/audit_drq_final_source_reconstruction_samples.py",
        "scripts/diagnose_drq_final_source_reconstruction.py",
    }
    _require(set(code_hashes) == set(parent_hashes) | new_sources,
             "reconstruction code map has missing or unreviewed sources")
    for name in new_sources:
        path = root / name
        _require(path.is_file() and not path.is_symlink()
                 and hashlib.sha256(path.read_bytes()).hexdigest() == code_hashes[name],
                 f"reconstruction source hash mismatch: {name}")
    _require(code_hashes.get("scripts/train_drq_final_source_replay.py")
             == trainer.r7.sha(root / "scripts/train_drq_final_source_replay.py"),
             "parent trainer hash is not preserved in the reconstruction")


def _nvidia_query(query: str) -> list[str]:
    completed = subprocess.run(
        ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _cgroup_memory() -> tuple[int, dict[str, int]]:
    directory = Path("/sys/fs/cgroup")
    limit = (directory / "memory.max").read_text(encoding="ascii").strip()
    current = int((directory / "memory.current").read_text(encoding="ascii").strip())
    events = {}
    for line in (directory / "memory.events").read_text(encoding="ascii").splitlines():
        name, value = line.split()
        events[name] = int(value)
    if limit == "max":
        available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    else:
        available = int(limit) - current
    return available, events


def _verify_runtime_and_resources(root: Path, reconstruction: dict[str, Any]) -> dict[str, Any]:
    import torch

    _require(sys.version.split()[0] == REQUIRED_PYTHON, "Python runtime version mismatch")
    executable = Path(sys.executable)
    prefix = Path(sys.prefix).resolve()
    _require(prefix.is_relative_to(Path("/tmp/kilo/haic-drq-train211")),
             "must run inside the isolated historical-compatible DrQ venv")
    versions = {}
    for package, expected in REQUIRED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise GateError(f"runtime package is missing: {package}") from exc
        versions[package] = actual
        _require(actual == expected, f"runtime package mismatch for {package}: {actual}")
    _require(torch.version.cuda == TORCH_CUDA_BUILD and torch.cuda.is_available()
             and torch.cuda.device_count() == 1,
             "Torch CUDA build/device availability mismatch")
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    _require(gpu_name == GPU_NAME and capability == GPU_CAPABILITY,
             "GPU model or compute capability mismatch")
    driver_rows = _nvidia_query("gpu=driver_version")
    device_rows = _nvidia_query("gpu=name,memory.total,memory.free")
    _require(len(driver_rows) == 1 and driver_rows[0] == GPU_DRIVER,
             "NVIDIA driver version mismatch")
    _require(len(device_rows) == 1, "expected exactly one visible NVIDIA GPU")
    device_fields = [field.strip() for field in device_rows[0].split(",")]
    _require(len(device_fields) == 3 and device_fields[0] == GPU_NAME,
             "nvidia-smi GPU identity mismatch")
    total_mib, free_mib = int(device_fields[1]), int(device_fields[2])
    gpu_free = free_mib * 1024**2
    disk_free = shutil.disk_usage(root).free
    memory_free, memory_events = _cgroup_memory()
    app_rows = _nvidia_query("compute-apps=pid,process_name,used_memory")
    active_apps = [row for row in app_rows if not row.lower().startswith("no running processes")]
    floors = reconstruction.get("resource_floors")
    if type(floors) is not dict or floors != RESOURCE_FLOORS:
        raise GateError("resource floors not frozen or malformed")
    _require(disk_free >= floors["min_disk_available_bytes"], "free disk below frozen floor")
    _require(memory_free >= floors["min_cgroup_available_bytes"], "raw cgroup free memory below frozen floor")
    _require(gpu_free >= floors["min_gpu_free_bytes"], "free GPU memory below frozen floor")
    _require(len(active_apps) <= floors["max_parallel_gpu_apps"],
             "another CUDA process is using the shared GPU")
    for name in THREAD_ENV:
        _require(os.environ.get(name) == "1", f"{name}=1 is required for the frozen runtime")
    _require(memory_events.get("oom", 0) == 0 and memory_events.get("oom_kill", 0) == 0,
             "cgroup reports an OOM event; resource gate requires investigation")
    return {
        "python": REQUIRED_PYTHON,
        "executable": str(executable),
        "packages": versions,
        "torch_cuda_build": torch.version.cuda,
        "gpu": {
            "name": gpu_name,
            "compute_capability": list(capability),
            "driver": driver_rows[0],
            "memory_total_bytes": total_mib * 1024**2,
            "memory_free_bytes": gpu_free,
        },
        "resources": {
            "disk_available_bytes": disk_free,
            "cgroup_available_bytes": memory_free,
            "cgroup_events": memory_events,
            "other_gpu_compute_apps": active_apps,
        },
        "thread_env": {name: os.environ[name] for name in THREAD_ENV},
        "torch_thread_setup": "delegated to parent trainer before pool/model loading",
    }


def _validate_exposure_snapshot(root: Path, protocol: dict[str, Any]) -> None:
    reconstruction = protocol["reconstruction"]
    receipt = root / EXPOSURE_RECEIPT
    _require(receipt.is_file() and not receipt.is_symlink(), "frozen exposure receipt missing/unsafe")
    expected_sha = reconstruction["exposure_audit"]["sha256"]
    _require(hashlib.sha256(receipt.read_bytes()).hexdigest() == expected_sha,
             "stored exposure receipt hash changed")
    live_sha = _canonical_json_sha(exposure_audit.audit(root))
    _require(live_sha == expected_sha,
             "TRAIN exposure/claims or another-lane run state changed since protocol freeze")


def _write_runtime_receipt(root: Path, arm: tuple[int, str], snapshot: dict[str, Any]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    relative = f"{RUN_ROOT}/preflight-runtime-{stamp}-seed{arm[0]}-{arm[1]}.json"
    path = root / relative
    _require(path.parent.is_dir() and not path.parent.is_symlink(), "runtime receipt directory missing/unsafe")
    raw = json.dumps({"format": "haic-drq-reconstruction-runtime-preflight-v1",
                      "study_id": STUDY_ID, "protocol_path": PROTOCOL_PATH,
                      "arm": {"source_seed": arm[0], "variant": arm[1]},
                      "recorded_at": stamp, "runtime": snapshot},
                     sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return relative


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--source-seed", type=int, choices=(0, 1))
    parser.add_argument("--variant", choices=trainer.r6.VARIANTS)
    args = parser.parse_args(argv)
    if (args.source_seed is None) != (args.variant is None):
        raise GateError("--source-seed and --variant must be supplied together")
    if not args.preflight_only and args.source_seed is None:
        raise GateError("select one arm for training; use --preflight-only for study-wide checks")
    root = Path.cwd().resolve(strict=True)
    protocol_path = root / PROTOCOL_PATH
    _require(protocol_path.is_file() and not protocol_path.is_symlink(),
             "dedicated reconstruction protocol missing/unsafe")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _validate_protocol(root, protocol)
    reconstruction = protocol["reconstruction"]
    _validate_exposure_snapshot(root, protocol)
    runtime = _verify_runtime_and_resources(root, reconstruction)

    trainer.STUDY_ID = STUDY_ID
    trainer.RUN_ROOT = Path(RUN_ROOT)
    trainer.PROTOCOL_PATH = Path(PROTOCOL_PATH)
    trainer_argv = ["--repo-root", str(root), "--protocol", str(protocol_path)]
    arm = None
    if args.source_seed is not None and args.variant is not None:
        arm = (args.source_seed, args.variant)
        trainer_argv.extend(["--source-seed", str(args.source_seed), "--variant", args.variant])
    if args.preflight_only:
        trainer_argv.append("--preflight-only")
        print(json.dumps({"reconstruction": STUDY_ID, "runtime": runtime,
                          "exposure_audit_sha256": protocol["reconstruction"]["exposure_audit"]["sha256"]},
                         sort_keys=True))
        with _parent_pool_receipt_context():
            return trainer.main(trainer_argv)

    assert arm is not None
    with _parent_pool_receipt_context():
        run_dir = root / RUN_ROOT / f"learner-{arm[0]}-{arm[1]}-final_source"
        _require(not run_dir.exists() and not run_dir.is_symlink(),
                 "selected arm path already exists; refusing overwrite")
        receipt = _write_runtime_receipt(root, arm, runtime)
        print(json.dumps({"arm": arm, "runtime_receipt": receipt}, sort_keys=True))
        return trainer.main(trainer_argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateError, exposure_audit.AuditError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
