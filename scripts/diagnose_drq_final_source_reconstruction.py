"""Run the frozen 192-episode TRAIN-DIAGNOSTIC gate for the child study.

The parent diagnostic and source-pool protocol remain byte-identical. This
adapter binds child treatment paths while temporarily validating each sealed
pool collection against its original parent study identity.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from scripts import diagnose_drq_final_replay as diagnostic
from scripts import train_drq_final_source_reconstruction as reconstruction


PROTOCOL_PATH = reconstruction.PROTOCOL_PATH
STUDY_ID = reconstruction.STUDY_ID
RUN_ROOT = reconstruction.RUN_ROOT
SAMPLE_AUDIT = reconstruction.SAMPLE_AUDIT_PATH
OUTPUT_ROOT = reconstruction.DIAGNOSTIC_OUTPUT_ROOT
PARENT_STUDY_ID = "drqv2-final-source-replay-v1"
PARENT_RUN_ROOT = reconstruction.exposure_audit.PARENT_RUN_ROOT
CPU_PREFIX = Path("/tmp/kilo/haic-cpu21")
CPU_PACKAGES = {
    "torch": "2.1.0+cpu",
    "gymnasium": "0.29.1",
    "box2d-py": "2.3.5",
    "numpy": "1.26.0",
    "opencv-python": "4.8.1.78",
    "pygame": "2.6.1",
    "stable-baselines3": "2.2.1",
}


class DiagnosticGateError(RuntimeError):
    """Raised when the child diagnostic protocol or CPU runtime is unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticGateError(message)


def _validate_cpu_runtime() -> None:
    import torch

    _require(sys.version.split()[0] == reconstruction.REQUIRED_PYTHON,
             "CPU21 diagnostic Python version mismatch")
    _require(Path(sys.prefix).resolve().is_relative_to(CPU_PREFIX),
             "diagnostic must run inside the isolated CPU21 environment")
    _require(torch.__version__ == CPU_PACKAGES["torch"]
             and torch.version.cuda is None and not torch.cuda.is_available(),
             "diagnostic must use the pinned CPU-only Torch 2.1 runtime")
    for package, expected in CPU_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise DiagnosticGateError(f"CPU21 runtime package missing: {package}") from exc
        _require(actual == expected, f"CPU21 package version mismatch for {package}: {actual}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _require(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1,
             "CPU21 Torch thread limit did not apply")


@contextmanager
def _parent_collection_identity_context() -> Iterator[None]:
    original_pools = diagnostic._pools
    child_study_id = diagnostic.STUDY_ID
    child_run_root = diagnostic.RUN_ROOT

    def pools_with_parent_identity(root: Path, protocol: dict[str, Any],
                                   sources: dict[int, dict[str, Any]],
                                   diagnostic_roads: set[int]) -> dict[int, dict[str, str]]:
        diagnostic.STUDY_ID = PARENT_STUDY_ID
        diagnostic.RUN_ROOT = Path(PARENT_RUN_ROOT)
        try:
            return original_pools(root, protocol, sources, diagnostic_roads)
        finally:
            diagnostic.STUDY_ID = child_study_id
            diagnostic.RUN_ROOT = child_run_root

    diagnostic._pools = pools_with_parent_identity
    try:
        yield
    finally:
        diagnostic._pools = original_pools
        diagnostic.STUDY_ID = child_study_id
        diagnostic.RUN_ROOT = child_run_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, default=Path(PROTOCOL_PATH))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    protocol_path = root / args.protocol
    if protocol_path != root / PROTOCOL_PATH:
        raise DiagnosticGateError("diagnostic is pinned to the reconstruction protocol")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    reconstruction._validate_protocol(root, protocol)
    reconstruction._validate_exposure_snapshot(root, protocol)
    _validate_cpu_runtime()

    diagnostic.STUDY_ID = STUDY_ID
    diagnostic.RUN_ROOT = Path(RUN_ROOT)
    diagnostic.SAMPLE_AUDIT = Path(SAMPLE_AUDIT)
    output_root = root / OUTPUT_ROOT
    diagnostic_argv = [
        "--repo-root", str(root),
        "--protocol", str(protocol_path),
        "--output-root", str(output_root),
    ]
    if args.preflight_only:
        diagnostic_argv.append("--preflight-only")
    with _parent_collection_identity_context():
        old_argv = sys.argv
        sys.argv = ["diagnose_drq_final_replay.py", *diagnostic_argv]
        try:
            diagnostic.main()
        finally:
            sys.argv = old_argv
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DiagnosticGateError, reconstruction.GateError,
            reconstruction.exposure_audit.AuditError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
