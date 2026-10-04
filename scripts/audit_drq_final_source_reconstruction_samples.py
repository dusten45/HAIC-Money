"""Run the frozen sample-provenance audit on the child DrQ run root.

The parent auditor remains byte-identical. This adapter changes only its fixed
study/root/receipt bindings and the one literal parent-study identity guard; the
actual child protocol digest and per-result child IDs stay unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from scripts import audit_drq_final_replay_samples as sample_audit
from scripts import train_drq_final_source_reconstruction as reconstruction


PROTOCOL_PATH = reconstruction.PROTOCOL_PATH
STUDY_ID = reconstruction.STUDY_ID
RUN_ROOT = reconstruction.RUN_ROOT
RECEIPT_PATH = reconstruction.SAMPLE_AUDIT_PATH
PARENT_STUDY_ID = "drqv2-final-source-replay-v1"
IDENTITY_GUARD = "wrong frozen six-arm protocol"


def _child_protocol_identity(protocol: dict[str, Any]) -> bool:
    return (
        protocol.get("format") == "haic-drq-final-source-replay-study-v1"
        and protocol.get("study_id") == STUDY_ID
        and protocol.get("run_root") == RUN_ROOT
        and protocol.get("r6_protocol_sha256") == sample_audit.R6_SHA
        and protocol.get("r7_protocol_sha256") == sample_audit.R7_SHA
        and protocol.get("lambda_preserve") == 0.5
    )


@contextmanager
def _child_audit_context(protocol: dict[str, Any]) -> Iterator[None]:
    old_root = sample_audit.ROOT
    old_protocol = sample_audit.PROTOCOL
    old_receipt = sample_audit.RECEIPT
    old_require = sample_audit.old.require

    def require_with_child_identity(condition: Any, message: str) -> None:
        if message == IDENTITY_GUARD:
            condition = _child_protocol_identity(protocol)
        old_require(condition, message)

    sample_audit.ROOT = Path(RUN_ROOT)
    sample_audit.PROTOCOL = Path(PROTOCOL_PATH)
    sample_audit.RECEIPT = Path(RECEIPT_PATH)
    sample_audit.old.require = require_with_child_identity
    try:
        yield
    finally:
        sample_audit.ROOT = old_root
        sample_audit.PROTOCOL = old_protocol
        sample_audit.RECEIPT = old_receipt
        sample_audit.old.require = old_require


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, default=Path(PROTOCOL_PATH))
    args = parser.parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    protocol_path = root / args.protocol
    if protocol_path != root / PROTOCOL_PATH:
        raise reconstruction.GateError("sample auditor is pinned to the reconstruction protocol")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    reconstruction._validate_protocol(root, protocol)
    reconstruction._validate_exposure_snapshot(root, protocol)
    with _child_audit_context(protocol):
        old_argv = sys.argv
        sys.argv = [
            "audit_drq_final_replay_samples.py",
            "--repo-root", str(root),
            "--protocol", str(protocol_path),
        ]
        try:
            sample_audit.main()
        finally:
            sys.argv = old_argv
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (reconstruction.GateError, reconstruction.exposure_audit.AuditError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
