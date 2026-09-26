"""Freeze a CPU-only, non-selecting DrQ diagnostic for an immutable road catalog."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.diagnose_drq_training_geometry import _catalog_rows


CATALOG_PROTOCOL = Path("experiments/drqv2-geometry-augmentation-v1.json")
CATALOG_RESULT = Path("experiments/drqv2-geometry-augmentation-v1-catalog-result.json")
OUTPUT = Path("experiments/drqv2-geometry-augmentation-v1-diagnostic-protocol.json")
CODE_SOURCES = (
    "core/vendor/car_racing.py", "core/finish_line.py", "common_adapter.py",
    "env_wrapper.py", "train.py", "agent.py", "drq_v2.py",
    "scripts/diagnose_drq_training_geometry.py", "scripts/freeze_drq_geometry_diagnostic.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root: Path) -> dict:
    root = root.resolve()
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError("diagnostic protocol already frozen")
    catalog_protocol_path = root / CATALOG_PROTOCOL
    result_path = root / CATALOG_RESULT
    catalog_protocol = json.loads(catalog_protocol_path.read_text(encoding="utf-8"))
    catalog_result = json.loads(result_path.read_text(encoding="utf-8"))
    if catalog_result.get("status") != "success" or catalog_result.get("protocol_sha256") != _sha(catalog_protocol_path):
        raise ValueError("catalog result does not belong to the frozen generation protocol")
    catalog_path = root / catalog_result["catalog_path"]
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_sha = _sha(catalog_path)
    if catalog_sha != catalog_result["catalog_sha256"]:
        raise ValueError("catalog file hash changed")
    audit_path = root / catalog_protocol["seed_audit_receipt"]["path"]
    if _sha(audit_path) != catalog_protocol["seed_audit_receipt"]["sha256"]:
        raise ValueError("blind-safe candidate seed audit changed")
    if catalog["seed_audit"] != json.loads(audit_path.read_text(encoding="utf-8")):
        raise ValueError("catalog candidate seed audit no longer matches its frozen receipt")
    if catalog.get("protocol_sha256") != _sha(catalog_protocol_path):
        raise ValueError("catalog generation protocol lineage changed")
    train = [row["geometry_seed"] for row in catalog["train"]]
    diagnostic = [row["geometry_seed"] for row in catalog["train_diagnostic"]]
    if len(train) != 120 or len(diagnostic) != 16 or len(set(train + diagnostic)) != 136:
        raise ValueError("catalog is not the frozen 120 TRAIN plus 16 TRAIN-DIAGNOSTIC allocation")
    exclusions = catalog_protocol["exclusion_seed_ids"]
    reserved = sorted(set(exclusions["reserved"] + exclusions["heldout"] + exclusions["blind"]))
    if set(train + diagnostic) & set(reserved):
        raise ValueError("selected training geometry overlaps known held-out or blind seed IDs")
    source_actors = catalog_protocol["source_actors"]
    for actor in source_actors:
        actor_path = root / actor["actor_path"]
        if _sha(actor_path) != actor["actor_sha256"]:
            raise ValueError("frozen source actor export hash changed")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              check=True, capture_output=True, text=True).stdout.strip()
    protocol = {
        "format": "haic-drq-training-geometry-diagnostics-v1",
        "study_id": catalog_protocol["study_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": revision,
        "source_sha256": {name: _sha(root / name) for name in CODE_SOURCES},
        "catalog_path": catalog_result["catalog_path"],
        "catalog_sha256": catalog_sha,
        "catalog_protocol_sha256": catalog["protocol_sha256"],
        "catalog_scan_path": catalog_result["scan_path"],
        "catalog_scan_sha256": catalog_result["scan_sha256"],
        "analysis_receipt": catalog_protocol["analysis_receipt"],
        "role": "training_diagnostic",
        "score_selection": False,
        "frame_skip": 4,
        "max_steps": 1200,
        "raw_reward": True,
        "obstacles": True,
        "source_actors": source_actors,
        "reserved_training_seeds": reserved,
        "partitions": {
            "train": {"seeds": train},
            "train_diagnostic": {"seeds": diagnostic},
            "screen": {"seeds": sorted(set(exclusions["heldout"]))},
            "confirmation": {"seeds": sorted(set(exclusions["heldout"]))},
            "blind": {"seeds": exclusions["blind"]},
        },
        "performance_scope": "training-only DrQ raw reward/progress/damage proxy; not official or fresh confirmation",
        "blind_use": "blind seed IDs excluded, never reset/read road or outcome",
    }
    # Validate the actual generated rows against the would-be protocol *before*
    # writing its immutable bytes. This validates quotas and every seed/hash.
    _catalog_rows(catalog, protocol, catalog_sha)
    with output.open("xb") as destination:
        destination.write((json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return {
        "protocol_path": OUTPUT.as_posix(),
        "protocol_sha256": _sha(output),
        "catalog_sha256": catalog_sha,
        "train_geometry_count": len(train),
        "train_diagnostic_geometry_count": len(diagnostic),
        "blind_geometry_reset_count": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        result = freeze(args.repo_root)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
