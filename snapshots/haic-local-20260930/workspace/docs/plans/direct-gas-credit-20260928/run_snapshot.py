"""Run the registered TRAIN-only HAIC profile from one immutable container copy."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
from pathlib import Path


SOURCE = Path("/source")
WORK = Path("/workspace")
RUN_ID = "direct-gas-credit-train-snapshot-20260928"
META = WORK / "docs/plans/direct-gas-credit-20260928"


def main() -> int:
    for directory in ("core", "haic_agent", "haic_research", "training"):
        shutil.copytree(
            SOURCE / directory,
            WORK / directory,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for filename in (
        "agent.py", "env_wrapper.py", "damage.py", "harness.config.json",
        "AGENTS.md", "PROJECT_INFO.md", "RESTRICTIONS.md", "README.md",
        "COMPETITION_INFO.md",
    ):
        shutil.copy2(SOURCE / filename, WORK / filename)
    (WORK / "docs/plans").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        SOURCE / "docs/plans/2026-09-28-direct-gas-credit-probe.md",
        WORK / "docs/plans/2026-09-28-direct-gas-credit-probe.md",
    )
    os.chdir(WORK)
    sys.path.insert(0, str(WORK))
    from haic_research.cli import main as haic_main

    manifest = json.loads((META / "manifest.json").read_text(encoding="utf-8-sig"))
    manifest["run_id"] = RUN_ID
    manifest["candidate_revision"] = "direct-gas-credit8-snapshot-4096steps"
    manifest["source_hashes"] = {
        name: hashlib.sha256((WORK / name).read_bytes()).hexdigest()
        for name in (
            "training/train_policy.py", "training/ppo.py", "haic_research/config.py",
            "haic_research/commands.py", "harness.config.json",
        )
    }
    manifest_path = META / "manifest-snapshot.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    plan_output = META / "plan-snapshot-output.json"
    with plan_output.open("w", encoding="utf-8") as stream:
        status = haic_main(
            ["plan", "--manifest", str(manifest_path), "--research", str(META / "hypothesis.json"),
             "--profile", "train_policy", "--arguments", str(META / "arguments.json")],
            stdout=stream,
        )
    if status:
        return status
    for stage in ("design", "implementation", "execution"):
        status = haic_main(
            ["approve", RUN_ID, stage, "--source-ref", "user-local-repeat-request-2026-09-28"],
            stdout=io.StringIO(),
        )
        if status:
            return status
    with (META / "run-snapshot-output.json").open("w", encoding="utf-8") as stream:
        status = haic_main(["run", RUN_ID, "--execute"], stdout=stream)
    print(json.dumps({"run_id": RUN_ID, "exit_code": status, "snapshot": True}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
