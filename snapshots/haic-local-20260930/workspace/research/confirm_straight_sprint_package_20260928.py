"""Run the frozen straight-sprint archive on unused confirmation cells."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

from research.confirm_full_road_guard_package_20260928 import CHILD, ROOT, digest


ARCHIVE = ROOT / "artifacts/haic-research-v2/straight-sprint-candidate-20260928/submission-straight-sprint.zip"
RUN = ROOT / "runs/haic-research-v2/score-linux-straight-sprint-package-confirmation-20260928"
CELLS = tuple((track, seed) for track in (1, 2) for seed in (216, 217, 218, 219))


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN.name, "purpose": "frozen package confirmation on unused cells",
        "split": "confirmation", "cells": [{"track_id": t, "seed": s} for t, s in CELLS],
        "max_decisions": 2000, "archive_sha256": digest(ARCHIVE),
        "source_sha256": digest(Path(__file__).resolve()), "created_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory)
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extractall(destination)
        completed = subprocess.run(
            [sys.executable, "-c", CHILD, str(ROOT), str(RUN), json.dumps(CELLS)],
            cwd=destination, text=True, capture_output=True, timeout=300, check=False,
        )
        print(completed.stdout, end="", flush=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr + completed.stdout)


if __name__ == "__main__":
    main()
