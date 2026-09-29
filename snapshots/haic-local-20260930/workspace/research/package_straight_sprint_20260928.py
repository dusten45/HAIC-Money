"""Build the frozen straight-sprint candidate as a local archive."""

import hashlib
import json
from pathlib import Path
import zipfile

from training.package_submission import static_violations
from research.package_full_road_guard_20260928 import (
    CHECKPOINT, EXPECTED_CHECKPOINT_SHA256, ROOT, RUNTIME_CONFIG,
    SOURCES as PARENT_SOURCES,
)


OUTPUT = ROOT / "artifacts/haic-research-v2/straight-sprint-candidate-20260928"
SOURCES = PARENT_SOURCES + ("haic_agent/straight_sprint_runtime.py",)
ENTRY = '''"""HAIC pixel-only straight sprint, actor-only runtime."""

from base_agent import Agent as _BaseAgent
from haic_agent.straight_sprint_runtime import StraightSprintAgent


class Agent:
    def __init__(self):
        base = _BaseAgent(planner_enabled=False, strict_checkpoint_loading=True)
        self._controller = StraightSprintAgent(base)

    def reset(self, observation):
        self._controller.reset(observation)

    def act(self, observation):
        return self._controller.act(observation)
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    checkpoint_bytes = CHECKPOINT.read_bytes()
    if digest(checkpoint_bytes) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("historical checkpoint bytes differ from frozen comparison")
    files = {"agent.py": ENTRY.encode("utf-8"), "base_agent.py": (ROOT / "agent.py").read_bytes(),
             "haic_agent/runtime_config.py": RUNTIME_CONFIG.encode("utf-8"),
             "policy.pt": checkpoint_bytes}
    files.update({name: (ROOT / name).read_bytes() for name in SOURCES})
    for name, data in files.items():
        if name.endswith(".py"):
            violations = static_violations(data.decode("utf-8"), name)
            if violations:
                raise ValueError(f"{name}: {violations}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive_path = OUTPUT / "submission-straight-sprint.zip"
    if archive_path.exists():
        raise FileExistsError(archive_path)
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(archive_path) as archive:
        if set(archive.namelist()) != set(files) or any(info.file_size > 500_000_000 for info in archive.infolist()):
            raise ValueError("archive inventory differs from frozen inputs")
    if archive_path.stat().st_size > 500_000_000 or sum(map(len, files.values())) > 2_000_000_000:
        raise ValueError("archive exceeds local mirrored size limits")
    manifest = {
        "archive": str(archive_path.relative_to(ROOT)),
        "archive_sha256": digest(archive_path.read_bytes()),
        "checkpoint_sha256": digest(checkpoint_bytes),
        "files": {name: digest(data) for name, data in files.items()},
        "source_sha256": digest(Path(__file__).read_bytes()),
        "tune_evidence": "runs/haic-research-v2/score-linux-straight-sprint-tune-20260928",
        "heldout_evidence": "runs/haic-research-v2/score-linux-straight-sprint-heldout-20260928",
        "official_submission": False,
    }
    (OUTPUT / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive_path), "bytes": archive_path.stat().st_size,
                      "sha256": manifest["archive_sha256"], "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
