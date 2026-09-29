"""Build the frozen pixel-only hybrid candidate as a local submission archive."""

import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from training.package_submission import static_violations


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/policy.pt"
EXPECTED_CHECKPOINT_SHA256 = "3ceb5ebbe2a4bd96944649a693b8ef924c0252fa620b0fd547c7fb494afff199"
OUTPUT = ROOT / "artifacts/haic-research-v2/score-confirm-hybrid-20260928"
SOURCES = (
    "haic_agent/__init__.py",
    "haic_agent/observation.py",
    "haic_agent/pixel_features.py",
    "haic_agent/networks.py",
    "haic_agent/dynamics.py",
    "haic_agent/planner.py",
    "haic_agent/corridor_agent.py",
    "haic_agent/hybrid_runtime.py",
)
ENTRY = '''"""HAIC pixel-only obstacle hybrid, actor-only runtime."""

from base_agent import Agent as _BaseAgent
from haic_agent.hybrid_runtime import ObstacleHybridAgent


class Agent:
    def __init__(self):
        base = _BaseAgent(planner_enabled=False, strict_checkpoint_loading=True)
        self._hybrid = ObstacleHybridAgent(base)

    def reset(self, observation):
        self._hybrid.reset(observation)

    def act(self, observation):
        return self._hybrid.act(observation)
'''
RUNTIME_CONFIG = '''"""Frozen actor-only submission runtime."""
PLANNER_ENABLED = False
PLANNER_SETTINGS = {"horizon": 4, "population": 16, "iterations": 2,
                    "candidate_batch_size": 8, "uncertainty_cost": 1.0}
STRICT_CHECKPOINT_LOADING = True
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    checkpoint_bytes = CHECKPOINT.read_bytes()
    if digest(checkpoint_bytes) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("historical checkpoint bytes differ from the frozen comparison")
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
    archive_path = OUTPUT / "submission-obstacle-hybrid.zip"
    if archive_path.exists():
        raise FileExistsError(archive_path)
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(archive_path) as archive:
        if set(archive.namelist()) != set(files) or any(info.file_size > 500_000_000 for info in archive.infolist()):
            raise ValueError("archive inventory differs from frozen inputs")
    if archive_path.stat().st_size > 500_000_000 or sum(map(len, files.values())) > 2_000_000_000:
        raise ValueError("archive exceeds the official size limits")
    manifest = {
        "archive": str(archive_path.relative_to(ROOT)),
        "archive_sha256": digest(archive_path.read_bytes()),
        "checkpoint_sha256": digest(checkpoint_bytes),
        "files": {name: digest(data) for name, data in files.items()},
        "source_sha256": digest(Path(__file__).read_bytes()),
        "selection_evidence": "runs/haic-research-v2/score-probe-hybrid-control-20260928",
        "confirmation_evidence": "runs/haic-research-v2/score-confirm-hybrid-20260928",
        "official_submission": False,
    }
    (OUTPUT / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive_path), "bytes": archive_path.stat().st_size,
                      "sha256": manifest["archive_sha256"], "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
