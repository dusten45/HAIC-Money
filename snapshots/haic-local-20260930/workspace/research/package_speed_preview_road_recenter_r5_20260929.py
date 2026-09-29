"""Add joint pixel-obstacle and road-edge arbitration."""

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r4-candidate-20260929/submission-speed-preview-road-recenter-r4.zip"
SOURCE = ROOT / "haic_agent/speed_preview_road_recenter_r5_runtime.py"
OUT_DIR = ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r5-candidate-20260929"
OUT = OUT_DIR / "submission-speed-preview-road-recenter-r5.zip"
CONTROL_SHA = "168093ce191011b36e37d3bcfd926c5d3d02f9397deb82ad2b160734541b0b96"
AGENT = '''"""Pixel-only preview steering, fast cruise, and road recentering."""

from base_agent import Agent as _BaseAgent
from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.anticipatory_bend_runtime import AnticipatoryBendAgent
from haic_agent.speed_coupled_preview_runtime import SpeedCoupledPreviewAgent
from haic_agent.speed_preview_road_recenter_r5_runtime import SpeedPreviewRoadRecenterAgent


class Agent:
    def __init__(self):
        base = _BaseAgent(planner_enabled=False, strict_checkpoint_loading=True)
        preview = AnticipatoryBendAgent(ObstacleFullRoadGuardAgent(base))
        speed = SpeedCoupledPreviewAgent(preview)
        self._controller = SpeedPreviewRoadRecenterAgent(speed)

    def reset(self, observation):
        self._controller.reset(observation)

    def act(self, observation):
        return self._controller.act(observation)
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if digest(CONTROL.read_bytes()) != CONTROL_SHA:
        raise RuntimeError("control package drift")
    if OUT_DIR.exists():
        raise RuntimeError("output already exists")
    OUT_DIR.mkdir(parents=True)
    with zipfile.ZipFile(CONTROL) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    payloads["agent.py"] = AGENT.encode("utf-8")
    payloads["haic_agent/speed_preview_road_recenter_r5_runtime.py"] = SOURCE.read_bytes()
    with zipfile.ZipFile(OUT, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in payloads.items():
            archive.writestr(name, content)
    manifest = {
        "archive": str(OUT.relative_to(ROOT)), "archive_sha256": digest(OUT.read_bytes()),
        "control_archive": str(CONTROL.relative_to(ROOT)), "control_sha256": CONTROL_SHA,
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": digest(SOURCE.read_bytes()),
        "builder_sha256": digest(Path(__file__).read_bytes()),
        "files": {name: digest(content) for name, content in sorted(payloads.items())},
    }
    (OUT_DIR / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive_sha256": manifest["archive_sha256"], "files": len(payloads), "bytes": OUT.stat().st_size}))


if __name__ == "__main__":
    main()
