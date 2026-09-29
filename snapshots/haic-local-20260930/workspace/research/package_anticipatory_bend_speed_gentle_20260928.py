"""Add the stable-exit speed gate to the fixed anticipatory ZIP."""

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts/haic-research-v2/anticipatory-bend-candidate-20260928/submission-anticipatory-bend.zip"
SPEED_SOURCE = ROOT / "haic_agent/anticipatory_bend_speed_gentle_runtime.py"
OUT_DIR = ROOT / "artifacts/haic-research-v2/anticipatory-bend-speed-gentle-candidate-20260928"
OUT = OUT_DIR / "submission-anticipatory-bend-speed-gentle.zip"
CONTROL_SHA = "ce28ab744956bc4898d102c41ff6184ee4ac5a819a8559a656a6aedda4008f66"
AGENT = '''"""Pixel-only early bend steering with steering-aware speed."""

from base_agent import Agent as _BaseAgent
from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.anticipatory_bend_runtime import AnticipatoryBendAgent
from haic_agent.anticipatory_bend_speed_gentle_runtime import AnticipatoryBendGentleSpeedAgent


class Agent:
    def __init__(self):
        base = _BaseAgent(planner_enabled=False, strict_checkpoint_loading=True)
        preview = AnticipatoryBendAgent(ObstacleFullRoadGuardAgent(base))
        self._controller = AnticipatoryBendGentleSpeedAgent(preview)

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
    with zipfile.ZipFile(CONTROL) as z:
        payloads = {name: z.read(name) for name in z.namelist()}
    payloads["agent.py"] = AGENT.encode("utf-8")
    payloads["haic_agent/anticipatory_bend_speed_gentle_runtime.py"] = SPEED_SOURCE.read_bytes()
    with zipfile.ZipFile(OUT, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, content in payloads.items():
            z.writestr(name, content)
    manifest = {
        "archive": str(OUT.relative_to(ROOT)), "archive_sha256": digest(OUT.read_bytes()),
        "control_archive": str(CONTROL.relative_to(ROOT)), "control_sha256": CONTROL_SHA,
        "speed_source": str(SPEED_SOURCE.relative_to(ROOT)),
        "speed_source_sha256": digest(SPEED_SOURCE.read_bytes()),
        "builder_sha256": digest(Path(__file__).read_bytes()),
        "files": {name: digest(content) for name, content in sorted(payloads.items())},
    }
    (OUT_DIR / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive_sha256": manifest["archive_sha256"], "files": len(payloads), "bytes": OUT.stat().st_size}))


if __name__ == "__main__":
    main()
