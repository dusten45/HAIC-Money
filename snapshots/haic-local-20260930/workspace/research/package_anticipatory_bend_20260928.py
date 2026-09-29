"""Package the frozen anticipatory bend source on the fixed control archive."""

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts/haic-research-v2/full-road-guard-candidate-20260928/submission-full-road-guard.zip"
SOURCE = ROOT / "haic_agent/anticipatory_bend_runtime.py"
OUTPUT_DIR = ROOT / "artifacts/haic-research-v2/anticipatory-bend-candidate-20260928"
OUTPUT = OUTPUT_DIR / "submission-anticipatory-bend.zip"
CONTROL_SHA = "5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40"
SOURCE_SHA = "27459733edf6d845b61296f68fa82eb17af9672e2a6c88337ff8ab6d9a2f91d2"
AGENT = '''"""HAIC pixel-only anticipatory bend, actor-only runtime."""

from base_agent import Agent as _BaseAgent
from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.anticipatory_bend_runtime import AnticipatoryBendAgent


class Agent:
    def __init__(self):
        base = _BaseAgent(planner_enabled=False, strict_checkpoint_loading=True)
        self._controller = AnticipatoryBendAgent(ObstacleFullRoadGuardAgent(base))

    def reset(self, observation):
        self._controller.reset(observation)

    def act(self, observation):
        return self._controller.act(observation)
'''


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if sha256(CONTROL.read_bytes()) != CONTROL_SHA:
        raise RuntimeError("control package changed")
    if sha256(SOURCE.read_bytes()) != SOURCE_SHA:
        raise RuntimeError("candidate source changed")
    if OUTPUT_DIR.exists():
        raise RuntimeError("package output already exists")
    OUTPUT_DIR.mkdir(parents=True)
    with zipfile.ZipFile(CONTROL) as control:
        payloads = {name: control.read(name) for name in control.namelist()}
    payloads["agent.py"] = AGENT.encode("utf-8")
    payloads["haic_agent/anticipatory_bend_runtime.py"] = SOURCE.read_bytes()
    with zipfile.ZipFile(OUTPUT, "x", compression=zipfile.ZIP_DEFLATED) as package:
        for name, data in payloads.items():
            package.writestr(name, data)
    manifest = {
        "archive": str(OUTPUT.relative_to(ROOT)),
        "archive_sha256": sha256(OUTPUT.read_bytes()),
        "control_archive": str(CONTROL.relative_to(ROOT)),
        "control_sha256": CONTROL_SHA,
        "candidate_source": str(SOURCE.relative_to(ROOT)),
        "candidate_source_sha256": SOURCE_SHA,
        "package_script_sha256": sha256(Path(__file__).read_bytes()),
        "files": {name: sha256(data) for name, data in sorted(payloads.items())},
    }
    (OUTPUT_DIR / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": manifest["archive"], "sha256": manifest["archive_sha256"], "file_count": len(payloads), "bytes": OUTPUT.stat().st_size}))


if __name__ == "__main__":
    main()
