"""Build the unchanged speed-budget pixel agent over a fixed speed-coupled ZIP."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from training.package_submission import static_violations


BASE_SHA256 = "54c117287390704acc52505d680c31574fdcf53ac062cbb0ce5258a57058c324"
SOURCE_SHA256 = "4c4dddbb3e33b0abb0b8daf0639784690dd6301e5a532ce7807d6d4e6ee6e9af"
ENTRY = '''"""Pixel-only bend preview with the measured speed budget."""

from base_agent import Agent as _BaseAgent
from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.anticipatory_bend_runtime import AnticipatoryBendAgent
from haic_agent.speed_coupled_preview_runtime import SpeedCoupledPreviewAgent
from haic_agent.high_speed_pivot_runtime import HighSpeedPivotAgent


class Agent:
    def __init__(self):
        base = _BaseAgent(planner_enabled=False, strict_checkpoint_loading=True)
        preview = AnticipatoryBendAgent(ObstacleFullRoadGuardAgent(base))
        speed = SpeedCoupledPreviewAgent(preview)
        self._controller = HighSpeedPivotAgent(speed, "speed_budget")

    def reset(self, observation):
        self._controller.reset(observation)

    def act(self, observation):
        return self._controller.act(observation)
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_package(base_archive: Path, candidate_source: Path, output: Path) -> dict:
    base_bytes = base_archive.read_bytes()
    source_bytes = candidate_source.read_bytes()
    if digest(base_bytes) != BASE_SHA256 or digest(source_bytes) != SOURCE_SHA256:
        raise ValueError("frozen base archive or speed-budget source differs from measured controller")
    with zipfile.ZipFile(base_archive) as base:
        names = base.namelist()
        if ("agent.py" not in names or len(names) != len(set(names))
                or any(name.startswith("/") or ".." in Path(name).parts or "\\" in name for name in names)):
            raise ValueError("unexpected base archive inventory")
        files = {name: base.read(name) for name in names}
    files["agent.py"] = ENTRY.encode("utf-8")
    files["haic_agent/high_speed_pivot_runtime.py"] = source_bytes
    if len(files) > 1000 or sum(map(len, files.values())) > 2_000_000_000:
        raise ValueError("archive exceeds mirrored file count or extracted size limit")
    for name, data in files.items():
        if len(data) > 500_000_000:
            raise ValueError(f"package file too large: {name}")
        if name.endswith(".py"):
            violations = static_violations(data.decode("utf-8"), name)
            if violations:
                raise ValueError(f"{name}: {violations}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    if output.stat().st_size > 500_000_000:
        raise ValueError("archive exceeds mirrored ZIP size limit")
    manifest = {"archive": str(output), "archive_sha256": digest(output.read_bytes()),
                "base_sha256": BASE_SHA256, "candidate_source_sha256": SOURCE_SHA256,
                "files": {name: digest(data) for name, data in sorted(files.items())},
                "official_submission": False}
    with (output.parent / "package_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-archive", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_package(args.control_archive, args.candidate_source, args.output)
    print(json.dumps({"archive_sha256": result["archive_sha256"], "file_count": len(result["files"])}))


if __name__ == "__main__":
    main()
