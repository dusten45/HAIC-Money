"""Package the measured road-clearance controller over the frozen control ZIP."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from training.package_submission import static_violations


CONTROL_SHA256 = "5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40"
SOURCE_SHA256 = "31eed1bc0e566d89b7b1d5fc33cb1f28f7662f383e1ae21f7ae814142997c86b"
ENTRY = '''"""Pixel-only road-clearance agent using the frozen stable controller."""

from base_agent import Agent as _BaseAgent
from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.stable_risk_envelope_runtime import StableRiskEnvelopeAgent


class Agent:
    def __init__(self):
        base = _BaseAgent(planner_enabled=False, strict_checkpoint_loading=True)
        self._controller = StableRiskEnvelopeAgent(ObstacleFullRoadGuardAgent(base))

    def reset(self, observation):
        self._controller.reset(observation)

    def act(self, observation):
        return self._controller.act(observation)
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_package(control_archive: Path, candidate_source: Path, output: Path) -> dict:
    control_bytes = control_archive.read_bytes()
    source_bytes = candidate_source.read_bytes()
    if digest(control_bytes) != CONTROL_SHA256 or digest(source_bytes) != SOURCE_SHA256:
        raise ValueError("frozen control archive or candidate source differs from confirmation")
    with zipfile.ZipFile(control_archive) as control:
        names = control.namelist()
        if len(names) != 16 or len(names) != len(set(names)) or any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("unexpected control archive inventory")
        files = {name: control.read(name) for name in names}
    files["agent.py"] = ENTRY.encode("utf-8")
    files["haic_agent/stable_risk_envelope_runtime.py"] = source_bytes
    if len(files) > 1000 or sum(map(len, files.values())) > 2_000_000_000:
        raise ValueError("archive exceeds local mirrored file or extracted-size limit")
    for name, data in files.items():
        if name.endswith(".py"):
            violations = static_violations(data.decode("utf-8"), name)
            if violations:
                raise ValueError(f"{name}: {violations}")
        if len(data) > 500_000_000:
            raise ValueError(f"individual package file exceeds limit: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    if output.stat().st_size > 500_000_000:
        raise ValueError("archive exceeds local mirrored ZIP size limit")
    manifest = {
        "archive": str(output),
        "archive_sha256": digest(output.read_bytes()),
        "control_sha256": CONTROL_SHA256,
        "candidate_source_sha256": SOURCE_SHA256,
        "files": {name: digest(data) for name, data in sorted(files.items())},
        "official_submission": False,
    }
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
