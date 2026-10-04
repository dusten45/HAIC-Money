"""Package launch and speed-gated early steering over the selected ZIP."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from training.package_submission import static_violations


CONTROL_SHA256 = "5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40"
SOURCES = (
    "haic_agent/launch_surge_runtime.py",
    "haic_agent/anticipatory_bend_runtime.py",
    "haic_agent/launch_preview_speed_gate_runtime.py",
)
ENTRY = '''"""Pixel-only opening acceleration and speed-gated bend preview."""

from base_agent import Agent as _BaseAgent
from haic_agent.launch_preview_speed_gate_runtime import LaunchPreviewSpeedGateAgent


class Agent:
    def __init__(self):
        base = _BaseAgent(planner_enabled=False, strict_checkpoint_loading=True)
        self._controller = LaunchPreviewSpeedGateAgent(base)

    def reset(self, observation):
        self._controller.reset(observation)

    def act(self, observation):
        return self._controller.act(observation)
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build_package(control_archive, candidate_source, output):
    if digest(control_archive.read_bytes()) != CONTROL_SHA256:
        raise ValueError("control archive differs from registered ZIP")
    if candidate_source.resolve() != Path(SOURCES[-1]).resolve():
        raise ValueError("unexpected composition source")
    with zipfile.ZipFile(control_archive) as archive:
        names = archive.namelist()
        if ("agent.py" not in names or len(names) != len(set(names))
                or any(name.startswith("/") or ".." in Path(name).parts or "\\" in name for name in names)):
            raise ValueError("unexpected control archive inventory")
        files = {name: archive.read(name) for name in names}
    files["agent.py"] = ENTRY.encode("utf-8")
    for name in SOURCES:
        files[name] = Path(name).read_bytes()
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
                "control_sha256": CONTROL_SHA256,
                "files": {name: digest(data) for name, data in sorted(files.items())},
                "official_submission": False}
    with (output.parent / "package_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-archive", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_package(args.control_archive, args.candidate_source, args.output)
    print(json.dumps({"archive_sha256": manifest["archive_sha256"],
                      "file_count": len(manifest["files"])}))


if __name__ == "__main__":
    main()
