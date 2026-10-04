"""Build an isolated adaptive-speed candidate without altering frozen sources."""

import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "c4e224d465eea12acc49d2700ba1e1e5b6146976"
RELEASE = "releases/arrival-speed-20260930"
WORKSPACE = "snapshots/haic-local-20260930/workspace"
ARRIVAL_HASH = "b168a17dac5fe5d6b28a265fb4adeeba6b3346391b35acd4a5818e7dc0b9693b"
CONTACT_HASH = "bd0518542d0c9165c2d5c87be81d4c77bb82183c4d0334269389154143d34e80"
CROSSING_HASH = "4447c8dfad1392ddbbb83f0b3d23b8d5237af2b9be785cdfd08edaf6c4623f64"
ENTRY = b"""from haic_agent.adaptive_avoidance_runtime import AdaptiveAvoidanceAgent
class Agent:
    def __init__(self): self.driver = AdaptiveAvoidanceAgent()
    def reset(self, observation): self.driver.reset(observation)
    def act(self, observation): return self.driver.act(observation)
    def last_step_diagnostics(self): return self.driver.last_step_diagnostics()
"""
SMOKE = """
import json
from pathlib import Path
import sys
import numpy as np
from agent import Agent
from haic_agent.contact_continuity_runtime import ContactContinuityAgent
frame = np.full((84,84), .1, dtype=np.float32)
frame[20:63,27:58] = .4
frame[77:83,10:13] = (.27 + .085*60)/18
clear = np.tile(frame[None], (4,1,1))
frame[48:52,34:37] = .7
obstacle = np.tile(frame[None], (4,1,1))
candidate = Agent()
baseline = ContactContinuityAgent('crossing_projection')
prefix_equal = True
for i in range(12):
    old = baseline.act(obstacle)
    new = candidate.act(obstacle)
    assert new.dtype == np.float32 and new.shape == (3,)
    assert np.isfinite(new).all()
    assert -1 <= new[0] <= 1 and 0 <= new[1] <= 1 and 0 <= new[2] <= 1
    assert old[0] == new[0]
    if i < 10:
        prefix_equal = prefix_equal and np.array_equal(old, new)
info = candidate.last_step_diagnostics()
assert prefix_equal and info['adaptive_speed_reason'] == 'adaptive'
assert new[2] < old[2] and info['adaptive_pass_speed'] > 44
obstacle_result = dict(baseline_action=old.tolist(), candidate_action=new.tolist(),
                       pass_speed=info['adaptive_pass_speed'], target=info['adaptive_speed_target'])
candidate.reset(clear)
baseline.reset(clear)
for _ in range(12):
    assert np.array_equal(candidate.act(clear), baseline.act(clear))
far_frame = clear[-1].copy()
far_frame[10:13,34:37] = .7
far = np.tile(far_frame[None], (4,1,1))
candidate.reset(far)
baseline.reset(far)
for _ in range(12):
    assert np.array_equal(candidate.act(far), baseline.act(far))
assert candidate.last_step_diagnostics()['adaptive_speed_reason'] == 'no_near_obstacle'
first = clear[-1].copy()
first[44:47,41:44] = .7
second = clear[-1].copy()
second[46:49,41:44] = .7
first = np.tile(first[None], (4,1,1))
second = np.tile(second[None], (4,1,1))
candidate.reset(first)
baseline.reset(first)
for _ in range(10):
    assert np.array_equal(candidate.act(first), baseline.act(first))
assert np.array_equal(candidate.act(second), baseline.act(second))
assert candidate.last_step_diagnostics()['contact_active']
assert candidate.last_step_diagnostics()['adaptive_speed_reason'] == 'steering_guard'
for name, module in list(sys.modules.items()):
    if name == 'agent' or name.startswith('haic_agent.'):
        assert Path(module.__file__).resolve().is_relative_to(Path.cwd())
print(json.dumps(dict(prefix_equal=True, no_obstacle_equal=True, far_only_equal=True,
                     moving_crossing_equal=True,
                     obstacle=obstacle_result, environment_resets=0)))
"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frozen_baseline_files(repo_root: Path = ROOT) -> dict[str, bytes]:
    def blob(path):
        return subprocess.run(
            ["git", "show", f"{SOURCE_COMMIT}:{path}"],
            cwd=repo_root, capture_output=True, check=True,
        ).stdout

    manifest = json.loads(blob(f"{RELEASE}/source-manifest.json"))
    original = blob(f"{RELEASE}/submission-arrival-speed.zip")
    if digest(original) != ARRIVAL_HASH or manifest["zip_sha256"] != ARRIVAL_HASH:
        raise ValueError("frozen arrival ZIP hash mismatch")
    with zipfile.ZipFile(io.BytesIO(original)) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    if set(files) != {row["path"] for row in manifest["files"]}:
        raise ValueError("frozen arrival layout mismatch")
    for row in manifest["files"]:
        data = files[row["path"]]
        if len(data) != row["bytes"] or digest(data) != row["sha256"]:
            raise ValueError(f"frozen source mismatch: {row['path']}")
    contact = blob(f"{WORKSPACE}/haic_agent/contact_continuity_runtime.py")
    if digest(contact) != CONTACT_HASH:
        raise ValueError("frozen contact module hash mismatch")
    tree = ast.parse(blob(f"{WORKSPACE}/training/confirm_contact_continuity.py"))
    entries = [
        ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ENTRY" for target in node.targets)
    ]
    if len(entries) != 1 or not isinstance(entries[0], str):
        raise ValueError("missing frozen crossing entry point")
    files["agent.py"] = entries[0].encode()
    files["haic_agent/contact_continuity_runtime.py"] = contact
    return files


def build_package(output: Path, repo_root: Path = ROOT) -> dict[str, Any]:
    output = Path(output).resolve()
    receipt = output.with_suffix(".manifest.json")
    if output.suffix != ".zip" or not output.parent.is_dir():
        raise ValueError("output must be a .zip in an existing directory")
    if output.exists() or receipt.exists():
        raise FileExistsError("refusing to overwrite a candidate or its manifest")
    baseline = frozen_baseline_files(repo_root)
    files = dict(baseline)
    files["agent.py"] = ENTRY
    files["haic_agent/adaptive_avoidance_runtime.py"] = (
        Path(repo_root) / "haic/algorithms/koi/adaptive_avoidance.py"
    ).read_bytes()
    for name, data in files.items():
        ast.parse(data, filename=name)
    with tempfile.TemporaryDirectory(prefix="koi-adaptive-", dir="/tmp/kilo") as temporary:
        directory = Path(temporary)
        for name, data in files.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        result = subprocess.run(
            [sys.executable, "-I", "-c", "import sys; sys.path.insert(0, '.');\n" + SMOKE],
            cwd=directory, capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if result.returncode:
            raise RuntimeError(f"synthetic package smoke failed: {result.stderr}")
        smoke = json.loads(result.stdout)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    packaged = buffer.getvalue()
    manifest = dict(
        candidate="koi-adaptive-avoidance-v1",
        status="synthetic-tested; driving improvement unverified",
        source_commit=SOURCE_COMMIT,
        reference_crossing_zip_sha256=CROSSING_HASH,
        baseline_restoration="frozen source reconstruction, not original crossing ZIP restoration",
        baseline_source_sha256={name: digest(data) for name, data in sorted(baseline.items())},
        candidate_zip_sha256=digest(packaged),
        files=[dict(path=name, bytes=len(data), sha256=digest(data)) for name, data in sorted(files.items())],
        smoke=smoke,
        official_submission=False,
        driving_evaluation_performed=False,
    )
    receipt_bytes = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    created = []
    try:
        for path, payload in ((output, packaged), (receipt, receipt_bytes)):
            with path.open("xb") as stream:
                identity = os.fstat(stream.fileno())
                created.append((path, identity.st_dev, identity.st_ino))
                stream.write(payload)
    except BaseException:
        # Roll back only our own files, never a pre-existing or replaced artifact.
        for path, device, inode in reversed(created):
            try:
                identity = path.stat()
                if (identity.st_dev, identity.st_ino) == (device, inode):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submissions/koi-adaptive-avoidance-v1.zip")
    args = parser.parse_args()
    manifest = build_package(args.output)
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
