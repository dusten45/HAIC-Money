"""Package isolated minimum-clearance KOI with verified frozen dependencies."""

import argparse
import ast
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import zipfile

from scripts.package_koi_adaptive_avoidance import (
    CROSSING_HASH, ROOT, SOURCE_COMMIT, digest, frozen_baseline_files,
)


ENTRY = b"""from haic_agent.minimum_clearance_runtime import MinimumClearanceAgent
class Agent:
    def __init__(self): self.driver = MinimumClearanceAgent()
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
def pixels(y=None, x=42):
    frame = np.full((84,84), .1, dtype=np.float32)
    frame[5:73,27:58] = .4
    frame[77:83,10:13] = (.27 + .085*45)/18
    if y is not None: frame[y:y+3,x-1:x+2] = .7
    return np.tile(frame[None], (4,1,1))
candidate = Agent()
baseline = ContactContinuityAgent('crossing_projection')
clear = pixels()
for _ in range(15):
    assert np.array_equal(candidate.act(clear), baseline.act(clear))
candidate.reset(clear)
baseline.reset(clear)
for _ in range(10):
    assert np.array_equal(candidate.act(pixels(28,45)), baseline.act(pixels(28,45)))
old = baseline.act(pixels(30,45))
new = candidate.act(pixels(30,45))
info = candidate.last_step_diagnostics()
assert new.shape == (3,) and new.dtype == np.float32 and np.isfinite(new).all()
assert np.array_equal(old[1:], new[1:])
assert info['minimum_clearance_reason'] == 'minimum_clearance'
assert info['minimum_clearance_changed']
assert abs(new[0]) < abs(old[0])
assert info['baseline_projection_active']
assert baseline.last['target_speed'] == info['target_speed']
obstacle_result = dict(baseline_action=old.tolist(), candidate_action=new.tolist(),
                       displacement_px=info['minimum_clearance_displacement_px'],
                       target_x=info['minimum_clearance_target_x'])
candidate.reset(clear)
baseline.reset(clear)
for _ in range(12):
    old = baseline.act(pixels(38))
    new = candidate.act(pixels(38))
assert np.array_equal(old, new)
assert candidate.last_step_diagnostics()['minimum_clearance_reason'] == 'applied_command_guard'
current = pixels(38,49)
previous = clear[-1].copy()
previous[42] = .1
previous[42,20:49] = .4
current[-2] = previous
candidate.reset(clear)
baseline.reset(clear)
for _ in range(12):
    old = baseline.act(current)
    new = candidate.act(current)
assert np.array_equal(old, new)
info = candidate.last_step_diagnostics()
assert abs(info['correction'] - .1375) < 1e-6
assert info['minimum_clearance_displacement_px'] == 0
assert info['minimum_clearance_reason'] == 'applied_command_guard'
deleted = pixels(30,45)
deleted[:,56] = .1
candidate.reset(clear)
baseline.reset(clear)
for _ in range(12):
    old = baseline.act(deleted)
    new = candidate.act(deleted)
assert np.array_equal(old, new)
assert candidate.last_step_diagnostics()['minimum_clearance_reason'] == 'unsupported_sweep'
candidate.reset(clear)
baseline.reset(clear)
for _ in range(12):
    old = baseline.act(pixels(38, 34))
    new = candidate.act(pixels(38, 34))
    assert np.array_equal(old[1:], new[1:])
info = candidate.last_step_diagnostics()
assert info['minimum_clearance_reason'] == 'return_centerline'
assert not info['baseline_projection_active'] and info['baseline_avoidance_active']
assert new[0] == info['minimum_clearance_road_steer']
candidate.driver.driver.impact_left = baseline.impact_left = 3
assert np.array_equal(candidate.act(pixels(36)), baseline.act(pixels(36)))
assert candidate.last_step_diagnostics()['minimum_clearance_reason'] == 'impact_guard'
candidate.reset(clear)
baseline.reset(clear)
for _ in range(12):
    assert np.array_equal(candidate.act(pixels(10)), baseline.act(pixels(10)))
candidate.reset(clear)
baseline.reset(clear)
for _ in range(12):
    assert np.array_equal(candidate.act(pixels(61)), baseline.act(pixels(61)))
info = candidate.last_step_diagnostics()
assert info['minimum_clearance_reason'] == 'no_near_obstacle'
assert not info['return_centerline_active']
for name, module in list(sys.modules.items()):
    if name == 'agent' or name.startswith('haic_agent.'):
        assert Path(module.__file__).resolve().is_relative_to(Path.cwd())
print(json.dumps(dict(prefix_equal=True, no_obstacle_equal=True, far_only_equal=True,
                     same_observation_pedals_equal=True, minimum_displacement_active=True,
                     projection_off_return_active=True, impact_fallback_equal=True, obstacle=obstacle_result,
                     centered_command_fallback_equal=True,
                     applied_correction_regression_equal=True, deleted_approach_regression_equal=True,
                     cropped_detector_fallback_equal=True,
                     environment_resets=0)))
"""


def build_package(output, repo_root=ROOT) -> dict[str, Any]:
    output = Path(output).resolve()
    receipt = output.with_suffix(".manifest.json")
    if output.suffix != ".zip" or not output.parent.is_dir():
        raise ValueError("output must be a .zip in an existing directory")
    if output.exists() or receipt.exists():
        raise FileExistsError("refusing to overwrite a candidate or its manifest")
    baseline = frozen_baseline_files(repo_root)
    files = dict(baseline)
    files["agent.py"] = ENTRY
    files["haic_agent/minimum_clearance_runtime.py"] = (
        Path(repo_root) / "haic/algorithms/koi/minimum_clearance.py"
    ).read_bytes()
    for name, data in files.items():
        ast.parse(data, filename=name)
    with tempfile.TemporaryDirectory(prefix="koi-minimum-clearance-", dir="/tmp/kilo") as temporary:
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
            raise RuntimeError(f"zero-reset real-source smoke failed: {result.stderr}")
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
        candidate="koi-minimum-clearance-v1",
        status="zero-reset source-tested; driving improvement unverified",
        source_commit=SOURCE_COMMIT, reference_crossing_zip_sha256=CROSSING_HASH,
        baseline_restoration="frozen source reconstruction, not original crossing ZIP restoration",
        baseline_source_sha256={name: digest(data) for name, data in sorted(baseline.items())},
        candidate_zip_sha256=digest(packaged),
        files=[dict(path=name, bytes=len(data), sha256=digest(data)) for name, data in sorted(files.items())],
        steering_only=True, baseline_speed_target_and_pedal_computation_unchanged=True,
        minimum_scope="baseline-selected side, .05px grid, geometric reference with additional full-current-command arc guard",
        physical_safety_certified=False,
        limitations=["Reference/kinematic tracking and wheel response are unverified dynamics hypotheses",
                     "Full-current-command guard deliberately retains baseline when future counter-steering is needed",
                     "Known current-car silhouette masks asphalt; no claim about hidden surface continuity",
                     "Detector crop/dropout retains baseline parity, not proof of physical full rear-clear"],
        smoke=smoke, official_submission=False, driving_evaluation_performed=False,
    )
    created = []
    try:
        for path, payload in ((output, packaged), (receipt, (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode())):
            with path.open("xb") as stream:
                identity = os.fstat(stream.fileno())
                created.append((path, identity.st_dev, identity.st_ino))
                stream.write(payload)
    except BaseException:
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
    parser.add_argument("--output", type=Path, default=ROOT / "submissions/koi-minimum-clearance-v1.zip")
    args = parser.parse_args()
    print(json.dumps(build_package(args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
