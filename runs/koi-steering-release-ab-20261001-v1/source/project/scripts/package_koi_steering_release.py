"""Package the isolated KOI steering lifecycle candidate without real resets."""

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


ENTRY = b"""from haic_agent.steering_release_runtime import SteeringReleaseAgent
class Agent:
    def __init__(self): self.driver = SteeringReleaseAgent()
    def reset(self, observation): self.driver.reset(observation)
    def act(self, observation): return self.driver.act(observation)
    def last_step_diagnostics(self): return self.driver.last_step_diagnostics()
"""
SMOKE = """
import json
import numpy as np
from agent import Agent
from haic_agent.contact_continuity_runtime import ContactContinuityAgent
def pixels(y=None, x=42, speed=45):
    frame = np.full((84,84), .1, dtype=np.float32)
    frame[5:73,27:58] = .4
    frame[77:83,10:13] = (.27 + .085 * speed) / 18
    if y is not None: frame[y:y+3,x-1:x+2] = .7
    return np.tile(frame[None], (4,1,1))
candidate, baseline = Agent(), ContactContinuityAgent('crossing_projection')
for _ in range(20):
    obs = pixels()
    assert np.array_equal(candidate.act(obs), baseline.act(obs))
    assert not candidate.last_step_diagnostics()['steering_release_changed']
    assert candidate.last_step_diagnostics()['steering_terms']['reconstruction_valid']
candidate.reset(pixels())
baseline.reset(pixels())
for _ in range(10):
    obs = pixels(30,35)
    assert np.array_equal(candidate.act(obs), baseline.act(obs))
obs = pixels(34,33)
old, new = baseline.act(obs), candidate.act(obs)
info = candidate.last_step_diagnostics()
assert info['steering_release_reason'] == 'near_release'
assert info['steering_release_changed'] and abs(new[0]) < abs(old[0])
assert np.array_equal(new[1:], old[1:])
assert info['target_speed'] == baseline.last_step_diagnostics()['target_speed']
assert info['steering_terms']['reconstruction_context'] == 'actual_postrelease'
assert info['steering_terms']['reconstruction_valid'] and info['steering_terms']['residual_valid']
assert info['steering_release_projection_band_px'] == 6
for y,x in ((38,33),(42,33),(46,33)):
    obs = pixels(y,x)
    action = candidate.act(obs)
    info = candidate.last_step_diagnostics()
    assert np.array_equal(action[1:], info['baseline_pedals'])
    assert info['steering_terms']['reconstruction_valid'] and info['steering_terms']['residual_valid']
candidate.driver.driver.impact_left = 3
action = candidate.act(pixels(50,33))
info = candidate.last_step_diagnostics()
assert not info['steering_release_changed']
assert info['impact_left_before_act'] == 3
assert info['steering_release_reason'] == 'protected_replacement'
candidate.driver.driver.impact_left = 0
candidate.act(pixels(56,33))
info = candidate.last_step_diagnostics()
assert info['steering_release_reason'] == 'near_release'
assert not info['steering_release_projection_available']
candidate = Agent()
for _ in range(10): candidate.act(pixels(30,35))
candidate.act(pixels(34,33))
obs = pixels(38,33)
obs[-1,77:83,10:13] = (.27 + .085 * 90) / 18
action = candidate.act(obs)
info = candidate.last_step_diagnostics()
assert not info['steering_release_changed']
assert info['pixel_speed'] < 80 and info['steering_release_reason'] == 'hud_uncertainty'
candidate, baseline = Agent(), ContactContinuityAgent('crossing_projection')
for _ in range(14):
    obs = pixels(18,32)
    assert np.array_equal(candidate.act(obs), baseline.act(obs))
    assert not candidate.last_step_diagnostics()['steering_release_changed']
print(json.dumps(dict(no_obstacle_equal=True, first10_equal=True, far_only_equal=True,
                      safe_release_active=True, staged_accounting_valid=True,
                      same_observation_pedals_target_equal=True, impact_fallback=True,
                      raw_saturation_fallback=True, original_projection_band_preserved=True,
                      unavailable_projection_tail_validated=True,
                      environment_constructions=0, environment_resets=0)))
"""


def build_package(output, repo_root=ROOT, candidate_name='koi-steering-release-v1') -> dict[str, Any]:
    if not candidate_name.startswith('koi-steering-release-'):
        raise ValueError('candidate name must identify the isolated steering-release study')
    root, output = Path(repo_root), Path(output).resolve()
    manifest_path = output.with_suffix('.manifest.json')
    if output.exists() or manifest_path.exists():
        raise FileExistsError('refusing to overwrite a frozen candidate or manifest')
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    files = frozen_baseline_files(root)
    baseline_members = {name: digest(source) for name, source in files.items()}
    input_hashes = {str(root / relative): digest((root / relative).read_bytes()) for relative in
                   ('scripts/package_koi_adaptive_avoidance.py', 'scripts/package_koi_steering_release.py')}
    inputs = (('haic/algorithms/koi/steering_release.py', 'haic_agent/steering_release_runtime.py'),
              ('haic/algorithms/koi/steering_terms.py', 'haic_agent/steering_terms.py'))
    for relative, member in inputs:
        source = (root / relative).read_bytes()
        ast.parse(source)
        input_hashes[str(root / relative)] = digest(source)
        files[member] = source
    files['agent.py'] = ENTRY
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, source in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source)
    package = buffer.getvalue()
    with tempfile.TemporaryDirectory(prefix='koi-steering-release-smoke-', dir='/tmp/kilo') as directory:
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            archive.extractall(directory)
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        smoke = subprocess.run([sys.executable, '-I', '-c', "import sys; sys.path.insert(0, '.');\n" + SMOKE],
                               cwd=directory, env=env, check=True, capture_output=True, text=True, timeout=60)
        receipt = json.loads(smoke.stdout.strip())
    for path, expected in input_hashes.items():
        if digest(Path(path).read_bytes()) != expected:
            raise ValueError(f'input changed during package smoke: {path}')
    manifest = dict(candidate=candidate_name, status='ZERO_RESET_TESTED_UNEVALUATED_LIFECYCLE_CANDIDATE',
                    source_commit=SOURCE_COMMIT, reference_crossing_zip_sha256=CROSSING_HASH,
                    reconstructed_from_frozen_sources=True, original_crossing_zip_missing=True,
                    candidate_zip_sha256=digest(package), input_sha256=input_hashes,
                    baseline_source_sha256=baseline_members,
                    files=[dict(path=name, sha256=digest(source), bytes=len(source)) for name, source in sorted(files.items())],
                    changed_members=['agent.py', 'haic_agent/steering_release_runtime.py', 'haic_agent/steering_terms.py'],
                    policy_change='release near/urgency term only after same-object validated projection clearance and reentry guards',
                    speed_target_changed=False, margin_reduction=False, projection_band_px=6.,
                    observation_contract='pixels/HUD/history only; no world state', smoke=receipt,
                    smoke_command='isolated frozen-source staged-accounting/near-release synthetic stacks; zero real resets',
                    limitations=['Current-command arc, lag and speed assumptions are not physical safety certification',
                                 'Pixel-component continuity is heuristic, not a physical object ID; circle enclosure is a raster assumption',
                                 'Unchanged six-pixel projection band and fixed footprint/raster allowances; no minimum displacement target',
                                 'No driving improvement or fresh/generalization evidence before separately frozen consumed-TRAIN A/B'])
    for path, data in ((output, package), (manifest_path, json.dumps(manifest, indent=2, allow_nan=False).encode() + b'\n')):
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'submissions/koi-steering-release-v1.zip')
    parser.add_argument('--candidate-name', default='koi-steering-release-v1')
    args = parser.parse_args()
    print(json.dumps(build_package(args.output, candidate_name=args.candidate_name), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
