"""Package frozen shield v1 without policy edits, driving evaluation, or upload."""

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

from package_submission import (
    BANNED_SUFFIXES, MAX_ARCHIVE_BYTES, MAX_COMPRESSION_RATIO,
    MAX_EXTRACTED_BYTES, MAX_FILE_COUNT, agent_static_violations,
)
from scripts.package_koi_adaptive_avoidance import digest


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/koi-collision-shield-v1'
VERSION = 'koi-collision-shield-v1-experimental-submission'
CROSSING_SHA = 'a4b35c56659555f5e60a372261df722628060e4afcd73c88ed9e9688cdb82bb8'
SHIELD_SHA = 'ad772bde9a9c3f4596fdfc742e7cac33f1b605ff52a2b3f76fbd4b75d6361d96'
RESULT_SHA = '25819a469f5e44dec9ae216feb67e38ca06cbc6ab71ee9440a7772e97a048ea5'
PYTHON = Path('/tmp/kilo/haic-cpu21/bin/python')
ENTRY = b'''from haic_agent.contact_continuity_runtime import ContactContinuityAgent
from haic_agent.collision_shield_runtime import CollisionShieldAgent

class Agent:
    def __init__(self):
        self.driver = CollisionShieldAgent(ContactContinuityAgent('crossing_projection'))

    def reset(self, observation):
        self.driver.reset(observation)

    def act(self, observation):
        return self.driver.act(observation)

    def last_step_diagnostics(self):
        return self.driver.last_step_diagnostics()
'''
SMOKE = r'''
import sys, time, json, resource, importlib.util, importlib.metadata
sys.path.insert(0, '.')
started = time.perf_counter()
from agent import Agent
agent = Agent()
init_time = time.perf_counter() - started
import numpy as np
import cv2
from haic_agent.contact_continuity_runtime import ContactContinuityAgent
assert sys.version_info[:2] == (3, 11)
assert np.__version__ == '1.26.0'
assert importlib.metadata.version('opencv-python') == '4.8.1.78'
spec = importlib.util.spec_from_file_location('frozen_shield_reference', sys.argv[1])
reference_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_module)
reference = reference_module.CollisionShieldAgent(ContactContinuityAgent('crossing_projection'))
def pixels(index):
    obs = np.full((4, 84, 84), .1, np.float32)
    obs[:, 5:73, 27:58] = .4
    obs[:, 77:83, 10:13] = (.27 + .085 * 45.) / 18.
    if index >= 12:
        y = 26 + 2 * (index - 12)
        obs[:, y:y+3, 40:43] = .7
    return obs
actions, timings, resets, changed = [], [], [], 0
for episode in range(2):
    start = time.perf_counter()
    agent.reset(pixels(0))
    resets.append(time.perf_counter() - start)
    reference.reset(pixels(0))
    sequence = []
    for index in range(32):
        obs = pixels(index)
        start = time.perf_counter()
        action = agent.act(obs)
        timings.append(time.perf_counter() - start)
        expected = reference.act(obs)
        assert action.shape == (3,) and action.dtype == np.float32
        assert np.isfinite(action).all()
        assert -1 <= action[0] <= 1 and np.all((action[1:] >= 0) & (action[1:] <= 1))
        assert np.array_equal(action, expected)
        assert agent.driver.last_shield == reference.last_shield
        changed += int(agent.driver.last_shield['active'])
        sequence.append(action.tolist())
    actions.append(sequence)
assert actions[0] == actions[1]
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
assert init_time < 10 and max(resets) < 5 and max(timings) < 5
assert rss < 1024 * 1024 * 1024
print(json.dumps(dict(python=sys.version.split()[0], numpy=np.__version__,
    opencv=importlib.metadata.version('opencv-python'),
    torch=importlib.metadata.version('torch'), import_construct_seconds=init_time,
    max_reset_seconds=max(resets), max_act_seconds=max(timings), peak_rss_bytes=rss,
    act_calls=len(timings), synthetic_changed_actions=changed,
    exact_frozen_action_and_diagnostic_parity=True, deterministic_reset=True,
    finite_bounded_float32_actions=True, environment_constructions=0, environment_resets=0)))
'''


def build_package(output):
    output = Path(output).resolve()
    receipt_path = output.with_suffix('.receipt.json')
    if output.exists() or receipt_path.exists():
        raise FileExistsError('refusing to overwrite any package or receipt')
    if not output.parent.is_dir() or output.suffix != '.zip':
        raise ValueError('output needs an existing parent and .zip suffix')
    baseline = RUN / 'crossing-projection-source-reconstruction.zip'
    shield = RUN / 'source/collision_shield.py'
    result = ROOT / 'experiments/koi-collision-shield-v1-result.json'
    for path, expected in ((baseline, CROSSING_SHA), (shield, SHIELD_SHA), (result, RESULT_SHA)):
        if digest(path.read_bytes()) != expected:
            raise ValueError(f'frozen input changed: {path}')
    if json.loads(result.read_text())['verdict'] != 'NOT_ADOPTED':
        raise ValueError('the original research verdict must remain NOT_ADOPTED')
    protected = set((ROOT / 'submissions').rglob('*.zip')) | {
        baseline, shield, result, ROOT / 'agent.py', RUN / 'protocol.json',
        ROOT / 'haic/algorithms/koi/collision_shield.py',
        ROOT / 'haic/algorithms/koi/steering_release.py',
        ROOT / 'submissions/koi-steering-release-v2.manifest.json',
        ROOT / 'runs/koi-steering-generalization-v1/crossing-projection-source-reconstruction.zip',
    }
    before = {str(path): digest(path.read_bytes()) for path in sorted(protected)}
    with zipfile.ZipFile(baseline) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('invalid frozen baseline archive')
        files = {name: archive.read(name) for name in archive.namelist()}
    baseline_hashes = {name: digest(data) for name, data in files.items()}
    files['agent.py'] = ENTRY
    files['haic_agent/collision_shield_runtime.py'] = shield.read_bytes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or path.suffix in BANNED_SUFFIXES:
                raise ValueError(f'unsafe archive member: {name}')
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    package = buffer.getvalue()
    if len(package) > MAX_ARCHIVE_BYTES or len(files) > MAX_FILE_COUNT:
        raise ValueError('archive size/count limit')
    with tempfile.TemporaryDirectory(prefix='koi-shield-submission-', dir='/tmp/kilo') as temp:
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            if archive.testzip() is not None or 'agent.py' not in archive.namelist():
                raise ValueError('CRC/root entrypoint failure')
            infos = archive.infolist()
            if sum(info.file_size for info in infos) > MAX_EXTRACTED_BYTES:
                raise ValueError('extracted size limit')
            for info in infos:
                if (info.file_size > MAX_ARCHIVE_BYTES
                        or info.file_size / max(1, info.compress_size) > MAX_COMPRESSION_RATIO):
                    raise ValueError('member size/compression limit')
                if archive.read(info.filename) != files[info.filename]:
                    raise ValueError('member byte parity failure')
            archive.extractall(temp)
        for name in files:
            if name.endswith('.py'):
                errors = agent_static_violations(Path(temp) / name)
                if errors:
                    raise ValueError(f'{name}: {errors}')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH', None)
        completed = subprocess.run([str(PYTHON), '-I', '-B', '-c', SMOKE, str(shield)],
                                   cwd=temp, env=env, capture_output=True, text=True, timeout=30)
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        smoke = json.loads(completed.stdout)
    if any(digest(Path(path).read_bytes()) != expected for path, expected in before.items()):
        raise ValueError('protected artifact changed while packaging')
    for name, expected in baseline_hashes.items():
        if name != 'agent.py' and digest(files[name]) != expected:
            raise ValueError('crossing dependency changed')
    receipt = dict(version=VERSION, status='EXPERIMENTAL_SUBMISSION_CANDIDATE_FROZEN_V1',
        package=str(output.relative_to(ROOT)), zip_sha256=digest(package), archive_bytes=len(package),
        extracted_bytes=sum(map(len, files.values())), policy_source_sha256=SHIELD_SHA,
        baseline_zip_sha256=CROSSING_SHA, unchanged_research_result_sha256=RESULT_SHA,
        research_verdict='NOT_ADOPTED', purpose='prevent imminent obstacle-collision DNF',
        ordinary_overavoidance='separate follow-up; no changes in this candidate',
        policy_and_parameters_byte_identical=True, crossing_dependencies_byte_identical=True,
        packaging_only_change='Static root Agent instantiates unchanged shield over unchanged crossing driver.',
        files=[dict(path=name, sha256=digest(data), bytes=len(data)) for name, data in sorted(files.items())],
        static_zip_checks_pass=True, runtime_smoke=smoke, protected_artifacts_unchanged=before,
        packager_sha256=digest(Path(__file__).read_bytes()),
        official_sources_checked_date='2026-10-01', official_sources=[
            'https://scholarships-hardwood-headers-influenced.trycloudflare.com/',
            'https://github.com/2026-HAIC/Participants'],
        additional_requirements=False, external_weights=False, official_upload_performed=False,
        official_server_acceptance_verified=False,
        limitations=['Synthetic packaging smoke only; no additional simulator evaluation or tuning.',
                     'Local packaging validation is not official Docker/server acceptance.'])
    with output.open('xb') as stream:
        stream.write(package)
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'submissions' / (VERSION + '.zip'))
    args = parser.parse_args()
    result = build_package(args.output)
    print(json.dumps({key: result[key] for key in
                      ('version', 'package', 'zip_sha256', 'archive_bytes', 'runtime_smoke')}, indent=2))


if __name__ == '__main__':
    main()
