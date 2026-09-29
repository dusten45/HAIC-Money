"""Frozen ZIP replay and untouched held-out comparison; no policy tuning."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from training.confirm_stable_package import check_archive
from training.package_fixed_high_speed_damping import CHILD as ORIGINAL_CHILD

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts/haic-research-v2/far-hazard-20260930'
CONTROL = ROOT/'artifacts/haic-research-v2/fast-completion-row-repair-20260929/submission-fast-completion.zip'
CONTROL_HASH = '7bc38a3bdeab7cfdf437cfbf5c06f15a81cfa18fad5317748ab13a259b8ea835'
ENTRY = '''from haic_agent.far_hazard_runtime import FarHazardAgent
class Agent:
    def __init__(self): self.driver = FarHazardAgent('arrival_speed')
    def reset(self, observation): self.driver.reset(observation)
    def act(self, observation): return self.driver.act(observation)
    def last_step_diagnostics(self): return self.driver.last_step_diagnostics()
'''
CHILD = ORIGINAL_CHILD.replace("haic_agent.__path__.append(str(root/'haic_agent'))", "for name, module in list(sys.modules.items()):\n    if name.startswith('haic_agent.') and getattr(module, '__file__', None):\n        assert Path(module.__file__).resolve().is_relative_to(Path.cwd())\nhaic_agent.__path__.append(str(root/'haic_agent'))")
CHILD = CHILD.replace('import numpy as np\n', '').replace('driver=agent.Agent()', 'driver=agent.Agent()\nimport numpy as np')

def digest(data):
    return hashlib.sha256(data).hexdigest()

def signature(row):
    keys = ('steer', 'gas', 'brake', 'car_x', 'car_y', 'car_yaw', 'progress')
    return [[t[k] for k in keys] for t in row['decision_trace']]

def run(output):
    start = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    assert digest(CONTROL.read_bytes()) == CONTROL_HASH
    manifest = json.loads((ROOT/'runs/haic-research-v2/far-hazard-20260930/run_manifest.json').read_text())
    with zipfile.ZipFile(CONTROL) as archive:
        files = {n: archive.read(n) for n in archive.namelist()}
    files['agent.py'] = ENTRY.encode()
    for name in ('haic_agent/acceleration_envelope_runtime.py', 'haic_agent/far_hazard_runtime.py'):
        files[name] = (ROOT/name).read_bytes()
    for name, data in files.items():
        if name in manifest['source_hashes']:
            assert digest(data) == manifest['source_hashes'][name], name
    candidate = output.parent/'submission-arrival-speed.zip'
    with zipfile.ZipFile(candidate, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    hashes = dict(control=CONTROL_HASH, candidate=digest(candidate.read_bytes()))
    checks = {arm: check_archive(path) for arm, path in [('control', CONTROL), ('candidate', candidate)]}
    references = json.loads((REFERENCE/'report.json').read_text())
    refs = {(r['track'], r['seed']): r for r in references['rows'] if r['arm']=='arrival_speed'}
    assert sum(r['arm']=='arrival_speed' for r in references['rows']) == 12
    assert set(refs) == {(t,s) for t in (1,2,3) for s in range(38300,38304)}
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        directories = {}
        for arm, path in [('control', CONTROL), ('candidate', candidate)]:
            destination = Path(temporary)/arm
            destination.mkdir()
            with zipfile.ZipFile(path) as archive:
                archive.extractall(destination)
            directories[arm] = destination
        cells = [('train_replay', t, s, ('candidate',)) for t, s in sorted(refs)]
        cells += [('held_out', t, s, ('candidate','control') if (t+s)%2 else ('control','candidate')) for t in (1,2,3) for s in range(49300,49304)]
        for split, track, seed, arms in cells:
            for arm in arms:
                begin = time.monotonic()
                child = subprocess.run([sys.executable, '-c', CHILD, str(ROOT), str(track), str(seed)], cwd=directories[arm], capture_output=True, text=True, timeout=90)
                if child.returncode:
                    raise RuntimeError(child.stderr[-3000:])
                row = json.loads(child.stdout.strip().splitlines()[-1])
                row.update(arm=arm, split=split, duration_s=time.monotonic()-begin)
                if split == 'train_replay':
                    ref = refs[track, seed]
                    raw = (REFERENCE/ref['evidence']).read_bytes()
                    assert digest(raw) == ref['sha256']
                    before = json.loads(raw)
                    row['same_full_trace'] = signature(row) == signature(before)
                    assert row['same_full_trace']
                    assert all(row[k] == before[k] for k in ('completed','lapTimeMs','collisions','damage','retire_reason'))
                path = output.parent/f'{split}-{track}-{seed}-{arm}.json'
                with path.open('x') as stream:
                    json.dump(row, stream)
                assert not row['error'] and not row['invalid_actions']
                assert row['import_create_s'] < 10 and row['reset_s'] < 5 and row['act_max_ms'] < 5000 and row['peak_rss_bytes'] < 1024**3
                compact = {k: row[k] for k in ('arm','split','track_id','seed','completed','lapTimeMs','progress','collisions','damage','retire_reason','import_create_s','reset_s','act_p50_ms','act_p95_ms','act_max_ms','peak_rss_bytes','duration_s')}
                compact.update(evidence=path.name, sha256=digest(path.read_bytes()), same_full_trace=row.get('same_full_trace'))
                compact['prefix_hash'] = digest(json.dumps(signature(row)[:10]).encode())
                rows.append(compact)
                print(json.dumps(compact), flush=True)
    summary = {}
    for track in (1,2,3):
        for seed in range(49300,49304):
            pair = [r for r in rows if r['split']=='held_out' and r['track_id']==track and r['seed']==seed]
            assert len(pair)==2 and len({r['prefix_hash'] for r in pair})==1
    for arm in ('control','candidate'):
        selected = [r for r in rows if r['split']=='held_out' and r['arm']==arm]
        times = [r['lapTimeMs'] for r in selected if r['completed']]
        summary[arm] = dict(completed=len(times), denominator=len(selected), median_finished_ms=statistics.median(times) if times else None)
    assert digest(candidate.read_bytes()) == hashes['candidate'] and digest(CONTROL.read_bytes()) == CONTROL_HASH
    with output.open('x') as stream:
        json.dump(dict(rows=rows, summary=summary, hashes=hashes, archive_checks=checks, candidate_archive=candidate.name, duration_s=time.monotonic()-start), stream, indent=2)
