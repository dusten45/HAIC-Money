"""Frozen extracted-ZIP comparison of post-impact steering recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import zipfile

from training.confirm_stable_package import check_archive
from training.package_fixed_high_speed_damping import CHILD as ORIGINAL_CHILD

ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'artifacts/haic-research-v2/fixed-high-speed-damping-package-20260929/submission-high-speed-damping.zip'
CONTROL_HASH='b196cfb653a0ceeae0779e5e3832f879a27f24cc662b7202a1a0c39888d469f9'
ENTRY='''from haic_agent.fixed_high_speed_recovery_v2 import FixedHighSpeedRecoveryV2

class Agent:
    def __init__(self):
        self._controller = FixedHighSpeedRecoveryV2("impact_clear")
    def reset(self, observation):
        self._controller.reset(observation)
    def act(self, observation):
        return self._controller.act(observation)
    def last_step_diagnostics(self):
        return self._controller.last_step_diagnostics()
'''
CHILD=ORIGINAL_CHILD.replace('import numpy as np\n','').replace('driver=agent.Agent()','driver=agent.Agent()\nimport numpy as np')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(output):
    output.parent.mkdir(parents=True,exist_ok=True)
    if digest(CONTROL.read_bytes())!=CONTROL_HASH:
        raise ValueError('Original delivered ZIP changed')
    with zipfile.ZipFile(CONTROL) as archive:
        files={name:archive.read(name) for name in archive.namelist()}
    files['agent.py']=ENTRY.encode()
    name='haic_agent/fixed_high_speed_recovery_v2.py'
    files[name]=(ROOT/name).read_bytes()
    candidate=output.parent/'submission-high-speed-recovery.zip'
    with zipfile.ZipFile(candidate,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in files.items(): archive.writestr(name,data)
    hashes=dict(control=CONTROL_HASH,candidate=digest(candidate.read_bytes()))
    archives=dict(control=CONTROL,candidate=candidate)
    checks={arm:check_archive(path) for arm,path in archives.items()}
    rows=[]
    with tempfile.TemporaryDirectory() as directory:
        directories={}
        for arm,path in archives.items():
            destination=Path(directory)/arm; destination.mkdir()
            with zipfile.ZipFile(path) as archive: archive.extractall(destination)
            directories[arm]=destination
        for track in (1,2,3):
            for seed in (38210,38211):
                order=('control','candidate') if (track+seed)%2 else ('candidate','control')
                for arm in order:
                    child=subprocess.run([sys.executable,'-c',CHILD,str(ROOT),str(track),str(seed)],
                                         cwd=directories[arm],text=True,capture_output=True,timeout=120)
                    if child.returncode: raise RuntimeError(child.stderr[-3000:])
                    row=json.loads(child.stdout.strip().splitlines()[-1]); row['arm']=arm
                    trace=row['decision_trace']
                    row['effective_interventions']=sum(bool(t.get('controller') and t['controller'].get('recovery_changed')) for t in trace)
                    prefix=[[t[k] for k in ('steer','gas','brake','car_x','car_y','car_yaw')] for t in trace[:10]]
                    row['prefix_hash']=digest(json.dumps(prefix).encode())
                    path=output.parent/f'{arm}-{track}-{seed}.json'
                    with path.open('x') as stream: json.dump(row,stream)
                    if row['error'] or row['invalid_actions'] or row['import_create_s']>=10 or row['reset_s']>=5 or row['act_max_ms']>=5000 or row['peak_rss_bytes']>=1024**3:
                        raise RuntimeError(f'Invalid package episode: {arm}:{track}:{seed}')
                    compact={k:row[k] for k in ('arm','track_id','seed','completed','lapTimeMs','progress','collisions','damage','retire_reason','import_create_s','reset_s','act_p95_ms','act_max_ms','peak_rss_bytes','effective_interventions','prefix_hash')}
                    compact.update(evidence=path.name,sha256=digest(path.read_bytes()))
                    rows.append(compact); print(json.dumps(compact),flush=True)
    pairs=[]
    for track in (1,2,3):
        for seed in (38210,38211):
            arms={r['arm']:r for r in rows if r['track_id']==track and r['seed']==seed}
            a,b=arms['candidate'],arms['control']
            pairs.append(dict(track=track,seed=seed,prefix_equal=a['prefix_hash']==b['prefix_hash'],
                              preserves_control_finish=not b['completed'] or a['completed']))
    if not all(p['prefix_equal'] for p in pairs): raise RuntimeError('Matched prefix mismatch')
    summary={arm:dict(completed=sum(r['completed'] for r in rows if r['arm']==arm),denominator=6,
                      median_finished_ms=statistics.median([r['lapTimeMs'] for r in rows if r['arm']==arm and r['completed']]) if any(r['completed'] for r in rows if r['arm']==arm) else None,
                      interventions=sum(r['effective_interventions'] for r in rows if r['arm']==arm)) for arm in archives}
    for arm,path in archives.items():
        if digest(path.read_bytes())!=hashes[arm]: raise RuntimeError('ZIP changed during run')
    with output.open('x') as stream:
        json.dump(dict(hashes=hashes,checks=checks,rows=rows,pairs=pairs,summary=summary,
                       candidate_archive=candidate.name,fresh_train_only=True,official_submission=False),stream,indent=2)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
