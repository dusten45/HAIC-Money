"""Frozen extracted-ZIP fresh comparison of completion coordination."""
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
CONTROL=ROOT/'artifacts/haic-research-v2/fixed-high-speed-impact-fresh-20260929/submission-high-speed-recovery.zip'
CONTROL_HASH='f1eb42a4492e87f5d61641322cae3c0d5e335e895bef7ee4206ad1a570eda018'
ENTRY='''from haic_agent.fast_completion_coordination import FastCompletionCoordination

class Agent:
    def __init__(self):
        self._controller = FastCompletionCoordination("preview_action_aware")
    def reset(self, observation):
        self._controller.reset(observation)
    def act(self, observation):
        return self._controller.act(observation)
    def last_step_diagnostics(self):
        return self._controller.last_step_diagnostics()
'''
CHILD=ORIGINAL_CHILD.replace('import numpy as np\n','').replace('driver=agent.Agent()','driver=agent.Agent()\nimport numpy as np')

CHILD=CHILD.replace("haic_agent.__path__.append(str(root/'haic_agent'))", "for name in ('fixed_high_speed_recovery_v2','fast_completion_coordination'):\n    if 'haic_agent.'+name in sys.modules:\n        assert Path(sys.modules['haic_agent.'+name].__file__).resolve().parent == Path.cwd()/'haic_agent'\nhaic_agent.__path__.append(str(root/'haic_agent'))")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(output):
    output.parent.mkdir(parents=True,exist_ok=True)
    if digest(CONTROL.read_bytes())!=CONTROL_HASH:
        raise ValueError('Original delivered ZIP changed')
    with zipfile.ZipFile(CONTROL) as archive:
        files={name:archive.read(name) for name in archive.namelist()}
    files['agent.py']=ENTRY.encode()
    name='haic_agent/fast_completion_coordination.py'
    files[name]=(ROOT/name).read_bytes()
    manifest=json.loads((ROOT/'runs/haic-research-v2/fast-completion-causal-repair-train-20260929/run_manifest.json').read_text())
    if digest(files[name])!=manifest['source_hashes'][name]: raise ValueError('Tested runtime changed')
    candidate=output.parent/'submission-fast-completion.zip'
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
            for seed in (38300,38301,38302,38303):
                order=('control','candidate') if (track+seed)%2 else ('candidate','control')
                for arm in order:
                    child=subprocess.run([sys.executable,'-c',CHILD,str(ROOT),str(track),str(seed)],
                                         cwd=directories[arm],text=True,capture_output=True,timeout=120)
                    if child.returncode: raise RuntimeError(child.stderr[-3000:])
                    row=json.loads(child.stdout.strip().splitlines()[-1]); row['arm']=arm
                    trace=row['decision_trace']
                    row['effective_interventions']=sum(bool(t.get('controller') and t['controller'].get('completion_changed',t['controller'].get('recovery_changed'))) for t in trace)
                    prefix=[[t[k] for k in ('steer','gas','brake','car_x','car_y','car_yaw')] for t in trace[:10]]
                    row['prefix_hash']=digest(json.dumps(prefix).encode())
                    path=output.parent/f'{arm}-{track}-{seed}.json'
                    with path.open('x') as stream: json.dump(row,stream)
                    if row['error'] or row['invalid_actions'] or row['import_create_s']>=10 or row['reset_s']>=5 or row['act_max_ms']>=5000 or row['peak_rss_bytes']>=1024**3:
                        raise RuntimeError(f'Invalid package episode: {arm}:{track}:{seed}')
                    row['mean_actual_speed']=statistics.mean(t['speed'] for t in trace[10:])
                    row['reduced_target_fraction']=sum(t['controller'].get('target_speed',60)<60 for t in trace[10:])/len(trace[10:])
                    row['braking_proxy_veto_count']=sum(t['controller'].get('braking_proxy_veto',False) for t in trace)
                    compact={k:row[k] for k in ('arm','track_id','seed','completed','lapTimeMs','progress','collisions','damage','retire_reason','import_create_s','reset_s','act_p95_ms','act_max_ms','peak_rss_bytes','effective_interventions','prefix_hash','mean_actual_speed','reduced_target_fraction','braking_proxy_veto_count')}
                    compact.update(evidence=path.name,sha256=digest(path.read_bytes()))
                    rows.append(compact); print(json.dumps(compact),flush=True)
    pairs=[]
    for track in (1,2,3):
        for seed in (38300,38301,38302,38303):
            arms={r['arm']:r for r in rows if r['track_id']==track and r['seed']==seed}
            a,b=arms['candidate'],arms['control']
            pairs.append(dict(track=track,seed=seed,prefix_equal=a['prefix_hash']==b['prefix_hash'],
                              preserves_control_finish=not b['completed'] or a['completed']))
    if not all(p['prefix_equal'] for p in pairs): raise RuntimeError('Matched prefix mismatch')
    summary={arm:dict(completed=sum(r['completed'] for r in rows if r['arm']==arm),denominator=12,
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
