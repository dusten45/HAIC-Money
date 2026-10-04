"""Exact accelerated road-commit ZIP replay and additional consumed TRAIN comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import zipfile
from training.confirm_fast_completion import CHILD as BASE_CHILD
from training.confirm_stable_package import check_archive

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts/haic-research-v2/fast-road-commit-20260929'
CONTROL = ROOT/'artifacts/haic-research-v2/fast-completion-row-repair-20260929/submission-fast-completion.zip'
CONTROL_HASH = '7bc38a3bdeab7cfdf437cfbf5c06f15a81cfa18fad5317748ab13a259b8ea835'
SOURCES = ('haic_agent/geometric_passing_runtime.py', 'haic_agent/fast_road_commit_runtime.py')
ENTRY = """from haic_agent.fast_road_commit_runtime import FastRoadCommitAgent

class Agent:
    def __init__(self): self.driver = FastRoadCommitAgent('commit_side')
    def reset(self, observation): self.driver.reset(observation)
    def act(self, observation): return self.driver.act(observation)
    def last_step_diagnostics(self): return self.driver.last_step_diagnostics()
"""
CHILD = BASE_CHILD.replace("haic_agent.__path__.append(str(root/'haic_agent'))", "for name in ('geometric_passing_runtime','fast_road_commit_runtime'):\n    if 'haic_agent.'+name in sys.modules:\n        assert Path(sys.modules['haic_agent.'+name].__file__).resolve().parent == Path.cwd()/'haic_agent'\nhaic_agent.__path__.append(str(root/'haic_agent'))")
CHILD = CHILD.replace("packaged_fixed_high_speed_damping", "packaged_fast_road_commit")


def digest(data): return hashlib.sha256(data).hexdigest()

def signature(row):
    return [[t[k] for k in ('steer','gas','brake','car_x','car_y','car_yaw','progress')] for t in row['decision_trace']]


def run(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    reference_report = json.loads((REFERENCE/'report.json').read_text())
    selected = [r for r in reference_report['rows'] if r['arm']=='commit_side']
    controls = [r for r in reference_report['rows'] if r['arm']=='control']
    expected={(t,s) for t in (1,2,3) for s in (38300,38302)}
    for group in (selected,controls):
        if len(group)!=6 or {(r['track'],r['seed']) for r in group}!=expected:
            raise ValueError('historical cell set mismatch')
        for reference in group:
            data=(REFERENCE/reference['evidence']).read_bytes()
            if digest(data)!=reference['sha256']:raise ValueError('historical evidence changed')
            raw=json.loads(data)
            if (raw['track_id'],raw['seed'])!=(reference['track'],reference['seed']):raise ValueError('historical identity mismatch')
            if any(raw[k]!=reference[k] for k in ('completed','lapTimeMs','progress','collisions','damage','retire_reason')):
                raise ValueError('historical compact outcome mismatch')
    if len(selected)!=6 or not all(r['completed'] for r in selected): raise ValueError('selection prerequisite failed')
    if not all(r['lapTimeMs'] < next(c['lapTimeMs'] for c in controls if (c['track'],c['seed'])==(r['track'],r['seed'])) for r in selected):
        raise ValueError('matched acceleration improvement prerequisite failed')
    if digest(CONTROL.read_bytes()) != CONTROL_HASH: raise ValueError('control ZIP changed')
    with zipfile.ZipFile(CONTROL) as z: files={n:z.read(n) for n in z.namelist()}
    files['agent.py'] = ENTRY.encode()
    frozen = json.loads((ROOT/'runs/haic-research-v2/fast-road-commit-20260929/run_manifest.json').read_text())
    for name in SOURCES:
        data=(ROOT/name).read_bytes()
        if digest(data)!=frozen['source_hashes'][name]: raise ValueError('selected source changed')
        files[name]=data
    # Base ZIP dependencies must also match the selected source-run identities.
    for name,data in files.items():
        if name in frozen['source_hashes'] and digest(data)!=frozen['source_hashes'][name]:
            raise ValueError('packaged dependency differs from selected runtime: '+name)
    candidate=output.parent/'submission-fast-road-commit.zip'
    with zipfile.ZipFile(candidate,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items(): z.writestr(name,data)
    archives={'control':CONTROL,'candidate':candidate}
    hashes={arm:digest(p.read_bytes()) for arm,p in archives.items()}
    checks={arm:check_archive(p) for arm,p in archives.items()}
    rows=[]
    with tempfile.TemporaryDirectory() as directory:
        directories={}
        for arm,archive in archives.items():
            folder=Path(directory)/arm;folder.mkdir()
            with zipfile.ZipFile(archive) as z:z.extractall(folder)
            directories[arm]=folder
        cells=[('replay',r['track'],r['seed'],'candidate',r) for r in selected]
        for track in (1,2,3):
            for seed in (38301,38303):
                order=('candidate','control') if (track+seed)%2 else ('control','candidate')
                cells.extend(('additional',track,seed,arm,None) for arm in order)
        for phase,track,seed,arm,reference in cells:
            completed=subprocess.run([sys.executable,'-c',CHILD,str(ROOT),str(track),str(seed)],
                                     cwd=directories[arm],capture_output=True,text=True,timeout=120)
            if completed.returncode:raise RuntimeError(completed.stderr[-3000:])
            row=json.loads(completed.stdout.strip().splitlines()[-1])
            same=None
            if reference is not None:
                data=(REFERENCE/reference['evidence']).read_bytes()
                if digest(data)!=reference['sha256']:raise ValueError('reference episode changed')
                before=json.loads(data)
                same=signature(row)==signature(before) and all(row[k]==before[k] for k in ('completed','lapTimeMs','collisions','damage','retire_reason'))
            row.update(phase=phase,arm=arm,same_source_trace=same)
            path=output.parent/f'{phase}-{track}-{seed}-{arm}.json'
            with path.open('x') as f:json.dump(row,f)
            if (row['error'] or row['invalid_actions'] or same is False or row['import_create_s']>=10 or row['reset_s']>=5
                    or row['act_max_ms']>=5000 or row['peak_rss_bytes']>=1024**3):raise RuntimeError(str(path))
            compact={k:row[k] for k in ('phase','arm','track_id','seed','completed','lapTimeMs','progress','collisions','damage','retire_reason','same_source_trace','import_create_s','reset_s','act_p50_ms','act_p95_ms','act_max_ms','peak_rss_bytes')}
            compact.update(evidence=path.name,sha256=digest(path.read_bytes()));rows.append(compact);print(json.dumps(compact),flush=True)
    for arm,archive in archives.items():
        if digest(archive.read_bytes())!=hashes[arm]:raise ValueError('archive changed')
    summary={}
    for arm in archives:
        additional=[r for r in rows if r['phase']=='additional' and r['arm']==arm]
        first=selected if arm=='candidate' else controls
        unique=first+additional
        summary[arm]=dict(completed=sum(r['completed'] for r in unique),denominator=len(unique),
                          median_finished_ms=statistics.median(r['lapTimeMs'] for r in unique if r['completed']),
                          additional_completed=sum(r['completed'] for r in additional),additional_denominator=len(additional),
                          collisions=sum(r['collisions'] for r in unique),damage=sum(r['damage'] for r in unique))
    key=lambda r:(r.get('track',r.get('track_id')),r['seed'])
    full_control={key(r):r for r in controls+[r for r in rows if r['phase']=='additional' and r['arm']=='control']}
    full_candidate={key(r):r for r in selected+[r for r in rows if r['phase']=='additional' and r['arm']=='candidate']}
    pairs=[dict(track=t,seed=s,control_completed=full_control[t,s]['completed'],candidate_completed=full_candidate[t,s]['completed'],
                control_ms=full_control[t,s]['lapTimeMs'],candidate_ms=full_candidate[t,s]['lapTimeMs']) for t,s in sorted(full_control)]
    for pair in pairs:
        both=pair['control_completed'] and pair['candidate_completed']
        pair['delta_ms']=pair['candidate_ms']-pair['control_ms'] if both else None
        pair['time_ratio']=pair['candidate_ms']/pair['control_ms'] if both else None
    with output.open('x') as f:json.dump(dict(summary=summary,pairs=pairs,rows=rows,hashes=hashes,checks=checks,
        files={n:digest(d) for n,d in files.items()},candidate_archive=candidate.name,
        reference_report_sha256=digest((REFERENCE/'report.json').read_bytes()),
        replay_is_not_independent=True,consumed_train_only=True,official_submission=False),f,indent=2)
    (output.parent/'agent.py').write_text(ENTRY)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
