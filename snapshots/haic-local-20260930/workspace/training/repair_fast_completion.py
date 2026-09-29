"""Replay exact missing-row repair ZIP against all24 frozen selected traces."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import zipfile
from training.confirm_fast_completion import CHILD
from training.confirm_stable_package import check_archive

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/haic-research-v2/fast-completion-exact-fresh-20260929'
TRAIN=ROOT/'artifacts/haic-research-v2/fast-completion-causal-repair-train-20260929'
ENTRY='''from haic_agent.fast_completion_coordination import FastCompletionCoordination
class Agent:
    def __init__(self): self.driver=FastCompletionCoordination("preview_row_repair")
    def reset(self,observation): self.driver.reset(observation)
    def act(self,observation): return self.driver.act(observation)
    def last_step_diagnostics(self): return self.driver.last_step_diagnostics()
'''

def digest(data): return hashlib.sha256(data).hexdigest()

def signature(trace):
    return digest(json.dumps([[t[k] for k in ('steer','gas','brake','car_x','car_y','car_yaw','progress')] for t in trace]).encode())

def run(output):
    reports={p:json.loads(p.read_text()) for p in (OLD/'report.json',TRAIN/'report.json')}
    report_hashes={str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in reports}
    fresh=reports[OLD/'report.json']
    archive=OLD/fresh['candidate_archive']
    if digest(archive.read_bytes())!=fresh['hashes']['candidate']: raise ValueError('Reference ZIP changed')
    references={}
    for p,report in reports.items():
        selected='candidate' if p.parent==OLD else 'preview_action_aware'
        for row in report['rows']:
            if row['arm']==selected:
                track=row.get('track',row.get('track_id'))
                references[track,row['seed']]=(p.parent/row['evidence'],row['sha256'])
    assert len(references)==24
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z: files={n:z.read(n) for n in z.namelist()}
    files['agent.py']=ENTRY.encode()
    files['haic_agent/fast_completion_coordination.py']=(ROOT/'haic_agent/fast_completion_coordination.py').read_bytes()
    candidate=output.parent/'submission-fast-completion.zip'
    with zipfile.ZipFile(candidate,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items(): z.writestr(name,data)
    candidate_hash=digest(candidate.read_bytes());checks=check_archive(candidate)
    cells=[(2,38302),(3,38300)]+[cell for cell in sorted(references) if cell not in ((2,38302),(3,38300))]
    rows=[]
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(candidate) as z:z.extractall(directory)
        for track,seed in cells:
            reference,expected=references[track,seed]
            if digest(reference.read_bytes())!=expected:raise ValueError('Reference episode changed')
            old=json.loads(reference.read_text());before=old['decision_trace']
            child=subprocess.run([sys.executable,'-c',CHILD,str(ROOT),str(track),str(seed)],cwd=directory,
                                 text=True,capture_output=True,timeout=120)
            if child.returncode:raise RuntimeError(child.stderr[-3000:])
            row=json.loads(child.stdout.strip().splitlines()[-1]);trace=row['decision_trace']
            row.update(reference_path=str(reference.relative_to(ROOT)),reference_sha256=expected,
                       reference_completed=old['completed'],same_prefix=signature(trace[:10])==signature(before[:10]),
                       same_full_trace=signature(trace)==signature(before),
                       geometry_repair_count=sum(t['controller']['geometry_repaired'] for t in trace),
                       mean_actual_speed=statistics.mean(t['speed'] for t in trace[10:]),
                       reduced_target_fraction=sum(t['controller']['target_speed']<60 for t in trace[10:])/len(trace[10:]))
            path=output.parent/f'candidate-{track}-{seed}.json'
            with path.open('x') as f:json.dump(row,f)
            if row['error'] or row['invalid_actions'] or not row['same_prefix'] or row['import_create_s']>=10 or row['reset_s']>=5 or row['act_max_ms']>=5000 or row['peak_rss_bytes']>=1024**3:raise RuntimeError(str(path))
            compact={k:row[k] for k in ('track_id','seed','completed','lapTimeMs','progress','collisions','damage','retire_reason','reference_completed','same_prefix','same_full_trace','geometry_repair_count','mean_actual_speed','reduced_target_fraction','import_create_s','reset_s','act_p95_ms','act_max_ms','peak_rss_bytes')}
            compact.update(evidence=path.name,sha256=digest(path.read_bytes()));rows.append(compact);print(json.dumps(compact),flush=True)
    for p in reports:
        if digest(p.read_bytes())!=report_hashes[str(p.relative_to(ROOT))]:raise ValueError('Reference report changed')
    if digest(candidate.read_bytes())!=candidate_hash:raise ValueError('Candidate ZIP changed')
    summary=dict(completed=sum(r['completed'] for r in rows),denominator=24,
                 reference_completed=sum(r['reference_completed'] for r in rows),
                 preserved_finishes=sum(r['reference_completed'] and r['completed'] for r in rows),
                 unchanged_finishes=sum(r['reference_completed'] and r['same_full_trace'] for r in rows),
                 median_finished_ms=statistics.median(r['lapTimeMs'] for r in rows if r['completed']),
                 collisions=sum(r['collisions'] for r in rows))
    with output.open('x') as f:json.dump(dict(summary=summary,rows=rows,reference_reports=report_hashes,
        reference_zip_sha256=fresh['hashes']['candidate'],candidate_sha256=candidate_hash,candidate_archive=candidate.name,
        checks=checks,consumed_train_only=True,official_submission=False),f,indent=2)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
