"""Read-only R12 kinematic model replay on its four consumed required traces.

No environment is imported or reset. Future executed actions are diagnostic
inputs only; this module is never imported by the pixel policy.
"""
import argparse
from pathlib import Path
import json, math, hashlib
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--trace-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
base=args.trace_dir
if args.output.exists():
 raise FileExistsError(args.output)
source_digest=hashlib.sha256((base/'source.py').read_bytes()).hexdigest()
if source_digest != '86ec69a9a6edab004b309f22256f9d78c13fb73d24f89c85d5c90b6238af19be':
 raise ValueError('This fixed model check requires the archived R12 controller source')
samples=[]
for track in range(1,5):
 rows=[json.loads(x) for x in (base/f't{track}.jsonl').read_text().splitlines()]
 for start in range(len(rows)):
  first=rows[start];d=first['policy_diagnostics'];h=first['before']['heading_rad'];origin=np.array(first['before']['position']);right=np.array([math.cos(h),math.sin(h)]);forward=np.array([-math.sin(h),math.cos(h)])
  v=d['speed'];rate=d['yaw_rate'];wheel=np.clip(d['wheel_angle'],-.4,.4);slip=d['slip'];yaw=0.;xy=np.zeros(2);elapsed=0.
  for j in range(12):
   if start+j>=len(rows):break
   row=rows[start+j]
   if abs(row['after']['sim_time_s']-row['before']['sim_time_s']-.08)>1e-6:break
   steer,gas,brake=row['action']
   for k in range(2):
    wheel+=np.clip(np.clip(steer,-.4,.4)-wheel,-.12,.12)
    v=max(0.,v+(gas*(61-.2*v)-.004*v*v-230*brake)*.04)
    rate+=.4*(np.clip(v*math.tan(wheel)/3.24,-6,6)-rate)
    yaw+=rate*.04;elapsed+=.04;travel=yaw+slip*math.exp(-elapsed/.24)
    xy+=np.array([math.sin(travel),math.cos(travel)])*v*.04
   if j+1 not in (1,2,4,8,12):continue
   delta=np.array(row['after']['position'])-origin;actual=np.array([delta@right,delta@forward])
   samples.append(dict(track=track,step=first['step'],horizon=j+1,action=first['action'],speed=first['before']['speed_m_s'],speed_error=v-row['after']['speed_m_s'],yaw_error=yaw+row['after']['heading_rad']-h,position_error=float(np.linalg.norm(xy-actual)),longitudinal_error=float(xy[1]-actual[1]),lateral_error=float(xy[0]-actual[0]),predicted=xy.tolist(),actual=actual.tolist()))
summary={}
for horizon in (1,2,4,8,12):
 for label,condition in [('moving',lambda r:r['speed']>20),('critical_track4',lambda r:r['track']==4 and 95<=r['step']<=110)]:
  selected=[r for r in samples if r['horizon']==horizon and condition(r)];data={}
  for key in ('speed_error','yaw_error','position_error','longitudinal_error','lateral_error'):
   values=np.array([r[key] for r in selected]);data[key]={'mean':float(values.mean()),'mean_abs':float(np.abs(values).mean()),'p90_abs':float(np.percentile(np.abs(values),90))}
  summary[f'{label}_{horizon}']={'count':len(selected),**data}
coast={}
for label,predicate in [('coast',lambda a:a[1]==0 and a[2]==0),('gas035',lambda a:.34<a[1]<.36 and a[2]==0),('fullgas',lambda a:a[1]>.99 and a[2]==0),('brake035',lambda a:.34<a[2]<.36)]:
 selected=[r for r in samples if r['horizon']==1 and r['speed']>30 and abs(r['action'][0])<.09 and predicate(r['action'])]
 coast[label]={'count':len(selected),'mean_speed_error':float(np.mean([r['speed_error'] for r in selected])) if selected else None}
result={'scope':'Zero resets, open-loop model replay of actual executed action sequences; runtime uses no future actions or telemetry. Partial terminal actions excluded.', 'summary':summary,'pedal_speed_error':coast,'critical_windows':[r for r in samples if r['track']==4 and 95<=r['step']<=110],'source_sha256':hashlib.sha256((base/'source.py').read_bytes()).hexdigest()}
result['provenance']={'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'traces_sha256':{str(base/f't{t}.jsonl'):hashlib.sha256((base/f't{t}.jsonl').read_bytes()).hexdigest() for t in range(1,5)}}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('x') as handle:
 handle.write(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items()if k.endswith('_12')or k.endswith('_1')},indent=2));print(coast)
