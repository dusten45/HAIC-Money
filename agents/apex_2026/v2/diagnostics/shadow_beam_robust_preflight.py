from pathlib import Path
import json,time,hashlib,numpy as np
from agents.apex_2026.v2.beam_robust_agent import Agent
src=Path('/tmp/apex-v2-beam-r3');rows=[json.loads(x) for x in (src/'t3-s4111953688.jsonl').read_text().splitlines()];obs=np.load(src/'failure_capture/observations.npz')['observations'];out=[]
for step in list(range(120,139))+[145,154]:
 a=Agent();gas=0.
 for prev in rows[:step-1]:
  for _ in range(4):gas+=min(float(prev['action'][1])-gas,.1)
 a.throttle=gas;a.last_steer=rows[step-2]['action'][0];a.slip=rows[step-2]['policy_diagnostics']['predicted_slip']
 saved={};original=a._search
 def capture(reference,arc,targets,free,initial):
  saved.update(free=free,initial=initial)
  return original(reference,arc,targets,free,initial)
 a._search=capture
 start=time.perf_counter();action=a.act(obs[step-120]);elapsed=time.perf_counter()-start;d=a.last_step_diagnostics();item={'same_first_action':bool(np.array_equal(action,np.array(rows[step-1]['action'],np.float32))),'step':step,'act_s':elapsed,'action':action.tolist(),'cost':d['beam_cost'],'sequence':d['beam_sequence'],'ticks':d['collision_checked_raw_ticks']};a.restore(a.shadow,saved['initial']);unsafe=0
 for action2 in d['beam_sequence']:
  for _ in range(4):
   x,y,h,_,_=a.shadow.step(action2);unsafe+=a._footprint_cost(saved['free'],x,y,h)
 item['selected_sequence_unsafe']=unsafe
 out.append(item);print(json.dumps({k:v for k,v in item.items() if k!='sequence'}),flush=True)
Path('/tmp/apex-v2-beam-r5/preflight.json').write_text(json.dumps({'source_sha256':hashlib.sha256(Path('agents/apex_2026/v2/beam_robust_agent.py').read_bytes()).hexdigest(),'results':out},indent=2))
