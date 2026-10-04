import json,hashlib,os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
import numpy as np
from pathlib import Path
from agents.apex_2026.evaluate import make_environment,diagnostics
src=Path('/tmp/apex-v2-beam-r5');primary=json.loads((src/'t3-s4111953688.json').read_text());assert primary['status']=='completed'
trace=src/'t3-s4111953688.jsonl';rows=[json.loads(x) for x in trace.read_text().splitlines()];out=src/'stall_capture';out.mkdir(exist_ok=True)
env=make_environment();obs,info=env.reset(seed=4111953688,options={'track_id':3});images=[];states=[];parity=[]
for r in rows[:140]:
 step=r['step'];before=diagnostics(env,info);action=np.array(r['action'],np.float32)
 assert np.array_equal(action,np.array(r['action']))
 if 60<=step<=140:images.append(obs.copy())
 obs,_,terminated,truncated,info=env.step(action);after=diagnostics(env,info)
 # The reset info differs from post-warmup initialization in the evaluator at
 # first decision only; preserve actual comparison results rather than hide it.
 parity.append({'step':step,'before_exact':before==r['before'],'after_exact':after==r['after']})
 if 60<=step<=140:states.append({'step':step,'before':before,'after':after,'action':action.tolist()})
np.savez_compressed(out/'observations.npz',observations=np.array(images));(out/'states.json').write_text(json.dumps(states));(out/'parity.json').write_text(json.dumps(parity))
report={'reset_count':1,'steps_replayed':min(140,len(rows)),'capture_decision_steps':[60,140],'original_trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((src/'source.py').read_bytes()).hexdigest(),'exact_before_count':sum(x['before_exact'] for x in parity),'exact_after_count':sum(x['after_exact'] for x in parity),'captured_exact':all(x['before_exact'] and x['after_exact'] for x in parity if x['step']>=60),'scope':'Diagnostic frozenaction replay only, not candidateevaluation; no policyact calls.'}
(out/'receipt.json').write_text(json.dumps(report,indent=2));env.close();print(json.dumps(report),flush=True)
