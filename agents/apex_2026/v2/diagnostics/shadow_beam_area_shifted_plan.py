from pathlib import Path
import math,json,numpy as np
from agents.apex_2026.v2.beam_area_agent import Agent
from agents.apex_2026.v2.shadow_physics import decode_wheel_omega
p=Path('/tmp/apex-v2-beam-r3');rows=[json.loads(x) for x in (p/'t3-s4111953688.jsonl').read_text().splitlines()];obs=np.load(p/'failure_capture/observations.npz')['observations'];pre=json.loads(Path('/tmp/apex-v2-beam-r4/preflight.json').read_text())['results'];sequence=next(r['sequence'] for r in pre if r['step']==137);step=138;stack=obs[18];d=rows[137]['policy_diagnostics'];a=Agent();_,slip,_=a.road._motion(stack[-2],stack[-1]);a.shadow.reset(d['speed']*math.cos(slip),-d['yaw_rate'],float(np.clip(-d['wheel_angle'],-.4,.4)),lateral_speed=d['speed']*math.sin(slip),throttle=0.,omegas=decode_wheel_omega(stack[-1]));initial=a.snapshot(a.shadow);free,_,_=a.road._free_space(stack[-1]);free[59:68,40:45]=1;out=[]
for label,seq in [('shifted_previous_sequence',sequence[1:]),('previous_firstaction_again',[sequence[0]]),('new_selected',next(r['sequence'] for r in pre if r['step']==138))]:
 a.restore(a.shadow,initial);ticks=[]
 for action in seq:
  costs=[]
  for _ in range(4):
   x,y,h,_,_=a.shadow.step(action);costs.append(a._footprint_cost(free,x,y,h))
  ticks.append({'action':action,'unsafe_cells_by_tick':costs})
 out.append({'label':label,'blocks':ticks,'total_unsafe':sum(sum(t['unsafe_cells_by_tick']) for t in ticks)});print(label,ticks[0],out[-1]['total_unsafe'])
Path('/tmp/apex-v2-beam-r4/shifted_plan.json').write_text(json.dumps({'results':out,'note':'No environment reset. Shifted previous R4 safe sequence evaluated against current138 pixel-reconstructed state and mask.'},indent=2))
