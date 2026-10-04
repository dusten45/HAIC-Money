from pathlib import Path
import math,json,numpy as np
from agents.apex_2026.v2.beam_agent import Agent
from agents.apex_2026.v2.shadow_physics import decode_wheel_omega
p=Path('/tmp/apex-v2-beam-r3');rows=[json.loads(x) for x in (p/'t3-s4111953688.jsonl').read_text().splitlines()];obs=np.load(p/'failure_capture/observations.npz')['observations'];out=[]
for step in (137,138):
 r=rows[step-1];a=Agent({});gas=0.
 for prev in rows[:step-1]:
  for _ in range(4):gas+=min(float(prev['action'][1])-gas,.1)
 a.throttle=gas;a.last_steer=rows[step-2]['action'][0];a.slip=rows[step-2]['policy_diagnostics']['predicted_slip'];stack=obs[step-120];act=a.act(stack);d=a.last_step_diagnostics();same=np.array_equal(act,np.array(r['action'],np.float32));costerr=d['beam_cost']-r['policy_diagnostics']['beam_cost'];print(step,same,costerr,gas)
 frame=stack[-1];_,slip,valid=a.road._motion(stack[-2],frame)
 if not valid:slip=rows[step-2]['policy_diagnostics']['predicted_slip']
 slip=float(np.clip(slip,-.7,.7));a.shadow.reset(d['speed']*math.cos(slip),-d['yaw_rate'],float(np.clip(-d['wheel_angle'],-.4,.4)),lateral_speed=d['speed']*math.sin(slip),throttle=gas,omegas=decode_wheel_omega(frame));free,obstacle,_=a.road._free_space(frame);free[59:68,40:45]=1;ticks=[]
 for i in range(4):
  state=a.shadow.step(act);ticks.append({'raw_tick':i+1,'pose':list(state[:3]),'speed':state[3],'yaw':state[4],'footprint_cost':a._footprint_cost(free,*state[:3]),'bodies':[{ 'position':list(b.position),'angle':float(b.angle),'polygons':[[list(v) for v in fixture.shape.vertices] for fixture in b.fixtures]} for b in [a.shadow.hull]+a.shadow.wheels]})
 h=r['before']['heading_rad'];delta=np.array(r['after']['position'])-r['before']['position'];actual=[math.cos(h)*delta[0]+math.sin(h)*delta[1],-math.sin(h)*delta[0]+math.cos(h)*delta[1],r['after']['heading_rad']-h];out.append({'step':step,'action_exact':bool(same),'beam_cost_delta':costerr,'pixel_only_throttle_from_actions':gas,'slip':slip,'ticks':ticks,'actual_final_pose':actual,'actual_final_footprint_cost':a._footprint_cost(free,*actual)})
(p/'failure_capture/selected_ticks.json').write_text(json.dumps(out,indent=2))
