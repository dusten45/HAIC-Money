from pathlib import Path
import math,json,numpy as np
from agents.apex_2026.v2.beam_agent import Agent
from agents.apex_2026.v2.shadow_physics import ShadowCar,decode_wheel_omega
p=Path('/tmp/apex-v2-beam-r3');rows=[json.loads(x) for x in (p/'t3-s4111953688.jsonl').read_text().splitlines()];obs=np.load(p/'failure_capture/observations.npz')['observations'];truth=json.loads((p/'failure_capture/truth.json').read_text());step=138;i=step-120;t=truth[i];r=rows[step-1];d=r['policy_diagnostics'];a=Agent({});frame=obs[i,-1];_,slip,_=a.road._motion(obs[i,-2],frame);free,_,_=a.road._free_space(frame);free[59:68,40:45]=1
h=t['before']['heading_rad'];vx,vy=t['velocity'];lat=math.cos(h)*vx+math.sin(h)*vy;fwd=-math.sin(h)*vx+math.cos(h)*vy
pixel={'speed':d['speed'],'yaw':-d['yaw_rate'],'wheel':float(np.clip(-d['wheel_angle'],-.4,.4)),'slip':slip,'omegas':decode_wheel_omega(frame)};actual={'speed':t['before']['speed_m_s'],'yaw':t['before']['angular_velocity'],'wheel':float(np.mean(t['wheel'][:2])),'slip':math.atan2(lat,fwd),'omegas':t['omega']};out=[]
for variant in ['pixel','speed','yaw','wheel','slip','omegas','all_true']:
 state=pixel.copy()
 if variant=='all_true':state.update(actual)
 elif variant!='pixel':state[variant]=actual[variant]
 car=ShadowCar();car.reset(state['speed']*math.cos(state['slip']),state['yaw'],state['wheel'],lateral_speed=state['speed']*math.sin(state['slip']),throttle=0.,omegas=state['omegas']);ticks=[]
 for _ in range(4):
  pred=car.step(r['action']);ticks.append({'pose':list(pred[:3]),'sparse_cost':a._footprint_cost(free,*pred[:3])})
 out.append({'variant':variant,'ticks':ticks});print(variant,[x['sparse_cost'] for x in ticks],ticks[-1]['pose'])
(p/'failure_capture/input_ablation.json').write_text(json.dumps({'scope':'Offline privileged calibration only; no policychange or newreset','pixel_inputs':{k:list(v) if isinstance(v,np.ndarray) else v for k,v in pixel.items()},'true_inputs':actual,'results':out},indent=2))
