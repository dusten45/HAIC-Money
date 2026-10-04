"""No-reset open-loop residual audit on previously consumed pixel/velocity capture."""
from pathlib import Path
import json
import hashlib
import numpy as np
from agents.apex_2026.candidate.agent import Agent
from agents.apex_2026.v2.shadow_physics import ShadowCar,decode_wheel_omega


def main():
    directory=Path('/tmp/apex-v2-motion-registration')
    truth=json.loads((directory/'truth.json').read_text())
    traces=[json.loads(l) for l in (directory/'trace.jsonl').read_text().splitlines()]
    obs=np.load(directory/'observations.npz')['observations']
    actions=[r['action'] for r in traces]
    throttles=[0.]
    for a in actions:
        gas=throttles[-1]
        for _ in range(4):gas+=min(a[1]-gas,.1)
        throttles.append(gas)
    model=ShadowCar();agent=Agent();results={}
    for mode in ('pixel_hud_true_slip','pixel_hud_lk_slip','pixel_hud_zero_slip'):
        rows=[]
        for i in range(1,min(len(obs),len(truth),len(actions))-5):
            s=truth[i];c,sn=np.cos(s['angle']),np.sin(s['angle']);inv=np.array([[c,sn],[-sn,c]])
            vel=inv@np.array(s['velocity']);frame=obs[i,-1]
            speed=float(np.clip((frame[77:83,10:13].sum()-.27)/.085,0,100))
            slip=np.arctan2(vel[0],vel[1])
            if mode=='pixel_hud_lk_slip':
                _,slip,valid=agent._motion(obs[i,-2],frame)
                if not valid:slip=0.
            if mode=='pixel_hud_zero_slip':slip=0.
            yaw,wheel=agent._hud_dynamics(frame)
            model.reset(speed*np.cos(slip),-yaw,-wheel,lateral_speed=speed*np.sin(slip),throttle=throttles[i],omegas=decode_wheel_omega(frame))
            for h in range(1,6):
                pred=model.step(actions[i+h-1],4)
                if h in (1,2,5):
                    target=truth[i+h]
                    local=inv@(np.array(target['position'])-np.array(s['position']))
                    rows.append(dict(index=i,horizon_s=h*.08,position_error_m=float(np.linalg.norm(np.array(pred[:2])-local)),yaw_error_rad=float(abs(pred[2]-(target['angle']-s['angle'])))))
        summary={}
        for h in (.08,.16,.4):
            sub=[r for r in rows if r['horizon_s']==h]
            summary[str(h)]={k:dict(mean=float(np.mean([r[k] for r in sub])),p95=float(np.quantile([r[k] for r in sub],.95))) for k in ('position_error_m','yaw_error_rad')}
        results[mode]=dict(sample_count=len(rows)//3,summary=summary,rows=rows)
    out=dict(source_sha256=hashlib.sha256(Path('agents/apex_2026/v2/shadow_physics.py').read_bytes()).hexdigest(),capture_artifacts={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (directory/'observations.npz',directory/'truth.json',directory/'trace.jsonl')},scope='Previously consumed required1 pixel+truth capture; no new reset. true_slip is calibration upper bound only; never runtime input.',results=results)
    Path('agents/apex_2026/v2/results/shadow-trace.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v['summary'] for k,v in results.items()},indent=2))

if __name__=='__main__':main()
