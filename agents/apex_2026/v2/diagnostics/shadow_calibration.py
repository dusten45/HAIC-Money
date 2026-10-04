"""Synthetic public-Car fidelity/latency audit; never imported by runtime agent."""
from pathlib import Path
import json
import time
import hashlib
import numpy as np
import Box2D
from core.vendor.car_dynamics import Car
from agents.apex_2026.v2.shadow_physics import ShadowCar, decode_wheel_omega
from agents.apex_2026.v2.diagnostics.shadow_rpm import render
from agents.apex_2026.candidate.agent import Agent as PixelAgent


class Tile:
    road_friction = 1.


def calibrate():
    world = Box2D.b2World(gravity=(0,0))
    car = Car(world,0,0,0)
    for wheel in car.wheels:
        wheel.tiles.add(Tile())
    actions = ([(0,1,0)]*100 + [(.3,.6,0)]*70 + [(-.3,.6,0)]*80 +
               [(.6,1,0)]*80 + [(0,0,.6)]*50 + [(-.5,1,0)]*90 +
               [(0,1,0)]*70 + [(.25,0,.2)]*50 + [(0,0,1)]*40)
    states=[]
    for a in actions:
        states.append(dict(position=np.array(car.hull.position),angle=car.hull.angle,
                           velocity=np.array(car.hull.linearVelocity),yaw=car.hull.angularVelocity,
                           wheel=car.wheels[0].joint.angle,throttle=car.wheels[2].gas,
                           omegas=[w.omega for w in car.wheels]))
        car.steer(-a[0]);car.gas(a[1]);car.brake(a[2]);car.step(.02);world.Step(.02,180,60)
    results={}
    for mode in ('exact_observable','rolling_rpm','zero_slip_rolling_rpm','pixel_hud_known_slip','oracle_speed_bicycle'):
        rows=[]; latencies=[]
        model=ShadowCar()
        for i in range(25,len(states)-41,5):
            s=states[i]; c=np.cos(s['angle']); sn=np.sin(s['angle'])
            inv=np.array([[c,sn],[-sn,c]])
            vx,vy=inv@s['velocity']
            if mode=='zero_slip_rolling_rpm':
                vx,vy=0.,np.linalg.norm(s['velocity'])
            yaw,wheel=s['yaw'],s['wheel']
            omegas=s['omegas'] if mode=='exact_observable' else None
            if mode=='pixel_hud_known_slip':
                frame=render(s['omegas'],np.linalg.norm(s['velocity']),s['yaw'],s['wheel'])
                yaw,wheel=PixelAgent._hud_dynamics(frame);yaw,wheel=-yaw,-wheel
                speed=float(np.clip((frame[77:83,10:13].sum()-.27)/.085,0,100))
                ratio=speed/max(np.linalg.norm(s['velocity']),.001)
                vx,vy=vx*ratio,vy*ratio
                omegas=decode_wheel_omega(frame)
            model.reset(vy,yaw,wheel,lateral_speed=vx,throttle=s['throttle'],omegas=omegas)
            bicycle=np.array([0.,0.,0.,s['yaw'],s['wheel'],np.arctan2(vx,max(vy,.001))])
            begin=time.perf_counter()
            for h in range(1,41):
                pred=model.step(actions[i+h-1])
                if mode=='oracle_speed_bicycle':
                    x,y,heading,rate,wheel,beta=bicycle
                    wheel+=np.clip(-actions[i+h-1][0]-wheel,-.06,.06)
                    wheel=float(np.clip(wheel,-.4,.4))
                    v=float(np.linalg.norm(states[i+h]['velocity']))
                    rate+=.2*(np.clip(v*np.tan(wheel)/3.24,-180/max(v,1),180/max(v,1))-rate)
                    heading+=.02*rate
                    beta+=.2*(-np.arctan(.5*np.tan(wheel))-beta)
                    x+=.02*v*np.sin(beta-heading);y+=.02*v*np.cos(beta-heading)
                    bicycle=np.array([x,y,heading,rate,wheel,beta])
                    pred=(x,y,heading,v,rate)
                if h in (4,10,20,40):
                    target=states[i+h]
                    position=inv@(target['position']-s['position'])
                    rows.append(dict(sample=i,horizon_s=h*.02,practical_state=bool(10<vy<75 and abs(vx)<.6*vy and abs(s['yaw'])<3 and min(s['omegas'])>=0 and max(s['omegas'])<350),position_error_m=float(np.linalg.norm(np.array(pred[:2])-position)),
                                    heading_error_rad=float(abs(pred[2]-(target['angle']-s['angle']))),
                                    speed_error_mps=float(abs(pred[3]-np.linalg.norm(target['velocity'])))))
            latencies.append(time.perf_counter()-begin)
        summary={}
        for h in (.08,.2,.4,.8):
            subset=[r for r in rows if r['horizon_s']==h]
            summary[str(h)]={k:dict(mean=float(np.mean([r[k] for r in subset])),p95=float(np.quantile([r[k] for r in subset],.95)),max=float(max(r[k] for r in subset))) for k in ('position_error_m','heading_error_rad','speed_error_mps')}
        practical={}
        for h in (.08,.2,.4,.8):
            sub=[r for r in rows if r['horizon_s']==h and r['practical_state']]
            practical[str(h)] = dict(count=len(sub),position_mae=float(np.mean([r['position_error_m'] for r in sub])),position_p95=float(np.quantile([r['position_error_m'] for r in sub],.95)))
        results[mode]=dict(practical=practical,sample_count=len(latencies),summary=summary,latency_40steps_s=dict(mean=float(np.mean(latencies)),max=float(max(latencies))),rows=rows)
    source=Path('agents/apex_2026/v2/shadow_physics.py')
    return dict(scope='Synthetic public-Car trajectories only; no geometry resets and no closed-loop performance claim.',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),results=results)


if __name__=='__main__':
    out=calibrate()
    Path('agents/apex_2026/v2/results/shadow-calibration.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:{'summary':v['summary'],'latency':v['latency_40steps_s']} for k,v in out['results'].items()},indent=2))
