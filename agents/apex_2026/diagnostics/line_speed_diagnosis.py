"""Reconstruct speed caps from existing required-track logs; no simulator use."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def describe(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    if not len(x):return {'n':0}
    return dict(n=len(x),mean=float(x.mean()),p10=float(np.quantile(x,.1)),median=float(np.median(x)),
                p90=float(np.quantile(x,.9)),max=float(x.max()))


def artifact(p):
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def caps(path,c,radius=3):
    d=63.-path[:,1];k=np.zeros(len(path))
    for i in range(len(path)):
        lo,hi=max(0,i-radius),min(len(path),i+radius+1)
        if hi-lo>=4:
            p=np.polyfit(d[lo:hi]/1.701,path[lo:hi,0]/1.3608,2)
            slope=2*p[0]*d[i]/1.701+p[1]
            k[i]=abs(2*p[0])/(1+slope*slope)**1.5
    speed=np.clip(np.sqrt(c['lateral_accel']/np.maximum(k,.001)),c['min_speed'],c['max_speed'])
    envelope=np.sqrt(speed*speed+2*c['braking_accel']*d/1.701)
    j=int(np.argmin(envelope))
    return float(envelope[j]),j,float(k[j]),k


def road_data(g):
    xy=np.array(g['track'])[:,2:4]
    ds=np.roll(xy,-1,axis=0)-xy;l=np.linalg.norm(ds,axis=1)
    angles=np.arctan2(ds[:,1],ds[:,0]);delta=np.angle(np.exp(1j*(angles-np.roll(angles,1))))
    k=np.abs(delta)/((l+np.roll(l,1))/2)
    return xy,ds,l,k


def nearest(pos,xy,ds):
    rel=pos-xy;a=np.clip(np.sum(rel*ds,axis=1)/np.sum(ds*ds,axis=1),0,1)
    distance=np.linalg.norm(rel-a[:,None]*ds,axis=1)
    return int(distance.argmin()),float(distance.min())


def analyze():
    records=[];receipts=[];sources=[];pairs={1:516237,2:644062,3:1007,4:18800}
    for tid,seed in pairs.items():
        folder=Path('/tmp/apex-line-fastlat100');p=folder/f't{tid}.jsonl';rp=p.with_suffix('.json')
        result=json.loads(rp.read_text());assert(result['track_id'],result['seed'])==(tid,seed)
        gpath=Path('/tmp/apex-feasibility')/f'geometry-t{tid}.json';g=json.loads(gpath.read_text())
        assert(g['track_id'],g['seed'])==(tid,seed)
        xy,ds,lengths,roadk=road_data(g)
        c=dict(min_speed=24.,max_speed=72.,lateral_accel=48.,braking_accel=65.,brake_gain=.025)
        c.update(result['config']['config']);sources.extend([artifact(p),artifact(rp),artifact(gpath)])
        rows=[json.loads(line) for line in p.read_text().splitlines()];prevspeed=0.;local=[]
        for row in rows:
            pd=row.get('policy_diagnostics',{});b=row['before'];v=b['speed_m_s'];accel=(v-prevspeed)/.08;prevspeed=v
            if not pd.get('valid'):continue
            path=np.asarray(pd['path'],dtype=np.float32);d=63.-path[:,1]
            pathcap,j,k,kappas=caps(path,c);widecap,_,_,_=caps(path,c,5);widercap,_,_,_=caps(path,c,7)
            corr=np.sqrt(c['lateral_accel']/max(abs(pd['curvature']),.001));horizon=32. if d[-1]<18 else float('inf')
            choices=dict(path=pathcap,correction=corr,horizon=horizon,max_speed=c['max_speed'])
            winner=min(choices,key=choices.get);target=max(c['min_speed'],choices[winner]);assert abs(target-pd['target_speed'])<1e-4,(tid,row['step'],target,pd['target_speed'])
            segment,deviation=nearest(np.array(b['position']),xy,ds)
            # Ground-road curvature in visible forward arc is diagnostic only.
            upcoming=[];distance=0.;idx=segment
            while distance<=d[-1]/1.701:
                upcoming.append(roadk[idx]);distance+=lengths[idx];idx=(idx+1)%len(xy)
            speed=pd['speed'];basegas=np.clip((target-speed)*.10+.25,0,1) if speed-target<=1 else 0
            excess=speed-target;gas,brake=row['action'][1:]
            r=dict(track_id=tid,step=row['step'],actual=v,hud=speed,acceleration=accel,target=target,
                   winner=winner,pathcap=pathcap,correction_cap=float(corr),horizon=float(d[-1]/1.701),
                   binding_index=j,binding_distance=float(d[j]/1.701),binding_kappa=k,
                   path_points=len(path),first_seven_x=path[:7,0].tolist(),
                   wider11_target=max(c['min_speed'],min(widecap,corr,horizon,c['max_speed'])),
                   wider15_target=max(c['min_speed'],min(widercap,corr,horizon,c['max_speed'])),
                   road_kappa=float(roadk[segment]),visible_road_kappa_max=float(max(upcoming)),
                   center_deviation=deviation,obstacle_pixels=pd['obstacle_pixels'],
                   speed_gap=target-v,hud_error=speed-v,brake=brake,gas=gas,gas_cap=pd.get('gas_cap',1),
                   gas_limited=bool(gas+1e-5<basegas),motion_valid=pd.get('motion_valid',False))
            records.append(r);local.append(r)
        bursts=[];count=0
        for z in local+[{'target':0}]:
            if z['target']>=80:count+=1
            elif count:bursts.append(count*.08);count=0
        receipts.append(dict(track_id=tid,lap_time_ms=result['lapTimeMs'],finished=result['finished'],damage=result['damage'],
                             steps=len(rows),valid_rows=len(local),summary=summarize(local),
                             target_over80_burst_duration_s=describe(bursts),
                             target_step_change=describe(np.abs(np.diff([x['target'] for x in local])))))
    hud=np.array([x['hud'] for x in records]);v=np.array([x['actual'] for x in records]);acc=np.array([x['acceleration'] for x in records])
    fit=np.linalg.lstsq(np.column_stack([v,acc,np.ones(len(v))]),hud,rcond=None)[0]
    straight=[x for x in records if x['visible_road_kappa_max']<.008 and x['center_deviation']<2 and x['obstacle_pixels']==0]
    quantization=[]
    for slope in [.0,.05,.1,.2,.3,.4,.5]:
        p=np.column_stack([42+np.rint(slope*np.arange(26)),np.arange(58,7,-2)])
        c=dict(min_speed=24.,max_speed=100.,lateral_accel=100.,braking_accel=65.)
        a,j,k,_=caps(p,c);b,_,_,_=caps(p,c,5);z,_,_,_=caps(p,c,7)
        quantization.append(dict(true_curvature=0,lateral_pixels_per_row=slope,cap7=min(a,100),cap11=min(b,100),cap15=min(z,100),
                                 binding_index=j,artificial_kappa=k))
    return dict(schema='apex-line-speed-diagnosis-v1',environment_resets=0,source=artifact(Path('/tmp/apex-line-fastlat100/source.py')),
                inputs=sources,tracks=receipts,all_rows=summarize(records),clear_straight_rows=summarize(straight),
                clear_straight_moving_rows=summarize([x for x in straight if x['actual']>20]),
                hud_fit=dict(expression='HUD = a*actual_speed + b*previous_interval_acceleration + intercept',coefficients=fit.tolist(),
                             implied_lag_s=float(-fit[1]/fit[0])),synthetic_straight_quantization=quantization,
                straight_low_target_examples=sorted(straight,key=lambda x:x['target'])[:12],
                path_limited_examples=sorted([x for x in records if x['winner']=='path'],key=lambda x:x['target'])[:12],
                findings={
                    'primary':'Path curvature caps bind66.8%, pursuit correction11.7%, max10021.5%; short-horizon32 never binds in1200valid rows.',
                    'quantization':'Zero-curvature straight quantized to integer pixel x can produce false0.237/m curvature, limiting target30.95m/s. Actual no-obstacle straight-road examples show identical false spike.',
                    'speed_gap':'Mean actual44.53m/s vs target59.18. Clearstraight subset mean42.13 vs target79.63; gas limited82.5% there. Finite acceleration distance and target changes also contribute; this does not causally assign entire speed gap to throttle.',
                    'hud':'Fitted delay39.6ms, about half an80ms action; speed bias+0.97m/s overall. Lag makes braking persist during deceleration.',
                    'recommendations_hypotheses':[
                        'Use a continuous clearance-constrained speed-reference path or quantization-aware fit, especially near first/end nodes; preserve raw collision checks and sharp physical bends.',
                        'Do not simply increase lateral_accel to defeat false curvature; it also raises real-turn and tracking-correction limits.',
                        'Widening local fit7->11/15 raises mean target only1.79/3.40m/s and can lower some caps or erase real turns; treat as limited ablation, not complete fix.',
                        'Review throttle cap near low predicted lateral demand: it commonly reaches0.5 long before physical demand approaches180. Use predicted near-future demand and slip to avoid power-oversteer while accelerating on straights.',
                        'Compensate observed speed lag with a bounded recent pixel-speed trend or use latest-frame HUD where independently calibrated; avoid differentiated noise.',
                        'Keep pursuit correction speed cap as distinct tracking guard; improve lag/heading tracking before relaxing it.',
                        'No evidence to change the short-horizon32 rule from these laps, since it never binds.'
                    ]},
                limitations=['Read-only posthoc cap recomputation; no counterfactual driving.',
                             'Ground centerline curvature is not the planned obstacle-avoidance or recovery path curvature.',
                             'Clear straight subset requires upcoming visible centerline curvature<0.008/m, center distance<2m, and zero detected obstacle pixels.',
                             'Wider fits can erase real sharp bends/obstacle maneuvers; target increases alone do not establish safety.',
                             'All four required tracks were previously consumed. No dev/holdout traces used.'])


def summarize(r):
    if not r:return {'n':0}
    n=len(r);counts=Counter(x['winner'] for x in r)
    return dict(n=n,cap_winner_counts=dict(counts),cap_winner_fractions={k:v/n for k,v in counts.items()},
                actual_speed=describe([x['actual'] for x in r]),target_speed=describe([x['target'] for x in r]),
                target_minus_actual=describe([x['speed_gap'] for x in r]),hud_minus_actual=describe([x['hud_error'] for x in r]),
                path_binding_distance=describe([x['binding_distance'] for x in r if x['winner']=='path']),
                path_binding_index_counts=dict(Counter(x['binding_index'] for x in r if x['winner']=='path')),
                wider11_target_increase=describe([x['wider11_target']-x['target'] for x in r]),
                wider15_target_increase=describe([x['wider15_target']-x['target'] for x in r]),
                gas_limited_fraction=sum(x['gas_limited'] for x in r)/n,
                gas_cap=describe([x['gas_cap'] for x in r]),
                brake_while_actual_below_target_fraction=sum(x['brake']>0 and x['speed_gap']>0 for x in r)/n,
                brake_fraction=sum(x['brake']>0 for x in r)/n,
                target_below_actual_1_fraction=sum(x['speed_gap']< -1 for x in r)/n,
                actual_below_target_5_fraction=sum(x['speed_gap']>5 for x in r)/n,
                motion_valid_fraction=sum(x['motion_valid'] for x in r)/n,
                near_path_spike_fraction=sum(x['winner']=='path' and x['binding_index']<4 for x in r)/n)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=analyze();r['script']=artifact(Path(__file__));a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(r['all_rows'],indent=2));print('HUD',r['hud_fit']);print('STRAIGHT',r['clear_straight_rows'])


if __name__=='__main__':main()
