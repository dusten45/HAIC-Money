"""Saved required-track pedal diagnostics and isolated, unmodified Car sweeps.

No CarRacing imports/resets. Uniform-road tests are synthetic physics, not laps.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summary(x):
    x=np.asarray(x,dtype=float)
    if not len(x):return {'n':0}
    return dict(n=len(x),mean=float(x.mean()),p10=float(np.quantile(x,.1)),median=float(np.median(x)),p90=float(np.quantile(x,.9)))


def artifact(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def traces():
    out=[];sources=[]
    for lane in ['apex-line-smooth200','apex-line-hud100']:
        for tid,seed in [(1,516237),(2,644062),(3,1007),(4,18800)]:
            p=Path('/tmp',lane,f't{tid}.jsonl' if lane.endswith('smooth200') else f't{tid}-s{seed}.jsonl')
            rp=p.with_suffix('.json');receipt=json.loads(rp.read_text());assert(receipt['track_id'],receipt['seed'])==(tid,seed)
            rows=[json.loads(l) for l in p.read_text().splitlines()];sources.extend([artifact(p),artifact(rp)])
            v=np.array([r['before']['speed_m_s'] for r in rows]);va=np.array([r['after']['speed_m_s'] for r in rows]);dt=np.array([r['after']['sim_time_s']-r['before']['sim_time_s'] for r in rows])
            actions=np.array([r['action'] for r in rows]);gas=actions[:,1];brake=actions[:,2]
            pd=[r['policy_diagnostics'] for r in rows];target=np.array([r['target_speed'] for r in pd]);hud=np.array([r['speed'] for r in pd]);gas_cap=np.array([r.get('gas_cap',1) for r in pd])
            dv=va-v;ax=dv/dt;excess=hud-target
            requested=np.where(excess<=1,np.clip((target-hud)*.1+.25,0,1),0)
            br=brake>0;accelerating=gas>.3;near=np.abs(target-v)<2
            bursts=[];rebound=[];reaccel=[]
            for j in np.flatnonzero(br&~np.r_[False,br[:-1]]):
                k=j
                while k<len(br) and br[k]:k+=1
                bursts.append((k-j)*.08)
                future=target[j:min(j+5,len(br))]
                rebound.append(float(future.max()-target[j]))
                reaccel.append(bool(np.any(gas[j:min(j+4,len(br))]>.3)))
            out.append(dict(lane=lane,track_id=tid,lap_time_ms=receipt['lapTimeMs'],rows=len(rows),
                            actual_speed=summary(v),target_speed=summary(target),gas=summary(gas),brake=summary(brake[br]),
                            brake_fraction=float(br.mean()),coast_fraction=float(((gas<.01)&(brake<.01)).mean()),
                            gas_limited_fraction=float((gas+1e-5<requested).mean()),
                            gas_acceleration=summary(ax[accelerating]),brake_deceleration=summary(-ax[br]),
                            accumulated_speed_reduction_during_brake=float(-dv[br].sum()),
                            brake_while_true_speed_below_target=int(np.sum(br&(v<target))),
                            brake_true_speed_undershoot_after_action=summary(np.maximum(target[br]-va[br],0)),
                            near_target_rows=int(near.sum()),near_target_gas_mean=float(gas[near].mean()),
                            near_target_brake_fraction=float(br[near].mean()),brake_burst_s=summary(bursts),
                            brake_onset_target_rebound_next_320ms=summary(rebound),
                            brake_onset_reaccelerates_within240ms_fraction=float(np.mean(reaccel)),
                            target_jump=summary(abs(np.diff(target)))))
    return out,sources


def physics(speeds=(20.,30.,40.,50.,60.,70.,80.,90.)):
    import Box2D
    from core.vendor.car_dynamics import Car, ENGINE_POWER, WHEEL_MOMENT_OF_INERTIA, FRICTION_LIMIT, SIZE
    class Road:road_friction=1.
    road=Road();out=[];raw=[]
    pedals=[('gas',g) for g in [0,.15,.3,.5,1]]+[('brake',b) for b in [.03,.1,.2,.4,.7]]
    for initial in speeds:
        for steer in [0.,.1,.2]:
            for kind,pedal in pedals:
                world=Box2D.b2World(gravity=(0,0));car=Car(world,0,0,0)
                for body in [car.hull]+car.wheels:body.linearVelocity=(0,initial)
                for w in car.wheels:w.tiles.add(road);w.omega=initial/w.wheel_rad
                mass=float(car.hull.mass+sum(w.mass for w in car.wheels));rad=car.wheels[0].wheel_rad
                # Coast into the commanded turn for .4s before applying pedal.
                for _ in range(20):
                    car.steer(-steer);car.gas(0);car.brake(0);car.step(.02);world.Step(.02,180,60)
                def snap():
                    vel=car.hull.linearVelocity;beta=np.angle(np.exp(1j*(np.arctan2(vel.y,vel.x)-car.hull.angle-np.pi/2)))
                    return dict(speed=float(vel.length),yaw=float(car.hull.angularVelocity),slip_deg=float(beta*180/np.pi))
                start=snap();samples=[]
                for tick in range(16):
                    gas=pedal if kind=='gas' else 0.;brake=pedal if kind=='brake' else 0.
                    car.steer(-steer);car.gas(gas);car.brake(brake)
                    demands=[]
                    for w in car.wheels:
                        f=w.GetWorldVector((0,1));s=w.GetWorldVector((1,0));v=w.linearVelocity
                        vf=np.dot(f,v);vs=np.dot(s,v)
                        omega=w.omega+.02*ENGINE_POWER*w.gas/WHEEL_MOMENT_OF_INERTIA/(abs(w.omega)+5)
                        omega+=-np.sign(omega)*min(abs(omega),15*w.brake)
                        ff=(-vf+omega*w.wheel_rad)*205000*SIZE*SIZE;sf=-vs*205000*SIZE*SIZE
                        demands.append(float(np.hypot(ff,sf)/FRICTION_LIMIT))
                    car.step(.02);world.Step(.02,180,60)
                    s=snap();s.update(time=(tick+1)*.02,force_demand_ratios=demands);samples.append(s)
                out.append(dict(initial_speed=initial,steer=steer,pedal_kind=kind,pedal=pedal,start=start,
                                valid_low_slip_start=abs(start['slip_deg'])<5,
                                acceleration_80ms=(samples[3]['speed']-start['speed'])/.08,
                                acceleration_320ms=(samples[-1]['speed']-start['speed'])/.32,
                                peak_slip_deg=max(abs(s['slip_deg']) for s in samples),end=samples[-1],
                                rear_saturation_fraction=float(np.mean([max(s['force_demand_ratios'][2:])>1 for s in samples])),
                                front_saturation_fraction=float(np.mean([max(s['force_demand_ratios'][:2])>1 for s in samples]))))
                raw.append(dict(initial_speed=initial,steer=steer,pedal_kind=kind,pedal=pedal,start=start,samples=samples))
    constants=dict(total_body_mass=mass,wheel_radius=rad,wheel_moment_of_inertia=WHEEL_MOMENT_OF_INERTIA,
                   effective_straight_acceleration_mass=mass+4*WHEEL_MOMENT_OF_INERTIA/rad**2,
                   rear_force_per_wheel_per_acceleration=mass/2+WHEEL_MOMENT_OF_INERTIA/rad**2,
                   friction_limit_per_wheel=FRICTION_LIMIT,engine_power_per_rear_wheel=ENGINE_POWER,
                   unsaturated_gas_acceleration_expression='2*ENGINE_POWER*gas/((v+5*radius)*(mass+4*I/radius^2))',
                   unsaturated_brake_deceleration_expression='4*I*15*brake/(.02*radius*(mass+4*I/radius^2))',
                   rear_longitudinal_acceleration_limit=FRICTION_LIMIT/(mass/2+WHEEL_MOMENT_OF_INERTIA/rad**2))
    return out,raw,constants


def model_validation(cases,raw,constants):
    """Validate analytic rolling model; never train on the high-slip rows."""
    m=constants['effective_straight_acceleration_mass'];r=constants['wheel_radius'];I=constants['wheel_moment_of_inertia']
    A=2*constants['engine_power_per_rear_wheel']/m
    B=4*I*15/(.02*r*m)
    coast=[c for c in cases if c['pedal_kind']=='gas' and c['pedal']==0 and c['steer']>0 and c['valid_low_slip_start'] and c['start']['speed']**2*abs(np.tan(c['steer']))/3.24<180]
    x=np.array([c['start']['speed']**3*np.tan(c['steer'])**2 for c in coast]);y=np.array([-c['acceleration_80ms'] for c in coast])
    drag=float(np.dot(x,y)/np.dot(x,x))
    predictions=[]
    for c in cases:
        v=c['start']['speed'];g=0.;steer=c['steer'];vs=[]
        for _ in range(16):
            gc=c['pedal'] if c['pedal_kind']=='gas' else 0
            g+=min(gc-g,.1)
            b=c['pedal'] if c['pedal_kind']=='brake' else 0
            a=A*g/(max(v,0)+5*r)-B*b-drag*v**3*np.tan(steer)**2
            v=max(0,v+.02*a);vs.append(v)
        nonsaturated=c['rear_saturation_fraction']==0 and c['front_saturation_fraction']==0 and c['valid_low_slip_start'] and c['peak_slip_deg']<5
        q=c['start']['speed']**2*abs(np.tan(steer))/3.24
        predictions.append(dict(initial_speed=c['initial_speed'],steer=steer,pedal_kind=c['pedal_kind'],pedal=c['pedal'],
                                rolling_unsaturated=nonsaturated,demand_proxy=q,
                                predicted_acceleration_80ms=(vs[3]-c['start']['speed'])/.08,
                                acceleration_error_80ms=(vs[3]-c['start']['speed'])/.08-c['acceleration_80ms'],
                                speed_error_320ms=vs[-1]-c['end']['speed']))
    groups={}
    for name,predicate in [('rolling_unsaturated',lambda p:p['rolling_unsaturated']),
                            ('straight',lambda p:p['rolling_unsaturated'] and p['steer']==0),
                            ('turning',lambda p:p['rolling_unsaturated'] and p['steer']>0),
                            ('excluded_saturated_or_slip',lambda p:not p['rolling_unsaturated'])]:
        subset=[p for p in predictions if predicate(p)]
        groups[name]=dict(n=len(subset),abs_acceleration_error_80ms=summary([abs(p['acceleration_error_80ms']) for p in subset]),
                          abs_speed_error_320ms=summary([abs(p['speed_error_320ms']) for p in subset]))
    per_speed=[]
    for speed in sorted(set(c['initial_speed'] for c in cases)):
        subset=[p for p in predictions if p['initial_speed']==speed and p['rolling_unsaturated']]
        per_speed.append(dict(initial_speed=speed,n=len(subset),abs_acceleration_error_80ms=summary([abs(p['acceleration_error_80ms']) for p in subset]),
                              abs_speed_error_320ms=summary([abs(p['speed_error_320ms']) for p in subset])))
    return dict(gas_coefficient=A,brake_coefficient=B,coast_drag_coefficient=drag,coast_calibration_rows=len(coast),
                expression='a = A*g_internal/(v+2.7) - B*brake - D*v^3*tan(steer)^2; integrate four .02s ticks; g_internal += min(command-g_internal,.1)',
                groups=groups,per_speed=per_speed,predictions=predictions,
                scope='Rolling moving car only20-90m/s. CoastD calibrated to the synthetic coasting subset; A/B derived from exact source constants. Saturated/high-slip rows excluded from applicability, retained for error disclosure.')


def allocator_guidance(constants,validation):
    return dict(status='default-off experimental proposal; no driving validation',
                measured_support='Unchanged uniform-road Car physics, initialized rolling moving wheels20-90m/s, no environment resets.',
                equations=[
                    'Maintain rear internal throttle memory at raw20ms ticks: g += min(g_command-g,0.1). Throttle reductions apply immediately.',
                    'Predict rolling longitudinal acceleration A*g/(v+2.7) - B*b - D*v^3*tan(steer)^2, A/B/D from model_validation. Integrate four ticks for next action speed.',
                    'Convert speed limit into next-action acceleration budget. Near a steady target, use coast rather than mandatory+0.25gas; do not brake if predicted coast speed is already below a bounded target allowance.',
                    'Choose gas or brake separately by monotone bisection on the four-tick predictor so predicted end speed approaches the reachable target without crossing its upper bound. Retain original braking-envelope safety limits.',
                    'Use current pixel speed plus bounded observed acceleration to compensate about40ms HUD lag; avoid unbounded differentiation.',
                    'Rolling rear tire force per wheel≈(m/2+I/r^2)*a_drive. Approximate lateral force per wheel≈m*a_lat/4. Set |a_drive|<=sqrt(max(0,(rho*400)^2-(m*a_lat/4)^2))/(m/2+I/r^2).',
                    'For initial conservative tests rho=0.8 gives straight drive cap35.02m/s² and lateral limit175.30m/s². Also preserve existing traction/slip guard, because equal lateral-force sharing is approximate.',
                    'Bound prediction with a_lat=max(|v*yaw|,v^2*|tan(steer)|/3.24) evaluated at predicted next speed; do not add extra throttle when slip grows or model residual is large.',
                    'Braking uses all four wheels. Keep initial braking command cap0.4 and cap deceleration by conservative residual total-friction circle; stronger brakes were measured but unnecessary for initial controller validation.',
                ],
                cautions=[
                    'Friction-circle formulas are approximate force allocation, not exact yaw stability guarantees.',
                    'Use a deadband tied to measured pixel-speed resolution/uncertainty, not a claim of exact speed tracking.',
                    'Startup near0m/s is outside calibrated moving domain; preserve existing startup behavior or separately validate stopped launch.',
                    'At skid/damage/grass or strong steering transients, rolling model is invalid; coast/brake safe fallback and existing protection must remain.',
                    'No claim that every short brake burst is unnecessary: targets change with real bends/obstacles.',
                ])


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',type=Path,required=True);a.add_argument('--raw',type=Path,required=True);args=a.parse_args()
    if args.output.exists() or args.raw.exists():raise FileExistsError('Choose unused output paths')
    t,sources=traces();p,raw,c=physics();args.raw.parent.mkdir(parents=True,exist_ok=True);args.raw.write_text(json.dumps(raw,indent=2)+'\n')
    validation=model_validation(p,raw,c)
    r=dict(schema='apex-pedal-allocation-v1',script=artifact(Path(__file__)),environment_resets=0,
           input_sources=sources,car_source=artifact(Path('core/vendor/car_dynamics.py')),traces=t,synthetic=p,
           constants=c,synthetic_raw=artifact(args.raw),model_validation=validation,allocator_guidance=allocator_guidance(c,validation),limitations=[
               'Only previously consumed required-track traces; no dev/holdout records used.',
               'Synthetic uniform road, initialized rolling wheels and .4s coast-turn conditioning; no scored episodes.',
               'Linear pedal equations assume rolling, no tire saturation and small steer; not valid during wheelspin/skid.',
               'Brake-onset rebound/undershoot is diagnostic, not proof braking was avoidable at future obstacles.',
               'Trajectories changed between smooth200 and HUD100; comparisons are descriptive, not action-matched causal attribution.'])
    args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print('saved',args.output,'cases',len(p))


if __name__=='__main__':main()
