"""Read-only saved required-track analysis and isolated Car physics identification.

Never imports CarRacing or resets an environment. Diagnostic geometry/telemetry
must not become runtime policy inputs. Synthetic bodies use uniform friction,
initial wheel velocities and wheel spin; these are not scored driving episodes.
Run from repository root with python -m agents.apex_2026.diagnostics.control_identification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


LANES = ["apex-release-r0", "apex-line-gain35", "apex-line-lookahead26",
         "apex-line-arcguard", "apex-line-lat100"]
PAIRS = {1: 516237, 2: 644062, 3: 1007, 4: 18800}


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def summary(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    return dict(n=len(a), median=float(np.median(a)), p90=float(np.quantile(a, .9)),
                p95=float(np.quantile(a, .95)), max=float(np.max(a)))


def provenance(p):
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def projection(pos, xy):
    d = np.roll(xy, -1, axis=0) - xy
    rel = pos[:, None] - xy[None]
    alpha = np.clip(np.sum(rel*d[None], axis=2)/np.sum(d*d, axis=1), 0, 1)
    residual = rel-alpha[:, :, None]*d[None]
    norm = np.linalg.norm(residual, axis=2)
    idx = norm.argmin(axis=1)
    dev = norm[np.arange(len(pos)), idx]
    chosen = d[idx]
    road_heading = np.arctan2(chosen[:, 1], chosen[:, 0])
    signed = np.sum(residual[np.arange(len(pos)),idx]*np.column_stack([chosen[:,1],-chosen[:,0]]),axis=1)/np.linalg.norm(chosen,axis=1)
    return dev, idx, signed, road_heading


def trace_analysis(base, geometry):
    receipts, records, sources = [], [], []
    for lane in LANES:
        for tid, seed in PAIRS.items():
            path = base/lane/f"t{tid}.jsonl"
            result_path = path.with_suffix(".json")
            result = json.loads(result_path.read_text())
            assert (result['track_id'], result['seed']) == (tid, seed)
            sources.extend([provenance(path), provenance(result_path)])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            b = [r['before'] for r in rows]; a = [r['after'] for r in rows]
            pos = np.array([r['position'] for r in b])
            nextpos = np.array([r['position'] for r in a])
            dt = np.array([y['sim_time_s']-x['sim_time_s'] for x,y in zip(b,a)])
            speed = np.array([r['speed_m_s'] for r in b])
            omega = np.array([r['angular_velocity'] for r in b])
            nextomega = np.array([r['angular_velocity'] for r in a])
            heading = np.array([r['heading_rad'] for r in b])
            deltaheading = wrap(np.array([r['heading_rad'] for r in a])-heading)
            displacement = nextpos-pos
            course = np.arctan2(displacement[:, 1], displacement[:, 0])
            slip = wrap(course-heading-deltaheading/2-np.pi/2)
            course_rate = np.r_[0, wrap(np.diff(course))/dt[1:]]
            action = np.array([r['action'] for r in rows])
            steer = action[:, 0]; prevsteer = np.r_[0, steer[:-1]]
            damage = np.array([r.get('info',{}).get('damage',0) for r in b])
            dev, nearest, signed_dev, road_heading = projection(pos, geometry[tid])
            heading_error = wrap(heading+np.pi/2-road_heading)
            valid = np.array([r.get('policy_diagnostics',{}).get('valid',False) for r in rows])
            curvature = np.array([r.get('policy_diagnostics',{}).get('curvature',np.nan) for r in rows])
            perceived_speed = np.array([r.get('policy_diagnostics',{}).get('speed',np.nan) for r in rows])
            clean = (speed>10)&(dev<3)&(damage==0)&(np.abs(slip)<.12)&(dt>.079)
            stable = clean & (np.abs(steer-prevsteer)<.015)&(np.abs(steer)>.03)&(np.abs(omega)>.03)
            off = np.flatnonzero((dev>6.667)&(speed>15))
            first = int(off[0]) if len(off) else None
            event = []
            if first is not None:
                for j in range(max(0,first-12), min(len(rows),first+3)):
                    event.append(dict(step=j+1,speed=float(speed[j]),distance_from_center=float(dev[j]),
                                      signed_center_deviation=float(signed_dev[j]),road_heading_error_deg=float(heading_error[j]*180/np.pi),
                                      slip_deg=float(slip[j]*180/np.pi),steer=float(steer[j]),gas=float(action[j,1]),brake=float(action[j,2]),
                                      yaw_rate=float(omega[j]),yaw_accel_proxy=float(abs(speed[j]*omega[j])),
                                      commanded_path_curvature=float(curvature[j]) if np.isfinite(curvature[j]) else None,
                                      road_progress=b[j].get('info',{}).get('progress',0),
                                      perception_valid=bool(valid[j])))
            receipt=dict(lane=lane,track_id=tid,seed=seed,steps=len(rows),finished=result['finished'],
                         lap_time_ms=result.get('lapTimeMs'),damage=result['damage'],progress=result['progress'],
                         speed=summary(speed),absolute_slip_deg=summary(np.abs(slip[speed>10])*180/np.pi),
                         center_deviation=summary(dev),clean_rows=int(clean.sum()),
                         clean_yaw_acceleration=summary(np.abs(speed[clean]*omega[clean])),
                         clean_course_acceleration=summary(np.abs(speed[clean]*course_rate[clean])),
                         steady_effective_steer_per_curvature=summary(np.abs(steer[stable]*speed[stable]/omega[stable])),
                         speed_estimation_error=summary(np.abs(perceived_speed-speed)),
                         first_center_offroad_step=first+1 if first is not None else None,
                         first_offroad_window=event)
            receipts.append(receipt)
            for j in np.flatnonzero(clean):
                records.append(dict(lane=lane,track_id=tid,v=speed[j],u=steer[j],up=prevsteer[j],
                                    yaw=omega[j],next_yaw=nextomega[j],slip=slip[j],du=steer[j]-prevsteer[j],gas=action[j,1],brake=action[j,2]))
    return receipts, records, sources


def fit_models(records):
    v=np.array([r['v'] for r in records]);u=np.array([r['u'] for r in records]);up=np.array([r['up'] for r in records])
    y=np.array([r['next_yaw'] for r in records]);w=np.array([r['yaw'] for r in records])
    tid=np.array([r['track_id'] for r in records])
    gas=np.array([r['gas'] for r in records]);brake=np.array([r['brake'] for r in records])
    features={'current_command':(-v*u)[:,None],
              'command_and_previous':np.column_stack([-v*u,-v*up]),
              'autoregressive':np.column_stack([w,-v*u,-v*up]),
              'speed_dependent_autoregressive':np.column_stack([w,-v*u,-v*up,-v*u*(v/60)**2]),
              'pedal_autoregressive':np.column_stack([w,-v*u,-v*up,-v*u*gas,w*brake])}
    result={}
    for name,x in features.items():
        coef=np.linalg.lstsq(x,y,rcond=None)[0]
        cv=[]
        for t in range(1,5):
            train=tid!=t;test=tid==t
            c=np.linalg.lstsq(x[train],y[train],rcond=None)[0]
            cv.append(dict(excluded_required_track=t,n=int(test.sum()),rmse=float(np.sqrt(np.mean((x[test]@c-y[test])**2)))))
        result[name]=dict(coefficients=coef.tolist(),rmse=float(np.sqrt(np.mean((x@coef-y)**2))),
                          absolute_error=summary(np.abs(x@coef-y)),required_track_cross_validation=cv)
    bins=[]
    for lo,hi in [(10,30),(30,45),(45,60),(60,75),(75,100)]:
        m=(v>=lo)&(v<hi)&(np.abs(u-up)<.015)&(np.abs(u)>.03)&(np.abs(w)>.03)
        bins.append(dict(speed_range=[lo,hi],effective_steer_per_curvature=summary(np.abs(u[m]*v[m]/w[m]))))
    return dict(clean_samples=len(records),models=result,steady_gain_by_speed=bins,
                feature_order={'current_command':['-v*u'],'command_and_previous':['-v*u','-v*u_previous'],
                               'autoregressive':['yaw_previous','-v*u','-v*u_previous'],
                               'speed_dependent_autoregressive':['yaw_previous','-v*u','-v*u_previous','-v*u*(v/60)^2'],
                               'pedal_autoregressive':['yaw_previous','-v*u','-v*u_previous','-v*u*gas','yaw_previous*brake']})


def synthetic(out):
    import Box2D
    from core.vendor.car_dynamics import Car, FRICTION_LIMIT, SIZE
    class Road:
        road_friction=1.
    road=Road();raw=[];table=[]
    for speed in [40.,60.,80.,100.]:
        for steer in [.025,.05,.1,.15,.2,.3,.4,.6]:
            for gas in [0.,.15,.3,.5,1.]:
                world=Box2D.b2World(gravity=(0,0));car=Car(world,0,0,0)
                mass=float(car.hull.mass+sum(w.mass for w in car.wheels))
                for body in [car.hull]+car.wheels:
                    body.linearVelocity=(0,speed)
                for wheel in car.wheels:
                    wheel.tiles.add(road);wheel.omega=speed/wheel.wheel_rad
                rows=[]
                for k in range(60):
                    command=0 if k<10 else steer
                    car.steer(-command);car.gas(0 if k<10 else gas);car.brake(0)
                    # Reproduce the pre-cap force magnitude from official Car.step,
                    # including wheel angular acceleration before friction force.
                    demands=[]
                    for wheel in car.wheels:
                        f=wheel.GetWorldVector((0,1));side=wheel.GetWorldVector((1,0));vel=wheel.linearVelocity
                        vf=f[0]*vel[0]+f[1]*vel[1];vs=side[0]*vel[0]+side[1]*vel[1]
                        om=wheel.omega+.02*40000*wheel.gas/1.6/(abs(wheel.omega)+5)
                        ff=(-vf+om*wheel.wheel_rad)*205000*SIZE*SIZE
                        sf=-vs*205000*SIZE*SIZE
                        demands.append(math.hypot(ff,sf)/FRICTION_LIMIT)
                    car.step(.02);world.Step(.02,180,60)
                    vel=car.hull.linearVelocity;v=float(vel.length)
                    beta=float(wrap(math.atan2(vel.y,vel.x)-car.hull.angle-math.pi/2))
                    rows.append(dict(time_since_step=round((k-9)*.02,3),speed=v,yaw=float(car.hull.angularVelocity),
                                     slip_deg=beta*180/math.pi,front_wheel_angle=float(car.wheels[0].joint.angle),
                                     saturated_wheels=sum(d>1 for d in demands),rear_saturated_wheels=sum(d>1 for d in demands[2:]),
                                     front_saturated_wheels=sum(d>1 for d in demands[:2]),max_force_demand_ratio=max(demands),
                                     vx=float(vel.x),vy=float(vel.y)))
                step=rows[10:]
                snap={str(t):step[round(t/.02)-1] for t in [.08,.16,.32,.64,1.]}
                yaw=np.array([r['yaw'] for r in step]);speeds=np.array([r['speed'] for r in step])
                slip=np.array([r['slip_deg'] for r in step]);angles=np.unwrap(np.arctan2([r['vy'] for r in step],[r['vx'] for r in step]))
                course_accel=speeds[1:]*np.diff(angles)/.02
                target=abs(float(np.median(yaw[-10:])))*.9
                ix=np.flatnonzero(abs(yaw)>=target)
                slip10=np.flatnonzero(abs(slip)>10)
                table.append(dict(initial_speed=speed,steer=steer,gas=gas,car_mass=mass,
                                  upper_bound_total_force_accel=4*FRICTION_LIMIT/mass,snapshots=snap,
                                  time_to_90pct_late_yaw=float((ix[0]+1)*.02) if len(ix) else None,
                                  peak_abs_slip_deg=float(abs(slip).max()),
                                  time_to_slip_over10deg=float((slip10[0]+1)*.02) if len(slip10) else None,
                                  peak_course_accel=float(abs(course_accel).max()),
                                  late_effective_steer_per_curvature=float(np.median(steer*speeds[-10:]/abs(yaw[-10:]))),
                                  fraction_saturated_raw_steps=float(np.mean([r['saturated_wheels']>0 for r in step]))))
                raw.append(dict(initial_speed=speed,steer=steer,gas=gas,rows=rows))
    p=out/'synthetic-raw.json';p.write_text(json.dumps(raw,indent=2)+'\n')
    return table,provenance(p)


def interpretation():
    return {
        'observations': [
            'Lat100: 0/4 finishes with zero collision damage. Arcguard: 4/4 finishes, zero damage, 26.14/33.26/29.06/28.24 seconds.',
            'Initial departures already have large road-relative heading errors with modest slip: T1 step33 +38deg heading and -3deg slip, T3 step32 +38deg and -1.2deg, T4 step44 -49deg and +2.8deg.',
            'Synthetic gas1 vs gas0 changes cornering stability causally. At initial60/steer0.1/.64s: gas0 yaw-1.89/slip-0.34deg vs gas1 yaw-4.40/slip14.6deg.',
            'At initial60/steer0.2, gas0 peak slip1.4deg; gas0.15 peak21.7deg in1s. At initial40/steer0.2, gas0.3 peak2.9deg while gas1 peak115deg.',
            'Steady observed steer/curvature median3.35-3.47 below60m/s. Only8 clean steady rows60-75, none75-100. Coasting nonsaturated synthetic ratio approximately3.1-3.2.',
            'Clean trace next-yaw RMSE improves from0.404rad/s using current command to0.298 using previous command and0.219 using previous measured yaw. This establishes state/history dependence, not a universal model.',
        ],
        'causal_limits': [
            'Trace logs lack wheel forces/contact state and instantaneous velocity vectors. They do not prove the exact first tire-saturation time.',
            'Large late slip after departure is an effect as well as possible cause; initial departures demonstrate overshoot/phase-lag before large slip.',
            'Pixel perception validity is an algorithm flag, not measured geometric accuracy; perception error cannot be ruled out.',
            'Lateral-acceleration100 failure does not establish physical tire limit100. Pure coast can sustain much greater acceleration on uniform road, but throttle consumes rear friction and destabilizes yaw.',
            'One-second initialized straight-body experiments are not guaranteed safe turning envelopes or long-duration equilibria.',
        ],
        'hypotheses_for_controller_testing': {
            'nominal_steer_per_curvature': [3.2,3.5],
            'no_supported_large_gain_increase_with_speed': True,
            'steering_mechanism': 'Wheel angle limited to+-0.4rad; motor<=3rad/s. Reversal from+0.2 to-0.2 needs at least0.133s; command smoothing0.35 adds approximately0.043s low-frequency delay at.08s decisions.',
            'yaw_prediction_seconds': 0.08,
            'simple_observed_fit': 'next_yaw = -v*(0.14525*steer + 0.10765*previous_steer)',
            'history_observed_fit': 'next_yaw = 0.60020*yaw - 0.16197*v*steer + 0.05583*v*previous_steer',
            'fit_use': 'Diagnostic initial model only; accuracy drops near saturation, road exit, damage, large steer and out-of-coverage speed.',
            'turning_gas_cap': 'Hypothesis: cap gas0.5 in turns, taper to0 as max(v^2*abs(steer)/3.2,abs(v*yaw)) increases from100 to180m/s^2; apply to predicted near-future speed too.',
            'slip_guard': 'Hypothesis: coast when trustworthy pixel slip exceeds5deg, and actively reduce speed for growing slip or demand>180m/s^2; do not wait for10deg.',
            'turn_speed_limit': 'Keep existing48 planner limit as measured baseline until tracking is stable. A later100-150m/s^2 cap requires joint steering/pedal/lag validation; do not infer safety from the219m/s^2 force upper bound.',
            'perception_and_lag': 'Increase preview in meters with speed and explicitly account for delayed yaw/course response. Test less steer smoothing or predictive compensation individually; lower smoothing trades lag against pixel noise.',
        },
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scratch',type=Path,default=Path('/tmp/apex-control-id'))
    args=p.parse_args()
    if args.output.exists() or (args.scratch/'synthetic-raw.json').exists():
        raise FileExistsError('Choose fresh receipt paths; evidence is never overwritten.')
    args.scratch.mkdir(parents=True,exist_ok=True)
    sources=[];geometry={}
    for tid,seed in PAIRS.items():
        path=Path('/tmp/apex-feasibility')/f'geometry-t{tid}.json'
        r=json.loads(path.read_text());assert (r['track_id'],r['seed'])==(tid,seed)
        geometry[tid]=np.array(r['track'])[:,2:4];sources.append(provenance(path))
    traces,records,trace_sources=trace_analysis(Path('/tmp'),geometry)
    synth,raw=synthetic(args.scratch)
    receipt=dict(schema='apex-control-identification-v1',script=provenance(Path(__file__)),
                 official_car_source=provenance(Path('core/vendor/car_dynamics.py')),
                 required_pairs=PAIRS,environment_resets=0,synthetic_cases=len(synth),
                 sources=sources+trace_sources,traces=traces,models=fit_models(records),synthetic=synth,synthetic_raw=raw,
                 interpretation=interpretation(),
                 limitations=['Only existing four required tracks; repeated lanes are dependent samples, not generalization.',
                              'Trace slip uses average displacement and midpoint heading, not instantaneous velocity.',
                              'Nearest centerline segment distance is diagnostic and can be ambiguous near crossing geometry.',
                              'Clean-row selection: speed>10, center deviation<3m, zero damage, |slip|<0.12rad, full .08s action.',
                              'Synthetic uniform-road grip removes road departure, obstacles, damage and perception; coasting and full gas are separate.',
                              'Synthetic speed is initialized, then evolves freely; no velocity clamping or external forces.',
                              'Cross-validation excludes each required track only for regression diagnostics; no new holdout data consumed.',
                              'No production policy changes and no privileged runtime inputs.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'receipt':str(args.output),'traces':len(traces),'synthetic_cases':len(synth),'clean_samples':len(records)}))


if __name__=='__main__':
    main()
