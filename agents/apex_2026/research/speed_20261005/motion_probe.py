"""Replay frozen camera actions; simulator truth is only a measurement target."""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from local_simulator.environment import create_environment, reset_environment
from local_simulator.schema import MapSpec


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--driver-source',required=True,type=Path)
    parser.add_argument('--trace',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--track',type=int,default=4)
    parser.add_argument('--seed',type=int,default=18800)
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--cache',type=Path)
    args=parser.parse_args()
    source_hash,driver_hash=sha(args.source),sha(args.driver_source)
    observer=load(args.source,'motion_observer').Agent()
    driver=load(args.driver_source,'motion_frozen_driver').Agent()
    recorded=json.loads(args.trace.read_text())
    spec=MapSpec(args.track,args.seed,'official',(),700,4)
    environment,raw=create_environment(spec,render_mode=None)
    rows=[];frames=[]
    try:
        observation,_=reset_environment(environment,spec)
        driver.reset(observation);observer.reset(observation)
        for step,saved in enumerate(recorded[:args.steps]):
            # Truth is read here and never passed into observer or driver.
            local=raw.car.hull.GetLocalVector(raw.car.hull.linearVelocity)
            true_speed=float(np.linalg.norm(raw.car.hull.linearVelocity))
            began=time.perf_counter()
            estimate=observer._estimate_motion(observation)
            latency=1000*(time.perf_counter()-began)
            true_heading=float(np.arctan2(local[0],local[1])) if true_speed>1. else 0.
            error=float(np.arctan2(np.sin(estimate['heading']-true_heading),
                                  np.cos(estimate['heading']-true_heading)))
            rows.append({'step':step,**estimate,'true_speed':true_speed,
                         'true_lateral_velocity':float(local[0]),
                         'true_heading':true_heading,'heading_error':error,
                         'true_hull_yaw_camera':-float(raw.car.hull.angularVelocity),
                         'true_hull_angle':float(raw.car.hull.angle),
                         'hud_yaw_camera':observer._yaw(observation[-1]),
                         'latency_ms':latency})
            if args.cache:frames.append(np.asarray(observation).copy())
            action=driver.act(observation)
            np.testing.assert_array_equal(action,np.asarray(saved['action'],np.float32))
            observation,_,terminated,truncated,_=environment.step(action)
            if terminated or truncated:break
    finally:
        environment.close()
    assert sha(args.source)==source_hash and sha(args.driver_source)==driver_hash
    selected=[r for r in rows if r['confidence']>=.7 and r['true_speed']>=20.]
    high_slip=[r for r in rows if r['true_speed']>=20. and abs(r['true_heading'])>.20]
    summary={'samples':len(rows),'confident_high_speed_samples':len(selected),
             'motion_latency_max_ms':max(r['latency_ms'] for r in rows),
             'true_large_slip_samples':len(high_slip),
             'large_slip_confident_samples':sum(r['confidence']>=.7 for r in high_slip)}
    if selected:
        heading=np.abs([r['heading_error'] for r in selected])*180/np.pi
        lateral=np.abs([r['lateral_velocity']-r['true_lateral_velocity'] for r in selected])
        summary.update({'heading_mae_deg':float(np.mean(heading)),
                        'heading_p90_deg':float(np.quantile(heading,.9)),
                        'lateral_velocity_mae_mps':float(np.mean(lateral)),
                        'lateral_velocity_p90_mps':float(np.quantile(lateral,.9))})
    report={'source_sha256':source_hash,'driver_sha256':driver_hash,
            'trace_sha256':sha(args.trace),'track_id':args.track,'seed':args.seed,
            'type':'diagnostic exact-action replay; not a new candidate lap benchmark',
            'truth_enters_inference':False,'summary':summary,'rows':rows}
    if args.cache:
        assert not args.cache.exists()
        np.savez_compressed(args.cache,observations=frames)
        report['camera_cache_sha256']=sha(args.cache)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(summary))


if __name__=='__main__':main()
