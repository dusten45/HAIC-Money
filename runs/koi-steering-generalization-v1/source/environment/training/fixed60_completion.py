"""Registered matched full-episode completion comparison, pixel-only actors."""
import argparse
import hashlib
import json
import resource
import statistics
import time
import numpy as np
from training.env_factory import create_training_environment
from core.vendor.car_racing import TRACK_WIDTH
from pathlib import Path
from haic_agent.fixed60_completion import Fixed60Completion
from training.evaluate_closed_loop import run_episode

SETTINGS = Path('docs/plans/active/fixed60-completion-settings.json')


def run(output):
    settings=json.loads(SETTINGS.read_text(encoding='utf-8-sig'))
    actor=Fixed60Completion
    if settings.get('actor')=='coordination':
        from haic_agent.fast_completion_coordination import FastCompletionCoordination
        actor=FastCompletionCoordination
    elif settings.get('actor')=='out_in_out':
        from haic_agent.out_in_out_runtime import OutInOutAgent
        actor=OutInOutAgent
    if settings.get('actor')=='out_in_out_flow':
        from haic_agent.out_in_out_flow_runtime import OutInOutFlowAgent
        actor=OutInOutFlowAgent
    if settings.get('actor')=='out_in_out_visibility':
        from haic_agent.out_in_out_visibility_runtime import OutInOutVisibilityAgent
        actor=OutInOutVisibilityAgent
    if settings.get('actor')=='high_acceleration_path':
        from haic_agent.high_acceleration_path_runtime import HighAccelerationPathAgent
        actor=HighAccelerationPathAgent
    if settings.get('actor')=='traction_path':
        from haic_agent.traction_path_runtime import TractionPathAgent
        actor=TractionPathAgent
    if settings.get('actor')=='geometric_passing':
        from haic_agent.geometric_passing_runtime import GeometricPassingAgent
        actor=GeometricPassingAgent
    if settings.get('actor')=='fast_committed_recovery':
        from haic_agent.fast_committed_recovery_runtime import FastCommittedRecoveryAgent
        actor=FastCommittedRecoveryAgent
    if settings.get('actor')=='fast_road_commit':
        from haic_agent.fast_road_commit_runtime import FastRoadCommitAgent
        actor=FastRoadCommitAgent
    if settings.get('actor')=='fast_clearance_cone':
        from haic_agent.fast_clearance_cone_runtime import FastClearanceConeAgent
        actor=FastClearanceConeAgent
    if settings.get('actor')=='preview_pedal':
        from haic_agent.preview_pedal_runtime import PreviewPedalAgent
        actor=PreviewPedalAgent
    if settings.get('actor')=='preview_pedal_memory':
        from haic_agent.preview_pedal_memory_runtime import PreviewPedalMemoryAgent
        actor=PreviewPedalMemoryAgent
    if settings.get('actor')=='preview_pedal_fast':
        from haic_agent.preview_pedal_fast_runtime import PreviewPedalFastAgent
        actor=PreviewPedalFastAgent
    if settings.get('actor')=='connected_preview':
        from haic_agent.connected_preview_runtime import ConnectedPreviewAgent
        actor=ConnectedPreviewAgent
    if settings.get('actor')=='connected_guarded_preview':
        from haic_agent.connected_guarded_preview_runtime import ConnectedGuardedPreviewAgent
        actor=ConnectedGuardedPreviewAgent
    if settings.get('actor')=='handoff_path':
        from haic_agent.handoff_path_runtime import HandoffPathAgent
        actor=HandoffPathAgent
    if settings.get('actor')=='unified_passing':
        from haic_agent.unified_passing_runtime import UnifiedPassingAgent
        actor=UnifiedPassingAgent
    if settings.get('actor')=='acceleration_envelope':
        from haic_agent.acceleration_envelope_runtime import AccelerationEnvelopeAgent
        actor=AccelerationEnvelopeAgent
    if settings.get('actor')=='acceleration_braking':
        from haic_agent.acceleration_braking_runtime import AccelerationBrakingAgent
        actor=AccelerationBrakingAgent
    if settings.get('actor')=='far_hazard':
        from haic_agent.far_hazard_runtime import FarHazardAgent
        actor=FarHazardAgent
    if settings.get('actor')=='far_hazard_package':
        from training.confirm_far_hazard import run as package_run
        return package_run(output)
    if settings.get('actor')=='contact_continuity':
        from haic_agent.contact_continuity_runtime import ContactContinuityAgent
        actor=ContactContinuityAgent
    if settings.get('actor')=='contact_package':
        from training.confirm_contact_continuity import run as package_run
        return package_run(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    start=time.monotonic()
    for track,seed in settings['cells']:
        arms=settings['arms']
        offset=(track+seed)%len(arms)
        for arm in arms[offset:]+arms[:offset]:
            begin=time.monotonic()
            agent=actor(arm)
            create=time.monotonic()-begin
            reset_times=[]
            reset=agent.reset
            def timed_reset(observation):
                begin=time.monotonic(); reset(observation); reset_times.append(time.monotonic()-begin)
            agent.reset=timed_reset
            geometry=[]
            def measured_environment(**kwargs):
                environment=create_training_environment(**kwargs)
                step=environment.step
                previous_index=[None]
                def measured_step(action):
                    raw=environment.unwrapped
                    points=np.asarray(raw.track,dtype=float)[:,2:4]
                    delta=np.roll(points,-1,axis=0)-points
                    length2=np.sum(delta*delta,axis=1)
                    position=np.asarray(raw.car.hull.position,dtype=float)
                    fraction=np.clip(np.sum((position-points)*delta,axis=1)/np.maximum(length2,1e-12),0,1)
                    error=position-(points+fraction[:,None]*delta)
                    distance2=np.sum(error*error,axis=1)
                    indices=np.arange(len(points)) if previous_index[0] is None else (previous_index[0]+np.arange(-12,13))%len(points)
                    index=int(indices[np.argmin(distance2[indices])])
                    previous_index[0]=index
                    signed=float((delta[index,0]*error[index,1]-delta[index,1]*error[index,0])/max(length2[index]**.5,1e-6))
                    geometry.append(dict(segment=index,left_distance=signed,normalized_left=signed/TRACK_WIDTH,
                                         association_distance=float(distance2[index]**.5),timing='pre_action',
                                         wheel_omega=[float(w.omega) for w in raw.car.wheels],
                                         wheel_angle=[float(w.joint.angle) for w in raw.car.wheels],
                                         wheel_gas=[float(w.gas) for w in raw.car.wheels],
                                         velocity=list(map(float,raw.car.hull.linearVelocity)),
                                         yaw_rate=float(raw.car.hull.angularVelocity),
                                         vehicle_mass=float(raw.car.hull.mass+sum(w.mass for w in raw.car.wheels))))
                    return step(action)
                environment.step=measured_step
                return environment
            row=run_episode(mode='fixed60_'+arm,track_id=track,seed=seed,agent=agent,max_decisions=1200,
                            environment_factory=measured_environment,plan_budget_seconds=4.5,capture_trace=True,fail_on_invalid_action=True)
            trace=row['decision_trace']
            if len(trace)!=len(geometry): raise RuntimeError('Geometry timing mismatch')
            for decision,measurement in zip(trace,geometry): decision['road_geometry']=measurement
            prefix=[[r[k] for k in ('steer','gas','brake','car_x','car_y','car_yaw')] for r in trace[:10]]
            row.update(arm=arm,create_seconds=create,reset_seconds=reset_times,
                       peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                       interventions=sum(t['controller']['completion_changed'] for t in trace),
                       cone_interventions=sum(t['controller'].get('cone_changed',False) for t in trace),
                       prefix_hash=hashlib.sha256(json.dumps(prefix).encode()).hexdigest())
            path=output.parent/f'{track}-{seed}-{arm}.json'
            with path.open('x') as f: json.dump(row,f)
            if row.get('error') or row['invalid_actions']: raise RuntimeError(str(path))
            compact={k:row[k] for k in ('arm','completed','lapTimeMs','progress','collisions','damage','retire_reason','interventions','prefix_hash')}
            compact.update(track=track,seed=seed,evidence=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append(compact); print(json.dumps(compact),flush=True)
    summary={}
    for arm in settings['arms']:
        selected=[r for r in rows if r['arm']==arm]
        times=[r['lapTimeMs'] for r in selected if r['completed']]
        summary[arm]=dict(completed=sum(r['completed'] for r in selected),denominator=len(selected),
                          median_finished_ms=statistics.median(times) if times else None,
                          interventions=sum(r['interventions'] for r in selected))
    same_prefix=all(len({r['prefix_hash'] for r in rows if r['track']==t and r['seed']==s})==1 for t,s in settings['cells'])
    with output.open('x') as f: json.dump(dict(settings=settings,rows=rows,summary=summary,same_prefix=same_prefix,
                                               duration_s=time.monotonic()-start),f,indent=2)
    if settings.get('same_prefix_required',True) and not same_prefix: raise RuntimeError('Unequal initial prefixes')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
