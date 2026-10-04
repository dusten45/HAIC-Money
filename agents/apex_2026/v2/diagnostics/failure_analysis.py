"""Read-only v2 diagnosis of consumed v1 regression evidence; never resets.

Uses saved policy-visible PNGs and evaluator traces. Simulator position/yaw are
used only for offline diagnosis and never supplied to a runtime candidate.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
FAILURES=('t2-s4031370700','t3-s4111953688')
K_MAX=math.tan(.4)/3.24


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]


def summarize_step(trace,index):
    r=trace[index];b=r['before'];a=r['after'];d=r.get('policy_diagnostics') or {};path=d.get('path',[])
    v=b['speed_m_s'];k=d.get('curvature');horizon=(63-path[-1][1])/1.701 if path else None
    slip=speed_error=None
    if index>=2 and v>=5:
        dt=b['sim_time_s']-trace[index-1]['before']['sim_time_s']
        velocity=(3*np.array(b['position'])-4*np.array(trace[index-1]['before']['position'])+np.array(trace[index-2]['before']['position']))/(2*dt)
        speed_error=float(np.linalg.norm(velocity)-v)
        slip=float(np.angle(np.exp(1j*(math.atan2(velocity[1],velocity[0])-b['heading_rad']-math.pi/2))))
    first_k=lateral=None
    if path:
        lateral=(path[0][0]-42)/1.3608;forward=(63-path[0][1])/1.701
        first_k=2*lateral/(forward**2+lateral**2) if forward**2+lateral**2 else 0.
    return dict(step=r['step'],time_s=b['sim_time_s'],position=b['position'],progress=a['info']['progress'],
        speed=v,next_speed=a['speed_m_s'],target=d.get('target_speed'),estimated_speed=d.get('speed'),
        action=r['action'],yaw_rate=b['angular_velocity'],actual_yaw_curvature_clockwise=-b['angular_velocity']/v if v>5 else None,
        commanded_curvature=k,command_exceeds_no_slip_steering_curvature=abs(k)>K_MAX if k is not None else None,
        nodes=len(path),horizon_m=horizon,first_node=path[0] if path else None,
        first_node_lateral_m=lateral,first_node_circular_connector_curvature=first_k,
        path_valid=d.get('valid'),flow_valid=d.get('flow_valid'),policy_slip=d.get('slip'),
        pre_action_slip_estimate=slip,velocity_reconstruction_speed_error=speed_error,
        slip_estimate_reliable=slip is not None and abs(speed_error)<max(1.5,.05*v),
        allocator_mode=d.get('allocator_mode'),predicted_next_speed=d.get('predicted_next_speed'),
        collision=a['info']['collision'],damage=a['info']['damage'],obstacle_pixels=d.get('obstacle_pixels'))


def pixel_arcs(frame,agent,commanded_curvature):
    """Static centerline feasibility only, not a rollout or collision guarantee."""
    free,obstacles,_=agent._free_space(frame)
    distances=np.arange(3.5,20.01,.25)
    def evaluate(k):
        x=(1-np.cos(k*distances))/k if abs(k)>1e-9 else np.zeros_like(distances)
        y=np.sin(k*distances)/k if abs(k)>1e-9 else distances
        px=42+x*1.3608;py=63-y*1.701
        inside=(px>=0)&(px<=83)&(py>=0)&(py<74)
        ix=np.clip(np.rint(px),0,83).astype(int);iy=np.clip(np.rint(py),0,83).astype(int)
        clear=inside&(free[iy,ix]>0)&(obstacles[iy,ix]==0)
        blocked=np.flatnonzero(~clear)
        return {'curvature':float(k),'clear_fraction':float(clear.mean()),
                'first_blocked_arc_distance_m':float(distances[blocked[0]]) if len(blocked) else None,
                'all_samples_clear':bool(clear.all())}
    candidates=[evaluate(k) for k in np.linspace(-K_MAX,K_MAX,101)]
    best=max(candidates,key=lambda x:(x['first_blocked_arc_distance_m'] or 21,x['clear_fraction']))
    return {'method':'car-anchored constant-curvature centerline samples at3.5..20m, road mask and already-inflated obstacle mask; no dynamic feasibility claim',
            'no_slip_curvature_limit':K_MAX,'candidate_count':len(candidates),'fully_clear_candidates':sum(c['all_samples_clear'] for c in candidates),
            'best_bounded_arc':best,'commanded_arc':evaluate(commanded_curvature) if commanded_curvature is not None else None,
            'first_anchor_reachable_lateral_at_2_94m':(1-math.sqrt(1-(K_MAX*(5/1.701))**2))/K_MAX,
            'mask_free_pixels':int(free.sum()),'mask_obstacle_pixels':int(obstacles.sum())}


def analyze(evidence,required,candidate):
    import cv2
    spec=importlib.util.spec_from_file_location('frozen_v1_analysis_only',candidate)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);agent=module.Agent()
    result={'scope':'v1 former holdout now consumed regression observations; never fresh validation','new_resets':0,
            'failures':{},'cohort':[],'method_caveats':[
             'Pre-action slip uses backward second-order position differences, not direct velocity; unreliable estimates flagged.',
             'Forward action displacement relative to initial heading includes yaw during action and must NOT be labeled instantaneous slip.',
             'Sparse saved frames do not capture either exact failure onset; no pixel counterfactual success is claimed.',
             'Static arcs sample centerline only, not swept vehicle footprint or transient dynamics; collision-free image path is necessary, not sufficient.',
             'Other-cell successes are contextual controls, not same-state causal interventions.']}
    for folder,label in [(evidence,'consumed_v1_regression'),(required,'required')]:
        for receipt in sorted(folder.glob('t*.json')):
            r=json.loads(receipt.read_text());trace=rows(receipt.with_suffix('.jsonl'));observations=[summarize_step(trace,i) for i in range(len(trace))]
            moving=[x for x in observations if x['speed']>20 and x['allocator_mode'] in ('rolling','coast_deadband')]
            model=[abs(x['predicted_next_speed']-x['next_speed']) for x in moving if x['predicted_next_speed'] is not None and not x['collision']]
            result['cohort'].append({'cell':[r['track_id'],r['seed']],'partition':label,'finished':r['finished'],'lapTimeMs':r['lapTimeMs'],
                 'receipt_sha256':sha(receipt),'trace_sha256':sha(receipt.with_suffix('.jsonl')),
                 'rolling_steps':len(moving),'rolling_unknown_flow_steps':sum(x['flow_valid'] is False for x in moving),
                 'rolling_model_next_speed_mae':float(np.mean(model)) if model else None,
                 'rolling_model_next_speed_p95_abs_error':float(np.percentile(model,95)) if model else None,
                 'moving_command_over_curvature_limit_steps':sum(x['speed']>20 and x['command_exceeds_no_slip_steering_curvature'] is True for x in observations)})
            if receipt.stem not in FAILURES:continue
            last_progress=max(i for i,t in enumerate(trace) if t['after']['info']['progress']>t['before']['info'].get('progress',0))
            selected=observations[max(0,last_progress-35):min(len(trace),last_progress+16)]
            frames=[]
            for frame_path in sorted((folder/(receipt.stem+'-frames')).glob('*.png')):
                step=int(frame_path.stem)
                if step>=len(trace):continue
                frame=cv2.imread(str(frame_path),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
                frames.append({'path':str(frame_path),'sha256':sha(frame_path),'observation_after_step':step,
                    'next_action_step':step+1,'pixel_geometry':pixel_arcs(frame,agent,(trace[step].get('policy_diagnostics') or {}).get('curvature'))})
            result['failures'][receipt.stem]={'receipt_path':str(receipt),'source_sha256':r['provenance']['agent_sha256'],
                 'steps':r['steps'],'progress':r['progress'],'damage':r['damage'],'last_progress_step':last_progress+1,
                 'selected_steps':selected,'saved_frame_geometry':frames,'all_steps_after_last_progress':len(trace)-last_progress-1}
    result['constraints']=[
      {'observed':'Both traces retain target100 until near bend entry; t2firstlarge drop at93.3m/s, t3at78.7m/s.',
       'architecture_requirement':'Unknown-beyond-visible-horizon must be represented in speed viability, with stop/turn-distance terminal constraint and temporal map horizon.'},
      {'observed':'Rolling allocator imposes desired deceleration>=−65m/s²; post-entry commands~.2 while target deficits and geometry remain large.',
       'architecture_requirement':'Separate emergency safety deceleration from nominal comfort/model rate, and couple steering+braking to reachable trajectory.'},
      {'observed':'Row-monotone paths start at arbitrary lateral offset; t3step201 starts6.50m left only2.94m ahead, unconnectable by current-pose no-slip bounded-curvature arc.',
       'architecture_requirement':'Pose-anchored 2D state-lattice/trajectory with heading, actual wheel state and swept footprint; forbid lateral teleport to first path node.'},
      {'observed':'Flowfalse sets slip0; rolling mode treats absent measurement as valid small slip. Corrected precollision slip estimates do not establish a>5degree violation here.',
       'architecture_requirement':'Keep unknown/uncertain state distinct from zero and use validity-aware model domains rather than inferred confidence.'},
      {'observed':'T2no-path fallback brakes to permanent stop; T3collision pin keeps high gas/large steer without progress.',
       'architecture_requirement':'Explicit stagnation/contact recovery state with progress memory and alternative feasible maneuvers; avoid unchanging terminal fallback.'}]
    return result


def figures(evidence,output_dir):
    """Plot existing telemetry and observations; no image synthesis or reset."""
    os.environ.setdefault('MPLCONFIGDIR','/tmp/apex-v2-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2
    output_dir.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for name in FAILURES:
        target=output_dir/(name+'.png')
        if target.exists():raise FileExistsError(target)
        trace=rows(evidence/(name+'.jsonl'))
        last=max(i for i,r in enumerate(trace) if r['after']['info']['progress']>r['before']['info'].get('progress',0))
        selected=trace[max(0,last-30):min(len(trace),last+12)]
        fig,axes=plt.subplots(2,2,figsize=(12,10))
        xy=np.array([r['before']['position'] for r in selected]);speed=[r['before']['speed_m_s'] for r in selected]
        axes[0,0].plot(xy[:,0],xy[:,1],color='gray',alpha=.5)
        scatter=axes[0,0].scatter(xy[:,0],xy[:,1],c=speed,cmap='viridis',s=20)
        for r in selected[::5]:axes[0,0].annotate(str(r['step']),r['before']['position'],fontsize=8)
        axes[0,0].axis('equal');axes[0,0].set_title('Preterminal trajectory, labels=actions');fig.colorbar(scatter,ax=axes[0,0],label='m/s')
        steps=[r['step'] for r in selected]
        axes[0,1].plot(steps,speed,label='true speed')
        axes[0,1].plot(steps,[(r.get('policy_diagnostics') or {}).get('target_speed',0) for r in selected],label='target')
        axes[0,1].legend();axes[0,1].set_title('Late target collapse');axes[0,1].set_xlabel('Action')
        for col,step in enumerate((100,200)):
            frame_path=evidence/(name+'-frames')/f'{step:04d}.png'
            frame=cv2.imread(str(frame_path),cv2.IMREAD_GRAYSCALE)
            axes[1,col].imshow(frame,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            diagnostic=trace[step].get('policy_diagnostics') or {};path=np.array(diagnostic.get('path',[]))
            if len(path):axes[1,col].plot(path[:,0],path[:,1],color='cyan')
            axes[1,col].set_title(f'After action{step}; next recorded path\nvalid={diagnostic.get("valid")}, nodes={len(path)}')
        fig.suptitle(name+' — consumed v1 regression, no new resets');fig.tight_layout();fig.savefig(target,dpi=140);plt.close(fig)
        outputs.append({'path':str(target),'sha256':sha(target)})
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir',type=Path,default=Path('/tmp/apex-final-holdout/holdout'))
    parser.add_argument('--required-dir',type=Path,default=Path('/tmp/apex-final-frozen/required'))
    parser.add_argument('--candidate',type=Path,default=ROOT/'agents/apex_2026/candidate/agent.py')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figure-dir',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=analyze(args.evidence_dir,args.required_dir,args.candidate)
    if args.figure_dir:result['figures']=figures(args.evidence_dir,args.figure_dir)
    result['generated_at']=datetime.now(timezone.utc).isoformat()
    result['provenance']={'script_sha256':sha(__file__),'candidate_sha256':sha(args.candidate),
                           'physics_source_sha256':sha(ROOT/'core/vendor/car_dynamics.py')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:{'last_progress_step':v['last_progress_step'],'progress':v['progress'],'damage':v['damage']} for k,v in result['failures'].items()},indent=2))


if __name__=='__main__':main()
