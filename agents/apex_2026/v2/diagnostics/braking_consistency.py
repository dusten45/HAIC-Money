"""Zero-reset world-aligned audit of frame-to-frame braking-plan constraints.

World pose is evaluator-only diagnostic truth; this file is never imported by
any runtime agent. It does not simulate counterfactual actions or claim rescue.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def geometry(row):
    d=row['policy_diagnostics']
    local=(np.asarray(d['path'])-[42.,63.])/[1.3608,-1.701]
    h=row['before']['heading_rad']
    rotation=np.array([[math.cos(h),-math.sin(h)],[math.sin(h),math.cos(h)]])
    world=local@rotation.T+row['before']['position']
    arc=np.asarray(d['path_arc']);curvature=np.asarray(d['path_curvature'])
    caps=np.clip(np.sqrt(100./np.maximum(abs(curvature),.001)),24.,100.)
    envelope=np.sqrt(caps*caps+130.*arc)
    index=int(np.argmin(envelope))
    return dict(world=world,arc=arc,curvature=curvature,caps=caps,
                envelope=envelope,index=index)


def compare(previous,current):
    p=geometry(previous);c=geometry(current);i=p['index']
    distances=np.linalg.norm(c['world']-p['world'][i],axis=1);j=int(np.argmin(distances))
    travel=float(np.linalg.norm(np.array(current['before']['position'])-previous['before']['position']))
    remaining=max(0.,float(p['arc'][i])-travel)
    previous_target=previous['policy_diagnostics']['target_speed']
    target=current['policy_diagnostics']['target_speed']
    return dict(previous_step=previous['step'],step=current['step'],
        previous_target=previous_target,target=target,target_rebound=target-previous_target,
        previous_curve_envelope=float(p['envelope'][i]),
        previous_connector_feasible=previous['policy_diagnostics']['connector_feasible'],
        current_connector_feasible=current['policy_diagnostics']['connector_feasible'],
        previous_speed=previous['before']['speed_m_s'],speed=current['before']['speed_m_s'],
        previous_brake=previous['action'][2],brake=current['action'][2],
        old_bend_world=p['world'][i].tolist(),new_nearest_world=c['world'][j].tolist(),
        old_bend_arc=float(p['arc'][i]),old_bend_curvature=float(p['curvature'][i]),
        old_bend_speed_cap=float(p['caps'][i]),new_nearest_arc=float(c['arc'][j]),
        new_nearest_curvature=float(c['curvature'][j]),new_nearest_speed_cap=float(c['caps'][j]),
        world_alignment_error_m=float(distances[j]),travel_m=travel,
        transported_remaining_distance_m=remaining,
        transported_target=float(np.sqrt(p['caps'][i]**2+130.*remaining)),
        new_binding_bend_world=c['world'][c['index']].tolist(),
        new_binding_arc=float(c['arc'][c['index']]),
        damage_after=current['after']['info']['damage'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plot',type=Path)
    args=parser.parse_args()
    source_sha='b182dc495c41201e00bb983b933a7ac4afb972ee864432743c2af16480009142'
    all_events=[];critical=[];provenance=[];plot_rows=None
    for trace in sorted(args.trace_dir.glob('*.jsonl')):
        receipt=trace.with_suffix('.json')
        if not receipt.exists():continue
        result=json.loads(receipt.read_text())
        if result['provenance']['agent_sha256']!=source_sha:
            raise ValueError('unexpected source: '+str(receipt))
        rows=[json.loads(line) for line in trace.read_text().splitlines()]
        provenance.append(dict(trace=str(trace),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
            receipt=str(receipt),receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest()))
        for previous,current in zip(rows,rows[1:]):
            if not all(x.get('policy_diagnostics',{}).get('valid') for x in (previous,current)):continue
            event=compare(previous,current)
            event.update(track_id=result['track_id'],seed=result['seed'],finished=result['finished'])
            if event['target_rebound']>8 and event['previous_brake']>.05:
                all_events.append(event)
            if result['track_id']==1 and result['seed']==3601050001 and 56<=current['step']<=65:
                critical.append(event);plot_rows=rows
    report=dict(scope='Read-only consumed regression traces; zero resets, no counterfactual driving. World coordinates are diagnostic-only.',
        source_sha256=source_sha,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trace_count=len(provenance),provenance=provenance,critical_window=critical,
        rebound_events=all_events,
        interpretation='Step60->61 drops a spatially retained imminent curvature constraint. This supports testing transported constraint memory, not unconditional target hysteresis; no rescue is established.',
        limitations=['Nearest-path alignment shows spatial proximity, not identical obstacle clearance or route homotopy.',
          'Only sparse rendered frames are retained, so no per-frame free-space or smoothing-stage replay is available.',
          'Scalar transported target uses prior path arc minus actual displacement, an approximation for this short interval.',
          'One saved braking action would alter subsequent pose and observations; trace-only analysis cannot establish finish outcomes.'])
    retained=[e for e in all_events if e['previous_connector_feasible']
              and abs(e['previous_curve_envelope']-e['previous_target'])<.01
              and e['world_alignment_error_m']<1.5
              and e['transported_remaining_distance_m']>0
              and e['new_nearest_speed_cap']>e['old_bend_speed_cap']+15]
    report['rebound_summary']={'all':len(all_events),
        'on_finished_episodes':sum(e['finished'] for e in all_events),
        'aligned_retained_bend_events':retained,
        'warning':'Rebounds also occur on successful episodes; indiscriminate scalar braking hysteresis is not supported.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if args.plot and plot_rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for step in (59,60,61,62):
            row=plot_rows[step-1];g=geometry(row);p=g['world'];i=g['index']
            axes[0].plot(p[:,0],p[:,1],label=f'step {step}')
            axes[0].scatter(*p[i],s=35)
        positions=np.array([r['before']['position'] for r in plot_rows[55:65]])
        axes[0].plot(positions[:,0],positions[:,1],'k.-',label='observed car')
        axes[0].set(xlim=(25,58),ylim=(52,89),aspect='equal',xlabel='world x (m)',ylabel='world y (m)',title='Plans aligned using evaluator pose')
        axes[0].legend()
        selected=plot_rows[55:65];steps=[r['step'] for r in selected]
        axes[1].plot(steps,[r['before']['speed_m_s'] for r in selected],'k.-',label='observed speed')
        axes[1].plot(steps,[r['policy_diagnostics']['target_speed'] for r in selected],'.-',label='new plan target')
        axes[1].plot([r['step'] for r in critical],[r['transported_target'] for r in critical],'.--',label='prior bend transported one step')
        axes[1].set(xlabel='decision',ylabel='m/s',title='Constraint disappears for one decision')
        axes[1].legend();fig.tight_layout();fig.savefig(args.plot,dpi=150)
    print(json.dumps(dict(trace_count=len(provenance),rebound_count=len(all_events),critical61=next(x for x in critical if x['step']==61)),indent=2))


if __name__=='__main__':main()
