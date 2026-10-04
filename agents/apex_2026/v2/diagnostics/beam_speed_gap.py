"""Lightweight saved-path/trace analysis only; no simulator or policy act calls."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def analyze():
    source=Path('/tmp/apex-v2-beam-r3/source.py')
    spec=importlib.util.spec_from_file_location('beam_gap_geometry',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    results=[]
    for folder,cells in [('beam-r3',[(1,516237),(2,644062),(3,1007),(4,18800)]),('beam-refined-r6',[(1,516237)])]:
        for track,seed in cells:
            root=Path('/tmp/apex-v2-'+folder);receipt_path=root/f't{track}-s{seed}.json';trace_path=receipt_path.with_suffix('.jsonl')
            receipt=json.loads(receipt_path.read_text());assert receipt['status']=='completed' and receipt['finished']
            rows=[json.loads(x) for x in trace_path.read_text().splitlines()]
            chord=np.array([np.linalg.norm(np.array(r['after']['position'])-r['before']['position']) for r in rows])
            speeds=np.array([r['before']['speed_m_s'] for r in rows]);ay=speeds*np.array([abs(r['before']['angular_velocity']) for r in rows])
            metrics=[]
            for r in rows:
                d=r.get('policy_diagnostics',{})
                if not d.get('valid'):continue
                reference=np.vstack(([0.,0.],(np.array(d['path'])-[42.,63.])/[1.3608,-1.701]))
                arc,kappa=module._Road._geometry(reference)
                local=np.clip(np.sqrt(100./np.maximum(abs(kappa),.001)),24.,100.)
                curvature_cap=float(np.min(np.sqrt(local**2+130.*arc)))
                frontier_cap=float(np.sqrt(24.**2+130.*arc[-1]))
                sequence=np.array(d['beam_sequence'])
                metrics.append(dict(frontier_binding=frontier_cap<=curvature_cap,horizon=float(arc[-1]),progress_fraction=float(d['beam_progress']/arc[-1]),at_end=bool(d['beam_progress']>=arc[-1]-.1),future_brake=bool(np.any(sequence[:,2]>.1)),last_brake=bool(sequence[-1,2]>.1),executed_brake=bool(r['action'][2]>.1),target_error=abs(min(curvature_cap,frontier_cap)-d['target_speed'])))
            total=float(chord.sum());lap=receipt['lapTimeMs']/1000.;launch=int(np.argmax(speeds>=60));loss=.08-chord/100.
            results.append(dict(variant=folder,track_id=track,seed=seed,receipt=str(receipt_path),trace=str(trace_path),source_sha256=receipt['provenance']['agent_sha256'],receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),lap_time_s=lap,trace_rows=len(rows),valid_policy_rows=len(metrics),chord_distance_m=total,chord_mean_speed_m_s=total/lap,required_mean_for13_m_s=total/13,required_mean_for10_m_s=total/10,instant100_chord_time_s=total/100,sampled_speed_deficit_time_vs100_s=float(loss.sum()),sampled_launch_deficit_until_first60_s=float(loss[:launch].sum()),sampled_deficit_at_abs_speed_yaw_ge100_s=float(loss[ay>=100].sum()),median_visible_horizon_m=float(np.median([x['horizon'] for x in metrics])),median_predicted_progress_over_horizon=float(np.median([x['progress_fraction'] for x in metrics])),frontier_binding_count=sum(x['frontier_binding'] for x in metrics),at_visible_end_count=sum(x['at_end'] for x in metrics),future_brake_sequence_count=sum(x['future_brake'] for x in metrics),last_action_brake_sequence_count=sum(x['last_brake'] for x in metrics),executed_brake_count=sum(x['executed_brake'] for x in metrics),target_reconstruction_max_error=max(x['target_error'] for x in metrics)))
    return dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_resets=0,scope='R3 required4 and completed R6 required1 only; no active or incomplete receipts analyzed',geometry_source=str(source),geometry_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),definitions={'distance':'Sum of before-to-after position chord lengths per80ms action; not50Hz rawcurvilinear distance. Actuallap may finish partway through last action, while trace includes its full endpoint. Hence distance/means and deficit decomposition are approximations, not physical lower bounds.','counts_denominator':'valid_policy_rows for frontier, endpoint and sequence counts; trace_rows for sampled speed deficit.','at_visible_end':'Predicted final progress >= visible reference arclength minus.1m.','frontier_binding':'sqrt(24²+130*H) <= minimum sqrt(clipped_curve_speed²+130*arc); exact current target reconstruction verified.','brake':'brake command>.1; future_brake includes any of8 planned actions including first.','speed_deficit':'sum(.08 - chord_distance/100); includes full final action and is not exactly lap_time-distance/100.','launch':'Initial sampled speed deficit before first preaction speed>=60m/s; not an optimized physical launch bound.','lateral_proxy':'abs(preaction speed*bodyyaw), not independently measured trajectory lateral acceleration during slip.'},cells=results,findings=['Visible route median~34.6m; R3 predicted terminalprogress reaches its end in48-58%decisions. Reward progress saturates beyond observed route while terminal24m/s speed and out-of-view footprint penalties remain. This is a controller-design restriction, not an official road speedlimit.','Most winning sequences contain planned braking but few firstactions execute braking. Receding-horizon terminalbraking intent is often deferred. This association alone does not prove that changing terminalcost safely improves time.','Path curvature binds most currenttargets, frontier25-33%. Actual speed exceeds these soft targets: target binding fraction is not a causal attribution of lap loss.','For recordedchordroute T2, instantaneous100m/s travel is11.39s, leaving limited13s launch/curvebudget. Physical Box2D rawtranslation limit approximately100m/s cannot be removed merely by increasing requested target. No global task impossibility claim.'],falsifiable_next_mechanism='Without newepisodes first decompose existing predicted sequence costs before/after visible-endarrival to test whether frontier saturation changes candidate rank. Then compare a justified terminalcontinuation-value/viability treatment that preserves uncertainty and stoppingfeasibility; do not simply remove unknown-horizon protection or sweep gains.',limitations=['Position/speed diagnostics are privileged offline measurements and never runtime inputs.','No new candidate, no policy search/act calls, no simulator runs.','Recordedroute budgets do not account for alternate shorter legal paths.','Current actualbodyyaw proxy and coarsechordlengths cannot establish exact friction utilization or global optimality.'])

if __name__=='__main__':
    p=Path('agents/apex_2026/v2/results/beam-speed-gap.json');p.write_text(json.dumps(analyze(),indent=2)+'\n');print(p)
