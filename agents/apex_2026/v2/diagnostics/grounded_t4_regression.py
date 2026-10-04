"""Saved-receipt comparison only; zero simulator resets."""
import hashlib,json
from pathlib import Path


def analyze():
    traces={};report={'cell':[4,3601050002],'episodes_run':0,'sources':{}}
    for label,folder in [('r8','terrain-r8'),('r9','terrain-grounded-r9')]:
        base=Path('/tmp/apex-v2-'+folder+'-regression/t4-s3601050002');receipt=json.loads(base.with_suffix('.json').read_text());p=base.with_suffix('.jsonl')
        traces[label]=[json.loads(x) for x in p.read_text().splitlines()]
        report['sources'][label]={'source_sha256':receipt['provenance']['agent_sha256'],'trace':str(p),'trace_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'finished':receipt['finished'],'lap_time_ms':receipt['lapTimeMs'],'progress':receipt['progress']}
    first=next(i for i,(a,b) in enumerate(zip(traces['r8'],traces['r9'])) if a['action']!=b['action'])
    report['first_action_divergence_step']=first+1
    report['matched_events']={}
    for step in [29,30,31,32,33,106,107,108,109,110,111,112,113]:
        report['matched_events'][step]={}
        for label,rows in traces.items():
            r=rows[step-1];d=r['policy_diagnostics'];report['matched_events'][step][label]={'action':r['action'],'pixel_target':d.get('target_speed'),'pixel_current_grip':d.get('current_wheel_grip'),'pixel_ground_sources':d.get('wheel_ground_sources'),'pixel_connector_feasible':d.get('connector_feasible'),'privileged_speed':r['before']['speed_m_s'],'privileged_position':r['before']['position'],'privileged_progress':r['after']['info']['progress'],'privileged_collision':r['after']['info']['collision']}
    report['findings']=[
        'Actions and trajectories match through29. Step30 historical front-left ground becomes.6; same steering but target32.4946->30.8271 and brake.194385->.192500. Thus first divergence is observed-grip conditioning, not a preceding trajectory difference.',
        'Steps31/32 also infer rear-left grass and use mean.8; lower-grip brake ceiling is65*.8/303.89578=.171111. Step31 R8 brake.191089 versus R9 .171111: lower target does not imply stronger braking when friction cap also falls.',
        'Only steps30-32 infer any grass before first impact111. Every step33-111 reports all-road. First-impact approach differences are propagated closed-loop trajectory effects, not a concurrent new grass classification.',
        'R9 connector becomes infeasible110 at58.2m/s; first collision111 starts52.9 and next state4.9m/s. R8 has no collision and finishes15.80s. Around same road segment R8 follows a different lateral line; direct same-step comparisons are not pose-matched causal counterfactuals.',
        'Postcollision connector remains infeasible, but target floor24 allows nearfull gas at nearzero speed. Car remains pinned atprogress.55636 until off_track retirement. Existing no-feasible-connector recovery is an additional structural weakness.',
        'Classification accuracy cannot be decided: raw frames29-32 and impact111 were not saved, only0/100/200/end. Historical projection, conservative erosion, road-boundary proximity or true grass are all unresolved explanations. No claim false-positive ground classification or specific obstacle shape.'
    ]
    report['limits']=['No new episodes or runtime changes.','Privileged pose/speed/collision used only offline; runtime receives pixels.','Frame100 exists but cannot establish wheel-ground truth at30 or collision object at111.','Changing inference can perturb a marginal controller without inference being wrong; this result rejects robustness promotion, not independently the observer calibration.']
    return report

if __name__=='__main__':
    p=Path('agents/apex_2026/v2/results/grounded-t4-regression.json');p.write_text(json.dumps(analyze(),indent=2)+'\n');print(p)
