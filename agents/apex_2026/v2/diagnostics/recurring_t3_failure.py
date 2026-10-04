"""Read saved consumed-regression receipts/traces; never construct a simulator."""
import hashlib
import json
from pathlib import Path
import numpy as np


def analyze():
    report={'cell':[3,4019195530],'episodes_run':0,'variants':{}}
    for name,folder in [('r6','geodesic-frenet-r6'),('r7','geodesic-footprint-r7'),('r8','terrain-r8')]:
        base=Path('/tmp/apex-v2-'+folder+'-regression/t3-s4019195530')
        receipt=json.loads(base.with_suffix('.json').read_text())
        trace=base.with_suffix('.jsonl');rows=[json.loads(x) for x in trace.read_text().splitlines()]
        first=next(i for i,r in enumerate(rows) if r['after']['info'].get('collision'))
        events=[]
        for r in rows[max(0,first-10):first+1]:
            d=r['policy_diagnostics'];arc=np.array(d['path_arc']);k=np.array(d['path_curvature'])
            curve=np.sqrt(100/np.maximum(abs(k),.001)).clip(24,100)
            j=int(np.argmin(curve**2+130*arc));v=r['before']['speed_m_s']
            events.append(dict(step=r['step'],pixel_target_speed=d['target_speed'],pixel_road_goal=d.get('road_goal'),pixel_binding_arc_m=float(arc[j]),pixel_curve_speed_road100=float(curve[j]),pixel_binding_curvature=float(k[j]),pixel_connector_feasible=d['connector_feasible'],privileged_speed_m_s=v,privileged_progress=r['after']['info']['progress'],privileged_damage=r['after']['info']['damage'],required_constant_decel_road100=max(0,float((v*v-curve[j]**2)/(2*arc[j]))),action=r['action']))
        report['variants'][name]=dict(receipt=str(base.with_suffix('.json')),trace=str(trace),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),source_sha256=receipt['provenance']['agent_sha256'],first_collision_step=rows[first]['step'],final_progress=receipt['progress'],events=events)
    report['findings']=[
        'All three first collide at progress approximately .572-.576; different final DNF positions are subsequent outcomes.',
        'R8 step108 already contains a left-turn curve .064/m at26.49m; step109 .070/m at21.11m. Turn is represented before frontier switch, so entirely-unseen-bend explanation is unsupported.',
        'R8 step109 top frontier[29,5] switches at110 to left frontier[0,53]; binding curve grows to .110/m at11.20m. Current66m/s needs approximately154m/s2 constant deceleration to30.2 within11.2m, versus configured65. Braking already active since109.',
        'R8 connector first infeasible114, collision116, then repeated closed-loop driving with no progress after about139 until239. Mask-invalid stop is not the primary failure.',
        'Only frame100 and200 are saved near these events. Pixel-derived path/goal diagnostics demonstrate representation discontinuity, but rawmask visibility of the left branch at109 and obstacle classification at impact cannot be determined without an additional episode, which was not run.',
        'Candidate structural hypotheses: temporally consistent frontier selection and braking viability using newly observed tighter curvature; static constant-arc fallback remains dynamically approximate. No fix is validated by this diagnostic.'
    ]
    report['limits']=['Curve binding reconstruction assumes road lateral100 and omits R8 planned per-wheel friction; it is an explicitly optimistic road reference, not exact R8 target reconstruction.','Privileged speed/progress/pose are offline evidence only; runtime only uses pixel observations.','Required constant deceleration is kinematic demand, not proof tire/friction feasibility during simultaneous turn.','No claim of obstacle image proof or actual grass contact.']
    return report

if __name__=='__main__':
    output=Path('agents/apex_2026/v2/results/recurring-t3-failure.json')
    output.write_text(json.dumps(analyze(),indent=2)+'\n')
    print(output)
