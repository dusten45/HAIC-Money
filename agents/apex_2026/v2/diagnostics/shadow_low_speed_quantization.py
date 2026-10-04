"""Offline public renderer bin ambiguity; never environment reset or runtime input."""
import hashlib,json
from pathlib import Path
import numpy as np
from agents.apex_2026.v2.diagnostics.shadow_rpm import render
from agents.apex_2026.v2.shadow_physics import decode_wheel_omega
from agents.apex_2026.v2.beam_robust_agent import Agent


def main():
    speed_rows=[];omega_rows=[]
    for speed in [0,.001,.03,.1,.5,1,2,3]:
        frame=render([0]*4,speed=speed)
        speed_rows.append({'true_speed':speed,'decoded_speed':Agent._decode_speed(frame),'speed_crop_sha256':hashlib.sha256(frame[74:83,10:13].tobytes()).hexdigest()})
    for omega in [0,.001,.01,.05,.1,.5,1,2,5,10]:
        frame=render([omega]*4,speed=0)
        omega_rows.append({'true_omega':omega,'decoded_omega':decode_wheel_omega(frame).tolist(),'rpm_crop_sha256':hashlib.sha256(frame[74:82,14:25].tobytes()).hexdigest()})
    report={'scope':'Synthetic public HUD renderer only; no environmentreset. No actualwheelstate inferred beyond observable bin ambiguity.','speed':speed_rows,'omega':omega_rows,'finding':'Lowestpositive bars alias nearzero and largerpositive values. Point decoder can inject predicted motion into almoststationary replanning. This is separate from objective procrastination; causal contribution notyetisolated.'}
    root=Path('agents/apex_2026/v2')
    report['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/'shadow_physics.py',root/'beam_robust_agent.py',root/'diagnostics/shadow_rpm.py',Path('core/vendor/car_racing.py')]}
    report['causal_limit']='Observerbias and separatebeampruning defect coexist. No claim that either alone causesallstalls.'
    Path('agents/apex_2026/v2/results/beam-r5-low-speed-quantization.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
