"""No-reset prediction audit of the frozen R5 stationary observation."""
from pathlib import Path
import json,hashlib,math,numpy as np
from agents.apex_2026.v2.beam_robust_agent import Agent
from agents.apex_2026.v2.shadow_physics import ShadowCar


def main():
    p=Path('/tmp/apex-v2-beam-r5');trace=p/'t3-s4111953688.jsonl';observations=p/'stall_capture/observations.npz'
    rows=[json.loads(x) for x in trace.read_text().splitlines()];obs=np.load(observations)['observations'];r=rows[119];d=r['policy_diagnostics'];a=Agent();_,slip,valid=a.road._motion(obs[60,-2],obs[60,-1]);slip=slip if valid else rows[118]['policy_diagnostics']['predicted_slip'];c=ShadowCar();c.reset(d['speed']*math.cos(slip),-d['yaw_rate'],float(np.clip(-d['wheel_angle'],-.4,.4)),lateral_speed=d['speed']*math.sin(slip),throttle=0,omegas=d['omegas']);pred=c.step(r['action'],4);actual=np.linalg.norm(np.array(r['after']['position'])-r['before']['position'])
    result={'step':120,'actual_displacement80ms':float(actual),'predicted_displacement80ms':math.hypot(pred[0],pred[1]),'hud_speed':d['speed'],'actual_speed':r['before']['speed_m_s'],'omegas':d['omegas'],'predicted_speed_after':pred[3],'action':r['action'],'scope':'No reset. Observedpixels/actioncarry only forprediction. Actualdiagnostics used only forcomparison.','hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [trace,observations,p/'source.py',Path('agents/apex_2026/v2/shadow_physics.py')]},'limitations':['No currenttruewheelRPM captured; syntheticpixelbin equivalence establishesambiguity, notexactwheelstate.','This predictionresidual does not alone prove stationarypolicycausality. Independent beampruningcounterexample alsoexists.','Noobserver runtimefix or newenvironmentreset.']}
    Path('agents/apex_2026/v2/results/beam-r5-stationary-prediction.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
