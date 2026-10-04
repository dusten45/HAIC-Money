"""Offline obstacle-label audit of saved antialiased camera detections."""
import hashlib,json
from pathlib import Path
import numpy as np
from agents.apex_2026.fast_clear_supported_agent import Agent as Old
from agents.apex_2026.fast_early_circle_agent import Agent as New
from agents.apex_2026.research.speed_20261005.graph_geometry import local
from local_simulator.environment import create_environment,reset_environment
from local_simulator.schema import MapSpec


def main():
    root=Path('.haic-artifacts/apex-speed-20261005');frames=np.load(root/'clear-supported-v2-probe-track2/frames.npz')['frames'];trace=json.loads((root/'clear-supported-v2-traces/track-2-seed-644062.json').read_text());spec=MapSpec(2,644062,'official',(),700,4);env,raw=create_environment(spec,None)
    try:reset_environment(env,spec);obstacles=np.asarray([list(b.position) for b in raw.obstacles]);initial=dict(position=list(raw.car.hull.position),angle=float(raw.car.hull.angle))
    finally:env.close()
    rows=[]
    for i,frame in enumerate(frames):
        old,new=Old(),New();road=new._road(frame)
        if road is None:continue
        before,after=old._circles(frame,road),new._circles(frame,road);state=trace[i-1] if i else initial;truth=local(obstacles,state['position'],state['angle']);errors=[]
        for y,x in after:errors.append(float(np.min(np.linalg.norm(truth-[x,y],axis=1))))
        if before or after:rows.append(dict(step=i,before=before,after=after,new_label_errors_m=errors,false_detections_over1_5m=sum(e>1.5 for e in errors)))
    result=dict(new_source_sha256=hashlib.sha256(Path('agents/apex_2026/fast_early_circle_agent.py').read_bytes()).hexdigest(),saved_frames=len(frames),new_driving_episodes=0,false_detections_over1_5m=sum(r['false_detections_over1_5m'] for r in rows),rows=rows)
    Path('agents/apex_2026/results/speed-20261005/early-circle-camera-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(frames=len(frames),false_detections_over1_5m=result['false_detections_over1_5m'],focus=[r for r in rows if 92<=r['step']<=97])))
if __name__=='__main__':main()
