"""Attribute solver memory erasure on exact saved missed-detection frames."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from agents.apex_2026.fast_early_circle_agent import Agent,_PathReference
from agents.apex_2026.research.speed_20261005.graph_geometry import distance_to_path


def main():
    root=Path('.haic-artifacts/apex-speed-20261005');frames=np.load(root/'early-circle-v1-probe-track2/frames.npz')['frames'];trace=json.loads((root/'early-circle-v1-traces/track-2-seed-644062.json').read_text());agent=Agent();rows=[]
    for i,frame in enumerate(frames):
        if i in (95,97):
            prior={name:getattr(agent,name) for name in ('pass_side','pass_x','pass_y','pass_missing','last_steer','last_target')};row=dict(step=i,prior=prior,variants={})
            for name in ('corridor','remembered_base_pass'):
                clone=copy.deepcopy(agent);road=clone._road(frame);circles=clone._circles(frame,road);speed,yaw=clone._speed(frame),clone._yaw(frame)
                if name=='corridor':path,_,_=clone._route(road[0],road[1],circles,speed,yaw)
                else:path,_,_=_PathReference._route(clone,road[0],road[1],circles,speed,yaw)
                remembered=np.array([clone.pass_x,clone.pass_y]);clearance=float(distance_to_path(remembered[None],np.column_stack((path,road[0])))[0]);desired=clone._steering(road[0],path,speed,yaw);emitted=clone.last_steer+float(np.clip(desired-clone.last_steer,-.24,.24));k=clone.reference_curvature;ff=float(np.arctan(clone.WHEELBASE*k));fb=clone.yaw_gain*clone.WHEELBASE*(speed*k-yaw)/max(speed,20.)
                row['variants'][name]=dict(camera_circles=circles,transported_circle=remembered.tolist(),pass_missing=clone.pass_missing,planned_center_clearance_m=clearance,reference_curvature=k,feedforward_steer=ff,yaw_feedback_steer=float(fb),emitted_steer=emitted)
            rows.append(row)
        action=agent.act(np.stack([frame]*4));np.testing.assert_array_equal(action,np.asarray(trace[i]['action'],np.float32))
    result=dict(source_sha256=hashlib.sha256(Path('agents/apex_2026/fast_early_circle_agent.py').read_bytes()).hexdigest(),exact_replayed_actions=len(frames),new_driving_episodes=0,rows=rows);Path('agents/apex_2026/results/speed-20261005/memory-corridor-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
