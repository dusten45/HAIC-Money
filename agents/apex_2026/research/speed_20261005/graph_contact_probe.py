"""Exact saved-image contact attribution; privileged obstacle labels offline."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from agents.apex_2026.fast_clear_supported_agent import Agent
from agents.apex_2026.research.speed_20261005.graph_geometry import local,distance_to_path
from local_simulator.environment import create_environment,reset_environment
from local_simulator.schema import MapSpec


def inner_components(frame,road):
    _,_,ys,lefts,rights=road;allowed=np.zeros((84,84),bool)
    for row in range(10,72):
        left=np.interp(row,ys[::-1],lefts[::-1]);right=np.interp(row,ys[::-1],rights[::-1]);allowed[row,max(0,int(left+1)):min(84,int(right))]=True
    bright=(frame>=.655)&(frame<=.705)&allowed;seen=np.zeros_like(bright);result=[]
    for y,x in np.argwhere(bright):
        if seen[y,x]:continue
        seen[y,x]=True;stack=[(int(y),int(x))];pixels=[]
        while stack:
            py,px=stack.pop();pixels.append((py,px))
            for ny,nx in ((py-1,px),(py+1,px),(py,px-1),(py,px+1)):
                if 0<=ny<84 and 0<=nx<84 and bright[ny,nx] and not seen[ny,nx]:seen[ny,nx]=True;stack.append((ny,nx))
        pixels=np.asarray(pixels);cy,cx=pixels.mean(axis=0)
        result.append(dict(area=len(pixels),width=int(np.ptp(pixels[:,1])+1),height=int(np.ptp(pixels[:,0])+1),pixel_center=[float(cx),float(cy)]))
    return result


def main():
    root=Path('.haic-artifacts/apex-speed-20261005');frames=np.load(root/'clear-supported-v2-probe-track2/frames.npz')['frames'];trace=json.loads((root/'clear-supported-v2-traces/track-2-seed-644062.json').read_text());spec=MapSpec(2,644062,'official',(),700,4);env,raw=create_environment(spec,None)
    try:reset_environment(env,spec);obstacles=np.asarray([list(b.position) for b in raw.obstacles]);initial=dict(position=list(raw.car.hull.position),angle=float(raw.car.hull.angle))
    finally:env.close()
    selected_obstacle=obstacles[np.argmin(np.linalg.norm(obstacles-trace[93]['position'],axis=1))];agent=Agent();rows=[]
    for i,frame in enumerate(frames):
        clone=copy.deepcopy(agent);road=clone._road(frame);circles=clone._circles(frame,road) if road is not None else [];plan=None
        if road is not None:plan,imminent,_=clone._route(road[0],road[1],circles,clone._speed(frame),clone._yaw(frame))
        action=agent.act(np.stack([frame]*4));np.testing.assert_array_equal(action,np.asarray(trace[i]['action'],np.float32))
        if not 88<=i<=100:continue
        state=trace[i-1] if i else initial;truth=local(selected_obstacle[None],state['position'],state['angle'])[0];box_gap=np.hypot(max(abs(truth[0])-1.6,0.),max(-2.4-truth[1],truth[1]-2.6,0.))-1.2
        row=dict(step=i,mode=agent.mode,circles=circles,inner_components=inner_components(frame,road) if road is not None else [],true_circle_local_m=truth.tolist(),preaction_rectangle_circle_gap_m=float(box_gap),speed_before=agent.last_speed,target=agent.last_target,action=action.tolist(),contact_after=trace[i]['collision'],speed_after=trace[i]['speed'],reference_curvature=getattr(agent,'reference_curvature',None))
        if plan is not None:
            route=np.column_stack((plan,road[0]));row['camera_planned_center_circle_distance_m']=float(distance_to_path(truth[None],route)[0])
        rows.append(row)
    result=dict(source_sha256=hashlib.sha256(Path('agents/apex_2026/fast_clear_supported_agent.py').read_bytes()).hexdigest(),exact_replayed_actions=len(frames),new_driving_episodes=0,first_contact_step=next(r['step'] for r in trace if r['collision']),rows=rows)
    Path('agents/apex_2026/results/speed-20261005/clear-supported-contact-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows[6:10]))
if __name__=='__main__':main()
