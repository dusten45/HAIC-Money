"""Exact early-circle replay and offline first-contact geometry attribution."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from agents.apex_2026.fast_early_circle_agent import Agent
from agents.apex_2026.research.speed_20261005.graph_geometry import local,distance_to_path
from local_simulator.environment import create_environment,reset_environment
from local_simulator.schema import MapSpec


def components(frame,road):
    _,_,ys,lefts,rights=road;allowed=np.zeros((84,84),bool)
    for row in range(10,72):
        left=np.interp(row,ys[::-1],lefts[::-1]);right=np.interp(row,ys[::-1],rights[::-1]);allowed[row,max(0,int(left+1)):min(84,int(right))]=True
    bright=(frame>=.60)&(frame<=.705)&allowed;seen=np.zeros_like(bright);result=[]
    for y,x in np.argwhere(bright):
        if seen[y,x]:continue
        seen[y,x]=True;stack=[(int(y),int(x))];pixels=[]
        while stack:
            py,px=stack.pop();pixels.append((py,px))
            for ny,nx in ((py-1,px),(py+1,px),(py,px-1),(py,px+1)):
                if 0<=ny<84 and 0<=nx<84 and bright[ny,nx] and not seen[ny,nx]:seen[ny,nx]=True;stack.append((ny,nx))
        pixels=np.asarray(pixels);cy,cx=pixels.mean(axis=0);width=int(np.ptp(pixels[:,1])+1);height=int(np.ptp(pixels[:,0])+1);aspect=width*1.701/(height*1.3608);surround=[]
        for py,px in ((cy,cx-4),(cy,cx+4),(cy-5,cx),(cy+5,cx)):
            iy,ix=int(round(py)),int(round(px));surround.append(float(frame[iy,ix]) if 0<=iy<73 and 0<=ix<84 else None)
        gates=dict(area=4<=len(pixels)<=48,width=2<=width<=8,height=2<=height<=9,aspect=.65<=aspect<=1.9,surround=sum(v is not None and .32<=v<=.51 for v in surround)>=3)
        result.append(dict(area=len(pixels),width=width,height=height,aspect=float(aspect),pixel_center=[float(cx),float(cy)],surround=surround,gates=gates))
    return result


def main():
    root=Path('.haic-artifacts/apex-speed-20261005');frames=np.load(root/'early-circle-v1-probe-track2/frames.npz')['frames'];trace=json.loads((root/'early-circle-v1-traces/track-2-seed-644062.json').read_text());old=json.loads((root/'clear-supported-v2-traces/track-2-seed-644062.json').read_text());spec=MapSpec(2,644062,'official',(),700,4);env,raw=create_environment(spec,None)
    try:reset_environment(env,spec);obstacles=np.asarray([list(b.position) for b in raw.obstacles]);initial=dict(position=list(raw.car.hull.position),angle=float(raw.car.hull.angle))
    finally:env.close()
    chosen=int(np.argmin(np.linalg.norm(obstacles-trace[97]['position'],axis=1)));old_chosen=int(np.argmin(np.linalg.norm(obstacles-old[96]['position'],axis=1)));agent=Agent();rows=[]
    for i,frame in enumerate(frames):
        clone=copy.deepcopy(agent);road=clone._road(frame);circles=clone._circles(frame,road) if road is not None else [];plan=None
        if road is not None:plan,imminent,_=clone._route(road[0],road[1],circles,clone._speed(frame),clone._yaw(frame))
        action=agent.act(np.stack([frame]*4));np.testing.assert_array_equal(action,np.asarray(trace[i]['action'],np.float32))
        if not 90<=i<=100:continue
        state=trace[i-1] if i else initial;truth=local(obstacles[chosen,None],state['position'],state['angle'])[0];box_gap=np.hypot(max(abs(truth[0])-1.6,0.),max(-2.4-truth[1],truth[1]-2.6,0.))-1.2
        row=dict(step=i,mode=agent.mode,circles=circles,components=components(frame,road) if road is not None else [],true_circle_local_m=truth.tolist(),preaction_rectangle_circle_gap_m=float(box_gap),speed_before=agent.last_speed,target=agent.last_target,action=action.tolist(),contact_after=trace[i]['collision'],speed_after=trace[i]['speed'],pass_side=agent.pass_side,pass_x=agent.pass_x,pass_y=agent.pass_y,pass_missing=agent.pass_missing)
        if plan is not None:row['camera_planned_center_circle_distance_m']=float(distance_to_path(truth[None],np.column_stack((plan,road[0])))[0])
        rows.append(row)
    result=dict(source_sha256=hashlib.sha256(Path('agents/apex_2026/fast_early_circle_agent.py').read_bytes()).hexdigest(),exact_replayed_actions=len(frames),new_driving_episodes=0,old_first_contact_step=97,new_first_contact_step=98,same_contact_obstacle=chosen==old_chosen,contact_obstacle_index=chosen,first_action_divergence=next(i for i,(a,b) in enumerate(zip(old,trace)) if a['action']!=b['action']),rows=rows)
    Path('agents/apex_2026/results/speed-20261005/early-circle-contact-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(same_obstacle=chosen==old_chosen,focus=rows[4:9])))
if __name__=='__main__':main()
