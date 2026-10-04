"""Privileged offline camera-road attribution; saved frames only, no driving.

Track pose is used solely for diagnostic ground truth and never inference.
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
from local_simulator.environment import create_environment, reset_environment
from local_simulator.schema import MapSpec


def local(points, position, angle):
    relative = points-np.asarray(position)
    c,s = np.cos(angle), np.sin(angle)
    return np.column_stack((c*relative[:,0]+s*relative[:,1], -s*relative[:,0]+c*relative[:,1]))


def distance_to_path(points, path):
    start, links = path[:-1], np.diff(path,axis=0)
    fraction=np.clip(np.sum((points[:,None]-start)*links,axis=2)/np.maximum(np.sum(links*links,axis=1),1e-8),0,1)
    return np.min(np.linalg.norm(points[:,None]-start-fraction[:,:,None]*links,axis=2),axis=1)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--track',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); root=Path('.haic-artifacts/apex-speed-20261005'); probe=root/f'corridor-v1-probe-track{args.track}'
    frames=np.load(probe/'frames.npz')['frames'];trace=json.loads((root/'corridor-v1-traces'/f'track-{args.track}-seed-{args.seed}.json').read_text())
    spec=importlib.util.spec_from_file_location('graph_reference','agents/apex_2026/fast_corridor_agent.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);agent=m.Agent()
    ms=MapSpec(args.track,args.seed,'official',(),700,4);env,raw=create_environment(ms,render_mode=None)
    try:
        reset_environment(env,ms);center=np.asarray(raw.track)[:,2:4];initial_position=list(raw.car.hull.position);initial_angle=float(raw.car.hull.angle)
    finally:env.close()
    rows=[]; overlays=[]
    for i,frame in enumerate(frames):
        state=trace[i-1] if i else dict(position=initial_position,angle=initial_angle)
        nearest=int(np.argmin(np.linalg.norm(center-np.asarray(state['position']),axis=1)))
        # Retain a small amount behind and the next 24 forward waypoints (~84m).
        road_points=center[(np.arange(nearest-3,nearest+25))%len(center)]
        true=local(road_points,state['position'],state['angle'])
        parsed=agent._road(frame)
        if parsed is None: rows.append(dict(step=i,lost=True));continue
        y,x=parsed[:2]; selected=(y>=0)&(y<=30);points=np.column_stack((x[selected],y[selected]));err=distance_to_path(points,true)
        angles=np.arctan2(np.diff(true[:,0]),np.diff(true[:,1]));turnback=bool(np.any(np.diff(true[:,1])[:15]<-.2))
        row=dict(step=i,lost=False,mean_center_error_m=float(err.mean()),max_center_error_m=float(err.max()),road_forward_extent_m=float(y[-1]),forward_branch_turnback=turnback,speed=float(state.get('speed',0)),progress=float(state.get('progress',0)))
        rows.append(row);overlays.append((row,frame,true,parsed))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(dict(track=args.track,seed=args.seed,source='fast_corridor_agent.py',frames=len(frames),no_driving_episodes=True,rows=rows),indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    selected=sorted(overlays,key=lambda item:item[0]['mean_center_error_m'],reverse=True)[:8]
    fig,axes=plt.subplots(2,4,figsize=(14,8))
    for ax,(row,frame,true,parsed) in zip(axes.flat,selected):
        ax.imshow(frame,cmap='gray',vmin=0,vmax=1);ax.plot(42+true[:,0]*1.3608,63-true[:,1]*1.701,'g.-',label='actual forward track');ax.plot(42+parsed[1]*1.3608,63-parsed[0]*1.701,'r.-',label='parsed row center');ax.set(xlim=(0,83),ylim=(73,0),title=f"step {row['step']} err {row['mean_center_error_m']:.1f}m");ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(args.output.with_suffix('.png'),dpi=130)
    print(json.dumps(dict(track=args.track,worst=sorted(rows,key=lambda r:r.get('mean_center_error_m',0),reverse=True)[:8])))
if __name__=='__main__':main()
