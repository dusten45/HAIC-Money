"""Offline pose-based accuracy attribution of a camera-only NumPy ridge."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from agents.apex_2026.fast_ridge_agent import Agent
from agents.apex_2026.research.speed_20261005.graph_ridge import field
from agents.apex_2026.research.speed_20261005.graph_geometry import local,distance_to_path
from local_simulator.environment import create_environment,reset_environment
from local_simulator.schema import MapSpec


def main():
    a=Agent();rows=[]
    for track,seed,steps in [(2,644062,[170,171,172,173,174]),(4,18800,[64,65,66,98,99,100,117,143])]:
        spec=MapSpec(track,seed,'official',(),700,4);env,raw=create_environment(spec,render_mode=None)
        try:reset_environment(env,spec);road=np.asarray(raw.track)[:,2:4]
        finally:env.close()
        frames=np.load(f'.haic-artifacts/apex-speed-20261005/corridor-v1-probe-track{track}/frames.npz')['frames'];trace=json.load(open(f'.haic-artifacts/apex-speed-20261005/corridor-v1-traces/track-{track}-seed-{seed}.json'))
        for i in steps:
            f=frames[i];start=time.perf_counter();path=a._ridge(f);elapsed=time.perf_counter()-start;state=trace[i-1];nearest=np.argmin(np.linalg.norm(road-state['position'],axis=1));true=local(road[np.arange(nearest-3,nearest+25)%len(road)],state['position'],state['angle']);parsed=a._road(f);sel=(parsed[0]>=0)&(abs(parsed[1])<30);old=np.column_stack((parsed[1][sel],parsed[0][sel]));olderr=distance_to_path(old,true);exact=field(f);cham=a._distance_field(f);mask=exact>0
            row=dict(track=track,step=i,ridge_available=path is not None,row_mean_error_m=float(olderr.mean()),ridge_ms=elapsed*1000,chamfer_vs_exact_mae_m=float(np.mean(abs(cham[mask]-exact[mask]))))
            if path is not None:
                newerr=distance_to_path(path,true);row.update(ridge_mean_error_m=float(newerr.mean()),ridge_max_error_m=float(newerr.max()))
            rows.append(row)
    p=Path('agents/apex_2026/results/speed-20261005/ridge-camera-proof.json');p.write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path('agents/apex_2026/fast_ridge_agent.py').read_bytes()).hexdigest(),no_driving_episodes=True,rows=rows),indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
