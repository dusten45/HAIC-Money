"""Exact saved-camera V1/V2 replay; no new driving or inference pose input."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np


def component(frame):
    asphalt=(frame>=.32)&(frame<=.51);asphalt[73:]=False
    candidates=np.argwhere(asphalt[53:73]);candidates[:,0]+=53
    metric=((candidates[:,1]-42)/1.3608)**2+((candidates[:,0]-63)/1.701)**2
    seed=candidates[np.argmin(metric)];seen=np.zeros((84,84),bool);seen[tuple(seed)]=True;stack=[tuple(seed)];pixels=[]
    while stack:
        y,x=stack.pop();pixels.append((y,x))
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                ny,nx=y+dy,x+dx
                if 0<=ny<73 and 0<=nx<84 and asphalt[ny,nx] and not seen[ny,nx]:seen[ny,nx]=True;stack.append((ny,nx))
    pixels=np.asarray(pixels)
    return dict(seed=seed.tolist(),nearest_distance_m=float(np.sqrt(metric.min())),pixels=len(pixels),row_span=int(np.ptp(pixels[:,0])+1),column_span=int(np.ptp(pixels[:,1])+1))


def main():
    root=Path('.haic-artifacts/apex-speed-20261005');report={}
    for version,source in [('v1',root/'graph-aborted/v1-source.py'),('v2',Path('agents/apex_2026/fast_graph_agent.py'))]:
        spec=importlib.util.spec_from_file_location('confidence_'+version,source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);agent=module.Agent()
        frames=np.load(root/f'graph-{version}-probe-track4/frames.npz')['frames'];trace=json.loads((root/f'graph-{version}-traces/track-4-seed-18800.json').read_text());agent.reset()
        rows=[]
        for i,frame in enumerate(frames):
            road=agent._road(frame);action=agent.act(np.stack([frame]*4));np.testing.assert_array_equal(action,np.asarray(trace[i]['action'],np.float32));row=component(frame)
            row.update(step=i,road_none=road is None,action=action.tolist(),speed_before=float(agent.last_speed),speed_after=trace[i]['speed'],progress_after=trace[i]['progress'])
            if i:
                delta=np.asarray(trace[i]['position'])-trace[i-1]['position'];angle=.5*(trace[i]['angle']+trace[i-1]['angle']);lateral=delta[0]*np.cos(angle)+delta[1]*np.sin(angle);forward=-delta[0]*np.sin(angle)+delta[1]*np.cos(angle);row['interval_sideslip_deg']=float(np.degrees(np.arctan2(lateral,forward)))
            rows.append(row)
        report[version]=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),exact_replayed_actions=len(rows),road_none_count=sum(r['road_none'] for r in rows),rows=rows)
    a=report['v1']['rows'];b=report['v2']['rows'];report['first_action_divergence']=next(i for i,(x,y) in enumerate(zip(a,b)) if x['action']!=y['action']);report['new_driving_episodes']=0
    Path('agents/apex_2026/results/speed-20261005/graph-confidence-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(first_action_divergence=report['first_action_divergence'],v1_none=report['v1']['road_none_count'],v2_none=report['v2']['road_none_count'],first_component=a[70])))
if __name__=='__main__':main()
