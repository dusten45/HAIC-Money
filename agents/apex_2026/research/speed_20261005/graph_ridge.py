"""Offline metric-distance ridge prototype from saved camera pixels only.

SciPy serves only as an exact research oracle for a prospective NumPy chamfer
field; the prototype is not a legal submission artifact or vehicle controller.
"""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_fill_holes, label, map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def field(frame):
    road=(frame[:73]>=.32)&(frame[:73]<=.51)
    holes=binary_fill_holes(road)&~road
    components,count=label(holes)
    for i in range(1,count+1):
        if np.sum(components==i)<=110:road[components==i]=True
    # Image edges are unknown, not an observed grass boundary.
    padded=np.pad(road,60,constant_values=True)
    return distance_transform_edt(padded,sampling=(1/1.701,1/1.3608))[60:-60,60:-60]


def sample(distance,points):
    coords=np.array([63-points[:,1]*1.701,42+points[:,0]*1.3608])
    return map_coordinates(distance,coords,order=1,mode='constant',cval=0.)


def route(frame):
    distance=field(frame)
    start=np.column_stack((np.linspace(-8,8,65),np.full(65,3.)))
    current=start[np.argmax(sample(distance,start))]
    direction=np.array([0.,1.]);points=[current.copy()]
    for _ in range(30):
        normal=np.array([direction[1],-direction[0]])
        offsets=np.linspace(-4.,4.,41)
        prediction=current+2*direction
        candidates=prediction+offsets[:,None]*normal
        score=sample(distance,candidates)-.035*offsets**2
        chosen=candidates[np.argmax(score)]
        if sample(distance,chosen[None])[0]<2. or not (-30<chosen[0]<30 and -5<chosen[1]<37):break
        heading=chosen-current;heading/=np.linalg.norm(heading)
        direction=.25*direction+.75*heading;direction/=np.linalg.norm(direction)
        current=chosen;points.append(current.copy())
    return np.asarray(points),distance


def main():
    selections={2:[170,171,172,173,174],4:[64,65,66,98,99,100,117,143]}
    rows=[];fig,axes=plt.subplots(3,5,figsize=(15,9))
    for ax,(track,step) in zip(axes.flat,[(t,s) for t,ss in selections.items() for s in ss]):
        frame=np.load(f'.haic-artifacts/apex-speed-20261005/corridor-v1-probe-track{track}/frames.npz')['frames'][step]
        path,distance=route(frame)
        ax.imshow(frame,cmap='gray',vmin=0,vmax=1);ax.plot(42+path[:,0]*1.3608,63-path[:,1]*1.701,'c.-');ax.set(xlim=(0,83),ylim=(73,0),title=f'track{track} step{step}')
        rows.append(dict(track=track,step=step,route=path.tolist(),arclength_m=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())))
    fig.tight_layout();p=Path('agents/apex_2026/results/speed-20261005/graph-ridge-prototype');fig.savefig(p.with_suffix('.png'),dpi=130);p.with_suffix('.json').write_text(json.dumps(dict(camera_only_prototype=True,legal_submission=False,rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
