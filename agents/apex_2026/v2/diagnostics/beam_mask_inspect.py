import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/apex-matplotlib')
import importlib.util,numpy as np,matplotlib.pyplot as plt,json,hashlib
from pathlib import Path
spec=importlib.util.spec_from_file_location('b','/tmp/apex-v2-beam-r3/source.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);road=m._Road()
a=np.load('/tmp/apex-v2-beam-r3/failure_capture/observations.npz')['observations'];rows=[json.loads(x) for x in open('/tmp/apex-v2-beam-r3/t3-s4111953688.jsonl')]
fig,axs=plt.subplots(2,5,figsize=(15,6))
for ax,step in zip(axs.flat,range(130,140)):
 f=a[step-120,-1];free,ob,dist=road._free_space(f);ax.imshow(f,cmap='gray',vmin=0,vmax=1);ax.contour(free,levels=[.5],colors=['cyan'],linewidths=.6);y,x=np.where(ob);ax.scatter(x,y,c='red',s=5);p=np.array(rows[step-1]['policy_diagnostics']['path']);ax.plot(p[:,0],p[:,1],'yellow',lw=.8);ax.set_title(str(step));ax.set_xlim(15,65);ax.set_ylim(74,15);ax.axis('off');print(step,'obpixels',int(ob.sum()))
fig.tight_layout();fig.savefig('/tmp/beam-mask-montage.png',dpi=130)

output=Path('agents/apex_2026/v2/results/beam-perception')
output.mkdir(exist_ok=True)
fig.savefig(output/'masks130-139.png',dpi=130)
events=[]
for step in range(130,140):
 f=a[step-120,-1];free,ob,_=road._free_space(f);yy,xx=np.where(ob)
 events.append(dict(step=step,obstacle_pixels=int(ob.sum()),bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None,collision=rows[step-1]['after']['info']['collision']))
actual_endpoint_checks=[]
for step in (137,138):
 r=rows[step-1];h=r['before']['heading_rad'];v=np.array(r['after']['position'])-r['before']['position'];pos=v@np.array([[np.cos(h),-np.sin(h)],[np.sin(h),np.cos(h)]])
 angle=r['after']['heading_rad']-h;c,si=np.cos(angle),np.sin(angle);free,ob,_=road._free_space(a[step-120,-1])
 for kind,body in [('sparse',m.Agent.FOOTPRINT),('dense',np.array([(x,y) for x in np.arange(-1.4,1.401,.1) for y in np.arange(-2.4,2.601,.1)]))]:
  pts=body@np.array([[c,si],[-si,c]])+pos;ix=np.rint(42+1.3608*pts[:,0]).astype(int);iy=np.rint(63-1.701*pts[:,1]).astype(int)
  actual_endpoint_checks.append(dict(step=step,footprint=kind,obstacle_samples=int(ob[iy,ix].sum()),outside_free_samples=int((free[iy,ix]==0).sum())))
paths=['/tmp/apex-v2-beam-r3/source.py','/tmp/apex-v2-beam-r3/t3-s4111953688.jsonl','/tmp/apex-v2-beam-r3/failure_capture/observations.npz']
report=dict(episodes_run_by_analysis=0,capture_receipt='/tmp/apex-v2-beam-r3/failure_capture/receipt.json',sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},events=events,actual_endpoint_checks=actual_endpoint_checks,findings=['Distinct on-road obstacle correctly segmented134-138, before first collision138; no roadside/grass merger in these frames.','Obstacle vanishes139 after impact; bright detector excludes y>=61 and vehicle overlaps, so this disappearance does not explain first collision.','At actual after138 pose transformed into before138 camera, sparse25 footprint detects1 obstacle sample and dense.1m footprint detects6. Actual after137 detects0 for both. Thus simple sparse-sampling miss is not established.','Thresholded bright core is not a guaranteed physical obstacle silhouette; raster/pose uncertainty remains. Shadow lane owns predicted trajectory comparison.'],limits=['Pixel image/mask conclusions use captured stacks; actual pose/collision are privileged offline diagnostics only.','No new simulator reset; capture was separately authorized single recorded-action replay.'])
(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
