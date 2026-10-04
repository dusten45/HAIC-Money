"""Offline selected-shadow-fixture versus captured obstacle pixel cells; zero resets."""
import importlib.util,numpy as np,json,cv2,hashlib
from pathlib import Path
s=importlib.util.spec_from_file_location('b','/tmp/apex-v2-beam-r3/source.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);road=m._Road();a=np.load('/tmp/apex-v2-beam-r3/failure_capture/observations.npz')['observations'];ticks=json.load(open('/tmp/apex-v2-beam-r3/failure_capture/selected_ticks.json'))
events=[]
for record in ticks:
 free,ob,_=road._free_space(a[record['step']-120,-1]);yy,xx=np.where(ob)
 for t in record['ticks']:
  hits=[]
  for bi,b in enumerate(t['bodies']):
   c,si=np.cos(b['angle']),np.sin(b['angle'])
   for pi,p in enumerate(b['polygons']):
    poly=np.array(p)@np.array([[c,si],[-si,c]])+b['position'];poly=(poly*np.array([1.3608,-1.701])+[42,63]).astype(np.float32)
    for x,y in zip(xx,yy):
     cell=np.array([[x-.5,y-.5],[x+.5,y-.5],[x+.5,y+.5],[x-.5,y+.5]],np.float32);area,_=cv2.intersectConvexConvex(poly,cell)
     if area>1e-7:hits.append(dict(body=bi,fixture=pi,pixel=[int(x),int(y)],overlap_pixel_area=float(area)))
  x,y,h=t['pose'];c,si=np.cos(h),np.sin(h)
  rect=np.array([(u,v) for u in np.arange(-1.4,1.401,.025) for v in np.arange(-2.4,2.601,.025)])
  pts=rect@np.array([[c,si],[-si,c]])+[x,y];ix=np.rint(42+1.3608*pts[:,0]).astype(int);iy=np.rint(63-1.701*pts[:,1]).astype(int)
  events.append(dict(step=record['step'],raw_tick=t['raw_tick'],original_sparse_cost=t['footprint_cost'],dense_rectangle_obstacle_samples=int(ob[iy,ix].sum()),exact_fixture_pixel_intersections=hits))

paths=['/tmp/apex-v2-beam-r3/source.py','/tmp/apex-v2-beam-r3/failure_capture/selected_ticks.json','/tmp/apex-v2-beam-r3/failure_capture/observations.npz']
report=dict(episodes_run=0,sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},events=events,conclusion='At step138 rawtick3 the predicted front-left wheel fixture intersects detected obstacle pixel39,53 by.235641px2 while original25point footprint cost is0. The same predicted trajectory therefore contains a mask-level collision missed by sparse sampling. This does not require changing observer state or widening rectangle1.4m; dense1.4m rectangle also detects it.',limits=['Polygon intersections use the square area represented by thresholded obstacle pixels, not privileged exact circular obstacle geometry. Raster uncertainty remains.','Exact physical wheel footprint intersections are nonzero only138tick3, not finaltick4; endpoint-only checks would miss this transient.','The script imports candidate definitions and processes captured predictions; no environment or shadow stepping occurs.','Dense rectangular footprint overbounds actual fixtures: tick4 rectangleoverlap but exactfixturesclear. Prefer exactpolygon-cell geometry or proven conservative envelope, not densepointcount interpretation as groundtruth.'])
output=Path('agents/apex_2026/v2/results/beam-perception/dense-footprint.json');output.write_text(json.dumps(report,indent=2)+'\n');print(output)
