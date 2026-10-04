"""Read-only observer-clone analysis of the authorized consumed-cell replay."""
import json,pathlib,hashlib,math
import cv2,numpy as np
from agents.apex_2026.v2.obstacle_steering_agent import Agent
from agents.apex_2026.v2.shadow_physics import decode_wheel_omega
ROOT=pathlib.Path('/tmp/apex-v2-obstacle-steering-audit')

def footprint(a):
 mask=np.zeros((84,84),np.uint8)
 for body in [a.shadow.hull]+a.shadow.wheels:
  for fixture in body.fixtures:
   q=np.array([body.GetWorldPoint(v) for v in fixture.shape.vertices])
   p=np.rint(np.column_stack((42+q[:,0]*a.PIXELS_X,63-q[:,1]*a.PIXELS_Y))).astype(np.int32)
   cv2.fillConvexPoly(mask,p,1)
 return mask

def main():
 rows=[]
 for p in sorted(ROOT.glob('*-state.json')):
  saved=json.loads(p.read_text());step=saved['step'];obs=np.load(ROOT/f'{step:04d}-observation.npy');frame=obs[-1];a=Agent()
  obstacle=a._obstacles(frame);n,labels,stats,_=cv2.connectedComponentsWithStats(obstacle,8)
  components=[]
  for i in range(1,n):
   y,x=np.where(labels==i);values=frame[y,x];q=np.column_stack((x/a.PIXELS_X,y/a.PIXELS_Y));e=np.linalg.eigvalsh(np.cov(q.T))
   components.append(dict(bounds=stats[i,:4].tolist(),area=int(stats[i,4]),gray_min=float(values.min()),gray_mean=float(values.mean()),gray_max=float(values.max()),metric_pca_aspect=float(np.sqrt(e[-1]/max(e[0],1e-8))),cannot_be_pure_orange=bool(values.max()>.71)))
  road=((frame>=.24)&(frame<=.52)).astype(np.uint8);road=cv2.morphologyEx(road,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8));road[71:]=0
  state=saved['state'];state['omega']=decode_wheel_omega(frame)
  a._initialize(state);road[footprint(a)>0]=1 # current self-occlusion, marked assumption
  path=np.array(saved['diagnostics']['path']);metric=np.column_stack(((path[:,0]-42)/a.PIXELS_X,(63-path[:,1])/a.PIXELS_Y))
  predictions={}
  for name,action in [('selected',saved['action']),('nominal',saved['diagnostics']['obstacle_shield_nominal'])]:
   a._initialize(state);samples=[];sweep=np.zeros_like(road)
   for tick in range(1,17):
    a.shadow.step(action,1);mask=footprint(a);sweep|=mask
    outside=(mask>0)&(road==0)
    samples.append(dict(tick=tick,road_outside_fraction=float(outside.sum()/max(mask.sum(),1)),obstacle_overlap_pixels=int((mask&obstacle).sum()),position=list(a.shadow.hull.position),speed=float(a.shadow.hull.linearVelocity.length)))
   predictions[name]=samples
   overlay=cv2.cvtColor(np.uint8(frame*255),cv2.COLOR_GRAY2BGR)
   overlay[(sweep>0)&(road>0)]=(0,180,0);overlay[(sweep>0)&(road==0)]=(0,0,255);overlay[obstacle>0]=(255,0,0)
   cv2.imwrite(str(ROOT/f'{step:04d}-{name}-sweep.png'),overlay)
  rows.append(dict(step=step,components=components,action=saved['action'],nominal=saved['diagnostics']['obstacle_shield_nominal'],predictions=predictions))
 result=dict(scope='One audit replay, not an additional independent benchmark cell',components_note='OrangeRGB255/165/0 yieldsgray173/255; resize cannot make pureorangepeak>.71. Highpeaks indicatewhitecurb/otherbrightcontent, notcertifiedobjectlabels.',road_note='Thresholdroad+morphclose; currentcarfootprintocclusion treatedasinitialroad. Outsidefractions are pixelproxy, not groundtruth.',interventions=rows)
 benchmark=pathlib.Path('/tmp/apex-v2-obstacle-steering-r0/t3-s4111953688.jsonl');replay=ROOT/'t3-s4111953688.jsonl'
 if replay.exists():
  orig=list(map(json.loads,benchmark.read_text().splitlines()));audit=list(map(json.loads,replay.read_text().splitlines()));result['action_trace_parity']=len(orig)==len(audit) and all(x['action']==y['action'] for x,y in zip(orig,audit));result['state_trace_parity']=len(orig)==len(audit) and all(x['before']==y['before'] and x['after']==y['after'] for x,y in zip(orig,audit));result['steps']=len(audit)
 for name in ['t3-s4111953688.json','t3-s4111953688.jsonl','run.py']:
  p=ROOT/name
  if p.exists():result[name+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 output=pathlib.Path('agents/apex_2026/v2/results/obstacle-steering-audit.json');output.write_text(json.dumps(result,indent=2)+'\n')
 for r in rows:
  print(r['step'],r['components'])
  for name,samples in r['predictions'].items():print(name,'first4maxroadoutside',max(s['road_outside_fraction'] for s in samples[:4]),'horizonmax',max(s['road_outside_fraction'] for s in samples))
 print('parity',result.get('action_trace_parity'),result.get('state_trace_parity'),result.get('steps'))
if __name__=='__main__':main()
