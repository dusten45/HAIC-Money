"""Read-only one-public-observation beam runtime audit; never resets a track."""
import cProfile,pstats,time,json,pathlib,hashlib,os,io,math
import numpy as np
from agents.apex_2026.v2.beam_agent import Agent

ROOT=pathlib.Path('/tmp/apex-beam-profile')
OBS=pathlib.Path('/tmp/apex-v2-obstacle-steering-audit/0068-observation.npy')
SOURCE=pathlib.Path('agents/apex_2026/v2/beam_agent.py')

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def projection_probe(report):
 path=np.array(report['diagnostics']['path']);ref=np.vstack(([0.,0.],(path-[42.,63.])/[1.3608,-1.701]));arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(ref,axis=0),axis=1))]
 start=ref[:-1];delta=np.diff(ref,axis=0);den=np.maximum(np.sum(delta*delta,axis=1),1e-8);ds=np.diff(arc);headings=np.array([math.atan2(q[0],q[1]) for q in delta])
 def cached(pos,previous,speed):
  fraction=np.clip(np.sum((pos-start)*delta,axis=1)/den,0,1)
  projected=start+fraction[:,None]*delta
  distance=np.sum((pos-projected)**2,axis=1)
  progress=arc[:-1]+fraction*ds
  valid=(progress>=max(0,previous-3.))&(progress<=previous+.08*speed+4.)
  distance=np.where(valid,distance,np.inf);i=int(np.argmin(distance))
  return float(progress[i]),float(np.sqrt(distance[i])),float(headings[i])
 rng=np.random.default_rng(7103);cases=[(rng.normal([0,15],[8,12]),float(rng.uniform(0,arc[-1])),float(rng.uniform(0,100))) for _ in range(2000)]
 a=[Agent._projection(x,ref,arc,s,v) for x,s,v in cases];b=[cached(x,s,v) for x,s,v in cases]
 parity=all(np.array_equal(np.array(x).view(np.uint64),np.array(y).view(np.uint64)) for x,y in zip(a,b));timings={}
 for name,fn in [('original',lambda x,s,v:Agent._projection(x,ref,arc,s,v)),('cached',cached)]:
  trials=[]
  for _ in range(5):
   t=time.process_time()
   for x,s,v in cases:fn(x,s,v)
   trials.append(time.process_time()-t)
  timings[name]=trials
 report['projection_cache_probe']=dict(cases=len(cases),bitwise_parity=parity,process_cpu_seconds=timings,median_speedup=float(np.median(timings['original'])/np.median(timings['cached'])),scope='Staticgeometrycache only; no wholeagentactionparity claim. Cases synthetically sample projectioninputs, not simulator episodes.')


def footprint_probe(report):
 a=Agent();obs=np.load(report['observation']);free,_,_=a.road._free_space(obs[-1]);free[59:68,40:45]=1
 rot=np.empty((2,2));points=np.empty((25,2));shift=np.empty(2);xf=np.empty(25);yf=np.empty(25);xi=np.empty(25,dtype=int);yi=np.empty(25,dtype=int);safe=np.empty(25,dtype=bool)
 def buffers(x,y,h):
  c,s=math.cos(h),math.sin(h);rot[0,0]=c;rot[0,1]=s;rot[1,0]=-s;rot[1,1]=c
  np.matmul(a.FOOTPRINT,rot,out=points);shift[0]=x;shift[1]=y;np.add(points,shift,out=points)
  np.multiply(points[:,0],1.3608,out=xf);np.add(xf,42,out=xf);np.rint(xf,out=xf);np.copyto(xi,xf,casting='unsafe')
  np.multiply(points[:,1],1.701,out=yf);np.subtract(63,yf,out=yf);np.rint(yf,out=yf);np.copyto(yi,yf,casting='unsafe')
  inside=(xi>=0)&(xi<84)&(yi>=0)&(yi<74);safe.fill(False);safe[inside]=free[yi[inside],xi[inside]]>0
  return int(np.count_nonzero(~safe))
 rng=np.random.default_rng(8103);cases=[(float(rng.uniform(-40,40)),float(rng.uniform(-10,45)),float(rng.uniform(-math.pi,math.pi))) for _ in range(2000)]
 parity=all(a._footprint_cost(free,*v)==buffers(*v) for v in cases);timings={}
 for name,fn in [('original',lambda x,y,h:a._footprint_cost(free,x,y,h)),('buffers',buffers)]:
  ts=[]
  for _ in range(5):
   t=time.process_time()
   for v in cases:fn(*v)
   ts.append(time.process_time()-t)
  timings[name]=ts
 report['footprint_buffer_probe']=dict(cases=len(cases),integer_cost_parity=parity,process_cpu_seconds=timings,median_speedup=float(np.median(timings['original'])/np.median(timings['buffers'])),scope='Same np.matmul and orderedcoordinateoperations, reusesstorage; notwholeagentactionparity.')

def main():
 ROOT.mkdir(exist_ok=True)
 before=sha(SOURCE);obs=np.load(OBS);agent=Agent()
 start=time.perf_counter();cpu=time.process_time();profile=cProfile.Profile();profile.enable()
 action=agent.act(obs.copy())
 profile.disable();wall=time.perf_counter()-start;cpu=time.process_time()-cpu
 profile.dump_stats(str(ROOT/'beam.prof'))
 stream=io.StringIO();stats=pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumulative');stats.print_stats(35)
 (ROOT/'profile.txt').write_text(stream.getvalue())
 rows=[]
 for key,value in stats.stats.items():
  cc,nc,tt,ct,callers=value
  rows.append(dict(file=key[0],line=key[1],function=key[2],primitive_calls=cc,calls=nc,self_s=tt,cumulative_s=ct))
 result=dict(scope='One savedpublic4x84x84observation, freshAgent state, no environment/trackreset. Profile context differs from realbeamtrajectory.',source_sha256_before=before,source_sha256_after=sha(SOURCE),observation=str(OBS),observation_sha256=sha(OBS),nice=os.nice(0),openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),wall_s=wall,process_cpu_s=cpu,profiled_action=action.tolist(),diagnostics=agent.diagnostics,functions=sorted(rows,key=lambda r:r['cumulative_s'],reverse=True)[:50],profile_artifact=str(ROOT/'beam.prof'),profile_text=str(ROOT/'profile.txt'))
 assert result['source_sha256_before']==result['source_sha256_after']
 projection_probe(result)
 footprint_probe(result)
 p=pathlib.Path('agents/apex_2026/v2/results/beam-profile.json');p.write_text(json.dumps(result,indent=2)+'\n')
 print(stream.getvalue());print('wall',wall,'cpu',cpu,'action',action)
if __name__=='__main__':main()
