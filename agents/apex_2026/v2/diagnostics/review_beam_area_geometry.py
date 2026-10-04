"""Independent saved-geometry review. No environment imports, steps or resets.

Run from the repository root with PYTHONPATH=. and single-threaded BLAS.
"""
import ast,json,math,hashlib,importlib.util,time
from pathlib import Path
import cv2,numpy as np
source=Path('agents/apex_2026/v2/beam_area_agent.py')
before=hashlib.sha256(source.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('review_area',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=m.Agent(); rng=np.random.default_rng(134138)
results=[]
for k in range(120):
 h=float(rng.uniform(-math.pi,math.pi));p=np.array([rng.uniform(-33,33),rng.uniform(-8,39)])
 for body in [a.shadow.hull]+a.shadow.wheels:
  body.position=tuple(p+rng.uniform(-2,2,2));body.angle=h+float(rng.uniform(-.4,.4))
 free=(rng.uniform(0,1,(84,84))>.06).astype(np.uint8)
 expected=set()
 for body in [a.shadow.hull]+a.shadow.wheels:
  for fixture in body.fixtures:
   poly=np.array([list(body.GetWorldPoint(v)) for v in fixture.shape.vertices],np.float32)*[1.3608,-1.701]+[42,63]
   lo=np.ceil(poly.min(axis=0)-.5).astype(int);hi=np.floor(poly.max(axis=0)+.5).astype(int)
   for y in range(lo[1],hi[1]+1):
    for x in range(lo[0],hi[0]+1):
     if 0<=x<84 and 0<=y<74 and free[y,x]:continue
     cell=np.array([[x-.5,y-.5],[x+.5,y-.5],[x+.5,y+.5],[x-.5,y+.5]],np.float32)
     area,_=cv2.intersectConvexConvex(poly.astype(np.float32),cell)
     if area>1e-7:expected.add((x,y))
 got=a._footprint_cost(free,*p,h)
 if got!=len(expected):results.append(dict(case=k,actual=got,oracle=len(expected)))
# API cache mutability probe; normal act supplies a fresh free mask each time.
a=m.Agent();a.shadow.reset(0,0,0);free=np.ones((84,84),np.uint8)
initial=a._footprint_cost(free,0,0,0);free[64,42]=0
cached=a._footprint_cost(free,0,0,0);fresh=a._footprint_cost(free.copy(),0,0,0)
report=dict(source_sha256=before,geometry_cases=120,oracle='Box2D GetWorldPoint plus independent OpenCV convex intersection, positive area >1e-7 px^2; random nontangent cases',mismatches=results,cache_probe=dict(clean=initial,same_object_mutated=cached,fresh_copy=fresh),source_unchanged=before==hashlib.sha256(source.read_bytes()).hexdigest(),resets=0)
base_ast=ast.parse(Path('agents/apex_2026/v2/beam_agent.py').read_text())
new_ast=ast.parse(source.read_text())
for tree in (base_ast,new_ast):
 for item in tree.body:
  if isinstance(item,ast.ClassDef) and item.name=='Agent':
   item.body=[method for method in item.body if getattr(method,'name',None) not in ('__init__','_footprint_cost')]
report['all_other_ast_nodes_identical']=ast.dump(base_ast)==ast.dump(new_ast)
micro=[]
for label,mask in [('all_safe',np.ones((84,84),np.uint8)),('all_unsafe',np.zeros((84,84),np.uint8))]:
 probe=m.Agent();probe.shadow.reset(40,0,0);probe._footprint_cost(mask,0,0,0)
 start=time.process_time()
 for _ in range(2000):value=probe._footprint_cost(mask,0,0,0)
 micro.append(dict(mask=label,calls=2000,process_cpu_per_call_us=(time.process_time()-start)/2000*1e6,collision_cost=value))
report['checker_microbenchmark']=micro
preflight_path=Path('/tmp/apex-v2-beam-r4/preflight.json')
preflight=json.loads(preflight_path.read_text())
if preflight['source_sha256']!=before:raise RuntimeError('Preflight source mismatch')
report['stored_observation_preflight']={'path':str(preflight_path),'sha256':hashlib.sha256(preflight_path.read_bytes()).hexdigest(),'timed_by':'shadow_physics','steps':[row['step'] for row in preflight['results']],'max_act_seconds':max(row['act_s'] for row in preflight['results'])}
report['limitations']=['Random oracle cases exclude exact tangent contacts; SAT intentionally counts boundary touches conservatively.','Pixel-cell occupied area approximates camera geometry, not physical obstacle silhouettes.','Checks occur every20ms; no continuous collision certificate between raw ticks.','Integral fast path requires free mask immutable during each search; current act satisfies this contract.','Six stored-observation latencies are below5s but do not bound worst-case episode or machine-load latency.','All-unsafe checker microbenchmark is a synthetic stress case, not a measured agent.act or episode.','Penalty is now unique unsafe pixel-cell count, not sparse sample count; coefficient200 and all other objective/search AST remain unchanged.']
report['review_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
Path('agents/apex_2026/v2/results/beam-area-geometry-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
if results or not report['source_unchanged'] or not report['all_other_ast_nodes_identical']:
 raise RuntimeError('Geometry or isolation review failed')
