"""Small source-bound packaging parity preflight; no environment resets."""
import ast,hashlib,importlib.util,json,os,pathlib,sys,time,io,tokenize
import cv2
import numpy as np
from agents.apex_2026.v2.diagnostics.build_beam_package import (build,DEFAULT_BEAM,DEFAULT_SHADOW,BEAM_SHA,SHADOW_SHA,remove_project_import,rename_shadow_road)

OUT=pathlib.Path('/tmp/apex-beam-standalone-preflight')
RESULT=pathlib.Path('agents/apex_2026/v2/results/beam-package-preflight.json')

def digest(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
 cv2.setNumThreads(1)
 manifest=build(DEFAULT_BEAM,DEFAULT_SHADOW,OUT)
 # Bind the archived original to its archived helper, not a mutable project file.
 shadow=load(DEFAULT_SHADOW,'_packaging_original_shadow')
 key='agents.apex_2026.v2.shadow_physics';prior=sys.modules.get(key);sys.modules[key]=shadow
 try:original=load(DEFAULT_BEAM,'_packaging_original_beam')
 finally:
  if prior is None:sys.modules.pop(key,None)
  else:sys.modules[key]=prior
 package=load(OUT/'agent.py','_packaging_standalone')
 source=(OUT/'agent.py').read_text();tree=ast.parse(source)
 expected=ast.parse(rename_shadow_road(DEFAULT_SHADOW.read_text())).body+ast.parse(remove_project_import(DEFAULT_BEAM.read_text())).body
 ast_equal=ast.dump(ast.Module(body=expected,type_ignores=[]),include_attributes=False)==ast.dump(tree,include_attributes=False)
 numbers=lambda s:[t.string for t in tokenize.generate_tokens(io.StringIO(s).readline) if t.type==tokenize.NUMBER]
 numbers_equal=numbers(source)==numbers(DEFAULT_SHADOW.read_text())+numbers(DEFAULT_BEAM.read_text())
 assert ast_equal and numbers_equal
 rows=[]
 def sequence(name,observations):
  left=original.Agent();right=package.Agent()
  assert left.shadow.wheels[0].tiles and next(iter(right.shadow.wheels[0].tiles)).__class__.__name__=='_ShadowRoad'
  assert right.road.__class__.__name__=='_Road'
  for i,obs in enumerate(observations):
   before=obs.copy();cv2.setRNGSeed(0);t=time.perf_counter();a=left.act(obs.copy());lt=time.perf_counter()-t
   cv2.setRNGSeed(0);t=time.perf_counter();b=right.act(obs.copy());rt=time.perf_counter()-t
   row=dict(sequence=name,index=i,action_original=a.tolist(),action_packaged=b.tolist(),action_bitwise=np.array_equal(a.view(np.uint32),b.view(np.uint32)),diagnostics_exact=left.diagnostics==right.diagnostics,state_exact=(left.throttle,left.last_steer,left.slip)==(right.throttle,right.last_steer,right.slip),observation_unchanged=np.array_equal(obs,before),original_wall_s=lt,package_wall_s=rt)
   assert row['action_bitwise'] and row['diagnostics_exact'] and row['state_exact'] and row['observation_unchanged'],row
   rows.append(row);print(name,i,'action/state/sequence/cost exact',flush=True)
 pure=np.full((4,84,84),.4,np.float32);pure[:,:74,:25]=.6;pure[:,:74,59:]=.6;pure[:,74:]=0
 sequence('pure_straight_from_rest',[pure])
 paths=[pathlib.Path('/tmp/apex-v2-obstacle-steering-audit')/f'{step:04d}-observation.npy' for step in (67,68)]
 sequence('saved_public_sequential_67_68',[np.load(p) for p in paths])
 result=dict(scope='Packagingpreflightonly; noselection,freeze,holdoutallocation orsimulatorreset.',environment_resets=0,beam_sha256=digest(DEFAULT_BEAM),shadow_sha256=digest(DEFAULT_SHADOW),package_manifest=manifest,package_path=str(OUT/'agent.py'),ast_equal_except_authorized_bindings=True,numeric_tokens_identical=True,mit_notice_preserved='MIT License' in source,rows=rows,observation_hashes={str(p):digest(p) for p in paths},nice=os.nice(0),openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),opencv_threads=cv2.getNumThreads(),limitations=['Threeobservations only; exactparityatthissubsetdoesnot replacefullcandidateevaluation.','Savedobservations are replayedsequentially fromfreshagent,not a newenvironmentepisode.','Walltimesarepreflightmeasurements,not a hardresourceeligibility guarantee.'])
 assert result['beam_sha256']==BEAM_SHA and result['shadow_sha256']==SHADOW_SHA
 RESULT.write_text(json.dumps(result,indent=2)+'\n')
 print('Packagingparity passed; originalruntimefiles untouched.')
if __name__=='__main__':main()
