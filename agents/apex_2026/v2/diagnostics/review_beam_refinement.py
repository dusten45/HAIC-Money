"""Instrument saved-observation R6 refinement; no real environment or reset."""
import ast,copy,hashlib,json,time
from pathlib import Path
import numpy as np
from agents.apex_2026.v2.beam_refined_agent import Agent
BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/'beam_refined_agent.py'
EXPECTED='cc0543c2bebbec0f2c1917a60b4e506cdd3da5bd562131e7090ae2bb55d65425'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert sha(SOURCE)==EXPECTED
 original_ast=ast.parse((BASE/'beam_robust_agent.py').read_text());refined_ast=ast.parse(SOURCE.read_text())
 # Strip only authorized instrumentation/refinement nodes, then compare all AST.
 original_ast.body=original_ast.body[1:];refined_ast.body=refined_ast.body[1:]
 for cls in refined_ast.body:
  if not isinstance(cls,ast.ClassDef) or cls.name!='Agent':continue
  cls.body=[node for node in cls.body if getattr(node,'name',None) not in ('_accept_refinement','_refine')]
  for method in cls.body:
   if getattr(method,'name',None)=='_advance':
    for node in ast.walk(method):
     if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':
      node.keywords=[kw for kw in node.keywords if kw.arg!='unsafe_count']
   if getattr(method,'name',None)=='act':
    method.body=[node for node in method.body if not (
       isinstance(node,ast.Assign) and (isinstance(node.targets[0],ast.Name) and node.targets[0].id=='original' or isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='_refine')
       or isinstance(node,ast.AugAssign) and isinstance(node.target,ast.Name) and node.target.id=='checked')]
    for node in ast.walk(method):
     if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':
      node.keywords=[kw for kw in node.keywords if kw.arg not in ('original_beam_cost','original_collision_count','refined_collision_count','refinement_shift','refinement_raw_ticks')]
 assert ast.dump(original_ast)==ast.dump(refined_ast),'Unauthorized non-refinement AST difference'
 fixtures=BASE/'results/beam_refinement_fixtures'
 observation=np.load(fixtures/'r5_stall_120_121.npz')['observations'][0]
 state=json.loads((fixtures/'r5_stall_120_121.json').read_text())['states'][0]
 agent=Agent()
 for key in ('throttle','last_steer','slip'):setattr(agent,key,state[key])
 original_refine=agent._refine
 report=dict(authorized_changes_only_ast_verified=True,source_sha256=EXPECTED,script_sha256=sha(__file__),resets=0,fixture_sha256={p.name:sha(p) for p in fixtures.iterdir() if p.is_file()})
 def audit(original,reference,arc,targets,free,initial):
  frozen=copy.deepcopy(original);sequence=copy.deepcopy(original['sequence']);start=copy.deepcopy(initial)
  advance=agent._advance;footprint=agent._footprint_cost;restore=agent.restore
  calls=[];collision=[];restores=[]
  def check_footprint(*args):
   count=footprint(*args);collision.append(count);return count
  def check_restore(shadow,record):
   restore(shadow,record)
   assert agent.snapshot(shadow)==record,'Restored state differs from input snapshot'
   restores.append(copy.deepcopy(record))
  def check_advance(node,action,ref,ar,tar,mask,terminal):
   i=len(calls);shift=i//8+1;depth=i%8
   if depth==0:
    assert node['state']==start and node['cost']==0 and node['unsafe_count']==0 and node['sequence']==[]
    assert node['last']==state['last_steer']
   intent=(sequence[depth+shift] if depth+shift<8 else (sequence[-1][0],0.,0.))
   pursuit,_=agent._pursuit(node,ref,ar)
   lattice=np.unique(np.clip(pursuit+np.array([-.14,0.,.14]),-.4,.4))
   expected=float(lattice[np.argmin(abs(lattice-intent[0]))])
   assert action==(expected,intent[1],intent[2]),'Illegal or incorrectly quantized proposal'
   assert terminal==(depth==7)
   before=len(collision);result=advance(node,action,ref,ar,tar,mask,terminal)
   assert len(collision)-before==4
   assert result['unsafe_count']==node['unsafe_count']+sum(collision[before:])
   calls.append(dict(shift=shift,depth=depth,cost=result['cost'],unsafe_count=result['unsafe_count']))
   return result
  agent._advance=check_advance;agent._footprint_cost=check_footprint;agent.restore=check_restore
  selected,shift,ticks=original_refine(original,reference,arc,targets,free,initial)
  agent._advance=advance;agent._footprint_cost=footprint;agent.restore=restore
  assert original==frozen and initial==start,'Original winner or initial snapshot mutated'
  assert len(calls)==56 and len(collision)==224 and len(restores)==56 and ticks==224
  eligible=[row for row in calls if row['depth']==7 and row['cost']<original['cost'] and row['unsafe_count']<=original['unsafe_count']]
  chosen=min(eligible,key=lambda row:row['cost']) if eligible else None
  assert shift==(chosen['shift'] if chosen else 0)
  assert selected['cost']==(chosen['cost'] if chosen else original['cost'])
  report.update(legal_lattice_actions=len(calls),exact_prefix_count_checks=len(calls),exact_snapshot_restore_checks=len(restores),extra_raw_ticks=len(collision),immutable_original_and_initial=True,selected_shift=shift,original_cost=original['cost'],selected_cost=selected['cost'],original_unsafe_count=original['unsafe_count'],selected_unsafe_count=selected['unsafe_count'],proposal_results=[row for row in calls if row['depth']==7])
  return selected,shift,ticks
 agent._refine=audit
 start=time.perf_counter();action=agent.act(observation)
 report['instrumented_act_seconds']=time.perf_counter()-start
 report['action']=action.tolist()
 assert agent.diagnostics['original_beam_cost']==state['expected_cost']
 assert sha(SOURCE)==EXPECTED
 report['limitations']=['Instrumented latency includes audit overhead; use separate source-bound preflight for resource assessment.','One saved stall observation exercises seven proposals; not a closed-loop performance claim.','Existing shadow snapshot/restore semantics are preserved; no physics model accuracy claim.']
 preflight_path=BASE/'results/beam-refined-r6-preflight.json'
 preflight=json.loads(preflight_path.read_text());assert preflight['source_sha256']==EXPECTED
 report['independent_lane_preflight']={'path':str(preflight_path),'sha256':sha(preflight_path),'cases':len(preflight['cases']),
     'rejected_full_action_memory_diagnostics_parity_cases':sum(row['unchanged_full_action_memory_diagnostics_parity'] for row in preflight['cases']),
     'maximum_act_seconds':preflight['maximum_act_s'],'timed_by':'rollout_agent'}
 report['limitations'].append('Saved-case maximum below5s does not guarantee full-episode worst-case latency; five required-frame probes use a repeated static image stack.')
 path=BASE/'results/beam-r6-independent-review.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
