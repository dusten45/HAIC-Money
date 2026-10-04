"""Read-only R5 isolation, mask and initial-footprint review; no track resets."""
import ast
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from agents.apex_2026.v2.beam_robust_agent import Agent

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'agents/apex_2026/v2'
SOURCE=BASE/'beam_robust_agent.py'
EXPECTED='4106f7ac579ed6a3f56b9389c919d5aa68a49061506600f67820ef28e6f93a57'
OBS=Path('/tmp/apex-v2-beam-r3/failure_capture/observations.npz')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    if sha(SOURCE)!=EXPECTED:raise ValueError('R5 source mismatch')
    old=ast.parse((BASE/'beam_area_agent.py').read_text())
    new=ast.parse(SOURCE.read_text())
    for cls in new.body:
        if isinstance(cls,ast.ClassDef) and cls.name=='Agent':
            cls.body=[node for node in cls.body if getattr(node,'name',None)!='_collision_mask']
            for method in cls.body:
                if getattr(method,'name',None)=='act':
                    method.body=[node for node in method.body if not (isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='collision_free')]
                    for node in ast.walk(method):
                        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='_search':
                            assert isinstance(node.args[3],ast.Name) and node.args[3].id=='collision_free'
                            node.args[3].id='free'
    assert ast.dump(old)==ast.dump(new),'Unapproved source change outside mask introduction'
    rng=np.random.default_rng(51383)
    for _ in range(100):
        raw=(rng.random((84,84))>.025).astype(np.uint8);before=raw.copy()
        padded=np.pad(raw,1,constant_values=0)
        oracle=np.minimum.reduce([padded[y:y+84,x:x+84] for y in range(3) for x in range(3)])
        actual=Agent._collision_mask(raw)
        assert np.array_equal(actual,oracle)
        assert np.array_equal(raw,before) and not np.shares_memory(raw,actual)
    with np.load(OBS,allow_pickle=False) as archive:observations=archive['observations']
    agent=Agent();nonzero=[]
    for index,obs in enumerate(observations):
        free,_,_=agent.road._free_space(obs[-1]);free[59:68,40:45]=1
        yaw,wheel=agent.road._hud_dynamics(obs[-1])
        agent.shadow.reset(0,-yaw,float(np.clip(-wheel,-.4,.4)))
        mask=agent._collision_mask(free)
        raw_cost=agent._footprint_cost(free,0,0,0)
        margin_cost=agent._footprint_cost(mask,0,0,0)
        if raw_cost or margin_cost:nonzero.append(dict(step=index+120,raw_cost=raw_cost,margin_cost=margin_cost))
    report=dict(source_sha256=EXPECTED,base_sha256=sha(BASE/'beam_area_agent.py'),review_script_sha256=sha(__file__),observation_sha256=sha(OBS),resets=0,
        ast_identical_after_removing_only_mask_helper_and_two_act_changes=True,
        independent_three_by_three_minimum_oracle_cases=100,mask_mutation=False,
        initial_pose_cases=41,nonzero_initial_pose_costs=nonzero,
        ordering='path/geodesic and speed targets use original free mask; current occlusion fill precedes erosion; only search collision mask changes',
        boundary='Explicit zero border; outside-image remains unsafe. One Chebyshev pixel margin: approximately .735m horizontal and .588m forward at fixed image scale.',
        cache='New independent array per action, never mutated during search; identity-based integral cache contract preserved.',
        limits=['One pixel is an empirical raster-resolution margin, not a proven pose or segmentation error bound.',
               '20ms sampling still leaves between-sample motion unchecked.',
               'Postimpact steps144-148 already violate the eroded margin at the current pose; no feasible-continuation guarantee.',
               'Agent.act timing supplied by separately source-bound shadow preflight; reviewer did not rerun it.',
               'No track episodes or policy inputs from privileged evaluator telemetry.'])
    preflight_path=Path('/tmp/apex-v2-beam-r5/preflight.json')
    preflight=json.loads(preflight_path.read_text())
    assert preflight['source_sha256']==EXPECTED
    report['stored_observation_preflight']={'path':str(preflight_path),'sha256':sha(preflight_path),'timed_by':'shadow_physics',
        'count':len(preflight['results']),'maximum_act_seconds':max(row['act_s'] for row in preflight['results']),
        'max_latency_step':max(preflight['results'],key=lambda row:row['act_s'])['step'],
        'selected_sequence_nonzero_unsafe':[{key:row[key] for key in ('step','selected_sequence_unsafe')} for row in preflight['results'] if row['selected_sequence_unsafe']],
        'scope':'Saved R3 observations with reconstructed R3 action memory; individual calls, not closed-loop R5 episodes.'}
    report['limits'].append('Measured21calls below5seconds do not guarantee worst-case latency; more eroded pixels invoke costly SAT more often.')
    assert sha(SOURCE)==EXPECTED
    dest=BASE/'results/beam-r5-independent-review.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
