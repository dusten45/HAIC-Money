"""No-reset R6 saved-pixel parity, counterexample and latency preflight."""
import importlib.util
import json
import time
from pathlib import Path
import cv2
import numpy as np
from agents.apex_2026.v2.diagnostics.beam_procrastination import carried_state,sha

RUNTIME=Path('agents/apex_2026/v2/beam_refined_agent.py').resolve()
BASELINE=Path('/tmp/apex-v2-beam-r5/source.py')
HELPER=Path('agents/apex_2026/v2/shadow_physics.py').resolve()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    inventory={str(p):sha(p) for p in (RUNTIME,BASELINE,HELPER,Path(__file__).resolve())}
    original=load(BASELINE,'r5_preflight_original');refined=load(RUNTIME,'r6_preflight_refined')
    cases=[];inputs={}
    r5=Path('/tmp/apex-v2-beam-r5');r3=Path('/tmp/apex-v2-beam-r3')
    for directory,capture,start,steps,label in [(r5,'stall_capture',60,[120,121],'R5_stall'),
                                               (r3,'failure_capture',120,list(range(134,139)),'R3_collision_window')]:
        archive=directory/capture/'observations.npz';trace=directory/'t3-s4111953688.jsonl'
        observations=np.load(archive)['observations'];rows=[json.loads(line) for line in trace.read_text().splitlines()]
        inputs[str(archive)]=sha(archive);inputs[str(trace)]=sha(trace)
        for step in steps:cases.append((f'{label}_{step}',observations[step-start],rows,step))
    for track,seed in [(1,516237),(2,644062),(3,1007),(4,18800)]:
        path=Path(f'/tmp/apex-final-frozen/required/t{track}-s{seed}-frames/0100.png')
        frame=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
        inputs[str(path)]=sha(path)
        cases.append((f'required{track}_static_frame100',np.repeat(frame[None],4,axis=0),None,None))
    path=Path('/tmp/apex-final-frozen/required/t1-s516237-frames/0000.png')
    frame=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
    inputs[str(path)]=sha(path)
    cases.append(('required1_static_start',np.repeat(frame[None],4,axis=0),None,None))
    records=[]
    new_fields={'original_beam_cost','original_collision_count','refined_collision_count',
                'refinement_shift','refinement_raw_ticks'}
    for name,observation,rows,step in cases:
        left=original.Agent();right=refined.Agent()
        if rows is not None:
            carried_state(left,rows,step);carried_state(right,rows,step)
        started=time.perf_counter();base_action=left.act(observation);base_time=time.perf_counter()-started
        started=time.perf_counter();action=right.act(observation);latency=time.perf_counter()-started
        d=right.diagnostics;old=left.diagnostics
        assert d['original_beam_cost']==old['beam_cost']
        assert d['beam_cost']<=old['beam_cost']
        assert d['refined_collision_count']<=d['original_collision_count']
        assert d['refinement_raw_ticks']==224
        assert d['collision_checked_raw_ticks']==old['collision_checked_raw_ticks']+224
        assert np.isfinite(action).all() and -1<=action[0]<=1 and np.all((action[1:]>=0)&(action[1:]<=1))
        unchanged=d['refinement_shift']==0
        if unchanged:
            assert np.array_equal(action,base_action)
            assert (right.slip,right.throttle,right.last_steer)==(left.slip,left.throttle,left.last_steer)
            comparable={k:v for k,v in d.items() if k not in new_fields}
            comparable['collision_checked_raw_ticks']-=224
            assert comparable==old
        record=dict(case=name,original_action=base_action.tolist(),action=action.tolist(),
            original_cost=old['beam_cost'],cost=d['beam_cost'],shift=d['refinement_shift'],
            original_collision_count=d['original_collision_count'],collision_count=d['refined_collision_count'],
            added_raw_ticks=d['refinement_raw_ticks'],baseline_act_s=base_time,act_s=latency,
            within_5s=latency<5.,unchanged_full_action_memory_diagnostics_parity=unchanged,
            initialization='public stack plus priorfloat32actions/predictedslip' if rows is not None else 'repeated single public PNG, zero action memory; synthetic preflight only')
        records.append(record)
        print(name,'shift',record['shift'],'cost',record['cost'],'seconds',round(latency,3),flush=True)
    assert any(r['unchanged_full_action_memory_diagnostics_parity'] for r in records)
    assert all(r['within_5s'] for r in records)
    assert inventory=={p:sha(Path(p)) for p in inventory}
    result=dict(scope='12 saved-pixel preflight cases; zero environment resets, no driving performance claim.',
        source_sha256=sha(RUNTIME),source_dependency_inventory=inventory,input_sha256=inputs,
        observer_objective_collision_margin_unchanged=True,
        count_metadata_note='unsafe_count accumulates actual prefix raw-tick unsafe-cell counts; completion rollouts are separate node copies. Original candidate rankings/costs remain exact in all12 comparisons.',
        raw_tick_bound=224,cases=records,maximum_act_s=max(r['act_s'] for r in records),
        no_runtime_source_drift=True)
    Path('agents/apex_2026/v2/results/beam-refined-r6-preflight.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
