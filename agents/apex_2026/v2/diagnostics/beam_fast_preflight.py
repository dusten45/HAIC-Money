"""No-reset FAST saved-pixel exactness and paired latency preflight."""
import importlib.util
import json
import time
from pathlib import Path
import cv2
import numpy as np
from agents.apex_2026.v2.diagnostics.beam_procrastination import carried_state,sha

RUNTIME=Path('agents/apex_2026/v2/beam_fast_agent.py').resolve()
BASELINE=Path('agents/apex_2026/v2/beam_refined_agent.py').resolve()
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
    new_fields={'transition_cache_hits','transition_cache_misses','transition_cache_entries',
                'memo_skipped_raw_ticks','actual_search_raw_ticks'}
    for name,observation,rows,step in cases:
        left=original.Agent();right=refined.Agent()
        if rows is not None:
            carried_state(left,rows,step);carried_state(right,rows,step)
        started=time.perf_counter();base_action=left.act(observation);base_time=time.perf_counter()-started
        started=time.perf_counter();action=right.act(observation);latency=time.perf_counter()-started
        d=right.diagnostics;old=left.diagnostics
        assert np.array_equal(action,base_action)
        assert (right.slip,right.throttle,right.last_steer)==(left.slip,left.throttle,left.last_steer)
        assert right.snapshot(right.shadow)==left.snapshot(left.shadow)
        assert {k:v for k,v in d.items() if k not in new_fields}==old
        assert d['actual_search_raw_ticks']+d['memo_skipped_raw_ticks']==old['collision_checked_raw_ticks']
        record=dict(case=name,action=action.tolist(),cost=d['beam_cost'],
            shift=d['refinement_shift'],baseline_act_s=base_time,act_s=latency,
            speedup=base_time/latency,within_5s=latency<5.,
            exact_action_carry_snapshot_legacy_diagnostics=True,
            cache={k:d[k] for k in sorted(new_fields)},
            initialization='public stack plus prior float32 actions/predicted slip' if rows is not None else 'repeated single public PNG, zero action memory; synthetic preflight only')
        records.append(record)
        print(name,'shift',record['shift'],'cost',record['cost'],'seconds',round(latency,3),flush=True)
    class AuditedFast(refined.Agent):
        # Diagnostic instrumentation only: records whether refinement reuses a
        # prefix cached by search. All calls delegate to unchanged runtime.
        def _refine(self,*args):
            self.search_keys=frozenset(self._transition_cache)
            self.cross_phase_hits=0
            self.in_refinement=True
            try:
                return super()._refine(*args)
            finally:
                self.in_refinement=False

        def _advance(self,node,action,reference,arc,targets,free,terminal):
            key=(tuple(tuple(float(v).hex() for v in a)
                       for a in node['sequence']+[action]),bool(terminal))
            cross_hit=(getattr(self,'in_refinement',False)
                       and key in self.search_keys and key in self._transition_cache)
            result=super()._advance(node,action,reference,arc,targets,free,terminal)
            if cross_hit:self.cross_phase_hits+=1
            return result

    left=original.Agent();right=AuditedFast()
    first=cases[0]
    carried_state(left,first[2],first[3]);carried_state(right,first[2],first[3])
    sequential=[]
    for name,observation,rows,step in cases[:2]:
        # Step121 deliberately receives step120 policy memory; no trace-based
        # reinitialization occurs here. These fixed recorded observations are
        # a parity test, not a counterfactual closed-loop driving trajectory.
        previous_cache=right._transition_cache
        base_action=left.act(observation);action=right.act(observation)
        assert right._transition_cache is not previous_cache
        assert np.array_equal(action,base_action)
        assert (right.slip,right.throttle,right.last_steer)==(left.slip,left.throttle,left.last_steer)
        assert right.snapshot(right.shadow)==left.snapshot(left.shadow)
        assert {k:v for k,v in right.diagnostics.items() if k not in new_fields}==left.diagnostics
        sequential.append(dict(step=step,action=action.tolist(),
            sequence=right.diagnostics['beam_sequence'],cost=right.diagnostics['beam_cost'],
            unsafe_count=right.diagnostics['refined_collision_count'],
            throttle=right.throttle,slip=right.slip,
            cross_phase_cache_hits=right.cross_phase_hits,
            all_action_carry_physical_snapshot_legacy_diagnostics_exact=True,
            cache_replaced_at_act_entry=True))
    assert sum(r['cross_phase_cache_hits'] for r in sequential)>0
    assert inventory=={p:sha(Path(p)) for p in inventory}
    result=dict(scope='12 independent saved-pixel contexts plus sequential120→121; zero environment resets, no driving performance claim.',
        cases=records,sequential_cases=sequential,
        prefix_contract='Cache fixed to initial state/reference/targets/mask within one act only; cleared before every observation including invalid ones.',
        source_sha256=sha(RUNTIME),source_dependency_inventory=inventory,input_sha256=inputs,
        observer_objective_collision_margin_unchanged=True,
        state_scope='Physical snapshot/controller state; unused fuel-spent/wheel-phase counters not claimed equal.',
        all_exact=True,
        maximum_act_s=max(r['act_s'] for r in records),
        no_runtime_source_drift=True)
    Path('agents/apex_2026/v2/results/beam-fast-r7-preflight.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
