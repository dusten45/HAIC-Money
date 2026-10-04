"""R5 stationary objective/search audit using an existing pixel replay capture.

No environment is created. Agent inputs use only captured pixels and prior
float32 actions/predicted slip. Alternative plans are internal model predictions,
not observed driving outcomes or proof of an actual safe escape.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np
from agents.apex_2026.v2.diagnostics.beam_procrastination import (
    carried_state,initialize,score_sequence,sha)

BASE=Path('/tmp/apex-v2-beam-r5')
SOURCE=BASE/'source.py'
TRACE=BASE/'t3-s4111953688.jsonl'
CAPTURE=BASE/'stall_capture/observations.npz'


def quantize_to_lattice(agent,desired,reference,arc,targets,free,initial):
    node=dict(cost=0.,progress=0.,state=initial,sequence=[],last=agent.last_steer)
    sequence=[]
    for depth,action in enumerate(desired):
        pursuit,_=agent._pursuit(node,reference,arc)
        steers=np.unique(np.clip(pursuit+np.array([-.14,0.,.14]),-.4,.4))
        steer=float(steers[np.argmin(abs(steers-action[0]))])
        candidate=(steer,action[1],action[2]);sequence.append(candidate)
        node=agent._advance(node,candidate,reference,arc,targets,free,depth==agent.DEPTH-1)
    return sequence


def pruning_witness(agent,sequence,reference,arc,targets,free,initial):
    beam=[dict(cost=0.,progress=0.,state=initial,sequence=[],last=agent.last_steer)]
    result=[];completion=None
    for depth in range(agent.DEPTH):
        children=[]
        for node in beam:
            pursuit,_=agent._pursuit(node,reference,arc)
            for steer in np.unique(np.clip(pursuit+np.array([-.14,0.,.14]),-.4,.4)):
                for gas,brake in ((.7,0.),(0.,0.),(0.,.65)):
                    child=agent._advance(node,(float(steer),gas,brake),reference,arc,targets,free,depth==agent.DEPTH-1)
                    child['priority']=agent._completion(child,reference,arc,targets,free,depth)
                    children.append(child)
        children.sort(key=lambda n:n['priority'])
        matches=[(i+1,n) for i,n in enumerate(children)
                 if all(np.array_equal(a,b) for a,b in zip(n['sequence'],sequence[:depth+1]))]
        assert len(matches)==1
        rank,node=matches[0]
        result.append(dict(depth=depth,prefix_rank=rank,children_count=len(children),
            prefix_actual_cost=node['cost'],completion_priority=node['priority'],
            retained_cutoff=children[agent.WIDTH-1]['priority']))
        if rank>agent.WIDTH:
            tail=node
            for future in range(depth+1,agent.DEPTH):
                steer,speed=agent._pursuit(tail,reference,arc)
                target=float(np.interp(tail['progress'],arc,targets))
                gas,brake=((0.,.65) if speed>target+1 else (.7,0.) if speed<target-3 else (0.,0.))
                tail=agent._advance(tail,(float(np.clip(steer,-.4,.4)),gas,brake),reference,arc,targets,free,future==agent.DEPTH-1)
            completion=score_sequence(agent,tail['sequence'],reference,arc,targets,free,initial)
            assert abs(completion['total_cost']-node['priority'])<1e-7
            break
        beam=children[:agent.WIDTH]
    return result,completion


def objective_alternatives(score):
    geometry=score['terminal_components']['cross']+score['terminal_components']['heading']
    # Normalize by N so an immediately attained constant progress s has the
    # same -4s reward scale. This changes timing preference, not collision cost.
    progress_mean=sum(s['progress'] for s in score['samples'])/len(score['samples'])
    return dict(original=score['total_cost'],
        remove_terminal_cross_heading_only=score['total_cost']-geometry,
        stagewise_normalized_progress=score['total_cost']-score['progress_reward']-4*progress_mean)


def main():
    assert sha(SOURCE)=='4106f7ac579ed6a3f56b9389c919d5aa68a49061506600f67820ef28e6f93a57'
    helper=Path('agents/apex_2026/v2/shadow_physics.py').resolve()
    assert sha(helper)=='8d5c21ce445bda2f192e4815739253ebf46d1099f635d966c12fe9e877990a56'
    capture_receipt=json.loads((BASE/'stall_capture/receipt.json').read_text())
    assert capture_receipt['captured_exact']
    assert capture_receipt['original_trace_sha256']==sha(TRACE)
    spec=importlib.util.spec_from_file_location('r5_objective_frozen',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[json.loads(line) for line in TRACE.read_text().splitlines()]
    observations=np.load(CAPTURE)['observations'];parity=[]
    for step in (120,121):
        agent=module.Agent();carry=carried_state(agent,rows,step)
        actual=agent.act(observations[step-60]);expected=rows[step-1]
        assert np.array_equal(actual,np.asarray(expected['action'],np.float32))
        assert agent.diagnostics['beam_cost']==expected['policy_diagnostics']['beam_cost']
        parity.append(dict(step=step,carried_state=carry,action=actual.tolist(),
            beam_cost=agent.diagnostics['beam_cost'],exact_action_and_cost_match=True))
    agent=module.Agent();carried_state(agent,rows,120)
    reference,arc,targets,free,initial,initialized=initialize(agent,module,observations[60])
    free=agent._collision_mask(free)
    initialized['one_pixel_margin_initial_unsafe']=agent._footprint_cost(free,0.,0.,0.)
    selected=rows[119]['policy_diagnostics']['beam_sequence']
    variants=dict(selected=selected,gas_now=[[selected[0][0],.7,0.]]+selected[1:],
        coast_all=[[selected[0][0],0.,0.]]*8,
        left_shift_coast_tail=selected[2:]+[[selected[-1][0],0.,0.]]*2,
        left_shift_brake_tail=selected[2:]+[[selected[-1][0],0.,.65]]*2)
    variants['lattice_early_acceleration']=quantize_to_lattice(agent,variants['left_shift_coast_tail'],reference,arc,targets,free,initial)
    scores={name:score_sequence(agent,sequence,reference,arc,targets,free,initial) for name,sequence in variants.items()}
    for score in scores.values():
        score['objective_alternatives']=objective_alternatives(score)
        points=np.array([[0.,0.]]+[s['position'] for s in score['samples']])
        lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
        jumps=np.array([s['delta_progress'] for s in score['samples']])-lengths
        score['physical_chord_distance_m']=float(lengths.sum())
        score['max_progress_increment_minus_chord_m']=float(jumps.max())
    pruning,completion=pruning_witness(agent,variants['lattice_early_acceleration'],reference,arc,targets,free,initial)
    result=dict(scope='Read-only R5 step120/121 captured-pixel audit; no resets, runtime edits or policy-driving evaluations.',
        provenance={str(p):sha(p) for p in (SOURCE,TRACE,CAPTURE,BASE/'stall_capture/receipt.json',helper,
            Path('agents/apex_2026/v2/diagnostics/beam_procrastination.py'),Path(__file__))},
        parity=parity,initialized_pixel_state=initialized,sequence_scores=scores,
        lattice_counterexample_sequence=variants['lattice_early_acceleration'],
        pruning_witness=pruning,rejected_prefix_feedback_completion=completion,
        objective_alternative_definition={
          'remove_terminal_cross_heading_only':'Subtract terminal4*cross^2+5*heading_error^2; retain terminal speed penalty and every stage/steering/collision cost.',
          'stagewise_normalized_progress':'Replace sum(-4*delta_progress) with -(4/N)*sum(progress_at_stage_end); retain every other original cost. One specified alternative, no gain sweep.'},
        findings=[
          'Terminal geometry is negligible for the selected R5 plan; removing it changes cost by about0.060, unlike the earlier R3 stall.',
          'Selected plan projection progresses approximately with physical displacement; no large arclength jump explains the reward.',
          'An immediate-gas substitution touches one unsafe pixel-cell sample and is expensive; indiscriminate gas is not justified.',
          'A different early-acceleration sequence admissible to the exact steering/pedal lattice has zero predicted unsafe full-area+margin samples and lower original cost than the selected plan.',
          'That better sequence prefix is pruned at depth1 because the feedback completion predicts collisions. The heuristic is not a lower bound on optimized completion cost.',
          'The pixel initialization predicts0.322m progress for the first coast despite the real car being nearly stationary. Decoder/kinetic-state bias is investigated separately; it is not repaired by this score audit.'],
        decision='No evidence supports removing terminal geometry as the principal R5 fix. Prioritize low-speed state uncertainty and search-prefix diversity/valid completion bounds. Stagewise progress may improve timing preference but does not address a known better original-objective branch being pruned.',
        limitations=['Zero predicted unsafe cells is not proof of a physically safe actual escape, especially with low-speed state bias.',
          'The constructed plan is a feasible model counterexample, not a global optimum or a tested runtime policy.',
          'The alternate scores rescore a fixed plan set; they are not full policy searches or driving outcomes.'])
    Path('agents/apex_2026/v2/results/beam-r5-objective.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(parity=parity,pruning=pruning,scores={n:{'cost':s['total_cost'],'unsafe':s['unsafe_sample_count'],'progress':s['terminal_progress'],'distance':s['physical_chord_distance_m'],'max_projection_excess':s['max_progress_increment_minus_chord_m'],'alternatives':s['objective_alternatives']} for n,s in scores.items()}),indent=2))


if __name__=='__main__':main()
