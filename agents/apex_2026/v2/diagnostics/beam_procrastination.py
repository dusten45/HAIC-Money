"""Pixel-only replay and score decomposition of the frozen R3 terminal stall.

Uses two already captured observation stacks and action-memory state only.
No environment reset, real-world state initialization, runtime edit or driving
counterfactual is performed. Alternative sequences run in the internal shadow.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

BASE=Path('/tmp/apex-v2-beam-r3')
SOURCE=BASE/'source.py'
TRACE=BASE/'t3-s4111953688.jsonl'
CAPTURE=BASE/'failure_capture/observations.npz'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def carried_state(agent,rows,step):
    agent.last_steer=float(np.float32(rows[step-2]['action'][0]))
    agent.slip=rows[step-2]['policy_diagnostics']['predicted_slip']
    throttle=0.
    for row in rows[:step-1]:
        command=float(np.float32(row['action'][1]))
        for _ in range(4):throttle+=min(command-throttle,.1)
    agent.throttle=throttle
    return dict(last_steer=agent.last_steer,slip=agent.slip,throttle=throttle)


def initialize(agent,module,observation):
    frame=observation[-1]
    free,_,_=agent.road._free_space(frame)
    path=agent.road._geodesic(free)
    reference=np.vstack(([0.,0.],(path-[42.,63.])/[1.3608,-1.701]))
    arc,kappas=agent.road._geometry(reference)
    targets=np.clip(np.sqrt(100./np.maximum(abs(kappas),.001)),24.,100.)
    targets[-1]=min(targets[-1],24.)
    for i in range(len(targets)-2,-1,-1):
        targets[i]=min(targets[i],math.sqrt(targets[i+1]**2+130.*(arc[i+1]-arc[i])))
    speed=agent._decode_speed(frame)
    yaw,wheel=agent.road._hud_dynamics(frame)
    _,measured,valid=agent.road._motion(observation[-2],frame)
    slip=float(np.clip(measured if valid else agent.slip,-.7,.7))
    omega=module.decode_wheel_omega(frame)
    agent.shadow.reset(speed*math.cos(slip),-yaw,float(np.clip(-wheel,-.4,.4)),
        lateral_speed=speed*math.sin(slip),throttle=agent.throttle,omegas=omega)
    initial=agent.snapshot(agent.shadow)
    raw_unsafe=agent._footprint_cost(free,0.,0.,0.)
    free[59:68,40:45]=1
    return reference,arc,targets,free,initial,dict(speed=speed,yaw=yaw,wheel=wheel,slip=slip,
        flow_valid=bool(valid),omegas=omega.tolist(),raw_mask_initial_unsafe=raw_unsafe,
        initialized_mask_initial_unsafe=agent._footprint_cost(free,0.,0.,0.))


def score_sequence(agent,sequence,reference,arc,targets,free,initial):
    node=dict(cost=0.,progress=0.,state=initial,sequence=[],last=agent.last_steer)
    samples=[]
    for index,action in enumerate(sequence):
        agent.restore(agent.shadow,node['state'])
        unsafe=[]
        for _ in range(4):
            x,y,h,v,_=agent.shadow.step(action)
            unsafe.append(agent._footprint_cost(free,x,y,h))
        next_node=agent._advance(node,action,reference,arc,targets,free,index==len(sequence)-1)
        progress,cross,ref_heading=agent._projection(np.array([x,y]),reference,arc,node['progress'],v)
        error=math.atan2(math.sin(-h-ref_heading),math.cos(-h-ref_heading))
        target=float(np.interp(progress,arc,targets))
        stage=agent.stage_cost(progress-node['progress'],cross,error,v,target)
        terminal=dict(cross=4.*cross**2,heading=5.*error**2,speed=.03*(v-target)**2) if index==len(sequence)-1 else dict(cross=0.,heading=0.,speed=0.)
        samples.append(dict(action=action,progress=progress,delta_progress=progress-node['progress'],
            position=[x,y],speed=v,unsafe_samples=unsafe,stage_cost=stage,
            steering_cost=.03*(action[0]-node['last'])**2,terminal_cost=terminal,
            cross_track=cross,heading_error=error,target=target,total_cost=next_node['cost']))
        node=next_node
    unsafe=sum(sum(s['unsafe_samples']) for s in samples)
    stage=sum(s['stage_cost'] for s in samples)
    terminal=sum(samples[-1]['terminal_cost'].values())
    steering=sum(s['steering_cost'] for s in samples)
    assert abs(node['cost']-(stage+200.*unsafe+terminal+steering))<1e-8
    assert abs(sum(s['delta_progress'] for s in samples)-node['progress'])<1e-10
    return dict(total_cost=node['cost'],terminal_progress=node['progress'],
        first_action_progress=samples[0]['progress'],progress_reward=-4.*node['progress'],
        running_tracking_speed_cost=stage+4.*node['progress'],
        terminal_cost=terminal,terminal_components=samples[-1]['terminal_cost'],
        collision_cost=200.*unsafe,unsafe_sample_count=unsafe,steering_cost=steering,
        integrated_progress_m_s=.08*sum(s['progress'] for s in samples),samples=samples)


def main():
    assert sha(SOURCE)=='aef90410f8ad7f3f022a7c5a07c9ba87b997304964a63e5ca0601d8e60e7e8fa'
    spec=importlib.util.spec_from_file_location('beam_r3_procrastination_source',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[json.loads(line) for line in TRACE.read_text().splitlines()]
    observations=np.load(CAPTURE)['observations']
    receipt=json.loads((BASE/'t3-s4111953688.json').read_text())
    helper=Path('agents/apex_2026/v2/shadow_physics.py').resolve()
    assert sha(helper)==receipt['provenance']['runtime_source_sha256_after'][str(helper)]
    parity=[]
    for step in (154,155):
        agent=module.Agent();carry=carried_state(agent,rows,step)
        action=agent.act(observations[step-120])
        expected=rows[step-1]
        assert np.array_equal(action,np.asarray(expected['action'],np.float32))
        assert agent.diagnostics['beam_cost']==expected['policy_diagnostics']['beam_cost']
        parity.append(dict(step=step,carried_state=carry,action=action.tolist(),beam_cost=agent.diagnostics['beam_cost'],
                           exact_action_and_cost_match=True))
    agent=module.Agent();carried_state(agent,rows,154)
    reference,arc,targets,free,initial,initialized=initialize(agent,module,observations[34])
    selected=rows[153]['policy_diagnostics']['beam_sequence']
    sequences=dict(chosen=selected,gas_now=[[selected[0][0],.7,0.]]+selected[1:],
        left_shift_coast_tail=selected[3:]+[[-.4,0.,0.]]*3,
        left_shift_brake_tail=selected[3:]+[[-.4,0.,.65]]*3,
        coast_all=[[-.4,0.,0.]]*8,gas_all=[[-.4,.7,0.]]*8)
    scores={name:score_sequence(agent,sequence,reference,arc,targets,free,initial) for name,sequence in sequences.items()}
    stalled=[r for r in rows if r['step']>=154]
    result=dict(scope='Zero new resets. Two pixel-only Agent.act replays; alternative sequences use the internal shadow only.',
        provenance={str(p):sha(p) for p in (SOURCE,TRACE,CAPTURE,BASE/'failure_capture/receipt.json',helper,Path(__file__))},
        parity=parity,initialized_pixel_state=initialized,sequence_scores=scores,
        repeated_plan_observation=dict(decisions=len(stalled),first_step=stalled[0]['step'],last_step=stalled[-1]['step'],
            coast_first_actions=sum(r['action'][1]==0 and r['action'][2]==0 for r in stalled),
            unchanged_beam_sequences=sum(r['policy_diagnostics']['beam_sequence']==selected for r in stalled),
            max_actual_speed=max(r['before']['speed_m_s'] for r in stalled),
            net_actual_displacement_m=float(np.linalg.norm(np.array(stalled[-1]['after']['position'])-stalled[0]['before']['position']))),
        findings=[
          'Progress rewards telescope exactly: sum(-4*delta_progress)=-4*terminal_progress. They do not reward earlier traversal at equal final progress.',
          'At154 and155 the same stationary observation/action memory yields the same three-coast-then-gas plan; only the first coast is executed each time.',
          'Chosen and immediate-gas-substitution sequences both have zero predicted unsafe samples. Their ranking difference is dominated by terminal cross-track/heading/speed cost, not collision occupancy.',
          'Left-shifting the chosen motion and appending coast reaches predicted unsafe pixels; forcing acceleration is not justified.',
          'The initialized prediction mask has zero initial unsafe samples, so this particular optimizer stall is not caused by an unavoidable initial collision penalty. Earlier real collisions and perception/state errors remain separate issues.'],
        structural_hypothesis='Use an explicit stagewise time-to-progress or discounted progress objective plus a terminal continuation/viability condition that cannot improve merely by delaying the hard maneuver beyond the horizon. Retain collision constraints; do not impose unconditional gas or infer a collision-free escape from this finite candidate set.',
        limitations=['Internal predictions are not actual alternative driving outcomes. No escape feasibility or rescue is established.',
          'A beam search can prune better action prefixes; this decomposition does not prove a global optimum.',
          'The car-occlusion fill and finite footprint mask limit collision fidelity; zero predicted occupancy does not prove no physical contact.'])
    output=Path('agents/apex_2026/v2/results/beam-procrastination.json')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(parity=parity,initialized=initialized,stall=result['repeated_plan_observation'],
        scores={n:{k:v for k,v in s.items() if k!='samples'} for n,s in scores.items()}),indent=2))


if __name__=='__main__':main()
