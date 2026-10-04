import json
from pathlib import Path
import numpy as np
from agents.apex_2026.v2.beam_refined_agent import Agent

FIXTURES=Path(__file__).parent/'results/beam_refinement_fixtures'


def test_public_pixel_stall_recovers_better_noncolliding_lattice_plan():
    obs=np.load(FIXTURES/'r5_stall_120_121.npz')['observations']
    states=json.loads((FIXTURES/'r5_stall_120_121.json').read_text())['states']
    for observation,state in zip(obs,states):
        agent=Agent()
        for key in ('throttle','last_steer','slip'):setattr(agent,key,state[key])
        action=agent.act(observation);d=agent.diagnostics
        assert d['original_beam_cost']==state['expected_cost']
        assert d['beam_cost']<d['original_beam_cost']-1.
        assert action[1]>.6 and action[2]==0
        assert d['refined_collision_count']<=d['original_collision_count']==0
        assert d['refinement_raw_ticks']==224
        assert 1<=d['refinement_shift']<=7


def test_rejected_refinement_preserves_original_object_and_ties():
    agent=Agent();original={'cost':1.,'unsafe_count':0,'sequence':[(.1,.7,0.)]*8}
    assert agent._accept_refinement(original,{'cost':1.,'unsafe_count':0},original) is original
    assert agent._accept_refinement(original,{'cost':0.,'unsafe_count':1},original) is original
    assert agent._accept_refinement(original,{'cost':2.,'unsafe_count':0},original) is original
    better={'cost':0.,'unsafe_count':0}
    assert agent._accept_refinement(original,better,original) is better
    tied={'cost':0.,'unsafe_count':0}
    assert agent._accept_refinement(better,tied,original) is better


def test_count_metadata_does_not_change_frozen_stage_objective():
    assert Agent.stage_cost(2.,.3,.2,40.,50.) == -4*2+.08*(2*.3**2+3*.2**2+.003*10**2)
