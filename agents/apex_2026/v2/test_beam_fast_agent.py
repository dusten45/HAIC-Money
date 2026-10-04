"""Small exactness checks; full stored-observation parity is separate preflight."""
import numpy as np
from agents.apex_2026.v2.beam_fast_agent import Agent
from agents.apex_2026.v2.beam_refined_agent import Agent as Original


def context(agent):
    agent.shadow.reset(20., .1, .02)
    node=dict(cost=0.,progress=0.,state=agent.snapshot(agent.shadow),sequence=[],last=0.)
    reference=np.array([[0.,0.],[0.,10.],[2.,20.],[3.,30.]])
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(reference,axis=0),axis=1))]
    return node,reference,arc,np.full(4,40.),np.ones((84,84),np.uint8)


def test_projection_cache_bitwise():
    fast=Agent()
    _,reference,arc,_,_=context(fast)
    rng=np.random.default_rng(431)
    for _ in range(100):
        p=rng.normal(size=2)*10
        progress=float(rng.uniform(0,25)); speed=float(rng.uniform(0,100))
        assert fast._projection(p,reference,arc,progress,speed)==Original._projection(p,reference,arc,progress,speed)
    assert fast._projection_reference is reference


def test_memo_restores_state_fresh_containers_and_terminal_key():
    fast=Agent(); node,ref,arc,targets,free=context(fast)
    action=(.1,.7,0.)
    first=fast._advance(node,action,ref,arc,targets,free,False)
    expected=fast.snapshot(fast.shadow)
    first['sequence'].append((9.,9.,9.)); first['priority']=999
    first['state'][0].clear()
    fast.shadow.step((-.3,0.,.7),4)
    again=fast._advance(node,action,ref,arc,targets,free,False)
    assert again['state']==expected==fast.snapshot(fast.shadow)
    assert again['sequence']==[action] and 'priority' not in again
    assert fast._memo_hits==1 and fast._memo_misses==1
    terminal=fast._advance(node,action,ref,arc,targets,free,True)
    assert fast._memo_misses==2 and terminal['cost']!=again['cost']


def test_signed_zero_cap_and_invalid_reset():
    fast=Agent(); node,ref,arc,targets,free=context(fast)
    fast._advance(node,(0.,0.,0.),ref,arc,targets,free,False)
    fast._advance(node,(-0.,0.,0.),ref,arc,targets,free,False)
    assert fast._memo_misses==2
    fast.CACHE_CAP=2
    fast._advance(node,(.1,0.,0.),ref,arc,targets,free,False)
    assert len(fast._transition_cache)==2
    fast.act(np.zeros((1,)))
    assert not fast._transition_cache and fast._memo_hits==fast._memo_misses==0
    assert fast._projection_reference is None


def test_cache_miss_and_hit_match_uncached_relevant_state():
    fast=Agent(); original=Original()
    node,ref,arc,targets,free=context(fast)
    original_node,*_=context(original)
    action=(.1,.7,0.)
    expected=original._advance(original_node,action,ref,arc,targets,free,False)
    assert fast._advance(node,action,ref,arc,targets,free,False)==expected
    assert fast._advance(node,action,ref,arc,targets,free,False)==expected
    assert fast.snapshot(fast.shadow)==original.snapshot(original.shadow)
