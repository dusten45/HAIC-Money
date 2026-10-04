import importlib.util
import numpy as np
from agents.apex_2026.candidate.agent import Agent as P1


def module():
    spec=importlib.util.find_spec('agents.apex_2026.v2.fallback_agent')
    assert spec is not None, 'geometry-triggered fallback is missing'
    return importlib.import_module(spec.name)


def straight():
    frame=np.full((84,84),.6,np.float32)
    frame[:74,27:57]=.4
    frame[74:]=0
    return np.repeat(frame[None],4,axis=0)


def test_certified_straight_sequence_matches_frozen_p1_exactly():
    hybrid=module().Agent();base=P1();obs=straight()
    for _ in range(6):
        assert np.array_equal(hybrid.act(obs),base.act(obs))
        assert hybrid.diagnostics['fallback_active'] is False


def test_unreachable_prefix_is_rejected():
    agent=module().Agent()
    path=np.array([[52.,58.],[54.,56.],[56.,54.],[58.,52.]])
    free=np.ones((84,84),np.uint8)
    certificate=agent.certificate(path,free,20.,.1)
    assert not certificate['certified']
    assert certificate['prefix_unreachable']


def test_nominal_arc_crossing_road_edge_is_rejected():
    agent=module().Agent()
    free=np.zeros((84,84),np.uint8);free[:74,34:50]=1
    path=np.array([[42.,58.],[42.,50.],[42.,40.],[42.,20.]])
    certificate=agent.certificate(path,free,40.,.09)
    assert not certificate['certified']
    assert certificate['arc_unsafe']


def test_hysteresis_requires_clear_evidence_before_return():
    agent=module().Agent()
    assert agent._select_mode(False)
    assert agent._select_mode(True)
    assert agent._select_mode(True)
    assert agent._select_mode(True)
    assert not agent._select_mode(True)


def test_executed_action_memory_is_shared():
    agent=module().Agent()
    agent._record_action(np.array([.35,.7,0]),previous_steer=-.2,prior_throttle=.1)
    for policy in (agent.nominal,agent.alternative):
        assert policy.last_steer==.35
        assert np.isclose(policy.internal_throttle,.5)
    assert agent.nominal.previous_steer==-.2
