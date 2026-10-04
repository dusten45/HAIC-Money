import importlib.util
import numpy as np


def test_shadow_controller_rejects_nonfinite_pixels():
    spec=importlib.util.find_spec('agents.apex_2026.v2.shadow_agent')
    assert spec is not None, 'shadow controller not implemented'
    agent=importlib.import_module(spec.name).Agent()
    action=agent.act(np.full((4,84,84),np.nan))
    assert np.isfinite(action).all()
    assert action[1]==0 and action[2]>0


def test_shadow_controller_is_deterministic_on_straight_road():
    from agents.apex_2026.v2.shadow_agent import Agent
    from agents.apex_2026.v2.diagnostics.shadow_rpm import render
    frame=render([65,65,70,70],speed=35.)
    frame[:74]=.6
    frame[:74,30:54]=.4
    obs=np.repeat(frame[None],4,axis=0)
    one,two=Agent(),Agent()
    a=one.act(obs);b=two.act(obs)
    assert np.array_equal(a,b)
    assert abs(a[0])<.15
    assert one.diagnostics['shadow_candidates']>=10
