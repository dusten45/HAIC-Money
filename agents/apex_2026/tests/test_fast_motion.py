"""Camera motion registration must separate translation from body rotation."""
import importlib.util
from pathlib import Path

import numpy as np

from agents.apex_2026.tests.test_fast_path import camera


SOURCE=Path(__file__).resolve().parents[1]/"fast_motion_agent.py"


def agent_type():
    assert SOURCE.is_file(), "camera motion estimator is not implemented"
    spec=importlib.util.spec_from_file_location("motion_test",SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.Agent


def transformed_road(dx,dy,rotation,curve=.007):
    ahead=np.linspace(-4.,42.,185)
    center=curve*ahead*ahead
    new_x=(center-dx)*np.cos(rotation)-(ahead-dy)*np.sin(rotation)
    new_y=(center-dx)*np.sin(rotation)+(ahead-dy)*np.cos(rotation)
    query=np.linspace(-2.,30.,65)
    current=(query,np.interp(query,new_y,new_x))
    return (ahead,center),current


def test_straight_road_measures_body_right_translation():
    old,new=transformed_road(1.2,np.sqrt(16.-1.2**2),0.,curve=0.)
    result=agent_type()()._register(old,new,4.,0.)
    assert abs(result["dx"]-1.2)<.08 and result["confidence"]>.7


def test_right_turn_rotation_is_not_mistaken_for_lateral_slip():
    rotation=.16;distance=4.
    old,new=transformed_road(distance*np.sin(rotation/2),
                             distance*np.cos(rotation/2),rotation,curve=0.)
    result=agent_type()()._register(old,new,distance,rotation)
    heading=np.arctan2(result["dx"],result["dy"])+rotation/2
    assert abs(heading)<.025 and result["confidence"]>.7


def test_curved_road_registration_preserves_both_translation_components():
    dx,dy,rotation=.8,4.7,.16
    old,new=transformed_road(dx,dy,rotation)
    result=agent_type()()._register(old,new,np.hypot(dx,dy),rotation)
    expected_x=dx*np.cos(rotation)-dy*np.sin(rotation)
    expected_y=dx*np.sin(rotation)+dy*np.cos(rotation)
    assert abs(result["dx"]-expected_x)<.12
    assert abs(result["dy"]-expected_y)<.12


def test_insufficient_or_incompatible_road_support_has_no_motion_confidence():
    agent=agent_type()()
    ahead=np.arange(0.,4.)
    result=agent._register((ahead,np.zeros(4)),(ahead,np.zeros(4)),4.,0.)
    assert result["confidence"]==0.
    old,new=transformed_road(0.,4.,0.)
    new=(new[0],new[1]+12.*np.sin(new[0]))
    assert agent._register(old,new,4.,0.)["confidence"]==0.


def test_invalid_and_duplicate_frames_leave_motion_unavailable_and_actions_bounded():
    agent=agent_type()()
    for observation in ([[1,2],[1]],np.full((4,84,84),np.nan),
                        np.full((4,84,84),.63)):
        action=agent.act(observation)
        assert np.isfinite(action).all() and action.dtype==np.float32
        assert np.all(action>=[-1.,0.,0.]) and np.all(action<=[1.,1.,1.])
        assert agent.motion_confidence==0.
    agent.last_motion_heading=.4;agent.motion_confidence=.8
    agent.reset()
    assert agent.last_motion_heading==0. and agent.motion_confidence==0.


def test_act_records_real_camera_translation_instead_of_bypassing_the_estimator():
    observation=np.asarray([camera(speed=60.,center=42.-2*i)[-1] for i in range(4)])
    agent=agent_type()()
    agent.act(observation)
    assert agent.motion_confidence>.7
    assert agent.last_lateral_velocity>12.


def test_agreeing_two_road_edges_preserve_translation_confidence():
    agent=agent_type()()
    assert hasattr(agent,"_register_boundaries"), "road boundary consistency is missing"
    ahead=np.linspace(-3.,35.,80);center=np.zeros_like(ahead)
    previous=(ahead,center,center-6.667,center+6.667)
    current=(ahead,center-1.2,center-1.2-6.667,center-1.2+6.667)
    result=agent._register_boundaries(previous,current,4.,0.)
    assert abs(result["dx"]-1.2)<.08 and result["confidence"]>.7


def test_matching_center_with_incompatible_width_cannot_claim_confident_motion():
    agent=agent_type()()
    assert hasattr(agent,"_register_boundaries"), "road boundary consistency is missing"
    ahead=np.linspace(-3.,35.,80);center=np.zeros_like(ahead)
    previous=(ahead,center,center-6.667,center+6.667)
    current=(ahead,center-1.2,center-1.2-11.,center-1.2+11.)
    assert agent._register_boundaries(previous,current,4.,0.)["confidence"]==0.
