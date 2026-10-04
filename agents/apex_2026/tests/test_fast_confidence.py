from pathlib import Path
import numpy as np
from agents.apex_2026.fast_confidence_agent import Agent


def test_saved_outer_road_support_loss_demands_controlled_recovery():
    frame=np.load(Path(__file__).parent/'fixtures'/'confidence-track4-step70.npz')['frame']
    agent=Agent();agent.last_center=30.2475
    action=agent.act(np.stack([frame]*4))
    assert action[1]==0., 'Do not accelerate when nearest ego component has no road support'
    assert action[2]>=.34
    assert action[0]<-.2


def test_normal_road_car_occlusion_remains_supported():
    frame=np.full((84,84),.6,np.float32);frame[:73,33:52]=.4;frame[57:69,40:45]=0.;frame[73:]=0.
    agent=Agent();assert agent._road(frame) is not None
    action=agent.act(np.stack([frame]*4));assert action[1]>.9 and action[2]==0.


def test_invalid_and_reset():
    agent=Agent();assert np.isfinite(agent.act(np.full((4,84,84),np.nan))).all()
    agent.reset();assert agent.lost_frames==0 and agent.last_steer==0.
