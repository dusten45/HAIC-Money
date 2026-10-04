from pathlib import Path
import numpy as np
from agents.apex_2026.fast_graph_agent import Agent


def test_disconnected_saved_road_branch_not_in_forward_route():
    frame=np.load(Path(__file__).parent/'fixtures'/'graph-track2-step174.npz')['frame']
    agent=Agent();road=agent._road(frame)
    assert road is not None
    assert road[0][-1] < 21., 'Disconnected upper road cannot extend actual forward route'


def test_connected_road_retains_car_occlusion_and_bounds():
    frame=np.full((84,84),.6,np.float32);frame[:73,33:52]=.4;frame[57:69,40:45]=0.
    agent=Agent();road=agent._road(frame)
    assert road is not None and road[0][-1]>32.
    assert np.max(abs(road[1]))<.5
    action=agent.act(np.stack([frame]*4));assert action.shape==(3,) and np.isfinite(action).all()


def test_invalid_frame_and_reset():
    agent=Agent();assert np.isfinite(agent.act(np.full((4,84,84),np.nan))).all()
    agent.reset();assert agent.last_steer==0.
