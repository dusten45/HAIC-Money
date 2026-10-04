from pathlib import Path
import numpy as np
from agents.apex_2026.fast_ridge_agent import Agent


def test_saved_nearly_horizontal_bend_has_parametric_forward_center():
    frame=np.load(Path(__file__).parent/'fixtures'/'ridge-track4-step65.npz')['frame']
    agent=Agent();path=agent._ridge(frame)
    assert path is not None
    assert np.min(path[:,0]) < -20.
    point=path[np.argmin(abs(path[:,0]+20.))]
    assert abs(point[1]-11.2)<1.5


def test_straight_route_center_and_action():
    frame=np.full((84,84),.6,np.float32);frame[:73,33:52]=.4;frame[57:69,40:45]=0.
    agent=Agent();path=agent._ridge(frame)
    assert path is not None and path[-1,1]>32.
    assert np.max(abs(path[:,0]))<.6
    action=agent.act(np.stack([frame]*4));assert action.shape==(3,) and np.isfinite(action).all()


def test_no_privileged_imports_and_reset():
    import ast
    source=Path(__file__).parents[1]/'fast_ridge_agent.py';tree=ast.parse(source.read_text())
    assert [n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)]==['numpy']
    assert not any(isinstance(n,ast.ImportFrom) for n in ast.walk(tree))
    agent=Agent();assert np.isfinite(agent.act(np.full((4,84,84),np.nan))).all();agent.reset();assert agent.last_steer==0.
