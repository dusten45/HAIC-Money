"""Use observed boundary support to reject false startup and horizon bends."""
from pathlib import Path
import numpy as np
from agents.apex_2026.fast_clear_supported_agent import Agent
from agents.apex_2026.fast_clear_ridge_agent import Agent as Unfiltered
from agents.apex_2026.fast_confidence_agent import Agent as Confidence
from agents.apex_2026.tests.test_fast_envelope import camera


def frame(name):
    return np.load(Path(__file__).parent/'fixtures'/name)['frame']


def test_saved_false_startup_falls_back_to_exact_confidence():
    image=frame('ridge-track2-step162.npz');observation=np.stack([image]*4)
    candidate=Agent()
    assert candidate._ridge(image) is None
    np.testing.assert_array_equal(candidate.act(observation),Confidence().act(observation))
    assert candidate.mode=='confidence'


def test_saved_false_horizon_bend_truncates_before_low_depth():
    image=frame('ridge-track2-step92.npz');candidate=Agent();path=candidate._ridge(image)
    original=Unfiltered()._ridge(image)
    assert path is not None and len(path)<len(original)
    depth=candidate._sample_distance(candidate._distance_field(image),path)
    assert np.min(depth)>=5.8
    np.testing.assert_array_equal(path,original[:len(path)])


def test_supported_horizontal_bend_keeps_original_ridge_and_action():
    image=frame('ridge-track4-step65.npz');observation=np.stack([image]*4)
    np.testing.assert_array_equal(Agent()._ridge(image),Unfiltered()._ridge(image))
    candidate,original=Agent(),Unfiltered()
    np.testing.assert_array_equal(candidate.act(observation),original.act(observation))
    assert candidate.mode=='ridge'


def test_obstacle_and_unsupported_ego_actions_stay_exact():
    for observation in [camera(speed=60,obstacle=(44,27)),np.stack([frame('confidence-track4-step70.npz')]*4)]:
        candidate,original=Agent(),Unfiltered()
        candidate.last_center=original.last_center=30.2475
        np.testing.assert_array_equal(candidate.act(observation),original.act(observation))


def test_invalid_reset_import_and_observation_contract():
    import ast
    candidate=Agent();observation=camera(speed=60,curvature=.003);saved=observation.copy()
    action=candidate.act(observation);np.testing.assert_array_equal(observation,saved)
    assert action.dtype==np.float32 and action.shape==(3,) and np.isfinite(action).all()
    assert np.isfinite(candidate.act(np.full((4,84,84),np.nan))).all()
    candidate.reset();np.testing.assert_array_equal(candidate.act(observation),Agent().act(observation))
    tree=ast.parse((Path(__file__).parents[1]/'fast_clear_supported_agent.py').read_text())
    assert [n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)]==['numpy']
    assert not any(isinstance(n,ast.ImportFrom) for n in ast.walk(tree))
