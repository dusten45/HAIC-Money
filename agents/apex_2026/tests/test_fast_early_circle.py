"""Detect antialiased real circles before the narrow raster core becomes large."""
from pathlib import Path
import numpy as np
from agents.apex_2026.fast_early_circle_agent import Agent
from agents.apex_2026.fast_clear_supported_agent import Agent as Reference


def road_camera():
    frame=np.full((84,84),.67,np.float32);frame[:73,33:52]=.4;frame[73:]=0.
    return frame


def test_saved_antialiased_circle_detected_two_actions_earlier():
    for step in [94,95]:
        frame=np.load(Path(__file__).parent/'fixtures'/f'early-circle-track2-step{step}.npz')['frame']
        agent=Agent();road=agent._road(frame)
        circles=agent._circles(frame,road)
        assert len(circles)==1
        assert circles[0][0]>19. and abs(circles[0][1])<1.
        assert Reference()._circles(frame,road)==[]


def test_grass_hud_and_white_road_paint_do_not_create_circles():
    frame=road_camera();frame[75:79,38:43]=.63;frame[20:25,34:36]=.9
    agent=Agent();assert agent._circles(frame,agent._road(frame))==[]


def test_long_bright_texture_and_tiny_specks_do_not_create_circles():
    frame=road_camera();frame[10:45,41:43]=.63;frame[52,40]=.63
    agent=Agent();assert agent._circles(frame,agent._road(frame))==[]


def test_clear_and_recovery_actions_remain_exact_and_finite():
    observations=[np.stack([road_camera()]*4),np.full((4,84,84),np.nan)]
    for observation in observations:
        np.testing.assert_array_equal(Agent().act(observation),Reference().act(observation))
