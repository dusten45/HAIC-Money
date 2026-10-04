import numpy as np
from agents.apex_2026.v2.speed_agent import Agent
from agents.apex_2026.v2.diagnostics.speed_hud import render


def test_current_hud_decoding_has_bounded_renderer_error():
    for speed in [0., 7.3, 26.8, 51.9, 78.1, 93.7, 119.4, 136.2]:
        assert abs(Agent._decode_speed(render(speed))-speed) < 3.


def test_scene_background_cannot_change_decoded_speed():
    for speed in [0., 48.2, 89.7, 135.]:
        assert Agent._decode_speed(render(speed, 0)) == Agent._decode_speed(render(speed, 255))


def test_speed_measurement_uses_current_frame_without_previous_frame_lag():
    class Probe(Agent):
        def _path(self, frame):
            path = np.array([[42.,60.],[42.,50.],[42.,40.],[42.,30.],[42.,20.],[42.,10.]])
            return path, np.ones((84,84),np.uint8), np.zeros((84,84),np.uint8)
    obs = np.stack([render(v) for v in [15.,25.,45.,70.]])
    a = Probe({'motion_preview':0.})
    a.act(obs)
    assert abs(a.diagnostics['speed']-70.) < 3.
    obs[:-1] = np.stack([render(v) for v in [80.,90.,100.]])
    b = Probe({'motion_preview':0.})
    b.act(obs)
    assert a.diagnostics['speed'] == b.diagnostics['speed']
