"""Behavior tests for an isolated pedal-allocation axis on frozen HUD100."""
import importlib.util
from pathlib import Path
import numpy as np

CONFIG = dict(lookahead=26,pursuit_gain=4.5,lateral_accel=100,motion_preview=.12,
              traction_accel=180,max_speed=100,path_smoothing=200,hud_dynamics=True)


def load_agent():
    path=Path(__file__).with_name('pedal_agent.py')
    assert path.exists(), 'Independent pedal allocator has not been implemented'
    spec=importlib.util.spec_from_file_location('isolated_pedal',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.Agent(config=CONFIG)


def test_coasts_at_a_steady_target_instead_of_adding_mandatory_gas():
    agent=load_agent()
    gas,brake=agent._allocate_pedals(50.,50.,0.,1.,0.)
    assert gas==0 and brake==0


def test_brake_inverse_reaches_requested_next_speed_without_overshoot():
    agent=load_agent()
    gas,brake=agent._allocate_pedals(50.,48.,0.,1.,0.)
    assert gas==0 and .02<brake<.08
    predicted,_=agent._predict_pedals(50.,0.,gas,brake,0.)
    assert abs(predicted-49.)<.1


def test_rolling_predictor_matches_independent_unsaturated_car_measurement():
    agent=load_agent()
    # Saved public Car sweep: speed50, steer0, commandgas.3 from internalgas0
    # gives acceleration12.1697902679m/s² during the first80ms.
    predicted,internal=agent._predict_pedals(50.,0.,.3,0.,0.)
    assert abs(predicted-(50.+12.1697902679*.08))<.15
    assert abs(internal-.3)<1e-9


def test_throttle_ramp_tracks_four_raw_ticks_and_reset_clears_memory():
    agent=load_agent()
    agent._allocate_pedals(0.,60.,0.,1.,0.)
    assert abs(agent.internal_throttle-.4)<1e-9
    agent._allocate_pedals(0.,60.,0.,1.,0.)
    assert abs(agent.internal_throttle-.8)<1e-9
    agent._allocate_pedals(50.,30.,0.,1.,0.)
    assert agent.internal_throttle==0
    agent.reset()
    assert agent.internal_throttle==0


def test_low_speed_and_observed_high_slip_use_original_law():
    agent=load_agent()
    assert agent._allocate_pedals(10.,10.,0.,1.,0.)==(.25,0.)
    assert agent._allocate_pedals(50.,50.,0.,1.,.3)==(.25,0.)


def test_original_traction_cap_is_preserved():
    agent=load_agent()
    gas,brake=agent._allocate_pedals(50.,90.,.1,.12,0.)
    assert 0<=gas<=.12
    assert brake==0


def test_invalid_observation_zero_gas_updates_throttle_memory():
    agent=load_agent()
    agent._allocate_pedals(0.,60.,0.,1.,0.)
    action=agent.act(np.full((4,84,84),np.nan,np.float32))
    assert action[1]==0
    assert agent.internal_throttle==0


def test_path_steering_and_traction_cap_match_frozen_hud100():
    from agents.apex_2026.diagnostics.hud_dynamics_calibration import render_hud
    path=Path(__file__).parent/'results'/'line_sources'/'5dd078d04dbf.py'
    spec=importlib.util.spec_from_file_location('frozen_hud100',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reference=module.Agent(config=CONFIG);candidate=load_agent()
    for speed in (0.,30.,50.,70.):
        frame=np.full((84,84),.63,np.float32);frame[:,32:52]=.4
        frame[74:]=render_hud(0.,0.,speed)[74:]
        observation=np.stack([frame]*4)
        expected=reference.act(observation);actual=candidate.act(observation)
        assert actual[0]==expected[0]
        assert candidate.diagnostics['target_speed']==reference.diagnostics['target_speed']
        assert candidate.diagnostics['gas_cap']==reference.diagnostics['gas_cap']
        np.testing.assert_array_equal(candidate.diagnostics['path'],reference.diagnostics['path'])
