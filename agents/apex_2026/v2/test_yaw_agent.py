from agents.apex_2026.v2.yaw_agent import Agent


def test_countersteering_does_not_hide_existing_high_yaw_load():
    a=Agent()
    assert a._observed_traction_limit(40.,0.,6.,0.) == 0.
    assert a._traction_gas_limit(40.,0.) == 1.


def test_actual_wheel_lag_also_limits_acceleration():
    assert Agent()._observed_traction_limit(70.,0.,0.,.2) == 0.


def test_observed_limit_never_increases_original_throttle_allowance():
    a=Agent()
    for speed in [25.,50.,80.]:
        for steer in [-.2,0.,.2]:
            cap=a._observed_traction_limit(speed,steer,1.,None)
            assert 0. <= cap <= a._traction_gas_limit(speed,steer)
