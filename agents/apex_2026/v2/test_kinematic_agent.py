import math
from agents.apex_2026.v2.kinematic_agent import Agent


def test_steering_inverts_geometry_instead_of_multiplying_curvature():
    for curvature in [-.1,-.02,0.,.02,.1]:
        steer=Agent._curvature_to_steer(curvature)
        assert abs(math.tan(steer)/3.24-curvature)<1e-12


def test_impossible_curvature_respects_physical_joint_limits():
    assert Agent._curvature_to_steer(1.) == .4
    assert Agent._curvature_to_steer(-1.) == -.4
