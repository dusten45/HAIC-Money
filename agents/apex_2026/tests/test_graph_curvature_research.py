"""Analytic checks for offline camera curvature diagnostics, not agent tuning."""
import numpy as np
from agents.apex_2026.research.speed_20261005.graph_curvature import fit,smooth,closest_s


def test_affine_route_smoothing_preserves_geometry():
    s=np.arange(0.,42.,2.);path=np.column_stack((.2*s-3.,s+1.))
    np.testing.assert_allclose(smooth(path,7.),path,atol=1e-11)
    curvature,heading=fit(path,20.)
    assert abs(curvature)<1e-11
    assert abs(heading-np.arctan(.2))<1e-11


def test_metric_circle_curvature_and_heading_match_analytic_values():
    theta=np.arange(0.,2.,.1);radius=20.
    path=np.column_stack((radius*(1.-np.cos(theta)),radius*np.sin(theta)))
    curvature,heading=fit(path,20.)
    # A quadratic fit over +/-7m approximates this circle; interpolation
    # and truncation error are bounded rather than called exact curvature.
    assert abs(curvature+1./radius)<.003
    assert abs(heading-1.)<.025


def test_polyline_projection_uses_arc_distance_and_handles_bends():
    path=np.array([[0.,0.],[0.,10.],[10.,10.]])
    at,gap=closest_s(path,np.array([6.,13.]))
    assert at==16. and gap==3.


def test_depth_support_rejects_saved_false_initial_bend():
    from pathlib import Path
    from agents.apex_2026.fast_ridge_agent import Agent
    from agents.apex_2026.research.speed_20261005.graph_curvature import supported_prefix
    frame=np.load(Path(__file__).parent/'fixtures'/'ridge-track2-step162.npz')['frame']
    agent=Agent();path=agent._ridge(frame)
    assert path is not None
    assert supported_prefix(agent,frame,path) is None


def test_depth_support_retains_saved_valid_horizontal_bend():
    from pathlib import Path
    from agents.apex_2026.fast_ridge_agent import Agent
    from agents.apex_2026.research.speed_20261005.graph_curvature import supported_prefix
    frame=np.load(Path(__file__).parent/'fixtures'/'ridge-track4-step65.npz')['frame']
    agent=Agent();path=agent._ridge(frame);supported=supported_prefix(agent,frame,path)
    np.testing.assert_array_equal(supported,path)
