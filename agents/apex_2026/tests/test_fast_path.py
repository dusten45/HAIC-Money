"""Camera steering feedback and selective fast-path routing behaviors."""
import importlib.util
import base64
import zlib
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "fast_path_agent.py"


def agent_type():
    assert SOURCE.is_file(), "fast path controller is not implemented"
    spec = importlib.util.spec_from_file_location("fast_path_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def camera(speed=0.0, yaw=0.0, center=42.0, curvature=0.0, obstacle=None):
    frame = np.full((84, 84), .63, np.float32)
    for row in range(73):
        distance = max(0.0, (63-row)/1.701)
        middle = center+1.3608*.5*curvature*distance**2
        left, right = int(round(middle-9)), int(round(middle+9))
        frame[row, max(0,left):min(84,right+1)] = .4
    if obstacle is not None:
        x,row = obstacle
        frame[row-2:row+3,x-2:x+3] = .671
    frame[73:] = 0.0
    top,bottom = 81.9-.042*speed,81.9
    for row in range(73,84):
        for col in range(10,14):
            width=max(0.0,min(col+1.,12.6)-max(float(col),10.5))
            height=max(0.0,min(row+1.,bottom)-max(float(row),top))
            frame[row,col] = width*height
    # Positive camera yaw is a right turn, whose red bar extends right.
    left,right = sorted((63.0,63.0+1.68*yaw))
    for row in range(75,81):
        height=max(0.,min(row+1.,79.8)-max(float(row),75.6))
        for col in range(51,83):
            frame[row,col] = .299*height*max(0.,min(col+1.,right)-max(float(col),left))
    return np.repeat(frame[None],4,axis=0)


def test_yaw_feedback_damps_excess_rotation_on_the_same_road():
    agent = agent_type()()
    ahead = np.arange(0.,35.,.5)
    path = .5*.015*ahead**2
    baseline = agent._steering(ahead,path,80.,0.)
    excessive = agent._steering(ahead,path,80.,3.)
    assert baseline > 0.0
    assert excessive < baseline-.03
    assert abs(baseline)<=.4 and abs(excessive)<=.4


def test_circle_feedforward_is_correct_after_yaw_settles():
    agent = agent_type()()
    radius=40.;ahead=np.arange(0.,30.,.5)
    path=radius-np.sqrt(radius**2-ahead**2)
    steering=agent._steering(ahead,path,60.,60./radius)
    assert abs(steering-np.arctan(3.24/radius))<.008


def test_yaw_hud_sign_and_speed_range_are_camera_only():
    agent=agent_type()()
    assert agent._yaw(camera(yaw=2.)[-1])>1.5
    assert agent._yaw(camera(yaw=-2.)[-1])<-1.5
    assert 110.<agent._speed(camera(speed=120.)[-1])<130.


def test_antialiased_yaw_area_keeps_moderate_and_large_turn_magnitudes():
    agent=agent_type()()
    for yaw in (-5.8,-1.5,.5,1.5,5.8):
        assert abs(agent._yaw(camera(yaw=yaw)[-1])-yaw)<.15


def test_a_circle_outside_the_swept_straight_does_not_force_a_lane_change():
    agent=agent_type()()
    clear=agent.act(camera(speed=55.))
    safe=agent_type()().act(camera(speed=55.,obstacle=(47,35)))
    assert abs(safe[0]-clear[0])<.015
    assert safe[1]>.5 and safe[2]==0.


def test_center_circle_routes_before_contact_and_imminent_contact_brakes():
    distant=agent_type()().act(camera(speed=55.,obstacle=(42,30)))
    close=agent_type()().act(camera(speed=75.,obstacle=(42,57)))
    assert abs(distant[0])>.025
    assert close[1]==0. and close[2]>0.


def test_near_pass_preview_cannot_aim_at_the_departure_back_through_the_circle():
    # Replay V1 turned back toward center at 9m because its 23m preview
    # already sampled the future departure segment, and collided next.
    action=agent_type()().act(camera(speed=70.,obstacle=(40,47)))
    assert action[0]>.05


def test_near_circle_already_outside_actual_turn_cannot_reverse_the_curve():
    # Frozen V4 T4 camera at step101, before any privileged diagnostic fields.
    # The circle at x4m/y4.4m was outside the actual left-turning hull; the
    # road-center conflict invented a pass and reversed the wheel toward it.
    compressed = (
        'eNrtnHlsFFUcx/HG+77v+76Nxrj7Bk1MjEf0DxNNPKMmmmgiHlHjkVABbQGhQLlLaQFLKW25QRERUBEBES8oyFEOoQhiKQjl'
        '1Pd5+JpxnJmd2ZnZXdhp8sueM/PmM9/3u+ZtK8ovMCoC2PBhFxu1NdcYY0ZfH7qx32FDLwo0PicbVX1VJGOuyGGeNaOujnnG'
        'PHOWJ+cc84x5xjxjnjHPmOeBzJPcIeYZ6zNXeUY15lzmWT3ySmNoxYUxz5hn2vbx8EuMutprY54xz5hnzDNneUY15phn/vAc'
        'WXVFzDPmmZc8R9ddl5c8o2AZ88wfntSb9C3I5+wM1l6NczRbPvLUTK2Gbs1W+fGljjai8jJbYz9R8XTTQCotWK+7+dpENd5c'
        'N/qA2uz0kEofTjrJV56xxRZbbLHlp5UPOV/ZkLLzlJUNPlfZ4NJz1OuYkbvBqW+fU42Puh1jdOp4iLGjpUVg27dta7W/tm5V'
        'xnO+w3fZJua7zwb0P8Mo/PAIY0tzs1i9apX4dckS8cvPP4vv589X9tOPP9qa/mxxfb1YsXy5WN/YKHbt3Cngy3XJJ4alg842'
        'unY5yvhj40axaOFC8cOCBeK7efPEt7Nn/8fmzpmjPnPiyTZm09fgtzVrRM/iEw5oP4jfK+l9sprD6M/KAnZeeWr+dqb3w+Of'
        'mzaJA8kXwJD5zBzc3NSkOMLCTlt2+sTstOmFp3n7gg5t9msfy7jR4gedDzM2/P674phqrobB007jWqsrGxpEt65H73dM+5Sc'
        'omIzscXJ9znxiIonxucw7f7Rscr/7A9xuuP7BxvEAs3Gr/+z4xF0e+s4uM65HKvwkcRq8hXmsVcednPejoddzuTHX2htmrdH'
        'p8yjXIvZ/fqepuIMY/TDww9Pr/p0m+t2+0CnjD9X6pgPPzjcIHdMpRGnOR9En354WrVpNmJl/36nZ50nmvQTU/z4QC88vF4L'
        'p2NrI++gbh044Mys8vQbE/zM+TB5umlTG/VqUWHbrMZ8v3GBc/VaM1q5BOHppk3zNWW+kZ9km2fQOR8mT7/aNO+DeU+uny2N'
        'hpX3eIkrdtr2wtNNm3bHRaPZ8qPp5IJ2XNLhaXcNrduk6vPZjZf99uh+XFY0GmbtmIqNX55uPT63PgpG/zAbvdOoc0m362DH'
        '04s2nXRp7Z8W9zg+6zy9ajTdOe/G04s2U+nSbNTN2eap600vGvWyrXU7KyMnnn7yXKfx8sj9A+s5s6bUvN5GG+9rC5NnmP0N'
        'N55Wrblp0/pd6uN1a9f+733r8Xr1PNF2jSXrwfhNrDZepzLW89sZ6wC1Bemh2eWFbucHg2VLl4pvZs1SRj3DPTsrT+t+rftk'
        'O3qfsMFHutWo9G+tPFm3FcVaRSxIvZNKR3zOvU6OU7/oEaN58yuqjzVo4ECB0fdvXPe8sfTXx1S/FcbWfdpdW/apf3ONbuDr'
        '1I/ivpOVJ2sho+IZtI9mnvOcF72zL2fOFLU1NTJfKVX31xtWPGXs3fOesXvXO2quDSkrk+dYLsgPW7a/qd6nruE9NIxm3eaJ'
        'lSffdeLJfq15aNQ8neo5rz0jerp/790rYLVi+ZNqnwP69xdww8fN/+4BY33jC0qHrEelZ818ZR0quuX9r7/6SkyeNEnM+vpu'
        'xUn7Aae8gvuCsOHRiaXWirU3GjXPdPuaurbDJ69seNrYtfNtpcPevU4y9Gf0zuE2dswNjmPgc3oY1DRbml81vpx5V+u9ltF1'
        'dWJkVZUynptzC66jWy9fj1GzzxRPt9zZKS6xjuP9goNUngHLPbvfNTY3tTeWL3tCsSPn8LtGfcL4m40Z09up+y34i7Fjxkgf'
        '8qjo2qWLZNJNHvt+8cW0aa787PTMODPN06tGic/EDjTI/VrWmc+dc6/0fw8rXU2ccEug8XB9OAY+AV8qdSt6FhfLOdsXnSnG'
        'fnlaY1ImeKaqIXkktsCR+3Y6L2auo6vx424KbUwzZ9xpbNzwosoFmON9SkpUPsBzYp1fntwLMfOsGnF5RnhqblamzG29Nssc'
        'K+EZ9v81wVegd3hy3RhX506dRFFhoSBm+WHppM9M8bRqlPnGOjenXiJzM4r/E0Ps13okh9rR8payEZWVafFkXpnHTR2TKZ66'
        'x018xYe59RCj4sn6JHL7z6ZMET26d5d5QnvJ5HXpa3qlNd/RRLZ4ak16uZ8dBU/i2Q8LHjLmzb1PrSOV8V78sfElZcQmc86U'
        'bnyPkqd5TSD5HOth7XoymeI5edKtxupVzxhrVj+r1qAx7/GdmMzJlGb98GS+WWv4KHnq3quO3X7WrIXNk9qJnJ6eELUWPgd+'
        'EydMUPb51Km+/Sfnx3qNTPHEV6LJQQPP8n2/JWye9H3gRi2Uah2TH55dio7MGE80me59qzB5kid9Mvk2qcvhKo4Tj4KydKo3'
        'o+QZ9DelYfyfQlhSX1H/T/n0dllzTg+Fpc7/6AvmC89xY2+UPvx+xXL7tjeURqkpwuJJ7Wqdf9R0ByrPz6cmjKY/X1Y9UOp/'
        '+nRhscToI1rHHS3PcpGuDRs6VIyqrlY9inSNuU2eSc5LXfnt7HtCNXpT1nET74KM2c3axH958VfQoUNyZcPTqu+JSR0nguxP'
        '6jIh53+yZ/EJQpusvRJRjF36lGTnTocKeQ4JjlHQoQ3HTcrXd/BeNngyHvrK/N4Ak+NKBtmfzCGSso5VvT9tQa+Rgw4S9Gup'
        '7fT6SXw4PUh8Dc+zxZOx0BvBwuApczJ1X0tbFDzZJ/chqeWoc6klli19vHVNOnVozNMPz3aJ9Y0vtN6nhSlsa2uuUa/37N6t'
        'eMoc+N/+4z3iX5+QWNnQ0Oob8BkY45Z+j7mVDOIr8HXUnvRoMI4RbB62YXzitzXPtRo+NQotbG5qL8d+C7+NTHIe3KfhOdx2'
        'tLSo80CnW7e8purTfZ9Nl752uvr+4vr6pNSS2q5PySkC9jzCNEj8kBriOMq4fgF5Jrgm6Fwb8SFb8ZaaZcH3DxrkiehS89QG'
        '06LCtq2xU45foNU4U3HUS3Jw6Tlin1ZU3FexXxt6grM2eAfVVPwX//0DWghjyA=='
    )
    frame=np.frombuffer(zlib.decompress(base64.b64decode(compressed)),dtype="<f4").reshape(84,84)
    clear=frame.copy()
    orange=(clear>=.655)&(clear<=.705)
    orange[73:]=False
    clear[orange]=.4
    baseline=agent_type()().act(clear)
    circle=agent_type()().act(frame)
    assert circle[0]<=baseline[0]+.025


def test_circle_with_small_raster_clearance_is_routed_with_tracking_margin():
    agent=agent_type()()
    ahead=np.arange(-3.,35.,.5)
    path,_,_=agent._route(ahead,np.zeros_like(ahead),[(18.,3.0)],55.,0.)
    assert agent.pass_side==-1
    assert float(np.interp(18.,ahead,path))<-.6


def test_three_point_five_meter_apparent_gap_keeps_raster_tracking_margin():
    agent=agent_type()()
    ahead=np.arange(-3.,35.,.5)
    agent._route(ahead,np.zeros_like(ahead),[(24.,3.5)],90.,0.)
    assert agent.pass_side==-1


def test_corner_pass_prefers_the_reachable_side_with_room_from_current_pose():
    agent=agent_type()()
    ahead=np.arange(-3.,35.,.5)
    # The circle is left of the road center but right of our current pose.
    # Automatically passing right would request a 5.7m displacement at 10m.
    path,_,_=agent._route(ahead,.3*ahead,[(10.,2.)],65.,0.)
    assert agent.pass_side==-1
    assert abs(float(np.interp(10.,ahead,path)))<2.


def test_steep_bend_pass_keeps_normal_clearance_after_road_margin_clipping():
    agent=agent_type()()
    ahead=np.arange(-3.,35.,.5)
    path,_,_=agent._route(ahead,ahead.copy(),[(16.,16.)],55.,0.)
    gap=abs(float(np.interp(16.,ahead,path))-16.)/np.sqrt(2.)
    assert gap>=3.60


def test_temporarily_occluded_circle_rotates_forward_distance_with_camera_yaw():
    agent=agent_type()()
    ahead=np.arange(-3.,35.,.5)
    agent._route(ahead,np.zeros_like(ahead),[(20.,2.5)],50.,0.)
    agent._route(ahead,np.zeros_like(ahead),[],50.,1.)
    expected=2.5*np.sin(.08)+16.*np.cos(.08)
    assert abs(agent.pass_y-expected)<.02


def test_turns_use_available_grip_but_excess_current_lateral_acceleration_brakes():
    low=agent_type()().act(camera(speed=30.,curvature=.02))
    high=agent_type()().act(camera(speed=90.,yaw=3.,curvature=.02))
    assert low[1]>.5 and low[2]==0.
    assert high[1]==0. and high[2]>0.


def test_top_speed_turn_does_not_spend_rear_grip_on_unproductive_full_gas():
    action=agent_type()().act(camera(speed=100.,yaw=1.,curvature=.01))
    assert action[1]<=.20


def test_visible_rear_wheel_spin_cuts_gas_while_preserving_steering():
    observation=camera(speed=55.,curvature=.03)
    frame=observation[-1]
    top,bottom=81.9-.021*300.,81.9
    for row in range(73,84):
        for col in range(18,25):
            width=max(0.,min(col+1.,23.1)-max(float(col),18.9))
            height=max(0.,min(row+1.,bottom)-max(float(row),top))
            frame[row,col]=.174*width*height
    action=agent_type()().act(observation)
    assert action[1]<=.20 and action[2]==0.


def test_road_antialias_above_hud_cannot_create_rear_wheel_spin():
    observation=camera()
    observation[-1,73,19:24]=.19
    agent=agent_type()()
    assert agent._rear_speed(observation[-1])==0.
    assert agent.act(observation)[1]>.8


def test_straight_launch_can_accelerate_despite_normal_powered_wheel_slip():
    observation=camera(speed=25.)
    # Calibration observed full-gas launch: hull25m/s and rear rolling58m/s.
    observation[-1,78:82,19:23]=.174
    action=agent_type()().act(observation)
    assert action[1]>.8 and action[2]==0.


def test_invalid_data_and_reset_produce_finite_bounded_actions():
    agent=agent_type()()
    agent.act(camera(speed=55.,obstacle=(42,35)))
    for observation in ([[1,2],[1]],np.full((4,84,84),np.nan),
                        np.full((4,84,84),.63)):
        for _ in range(6):
            action=agent.act(observation)
            assert action.dtype==np.float32 and np.isfinite(action).all()
            assert np.all(action>=[-1.,0.,0.]) and np.all(action<=[1.,1.,1.])
    agent.reset()
    assert agent.pass_side==0 and agent.last_steer==0.
    np.testing.assert_array_equal(agent.act(camera()),agent_type()().act(camera()))
