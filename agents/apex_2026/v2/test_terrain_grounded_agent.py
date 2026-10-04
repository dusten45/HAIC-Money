"""Ground classification must use visible past ground, not the vehicle sprite."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np


def agent():
    p=Path(__file__).with_name('terrain_grounded_agent.py')
    assert p.exists(), 'Occlusion-aware terrain observer is not implemented'
    s=importlib.util.spec_from_file_location('grounded',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m.Agent()


def hud(frame,speed=50.,yaw=0.):
    frame[74:]=0.
    frame[77:83,10:13]=(.27+.085*speed)/18.
    if yaw:
        difference=(-yaw-.024360133436811916)/.4580150260806972
        if difference>0:frame[74:82,52:63]=difference/88.
        else:frame[74:82,63:75]=-difference/96.


class GroundedTests(unittest.TestCase):
    def test_rigid_backprojection_couples_translation_and_yaw(self):
        a=agent();point=np.array([[1.,2.]])
        result=a._transport_to_past(point,np.full(4,20.),np.full(4,2.),2)
        angle=.32
        expected=np.array([[np.cos(angle)+2*np.sin(angle)+10*(1-np.cos(angle)),
                            -np.sin(angle)+2*np.cos(angle)+10*np.sin(angle)]])
        np.testing.assert_allclose(result,expected,atol=1e-9)

    def test_gray_vehicle_over_grass_does_not_imply_road(self):
        a=agent();obs=np.full((4,84,84),.65,np.float32)
        for f in obs:hud(f)
        obs[-1,58:70,39:46]=.4
        np.testing.assert_allclose(a._current_grip(obs,50.),[.6]*4)
        self.assertTrue(all(x.startswith('history') for x in a.ground_diagnostics['wheel_ground_sources']))

    def test_visible_historical_road_overrides_current_side_grass(self):
        a=agent();obs=np.full((4,84,84),.65,np.float32)
        for f in obs:hud(f)
        obs[:-1,40:56,28:57]=.4
        obs[-1,58:70,39:46]=.4
        np.testing.assert_allclose(a._current_grip(obs,50.),[1.]*4)

    def test_stationary_occluded_wheels_are_explicit_fallback(self):
        a=agent();obs=np.full((4,84,84),.65,np.float32)
        for f in obs:hud(f,0.);f[58:70,39:46]=.4
        np.testing.assert_allclose(a._current_grip(obs,0.),[.6]*4)
        self.assertEqual(a.ground_diagnostics['wheel_ground_sources'],['visible_side_fallback']*4)

    def test_crop_and_vehicle_occlusion_are_explicit_unknown(self):
        a=agent();frame=np.full((84,84),.4,np.float32)
        grip,valid=a._sample_ground(np.array([[50.,0.],[0.,-10.],[0.,0.],[4.,10.]]),frame)
        np.testing.assert_array_equal(valid,[False,False,False,True])
        np.testing.assert_allclose(grip,[.6,.6,.6,1.])

    def test_straight_start_preserves_forward_drive(self):
        a=agent();f=np.full((84,84),.65,np.float32);f[:74,28:57]=.4;hud(f,0.)
        action=a.act(np.stack([f]*4))
        self.assertLess(abs(action[0]),.1);self.assertGreater(action[1],.5)
        self.assertTrue(np.isfinite(action).all())

if __name__=='__main__':unittest.main()
