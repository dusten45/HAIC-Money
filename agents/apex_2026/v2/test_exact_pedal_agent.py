import unittest
import numpy as np
from agents.apex_2026.candidate.agent import Agent as P1

class ExactPedalTests(unittest.TestCase):
    def make(self):
        from agents.apex_2026.v2.exact_pedal_agent import Agent
        return Agent()
    def test_inherits_frozen_path_and_speed_logic(self):
        a=self.make()
        self.assertIs(type(a)._path,P1._path)
        self.assertEqual(a.config,P1().config)
    def test_repeat_predictions_reset_world(self):
        a=self.make();a._sensor=(0.,0.,None);a._initial_slip=0.
        p=a._predict_exact(60.,.8,0.,0.,0.)
        a._predict_exact(30.,-.8,1.,.4,1.)
        self.assertAlmostEqual(p,a._predict_exact(60.,.8,0.,0.,0.),places=4)
        self.assertGreater(p,55.)
    def test_braking_and_gas_monotonic_straight(self):
        a=self.make();a._sensor=(0.,0.,None);a._initial_slip=0.
        gas=[a._predict_exact(50.,0.,x,0.,0.) for x in np.linspace(0,.5,9)]
        brake=[a._predict_exact(50.,0.,0.,x,0.) for x in np.linspace(0,.4,9)]
        self.assertTrue(np.all(np.diff(gas)>=-1e-4))
        self.assertTrue(np.all(np.diff(brake)<=1e-4))
    def test_limits_and_unknown_state_explicit(self):
        a=self.make();gas,brake=a._allocate_pedals(60.,24.,.8,.3,0.)
        self.assertEqual(gas,0.);self.assertGreater(brake,0.)
        self.assertLessEqual(brake,.4)
        self.assertEqual(a.pedal_diagnostics['exact_slip_source'],'propagated_unknown')
    def test_legacy_domain_preserved(self):
        for speed in (10.,95.):
            a=self.make();b=P1()
            self.assertEqual(a._allocate_pedals(speed,40.,.3,.4,0.),b._allocate_pedals(speed,40.,.3,.4,0.))
    def test_path_and_steer_same_pixels(self):
        a=self.make();b=P1();obs=np.full((4,84,84),.35,np.float32)
        obs[:,74:]=0;obs[:,77:83,10:13]=.25
        aa=a.act(obs);bb=b.act(obs)
        self.assertEqual(float(aa[0]),float(bb[0]))
        self.assertEqual(a.diagnostics['target_speed'],b.diagnostics['target_speed'])
        self.assertEqual(a.diagnostics['path'],b.diagnostics['path'])
if __name__=='__main__':unittest.main()
