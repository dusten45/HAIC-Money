import unittest
import numpy as np
from agents.apex_2026.v2.force_agent import Agent


class ForceBudgetTests(unittest.TestCase):
    def agent(self):
        return Agent({'traction_accel':180.})

    def test_mild_fast_curve_can_use_more_than_half_throttle(self):
        a=self.agent();a._traction_yaw=.3;a._traction_wheel=.015
        self.assertGreater(a._traction_gas_limit(80.,.015),.8)

    def test_tight_corner_reserves_force_for_turning(self):
        a=self.agent();a._traction_yaw=3.5;a._traction_wheel=.2
        self.assertEqual(a._traction_gas_limit(60.,.2),0.)

    def test_observed_slip_disables_extra_drive(self):
        a=self.agent();a._traction_slip=.2;a._traction_flow_valid=True
        self.assertEqual(a._traction_gas_limit(60.,.01),0.)

    def test_mirror_and_finite_bounds(self):
        for speed in [0.,10.,20.,40.,60.,80.,100.]:
            for steer in [0.,.01,.1,.4,.8]:
                a=self.agent();b=self.agent()
                a._traction_yaw,b._traction_yaw=1.,-1.
                a._traction_wheel,b._traction_wheel=.1,-.1
                x,y=a._traction_gas_limit(speed,steer),b._traction_gas_limit(speed,-steer)
                self.assertTrue(np.isfinite(x));self.assertGreaterEqual(x,0.);self.assertLessEqual(x,1.)
                self.assertAlmostEqual(x,y)

    def test_reset_removes_observed_state(self):
        a=self.agent();a._traction_yaw=5;a._traction_slip=.5;a.reset()
        self.assertEqual(a._traction_yaw,0.)
        self.assertFalse(a._traction_flow_valid)


if __name__=='__main__':unittest.main()
