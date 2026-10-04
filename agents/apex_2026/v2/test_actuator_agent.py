import unittest
from agents.apex_2026.pedal_agent import Agent as Before
from agents.apex_2026.v2.actuator_agent import Agent

class ActuatorDomainTests(unittest.TestCase):
    def test_impossible_instantaneous_wheel_angle_does_not_cancel_braking(self):
        a=Agent();gas,brake=a._allocate_pedals(64.,24.,-.8,.5,0.)
        self.assertEqual(gas,0.);self.assertGreater(brake,.6)
        self.assertEqual(a.pedal_diagnostics['allocator_mode'],'legacy_outside_rolling')
    def test_identified_domain_behavior_is_unchanged(self):
        for speed,target,steer in [(60.,60.,.03),(60.,58.,.1),(60.,65.,-.1),(70.,25.,.2)]:
            self.assertEqual(Agent()._allocate_pedals(speed,target,steer,.5,0.),Before()._allocate_pedals(speed,target,steer,.5,0.))

if __name__=='__main__':unittest.main()
