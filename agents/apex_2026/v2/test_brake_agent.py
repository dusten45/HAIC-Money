import unittest
from agents.apex_2026.pedal_agent import Agent as Before
from agents.apex_2026.v2.brake_agent import Agent

class BrakeAuthorityTests(unittest.TestCase):
    def test_emergency_uses_existing_actuator_authority(self):
        a=Agent();gas,brake=a._allocate_pedals(70.,25.,0.,1.,0.)
        self.assertEqual(gas,0.);self.assertGreater(brake,.38);self.assertLessEqual(brake,.4)
        p,_=a._predict_pedals(70.,0.,gas,brake,0.)
        self.assertLess(p,70.-65.*.08)
    def test_nonurgent_actions_preserved(self):
        for speed,target in [(60.,60.),(60.,58.),(60.,65.),(10.,40.),(95.,80.)]:
            self.assertEqual(Agent()._allocate_pedals(speed,target,.03,.5,0.),Before()._allocate_pedals(speed,target,.03,.5,0.))

if __name__=='__main__':unittest.main()
