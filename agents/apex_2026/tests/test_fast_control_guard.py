import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from agents.apex_2026.fast_rear_clear_agent import Agent as Parent, _ConfidenceReference
from agents.apex_2026.fast_control_guard_agent import Agent


def decode(value):
    if isinstance(value,dict):
        if '__array__' in value:return np.asarray(value['__array__'])
        if '__tuple__' in value:return tuple(decode(x) for x in value['__tuple__'])
        return {k:decode(v) for k,v in value.items()}
    if isinstance(value,list):return [decode(x) for x in value]
    return value


class ControlGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=np.load(Path(__file__).parent/'fixtures/control-guard-normal-track2.npz')

    def state(self,cls,step):
        agent=cls();agent.__dict__=decode(json.loads(str(self.fixture['state'+str(step)])))
        return agent

    def confidence(self,step):
        agent=self.state(Parent,step);agent.arc_guard_used=False;agent.arc_guard_curve=0.
        action=_ConfidenceReference.act(agent,self.fixture['frame'+str(step)])
        if agent.arc_guard_used and action[1]>0:
            demand=max(agent.last_speed*abs(agent.last_yaw),agent.last_speed**2*agent.arc_guard_curve)
            if demand>120:action[1]=min(float(action[1]),.30)
            elif demand>60:action[1]=min(float(action[1]),.60)
        return agent,action

    def test_saved_unsupported_ridge_retains_supported_confidence(self):
        for step in (149,154):
            with self.subTest(step=step):
                expected,action=self.confidence(step);agent=self.state(Agent,step)
                np.testing.assert_array_equal(agent.act(self.fixture['frame'+str(step)]),action)
                self.assertEqual(agent.mode,'confidence')
                for key in ('last_steer','last_target','reference_curvature','last_center','spin_frames','pass_side','pass_missing'):
                    self.assertEqual(getattr(agent,key),getattr(expected,key))

    def test_supported_ridge_and_both_unsupported_are_exact_parent(self):
        for step in (148,150,155):
            with self.subTest(step=step):
                parent=self.state(Parent,step);agent=self.state(Agent,step);frame=self.fixture['frame'+str(step)]
                np.testing.assert_array_equal(agent.act(frame),parent.act(frame))
                for key in ('last_steer','last_target','spin_frames','pass_side','pass_missing','circle_cooldown','mode'):
                    self.assertEqual(getattr(agent,key),getattr(parent,key))

    def test_pass_memory_transports_exactly_once(self):
        class Counting(Agent):
            def _route(self,*args):
                self.route_calls+=1
                return super()._route(*args)
        agent=self.state(Counting,148);parent=self.state(Parent,148)
        for a in (agent,parent):a.pass_side=1;a.pass_y=25.;a.pass_x=-3.;a.pass_missing=1
        agent.route_calls=0;frame=self.fixture['frame148']
        np.testing.assert_array_equal(agent.act(frame),parent.act(frame))
        self.assertEqual(agent.route_calls,1)
        self.assertEqual(agent.pass_missing,2)
        self.assertEqual(agent.pass_y,parent.pass_y)
        self.assertEqual(agent.circle_cooldown,parent.circle_cooldown)

    def test_rejected_trial_restores_confidence_spin(self):
        agent=self.state(Agent,149);expected=self.state(Parent,149);frame=self.fixture['frame149']
        speed=agent._speed(frame)
        with patch.object(agent,'_rear_speed',side_effect=[speed+20.,speed]):
            action=agent.act(frame)
        with patch.object(expected,'_rear_speed',return_value=speed+20.):
            expected.arc_guard_used=False;expected.arc_guard_curve=0.
            original=_ConfidenceReference.act(expected,frame)
        np.testing.assert_array_equal(action,original)
        self.assertEqual(expected.spin_frames,6)
        self.assertEqual(agent.spin_frames,6)

    def test_reset_and_missing_camera_match_parent(self):
        agent=self.state(Agent,149);parent=self.state(Parent,149)
        agent.reset();parent.reset()
        np.testing.assert_array_equal(agent.act(np.full((4,84,84),np.nan)),parent.act(np.full((4,84,84),np.nan)))
        self.assertEqual(agent.lost_frames,parent.lost_frames)


if __name__=='__main__':unittest.main()
