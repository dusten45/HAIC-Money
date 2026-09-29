import unittest

import numpy as np

from haic_agent.high_speed_pivot_runtime import HighSpeedPivotAgent


class _Base:
    def reset(self, observation):
        pass

    def act(self, observation):
        return np.asarray((0.0, 0.1, 0.0), dtype=np.float32)


def _observation(*, near=42, middle=42, far=42, speed=40, obstacle=None, road=True):
    frame = np.zeros((84, 84), dtype=np.float32)
    if road:
        for row, center in ((54, near), (50, near), (46, middle), (42, middle),
                            (38, middle), (34, far), (30, far)):
            x = int(center)
            frame[row, x - 9:x + 10] = 0.35
    if obstacle is not None:
        x, y = obstacle
        frame[y:y + 3, x:x + 3] = 0.8
    frame[77:83, 10:13] = (speed * 0.085 + 0.27) / 18.0
    return np.stack((frame, frame, frame, frame))


class HighSpeedPivotTests(unittest.TestCase):
    def test_lost_road_uses_last_visible_turn_and_stops_gas(self):
        agent = HighSpeedPivotAgent(_Base(), "road_memory")
        agent.reset(_observation())
        agent.act(_observation(near=42, middle=37, far=32))
        action = agent.act(_observation(road=False, speed=45))
        self.assertLess(float(action[0]), 0)
        self.assertEqual(float(action[1]), 0.0)
        self.assertGreater(float(action[2]), 0.0)
        self.assertEqual(agent.activation_count, 1)

    def test_path_target_changes_steering_before_left_bend(self):
        agent = HighSpeedPivotAgent(_Base(), "path_exit")
        agent.reset(_observation())
        action = agent.act(_observation(near=42, middle=38, far=34))
        self.assertLess(float(action[0]), 0)
        self.assertEqual(agent.activation_count, 1)

    def test_obstacle_on_right_selects_left_passage(self):
        agent = HighSpeedPivotAgent(_Base(), "obstacle_side")
        agent.reset(_observation())
        action = agent.act(_observation(obstacle=(45, 40)))
        self.assertLess(float(action[0]), 0)
        self.assertEqual(agent.activation_count, 1)

    def test_speed_budget_gases_on_straight_and_brakes_before_bend(self):
        agent = HighSpeedPivotAgent(_Base(), "speed_budget")
        agent.reset(_observation())
        straight = agent.act(_observation(speed=40))
        bend = agent.act(_observation(near=42, middle=37, far=32, speed=55))
        self.assertGreaterEqual(float(straight[1]), 0.25)
        self.assertEqual(float(bend[1]), 0.0)
        self.assertGreater(float(bend[2]), 0.0)
        self.assertEqual(agent.activation_count, 2)


if __name__ == "__main__":
    unittest.main()
