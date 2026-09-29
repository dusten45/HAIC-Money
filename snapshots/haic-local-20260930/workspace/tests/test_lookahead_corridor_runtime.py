import unittest
from unittest.mock import patch

import numpy as np

from haic_agent.lookahead_corridor_runtime import LookaheadCorridorAgent


class LookaheadCorridorTests(unittest.TestCase):
    def test_preview_turns_toward_far_road(self):
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        agent = LookaheadCorridorAgent()
        agent.reset(obs)
        centers = {54: 42.0, 50: 42.0, 46: 42.0, 42: 42.0,
                   38: 44.0, 34: 46.0, 30: 48.0}
        with patch("haic_agent.lookahead_corridor_runtime.road_centers", return_value=centers):
            action = agent.act(obs)
        self.assertGreater(float(action[0]), 0.0)
        self.assertEqual(agent.lookahead_count, 1)
        self.assertTrue(np.all(np.isfinite(action)))


if __name__ == "__main__":
    unittest.main()
