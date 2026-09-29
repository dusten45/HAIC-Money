import unittest

from training.confirm_speed_budget_package import cells_for, decide, summarize


class SpeedBudgetConfirmationTests(unittest.TestCase):
    def test_replication_cells_are_new_and_complete(self):
        cells = cells_for(2024)
        self.assertEqual(len(cells), 12)
        self.assertEqual(cells[0], (1, 2024))
        self.assertEqual(cells[-1], (3, 2027))
        self.assertNotIn((1, 2023), cells)

    def test_summarize_keeps_packaged_action_activation(self):
        rows = [{"completed": True, "lapTimeMs": 23000, "progress": 1.0,
                 "collisions": 0, "damage": 0.0, "invalid_actions": 0,
                 "import_create_s": 1.0, "reset_s": 0.01, "peak_rss_bytes": 250000000,
                 "act_max_ms": 20.0, "act_p95_ms": 12.0, "activation_count": 17}]
        result = summarize(rows)
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["activation_count"], 17)

    def test_completion_precedes_faster_finished_laps(self):
        self.assertEqual(decide({"completed": 11, "invalid_actions": 0,
                                 "median_finished_lap_ms": 22000},
                                {"completed": 12, "invalid_actions": 0,
                                 "median_finished_lap_ms": 25000}), "REJECT")


if __name__ == "__main__":
    unittest.main()
