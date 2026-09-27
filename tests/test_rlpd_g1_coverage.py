import copy
import hashlib
import unittest

from haic.algorithms.rlpd.g1_coverage import ACTOR_HASHES, CoverageRules, assess_coverage


class TestG1CoverageGate(unittest.TestCase):
    def setUp(self):
        self.cells = tuple((1, 410000 + index) for index in range(24))
        self.rules = CoverageRules(
            cells=self.cells,
            primary_actor_id="entropy-v5-author-seed50",
            comparator_actor_id="long-horizon-seed11",
            image_rubric_sha256="a" * 64,
        )
        self.rows = []
        for index, (track_id, seed) in enumerate(self.cells):
            road_sha = hashlib.sha256(f"synthetic-road-{seed}".encode("ascii")).hexdigest()
            for actor_id in (self.rules.primary_actor_id, self.rules.comparator_actor_id):
                is_primary = actor_id == self.rules.primary_actor_id
                finishes = is_primary and index in (3, 4, 5)
                self.rows.append({
                    "partition": "TRAIN", "track_id": track_id,
                    "geometry_seed": seed, "obstacles": True,
                    "actor_id": actor_id, "actor_sha256": ACTOR_HASHES[actor_id],
                    "road_centerline_sha256": road_sha,
                    "status": "complete", "outcome": "finished" if finishes else "off_track",
                    "finished": finishes, "terminated": True, "truncated": finishes,
                    "decisions": 120, "driven_raw_frames": 480,
                    "reset_initial_raw_frames": 1, "reset_noop_raw_frames": 50,
                    "image_flag": "yes" if is_primary and index in (0, 1, 2) else "no",
                    "image_rubric_sha256": self.rules.image_rubric_sha256,
                    "image_review_blinded": True,
                    "image_review_receipt_sha256": "b" * 64,
                })

    def evaluate(self, rows=None):
        return assess_coverage(self.rules, self.rows if rows is None else rows,
                               core_hours_used=0.15)

    def test_complete_paired_cohort_passes_only_distinct_primary_roads(self):
        result = self.evaluate()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["geometry_clusters"], 24)
        self.assertEqual(result["scheduled_episodes"], 48)
        self.assertEqual(result["primary_positive_geometries"], [410000, 410001, 410002])
        self.assertEqual(result["primary_finished_parent_geometries"], [410003, 410004, 410005])
        self.assertEqual(result["spent_decisions"], 48 * 120)
        self.assertEqual(result["reset_noop_raw_frames"], 48 * 50)
        self.assertEqual(result["uncertain_slots"], [])

    def test_two_image_positive_or_two_finished_parent_roads_stop_inconclusive(self):
        rows = copy.deepcopy(self.rows)
        rows[4]["image_flag"] = "unknown"
        result = self.evaluate(rows)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["unknown_image_slots"], [(410002, self.rules.primary_actor_id)])
        rows = copy.deepcopy(self.rows)
        rows[6]["outcome"] = "off_track"
        rows[6]["finished"] = False
        rows[6]["truncated"] = False
        self.assertEqual(self.evaluate(rows)["status"], "inconclusive")

    def test_duplicate_generated_centerline_is_not_independent_geometry(self):
        rows = copy.deepcopy(self.rows)
        for index in (0, 1, 2):
            for row in rows[2 * index:2 * index + 2]:
                row["road_centerline_sha256"] = "c" * 64
        for index in (3, 4, 5):
            for row in rows[2 * index:2 * index + 2]:
                row["road_centerline_sha256"] = "d" * 64
        result = self.evaluate(rows)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(len(result["primary_positive_geometries"]), 3)
        self.assertEqual(len(result["distinct_positive_road_hashes"]), 1)
        self.assertEqual(len(result["distinct_finished_parent_road_hashes"]), 1)

    def test_duplicate_nonqualifying_roads_still_block_fixed_24_road_cohort(self):
        rows = copy.deepcopy(self.rows)
        repeated = rows[12]["road_centerline_sha256"]
        for index in range(6, 24):
            for row in rows[2 * index:2 * index + 2]:
                row["road_centerline_sha256"] = repeated
        result = self.evaluate(rows)
        self.assertEqual(len(result["primary_positive_geometries"]), 3)
        self.assertEqual(len(result["primary_finished_parent_geometries"]), 3)
        self.assertEqual(result["distinct_observed_road_geometries"], 7)
        self.assertEqual(result["status"], "inconclusive")

    def test_unknown_image_review_prevents_pass_even_outside_required_categories(self):
        rows = copy.deepcopy(self.rows)
        rows[12]["image_flag"] = "unknown"
        result = self.evaluate(rows)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["unknown_image_slots"], [(410006, self.rules.primary_actor_id)])

    def test_censoring_and_unrun_keep_entire_schedule_and_never_count_failure(self):
        rows = copy.deepcopy(self.rows)
        row = rows[12]
        row.update(status="collection_censored", outcome="unknown", finished=False,
                   terminated=False, truncated=False, decisions=30, driven_raw_frames=120,
                   image_flag="unknown")
        result = self.evaluate(rows)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["uncertain_slots"], [(410006, self.rules.primary_actor_id, "collection_censored")])
        rows = copy.deepcopy(self.rows)
        row = rows[12]
        row.update(status="unrun", outcome=None, finished=None, terminated=None,
                   truncated=None, decisions=0, driven_raw_frames=0,
                   reset_initial_raw_frames=0, reset_noop_raw_frames=0,
                   image_flag=None, road_centerline_sha256=None)
        result = self.evaluate(rows)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["uncertain_slots"], [(410006, self.rules.primary_actor_id, "unrun")])
        with self.assertRaisesRegex(ValueError, "every scheduled"):
            self.evaluate(rows[:-1])

    def test_primary_source_and_exact_road_pair_are_immutable(self):
        with self.assertRaisesRegex(ValueError, "source actors"):
            CoverageRules(self.cells, "long-horizon-seed11", "entropy-v5-author-seed50", "a" * 64)
        rows = copy.deepcopy(self.rows)
        rows[1]["actor_id"] = self.rules.primary_actor_id
        with self.assertRaisesRegex(ValueError, "actor-road schedule"):
            self.evaluate(rows)
        rows = copy.deepcopy(self.rows)
        rows[1]["road_centerline_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "same road bytes"):
            self.evaluate(rows)
        for wrong_track, wrong_seed in ((True, 410000), (1, 410000.0)):
            with self.subTest(wrong_track=wrong_track, wrong_seed=wrong_seed):
                rows = copy.deepcopy(self.rows)
                rows[0]["track_id"] = wrong_track
                rows[0]["geometry_seed"] = wrong_seed
                with self.assertRaisesRegex(ValueError, "actor-road schedule"):
                    self.evaluate(rows)

    def test_image_review_and_true_finish_flags_are_not_replaced_by_proxy(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["image_review_blinded"] = False
        with self.assertRaisesRegex(ValueError, "blinded image review"):
            self.evaluate(rows)
        rows = copy.deepcopy(self.rows)
        rows[6]["finished"] = False
        with self.assertRaisesRegex(ValueError, "not a finish"):
            self.evaluate(rows)
        rows = copy.deepcopy(self.rows)
        rows[6]["truncated"] = False
        with self.assertRaisesRegex(ValueError, "not a finish"):
            self.evaluate(rows)
        rows = copy.deepcopy(self.rows)
        rows[0].update(outcome="task_timeout", terminated=False, truncated=True)
        with self.assertRaisesRegex(ValueError, "original full deadline"):
            self.evaluate(rows)

    def test_budget_and_invalid_censoring_fail_closed(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["driven_raw_frames"] = 481
        with self.assertRaisesRegex(ValueError, "raw-frame accounting"):
            self.evaluate(rows)
        rows = copy.deepcopy(self.rows)
        rows[0]["driven_raw_frames"] = 120
        with self.assertRaisesRegex(ValueError, "raw-frame accounting"):
            self.evaluate(rows)
        rows = copy.deepcopy(self.rows)
        rows[0]["decisions"] = 1
        rows[0]["driven_raw_frames"] = 4
        with self.assertRaisesRegex(ValueError, "101-negative-decision boundary"):
            self.evaluate(rows)
        rows[0]["outcome"] = "crash"
        with self.assertRaisesRegex(ValueError, "five damage increments"):
            self.evaluate(rows)
        rows = copy.deepcopy(self.rows)
        rows[0].update(status="collection_censored", outcome="off_track")
        with self.assertRaisesRegex(ValueError, "censoring must remain unknown"):
            self.evaluate(rows)
        with self.assertRaisesRegex(ValueError, "core-hour cap"):
            assess_coverage(self.rules, self.rows, core_hours_used=4.01)


if __name__ == "__main__":
    unittest.main()
