from __future__ import annotations

import unittest

from scripts.freeze_drq_geometry_mix import VARIANTS


class GeometryMixProtocolDefinitionTests(unittest.TestCase):
    def test_three_predeclared_distributions_cover_all_six_families(self) -> None:
        self.assertEqual([row["name"] for row in VARIANTS],
                         ["uniform", "failure_weighted", "easy_retention"])
        family_set = set(VARIANTS[0]["family_tiers"])
        self.assertEqual(len(family_set), 6)
        for variant in VARIANTS:
            self.assertEqual(set(variant["family_tiers"]), family_set)
            self.assertEqual(set(variant["tier_weights"]), {"easy", "boundary", "difficult"})
            self.assertAlmostEqual(sum(variant["tier_weights"].values()), 1.0)
            self.assertEqual(set(variant["family_tiers"].values()) -
                             {"easy", "boundary", "difficult"}, set())

    def test_each_idea_retains_easy_roads_and_applies_a_distinct_fixed_distribution(self) -> None:
        easy_anchor = "easy-curvature-anchor"
        distributions = {row["name"]: row for row in VARIANTS}
        self.assertTrue(all(row["family_tiers"][easy_anchor] == "easy" for row in (
            distributions["failure_weighted"], distributions["easy_retention"]
        )))
        uniform = distributions["uniform"]
        self.assertEqual(set(uniform["family_tiers"].values()), {"boundary"})
        self.assertNotEqual(distributions["failure_weighted"]["tier_weights"],
                            distributions["easy_retention"]["tier_weights"])
        self.assertNotEqual(distributions["uniform"]["tier_weights"],
                            distributions["failure_weighted"]["tier_weights"])


if __name__ == "__main__":
    unittest.main()
