import unittest

from scripts.diagnose.drqv2_residual_option_prefix_branches import (
    _seed_values,
    compare_prefix_signature,
)


class TestPrefixBranchDiagnostics(unittest.TestCase):
    def test_exact_step_signature_passes_when_every_contract_field_matches(self):
        expected = {
            "observation_sha256": "obs",
            "official_action": [0.1, 0.5, 0.0],
            "reward": 2.25,
            "terminated": False,
            "truncated": False,
            "info": {"progress": 0.25, "damage": 0.0},
            "state": {"position": [1.0, 2.0], "damage": 0.0},
        }
        self.assertIsNone(compare_prefix_signature(expected, dict(expected)))

    def test_prefix_signature_reports_first_observation_action_or_physics_mismatch(self):
        expected = {
            "observation_sha256": "obs",
            "official_action": [0.1, 0.5, 0.0],
            "reward": 2.25,
            "terminated": False,
            "truncated": False,
            "info": {"progress": 0.25},
            "state": {"position": [1.0, 2.0]},
        }
        for key in ("observation_sha256", "official_action", "reward", "terminated", "truncated", "info", "state"):
            observed = dict(expected)
            observed[key] = "different"
            with self.subTest(key=key):
                self.assertEqual(compare_prefix_signature(expected, observed), key)

    def test_reset_signature_compares_reset_identity(self):
        expected = {
            "observation_sha256": "obs",
            "reset_identity": {"track_id": 1, "seed": 4200000001},
            "state": {"damage": 0.0},
        }
        observed = dict(expected)
        observed["reset_identity"] = {"track_id": 2, "seed": 4200000001}
        self.assertEqual(compare_prefix_signature(expected, observed), "reset_identity")

    def test_seed_audit_collects_all_seed_named_fields(self):
        document = {
            "reserved_training_seeds": [101, 102],
            "partitions": {"screen": {"seeds": [201, 202]}},
            "nested": {"development_screen": {"geometry_seeds": [301]}},
            "other": [{"environment_sampler_seed": 401}],
        }
        self.assertEqual(_seed_values(document), {101, 102, 201, 202, 301, 401})


if __name__ == "__main__":
    unittest.main()
