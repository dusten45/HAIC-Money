from pathlib import Path
import tempfile
import unittest

from scripts.recover_rlpd_entropy_v5_attempt4 import (
    _assert_actor_identity,
    _confirmation_target_winner,
    _write_exclusive_json,
    _write_json_if_absent_or_equal,
)


def _confirmation(finishes, *, eligible=True):
    return {
        "canonical_episodes": 32,
        "canonical_finishes": finishes,
        "eligible": eligible,
        "determinism_audited": True,
        "cpu_reload_matches": True,
        "operational_failures": 0,
    }


class TestEntropyV5Attempt4Gate(unittest.TestCase):
    def setUp(self):
        self.protocol = {
            "student_training": {
                "arms": ["rlpd-author-target", "rlpd-positive-target"],
                "learner_seeds": [50, 51],
            }
        }

    def test_target_winner_requires_paired_strict_dominance(self):
        confirmations = {
            "rlpd-author-target-seed50": _confirmation(17),
            "rlpd-author-target-seed51": _confirmation(12),
            "rlpd-positive-target-seed50": _confirmation(8),
            "rlpd-positive-target-seed51": _confirmation(12),
        }
        self.assertEqual(
            _confirmation_target_winner(self.protocol, confirmations),
            "rlpd-author-target",
        )

    def test_target_winner_rejects_crossed_tied_or_ineligible_cells(self):
        crossed = {
            "rlpd-author-target-seed50": _confirmation(17),
            "rlpd-author-target-seed51": _confirmation(5),
            "rlpd-positive-target-seed50": _confirmation(8),
            "rlpd-positive-target-seed51": _confirmation(12),
        }
        self.assertIsNone(_confirmation_target_winner(self.protocol, crossed))

        tied = {
            "rlpd-author-target-seed50": _confirmation(17),
            "rlpd-author-target-seed51": _confirmation(12),
            "rlpd-positive-target-seed50": _confirmation(17),
            "rlpd-positive-target-seed51": _confirmation(12),
        }
        self.assertIsNone(_confirmation_target_winner(self.protocol, tied))

        tied["rlpd-positive-target-seed51"] = _confirmation(11, eligible=False)
        self.assertIsNone(_confirmation_target_winner(self.protocol, tied))


class TestEntropyV5Attempt4RecoverySafety(unittest.TestCase):
    def test_recovery_claim_is_exclusive_and_preserves_first_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "attempt.json"
            first = {"status": "running"}
            _write_exclusive_json(path, first)
            with self.assertRaises(FileExistsError):
                _write_exclusive_json(path, {"status": "second"})
            self.assertEqual(path.read_text(), '{\n  "status": "running"\n}\n')

    def test_result_finalization_writes_missing_identical_artifacts_only(self):
        with tempfile.TemporaryDirectory() as temp:
            existing = Path(temp) / "existing.json"
            missing = Path(temp) / "missing.json"
            result = {"status": "complete", "finishes": 7}
            _write_exclusive_json(existing, result)
            _write_json_if_absent_or_equal(existing, result)
            _write_json_if_absent_or_equal(missing, result)
            self.assertEqual(existing.read_text(), missing.read_text())
            with self.assertRaises(ValueError):
                _write_json_if_absent_or_equal(existing, {"status": "changed"})

    def test_actor_embedded_identity_rejects_relabelled_selection(self):
        selected = {"actor_sha256": "actor-sha", "environment_steps": 131072}
        actor = {
            "archive_sha256": "actor-sha",
            "algorithm": "rlpd",
            "export_metadata": {
                "format": "haic-rlpd-pixel-actor-v1",
                "protocol_sha256": "protocol-sha",
                "training_seed": 50,
                "environment_steps": 131072,
                "environment_contract": {
                    "arm": "rlpd-author-target",
                    "training_seed": 50,
                    "offline_dataset_sha256": "dataset-sha",
                },
            },
        }
        _assert_actor_identity(
            actor, selected, "rlpd-author-target", 50, "protocol-sha", "dataset-sha", "seed50"
        )
        actor["export_metadata"]["environment_contract"]["arm"] = "rlpd-positive-target"
        with self.assertRaisesRegex(ValueError, "embedded V5 arm/seed/data identity"):
            _assert_actor_identity(
                actor, selected, "rlpd-author-target", 50, "protocol-sha", "dataset-sha", "seed50"
            )


if __name__ == "__main__":
    unittest.main()
