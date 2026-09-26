import copy
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.rlpd_entropy_common import EXPECTED_ARM_SPECS, STUDY_NAME, read_entropy_protocol
from scripts.run_rlpd_entropy_ablation import _paired_entropy_winner
from scripts.rlpd_common import write_json
from tests.test_rlpd_run import followup_protocol


def entropy_protocol():
    protocol = followup_protocol()
    protocol["name"] = STUDY_NAME
    protocol["student_training"]["arms"] = list(EXPECTED_ARM_SPECS)
    protocol["student_training"]["arm_specs"] = copy.deepcopy(EXPECTED_ARM_SPECS)
    protocol["student_training"]["learner_seeds"] = [40, 41]
    return protocol


class TestEntropyTargetAblation(unittest.TestCase):
    def test_entropy_arms_differ_only_in_target(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "protocol.json"
            write_json(path, entropy_protocol())
            protocol = read_entropy_protocol(path, verify_sources=False)
        specs = protocol["student_training"]["arm_specs"]
        self.assertEqual(specs["rlpd-author-target"]["target_entropy"], -1.5)
        self.assertEqual(specs["rlpd-positive-target"]["target_entropy"], 1.5)
        self.assertEqual(specs["rlpd-author-target"]["algorithm"], specs["rlpd-positive-target"]["algorithm"])
        self.assertEqual(specs["rlpd-author-target"]["use_offline"], specs["rlpd-positive-target"]["use_offline"])
        self.assertEqual(protocol["student_training"]["steps_per_run"], 131072)
        self.assertEqual(protocol["student_training"]["learner_seeds"], [40, 41])

    def test_entropy_arm_change_cannot_smuggle_a_second_hyperparameter(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "protocol.json"
            protocol = entropy_protocol()
            protocol["student_training"]["arm_specs"]["rlpd-positive-target"]["use_offline"] = False
            write_json(path, protocol)
            with self.assertRaisesRegex(ValueError, "differ only"):
                read_entropy_protocol(path, verify_sources=False)

    def test_fixed_teacher_budget_allows_frozen_accounting_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "protocol.json"
            protocol = entropy_protocol()
            protocol["teacher_data_budget"].update({
                "partial_episode": "drop and account, never synthesize terminal",
                "source_repeats": "one attempt per geometry",
            })
            write_json(path, protocol)
            checked = read_entropy_protocol(path, verify_sources=False)
        self.assertEqual(checked["teacher_data_budget"]["decisions"], 16384)
        self.assertEqual(checked["teacher_data_budget"]["minimum_distinct_finishes"], 4)
        self.assertIn("partial_episode", checked["teacher_data_budget"])

    def test_target_scalar_only_reverses_temperature_gradient_at_v3_entropy(self):
        entropy = torch.tensor(-1.25)
        gradients = {}
        for target in (-1.5, 1.5):
            beta = torch.tensor(torch.log(torch.tensor(0.1)).item(), requires_grad=True)
            alpha = beta.exp()
            (alpha * (entropy - target)).backward()
            gradients[target] = float(beta.grad)
        self.assertGreater(gradients[-1.5], 0.0)
        self.assertLess(gradients[1.5], 0.0)

    def test_confirmation_requires_seedwise_paired_dominance(self):
        def row(finishes):
            return {
                "canonical_episodes": 32,
                "eligible": True,
                "determinism_audited": True,
                "cpu_reload_matches": True,
                "operational_failures": 0,
                "canonical_finishes": finishes,
            }

        author_wins = {
            "rlpd-author-target-seed20": row(5),
            "rlpd-author-target-seed21": row(7),
            "rlpd-positive-target-seed20": row(4),
            "rlpd-positive-target-seed21": row(6),
        }
        self.assertEqual(_paired_entropy_winner(author_wins, [20, 21]), "rlpd-author-target")

        positive_wins = {
            "rlpd-author-target-seed20": row(4),
            "rlpd-author-target-seed21": row(6),
            "rlpd-positive-target-seed20": row(5),
            "rlpd-positive-target-seed21": row(7),
        }
        self.assertEqual(_paired_entropy_winner(positive_wins, [20, 21]), "rlpd-positive-target")

        crossed = dict(author_wins)
        crossed["rlpd-positive-target-seed21"] = row(8)
        self.assertIsNone(_paired_entropy_winner(crossed, [20, 21]))

        zero = dict(author_wins)
        zero["rlpd-positive-target-seed20"] = row(0)
        self.assertIsNone(_paired_entropy_winner(zero, [20, 21]))


if __name__ == "__main__":
    unittest.main()
