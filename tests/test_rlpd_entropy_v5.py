from pathlib import Path
import tempfile
import unittest

from scripts.rlpd_common import write_json
from scripts.rlpd_entropy_common import EXPECTED_ARM_SPECS
from scripts.rlpd_entropy_v5_common import STUDY_NAME, read_entropy_v5_protocol
from tests.test_rlpd_entropy_ablation import entropy_protocol


class TestEntropyV5Protocol(unittest.TestCase):
    def test_v5_name_seed_and_budget_contract_roundtrips(self):
        protocol = entropy_protocol()
        protocol["name"] = STUDY_NAME
        protocol["student_training"]["learner_seeds"] = [50, 51]
        protocol["student_training"]["arms"] = list(EXPECTED_ARM_SPECS)
        protocol["student_training"]["arm_specs"] = EXPECTED_ARM_SPECS
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "entropy-v5.json"
            write_json(path, protocol)
            checked = read_entropy_v5_protocol(path, verify_sources=False)
        self.assertEqual(checked["name"], "pixel-rlpd-entropy-target-ablation-v5")
        self.assertEqual(checked["student_training"]["learner_seeds"], [50, 51])
        self.assertEqual(checked["teacher_data_budget"]["decisions"], 16384)
        self.assertEqual(checked["student_training"]["steps_per_run"], 131072)

    def test_v5_rejects_relabelled_entropy_arm_or_seed(self):
        base = entropy_protocol()
        for mutate, error in (
            (lambda p: p["student_training"]["arm_specs"]["rlpd-positive-target"].update({"target_entropy": -1.5}), "target"),
            (lambda p: p["student_training"].update({"learner_seeds": [40, 41]}), "budget"),
        ):
            protocol = dict(base)
            protocol["student_training"] = {
                key: value.copy() if isinstance(value, dict) else value
                for key, value in base["student_training"].items()
            }
            protocol["student_training"]["arm_specs"] = {
                key: dict(value) for key, value in base["student_training"]["arm_specs"].items()
            }
            protocol["name"] = STUDY_NAME
            protocol["student_training"]["learner_seeds"] = [50, 51]
            mutate(protocol)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "invalid.json"
                write_json(path, protocol)
                with self.assertRaises(ValueError):
                    read_entropy_v5_protocol(path, verify_sources=False)


if __name__ == "__main__":
    unittest.main()
