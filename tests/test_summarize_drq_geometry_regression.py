import unittest

from scripts.summarize_drq_geometry_regression import markers


class TestPredeclaredFailureMarkers(unittest.TestCase):
    def test_majority_rules_and_nonexclusive_labels(self):
        pair = {"variant": "uniform", "source_seed": 0, "geometry_seed": 7,
                "family": "example", "transition": "lost", "source_steps": 400,
                "candidate_steps": 300, "source_max_progress": 1.0,
                "candidate_max_progress": .90,
                "divergence": {"initial_native_linf_difference": .10,
                               "first_sustained_action_decision": 1}}
        early = [{"variant": "uniform", "source_seed": 0, "geometry_seed": 7,
                  "decision": i, "action_linf_difference": .10,
                  "source_q1_ranking_variant_minus_source": -1 if i <= 10 else 1,
                  "variant_q1_ranking_variant_minus_source": 1 if i <= 10 else -1,
                  "source_q_min_ranking_variant_minus_source": -1,
                  "variant_q_min_ranking_variant_minus_source": 1,
                  "source_actor_encoder_cosine": .89 if i <= 10 else .91,
                  "variant_critic": {"q1": {"source_action": 1}, "q2": {"source_action": 1}},
                  "source_critic": {"q1": {"source_action": 1}, "q2": {"source_action": 1}}}
                 for i in range(1, 21)]
        result = markers(pair, early)
        self.assertEqual(result["labels"], ["early_policy_drift_marker",
                                            "critic_q1_reversal_marker",
                                            "actor_encoder_shift_marker",
                                            "finish_or_late_stage_marker"])
        self.assertEqual(result["first20_source_critic_q1_prefers_source_r6_critic_q1_prefers_r6"], 10)
