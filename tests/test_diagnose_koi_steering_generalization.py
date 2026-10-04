from copy import deepcopy

from scripts import diagnose_koi_steering_generalization as diagnose
from tests.test_analyze_koi_steering_generalization import full_pairs


def test_paired_denominator_excludes_unmatched_episode_and_keeps_counterexamples():
    slots, measured = full_pairs()
    excluded_key = (3, diagnose.analysis.SEEDS[-1], diagnose.analysis.ARMS[1])
    del measured[excluded_key]
    defect = (1, diagnose.analysis.SEEDS[2], diagnose.analysis.ARMS[1])
    measured[defect][1]["completed"] = False
    measured[defect][0]["completed"] = False
    result = diagnose.analysis.summarize(slots, measured)
    episodes = {}
    for k, (e, r) in measured.items():
        e = deepcopy(e)
        states = r["obstacles"][0]["_states"]
        for s in states:
            s["clearance"] = [2.] * 6
            s["contacts"] = []
        e["initial_state"] = states[0]
        episodes[k] = e, states[1:]
    output = diagnose.analyze(result, episodes)
    assert output["matched_pairs"] == 71
    assert output["excluded_unmatched_episodes"] == 1
    assert all(a["episode_denominator"] == 71 and a["object_denominator"] == 426 for a in output["paired_arm_summaries"].values())
    assert output["paired_completion"]["lost"] == 1
    assert output["irreversible_predeclared_failure"]
    assert output["decision"] == "NOT_GENERALIZED_ADOPTION_CANDIDATE"
    assert len(output["lost_finish_failure_diagnostics"]) == 1
