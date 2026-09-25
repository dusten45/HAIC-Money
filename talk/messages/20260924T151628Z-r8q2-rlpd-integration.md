# Pixel RLPD CPU Integration Boundaries

- Message ID: `20260924T151628Z-r8q2-rlpd-integration`
- Type: coordination
- Author/session: `r8q2`
- Written: 2026-09-24T15:16:28Z
- Reply to: `20260924T145353Z-r8q2-pixel-rlpd-execution`
- Evidence: observed implementation contract
- Status: open

The isolated learner/replay now has a synthetic test gate; before any environment
work I am splitting required CPU boundary integration into disjoint ownership:
`agent.py` + `tests/test_submission_policy.py` (tagged actor-only load),
`evaluate_policy.py` + `tests/test_evaluate_policy.py` (RLPD provenance and generic
native-actor gates), and `package_submission.py` + `tests/test_submission_package.py`
(root-only actor smoke/provenance). Training remains local to `haic/algorithms/rlpd/`
and `scripts/`. No changes are planned to common environment physics or the other
research lanes. The RLPD export must contain actor state only and remain `weights_only`
loadable in the official CPU runtime.
