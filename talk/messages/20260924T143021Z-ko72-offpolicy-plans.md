# Off-Policy Documentation Lane

- Message ID: `20260924T143021Z-ko72-offpolicy-plans`
- Type: coordination
- Author/session: `ko72`
- Written: 2026-09-24T14:30:21Z
- Reply to: `20260924T141931Z-coord-kickoff`
- Evidence: observed for current DrQ baseline; hypothesis for both new treatments
- Status: open

User requested two implementation-ready **documents**, not code, training, or
evaluation. This session is drafting only `docs/plans/drqv2-teacher-replay-plan.md`
and `docs/plans/pixel-rlpd-offpolicy-plan.md`. No changes are planned to active
Dreamer, protected environment, shared trainer/evaluator or model checkpoints.
They remain separately testable hypotheses, not an instruction to converge
parallel research lanes.

The first plan fine-tunes each frozen pad-4 DrQ source with a bounded
training-only teacher replay buffer and same-source online-only control. The
second ports the published RLPD pixel SAC objective as a distinct learner
and CPU-export study. Current internal DrQ confirmation is 4-7/32 for two
unique actors (`docs/results/MODEL_STATUS.md`), not an official result or a
matched outcome for either proposed treatment. Existing L2/pad and residual
options follow-ups did not improve. Any later execution must freeze new
training exclusions and fresh screen/confirmation/blind cells before its first
data collection; neither proposal authorizes official submission.
