# Dreamer B1 Failure Audit Result
- Message ID: 20260925T002014Z-q1d8-dreamer-b1-diagnosis-result
- Type: result
- Author/session: q1d8
- Written: 2026-09-25T00:20:14Z
- Reply to: 20260924T235100Z-q1d8-dreamer-diagnostic-scope
- Evidence: measured and source-audited
- Status: complete for diagnosis; B1/policy gate remains blocked

The standalone diagnosis is `experiments/dreamerv3-b1-failure-diagnosis-v1.json`.
It verifies protocol/checkpoint/dataset hashes for all 18 v1-v9 B1 runs and
keeps all frozen results intact. v9's `pos_weight=56` scaled the **majority
continue=1** class instead of terminal=0, a confirmed implementation/intent
mismatch. B1's terminal-anchor sampling enriches positives in scored windows
but compares BCE to a constant based on **training** prevalence, so v8's
reported BCE improvement over that baseline disappears against the matched
sampled-window constant. Its reward Spearman ranks episode averages, not
alternative actions at one state. All studies still genuinely fail B1 as
frozen; none proves a Dreamer actor cannot learn.

The exact-source v9 latent probe found both posterior/prior linear
10-decision-before-failure AUC near chance despite small video MSE gains.
Training replay terminal frequency was 35/12288 or 36/12288; final progress
never reached 20% on completed random episodes. All 32 dev episodes were
off-track failures, with no per-transition progress or damage saved. An actual
four-decision training-only local trace showed no action/result offset in the
tested path; v8/v9 logs had zero timeout/finish events. A new read-only matched
short/8/10/full-context diagnostic on the same v9 anchors found zero terminal
recall at every context in both seeds; the seed0 report reproduced byte-for-byte.
Results are `experiments/dreamerv3-b1-context-audit-v9-seed{0,1}.json`, produced
by `scripts/diagnose/dreamerv3_b1_context_audit.py`.

No new learner training, actor rollout, policy screen, confirmation/blind,
official site action, or changes to another agent's DrQ/RLPD/Agent code were
made. Source-of-truth pointers were added only to the Dreamer plan,
`docs/experiments/INDEX.md`, and `docs/context/current-state.md`; neighboring
concurrent edits were preserved.
