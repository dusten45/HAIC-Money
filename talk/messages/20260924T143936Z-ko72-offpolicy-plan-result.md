# Two Independent Off-Policy Plans Documented

- Message ID: `20260924T143936Z-ko72-offpolicy-plan-result`
- Type: result
- Author/session: `ko72`
- Written: 2026-09-24T14:39:36Z
- Reply to: `20260924T143021Z-ko72-offpolicy-plans`
- Evidence: observed for documentation/code contracts; hypothesis for performance
- Status: closed

Added `docs/plans/drqv2-teacher-replay-plan.md` and
`docs/plans/pixel-rlpd-offpolicy-plan.md`. They specify independent study
protocols, data and source lineage, full learner/inference boundaries,
same-initialization online-only controls, fresh CPU cells, costs, correctness
tests, and stopping gates. Neither claims an HAIC performance result or
changes the currently active Dreamer plan.

Material audit finding: new evaluation geometry must be disjoint not only
from both studies' new online/teacher collection but also from the original
pad-4 DrQ teacher actors' **131,072-decision training episode ledgers**.
The plans require a source `episodes.jsonl`/sampler audit before either study
freezes fresh cells. Independent pixel RLPD uses actual pixel-config Q10,
target subset one and no entropy Bellman backup; DrQ prior replay keeps its
native actor/tag and does not claim to reproduce RLPD.

The only actions in this session were document editing and read-only source
checks. No code, dataset, training, evaluation, package, or official external
action was executed. All planned budgets and success thresholds remain
unmeasured proposals until their respective protocol freezes.
