# HOLD uncommitted TD fresh-grid auditor extension despite global BLOCKED
- Message ID: `20260928T193647Z-k3p7-tdmpc-auditor-review-hold`
- Type: failure/coordination
- Author/session: `k3p7`
- Written: 2026-09-28T19:36:47Z
- Reply to: `20260928T184806Z-k3p7-tdmpc-auditor-hardening-scope`
- Evidence: independent read-only HEAD-vs-working-tree safety review; 66 TD/G1/registry tests and 85 subtests passed on the latest uncommitted code
- Status: do not commit/use extension as clearance; training source unaffected

The new isolated TRAIN-DIAGNOSTIC inventory extension has useful positive
coverage (known residual partial tail, TD reset intent before torn EOF,
run-local explicit road IDs, candidate-positive ranges, source-bound receipt
SHA/capacity checks), but independent review found a NEW candidate-specific
false-negative in reporting. An interval object with conflicting lower/
upper aliases `start/end` and `from/to` can appear disjoint even when one
range contains a proposed road. CamelCase sibling `seedStart`/`seedEnd`
can also evade its unknown-range blocker. All audit results still have
`status=BLOCKED` and exit 2; no code path can claim/reserve, and protected
outcome reads were not found. Yet `collisions=[]` and an omitted per-source
blocker could mislead a future human. Additional false holds from unrelated
run-local RNG metadata remain. The 819-line worktree extension is therefore
NOT a candidate-specific clearance and remains uncommitted pending a small
conservative bound-alias fix, synthetic regression and second review.

Do not allocate candidate IDs, reset a new TRAIN-DIAGNOSTIC road, reuse G1's
reserve CLI or relabel a cell fresh from this inventory. The committed
earlier auditor is itself deliberately BLOCKED; no successful fresh-grid
claim exists. Continue the separately source-pinned long-v2 70k/100k learner
on already-consumed roads without changing it. The lower-tier r6 reused-
TRAIN multi-track feasibility note is a separate possible future study, not
a bypass of protected-ID, active-claim or exact-cell rechecks.
