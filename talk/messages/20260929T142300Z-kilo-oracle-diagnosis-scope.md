# Oracle-guided failure diagnosis scope
- Message ID: `20260929T142300Z-kilo-oracle-diagnosis-scope`
- Type: coordination
- Author/session: `kilo`
- Written: 2026-09-29T14:23:00Z
- Reply to: none
- Evidence: observed repository state
- Status: open

User-directed work will treat `haic/oracle_v1/` as immutable and will not access
Fresh or recreate a Fresh link. I am evaluating a read-only comparison of existing
canonical DrQ source traces (16 consumed TRAIN-DIAGNOSTIC roads, two actors) and
RLPD G0 traces (12 consumed TRAIN roads, two actors). Candidate files are isolated
to new `scripts/diagnose_oracle_policy_failures.py`, a focused test, a new
`experiments/oracle-policy-diagnosis-v1*.json` protocol/result, a new report under
`docs/experiments/`, and a dedicated `runs/oracle-policy-diagnosis-v1/` output.
No current shared docs, policy weights, learner, live RLPD G1 run, or DrQ source
artifact will be edited. Existing cell status, source/runtime parity, available
trace fields and a safe action-intervention gate are still being verified before
collection; no new training or protected evaluation is in scope.
