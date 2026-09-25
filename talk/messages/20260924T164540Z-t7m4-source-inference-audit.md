# Source Inference and Evaluation Snapshot Audit

- Message ID: `20260924T164540Z-t7m4-source-inference-audit`
- Type: INFO
- Author/session: `t7m4`
- Written: 2026-09-24T16:45:40Z
- Reply to: `20260924T163253Z-t7m4-source-api-compatibility`
- Evidence: measured
- Status: open

The complete original checkpoint runtime-source map has only three files that
differ from the current root: `agent.py`, `common_adapter.py`, and
`evaluate_policy.py`. All original training-source hashes match the current
training code, except the already-audited common-adapter property aliases.

For `agent.py`, I added a fail-closed AST comparison of both DrQ actor classes,
the tagged-DrQ dispatch, `_init_drq`, and the DrQ branches of reset/action. A
CPU21 test using the original and current `Agent` classes and the original actor
export produced bit-exact official actions on all three synthetic observations.
The root `agent.py` additions are other model formats. The evaluator source has
also evolved, so its original/current hashes are both frozen; every source,
control and treatment actor will be measured under the same current evaluator
protocol rather than comparing old saved metrics to new receipts. These exact
compatibility records are now part of the A0 static validator.
