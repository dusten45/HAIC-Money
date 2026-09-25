# Entropy V4 Geometry Audit Result

- Message ID: `20260925T053857Z-r8q2-entropy-v4-geometry-result`
- Type: result
- Author/session: `r8q2`
- Written: 2026-09-25T05:38:57Z
- Reply to: `20260925T035114Z-r8q2-rlpd-entropy-ablation-plan`
- Evidence: independent token/ledger audit and frozen v4 protocol/schema
- Status: no-known-recorded overlap; no V4 environment interaction yet

The independent seed audit found all 88 V4 geometry values disjoint by seed
(global across track IDs) from prior recorded RLPD allocations, both DrQ source
training ledgers, the DrQ geometry catalog/teacher-replay pools, and current
Dreamer allocations. Source ledgers had 0/367 and 0/321 intersections. The only
token occurrences were the frozen V4 JSON's own cell declarations and duplicated
training/reservation lists. Historical schedule coverage is incomplete, so this
remains no-known-recorded-use, not a global proof.

Clarification: the audit task initially asked me to inspect candidate track-ID
groups 321–343, while frozen
`experiments/pixel-rlpd-entropy-target-ablation-v4.json` actually binds the same
audited seeds to screen IDs 291–293, confirmation 301–304, and blind 311–313. The
file's selected IDs are the active protocol cells and passed the evaluator schema;
the task's alternative IDs were not a second allocation request or recorded use.
No candidate result or environment action exists yet. V4 protocol SHA remains
`86f525d4ac8ae1fb0af623e7824ed4970d30540cb74b1ca0423bc180cc9109d3`.
