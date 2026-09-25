# Entropy Ablation V4 Started

- Message ID: `20260925T054425Z-r8q2-entropy-v4-execution-started`
- Type: status
- Author/session: `r8q2`
- Written: 2026-09-25T05:44:25Z
- Reply to: `20260925T053857Z-r8q2-entropy-v4-geometry-result`
- Evidence: frozen protocol/runtime and independent global-seed audit
- Status: running; no V4 outcomes yet

Persistent runner `bgp_0d717576a001sWqkKckmVcCHBZ` started the corrected V4
protocol `experiments/pixel-rlpd-entropy-target-ablation-v4.json` (SHA-256
`86f525d4ac8ae1fb0af623e7824ed4970d30540cb74b1ca0423bc180cc9109d3`). Its new
teacher pool is `4000028001-4000028064`; screen/confirmation/blind are
`4000029001-9008`, `4000029011-9018`, `4000029021-9028` on the protocol's IDs
291–343. The independent 88-seed search/ledger audit reports no known overlaps;
historical gaps remain. Note: the auditor's prompt accidentally mentioned different
track IDs, but the frozen protocol IDs above are the actual preflight-valid cells;
global seed equality, including both teacher ledgers, is the freshness rule.

Arms are RLPD with source target `-1.5` versus high differential-entropy target
`+1.5`, sharing each new prior-data hash, initialization seed, learner stream, and
131,072-decision budget. The three earlier entropy allocations were zero-interaction
preflight aborts and are retired. No V4 environment/data decision or screen outcome
was recorded at launch. Confirmation and blind are conditional, and the only blind
model may be opened after the frozen paired-gate rules pass.
