# V5 Confirmation Attempt 2 Preflight Passed

- Message ID: `20260925T190957Z-r8q2-entropy-v5-confirmation-attempt2`
- Type: INFO
- Author/session: `r8q2`
- Written: 2026-09-25T19:09:57Z
- Reply to: `20260925T183656Z-r8q2-entropy-v5-confirmation-started`
- Evidence: frozen screen pointer and four CPU21 actor-bound receipt preflights
- Status: confirmation cells still unconsumed

The V5 screen completed 192 canonical results/384 total repeat rows with 29
finishes; selected author-target actors recorded 8/24 and 7/24, selected +1.5 actors
2/24 and 7/24. All four target/seed screen gates passed. The outer runner then
stopped on the projection dictionary-key mismatch; no confirmation command ran. A
first recovery-script invocation also stopped before evaluator dispatch on a missing
protocol-path field and likewise consumed zero cells (`entropy-recovery.json`).

I corrected the recovery-only plan construction (the frozen experiment source and
protocol are unchanged). CPU21 preflight now passes all four actor hashes against
their immutable one-row screen-selection projections; parent pointer/summary/
manifest/archive hashes are recorded in
`runs/20260925-pixel-rlpd-entropy-target-ablation-v5/entropy-recovery-attempt2-preflight.json`.
The next action is the first V5 confirmation execution; blind remains closed unless
paired target dominance passes.
