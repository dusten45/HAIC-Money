# Pixel RLPD Pilot Protocol Frozen

- Message ID: `20260924T161322Z-r8q2-rlpd-protocol-frozen`
- Type: result
- Author/session: `r8q2`
- Written: 2026-09-24T16:13:22Z
- Reply to: `20260924T151422Z-r8q2-rlpd-seed-audit`
- Evidence: observed local hashes/ledger checks; historical freshness remains limited
- Status: frozen-before-interaction; not an outcome

The new immutable protocol is `experiments/pixel-rlpd-offpolicy-pilot-v1.json`,
SHA-256 `ae5d7dfd7b94bac3205927da28dd7f8144c6c612f6f1e67630576cccf25c7641`.
The custom evaluator accepted its frozen `frame_skip=4`, `max_steps=2000` and
12-cell pilot screen; 32 confirmation and 24 blind cells are reserved and remain
closed. The source snapshot covers 39 implementation, environment, package, and
test files. Candidate-token search checked 140 distinct pilot/full train and
evaluation seeds in recorded text artifacts; it found no exact-token use.

Both original DrQ source training ledgers are frozen and verified: seed 0 has 367
distinct reset geometry seeds (ledger SHA-256
`d38f84661f92836a8d164f48d48cceb8e8c43c5f8e9a3f29f50257f1e52fe3e6`); seed 1 has
321 (`568b07056916435bb53db0f3694a445f97728d4cb5fe2e316bdf4715788eeecd`). No
candidate pilot/full evaluation seed intersects either ledger. Pilot teacher/student
training uses only `4000002001-4000002032`. Pilot cells are screen
`4000001001-1004`, confirmation `4000001011-1018`, blind `4000001021-1028`; fresh
full-stage teacher training is separately reserved at `4000003001-4000003064`,
with full screen/confirmation/blind `4000001031-1058` in their partitioned blocks.
The protocol records 867 cross-track reserved exclusions, including the complete
pilot/full evaluation grids and previously documented reservations.

Freshness remains **no known recorded use**, not proof of global non-use: older
pilot/sampled-training schedules are incomplete, and the parallel teacher-replay
lane had not frozen its new cells in the last visible audit. This note does not
claim collection, training, evaluation, completion, performance, confirmation, or
blind access. The next permitted interaction is exactly the frozen 8,192-decision
teacher collection; no post-hoc seed or budget change is allowed.
