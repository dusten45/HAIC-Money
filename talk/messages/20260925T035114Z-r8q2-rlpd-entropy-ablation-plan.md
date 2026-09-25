# RLPD Entropy Ablation Scope

- Message ID: `20260925T035114Z-r8q2-rlpd-entropy-ablation-plan`
- Type: coordination
- Author/session: `r8q2`
- Written: 2026-09-25T03:51:14Z
- Reply to: `20260925T033804Z-r8q2-rlpd-followup-v1-result`
- Evidence: V3 result receipt; next mechanism is untested
- Status: new study proposed within owner-directed iteration

V3 completed screen, strict confirmation and one final blind; it supports the
expanded RLPD recipe as an internal candidate, but does not identify why it
outperformed its SAC control. The temperature target remains a plausible, untested
mechanism. I propose a one-factor matched RLPD ablation: same fresh teacher dataset,
same online/offline ratio, architecture, 131,072 student decisions and learner RNG
seeds, changing only `target_entropy` from pinned author `-1.5` to an explicit
positive-entropy alternative `+1.5`. Use two new learner seeds (20/21) and new
geometry pools: teacher data `4000022001-2064`, screen `4000023001-3008`, confirmation
`4000023011-3018`, blind `4000023021-3028`. These candidate ranges are not locked
until an independent artifact/ledger audit completes. Neither V3 data/checkpoints
nor V1/V2/V3 consumed or reserved cells will be reused.

New-only implementation/protocol files planned are
`scripts/freeze_rlpd_entropy_ablation.py`, `scripts/train_rlpd_entropy_ablation.py`,
`scripts/run_rlpd_entropy_ablation.py`, `scripts/project_rlpd_entropy_receipts.py`,
`tests/test_rlpd_entropy_ablation.py`, and an independent frozen experiment JSON.
I will leave all V3 hashed sources and artifacts untouched. The two treatments will
share the new offline dataset and per-seed indexed environment stream; screen and
confirmation compare target variants on the same cells, and blind is one
confirmation-selected model only if all predeclared gates pass. This is an
exploratory entropy-convention ablation, not an assumption that changing the sign
will help. Do not open any official competition evaluation/submission path.
