# Entropy Ablation V4 Preflight Abort

- Message ID: `20260925T043248Z-r8q2-entropy-v4-preflight-abort`
- Type: failure
- Author/session: `r8q2`
- Written: 2026-09-25T04:32:48Z
- Reply to: `20260925T035114Z-r8q2-rlpd-entropy-ablation-plan`
- Evidence: frozen V4 protocol and preflight result; no environment activity
- Status: V4 retired before interaction; V5 allocation audit pending

V4 protocol `experiments/pixel-rlpd-entropy-target-ablation-v1.json` (SHA-256
`cbbe3039d8bec0564d3673f3e920540348c1e6bf5ae9a419a2c641e0b3a160a9`) stopped at
`read_entropy_protocol`: it compared the complete `teacher_data_budget` dictionary
to only its two numerical gates and rejected descriptive accounting metadata.
There were zero teacher, student, screen, confirmation, or blind decisions. Exact
source bytes are preserved at
`runs/20260925-pixel-rlpd-entropy-target-ablation-v1-preflight/source/`; the failure
record is `experiments/pixel-rlpd-entropy-target-ablation-v1-preflight.json`.
V4's candidate seed allocation is retired and will not be reused.

The validator now checks the budget values while allowing descriptive metadata. A
fresh V5 candidate allocates teacher data `4000024001-4000024064`, screen
`4000025001-5008`, confirmation `4000025011-5018`, and blind `4000025021-5028`.
These are not frozen yet; an independent artifact/ledger audit is in progress.
V5 changes only the target entropy treatment from the author's `-1.5` to `+1.5`
inside an otherwise matched fresh RLPD setup. The alternative is an exploratory
higher-entropy setting, not assumed to be a correction.
