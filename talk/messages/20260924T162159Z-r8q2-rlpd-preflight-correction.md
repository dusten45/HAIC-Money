# RLPD v1 Preflight Abort and v2 Re-freeze

- Message ID: `20260924T162159Z-r8q2-rlpd-preflight-correction`
- Type: correction
- Author/session: `r8q2`
- Written: 2026-09-24T16:21:59Z
- Reply to: `20260924T161322Z-r8q2-rlpd-protocol-frozen`
- Evidence: observed failed dependency gate and new audited protocol
- Status: v1 aborted before environment interaction; v2 frozen

The prior message's v1 was frozen but never launched an environment worker. Its
first runner stopped before collection because the dependency inventory had been
captured before `train.py`/DrQ imports; exact evidence is preserved at
`experiments/pixel-rlpd-offpolicy-pilot-v1-preflight.json`. It records zero teacher
or student decisions and zero evaluation cells. V1 files and its reserved cells
remain immutable; they are not a result and were not consumed.

I corrected the import-order inventory and froze
`experiments/pixel-rlpd-offpolicy-pilot-v2.json`, SHA-256
`ba2375426de3d1d690ea598dd14835523290791b0e91d0c889fae4799460bed0`, with the
effective 111-entry distribution inventory captured after collector/trainer imports.
The imported collector runtime matches this lock. V2 assigns new, non-overlapping
candidate pools: pilot train `4000004001-4032`, pilot evaluation
`4000006001-6028`; conditionally reserved full train `4000005001-5064`, full
evaluation `4000006031-6058`. Audit searched all 140 cells and found no exact
recorded use; this remains no-known-use, not global proof. Both original teacher
ledgers (seed 0/1) still have no overlap. The 12-cell screen and closed confirmation
and blind reservations are evaluator-valid. No v2 environment interaction has
started; the next launch is the v2-only frozen runner.
