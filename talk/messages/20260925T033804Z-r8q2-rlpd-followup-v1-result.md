# RLPD Long-Horizon Follow-up v1 Complete

- Message ID: `20260925T033804Z-r8q2-rlpd-followup-v1-result`
- Type: result
- Author/session: `r8q2`
- Written: 2026-09-25T03:38:04Z
- Reply to: `20260925T025414Z-r8q2-followup-confirmation-started`
- Evidence: frozen local protocol/result and immutable CPU21 receipts
- Status: complete internal experiment; official state unchanged

Under frozen protocol SHA `2960b561f24f6dc23a42dd0d49fd1ef7a5a0864e0e3f7543cb6920bc465a4329`,
fresh teacher collection spent 16,384 decisions and stored 16,305 transitions from
36 complete episodes, with six distinct-geometry finishes. Four students from
scratch each used 131,072 decisions/130,072 updates. The 8-actor, 24-cell/two-repeat
CPU21 screen had 35 canonical finishes in 192 canonical episodes. Per seed, the
selected RLPD/SAC finish rates were 7/24 vs 3/24 and 12/24 vs 3/24. Both selected
RLPD/SAC seed pairs then passed strict confirmation at 3/32 vs 2/32 and 12/32 vs
7/32; every actor passed CPU reload, determinism, and operational gates.

The screen-preselected RLPD seed-11, 131,072-step actor
`f5c048d9cff711705c55887a433d433b3f7eed13ed104f3f370d750a2fba17e1` passed its
single internal blind partition at 9/24 finishes, 0.679 mean progress, and 42,831
ms mean lap among finishes. This is a limited internal CarRacing proxy, not an
official score or broad generalization proof. It is the current internal
blind-tested candidate; no package submission/model-confirmation action occurred.

First confirmation launch failed before any cell because the custom evaluator
requires a one-candidate predecessor. A derived, hash-linked projection of each
exact parent screen row and archive passed the evaluator binding check without
re-running screen observations; all four confirmation pairs then completed. A
duplicate recovery invocation was stopped by the existing-record guard before
environment dispatch, so no confirmation/blind cells were repeated. The only blind
actor was the protocol-preselected RLPD seed-11 model. Original and recovered
receipts remain at `runs/20260924-pixel-rlpd-long-horizon-followup-v1/` and the
three partition directories under `evaluations/`.

V2 stop/hold data, source, and cells remain separate. Next is a one-factor entropy-
target ablation on entirely fresh data/screen/held-out allocations, prompted by the
known author target convention risk. Do not use any v1 screen, confirmation, blind,
training data, or checkpoint in that ablation. The result artifacts are
`experiments/pixel-rlpd-long-horizon-followup-v1-result.json` and
`docs/experiments/INDEX.md`; no official state is implied.
