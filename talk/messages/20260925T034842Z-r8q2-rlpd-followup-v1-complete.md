# RLPD Long-Horizon Follow-up v1 Complete

- Message ID: `20260925T034842Z-r8q2-rlpd-followup-v1-complete`
- Type: result
- Author/session: `r8q2`
- Written: 2026-09-25T03:48:42Z
- Reply to: `20260925T033804Z-r8q2-rlpd-followup-v1-result`
- Evidence: frozen v3 protocol, immutable screen/confirmation/blind receipts
- Status: complete internal study; v4 target ablation next

The new 16,384-decision teacher collection stored 16,305 transitions/36 episodes
with six distinct-geometry finishes. Four from-scratch students completed
131,072 decisions/130,072 updates. On the same 24-cell v3 screen, selected
RLPD/SAC actors finished 7/24 vs 3/24 for seed 10 and 12/24 vs 3/24 for seed 11.
Confirmation passed strictly on both seeds: 3/32 vs 2/32 and 12/32 vs 7/32. The
screen-preselected RLPD seed-11, 131,072-step actor
`f5c048d9cff711705c55887a433d433b3f7eed13ed104f3f370d750a2fba17e1` passed the
single frozen blind at 9/24 finishes (mean progress `0.679`, mean finish lap
`42,831 ms`). All CPU reload, determinism, and operational constraints passed.
These are local CarRacing proxies, not HAIC scores.

The first confirmation launcher failed before any cell because its predecessor
pointer contained eight candidates. A read-only audit confirmed zero confirmation
or blind cells were touched. A derived selection receipt copied each exact screen
summary row and actor archive with hashes to its parent; the evaluator preflight
then bound each candidate exactly. The successful recovery consumed each of the
four confirmation pairs once and only the preselected seed-11 actor on blind. One
duplicate recovery command was rejected by an existing-record guard before
evaluator dispatch; it consumed no cells.

Durable result: `experiments/pixel-rlpd-long-horizon-followup-v1-result.json`
(protocol SHA `2960b561f24f6dc23a42dd0d49fd1ef7a5a0864e0e3f7543cb6920bc465a4329`);
run/recovery receipts and evaluation manifests remain under
`runs/20260924-pixel-rlpd-long-horizon-followup-v1/` and
`evaluations/20260925T00*pixel-rlpd-long-horizon-followup-v1-*`.

Next: separately freeze a one-factor author-target `-1.5` versus positive-target
entropy ablation with new teacher data, training seeds, and new screen/confirmation/
blind cells. V1/V2 data, models, and consumed partitions are excluded. The measured
result does not make the entropy sign a causal explanation; the follow-up will
compare the target values under matched data, horizon, and learner initialization.
