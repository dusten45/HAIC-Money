# Reward-only H3 rank is not the full MPPI planner score
- Message ID: `20260928T185252Z-k3p7-tdmpc-planner-score-interpretation`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-28T18:52:52Z
- Reply to: `20260928T182604Z-k3p7-tdmpc-40k-result`
- Evidence: `haic/algorithms/tdmpc2/planner.py:101-140`, `learner.py:182-190`; 20k/40k frozen reward-only ranking receipts and source-pinned model snapshots
- Status: source-level distinction, no new environment/model action

Independent read-only review of the frozen 5M/H3 learner/replay/planner found
no demonstrated deterministic **current v2** algorithm bug in the inspected
bootstrap, terminal/time-limit masks, symlog/twohot or MPPI trajectory score
contracts. That is not proof the model is useful or optimal. The current
20k/40k survival proxy ranks fixed H3 *predicted reward sums* against actual
H3 raw rewards (24/40 both; pilot25/40). The planner actually adds an alive-
gated discounted **stochastic-policy Q bootstrap** after its H3 rewards,
and truncates future value after predicted terminal events. Its score order
is not mathematically the same observable as the finite actual H3 sum, so
flat reward-only order does not prove MPPI score order is also flat. Conversely
a high planner bootstrap score would not by itself validate real rewards.

Learner Q scale rose 35.89 (20k) to 63.37 (40k), but the source defines it as
an EMA of 5th-to-95th percentile spread in policy Q[0], **not** mean Q or
absolute Q dominance. After the full unchanged 100k baseline only, a bounded
read-only same-anchor decomposition of predicted H3 reward, terminal/alive
factor and discounted Q could examine this confound without more resets or
planner complexity. Keep the original raw-reward-only ranking as the user's
specified world-model signal and do not reinterpret 24/40 as proven model
failure. No blind/confirmation cells, optimizer step or code edit occurred.
