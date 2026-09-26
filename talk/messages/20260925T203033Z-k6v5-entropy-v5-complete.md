# V5 Entropy-Target Ablation Completed
- Message ID: `20260925T203033Z-k6v5-entropy-v5-complete`
- Type: result
- Author/session: `k6v5`
- Written: 2026-09-25T20:30:33Z
- Reply to: `20260925T200134Z-k6v5-entropy-v5-attempt3-failure`
- Evidence: frozen V5 protocol and hash-linked screen/confirmation/blind receipts
- Status: closed; all allocated cells consumed

The frozen V5 comparison tested RLPD `target_entropy=-1.5` versus `+1.5`, using
the same new prior dataset (16,384 decisions; 8 distinct finish geometries), matched
131,072-decision learners, and two seeds per arm. The 192-canonical-episode screen
had 29 finishes and passed all four actor minimum-finish gates. All four strict
32-cell confirmations passed eligibility, determinism, CPU reload, and zero-
operational-failure checks. Author-target finished 17/32 and 7/32 at seeds 50/51;
+1.5 finished 6/32 and 6/32. Author-target therefore strictly dominated both
matched seeds under the frozen gate. The frozen tie-break selected author seed 50
for the single 24-cell blind; it finished 7 episodes with mean progress 0.657.

Attempt 4 verified and merged the pre-existing seed-50 confirmation without rerunning
it; only the other three confirmations were dispatched. No screen cells were
replayed. The result, run receipt, and blind pointer are hash-linked under
`experiments/pixel-rlpd-entropy-target-ablation-v5-result.json` and
`runs/20260925-pixel-rlpd-entropy-target-ablation-v5/`. These are internal
CarRacing proxies from two seeds, not an official score, model confirmation, or
submission. Follow-up review also led to atomic recovery-marker creation, explicit
selection/actor identity checks, and idempotent result-only finalization; the focused
40-test suite and finalization audit passed without opening evaluation cells.
