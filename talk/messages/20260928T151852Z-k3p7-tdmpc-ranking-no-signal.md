# TD-MPC2 v2 logged H3 ranking has no discriminating targets
- Message ID: `20260928T151852Z-k3p7-tdmpc-ranking-no-signal`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-28T15:18:52Z
- Reply to: `20260928T150809Z-k3p7-tdmpc-exploration-result`
- Evidence: SHA-bound v2 replay/checkpoint/CPU export and `experiments/tdmpc2-v2-logged-ranking-diagnostic.json`
- Status: no valid ranking inference; same-state prefix gate remains open

Read-only `python -m scripts.diagnose_tdmpc2_prefix_ranking` matched 37
complete v2 TRAIN replay episodes and all 12,058 actions/rewards to its
source-pinned ledger, without resetting the environment. Every logged H=3
first-action-window return is approximately -1.19401, with a cross-window
range 6.66e-16. All 666 across-start pairs are ties at absolute 1e-6;
concordance has **zero comparable pairs** and is null. An initial exact-float
comparator would have mislabeled 278+235=513 roundoff-only pairs as ordered;
that bug was caught before interpreting or publishing a ranking and corrected
with a tested tolerance. This is neither proof of world-model failure nor a
positive survival signal. It is observational training replay from different
initial states, not a parity-verified same-state branch. Later anchors and
independently verified action-suffix branches remain required for the user's
actual predicted-versus-real ranking question. No new environment cell,
protected partition, learner step, or official action was used.
