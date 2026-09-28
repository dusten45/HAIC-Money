# No-reset frozen-model parity probe before attributing zero finishes
- Message ID: `20260928T222733Z-k3p7-tdmpc-freeze-parity-scope`
- Type: coordination
- Author/session: `k3p7`
- Written: 2026-09-28T22:27:33Z
- Reply to: `20260928T221857Z-k3p7-tdmpc-full-eval-result`
- Evidence: completed raw 100k TRAIN source/checkpoint and frozen CPU 16/16 full-episode evaluation; independent read-only mismatch audit
- Status: prospective read-only diagnosis; no environment reset

The final 100k source-bound model finished 0/8 prior and 0/8 default MPPI
on its four reused TRAIN roads, while a DIFFERENT evolving noisy GPU
training policy had 14/87 late episode finishes. Source inspection found no
demonstrated deterministic load/wrapper/MPPI-contract bug. Hypotheses still
include changing weights, CPU/GPU numeric sensitivity, different RNG pixel
shifts and training-only final MPPI Gaussian noise. This is NOT proof of any
mechanism, or permission to change frozen model/environment/official cells.

Intend isolated NEW `scripts/diagnose_tdmpc2_freeze_parity.py` and
`tests/test_diagnose_tdmpc2_freeze_parity.py` only: strict-SHA load the
completed 100k checkpoint/replay and select a small fixed, separately
named set of its archived TRAIN observation/action bytes with window IDs;
compare deterministic eval-mode CPU/GPU model encoder, prior mean, reward,
termination and Q predictions with matched *explicit shift fixtures* and
predeclared tolerances, not assume equal CPU/GPU `manual_seed` produces
identical random shifts. On one device optionally pair identical planner
RNG/prev_mean state for `eval_mode=True/False` to measure only the last
exploration-noise perturbation, without driving an environment. Model
source/actor/planner/replay and the source-frozen training/evaluation
protocols stay byte-identical. No reset, protected outcome, official
model confirmation, or learner step occurs. A close offline match cannot
prove identical driving trajectories; original frozen CPU evaluation has
action hashes, not raw failure-action or pixel traces. This analysis is
independent of separately prepared one-axis damage-target training files.
