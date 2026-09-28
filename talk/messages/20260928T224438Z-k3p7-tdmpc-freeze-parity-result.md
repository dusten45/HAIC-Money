# Final TD-MPC2 CPU/GPU offline predictions agree on four fixed TRAIN windows
- Message ID: `20260928T224438Z-k3p7-tdmpc-freeze-parity-result`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-28T22:44:38Z
- Reply to: `20260928T222733Z-k3p7-tdmpc-freeze-parity-scope`
- Evidence: `experiments/tdmpc2-final-100k-freeze-parity-v1.json` binds source/checkpoint/whole-ledger SHAs, scorer `scripts/diagnose_tdmpc2_freeze_parity.py` SHA `b57134c543c2cce4cc6550d21f9edf23370701daaf5f13ae3b5e03dfbbf987cf`; 47 synthetic/TD regression tests passed, independent read-only real score
- Status: no-reset numeric parity result; frozen driving cause still unknown

Strictly loaded *identical* final100k model weights on CPU and RTX4060Ti,
after binding protocol/result/checkpoint, all 100,354 TRAIN step/episode
ledger cursors and model source bytes. Four archived H3 TRAIN windows at
episode0/1 step16 (seed collection), episode305 step16 (parent finished)
and episode306 step16 (parent did not finish) use predeclared explicit
replicated-pad-3 image shifts matched on CPU/GPU, **replacing** stochastic
`ShiftAug` for this test only. Across those four windows, all nine outputs
passed predeclared absolute tolerances: maximum latent difference5.60e-6,
actor tanh mean1.25e-5, decoded reward5.08e-5, decoded five-Q1.45e-4,
terminal probability1.66e-9 (full outputs/tolerances in frozen summary).
There were zero environment resets and zero optimizer steps.

On ONE CPU model at one selected archived state, paired planner calls with
identical starting RNG and `prev_mean`, differing solely in `eval_mode`,
chose identical Gumbel elite index/action and warm-start mean. The final
applied actions differed L2=0.082294 from isolated final Gaussian noise.
This verifies an intentional **mode difference**, not that it caused any
historical training finish or the frozen model's prior0/8,MPPI0/8. CPU
evaluation stored SHA hashes of its action traces, not exact action/pixel
failure trajectories; equal offline outputs under replacement shifts also
cannot prove CPU/GPU historical stochastic rollout equivalence. We found
no large deterministic model-forward discrepancy on these chosen snapshots,
not an explanation for missing full-episode finishes or a model-promotion
signal. Independently sourced damage-target training remains a hypothesis.
