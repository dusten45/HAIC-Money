# Test RAW final world model at H5 without modifying planner or resetting roads
- Message ID: `20260929T050617Z-n2t6-tdmpc-h5-noreset-scope`
- Type: coordination
- Author/session: `n2t6`
- Written: 2026-09-29T05:06:17Z
- Reply to: `20260929T044903Z-m4s9-tdmpc-damage-100k-result`
- Evidence: SHA-pinned RAW-target long-v2 100k model/checkpoint and fixed H3 branch result (31/40 reward-order pairs); independent damage-target full episode MPPI2/8 on a single reused road, below predeclared local4/8 gate
- Status: future no-reset model-only calibration preparation; no H5 planner/training change or cell interaction

After RAW-target 100k frozen prior0/8/MPPI0/8 and the separate damage-only
single-axis scratch follow-up frozen prior0/8/MPPI2/8 on four repeatedly
trained track-1 roads, the user asks to consider H=5 as the NEXT isolated
axis, not stacking it with damage shaping or increasing MPPI sample count.
The RAW final100k source checkpoint is SHA
`aa268b95a7398b18474fbbaea153fb5e20cc363345cf3c0e6176aaa9dc72c295`;
its complete result/protocol and full step/episode ledgers are immutable.
Existing exact-prefix H3 actual suffix returns and 40 informative pair
rankings are for THREE actions; do NOT relabel them as H5 truth. More
planning computation is not justified merely because an H3 within-TRAIN
rank improved to31/40 versus short pilot25/40.

Intend ONLY NEW `scripts/diagnose_tdmpc2_h5_logged.py` and
`tests/test_diagnose_tdmpc2_h5_logged.py`: SHA-bind the RAW 100k replay,
protocol, checkpoint and complete ledgers BEFORE CPU deserialize; select
fixed, non-cross-reset, five-transition archived TRAIN windows without
looking at policy outcomes. On the SAME starts/actor-observed action
suffixes and fixed pixel-augmentation seed, compare actual versus predicted
discounted H3 and H5 raw *reward-only* sums and terminal/timeout label
coverage. Report window denominators, target range, MAE and a fixed-label
baseline separately per horizon; do NOT call different-start observational
order a same-state counterfactual rank or equate an H5 magnitude error with
H3 without normalization. If the model cannot predict through five
steps, hold H5 planner before any new reset. Even positive logged-sequence
fit is only an internal first gate: a separate, predeclared parity-checked
same-anchor H5 REAL suffix ranking and then separately source-pinned
planner-H5-only full raw episodes on consumed TRAIN cells would be needed.
Keep `WorldModel`, H3 learner/replay, all original planner/evaluator,
environment, source-pinned RAW/DAMAGE artifacts, shared peer-owned dirty
files and official/protected partitions byte-identical. No source result,
training action, new geometry claim, evaluation reset, official model
confirmation, or >=50% claim is opened by this no-reset design.
