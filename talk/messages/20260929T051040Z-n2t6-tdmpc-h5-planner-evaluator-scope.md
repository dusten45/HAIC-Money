# Prepare H5 planner-only CPU source without altering RAW H3 model or roads
- Message ID: `20260929T051040Z-n2t6-tdmpc-h5-planner-evaluator-scope`
- Type: coordination
- Author/session: `n2t6`
- Written: 2026-09-29T05:10:40Z
- Reply to: `20260929T050617Z-n2t6-tdmpc-h5-noreset-scope`
- Evidence: RAW100k checkpoint SHA `aa268b95a7398b18474fbbaea153fb5e20cc363345cf3c0e6176aaa9dc72c295`, its completed source/ledger; frozen RAW H3 MPPI0/8 full-episode reused-TRAIN result; DAMAGE-only distinct target MPPI2/8 on one reused road below local4/8 gate
- Status: NEW evaluator code/tests preparation only, NO H5 environment reset or protocol freeze

The user requests continuing isolated research toward >=50% full-episode
completion. First source-bound, NO-RESET logged H5 reward/terminal checks
on the RAW checkpoint are separately being implemented; H3 real branch
suffixes cannot be labeled H5 truth. Only if an informative, predeclared
model-quality gate passes, a separately frozen H5 same-anchor real-return
branch study is needed before extra planning resets. In parallel with this
read-only probe, prepare but do NOT execute NEW
`scripts/evaluate_tdmpc2_h5_planner.py` and
`tests/test_evaluate_tdmpc2_h5_planner.py` for a later planning-only test.

Strictly SHA-bind the unchanged RAW100k final H3-trained WorldModel,
completed training/evaluation result and full step/checkpoint cursors before
trusted CPU `torch.load`; same raw official environment and four already-
consumed obstacle-enabled track-1 TRAIN roads, two canonical repeats each,
base eval seed20260928, full2,000-decision episodes. The only intended policy
change is **PlannerConfig.horizon=5** rather than default3: retain 3D
action, model weights, 512 MPPI samples including24 actor proposals, six
iterations,64 elites, discount.995, terminal handling, and `eval_mode=True`.
Primary mode is H5 MPPI n8, compared descriptively with frozen H3 MPPI0/8
on matched reset *cells*, not matched actions or trajectories. No additional
prior arm is needed because prior doesn't use the planner horizon. Report
per-road finishes/8, censoring, raw return/progress/damage and CPU latency;
an observed>=4/8 on these four reused roads is only a descriptive LOCAL
target, NOT independent cross-track generalization or official score.

Preflight must reject absent failed H5 model-quality/branch gate, incomplete
RAW source, unbounded latency/resources, wrong planner horizon or a reused
output path before ANY environment constructor. Freeze exact H5 protocol/
code SHA only AFTER the no-reset gate and any required real H5 branch result
exist. The code's synthetic tests may use fake env/model without actual
reset. Do not alter old H3 trainer/evaluator/planner/model/replay, DAMAGE
trainer/evaluator, shared teammate-owned subsystems, checkpoint metadata,
reserved/protected roads, official submission or model confirmation.
