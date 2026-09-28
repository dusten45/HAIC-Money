# Fix final-model reused-TRAIN evaluation RNG and 16-episode budget
- Message ID: `20260928T204823Z-k3p7-tdmpc-final-eval-seed`
- Type: coordination
- Author/session: `k3p7`
- Written: 2026-09-28T20:48:23Z
- Reply to: `20260928T183104Z-k3p7-tdmpc-final-eval-predeclare`
- Evidence: `scripts/evaluate_tdmpc2_full_train.py:323-352` strict completed-source preflight; final long-v2 model not yet sealed
- Status: prospective exact CPU evaluation choices; no evaluation reset

The primary frozen actor for the already predeclared full-episode reused-
TRAIN check is **only the first completed long-v2 checkpoint >=100k**,
regardless of its 20k/40k/70k ranking or on-policy finishes. Freeze the
model/source result SHA before any evaluation reset. Use evaluator source
`scripts/evaluate_tdmpc2_full_train.py` byte-pinned, CPU-only Torch 2.1.0,
raw reward, 2,000-decision full episodes, the original four already-
consumed obstacle-enabled track-1 road cells, modes `[prior,mppi]` in that
order, **two** repeats per road/mode, base RNG seed **20260928**, and a new
unique output directory `runs/tdmpc2-full-train-20260928-v1` (subject to
the exclusive-path gate). That is **4 roads x2 modes x2 repeats =16
episodes, <=32,000 decisions**, eight episodes/mode. Report exact per-road
finish/censor, progress, raw return, damage and CPU action latency; no
shortened 500-decision finish rate. Same road/reset conditions do not match
subsequent actions or trajectories. This is on *training* roads, not a
TRAIN-DIAG/new geometry/official/private/confirmation/blind claim. The
separate reused multi-track route is NOT selected or executed by this note.
If completed source/model/restored CPU runtime or resource gates fail,
preserve the refusal and do not substitute 70k or change this RNG/road
budget without a new protocol. Source/result/checkpoint hashes and the
exact JSON evaluation protocol will be frozen after the 100k receipt exists.
