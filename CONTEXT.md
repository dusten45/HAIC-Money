# Project Context

## Current State

The goal is a fast, reliable, rule-compliant 2026 HAIC agent. The live bare
`Agent` route remains `_CompoundClearingBrakeCarryController`; neural weights in
`model.pt` are bypassed after road detection. Diagnostic controllers in
`agent.py` are not active or submission-ready.

The source-bound `camera-policy-competition-v3` study tested the unchanged
`_BoundedSideHoldController` on new geometry under a competition-aligned gate.
Screen retained (29/32 finishes versus 18/32, no lost control finish), but
confirmation **REJECTED**: 46/64 versus 31/64 finishes and five lost control
finishes against its fixed limit of four. All 136 confirmation receipts and four
exact repeats validated; the saved summary and independent recomputation
agree. Candidate crashes/contacts/damage were 2/38/7.6 versus 8/102/20.4,
but shared finishes were about 6% slower. Three lost finishes were on track ID
3 and two on ID 4. The blind phase is sealed and the live route is unchanged.
See `experiments/camera-policy-competition-v3-result.json` and its bound
protocol. The 24 opened V3 geometries are now development data.

Earlier V2 and V1 camera candidates were also rejected under their own fixed
screen gates despite aggregate finish gains. Their confirmation/blind phases
remain sealed. V2 lost-control mechanisms include on-road post-contact stalls,
side-switch vetoes and a crash without the proposed intervention; static
camera-pixel corridor widths did not certify physical clearance. The prior
results and consumed-cell diagnostics remain in `experiments/`.

Current priority: diagnose the five V3 confirmation finish losses with
source-bound action/camera replays, then test a minimal context-sensitive
controller on consumed cells. Screen failures also show two road exits on one
geometry and one on-road stall; first-dropout camera evidence is not yet saved,
so full-width road reacquisition remains a hypothesis. The candidate's
systematic 6–7% shared-finish slowdown also needs attention. Bind a new fresh
protocol before any subsequent promotion claim; do not open V3 blind.

## Frozen Competition Contract

- Do not modify `core/`, `env_wrapper.py`, or `damage.py` for experiments.
- Input: four 84x84 grayscale frames, CHW float32 in `[0,1]`. Output:
  continuous `[steer, gas, brake]` with steer in `[-1,1]` and pedals in `[0,1]`.
  There is one decision per step; frame skip 4 is inside the environment.
- Use raw official reward. Pure time limits bootstrap; finish, crash and
  off-track endpoints do not. Finish needs at least 95% progress and a valid
  forward finish crossing. Geometry is seed-driven; track ID changes obstacles.
  Reserve a geometry seed across every track ID.
- Official ranking prioritizes completion, then DNF progress or completed lap
  time. Require source-bound cold CPU receipts and exact reload checks before
  promotion. Never select on confirmation/blind or alter gates after data.
- Inference gate: Python 3.11, Torch 2.1 CPU, NumPy 1.26; init 10 s,
  reset/action 5 s, worker 1,024 MiB, ZIP 500 MiB. No trainer, SB3 or prohibited
  imports in submission inference. The official participant source checked on
  2026-10-04 KST was commit `dfb7a2de2178825ca5c5ce20bab01ba67052ba31`.

## Other Evidence and Infrastructure

- The completed matched DrQ-v2 steering-logit L2 study rejected coefficient
  0.001 in both training seeds: paired confirmation finish deltas were -3 and
  -7. The separately planned augmentation-padding comparison has no verified
  completion in this Windows checkout. Demonstration replay is implemented but
  not trained or performance-evaluated. Read the corresponding
  `experiments/drqv2-*-result.json` and execution records before resuming ML.
- Earlier PPO variants did not establish competitive completion. Video-recovered
  maps `(1,516237)`, `(2,644062)`, `(3,1007)` and untracked `evaluation_videos/`
  are diagnostic, not fresh evaluation. Track Lab can replay camera, action,
  trajectory and collision telemetry.
- Local `.venv`: Windows Python 3.11.15, Torch 2.1 CPU, NumPy 1.26. Windows
  lacks `fcntl` for `tests/test_drqv2_matched.py`; the latest full suite
  excluding that file passed 907 tests and skipped 10. Official-like Linux
  submission-container certification is unavailable here: Docker is stopped
  and WSL lacks the required runtime/dependencies.
- Preserve untracked user videos, prior submission directories and
  `submission.zip`; none is the current candidate package. Keep large traces,
  videos, models and run artifacts out of Git. Commit and push coherent,
  validated progress to the current upstream branch.
